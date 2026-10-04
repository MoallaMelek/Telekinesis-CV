"""Telekinesis CV - Reality Manipulation.

Touch a real object in the (mirrored) webcam image -> its silhouette highlights ->
pinch to lift its extracted appearance -> move, throw, scale, rotate, hide, reset.
The physical object never moves; only its appearance in the video does.
"""
import argparse
from collections import deque
import math
import sys
import time

import cv2
import numpy as np

from hand_tracker import HandTracker, draw_hand_skeleton
from manipulation import Group, Manipulator, Sprite, render_sprite
from scene import (ObjectTracker, PersonSegmenter, Reconstruction, SceneMemory, bbox_of,
                   feathered_alpha, make_inpainter)
from segmentation import MODEL_DIR, Job, SegmentationWorker, iou
from selection import Selector, hand_zone, hand_core, negative_prompts
from precision import SelectionEditor
from hud import draw_hud

WINDOW = "Telekinesis CV | Reality Manipulation"
CYAN, GREEN, ORANGE, WHITE, GREY = (255, 220, 90), (120, 255, 140), (60, 170, 255), (235, 235, 235), (160, 160, 160)


def text(frame, message, org, color=WHITE, scale=.42, thickness=1):
    for t, shade in ((thickness + 2, (0, 0, 0)), (thickness, color)):
        cv2.putText(frame, message, org, cv2.FONT_HERSHEY_SIMPLEX, scale, shade, t, cv2.LINE_AA)


def ipt(p):
    return tuple(int(v) for v in np.rint(p))


def shifted(mask, offset):
    """Translate a bool mask by an integer pixel offset (no wrap-around)."""
    dx, dy = (int(v) for v in np.rint(offset))
    if dx == 0 and dy == 0:
        return mask
    out = np.zeros_like(mask)
    h, w = mask.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


class App:
    """All per-frame logic. `step()` is pure w.r.t. I/O, so tests can drive it."""

    def __init__(self, width, height, worker, args):
        self.width, self.height = width, height
        self.worker = worker
        self.args = args
        self.selector = Selector()
        self.manip = Manipulator(width, height, args.pinch_threshold, args.release_threshold,
                                 args.min_scale, args.max_scale)
        self.memory = SceneMemory()
        self.inpainter = make_inpainter()
        self.editor = None
        self.clicked = None            # intentional mouse request, independent of dwell
        self.mouse_until = -1.0
        self.pending_lock = None       # (slot, deadline) pinch arrived before the mask did
        self.refining = None           # (group id, snapshot id, submitted time)
        self.refine_counter = 10 ** 6  # refine jobs use their own id range
        self.debug = args.debug
        self.occlusion = "hands"
        self.show_keys = False
        self.messages = deque(maxlen=3)
        self.mouse = {"point": None, "down": False, "moved": -10.0}
        self.pointer_slot = None
        self.last_candidate = None
        self.counters = dict(previews=0, locks=0, rejected_stale=0, refine_accepted=0,
                             refine_rejected=0)
        self.started = None
        self.next_group = 0            # unique ids: refinement results find the right object

    # ------------------------------------------------------------------ helpers
    def say(self, message, now, seconds=2.0):
        self.messages.append((message, now + seconds))
        print(f"[{now:7.2f}s] {message}", flush=True)

    def active(self):
        return self.manip.object(self.manip.active) if self.manip.active is not None else None

    def groups(self):
        seen = {}
        for obj in self.manip.objects:
            seen.setdefault(obj.group.id, obj.group)
        return list(seen.values())

    def blocked(self, x, y):
        for group in self.groups():
            ox, oy = (int(v) for v in np.rint(group.tracker.offset))
            yy, xx = y - oy, x - ox
            if 0 <= yy < self.height and 0 <= xx < self.width and group.reconstruction.hole[yy, xx]:
                return True
        return self.manip.hit((x, y)) is not None

    def choose_pointer(self, now):
        hands = self.manip.hands
        if any(h.held is not None for h in hands):
            return None                # while holding, a 2nd-hand pinch means scale/rotate
        if self.mouse["point"] is not None and now < self.mouse_until and hands[2].held is None:
            return 2
        eligible = [s for s in (0, 1) if hands[s].present and hands[s].held is None]
        if self.pointer_slot in eligible:
            return self.pointer_slot
        if eligible:
            return eligible[0]
        return 2 if self.mouse["point"] is not None else None

    # ------------------------------------------------------------------ main step
    def person(self):
        """Person-probability mask for the current frame, computed only when needed."""
        if self._person is None:
            source = self._person_source
            value = source() if callable(source) else source
            self._person = np.zeros(self._shape, np.float32) if value is None else value
        return self._person

    def step(self, frame, now, dt, observations, person=None):
        """frame: mirrored BGR camera image; observations: MediaPipe hands (mirrored px);
        person: float person-probability mask, a zero-argument callable producing it
        lazily, or None. Returns the composited display."""
        if self.started is None:
            self.started = now
        h, w = frame.shape[:2]
        self._shape, self._person_source, self._person = (h, w), person, None
        if self.editor is not None:
            return self.edit_step(now)
        if dt > .25:
            self.selector.reset_dwell(now)
        mouse = self.mouse if self.mouse["point"] is not None else None
        events = self.manip.update_hands(observations, now, dt, mouse)
        hands = self.manip.hands
        hand_points = [hands[s].points for s in (0, 1) if hands[s].present]
        zone = hand_zone((h, w), hand_points)
        core = hand_core((h, w), hand_points)
        self.last_frame, self.last_zone = frame, zone
        if self.manip.objects or self.debug:
            person = self.person()     # occlusion, reconstruction and tracking need it
        if now >= self.memory.next_at:
            # Remember where people/hands were, so memory never offers a hand as background.
            seen = zone.astype(np.float32) if self._person is None else np.maximum(self._person, zone)
            self.memory.observe(frame, seen, now)

        # 1) Selection: aim -> (snapshot + segmentation in the background) -> preview.
        self.pointer_slot = self.choose_pointer(now)
        pointer = hands[self.pointer_slot] if self.pointer_slot is not None else None
        aim = None if pointer is None else pointer.aim
        engaged = pointer is not None and pointer.pinch.engaged
        hovering = None if aim is None else self.manip.hit(aim)
        negatives = [p for pts in hand_points for p in negative_prompts(pts)]
        job = None
        if self.clicked is not None:
            point, self.clicked = self.clicked, None
            if not self.blocked(*point) and not self.worker.error:
                job = self.selector.prompt(now, frame, [point], negatives=negatives,
                                           hands=hand_points, zone=core)
                # First click requests an outline. A second click/pinch confirms it.
                events = [(s, e) for s, e in events if s != 2]
        elif not self.worker.error and (self.worker.ready or self.worker.model is not None):
            job = self.selector.update(now, None if hovering else aim, frame, hand_points,
                                       core, self.blocked, engaged, negatives=negatives)
        if job is not None:
            job.max_area = self.args.max_area
            if job.frame is not None:
                # Approximate foreground is a ranking hint, never object ownership.
                job.person = self.person() > .5
            self.worker.request(job)

        polled = self.worker.poll()
        if polled is not None:
            kind, result = polled
            if kind == "select":
                if self.selector.accept(result, now):
                    self.counters["previews"] += 1
                    c = self.selector.candidate
                    if c is not None:
                        print(f"[{now:7.2f}s] preview: {len(result.candidates)} candidates, chose "
                              f"area {c.area:.3f} score {c.score:.2f} stability {c.stability:.2f}; "
                              f"encoder {result.encoder_ms:.0f} ms decoder {result.decoder_ms:.0f} ms",
                              flush=True)
                else:
                    self.counters["rejected_stale"] += 1
            # Refinement is applied after this frame's motion below. Rebasing here
            # would anchor yesterday's image pose to today's grip and lose movement.

        # 2) Gesture events.
        for slot, event in events:
            self.handle(slot, event, now)
        if self.pending_lock is not None:
            slot, deadline = self.pending_lock
            hand = hands[slot]
            if not hand.pinch.pinched or now > deadline or self.worker.error:
                self.pending_lock = None
                self.say("Pinch cancelled - no outline was ready. Hold still, then pinch.", now)
            elif self.selector.candidate is not None:
                self.pending_lock = None
                self.lock(slot, now)

        # 3) Track the physical originals; refine the cut-out once the hand is away.
        if self.groups():
            person = self.scene_person(core)
            occluder = (person > .5) | core
            for group in self.groups():
                group.tracker.update(frame, occluder, now)
            self.schedule_refine(frame, core, now)
        self.manip.update_objects(now, dt)
        if polled is not None and polled[0] == "refine":
            self.finish_refine(polled[1], now)
        person = self.scene_person(core) if self.manip.objects else self.person() if self.debug else None

        # 4) Composite: live -> reconstructed originals -> sprites -> hands in front.
        out = frame.copy()
        for group in self.groups():
            group.reconstruction.render(out, frame, group.tracker.offset, person)
        dirty = []
        for obj in self.manip.draw_order():
            glow = GREEN if obj.holders else None
            rect = render_sprite(out, obj, glow)
            if rect:
                dirty.append(rect)
        if self.occlusion != "off" and dirty:
            if self.occlusion == "hands":
                soft_zone = cv2.GaussianBlur(core.astype(np.float32), (9, 9), 0)
                fg = person * soft_zone
            else:
                fg = person
            for x0, y0, x1, y1 in dirty:
                a = fg[y0:y1, x0:x1, None]
                out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * (1 - a) + frame[y0:y1, x0:x1] * a).astype(np.uint8)
        self.draw_feedback(out, frame, now, aim, hovering, person, zone)
        return out

    def scene_person(self, core):
        """A selected object's pixels take precedence over coarse selfie foreground."""
        person = self.person().copy()
        for group in self.groups():
            mask = shifted(group.mask, group.tracker.offset)
            person[mask & ~core] = 0
        return person

    # ------------------------------------------------------------------ events
    def handle(self, slot, event, now):
        m = self.manip
        hand = m.hands[slot]
        if event == "press":
            target = m.hit(hand.pinch_point) or m.hit(hand.aim)
            other = next((m.object(m.hands[s].held) for s in (0, 1, 2)
                          if s != slot and m.hands[s].held is not None), None)
            if target is not None:
                m.grab(slot, target, now)
                self.say("Two hands: spread = scale, turn = rotate" if len(target.holders) > 1
                         else "Grabbed - move it; release to place, flick to throw", now)
            elif other is not None and slot != 2:
                m.grab(slot, other, now)
                self.say("Two hands: spread = scale, turn = rotate", now)
            elif slot == self.pointer_slot and self.selector.candidate is not None:
                self.lock(slot, now)
            elif slot == self.pointer_slot and self.selector.state == "ANALYZING":
                self.pending_lock = (slot, now + 2.0)
        elif event == "release":
            if self.pending_lock and self.pending_lock[0] == slot:
                self.pending_lock = None
            outcome = m.release(slot, now)
            if outcome == "throw":
                self.say("Thrown!", now, 1.2)
            elif outcome == "drop":
                self.say("Placed - it floats there. Pinch it to grab again.", now)
        elif event == "lost" and slot != 2:
            self.say("Hand lost - object stays where it was (no throw)", now)
        elif event == "fist":
            other = next((m.object(m.hands[s].held) for s in (0, 1) if s != slot
                          and m.hands[s].held is not None), None)
            if other is not None:
                copy = m.duplicate(other)
                self.say("Duplicated (experimental)" if copy else "Object limit reached", now)
            elif self.active() is not None:
                self.say(f"Object {m.toggle_visibility(self.active())}", now)
        elif event == "palm":
            obj = self.active()
            if obj is not None and not any(o.holders for o in m.objects):
                m.reset(obj)
                self.say("Reset to original position, scale, rotation", now)

    # ------------------------------------------------------------------ lock / extract
    def lock(self, slot, now):
        result, candidate = self.selector.result, self.selector.candidate
        if candidate is None or not candidate.selectable:
            return
        if len(self.manip.objects) >= self.manip.max_objects:
            self.say("Object limit reached - Esc releases one, X releases all", now)
            self.selector.clear()
            return
        embedding = result.embedding
        snap = embedding.frame
        mask = candidate.mask.copy()
        # Trust the confirmed silhouette. Selfie masks/hulls cannot distinguish an
        # object held in front of a person from skin; trimming them destroys objects.
        for group in self.groups():
            if iou(mask, shifted(group.mask, group.tracker.offset)) > .5:
                self.say("That object is already extracted - pinch its image instead", now)
                self.selector.clear()
                return
        self.next_group += 1
        group, sprite = self.extract(snap, mask, now, self.next_group)
        group.snapshot_id = result.snapshot_id
        if result.explicit:
            group.refined = True
            group.refine = "User-corrected silhouette; automatic refinement disabled"
        group.score, group.stability = candidate.score, candidate.stability
        obj = self.manip.add(group, sprite)
        self.manip.grab(slot, obj, now)
        self.last_candidate = (mask, candidate)
        self.counters["locks"] += 1
        self.selector.clear()
        self.say(f"Locked ({group.reconstruction.source}). Move it; release to place, flick to throw", now, 3)

    def extract(self, frame, mask, now, gid):
        """Build the RGBA sprite + tracker + background reconstruction for one object."""
        box = bbox_of(mask, margin=3)
        x0, y0, x1, y1 = box
        ys, xs = np.nonzero(mask)
        centroid = np.array([xs.mean(), ys.mean()])
        sprite = Sprite(frame[y0:y1, x0:x1].copy(), feathered_alpha(mask)[y0:y1, x0:x1],
                        centroid - (x0, y0))
        tracker = ObjectTracker(frame, mask)
        reconstruction = Reconstruction(frame, mask, self.memory, now, self.inpainter)
        return Group(gid, mask, centroid, tracker, reconstruction), sprite

    def schedule_refine(self, frame, zone, now):
        """Re-segment once the hand has left the original region: a hand-free view
        gives a complete silhouette (no finger notch) and clean pixels."""
        if self.refining is not None:
            if now - self.refining[2] > 6:
                self.refining = None           # replaced/lost in the one-slot queue
            return
        if self.worker.error or not self.worker.ready or self.worker.busy or self.selector.state == "ANALYZING":
            return
        for group in self.groups():
            if group.refined:
                continue
            region = shifted(group.reconstruction.hole, group.tracker.offset)
            region = cv2.dilate(region.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            clear = not (region & zone).any() and group.tracker.state == "TRACKING"
            if not clear:
                group.clear_since = None
                continue
            if group.clear_since is None:
                group.clear_since = now
            if now - group.clear_since < .4:
                continue
            box = bbox_of(shifted(group.mask, group.tracker.offset), margin=6)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            self.refine_counter += 1
            job = Job("refine", self.refine_counter, self.refine_counter, frame=frame,
                      positives=(tuple(group.origin + group.tracker.offset),), box=(x0, y0, x1, y1),
                      hand_points=(), hand_zone=zone, person=self.person() > .5,
                      captured_at=now, max_area=self.args.max_area)
            self.refining = (group.id, self.refine_counter, now, group.tracker.offset.copy())
            group.refine = "re-segmenting from a hand-free view..."
            self.worker.request(job)
            return

    def finish_refine(self, result, now):
        if self.refining is None or result.snapshot_id != self.refining[1]:
            return
        gid, submit_offset = self.refining[0], self.refining[3]
        self.refining = None
        group = next((g for g in self.groups() if g.id == gid), None)
        if group is None:
            return
        if group.tracker.state != "TRACKING":
            # The scene may have changed while the model ran; re-anchoring to that frame
            # could be wrong. Skip now; it is retried once tracking is confident again.
            group.refine = "refinement skipped (tracking not confident); will retry"
            group.clear_since = None
            return
        group.refined = True
        # Compare in the coordinates of the frame that was re-segmented.
        old = shifted(group.mask, submit_offset)
        best, best_iou = None, 0.0
        for c in result.candidates:
            if not c.selectable:
                continue
            value = iou(c.mask, old)
            if value > best_iou:
                best, best_iou = c, value
        ratio = best.mask.sum() / max(old.sum(), 1) if best is not None else 0
        if best is None or best_iou < .5 or not .6 <= ratio <= 1.8:
            # Never silently switch objects: disagreement keeps the original cut-out.
            group.refine = f"kept original (refinement IoU {best_iou:.2f})"
            self.counters["refine_rejected"] += 1
            return
        mask = best.mask
        new_group, sprite = self.extract(result.embedding.frame, mask, now, group.id)
        # The result describes the SUBMITTED frame, not the current camera pose.
        new_group.tracker.offset = group.tracker.offset - submit_offset
        delta = new_group.origin - (group.origin + submit_offset)
        group.mask, group.origin = new_group.mask, new_group.origin
        group.tracker, group.reconstruction = new_group.tracker, new_group.reconstruction
        group.refine = f"refined from hand-free view (IoU {best_iou:.2f})"
        for obj in self.manip.objects:
            if obj.group is group:
                # Preserve the image transform when refinement changes the centroid.
                obj.position += obj.matrix()[:, :2] @ delta
                obj.sprite = sprite
                self.manip.two.pop(obj.id, None)
                self.manip._rebase(obj)
        self.counters["refine_accepted"] += 1
        print(f"[{now:7.2f}s] refinement accepted, IoU {best_iou:.2f}, "
              f"background: {group.reconstruction.source}", flush=True)

    # ------------------------------------------------------------------ keyboard / mouse
    def key(self, key, now):
        if self.editor is not None:
            if key == 27:
                self.editor = None
                self.cancel_selection()
            elif key in (ord("e"), ord("E")) and self.worker.error:
                self.worker.retry()
                self.submit_edit(now)
            elif key in (10, 13) and self.selector.candidate is not None and self.selector.state == "PREVIEW":
                self.editor = None
                self.selector.last_seen = now
                self.selector.pinned = True
                self.mouse.update(down=False)
                for hand in self.manip.hands:
                    hand.pinch.reset()
                    hand.fist.since = hand.palm.since = None
                self.say("Outline ready. Pinch it or click and drag to lift.", now)
            elif key in (8, 127) and self.editor.undo():
                self.submit_edit(now)
            elif key in (ord("m"), ord("M")):
                self.selector.cycle(now)
            return
        obj = self.active()
        m = self.manip
        if key == 27:                                          # Esc
            self.cancel_selection()
            if obj is not None:
                m.remove(obj)
                self.say("Released back to reality (original shown again)", now)
            else:
                self.say("Selection cancelled", now)
        elif key in (ord("s"), ord("S")) and self.last_frame is not None:
            if any(o.holders for o in m.objects):
                self.say("Place your object before precise selection.", now)
                return
            self.cancel_selection()
            self.editor = SelectionEditor(self.last_frame)
            self.mouse.update(down=False)
            self.say("View paused. Draw a box around any object, or click inside it.", now, 4)
        elif key in (ord("r"), ord("R")) and obj is not None:
            m.reset(obj)
            self.say("Reset", now)
        elif key in (ord("h"), ord("H")) and obj is not None:
            self.say(f"Object {m.toggle_visibility(obj)}", now)
        elif key in (ord("c"), ord("C")) and obj is not None:
            self.say("Duplicated (experimental)" if m.duplicate(obj) else "Object limit reached", now)
        elif key in (ord("z"), ord("Z")) and obj is not None:
            outcome = m.toggle_freeze(obj)
            if outcome:
                self.say(outcome.capitalize(), now)
        elif key in (ord("m"), ord("M")):
            self.selector.cycle(now)
        elif key in (ord("b"), ord("B")) and obj is not None:
            self.say(f"Background: {obj.group.reconstruction.cycle()}", now)
        elif key in (ord("p"), ord("P")):
            seen = np.maximum(self.person(), self.last_zone.astype(np.float32))
            self.memory.capture_plate(self.last_frame, seen, now)
            self.say("Clean plate captured - valid only if the objects were physically removed", now, 3)
        elif key in (ord("o"), ord("O")):
            self.occlusion = {"hands": "person", "person": "off", "off": "hands"}[self.occlusion]
            self.say(f"Occlusion: {self.occlusion} in front of objects", now)
        elif key in (ord("t"), ord("T")):
            m.roll_rotation = not m.roll_rotation
            self.say(f"One-hand twist rotation {'on' if m.roll_rotation else 'off'}", now)
        elif key in (ord("d"), ord("D")):
            self.debug = not self.debug
        elif key in (ord("k"), ord("K")):
            self.show_keys = not self.show_keys
        elif key == 9 and m.objects:                            # Tab
            ids = [o.id for o in m.objects]
            m.active = ids[(ids.index(m.active) + 1) % len(ids)] if m.active in ids else ids[0]
        elif key in (ord("x"), ord("X")):
            self.cancel_selection()
            for o in list(m.objects):
                m.remove(o)
            self.say("All objects released", now)
        elif key in (ord("e"), ord("E")) and self.worker.error:
            self.cancel_selection()
            self.refining = None
            self.worker.retry()
            self.say("Retrying segmentation", now)

    def cancel_selection(self):
        self.selector.clear()
        self.clicked = None
        self.pending_lock = None
        self.worker.discard_pending()

    def submit_edit(self, now):
        editor = self.editor
        self.worker.discard_pending()
        if not editor.has_prompt:
            self.selector.clear()
            return
        job = self.selector.prompt(now, editor.frame, editor.positives, editor.negatives,
                                   box=editor.box, embedding=editor.embedding, explicit=True)
        job.max_area = .95
        self.worker.request(job)

    def edit_step(self, now):
        editor = self.editor
        polled = self.worker.poll()
        if polled is not None and polled[0] == "select":
            result = polled[1]
            # Cache the same frozen image even when a newer correction supersedes it.
            if result.embedding is not None and np.array_equal(result.embedding.frame, editor.frame):
                editor.embedding = result.embedding
            self.selector.accept(result, now)
        out = editor.frame.copy()
        if self.selector.mask is not None:
            mask = self.selector.mask
            out[mask] = (out[mask] * .75 + np.array(CYAN) * .25).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, CYAN, 2)
        box = editor.box
        if editor.drag is not None and editor.cursor is not None:
            box = (*editor.drag, *editor.cursor)
        if box is not None:
            cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), CYAN, 1)
        for points, colour in ((editor.positives, GREEN), (editor.negatives, ORANGE)):
            for x, y in points:
                cv2.circle(out, (x, y), 5, (0, 0, 0), 3)
                cv2.circle(out, (x, y), 4, colour, -1)
        status = ("Selection unavailable. Esc to return; E to retry." if self.worker.error else
                  "Finding outline... You can keep adding corrections." if self.selector.state == "ANALYZING" else
                  "No outline yet. Draw a box, or click inside the object." if self.selector.mask is None else
                  "Left click includes; right click excludes. Enter accepts this outline.")
        draw_hud(out, status, "Precise selection / paused", stage=1,
                 hint="Drag box / click include / right click exclude / Backspace undo / M outline / Enter done / Esc cancel")
        return out

    def on_mouse(self, event, x, y, flags, _param=None):
        now = time.perf_counter() - (self.started_wall or 0)
        x, y = min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)
        if self.editor is not None:
            point = self.editor.point(x, y)
            if event == cv2.EVENT_MOUSEMOVE:
                self.editor.cursor = point
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.editor.drag = self.editor.cursor = point
            elif event == cv2.EVENT_LBUTTONUP and self.editor.finish_drag(point):
                self.submit_edit(now)
            elif event == cv2.EVENT_RBUTTONDOWN and self.editor.exclude(point):
                self.submit_edit(now)
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse.update(point=(x, y), moved=now)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse.update(point=(x, y), down=True, moved=now)
            self.mouse_until = now + 10
            if (self.manip.hit((x, y)) is None and
                    (self.selector.mask is None or not self.selector.mask[y, x])):
                self.clicked = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse.update(point=(x, y), down=False, moved=now)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selector.cycle(now)
        elif event == cv2.EVENT_MOUSEWHEEL and self.active() is not None:
            obj = self.active()
            # HighGUI packs a signed 16-bit wheel delta into the upper flags word.
            # getMouseWheelDelta is a C++ helper absent from Python OpenCV bindings.
            delta = (flags >> 16) & 0xFFFF
            delta = delta - 0x10000 if delta & 0x8000 else delta
            if delta == 0:
                return
            up = delta > 0
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                obj.angle += 10 if up else -10
            else:
                obj.scale = float(np.clip(obj.scale * (1.1 if up else 1 / 1.1),
                                          self.manip.min_scale, self.manip.max_scale))
            if obj.mode == "home":
                obj.mode = "floating"
            self.manip.two.pop(obj.id, None)
            self.manip._rebase(obj)

    started_wall = None
    last_frame = last_zone = None

    # ------------------------------------------------------------------ drawing
    def draw_feedback(self, out, frame, now, aim, hovering, person, zone):
        s = self.selector
        m = self.manip
        # Candidate preview: thin outline + faint tint = TARGETED.
        mask = s.mask
        if mask is not None:
            tint = out[mask].astype(np.float32) * .78 + np.array(CYAN, np.float32) * .22
            out[mask] = tint.astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, CYAN, 1, cv2.LINE_AA)
        if hovering is not None and not hovering.holders:
            cv2.circle(out, ipt(hovering.position), 5, GREEN, 1, cv2.LINE_AA)
        # Show actual detections in normal use; grace-period poses are stale, not live hands.
        real_hands = [hand for hand in m.hands[:2] if hand.present and not hand.missing]
        for hand in real_hands:
            draw_hand_skeleton(out, hand.points, hand.pinch.pinched)
        # Reticle at the aim point; an arc shows dwell/analysis progress.
        if aim is not None:
            centre = ipt(aim)
            if self.pointer_slot in (0, 1):
                cv2.line(out, ipt(m.hands[self.pointer_slot].points[8]), centre, CYAN, 1, cv2.LINE_AA)
            text(out, "TARGET", (centre[0] + 15, centre[1] - 12), CYAN, .35)
            colour = CYAN if s.state == "PREVIEW" else WHITE
            cv2.circle(out, centre, 7, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.circle(out, centre, 7, colour, 1, cv2.LINE_AA)
            if s.state == "AIMING" and s.anchor_since is not None:
                p = min(1.0, (now - s.anchor_since) / s.dwell)
                cv2.ellipse(out, centre, (12, 12), -90, 0, 360 * p, WHITE, 2, cv2.LINE_AA)
            elif s.state == "ANALYZING":
                start = (now * 360) % 360
                cv2.ellipse(out, centre, (12, 12), start, 0, 90, CYAN, 2, cv2.LINE_AA)
        # Held objects: tether from fingers; transform indicators while two-handed.
        for slot, hand in enumerate(m.hands):
            if hand.held is None:
                continue
            obj = m.object(hand.held)
            if obj is None:
                continue
            cv2.line(out, ipt(hand.pinch_point), ipt(obj.position), GREEN, 1, cv2.LINE_AA)
            cv2.circle(out, ipt(hand.pinch_point), 4, GREEN, -1, cv2.LINE_AA)
            if len(obj.holders) > 1:
                text(out, f"x{obj.scale:.2f}  {obj.angle:+.0f} deg", ipt(obj.position + (-40, -obj.half_extent()[1] - 8)), GREEN, .45)
            elif abs(obj.angle) >= 1:
                text(out, f"{obj.angle:+.0f} deg", ipt(obj.position + (-18, -obj.half_extent()[1] - 8)), GREEN, .4)
        # Fist / palm hold progress rings (so an action never happens by surprise).
        for hand in m.hands[:2]:
            if not hand.present:
                continue
            for gesture, colour, label in ((hand.fist, ORANGE, "hide/show"), (hand.palm, CYAN, "reset")):
                p = gesture.progress(now)
                if p > .15 and m.objects:
                    c = ipt(hand.palm_center)
                    cv2.ellipse(out, c, (26, 26), -90, 0, 360 * p, colour, 3, cv2.LINE_AA)
                    text(out, label, (c[0] - 28, c[1] + 42), colour, .4)
        hidden = sum(1 for o in m.objects if not o.visible)
        lost = [g for g in self.groups() if g.tracker.state.startswith("LOST")]
        message = self.messages[-1][0] if self.messages and now < self.messages[-1][1] else None
        summary = "Hand detected" if real_hands else "No hand detected"
        if len(real_hands) == 2:
            summary = "2 hands detected"
        if self.pointer_slot == 2:
            summary = "Mouse control"
        if m.active is not None:
            summary = f"Object {m.active} selected"
        if hidden:
            summary += f" / {hidden} hidden"
        holding = any(o.holders for o in m.objects)
        previewing = s.state == "PREVIEW" or self.pending_lock is not None
        selecting = s.state in ("AIMING", "ANALYZING", "NO_OBJECT")
        if not holding and (previewing or selecting):
            summary = "New object outline" if previewing else "Finding object" if s.state == "ANALYZING" else summary
        stage = 3 if holding else 2 if previewing else 1 if selecting else 3 if m.objects else 1
        hint = ("Open fingers to place / R reset / K help / Q quit" if holding else
                "M change outline / S precise selection / K help / Q quit" if previewing else
                "Click to select / S precise selection / K help / Q quit" if selecting else
                "R reset / H hide / S precise selection / Esc restore / K help / Q quit" if m.objects else
                "Click to select / S precise selection / K help / Q quit")
        self.hud_bottom = draw_hud(out, self.status(now), summary, message,
                                  bool(lost), self.show_keys and not self.debug, stage=stage, hint=hint)
        if self.debug:
            self.draw_debug(out, now, aim, person, zone)

    def status(self, now):
        s, m = self.selector, self.manip
        if self.worker.error:
            return "Object selection unavailable. Press E to retry loading."
        if not self.worker.ready and self.worker.model is None:
            return "Getting ready... Wait for object selection to finish loading."
        held = [o for o in m.objects if o.holders]
        if held:
            if len(held[0].holders) > 1:
                return "Move hands apart to enlarge. Turn your hands to rotate."
            return "Keep pinching and move your hand. Open your fingers to place it."
        if self.pending_lock:
            return "Keep pinching. Your object is still being found..."
        if s.state == "PREVIEW":
            c = s.candidate
            if c.score < .50 or c.stability < .70:
                return "Uncertain outline. Press S to correct it, or M for another outline."
            return "Outline looks right? Pinch to lift, or click and drag. S corrects it."
        if s.state == "ANALYZING":
            return "Finding your object... Keep the target ring in place."
        if s.state == "AIMING":
            return "Keep the target ring on the object. Hold your finger still."
        if s.state == "NO_OBJECT":
            return "No outline at this point. Click to retry, or S to frame the object."
        if m.objects:
            obj = self.active()
            if obj is not None and not obj.visible:
                return "Your object is hidden. Press H to show it again."
            return "Object placed. Pinch its image to pick it up again."
        if not any(h.present for h in m.hands[:2]) and self.pointer_slot != 2:
            return "Raise one hand with your index finger extended, or use the mouse."
        return "Point at an object. Put the TARGET ring over its image."

    def draw_debug(self, out, now, aim, person, zone):
        m, s = self.manip, self.selector
        lines = [f"FPS {self.fps:.1f}   frame {self.frame_ms:.0f} ms (hands{'x2' if self.two_hand_mode else ''} {self.hands_ms:.0f}, "
                 f"person {self.person_ms:.0f}, app {self.app_ms:.0f})",
                 f"model: encoder {self.worker.last_encoder_ms:.0f} ms decoder {self.worker.last_decoder_ms:.0f} ms"
                 f"   aim->preview {s.latency_ms or 0:.0f} ms   busy {self.worker.busy}",
                 f"selection {s.state}  snapshot {s.snapshot_id} prompt {s.prompt_id}  stale rejected "
                 f"{self.counters['rejected_stale']}"]
        if s.result is not None:
            for i, c in enumerate(s.result.candidates):
                mark = ">" if i == s.choice else " "
                lines.append(f" {mark}{i} area {c.area:.3f} score {c.score:.2f} stab {c.stability:.2f} "
                             f"hand {c.hand_overlap:.2f} body {c.person_overlap:.2f} {c.rank_reason}")
        for slot, hand in enumerate(m.hands):
            if hand.present:
                ratio = "--" if hand.ratio is None else f"{hand.ratio:.2f}"
                lines.append(f"hand{slot}{'(mouse)' if hand.virtual else ''}: pinch {ratio} {hand.pinch.state} "
                             f"v=({hand.velocity[0]:.0f},{hand.velocity[1]:.0f}) roll {hand.roll:.0f} "
                             f"tip {ipt(hand.pinch_point)} held {hand.held}")
        for obj in m.objects:
            g = obj.group
            lines.append(f"obj{obj.id}{'*' if obj.id == m.active else ''}{' dup' if obj.duplicate else ''}: "
                         f"{obj.mode} pos {ipt(obj.position)} x{obj.scale:.2f} {obj.angle:+.0f}deg "
                         f"v=({obj.velocity[0]:.0f},{obj.velocity[1]:.0f}) vis {obj.visible}")
            lines.append(f"   track {g.tracker.state} ncc {g.tracker.score:.2f} occl {g.tracker.occluded_fraction:.2f} "
                         f"off {ipt(g.tracker.offset)} | bg: {g.reconstruction.source} | {g.refine}")
        y = self.hud_bottom + 16
        for line in lines:
            if y >= out.shape[0] - 132:
                break
            text(out, line, (10, y), (200, 255, 200), .36)
            y += 15
        for slot, hand in enumerate(m.hands[:2]):
            if hand.present:
                for p in negative_prompts(hand.points):
                    cv2.drawMarker(out, ipt(p), (60, 60, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
        if s.result is not None and s.result.point is not None:
            cv2.drawMarker(out, ipt(s.result.point), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        # Thumbnails: raw binary mask, person segmentation, active sprite alpha.
        th, tw = 90, 120
        y0 = out.shape[0] - th - 24
        thumbs = []
        raw = s.mask if s.mask is not None else (self.last_candidate[0] if self.last_candidate else None)
        if raw is not None:
            thumbs.append(("binary mask", raw.astype(np.uint8) * 255))
        thumbs.append(("person seg", (person * 255).astype(np.uint8)))
        thumbs.append(("hand zone", zone.astype(np.uint8) * 255))
        obj = self.active()
        if obj is not None:
            thumbs.append(("sprite alpha", (obj.sprite.alpha * 255).astype(np.uint8)))
        for i, (label, image) in enumerate(thumbs):
            x0 = 10 + i * (tw + 8)
            if x0 + tw > out.shape[1]:
                break
            out[y0:y0 + th, x0:x0 + tw] = cv2.cvtColor(cv2.resize(image, (tw, th), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            text(out, label, (x0, y0 - 4), WHITE, .33)

    fps = frame_ms = hands_ms = person_ms = app_ms = 0.0
    two_hand_mode = False
    _person = _person_source = None
    _shape = None

    def close(self):
        self.inpainter.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------- camera loop
def create_window(width, height):
    # Enlarge only the display; inference keeps using the 640x480 camera frame.
    scale = 2.0
    if sys.platform == "win32":
        import ctypes
        screen = ctypes.windll.user32
        scale = min(scale, screen.GetSystemMetrics(0) * .92 / width, screen.GetSystemMetrics(1) * .86 / height)
    size = (int(width * scale), int(height * scale))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, *size)
    return size


def open_camera(index, backend):
    choices = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    backends = ([choices[backend]] if backend != "auto" else
                [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY])
    for api in backends:
        camera = cv2.VideoCapture(index, api)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(10):
                ok, frame = camera.read()
                if ok and frame is not None and frame.size:
                    print(f"Camera {index}: {camera.getBackendName()}, {frame.shape[1]}x{frame.shape[0]}", flush=True)
                    return camera, frame
        camera.release()
    raise RuntimeError(
        f"Cannot read webcam {index}. Close other camera apps, check the privacy shutter and "
        "Windows Settings > Privacy & security > Camera. Try --camera 1 or --backend msmf.")


def run(args):
    tracker = tracker2 = camera = worker = person_model = app = None
    frames = hand_frames = 0
    stamps = deque(maxlen=31)
    stage = deque(maxlen=60)
    started = None
    try:
        tracker = HandTracker(num_hands=1 if args.hands != "2" else 2)
        # MediaPipe re-runs its palm detector on every frame while it sees fewer hands than
        # num_hands (measured ~60-125 ms here vs ~20 ms while tracking one hand). So track
        # one hand while pointing, and switch to two only while an object is held.
        tracker2 = HandTracker(num_hands=2) if args.hands == "auto" else None
        if not args.no_person:
            try:
                person_model = PersonSegmenter()
            except Exception as error:     # occlusion degrades to "off", app keeps running
                print(f"Person segmentation unavailable ({error}); occlusion disabled.", file=sys.stderr)
        camera, frame = open_camera(args.camera, args.backend)
        height, width = frame.shape[:2]
        if height < 240 or width < 360:
            raise RuntimeError("Camera must deliver at least 360x240 pixels.")
        worker = SegmentationWorker(args.model_dir, args.threads)
        worker.warm_up()
        app = App(width, height, worker, args)
        if person_model is None:
            app.occlusion = "off"
        fullscreen = False
        if not args.headless:
            window_size = create_window(width, height)
            cv2.setMouseCallback(WINDOW, app.on_mouse)
        started = previous = time.perf_counter()
        app.started_wall = started
        next_report = 5.0
        while True:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Webcam resolution changed during the session. Restart to realign the scene.")
            captured = time.perf_counter()
            dt, now = captured - previous, captured - started
            previous = captured
            # Mirror FIRST: landmarks, prompts, masks and display all share this one
            # coordinate system, so the image behaves like a mirror for the user.
            frame = cv2.flip(frame, 1)
            t0 = time.perf_counter()
            holding = any(o.holders for o in app.manip.objects)
            active_tracker = tracker2 if tracker2 is not None and holding else tracker
            observations = active_tracker.detect_all(frame, captured)
            t1 = time.perf_counter()
            timing = {"person": 0.0}

            def segment_person(frame=frame, captured=captured):
                started_seg = time.perf_counter()
                mask = person_model.segment(frame, captured) if person_model else None
                timing["person"] = (time.perf_counter() - started_seg) * 1e3
                return mask

            out = app.step(frame, now, dt, observations, segment_person)
            t3 = time.perf_counter()
            t2 = t1 + timing["person"] / 1e3
            app.two_hand_mode = active_tracker is tracker2
            hand_frames += bool(observations)
            stamps.append(t3)
            app.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            app.hands_ms, app.person_ms = (t1 - t0) * 1e3, timing["person"]
            app.app_ms = (t3 - t1) * 1e3 - timing["person"]
            app.frame_ms = (t3 - captured) * 1e3
            stage.append((app.hands_ms, app.person_ms, app.app_ms, app.frame_ms))
            frames += 1
            if now >= next_report:
                avg = np.mean(stage, axis=0)
                print(f"Live {app.fps:.1f} FPS | hands{'x2' if app.two_hand_mode else ''} {avg[0]:.0f} ms person {avg[1]:.0f} ms "
                      f"app {avg[2]:.0f} ms (frame {avg[3]:.0f} ms) | hands seen {len(observations)} | "
                      f"objects {len(app.manip.objects)} | model ready {worker.model is not None}", flush=True)
                next_report = now + 5
            if not args.headless:
                cv2.imshow(WINDOW, out)
                key = cv2.waitKeyEx(1)
                if key != -1:
                    key &= 0xFF
                    if key in (ord("q"), ord("Q")):
                        break
                    if key in (ord("f"), ord("F")):
                        fullscreen = not fullscreen
                        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                        if not fullscreen:
                            cv2.resizeWindow(WINDOW, *window_size)
                    else:
                        app.key(key, now)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.seconds and time.perf_counter() - started >= args.seconds:
                break
            ok, frame = camera.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Webcam stopped delivering frames. Reconnect it and restart.")
    finally:
        finished = time.perf_counter()
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        if tracker2 is not None:
            tracker2.close()
        if person_model is not None:
            person_model.close()
        cv2.destroyAllWindows()
        if app is not None:
            app.close()
        if worker is not None:
            worker.close()
        if started is not None and frames:
            print(f"Run: {frames} frames / {finished - started:.1f}s = {frames / (finished - started):.1f} FPS; "
                  f"hands in {hand_frames} frames; "
                  f"{app.counters if app else ''}; manipulation {app.manip.stats if app else ''}", flush=True)
        print(f"Shutdown complete in {time.perf_counter() - finished:.2f}s", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pinch-threshold", type=float, default=.30)
    parser.add_argument("--release-threshold", type=float, default=.45)
    parser.add_argument("--min-scale", type=float, default=.25)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument("--max-area", type=float, default=.30,
                        help="largest plausible object as a fraction of the frame")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--no-person", action="store_true", help="disable person segmentation/occlusion")
    parser.add_argument("--hands", choices=("auto", "1", "2"), default="auto",
                        help="auto: track 1 hand, 2 only while holding (faster)")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    values = (args.pinch_threshold, args.release_threshold, args.min_scale, args.max_scale, args.max_area, args.seconds)
    if not all(math.isfinite(v) for v in values):
        parser.error("Numeric settings must be finite.")
    if not 0 < args.pinch_threshold < args.release_threshold:
        parser.error("Thresholds must satisfy 0 < pinch < release.")
    if not 0 < args.min_scale <= 1 <= args.max_scale:
        parser.error("Scale bounds must satisfy 0 < min <= 1 <= max.")
    if not 0 < args.max_area <= 1:
        parser.error("--max-area must be in (0, 1].")
    if args.seconds < 0 or (args.headless and args.seconds <= 0):
        parser.error("Headless runs require a positive --seconds duration.")
    if args.camera < 0:
        parser.error("--camera must be a non-negative device index.")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Telekinesis CV: {error}", file=sys.stderr)
        sys.exit(1)
