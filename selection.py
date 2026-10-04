"""Temporal gesture logic and touch-selection state, separate from inference and drawing.

Why not simply `if distance < threshold: grab()`?  Landmark distances jitter every frame.
A single noisy low reading would grab, a single high reading would drop, and a reading
oscillating around the threshold would grab/release repeatedly. The state machine below
needs *sustained* evidence to change state (dwell), uses two thresholds (hysteresis), and
needs the fingers to have been open before a new pinch counts (arming).
"""
import math

import cv2
import numpy as np

from segmentation import Job


class PinchGesture:
    """OPEN -> PINCH_CANDIDATE -> PINCHED -> RELEASE_CANDIDATE -> OPEN."""

    def __init__(self, close=.30, release=.45, close_time=.07, release_time=.06):
        self.close, self.release = close, release
        self.close_time, self.release_time = close_time, release_time
        self.reset()

    def reset(self):
        self.state = "OPEN"
        self.since = None
        self.armed = False           # fingers must be seen open before a pinch counts
        self.release_started = None

    @property
    def pinched(self):
        return self.state in ("PINCHED", "RELEASE_CANDIDATE")

    @property
    def engaged(self):
        """True once the fingers start closing: selection must stop retargeting."""
        return self.state != "OPEN"

    def update(self, ratio, now):
        """Return "press", "release", "lost" or None for this frame."""
        if ratio is None or not math.isfinite(ratio):
            was = self.pinched
            self.reset()
            return "lost" if was else None
        state = self.state
        if state == "OPEN":
            if ratio >= self.release:
                self.armed = True
            elif ratio <= self.close and self.armed:
                self.state, self.since = "PINCH_CANDIDATE", now
        elif state == "PINCH_CANDIDATE":
            if ratio > self.close:
                # The hysteresis band alone can never complete a pinch.
                self.state = "OPEN"
            elif now - self.since >= self.close_time:
                self.state, self.armed = "PINCHED", False
                return "press"
        elif state == "PINCHED":
            if ratio >= self.release:
                self.state, self.since = "RELEASE_CANDIDATE", now
                self.release_started = now
        elif state == "RELEASE_CANDIDATE":
            if ratio < self.release:
                self.state = "PINCHED"
            elif now - self.since >= self.release_time:
                self.state, self.armed = "OPEN", True
                return "release"
        return None


class HoldGesture:
    """A pose (fist / open palm) must be held for `duration` to fire once; then cooldown."""

    def __init__(self, duration, cooldown=1.0):
        self.duration, self.cooldown = duration, cooldown
        self.since = None
        self.fired = False
        self.ready_at = 0.0

    def update(self, active, now):
        if not active:
            self.since, self.fired = None, False
            return False
        if self.since is None:
            self.since = now
        if not self.fired and now >= self.ready_at and now - self.since >= self.duration:
            self.fired, self.ready_at = True, now + self.cooldown
            return True
        return False

    def progress(self, now):
        if self.since is None or self.fired or now < self.ready_at:
            return 0.0
        return min(1.0, (now - self.since) / self.duration)


def aim_point(points, shape, lead=.28):
    """Touch-selection aim: slightly AHEAD of the index fingertip along the finger.

    The landmark sits on the finger itself; pixels there are skin, not the object.
    Placing the prompt ~0.28 palm lengths beyond the tip (along PIP -> tip) puts it on
    what the finger is touching in the image, and the reticle is no longer hidden
    under the finger. It is still touch selection, not ray casting.
    """
    tip, pip = points[8], points[6]
    direction = tip - pip
    norm = float(np.linalg.norm(direction))
    palm = (np.linalg.norm(points[0] - points[9]) + np.linalg.norm(points[5] - points[17])) / 2
    if norm < 1e-6 or not np.isfinite(palm):
        point = tip.copy()
    else:
        point = tip + direction / norm * lead * palm
    height, width = shape[:2]
    return np.clip(point, [0, 0], [width - 1, height - 1])


def hand_zone(shape, hands, extra=0):
    """Approximate hand/forearm region from landmarks (NOT a segmentation)."""
    zone = np.zeros(shape[:2], np.uint8)
    for points in hands:
        if points is None or not np.isfinite(points).all():
            continue
        palm = np.linalg.norm(points[0] - points[9])
        one = np.zeros(shape[:2], np.uint8)
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(one, hull, 255)
        size = int(np.clip(palm * .35 + extra, 9, 80)) | 1
        one = cv2.dilate(one, np.ones((size, size), np.uint8))
        wrist, direction = points[0], points[0] - points[9]
        cv2.line(one, tuple(np.rint(wrist).astype(int)),
                 tuple(np.rint(wrist + direction * 3).astype(int)), 255, max(15, int(palm * .8)))
        zone = np.maximum(zone, one)
    return zone.astype(bool)


def negative_prompts(points):
    """Points on the pointing hand tell the model 'not this' (index PIP, palm, thumb)."""
    return [tuple(points[i]) for i in (6, 9, 4)]


class Selector:
    """AIMING -> ANALYZING -> PREVIEW; latches while the user is pinching.

    The prompt is taken from the LIVE frame at the moment the aim settles; the hand is
    in that frame, so negative prompts and a hand-overlap filter keep it out of the mask.
    A cached encoding is reused for new aim points while it is recent and the aim is
    not where the hand was in that snapshot.
    """

    def __init__(self, dwell=.22, radius=16, reuse_seconds=2.5, carousel=1.6):
        self.dwell, self.radius, self.reuse_seconds = dwell, radius, reuse_seconds
        self.carousel = carousel       # keep pointing at a preview -> next outline
        self.preview_since = None
        self.snapshot_id = self.prompt_id = 0
        self.expected_snapshot = None
        self.embedding = None
        self.result = None
        self.choice = None
        self.latched = None            # dilated preview mask used for hover tolerance
        self.anchor = self.anchor_since = None
        self.sent = False
        self.left_since = None
        self.last_seen = None
        self.state = "IDLE"
        self.version = 0               # bumps whenever the visible candidate changes
        self.submitted_at = None
        self.latency_ms = None         # aim settled -> preview shown (measured)

    @property
    def candidate(self):
        if self.result is None or self.choice is None:
            return None
        return self.result.candidates[self.choice]

    @property
    def mask(self):
        c = self.candidate
        return None if c is None else c.mask

    def clear(self, state="IDLE"):
        self.prompt_id += 1            # late results for the old intent are now stale
        self.result = self.choice = self.latched = None
        self.anchor = self.anchor_since = self.left_since = None
        self.sent = False
        self.state = state
        self.expected_snapshot = None
        self.version += 1

    def reset_dwell(self, now):
        """Unobserved time (camera gap) must not count as a steady aim."""
        if self.anchor is not None and not self.sent:
            self.anchor_since = now

    def cycle(self, now=None):
        """Show the next alternative outline (ordered by size, wrapping around)."""
        if self.result:
            order = sorted((i for i, c in enumerate(self.result.candidates) if c.selectable),
                           key=lambda i: self.result.candidates[i].area)
            if not order or self.choice is None:
                return
            position = order.index(self.choice) if self.choice in order else -1
            self.choice = order[(position + 1) % len(order)]
            self._latch()
            self.version += 1
            self.preview_since = now

    def _latch(self):
        mask = self.mask
        self.latched = None if mask is None else cv2.dilate(
            mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)

    def update(self, now, aim, frame, hands, zone, blocked, engaged, busy_encoding=False,
               negatives=()):
        """Advance selection; return a segmentation Job to submit, or None.

        aim: (x, y) or None; hands: landmark arrays in this frame; zone: hand-zone mask;
        blocked: callable (x, y) -> True where selection is not allowed (hidden originals);
        engaged: the aiming input has started a pinch -> freeze the target.
        """
        if aim is None:
            if self.state == "AIMING":
                self.state = "IDLE"    # nothing was submitted; nothing to keep
            elif self.last_seen is not None and now - self.last_seen > .8 and self.state != "IDLE":
                self.clear()           # a preview survives brief hand loss, not longer
            self.anchor = None
            return None
        self.last_seen = now
        if engaged:
            return None                # never retarget while the fingers are closing
        aim = np.asarray(aim, float)
        h, w = frame.shape[:2]
        x, y = int(aim[0]), int(aim[1])
        if self.latched is not None:
            if self.latched[y, x]:
                self.left_since = None
                self.anchor = None
                # Hands-only correction: keep pointing steadily near the prompt and the
                # preview steps through the other plausible outlines (part/whole).
                near = (self.result.point is not None and
                        np.linalg.norm(aim - np.asarray(self.result.point)) <= 2.5 * self.radius)
                if (near and self.preview_since is not None and len(self.result.candidates) > 1
                        and now - self.preview_since >= self.carousel):
                    self.cycle(now)
                elif not near:
                    self.preview_since = now
                return None
            if self.left_since is None:
                self.left_since = now
            if now - self.left_since < .15:
                return None            # brief jitter outside the silhouette is forgiven
            self.clear("AIMING")
        if blocked is not None and blocked(x, y):
            if self.state in ("AIMING", "ANALYZING"):
                self.clear()
            return None
        if self.anchor is None or np.linalg.norm(aim - self.anchor) > self.radius:
            if self.state == "ANALYZING":
                self.clear("AIMING")   # aim moved on before the answer: abandon it
            self.anchor, self.anchor_since, self.sent = aim.copy(), now, False
            self.state = "AIMING"
            return None
        if self.sent or now - self.anchor_since < self.dwell:
            return None
        self.sent = True
        self.submitted_at = now
        self.state = "ANALYZING"
        self.prompt_id += 1
        point = tuple(self.anchor)
        job = Job("select", self.snapshot_id, self.prompt_id, positives=(point,),
                  negatives=tuple(negatives))
        if self._can_reuse(now, aim, frame, zone):
            job.embedding = self.embedding
            job.snapshot_id = self.embedding.snapshot_id
            job.negatives = tuple(negatives_from_embedding(self.embedding))
        else:
            self.snapshot_id += 1
            job.snapshot_id = self.snapshot_id
            job.frame, job.hand_points, job.hand_zone = frame, tuple(hands), zone
            job.captured_at = now
        self.expected_snapshot = job.snapshot_id
        return job

    def _can_reuse(self, now, aim, frame, zone):
        e = self.embedding
        if e is None or now - e.captured_at > self.reuse_seconds or e.frame.shape != frame.shape:
            return False
        x, y = int(aim[0]), int(aim[1])
        if e.hand_zone[y, x]:
            return False               # the snapshot shows the hand at this spot
        # The region around the aim must still look like the snapshot (scene unchanged).
        h, w = frame.shape[:2]
        y0, y1, x0, x1 = max(0, y - 30), min(h, y + 31), max(0, x - 30), min(w, x + 31)
        visible = ~(zone[y0:y1, x0:x1] | e.hand_zone[y0:y1, x0:x1])
        if visible.mean() < .4:
            return False
        diff = cv2.absdiff(frame[y0:y1, x0:x1], e.frame[y0:y1, x0:x1]).mean(axis=2)
        return float(diff[visible].mean()) < 14

    def accept(self, result, now=None):
        """Accept a finished job only if it answers the CURRENT prompt."""
        if (result.prompt_id != self.prompt_id or result.snapshot_id != self.expected_snapshot
                or self.state != "ANALYZING"):
            return False
        if now is not None and self.submitted_at is not None:
            self.latency_ms = (now - self.submitted_at) * 1000
        if result.embedding is not None:
            self.embedding = result.embedding
        self.result, self.choice = result, result.choice
        self.state = "PREVIEW" if self.choice is not None else "NO_OBJECT"
        self.preview_since = now
        self._latch()
        self.version += 1
        return True


def negatives_from_embedding(embedding):
    negatives = []
    for points in embedding.hand_points:
        negatives += negative_prompts(points)
    return negatives
