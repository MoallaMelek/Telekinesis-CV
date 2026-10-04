"""Extracted real-object sprites: grab, move, two-hand scale/rotate, throw, render.

Everything here is our own geometry: affine transforms, velocity estimation, hand
identity matching, screen-space physics and premultiplied-alpha compositing.
"""
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations
import math

import cv2
import numpy as np

from hand_tracker import estimate_velocity, smooth_pointer, update_pinch
from selection import HoldGesture, PinchGesture, aim_point


# ----------------------------------------------------------------------------- objects
@dataclass
class Sprite:
    """RGBA content cut from a real camera frame: OBJECT_RGBA = pixels x alpha."""
    bgr: np.ndarray            # uint8 crop of the camera frame
    alpha: np.ndarray          # float32 0..1 soft mask for the crop
    anchor: np.ndarray         # rotation/scale centre inside the crop (mask centroid)

    @property
    def premultiplied(self):
        # Premultiplying (colour x alpha) before warping stops the transparent pixels'
        # colours from bleeding in as a dark/coloured fringe during interpolation.
        return self.bgr.astype(np.float32) * self.alpha[..., None]

    @property
    def size(self):
        return np.array(self.bgr.shape[1::-1], float)


@dataclass
class Group:
    """One PHYSICAL object: its lock-time mask, tracker and background reconstruction."""
    id: int
    mask: np.ndarray
    origin: np.ndarray         # mask centroid in the lock frame
    tracker: object
    reconstruction: object
    snapshot_id: int = 0
    refine: str = "waiting for your hand to leave the object"
    refined: bool = False
    clear_since: float = None
    score: float = 0.0
    stability: float = 0.0


@dataclass
class ManipulatedObject:
    id: int
    group: Group
    sprite: Sprite
    duplicate: bool = False
    position: np.ndarray = None
    scale: float = 1.0
    angle: float = 0.0          # degrees, image plane only (2D rotation)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    spin: float = 0.0           # degrees / second
    mode: str = "home"          # home | held | floating | flying
    visible: bool = True
    fade: float = 1.0
    holders: set = field(default_factory=set)

    def home_position(self):
        return self.group.origin + self.group.tracker.offset

    def matrix(self):
        """2x3 affine: crop pixel -> frame pixel.  p' = T(position) R(angle) S(scale) T(-anchor) p"""
        a = math.radians(self.angle)
        c, s = math.cos(a) * self.scale, math.sin(a) * self.scale
        linear = np.array([[c, -s], [s, c]])
        return np.hstack([linear, (self.position - linear @ self.sprite.anchor)[:, None]])

    def half_extent(self):
        w, h = self.sprite.size * self.scale
        a = math.radians(self.angle)
        return np.array([abs(math.cos(a)) * w + abs(math.sin(a)) * h,
                         abs(math.sin(a)) * w + abs(math.cos(a)) * h]) / 2

    def contains(self, point, margin=14):
        """Hit test against the transformed alpha, with a forgiving margin."""
        inverse = cv2.invertAffineTransform(self.matrix())
        x, y = inverse @ np.array([point[0], point[1], 1.0])
        h, w = self.sprite.alpha.shape
        m = margin / max(self.scale, 1e-3)
        if not (-m <= x < w + m and -m <= y < h + m):
            return False
        x0, x1 = int(max(0, x - m)), int(min(w, x + m + 1))
        y0, y1 = int(max(0, y - m)), int(min(h, y + m + 1))
        return x1 > x0 and y1 > y0 and bool((self.sprite.alpha[y0:y1, x0:x1] > .5).any())


def render_sprite(out, obj, glow=None):
    """Warp pixels AND alpha with the same affine matrix, then alpha-composite."""
    if obj.fade <= .01:
        return None
    h, w = out.shape[:2]
    matrix = obj.matrix()
    sh, sw = obj.sprite.alpha.shape
    corners = np.array([[0, 0, 1], [sw, 0, 1], [0, sh, 1], [sw, sh, 1]], float) @ matrix.T
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    local = matrix.copy()
    local[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    colour = cv2.warpAffine(obj.sprite.premultiplied, local, size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpAffine(obj.sprite.alpha, local, size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None] * obj.fade
    roi = out[y0:y1, x0:x1].astype(np.float32)
    # "Over" operator with premultiplied colour: out = src + dst * (1 - alpha_src)
    blended = colour * obj.fade + roi * (1 - alpha)
    if glow is not None:
        edge = cv2.morphologyEx((alpha[..., 0] > .5).astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)).astype(bool)
        blended[edge] = blended[edge] * .35 + np.array(glow, np.float32) * .65
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return (x0, y0, x1, y1)


def step_physics(obj, dt, width, height, gravity=1500.0, restitution=.45, damping=.25):
    """Screen-space ballistic motion for a thrown sprite: pixels and seconds."""
    if obj.mode != "flying":
        return
    remaining = min(max(dt, 0.0), .1)      # a long stall must not teleport the sprite
    while remaining > 1e-9:
        step = min(remaining, 1 / 240)
        remaining -= step
        obj.velocity[1] += gravity * step
        obj.velocity *= math.exp(-damping * step)
        obj.position += obj.velocity * step
        obj.angle += obj.spin * step
        obj.spin *= math.exp(-1.2 * step)
        half = obj.half_extent()
        for axis, extent in ((0, width), (1, height)):
            lo, hi = half[axis], extent - half[axis]
            if lo > hi:                     # bigger than the screen: keep it centred
                obj.position[axis] = extent / 2
                obj.velocity[axis] = 0
            elif obj.position[axis] < lo:
                obj.position[axis] = lo
                obj.velocity[axis] = abs(obj.velocity[axis]) * restitution
            elif obj.position[axis] > hi:
                obj.position[axis] = hi
                obj.velocity[axis] = -abs(obj.velocity[axis]) * restitution
                if axis == 1:
                    obj.velocity[0] *= math.exp(-6 * step)   # floor friction
                    obj.spin *= .9
        on_floor = obj.position[1] >= height - half[1] - .5
        if on_floor and abs(obj.velocity[1]) < 60 and np.linalg.norm(obj.velocity) < 25:
            obj.velocity[:] = 0
            obj.spin = 0
            obj.mode = "floating"          # came to rest on the bottom edge


# ------------------------------------------------------------------------------ hands
def finger_pose(points):
    """Count extended / curled fingers (index..pinky) from joint angles and wrist distance."""
    extended = curled = 0
    for base, joint, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b = points[base] - points[joint], points[tip] - points[joint]
        cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
        tip_d = np.linalg.norm(points[tip] - points[0])
        joint_d = np.linalg.norm(points[joint] - points[0])
        extended += cosine < -.80 and tip_d > joint_d * 1.10
        curled += cosine > -.25 and tip_d < joint_d * 1.05
    palm = (np.linalg.norm(points[0] - points[9]) + np.linalg.norm(points[5] - points[17])) / 2
    spread = float(np.linalg.norm(points[8] - points[20]) / max(palm, 1e-6))
    return extended, curled, spread, float(palm)


@dataclass
class Hand:
    points: np.ndarray = None
    label: str = ""
    pinch_point: np.ndarray = None   # thumb/index midpoint: WHERE the pinch happens
    grip: np.ndarray = None          # palm centre: how the hand MOVES (unaffected by
                                     # the fingers opening, so releasing adds no motion)
    aim: np.ndarray = None
    palm_center: np.ndarray = None
    palm_size: float = 0.0
    ratio: float = None
    roll: float = 0.0
    pinch: PinchGesture = None
    fist: HoldGesture = None
    palm: HoldGesture = None
    fist_pose: bool = False
    palm_pose: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    palm_history: deque = field(default_factory=lambda: deque(maxlen=40))
    held: int = None
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    roll_base: float = 0.0
    angle_base: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    virtual: bool = False       # the mouse fallback
    missing: bool = False       # briefly unseen while holding (grace period)
    last_seen: float = 0.0

    @property
    def present(self):
        return self.pinch_point is not None


def release_velocity(history, cutoff, horizon=.10, lookback=.25):
    """Hand velocity for a throw, from samples strictly BEFORE the release began.

    People often decelerate a moment before their fingers open, so the instantaneous
    estimate can miss a real flick. Take the fastest short-window least-squares
    velocity in the last `lookback` seconds (each window needs >= 3 samples).
    """
    samples = [s for s in history if s[0] < cutoff - 1e-9]
    best = np.zeros(2)
    if not samples:
        return best
    ends = [s[0] for s in samples if s[0] >= samples[-1][0] - lookback]
    for end in ends:
        window = [s for s in samples if s[0] <= end + 1e-9]
        v = estimate_velocity(window, end, horizon)
        if np.linalg.norm(v) > np.linalg.norm(best):
            best = v
    return best


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


class Manipulator:
    """Hand slots -> gesture events, and held-object transforms (1 or 2 hands)."""

    max_objects = 6

    def __init__(self, width, height, pinch=.30, release=.45, min_scale=.25, max_scale=4.0,
                 throw_speed=380.0, roll_rotation=True, roll_deadzone=20.0, grace=.25):
        self.width, self.height = width, height
        self.pinch_thresholds = (pinch, release)
        self.min_scale, self.max_scale = min_scale, max_scale
        self.throw_speed = throw_speed
        self.grace = grace
        self.roll_rotation, self.roll_deadzone = roll_rotation, roll_deadzone
        self.hands = [self._new_hand() for _ in range(3)]   # slot 2 = mouse
        self.hands[2].virtual = True
        self.objects = []
        self.active = None
        self.two = {}
        self.next_id = 1
        self.stats = dict(grabs=0, throws=0, drops=0, tracking_releases=0)

    def _new_hand(self, virtual=False):
        hand = Hand(pinch=PinchGesture(*self.pinch_thresholds), fist=HoldGesture(.6, 1.0),
                    palm=HoldGesture(1.5, 2.0))
        hand.virtual = virtual
        return hand

    def object(self, oid):
        return next((o for o in self.objects if o.id == oid), None)

    # ---------------------------------------------------------------- hand updates
    def match(self, observations):
        """MediaPipe's result order can swap between frames: assign detections to slots
        by minimum wrist travel (plus a soft handedness penalty), not by list index."""
        observations = observations[:2]
        if not observations:
            return {}

        def cost(assignment):
            total = 0.0
            for detection, slot in zip(observations, assignment):
                hand = self.hands[slot]
                total += (160 if hand.points is None else
                          float(np.linalg.norm(detection["points"][0] - hand.points[0])))
                if hand.points is not None and hand.label != detection["label"]:
                    total += 50
            return total

        slots = min(permutations(range(2), len(observations)), key=cost)
        return dict(zip(slots, observations))

    def update_hands(self, observations, now, dt, mouse=None):
        """Return a list of (slot, event) with event in press/release/lost/fist/palm."""
        events = []
        observations = [o for o in observations
                        if o["points"].shape == (21, 2) and np.isfinite(o["points"]).all()]
        matches = self.match(observations)
        for slot in (0, 1):
            hand = self.hands[slot]
            observed = matches.get(slot)
            points = observed["points"] if observed else None
            limit = self.width * (.25 if hand.missing else .45)
            jump = (points is not None and hand.points is not None and
                    np.linalg.norm(points[0] - hand.points[0]) > limit)
            if (points is None and hand.held is not None and dt <= .25
                    and now - hand.last_seen <= self.grace):
                # A holding hand vanished for a moment (landmark dropout, motion blur):
                # keep the object where it is and wait briefly instead of dropping it.
                hand.missing = True
                continue
            if points is None or dt > .25 or jump:
                # Hand lost (or identity uncertain): release safely, never a throw.
                if hand.present:
                    if hand.held is not None:
                        events.append((slot, "lost"))
                    self.release(slot, now, lost=True)
                self.hands[slot] = hand = self._new_hand()
                if points is None:
                    continue
            hand.missing, hand.last_seen = False, now
            hand.points, hand.label = points, observed["label"]
            tip_mid = (points[4] + points[8]) / 2
            hand.pinch_point = smooth_pointer(hand.pinch_point, tip_mid, dt, .03)
            hand.grip = smooth_pointer(hand.grip, points[[0, 5, 9, 13, 17]].mean(axis=0), dt, .03)
            hand.aim = smooth_pointer(hand.aim, aim_point(points, (self.height, self.width)), dt, .05)
            extended, curled, spread, palm = finger_pose(points)
            hand.palm_size = palm
            hand.palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
            hand.roll = math.degrees(math.atan2(*(points[9] - points[0])[::-1]))
            _, hand.ratio = update_pinch(points, False, *self.pinch_thresholds)
            event = hand.pinch.update(hand.ratio, now)
            hand.history.append((now, *hand.grip))
            hand.palm_history.append((now, *hand.palm_center))
            hand.velocity = estimate_velocity(hand.history, now)
            palm_speed = float(np.linalg.norm(estimate_velocity(hand.palm_history, now, .25)))
            # Poses: a fist needs ALL four fingers curled and no pinch in progress;
            # the reset palm needs four extended, spread fingers held nearly still.
            hand.fist_pose = curled == 4 and not hand.pinch.engaged and hand.held is None
            hand.palm_pose = (extended == 4 and spread > 1.0 and palm_speed < 220
                              and not hand.pinch.engaged and hand.held is None)
            if event:
                events.append((slot, event))
            if hand.fist.update(hand.fist_pose, now):
                events.append((slot, "fist"))
            if hand.palm.update(hand.palm_pose, now):
                events.append((slot, "palm"))
        # Mouse: a virtual single hand whose "pinch" is the left button.
        hand = self.hands[2]
        if mouse is None or mouse.get("point") is None:
            if hand.present and hand.pinch.pinched:
                events.append((2, "release"))
                self.release(2, now)
            self.hands[2] = self._new_hand(True)
        else:
            point = np.asarray(mouse["point"], float)
            hand.pinch_point = hand.aim = hand.grip = point
            hand.history.append((now, *point))
            hand.velocity = estimate_velocity(hand.history, now)
            was = hand.pinch.pinched
            if mouse.get("down") and not was:
                hand.pinch.state = "PINCHED"
                events.append((2, "press"))
            elif not mouse.get("down") and was:
                hand.pinch.state = "OPEN"
                hand.pinch.release_started = now
                events.append((2, "release"))
        return events

    # ---------------------------------------------------------------- grab / release
    def grab(self, slot, obj, now):
        hand = self.hands[slot]
        if hand.held is not None:
            return
        if obj.mode == "home":
            obj.position = obj.home_position()
        # Keep the grab offset: the object must not snap its centre to the fingers.
        hand.held, hand.offset = obj.id, obj.position - hand.grip
        hand.roll_base, hand.angle_base = hand.roll, obj.angle
        obj.holders.add(slot)
        obj.mode, obj.velocity[:], obj.spin = "held", 0, 0
        self.active = obj.id
        self.stats["grabs"] += 1
        self._rebase(obj)

    def release(self, slot, now, lost=False):
        hand = self.hands[slot]
        obj = self.object(hand.held) if hand.held is not None else None
        hand.held = None
        if obj is None:
            return None
        obj.holders.discard(slot)
        self.two.pop(obj.id, None)
        if obj.holders:
            self._rebase(obj)          # 2 hands -> 1 hand: re-derive offsets, no jump
            return "handover"
        if lost:
            obj.mode, obj.velocity[:] = "floating", 0
            self.stats["tracking_releases"] += 1
            return "lost"
        velocity = release_velocity(hand.history, hand.pinch.release_started or now)
        speed = float(np.linalg.norm(velocity))
        if speed >= self.throw_speed:
            obj.mode = "flying"
            obj.velocity = velocity * min(1.0, 2600 / speed)
            obj.spin = float(np.clip(velocity[0] / 6, -540, 540))
            self.stats["throws"] += 1
            return "throw"
        obj.mode, obj.velocity[:] = "floating", 0    # frozen in space where released
        self.stats["drops"] += 1
        return "drop"

    def _rebase(self, obj):
        for slot in obj.holders:
            hand = self.hands[slot]
            hand.offset = obj.position - hand.grip
            hand.roll_base, hand.angle_base = hand.roll, obj.angle

    # ---------------------------------------------------------------- per-frame motion
    def update_objects(self, now, dt):
        for obj in self.objects:
            target = 1.0 if obj.visible else 0.0
            obj.fade += np.clip(target - obj.fade, -dt / .25, dt / .25)   # 0.25 s fades
            if obj.mode == "home":
                obj.position = obj.home_position()
            holders = sorted(obj.holders)
            if len(holders) >= 2:
                self._two_hand(obj, holders[:2], dt)
            elif len(holders) == 1:
                self.two.pop(obj.id, None)
                hand = self.hands[holders[0]]
                old = obj.position.copy()
                obj.position = hand.grip + hand.offset
                if self.roll_rotation and not hand.virtual:
                    # One-hand twist: only rotation beyond a dead zone counts, so the
                    # natural wobble of a moving hand does not spin the object.
                    delta = wrap_degrees(hand.roll - hand.roll_base)
                    excess = math.copysign(max(0.0, abs(delta) - self.roll_deadzone), delta)
                    goal = hand.angle_base + excess
                    obj.angle += wrap_degrees(goal - obj.angle) * (1 - math.exp(-dt / .08))
                obj.velocity = (obj.position - old) / max(dt, 1e-3)
            else:
                step_physics(obj, dt, self.width, self.height)

    def _two_hand(self, obj, slots, dt):
        a, b = (self.hands[s] for s in slots)
        line = b.grip - a.grip
        distance = float(np.linalg.norm(line))
        midpoint = (a.grip + b.grip) / 2
        angle = math.degrees(math.atan2(line[1], line[0]))
        if obj.id not in self.two:
            if distance < 20:
                return
            # Normalise by the separation when the second hand joined.
            self.two[obj.id] = dict(distance=distance, angle=angle, scale=obj.scale,
                                    rotation=obj.angle, offset=obj.position - midpoint)
        base = self.two[obj.id]
        goal_scale = float(np.clip(base["scale"] * distance / base["distance"],
                                   self.min_scale, self.max_scale))
        turn = wrap_degrees(angle - base["angle"])
        goal_angle = base["rotation"] + turn
        k = 1 - math.exp(-dt / .06)                      # ~60 ms smoothing
        obj.scale += (goal_scale - obj.scale) * k
        obj.angle += wrap_degrees(goal_angle - obj.angle) * k
        r = math.radians(turn)
        rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        obj.position = midpoint + rot @ base["offset"] * (obj.scale / base["scale"])

    # ---------------------------------------------------------------- actions
    def toggle_visibility(self, obj):
        obj.visible = not obj.visible
        return "hidden" if not obj.visible else "shown"

    def reset(self, obj):
        for slot in list(obj.holders):
            self.hands[slot].held = None
        obj.holders.clear()
        self.two.pop(obj.id, None)
        obj.scale, obj.angle, obj.spin = 1.0, 0.0, 0.0
        obj.velocity[:] = 0
        obj.visible = True
        obj.mode = "home"
        obj.position = obj.home_position()

    def duplicate(self, obj):
        if len(self.objects) >= self.max_objects:
            return None
        copy = ManipulatedObject(self.next_id, obj.group, obj.sprite, duplicate=True,
                                 position=obj.position + (28, 22), scale=obj.scale,
                                 angle=obj.angle, mode="floating", visible=True, fade=0.0)
        self.next_id += 1
        self.objects.append(copy)
        self.active = copy.id
        return copy

    def toggle_freeze(self, obj):
        if obj.mode == "flying":
            obj.mode, obj.velocity[:] = "floating", 0
            obj.spin = 0
            return "frozen"
        if obj.mode in ("floating", "home"):
            obj.mode = "flying"                           # gravity takes over
            return "unfrozen (falling)"
        return None

    def remove(self, obj):
        for slot in list(obj.holders):
            self.hands[slot].held = None
        self.two.pop(obj.id, None)
        self.objects.remove(obj)
        if self.active == obj.id:
            self.active = self.objects[-1].id if self.objects else None

    def add(self, group, sprite):
        obj = ManipulatedObject(self.next_id, group, sprite, position=group.origin.copy())
        self.next_id += 1
        self.objects.append(obj)
        self.active = obj.id
        return obj

    def hit(self, point):
        """Topmost visible sprite under the point (held/newest drawn last = on top)."""
        for obj in reversed(self.draw_order()):
            if obj.fade > .5 and obj.contains(point):
                return obj
        return None

    def draw_order(self):
        return sorted(self.objects, key=lambda o: (bool(o.holders), o.id))
