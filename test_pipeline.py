"""End-to-end tests with the REAL EdgeSAM model on reproducible synthetic scenes.

A textured desk with three distinct objects (an irregular mug with a handle, a phone,
a bottle) and a drawn skin-coloured hand whose synthetic landmarks drive the app. The
hand is really present in the frames the model segments. Skipped if the model is absent.
This proves the pipeline mechanics; it does not prove webcam behaviour with real hands.
"""
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from main import App, parse_args
from segmentation import MODEL_DIR, MODEL_FILES, EdgeSAM, iou, rank_candidates
from test_core import make_hand

HAVE_MODEL = all((MODEL_DIR / name).is_file() for name in MODEL_FILES)
SKIN = (120, 150, 205)


def desk_scene():
    rng = np.random.default_rng(7)
    base = rng.integers(95, 150, (60, 80, 3), np.uint8)
    frame = cv2.resize(base, (640, 480), interpolation=cv2.INTER_CUBIC)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame[..., 0] = (frame[..., 0] * .6).astype(np.uint8)          # warm wood-ish tone
    truth = {}
    mug = np.zeros((480, 640), np.uint8)
    cv2.ellipse(mug, (170, 250), (45, 55), 0, 0, 360, 1, -1)
    cv2.ellipse(mug, (222, 250), (22, 28), 0, 0, 360, 1, 12)      # handle (a ring)
    frame[mug > 0] = (40, 50, 190)
    cv2.ellipse(frame, (170, 205), (38, 10), 0, 0, 360, (60, 70, 210), -1)
    truth["mug"] = mug > 0
    phone = np.zeros((480, 640), np.uint8)
    box = cv2.boxPoints(((360, 270), (80, 150), 15)).astype(np.int32)
    cv2.fillConvexPoly(phone, box, 1)
    frame[phone > 0] = (60, 30, 20)
    inner = cv2.boxPoints(((360, 270), (64, 128), 15)).astype(np.int32)
    cv2.fillConvexPoly(frame, inner, (150, 120, 70))
    truth["phone"] = phone > 0
    bottle = np.zeros((480, 640), np.uint8)
    cv2.rectangle(bottle, (505, 180), (555, 330), 1, -1)
    cv2.rectangle(bottle, (520, 140), (540, 185), 1, -1)
    frame[bottle > 0] = (60, 160, 50)
    cv2.rectangle(frame, (505, 230), (555, 270), (230, 230, 230), -1)   # label
    truth["bottle"] = bottle > 0
    return frame, truth


def draw_hand(frame, points):
    """Paint a skin-coloured hand over the scene; return its pixel mask."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    p = np.rint(points).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(p[[0, 1, 5, 9, 13, 17]]), 1)
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        for a, b in zip(chain, chain[1:]):
            cv2.line(mask, tuple(p[a]), tuple(p[b]), 1, 16)
    wrist, direction = points[0], points[0] - points[9]
    cv2.line(mask, tuple(p[0]), tuple(np.rint(wrist + direction * 3).astype(int)), 1, 40)
    frame[mask > 0] = SKIN
    return mask > 0


def hand_obs(points, label="Right"):
    return {"points": points, "z": np.zeros(21), "label": label}


@unittest.skipUnless(HAVE_MODEL, "EdgeSAM model not downloaded; run the app once")
class SegmentationQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EdgeSAM(threads=4)
        cls.scene, cls.truth = desk_scene()

    def select(self, frame, point, negatives=(), zone=None):
        embedding = self.model.encode(frame, hand_zone=zone)
        candidates = self.model.decode(embedding, [point], list(negatives))
        choice = rank_candidates(candidates)
        return None if choice is None else candidates[choice].mask

    def test_mouse_point_selects_each_whole_object(self):
        for name, point in (("mug", (160, 260)), ("phone", (360, 300)), ("bottle", (530, 300))):
            mask = self.select(self.scene, point)
            self.assertGreater(iou(mask, self.truth[name]), .85, name)

    def test_touch_with_hand_in_frame_excludes_hand(self):
        from selection import aim_point, hand_zone, negative_prompts
        for name, tip in (("mug", (165, 320)), ("phone", (365, 355)), ("bottle", (530, 345))):
            frame = self.scene.copy()
            points = make_hand(tip, palm=70)
            hand = draw_hand(frame, points)
            aim = aim_point(points, frame.shape)
            self.assertTrue(self.truth[name][int(aim[1]), int(aim[0])], f"aim misses {name}")
            mask = self.select(frame, aim, negative_prompts(points), hand_zone(frame.shape, [points]))
            visible_truth = self.truth[name] & ~hand
            self.assertGreater(iou(mask, visible_truth), .8, name)
            self.assertLess((mask & hand).sum() / mask.sum(), .05, name)

    def test_mirrored_frame_gives_mirrored_mask(self):
        mask = self.select(self.scene, (360, 300))
        mirrored = self.select(cv2.flip(self.scene, 1), (639 - 360, 300))
        self.assertGreater(iou(cv2.flip(mask.astype(np.uint8), 1).astype(bool), mirrored), .95)

    def test_dark_noisy_selection_improves_without_changing_the_sprite_pixels(self):
        rng = np.random.default_rng(42)
        rng.normal(0, 8, self.scene.shape)  # same controlled fixture as the precision probe
        dim = np.clip(self.scene.astype(float) * .18 + rng.normal(0, 8, self.scene.shape), 0, 255).astype(np.uint8)
        with patch("segmentation.enhance_low_light", side_effect=lambda frame: frame):
            baseline = self.select(dim, (530, 300))
        improved = self.select(dim, (530, 300))
        before, after = iou(baseline, self.truth["bottle"]), iou(improved, self.truth["bottle"])
        self.assertGreater(after, .93)
        self.assertGreater(after - before, .03)
        embedding = self.model.encode(dim)
        np.testing.assert_array_equal(embedding.frame, dim)

    def test_shadow_crossing_object_keeps_the_right_silhouette(self):
        y, x = np.indices(self.scene.shape[:2])
        gain = np.where((x > 180) & (y > 220), .45, 1.)
        shadow = (self.scene.astype(float) * gain[..., None]).astype(np.uint8)
        for name, point in (("mug", (160, 260)), ("phone", (360, 300)), ("bottle", (530, 300))):
            mask = self.select(shadow, point)
            self.assertGreater(iou(mask, self.truth[name]), .90, name)

    def test_focused_boundary_pass_improves_shadowed_bottle(self):
        y, x = np.indices(self.scene.shape[:2])
        gain = np.where((x > 180) & (y > 220), .45, 1.)
        shadow = (self.scene.astype(float) * gain[..., None]).astype(np.uint8)
        embedding = self.model.encode(shadow)
        candidates = self.model.decode(embedding, [(530, 300)])
        coarse = candidates[rank_candidates(candidates)]
        focused = self.model.refine_candidate(embedding, coarse, [(530, 300)])
        self.assertGreater(iou(focused.mask, self.truth["bottle"]), .98)
        self.assertGreater(iou(focused.mask, self.truth["bottle"]) - iou(coarse.mask, self.truth["bottle"]), .02)


@unittest.skipUnless(HAVE_MODEL, "EdgeSAM model not downloaded; run the app once")
class AppFlowTests(unittest.TestCase):
    """Drive App.step() with drawn hands and synthetic landmarks, in real time."""

    def setUp(self):
        from segmentation import SegmentationWorker
        self.scene, self.truth = desk_scene()
        self.worker = SegmentationWorker(threads=4)
        self.addCleanup(self.worker.close)
        self.worker.warm_up()
        self.app = App(640, 480, self.worker, parse_args([]))
        self.addCleanup(self.app.close)
        self.t = 0.0
        self.out = None
        self.last = None
        self.assertTrue(self.wait(lambda: self.worker.ready, 60), "model startup timed out")

    def frame(self, hands=(), scene=None, dt=1 / 25, pace=False):
        """One app step. hands: list of (landmarks, label)."""
        started = time.perf_counter()
        self.t += dt
        frame = (self.scene if scene is None else scene).copy()
        person = np.zeros((480, 640), np.float32)
        for points, _ in hands:
            person[draw_hand(frame, points)] = 1.0     # stands in for a perfect person mask
        self.last = frame
        self.out = self.app.step(frame, self.t, dt, [hand_obs(p, l) for p, l in hands], person)
        time.sleep(max(0, dt - (time.perf_counter() - started)) if pace else .004)
        return self.out

    def wait(self, condition, seconds=15, hands=()):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.worker.error:
                self.fail(f"model failed: {self.worker.error}")
            if condition():
                return True
            if hasattr(self, "app"):
                # Match simulated time to wall time while waiting for real inference.
                # Fast synthetic time must not expire an otherwise valid model result.
                self.frame(hands, pace=True)
            else:
                time.sleep(.02)
        return condition()

    def hold(self, hands, n):
        for _ in range(n):
            self.frame(hands)

    def aim_and_preview(self, tip, label="Right"):
        hand = [(make_hand(tip, palm=70), label)]
        self.assertTrue(self.wait(lambda: self.app.selector.state == "PREVIEW", 20, hand),
                        f"no preview, state {self.app.selector.state}")
        return hand

    def pinch(self, tip, n=4, label="Right"):
        self.hold([(make_hand(tip, palm=70, pinch=True), label)], n)

    def red_fraction(self, image, mask):
        pixels = image[mask].astype(int)
        return float(((pixels[:, 2] > 150) & (pixels[:, 1] < 90)).mean())

    def test_full_manipulation_sequence(self):
        app, mug = self.app, self.truth["mug"]
        self.aim_and_preview((165, 320))
        self.assertGreater(iou(app.selector.mask, mug), .75)
        self.pinch((165, 320))
        self.assertEqual(len(app.manip.objects), 1)
        obj = app.manip.objects[0]
        self.assertEqual(obj.mode, "held")
        start_offset = obj.position - app.manip.hands[0].grip
        # Drag right by 250 px (the physical mug stays put in every raw frame).
        for i in range(1, 26):
            self.pinch((165 + 10 * i, 320), n=1)
        np.testing.assert_allclose(obj.position - app.manip.hands[0].grip, start_offset, atol=2)
        core = cv2.erode(mug.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        self.assertLess(self.red_fraction(self.out, core), .05, "original still visible")
        # Stationary release: stays where placed.
        self.hold([(make_hand((415, 320), palm=70, pinch=True), "Right")], 6)
        self.hold([(make_hand((415, 320), palm=70), "Right")], 4)
        self.assertEqual(obj.mode, "floating")
        # Refinement can change the centroid; compare the image transform relative to
        # the physical object rather than requiring its centroid to remain identical.
        placed = obj.position - obj.matrix()[:, :2] @ obj.home_position()
        self.hold([], 5)
        np.testing.assert_allclose(obj.position - obj.matrix()[:, :2] @ obj.home_position(), placed)
        # Hide -> region shows background; show; reset -> back home.
        self.assertGreater(self.red_fraction(self.out, self.sprite_mask(obj)), .5)
        app.key(ord("h"), self.t)
        self.hold([], 12)
        self.assertLess(self.red_fraction(self.out, self.sprite_mask(obj)), .05)
        app.key(ord("h"), self.t)
        self.hold([], 12)
        app.key(ord("r"), self.t)
        self.hold([], 3)
        np.testing.assert_allclose(obj.position, obj.group.origin + obj.group.tracker.offset, atol=1)
        self.assertGreater(self.red_fraction(self.out, core), .9)
        # Duplicate, then release both back to reality.
        app.key(ord("c"), self.t)
        self.hold([], 12)
        self.assertEqual(len(app.manip.objects), 2)
        app.key(27, self.t)
        app.key(27, self.t)
        self.hold([], 2)
        self.assertEqual(len(app.manip.objects), 0)
        np.testing.assert_array_equal(self.out[core], self.scene[core])
        # Repeat with another object, no object-specific code.
        self.aim_and_preview((365, 355))
        self.assertGreater(iou(app.selector.mask, self.truth["phone"]), .75)

    def sprite_mask(self, obj):
        canvas = np.zeros((480, 640, 3), np.uint8)
        from manipulation import render_sprite
        old = obj.fade
        obj.fade = 1.0
        render_sprite(canvas, obj)
        obj.fade = old
        white = np.zeros((480, 640, 3), np.uint8)
        return cv2.erode(((canvas != white).any(axis=2)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    def test_throw_lands_and_hand_loss_never_throws(self):
        app = self.app
        self.aim_and_preview((530, 345))
        self.pinch((530, 345))
        obj = app.manip.objects[0]
        # Tracking loss mid-swing (0.4 s, longer than the 0.25 s grace): no throw.
        for i in range(5):
            self.pinch((530 - 25 * i, 345), n=1)
        self.hold([], 10)
        self.assertEqual(obj.mode, "floating")
        self.assertEqual(app.manip.stats["throws"], 0)
        # Re-grab the sprite where it is and throw it left/up.
        tip = obj.position + (0, 40)
        self.hold([(make_hand(tip, palm=70), "Right")], 5)
        self.pinch(tip)
        self.assertEqual(obj.mode, "held")
        for i in range(8):
            self.pinch(tip + (-30 * i, -12 * i), n=1)
        end = tip + (-210, -84)
        self.hold([(make_hand(end, palm=70), "Right")], 3)
        self.assertEqual(obj.mode, "flying")
        self.assertLess(obj.velocity[0], -300)
        self.hold([], 120)
        self.assertEqual(obj.mode, "floating")
        self.assertGreater(obj.position[1], 480 - obj.half_extent()[1] - 2)   # resting on the floor

    def test_two_hand_scale_and_rotation(self):
        app = self.app
        self.aim_and_preview((365, 355))
        self.pinch((365, 355))
        obj = app.manip.objects[0]
        second = make_hand((470, 355), palm=70)
        self.hold([(make_hand((365, 355), palm=70, pinch=True), "Right"), (second, "Left")], 5)
        self.hold([(make_hand((365, 355), palm=70, pinch=True), "Right"),
                   (make_hand((470, 355), palm=70, pinch=True), "Left")], 5)
        self.assertEqual(len(obj.holders), 2)
        for i in range(1, 21):
            self.hold([(make_hand((365 - 3 * i, 355 + 2 * i), palm=70, pinch=True), "Right"),
                       (make_hand((470 + 3 * i, 355 - 2 * i), palm=70, pinch=True), "Left")], 1)
        self.hold([(make_hand((305, 395), palm=70, pinch=True), "Right"),
                   (make_hand((530, 315), palm=70, pinch=True), "Left")], 10)
        self.assertGreater(obj.scale, 1.8)
        self.assertLess(obj.angle, -10)               # hand line turned counter-clockwise

    def test_stale_selection_is_rejected_and_camera_shift_is_tracked(self):
        app = self.app
        # Settle on the mug, then move to the bottle before the model answers.
        for _ in range(8):
            self.frame([(make_hand((165, 320), palm=70), "Right")])
        self.assertEqual(app.selector.state, "ANALYZING")
        self.aim_and_preview((530, 345))
        self.assertGreater(iou(app.selector.mask, self.truth["bottle"]), .75)
        self.assertGreaterEqual(app.counters["rejected_stale"], 1)
        self.pinch((530, 345))
        obj = app.manip.objects[0]
        self.hold([(make_hand((530, 345), palm=70), "Right")], 3)
        self.assertTrue(self.wait(lambda: obj.group.refined, 15))   # hand-free refinement done
        home = obj.home_position().copy()   # where the physical bottle is seen
        # Camera nudged 6 px right, 4 px down: the tracker follows the physical bottle.
        shifted = np.roll(self.scene, (4, 6), axis=(0, 1))
        for _ in range(15):
            self.frame([], scene=shifted)
        np.testing.assert_allclose(obj.home_position(), home + (6, 4), atol=1.6)
        home = obj.home_position().copy()
        # A completely different scene: tracking is lost, never switched.
        rng = np.random.default_rng(99)
        other = cv2.resize(rng.integers(0, 255, (48, 64, 3), np.uint8), (640, 480))
        for _ in range(60):
            self.frame([], scene=other)
        self.assertTrue(obj.group.tracker.state.startswith("LOST"))
        np.testing.assert_allclose(obj.home_position(), home, atol=.5)


if __name__ == "__main__":
    unittest.main()
