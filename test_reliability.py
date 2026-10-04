"""Failure paths and control regressions, offline and without a webcam."""
from concurrent.futures import Future, CancelledError
import hashlib
import io
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from hud import draw_hud, wrap_text, FONT
from main import App, parse_args, shifted
from manipulation import Manipulator
from model_assets import ensure_asset
from segmentation import EdgeSAM, MaskResult, SegmentationWorker, rank_candidates, clean_mask, stability_score
from scene import feathered_alpha
from lighting import enhance_low_light
from selection import Selector, hand_zone
from test_core import make_object, make_hand
from test_selection import candidate, fake_embedding, rect_mask, textured


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.bin"
        self.payload = b"verified fixture model"
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def fetch(self, data=None, **kwargs):
        with patch("model_assets.urlopen", return_value=io.BytesIO(self.payload if data is None else data)):
            return ensure_asset(self.path, "https://example.invalid/model", self.digest, **kwargs)

    def test_verified_cache_works_offline(self):
        self.path.write_bytes(self.payload)
        with patch("model_assets.urlopen", side_effect=AssertionError("must not use network")):
            ensure_asset(self.path, "unused", self.digest)

    def test_corrupt_cache_is_repaired_atomically(self):
        self.path.write_bytes(b"corrupt")
        self.fetch()
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertEqual(list(self.path.parent.glob("*.download")), [])

    def test_bad_response_preserves_cache_and_removes_temporary(self):
        self.path.write_bytes(b"previous")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.fetch(b"bad data")
        self.assertEqual(self.path.read_bytes(), b"previous")
        self.assertEqual(list(self.path.parent.glob("*.download")), [])

    def test_size_limit_and_cancellation_never_install_file(self):
        with self.assertRaisesRegex(RuntimeError, "size limit"):
            self.fetch(max_bytes=3)
        cancel = Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            self.fetch(cancel=cancel)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.glob("*.download")), [])

    def test_total_timeout_cleans_up(self):
        with patch("model_assets.time.monotonic", side_effect=[0, 181]):
            with self.assertRaisesRegex(RuntimeError, "180 seconds"):
                self.fetch()
        self.assertFalse(self.path.exists())


class RankingRecoveryTests(unittest.TestCase):
    def test_focused_pass_cannot_switch_to_another_object(self):
        model = object.__new__(EdgeSAM)
        coarse = candidate(rect_mask(20, 20, 70, 80), stability=.90)
        unrelated = candidate(rect_mask(100, 100, 180, 190), stability=.99)
        model.decode = Mock(return_value=[unrelated])
        self.assertIs(model.refine_candidate(None, coarse, [(40, 40)]), coarse)

    def test_focused_pass_accepts_agreeing_more_stable_boundary(self):
        model = object.__new__(EdgeSAM)
        coarse = candidate(rect_mask(20, 20, 70, 80), stability=.90)
        focused = candidate(rect_mask(21, 21, 70, 80), stability=.98)
        model.decode = Mock(return_value=[focused])
        self.assertIs(model.refine_candidate(None, coarse, [(40, 40)]), focused)

    def test_focus_rejects_weaker_score_and_more_hand_pixels(self):
        model = object.__new__(EdgeSAM)
        coarse = candidate(rect_mask(20, 20, 70, 80), score=.95, stability=.90)
        weak = candidate(coarse.mask.copy(), score=.55, stability=.99)
        hand = candidate(coarse.mask.copy(), score=.95, stability=.99, overlap=.15)
        model.decode = Mock(return_value=[weak, hand])
        self.assertIs(model.refine_candidate(None, coarse, [(40, 40)]), coarse)

    def test_nearby_equal_sized_neighbour_is_not_added(self):
        mask = rect_mask(20, 20, 60, 70, (100, 120))
        mask[20:70, 62:102] = True
        cleaned = clean_mask(mask, (40, 40))
        self.assertTrue(cleaned[40, 40])
        self.assertFalse(cleaned[40, 80])

    def test_prompted_component_does_not_import_distant_neighbour(self):
        mask = rect_mask(20, 20, 50, 50, (160, 160))
        mask[80:145, 80:145] = True
        cleaned = clean_mask(mask, (30, 30))
        self.assertTrue(cleaned[30, 30])
        self.assertFalse(cleaned[100, 100])

    def test_exact_prompt_wins_over_larger_nearby_component(self):
        mask = rect_mask(20, 20, 25, 25, (80, 80))
        mask[20:45, 27:50] = True
        cleaned = clean_mask(mask, (23, 23))
        self.assertTrue(cleaned[23, 23])
        self.assertFalse(cleaned[30, 30])

    def test_split_handle_retained_but_distant_fragment_removed(self):
        mask = rect_mask(20, 20, 70, 80, (160, 160))
        mask[30:65, 72:82] = True
        mask[30:65, 120:130] = True
        cleaned = clean_mask(mask, (40, 40))
        self.assertTrue(cleaned[40, 76])
        self.assertFalse(cleaned[40, 125])

    def test_handle_hole_and_thin_details_are_preserved(self):
        mask = rect_mask(20, 20, 80, 80, (100, 100))
        mask[35:45, 35:45] = False
        cleaned = clean_mask(mask, (25, 25))
        self.assertFalse(cleaned[40, 40])
        cord = rect_mask(20, 20, 22, 80, (100, 100))
        alpha = feathered_alpha(cord)
        self.assertGreater(float(alpha[45, 20]), .5)
        self.assertEqual(float(alpha[45, 19]), 0)

    def test_unrelated_logits_do_not_change_target_stability(self):
        logits = np.full((100, 100), -4, np.float32)
        logits[10:30, 10:30] = 4
        support = rect_mask(8, 8, 32, 32, (100, 100))
        before = stability_score(logits, support=support)
        logits[50:90, 50:90] = .5
        self.assertEqual(stability_score(logits, support=support), before)

    def test_low_confidence_background_is_not_selectable(self):
        weak = candidate(rect_mask(20, 20, 140, 150), score=.50, stability=.68)
        self.assertIsNone(rank_candidates([weak]))
        self.assertFalse(weak.selectable)

    def test_lighting_preparation_keeps_original_pixels_untouched(self):
        frame = np.full((60, 80, 3), 20, np.uint8)
        original = frame.copy()
        enhanced = enhance_low_light(frame)
        np.testing.assert_array_equal(frame, original)
        self.assertGreater(float(enhanced.mean()), float(frame.mean()))
        bright = np.full((60, 80, 3), 140, np.uint8)
        self.assertIs(enhance_low_light(bright), bright)

    def test_two_hand_zone_does_not_expand_first_hand_again(self):
        a, b = make_hand((150, 200)), make_hand((450, 200), palm=100)
        expected = hand_zone((480, 640), [a]) | hand_zone((480, 640), [b])
        np.testing.assert_array_equal(hand_zone((480, 640), [a, b]), expected)
        np.testing.assert_array_equal(hand_zone((480, 640), [b, a]), expected)

    def test_conflicting_score_and_stability_never_crash(self):
        crisp_part = candidate(rect_mask(100, 100, 130, 130), score=.99, stability=.76)
        whole = candidate(rect_mask(80, 80, 180, 200), score=.60, stability=.99)
        self.assertEqual(rank_candidates([crisp_part, whole]), 0)

    def test_cycle_never_selects_body_surface_or_hand(self):
        good = candidate(rect_mask(80, 80, 180, 200))
        hand = candidate(rect_mask(240, 240, 300, 340), overlap=.7)
        body = candidate(rect_mask(300, 180, 400, 400))
        body.person_overlap = .8
        surface = candidate(rect_mask(0, 380, 640, 480))
        masks = [hand, body, good, surface]
        choice = rank_candidates(masks)
        selector = Selector()
        selector.result = MaskResult(1, 1, (100, 100), masks, choice, 0, 0, 0)
        selector.choice = choice
        for t in range(8):
            selector.cycle(t)
            self.assertIs(selector.candidate, good)

    def test_all_rejected_cannot_be_revived_by_cycle(self):
        masks = [candidate(rect_mask(0, 0, 640, 480))]
        choice = rank_candidates(masks)
        selector = Selector()
        selector.result = MaskResult(1, 1, None, masks, choice, 0, 0, 0)
        selector.choice = choice
        selector.cycle(1)
        self.assertIsNone(selector.candidate)

    def test_wrong_snapshot_is_rejected_even_with_matching_prompt(self):
        selector = Selector(dwell=.1)
        frame = textured()
        zone = np.zeros(frame.shape[:2], bool)
        selector.update(0, (100, 100), frame, [], zone, None, False)
        job = selector.update(.2, (100, 100), frame, [], zone, None, False)
        result = MaskResult(job.snapshot_id + 1, job.prompt_id, (100, 100), [], None, 0, 0, 0)
        self.assertFalse(selector.accept(result, .3))

    def test_failed_startup_retry_reloads_without_user_aim(self):
        worker = SegmentationWorker()
        self.addCleanup(worker.close)
        worker.error = "offline"
        with patch.object(worker, "_load", return_value=None) as load:
            worker.retry()
            worker.future.result(timeout=2)
            worker.poll()
            load.assert_called_once()
        self.assertIsNone(worker.error)

    def test_inference_failure_clears_pending_and_marks_not_ready(self):
        worker = SegmentationWorker()
        self.addCleanup(worker.close)
        worker.ready = True
        worker.pending = Mock()
        worker.future = Future()
        worker.future.set_exception(RuntimeError("bad inference"))
        worker.poll()
        self.assertFalse(worker.ready)
        self.assertIsNone(worker.pending)
        self.assertIn("bad inference", worker.error)


class AppControlTests(unittest.TestCase):
    def setUp(self):
        self.worker = Mock(ready=True, error=None, model=object())
        self.app = App(640, 480, self.worker, parse_args([]))
        self.addCleanup(self.app.close)
        self.obj = make_object(self.app.manip)

    def test_wheel_scale_rotate_and_zero_delta(self):
        for delta, multiplier in ((120, 1.1), (-120, 1 / 1.1)):
            flags = (delta & 0xFFFF) << 16
            before = self.obj.scale
            self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, flags)
            self.assertAlmostEqual(self.obj.scale, before * multiplier)
            before = self.obj.angle
            self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, flags | cv2.EVENT_FLAG_CTRLKEY)
            self.assertEqual(self.obj.angle, before + (10 if delta > 0 else -10))
        before = (self.obj.scale, self.obj.angle)
        self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, 0)
        self.assertEqual((self.obj.scale, self.obj.angle), before)

    def test_cancel_clears_early_pinch_and_pending_work(self):
        self.app.pending_lock = (2, 5)
        self.app.selector.state = "ANALYZING"
        self.app.key(27, 1)
        self.assertIsNone(self.app.pending_lock)
        self.assertEqual(self.app.selector.state, "IDLE")
        self.worker.discard_pending.assert_called_once()
        self.assertEqual(self.app.manip.objects, [])

    def test_wheel_rotation_persists_while_holding(self):
        hand = self.app.manip.hands[0]
        hand.grip = hand.pinch_point = np.array([200., 200.])
        self.app.manip.grab(0, self.obj, 1)
        self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, (120 << 16) | cv2.EVENT_FLAG_CTRLKEY)
        self.app.manip.update_objects(1.1, .1)
        self.assertEqual(self.obj.angle, 10)

    def test_wheel_scale_persists_while_two_hands_hold(self):
        m = self.app.manip
        for slot, x in ((0, 150), (1, 250)):
            m.hands[slot].grip = m.hands[slot].pinch_point = np.array([float(x), 200.])
            m.grab(slot, self.obj, 1)
        m.update_objects(1.1, .1)
        self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, 120 << 16)
        m.update_objects(1.2, .1)
        self.assertAlmostEqual(self.obj.scale, 1.1)

    def test_refinement_rejects_body_even_if_overlap_is_high(self):
        app = self.app
        mask = rect_mask(100, 100, 140, 140)
        group, sprite = app.extract(textured(), mask, 1, 1)
        app.manip.objects.clear()
        app.manip.add(group, sprite)
        app.refining = (1, 1000001, 1, np.zeros(2))
        rejected = candidate(mask)
        rejected.person_overlap = .9
        rank_candidates([rejected])
        result = MaskResult(1000001, 1000001, (120, 120), [rejected], None, 0, 0, 0,
                            fake_embedding(textured(), 1000001, 1))
        app.finish_refine(result, 2)
        self.assertEqual(app.counters["refine_rejected"], 1)
        self.assertIs(group.mask, mask)

    def test_retry_clears_stale_selection(self):
        self.worker.error = "offline"
        self.app.selector.state = "ANALYZING"
        self.app.pending_lock = (2, 5)
        self.app.key(ord("e"), 1)
        self.assertEqual(self.app.selector.state, "IDLE")
        self.assertIsNone(self.app.pending_lock)
        self.worker.retry.assert_called_once()

    def test_hand_skeleton_is_visible_without_debug(self):
        self.app.manip.objects.clear()
        self.worker.poll.return_value = None
        self.assertFalse(self.app.debug)
        self.assertFalse(self.app.show_keys)
        with patch("main.draw_hand_skeleton") as draw:
            self.app.step(textured(), 1, .04, [{"points": make_hand(), "label": "Right"}])
            self.assertEqual(draw.call_count, 1)
            self.app.step(textured(), 1.1, .04, [])
            self.assertEqual(draw.call_count, 1)  # no invented skeleton after tracking loss

    def test_refinement_preserves_current_camera_translation_and_sprite_transform(self):
        app, obj = self.app, self.obj
        mask = rect_mask(100, 100, 140, 140)
        old_group, old_sprite = app.extract(textured(), mask, 1, 1)
        app.manip.objects.clear()
        obj = app.manip.add(old_group, old_sprite)
        obj.mode, obj.scale, obj.angle = "floating", 1.7, 32
        obj.position = np.array([380., 260.])
        duplicate = app.manip.duplicate(obj)
        duplicate.angle = -20
        duplicate_start = duplicate.position.copy()
        duplicate_linear = duplicate.matrix()[:, :2]
        old_group.tracker.offset = np.array([9., 6.])
        submit_offset = np.array([4., 3.])
        new_mask = rect_mask(104, 103, 146, 145)
        app.refining = (1, 1000001, 1, submit_offset)
        embedding = fake_embedding(textured(), 1000001, 1)
        result = MaskResult(1000001, 1000001, (120, 120), [candidate(new_mask)], 0, 0, 0, 0, embedding)
        old_origin = old_group.origin.copy()
        old_linear = obj.matrix()[:, :2]
        old_position = obj.position.copy()
        app.finish_refine(result, 2)
        np.testing.assert_allclose(old_group.tracker.offset, [5, 3])
        delta = old_group.origin - (old_origin + submit_offset)
        np.testing.assert_allclose(obj.position, old_position + old_linear @ delta)
        np.testing.assert_allclose(duplicate.position, duplicate_start + duplicate_linear @ delta)
        self.assertIs(duplicate.sprite, obj.sprite)
        self.assertTrue(old_group.refined)

    def test_translation_beyond_frame_returns_empty_mask(self):
        mask = rect_mask(10, 10, 20, 20, (40, 40))
        for offset in ((45, 3), (-45, 3), (4, 50), (4, -50)):
            self.assertFalse(shifted(mask, offset).any())


class HudTests(unittest.TestCase):
    def test_narrow_wrap_always_terminates(self):
        self.assertEqual(wrap_text("test", 0), [])
        self.assertLessEqual(len(wrap_text("test", 1)), 3)

    def test_long_feedback_fits_small_and_standard_camera_widths(self):
        status = "Move it - release to place - flick + release to throw - 2nd hand pinch: scale"
        for width, height in ((360, 240), (640, 480), (1280, 720)):
            for line in wrap_text(status, width - 24, .45, 2):
                self.assertLessEqual(cv2.getTextSize(line, FONT, .45, 1)[0][0], width - 24)
            image = np.zeros((height, width, 3), np.uint8)
            bottom = draw_hud(image, status, "6 objects / active 6 / 2 hidden", status, True, True)
            self.assertLess(bottom, height / 2)
            self.assertTrue(image.any())


if __name__ == "__main__":
    unittest.main()
