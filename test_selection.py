"""Selection, coordinate transforms, ranking, tracking and reconstruction.
No model files or webcam needed (the real model is exercised in test_pipeline.py)."""
from concurrent.futures import CancelledError
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from scene import ObjectTracker, Reconstruction, SceneMemory, feathered_alpha, push_pull_fill
from segmentation import (Candidate, Embedding, Job, MaskResult, SegmentationWorker, check_cancelled,
                          clean_mask, preprocess, rank_candidates, restore_logits, to_model_coords,
                          touched_borders)
from selection import Selector, hand_zone
from test_core import make_hand


def candidate(mask, score=.9, stability=.95, overlap=0.0):
    return Candidate(mask, mask.astype(np.float32), score, stability, float(mask.mean()), overlap,
                     touched_borders(mask))


def rect_mask(x0, y0, x1, y1, shape=(480, 640)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def textured(shape=(480, 640), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, (shape[0] // 8, shape[1] // 8, 3), np.uint8)
    return cv2.resize(base, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


class TransformTests(unittest.TestCase):
    def test_normalization_and_padding_on_non_square_frames(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[:] = (0, 0, 255)                         # pure red in BGR
        tensor, (h, w) = preprocess(frame)
        self.assertEqual((h, w), (768, 1024))
        np.testing.assert_allclose(tensor[0, 0, 0, 0], (255 - 123.675) / 58.395, rtol=1e-5)  # R
        np.testing.assert_allclose(tensor[0, 2, 0, 0], (0 - 103.53) / 57.375, rtol=1e-5)     # B
        self.assertFalse(tensor[0, :, 768:, :].any())  # bottom padding is exactly zero

    def test_point_and_mask_inverse_transforms_round_trip(self):
        np.testing.assert_allclose(to_model_coords([(640, 480)], (768, 1024), (480, 640)), [[1024, 768]])
        # A 256x256 logit map describing a rectangle in PADDED model space must come back
        # at the same camera location (not stretched by the padding).
        low = np.full((256, 256), -10, np.float32)
        # camera rect x 160..480, y 120..360 -> model x 256..768, y 192..576 -> /4
        low[48:144, 64:192] = 10
        restored = restore_logits(low, (768, 1024), (480, 640)) > 0
        ys, xs = np.nonzero(restored)
        self.assertAlmostEqual(xs.min(), 160, delta=3)
        self.assertAlmostEqual(xs.max(), 479, delta=3)
        self.assertAlmostEqual(ys.min(), 120, delta=3)
        self.assertAlmostEqual(ys.max(), 359, delta=3)

    def test_mirroring_is_applied_before_every_consumer(self):
        # The app flips the frame once; points and masks then share mirrored pixels.
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[100:120, 50:70] = 255
        mirrored = cv2.flip(frame, 1)
        self.assertTrue(mirrored[110, 640 - 60].all())


class RankingTests(unittest.TestCase):
    def test_whole_object_beats_higher_scoring_part(self):
        part = candidate(rect_mask(100, 100, 130, 130), score=.97, stability=.97)
        whole = candidate(rect_mask(80, 80, 200, 220), score=.90, stability=.96)
        self.assertEqual(rank_candidates([part, whole]), 1)

    def test_merge_of_neighbours_with_unstable_seam_is_not_chosen(self):
        single = candidate(rect_mask(100, 100, 160, 220), score=.99, stability=.98)
        merged = candidate(rect_mask(100, 100, 230, 220), score=.86, stability=.87)
        self.assertEqual(rank_candidates([single, merged]), 0)

    def test_surfaces_hand_and_huge_masks_rejected(self):
        desk = candidate(rect_mask(0, 380, 640, 480), score=.95)   # touches 3 borders
        hand = candidate(rect_mask(300, 200, 360, 300), overlap=.6)
        huge = candidate(rect_mask(20, 20, 620, 460))
        self.assertIsNone(rank_candidates([desk, hand, huge]))
        self.assertEqual(desk.rank_reason, "surface (touches borders)")

    def test_users_body_is_never_the_object(self):
        torso = candidate(rect_mask(200, 150, 440, 470))
        torso.person_overlap = .9
        mug = candidate(rect_mask(100, 300, 150, 360))
        self.assertEqual(rank_candidates([torso, mug]), 1)
        self.assertEqual(torso.rank_reason, "your body (person mask)")

    def test_clean_mask_keeps_prompted_component_and_fills_holes(self):
        m = rect_mask(100, 100, 200, 200)
        m[140:150, 140:150] = False            # specular highlight hole
        m[300:305, 300:305] = True             # unrelated speck
        cleaned = clean_mask(m, (120, 120))
        self.assertTrue(cleaned[145, 145])
        self.assertFalse(cleaned[302, 302])


def fake_embedding(frame, sid, now, zone=None):
    return Embedding(sid, frame, None, (768, 1024), [], zone if zone is not None else np.zeros(frame.shape[:2], bool), now, 0)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.s = Selector()
        self.frame = textured()
        self.zone = np.zeros((480, 640), bool)

    def aim(self, point, now, engaged=False, frame=None):
        return self.s.update(now, point, self.frame if frame is None else frame, [], self.zone,
                             None, engaged)

    def settle(self, point, start=0.0):
        job = None
        for i in range(20):
            job = self.aim(point, start + i * .03) or job
        return job

    def respond(self, job, mask, now):
        embedding = job.embedding or fake_embedding(job.frame, job.snapshot_id, job.captured_at)
        result = MaskResult(job.snapshot_id, job.prompt_id, job.positives[0], [candidate(mask)], 0, 0, 0, 0, embedding)
        return self.s.accept(result, now)

    def test_dwell_submits_live_snapshot_with_hand_negatives(self):
        job = self.s.update(0, (100, 100), self.frame, [], self.zone, None, False, negatives=[(1, 2)])
        self.assertIsNone(job)
        job = None
        for i in range(1, 12):
            job = self.s.update(i * .03, (100, 100), self.frame, [], self.zone, None, False,
                                negatives=[(1, 2)]) or job
        self.assertIsNotNone(job)
        self.assertIsNotNone(job.frame)
        self.assertEqual(job.negatives, ((1, 2),))

    def test_jitter_inside_preview_and_pinch_motion_never_retarget(self):
        job = self.settle((100, 100))
        self.assertTrue(self.respond(job, rect_mask(60, 60, 160, 160), .5))
        version = self.s.version
        # Pinch closing moves the index tip ~45 px, and some jitter leaves the outline briefly.
        for i, p in enumerate([(110, 110), (130, 140), (150, 150), (175, 150), (150, 150)]):
            self.assertIsNone(self.aim(p, .6 + i * .03))
        self.assertEqual(self.s.version, version)
        # Once the fingers start closing (engaged), even a big move cannot retarget.
        for i in range(10):
            self.assertIsNone(self.aim((400, 400), 1 + i * .03, engaged=True))
        self.assertEqual(self.s.state, "PREVIEW")

    def test_moving_away_while_analyzing_rejects_stale_result(self):
        job = self.settle((100, 100))
        self.aim((300, 300), .5)                       # user moved to another object
        self.assertFalse(self.respond(job, rect_mask(60, 60, 160, 160), .6))
        self.assertIsNone(self.s.mask)

    def test_cached_embedding_reused_unless_hand_was_there(self):
        job = self.settle((100, 100))
        zone = np.zeros((480, 640), bool)
        zone[250:350, 250:350] = True                  # the hand, as seen in that snapshot
        job.hand_zone = zone
        embedding = fake_embedding(job.frame, job.snapshot_id, .3, zone)
        result = MaskResult(job.snapshot_id, job.prompt_id, (100, 100), [candidate(rect_mask(60, 60, 160, 160))], 0, 0, 0, 0, embedding)
        self.s.accept(result, .4)
        reuse = self.settle((500, 100), start=1)
        self.assertIsNotNone(reuse.embedding)          # decode-only: fast
        self.assertIsNone(reuse.frame)
        self.s.accept(MaskResult(reuse.snapshot_id, reuse.prompt_id, (500, 100), [candidate(rect_mask(460, 60, 560, 160))], 0, 0, 0, 0, embedding), 1.5)
        fresh = self.settle((300, 300), start=2)
        self.assertIsNone(fresh.embedding)             # the hand hid this spot: re-encode
        self.assertIsNotNone(fresh.frame)

    def test_changed_scene_forces_new_encoding(self):
        job = self.settle((100, 100))
        self.respond(job, rect_mask(60, 60, 160, 160), .4)
        changed = self.frame.copy()
        changed[:] = 255 - changed
        job2 = None
        for i in range(20):
            job2 = self.aim((500, 300), 1 + i * .03, frame=changed) or job2
        self.assertIsNotNone(job2.frame)

    def test_carousel_cycles_outlines_while_pointing_steadily(self):
        job = self.settle((100, 100))
        small, large = rect_mask(80, 80, 120, 120), rect_mask(50, 50, 180, 180)
        embedding = fake_embedding(job.frame, job.snapshot_id, .3)
        self.s.accept(MaskResult(job.snapshot_id, job.prompt_id, (100, 100), [candidate(small), candidate(large)], 1, 0, 0, 0, embedding), .4)
        self.aim((100, 100), .5)
        self.assertEqual(self.s.choice, 1)
        self.aim((100, 100), 2.1)
        self.assertEqual(self.s.choice, 0)             # next outline after 1.6 s
        self.aim((100, 100), 3.8)
        self.assertEqual(self.s.choice, 1)             # wraps around

    def test_camera_gap_restarts_dwell_and_hand_loss_clears(self):
        self.aim((100, 100), 0)
        self.s.reset_dwell(1.0)
        self.assertIsNone(self.aim((100, 100), 1.05))
        job = self.settle((100, 100), start=1.1)
        self.respond(job, rect_mask(60, 60, 160, 160), 1.6)
        self.s.update(1.7, None, self.frame, [], self.zone, None, False)
        self.assertEqual(self.s.state, "PREVIEW")      # brief loss is tolerated
        self.s.update(2.7, None, self.frame, [], self.zone, None, False)
        self.assertEqual(self.s.state, "IDLE")

    def test_blocked_regions_are_not_selectable(self):
        blocked = lambda x, y: x < 200
        for i in range(12):
            self.assertIsNone(self.s.update(i * .03, (100, 100), self.frame, [], self.zone, blocked, False))

    def test_hand_zone_covers_landmarks(self):
        zone = hand_zone((480, 640), [make_hand((320, 200))])
        self.assertTrue(zone[200, 320] and zone[330, 320])
        self.assertFalse(zone[50, 50])


class WorkerTests(unittest.TestCase):
    def test_only_latest_pending_job_runs_and_input_is_owned(self):
        worker = SegmentationWorker()
        frame = np.zeros((10, 10, 3), np.uint8)
        worker.future = Mock(done=Mock(return_value=False))
        worker.request(Job("select", 1, 1, frame=frame))
        worker.request(Job("select", 1, 2, frame=frame))
        frame[:] = 255
        self.assertEqual(worker.pending.prompt_id, 2)
        self.assertFalse(worker.pending.frame.any())
        worker.future = None
        worker.close()

    def test_cancellation(self):
        worker = SegmentationWorker()
        worker.close()
        self.assertTrue(worker.run_options.terminate)
        with self.assertRaises(CancelledError):
            check_cancelled(worker.cancel)

    def test_missing_model_reports_error_without_crashing(self):
        import model_assets
        original = model_assets.urlopen
        model_assets.urlopen = Mock(side_effect=OSError("offline"))
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as empty:
                worker = SegmentationWorker(empty)
                worker.warm_up()
                deadline = time.time() + 10
                while worker.error is None and time.time() < deadline:
                    worker.poll()
                    time.sleep(.01)
                worker.close()
        finally:
            model_assets.urlopen = original
        self.assertIn("could not be downloaded", worker.error)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = textured(seed=1)
        self.mask = np.zeros((480, 640), bool)
        cv2.ellipse(self.mask.view(np.uint8), (300, 240), (40, 60), 0, 0, 360, 1, -1)
        self.frame[self.mask] = (40, 90, 200)
        cv2.putText(self.frame, "MUG", (272, 250), 0, .8, (255, 255, 255), 2)

    def test_static_small_motion_and_lighting(self):
        tracker = ObjectTracker(self.frame, self.mask)
        tracker.update(self.frame, None, 0)
        np.testing.assert_allclose(tracker.offset, 0, atol=.1)
        moved = np.roll(self.frame, (-4, 7), axis=(0, 1))
        brighter = cv2.convertScaleAbs(moved, alpha=1.15, beta=12)
        for i in range(8):
            tracker.update(brighter, None, .1 * i)
        np.testing.assert_allclose(tracker.offset, (7, -4), atol=1.6)
        self.assertEqual(tracker.state, "TRACKING")

    def test_occlusion_holds_pose_and_never_jumps_to_other_object(self):
        tracker = ObjectTracker(self.frame, self.mask)
        occluder = np.zeros((480, 640), bool)
        occluder[150:330, 240:360] = True
        tracker.update(self.frame, occluder, 0)
        self.assertEqual(tracker.state, "OCCLUDED")
        np.testing.assert_allclose(tracker.offset, 0)
        # The object vanishes and a look-alike appears 120 px away: no switch.
        other = self.frame.copy()
        other[self.mask] = textured(seed=5)[self.mask]
        other[np.roll(self.mask, 120, axis=1)] = (40, 90, 200)
        for i in range(30):
            tracker.update(other, None, .1 * i)
        np.testing.assert_allclose(tracker.offset, 0, atol=.1)
        self.assertTrue(tracker.state.startswith("LOST"))


class ReconstructionTests(unittest.TestCase):
    def test_scene_memory_is_used_only_if_it_saw_the_background(self):
        desk = textured(seed=2)
        with_object = desk.copy()
        mask = rect_mask(280, 200, 360, 300)
        with_object[mask] = (30, 30, 220)
        memory = SceneMemory()
        memory.observe(with_object, np.zeros((480, 640)), 0.0)
        rec = Reconstruction(with_object, mask, memory, 5.0)
        self.assertTrue(rec.source.startswith("inpaint"))       # memory only had the object
        memory.observe(desk, np.zeros((480, 640)), 2.0)          # background once visible
        rec = Reconstruction(with_object, mask, memory, 8.0)
        self.assertTrue(rec.source.startswith("scene memory"))
        out = with_object.copy()
        rec.render(out, with_object, np.zeros(2), None)
        inside = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.assertLess(np.abs(out[inside].astype(int) - desk[inside]).mean(), 3)

    def test_clean_plate_requires_absent_object_and_person_stays_in_front(self):
        desk = textured(seed=3)
        scene = desk.copy()
        mask = rect_mask(100, 100, 180, 200)
        scene[mask] = (20, 200, 20)
        memory = SceneMemory()
        memory.capture_plate(scene, np.zeros((480, 640)), 0)     # plate still shows object
        self.assertFalse(Reconstruction(scene, mask, memory, 5).source == "clean plate")
        memory.capture_plate(desk, np.zeros((480, 640)), 0)
        rec = Reconstruction(scene, mask, memory, 5)
        self.assertEqual(rec.source, "clean plate")
        person = np.zeros((480, 640), np.float32)
        person[140:160, 120:160] = 1.0                            # a hand in front
        live = scene.copy()
        live[person > 0] = (200, 180, 160)
        out = live.copy()
        rec.render(out, live, np.zeros(2), person)
        np.testing.assert_array_equal(out[150, 140], (200, 180, 160))

    def test_lighting_adaptation_and_fill_quality(self):
        desk = np.full((480, 640, 3), 120, np.uint8)
        mask = rect_mask(300, 200, 340, 260)
        scene = desk.copy()
        scene[mask] = 20
        rec = Reconstruction(scene, mask, None, 0)
        np.testing.assert_allclose(rec.patch[rec.hole[rec.box[1]:rec.box[3], rec.box[0]:rec.box[2]]], 120, atol=1)
        brighter = cv2.add(scene, 30)
        out = brighter.copy()
        rec.render(out, brighter, np.zeros(2), None)
        self.assertAlmostEqual(float(out[230, 320].mean()), 150, delta=3)

    def test_soft_alpha_and_push_pull(self):
        alpha = feathered_alpha(rect_mask(100, 100, 200, 200))
        self.assertEqual(alpha[150, 150], 1.0)
        self.assertEqual(alpha[50, 50], 0.0)
        self.assertTrue(0 < alpha[100, 150] < 1)                  # soft edge
        image = np.zeros((50, 50, 3), np.uint8)
        image[:, :25], image[:, 25:] = 50, 150
        hole = np.zeros((50, 50), bool)
        hole[10:40, 15:35] = True
        filled = push_pull_fill(image, hole)
        self.assertTrue(50 <= filled[25, 20, 0] <= 150)
        self.assertLess(filled[25, 16, 0], filled[25, 34, 0])     # gradient between borders


if __name__ == "__main__":
    unittest.main()
