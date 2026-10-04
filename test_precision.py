"""Category-independent selection and native-buffer ownership regressions."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from main import App, parse_args
from precision import SelectionEditor
from scene import PersonSegmenter
from selection import Selector, hand_core, hand_zone
from segmentation import MaskResult, rank_candidates, clean_mask
from test_core import make_hand
from test_selection import candidate, rect_mask, fake_embedding, textured


class OwnershipTests(unittest.TestCase):
    def test_person_mask_owns_memory_after_native_result_released(self):
        segmenter = object.__new__(PersonSegmenter)
        segmenter.last_ms = -1
        borrowed = np.ones((10, 12), np.float32)
        result = SimpleNamespace(confidence_masks=[Mock(numpy_view=Mock(return_value=borrowed))])
        segmenter.model = Mock(segment_for_video=Mock(return_value=result))
        # This is an offline ownership test, not MediaPipe native initialization.
        # Headless Linux runners may not have its optional GLES runtime installed.
        with patch('scene.mp.Image', return_value=Mock()):
            owned = segmenter.segment(np.zeros((10, 12, 3), np.uint8), 0)
        self.assertFalse(np.shares_memory(owned, borrowed))
        borrowed[:] = 0
        self.assertTrue(owned.all())

    def test_core_does_not_invent_a_forearm_on_nearby_flat_objects(self):
        hand = make_hand((320, 200))
        broad = hand_zone((480, 640), [hand])
        core = hand_core((480, 640), [hand])
        self.assertGreater((broad & ~core).sum(), core.sum())
        self.assertTrue(core[200, 320])

    def test_false_person_and_hand_masks_do_not_veto_objects(self):
        held = candidate(rect_mask(230, 230, 330, 360), score=.98)
        held.person_overlap = .95
        held.hand_overlap = .8
        self.assertEqual(rank_candidates([held]), 0)
        self.assertTrue(held.selectable)
        flat = candidate(rect_mask(100, 350, 180, 380), score=.99, overlap=1)
        flat.person_overlap = .02
        table = candidate(rect_mask(20, 200, 390, 420), score=.65, stability=.8)
        self.assertEqual(rank_candidates([table, flat]), 1)

    def test_explicit_box_allows_edge_objects_but_not_nonfinite_or_whole_frame(self):
        edge = candidate(rect_mask(0, 200, 640, 480))
        bad = candidate(rect_mask(20, 20, 100, 100), score=float('nan'))
        full = candidate(np.ones((480, 640), bool))
        self.assertIsNone(rank_candidates([edge]))
        self.assertEqual(rank_candidates([edge, bad, full], .95, explicit=True), 0)

    def test_all_positive_components_survive_cleanup_without_importing_neighbours(self):
        mask = rect_mask(20, 20, 50, 50, (160, 160))
        mask[90:120, 90:120] = True
        mask[90:120, 130:155] = True
        cleaned = clean_mask(mask, (30, 30), [(100, 100)])
        self.assertTrue(cleaned[30, 30] and cleaned[100, 100])
        self.assertFalse(cleaned[100, 140])


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.worker = Mock(error=None, ready=True, model=None, poll=Mock(return_value=None))
        self.app = App(640, 480, self.worker, parse_args([]))
        self.frame = textured()
        self.app.step(self.frame, 0, .04, [])
        self.app.started_wall = 0

    def click(self, x, y):
        with patch('main.time.perf_counter', return_value=1):
            self.app.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0)

    def test_click_without_hover_submits_and_never_extracts_unconfirmed_mask(self):
        self.click(100, 100)
        self.app.step(self.frame, 1, .04, [])
        job = self.worker.request.call_args.args[0]
        self.assertEqual(job.positives, ((100, 100),))
        self.assertEqual(self.app.selector.state, 'ANALYZING')
        self.assertEqual(self.app.counters['locks'], 0)

    def test_failed_point_can_be_retried_at_exact_same_pixel(self):
        for t in (1, 2):
            self.app.selector.clear('NO_OBJECT')
            self.click(100, 100)
            self.app.step(self.frame, t, .04, [])
        self.assertEqual(self.worker.request.call_count, 2)

    def test_mouse_motion_does_not_steal_real_hand_selection(self):
        hand = self.app.manip.hands[0]
        hand.points = make_hand((300, 200))
        hand.pinch_point = np.array([300., 200.])
        self.app.mouse.update(point=(100, 100), moved=1)
        self.assertEqual(self.app.choose_pointer(1), 0)
        self.click(100, 100)
        self.assertEqual(self.app.choose_pointer(1), 2)

    def test_box_corrections_undo_and_stale_results_share_frozen_encoding(self):
        self.app.key(ord('s'), .2)
        editor = self.app.editor
        self.assertFalse(np.shares_memory(editor.frame, self.frame))
        editor.drag = (180, 150)
        editor.finish_drag((300, 300))
        self.app.submit_edit(.3)
        first = self.worker.request.call_args.args[0]
        self.assertEqual(first.box, (180, 150, 300, 300))
        self.assertTrue(first.explicit)
        embedding = fake_embedding(editor.frame, first.snapshot_id, .3)
        result = MaskResult(first.snapshot_id, first.prompt_id, None,
                            [candidate(rect_mask(190, 170, 290, 280))], 0, 0, 0, 0, embedding)
        editor.exclude((290, 270))
        self.app.submit_edit(.4)
        self.worker.poll.return_value = ('select', result)
        self.app.edit_step(.5)
        self.assertIsNone(self.app.selector.candidate)  # obsolete outline cannot overwrite correction
        self.assertIs(editor.embedding, embedding)
        self.worker.poll.return_value = None
        self.app.submit_edit(.6)
        next_job = self.worker.request.call_args.args[0]
        self.assertIs(next_job.embedding, embedding)
        self.assertEqual(next_job.negatives, ((290, 270),))
        editor.undo()
        self.assertEqual(editor.negatives, [])

    def test_confirmed_correction_persists_without_hand_until_clicked(self):
        self.app.key(ord('s'), .2)
        self.app.selector.result = MaskResult(1, 1, None,
                                              [candidate(rect_mask(100, 100, 180, 180))], 0, 0, 0, 0)
        self.app.selector.choice = 0
        self.app.selector.state = 'PREVIEW'
        self.app.key(13, .4)
        self.app.step(self.frame, 10, .04, [])
        self.assertIsNone(self.app.editor)
        self.assertEqual(self.app.selector.state, 'PREVIEW')

    def test_selected_object_ownership_overrides_false_person_for_reconstruction(self):
        mask = rect_mask(100, 100, 180, 180)
        group, sprite = self.app.extract(self.frame, mask, 0, 1)
        self.app.manip.add(group, sprite)
        self.app._person = mask.astype(np.float32)
        core = np.zeros(mask.shape, bool)
        person = self.app.scene_person(core)
        self.assertFalse(person[mask].any())
        core[120:130, 120:130] = True
        self.assertTrue(self.app.scene_person(core)[125, 125])

    def test_manual_corrections_cannot_be_undone_by_hand_free_refinement(self):
        mask = rect_mask(100, 100, 180, 180)
        embedding = fake_embedding(self.frame, 1, 0)
        self.app.selector.result = MaskResult(1, 1, None, [candidate(mask)], 0,
                                              0, 0, 0, embedding, explicit=True)
        self.app.selector.choice = 0
        self.app.manip.hands[2].grip = self.app.manip.hands[2].pinch_point = np.array([130., 130.])
        self.app.lock(2, 1)
        group = self.app.groups()[0]
        self.assertTrue(group.refined)
        self.worker.request.reset_mock()
        self.worker.busy = False
        self.app.schedule_refine(self.frame, np.zeros(mask.shape, bool), 3)
        self.worker.request.assert_not_called()

    def test_new_selection_guidance_takes_precedence_over_existing_objects(self):
        group, sprite = self.app.extract(self.frame, rect_mask(100, 100, 180, 180), 0, 1)
        self.app.manip.add(group, sprite)
        self.app.selector.result = MaskResult(1, 1, None,
                                              [candidate(rect_mask(250, 150, 300, 220))], 0, 0, 0, 0)
        self.app.selector.choice = 0
        for state, stage in [('PREVIEW', 2), ('ANALYZING', 1)]:
            self.app.selector.state = state
            with patch('main.draw_hud', return_value=80) as hud:
                self.app.draw_feedback(self.frame.copy(), self.frame, 1, None, None,
                                       np.zeros((480, 640)), np.zeros((480, 640), bool))
            self.assertEqual(hud.call_args.kwargs['stage'], stage)
            self.assertIn('S precise selection', hud.call_args.kwargs['hint'])


if __name__ == '__main__':
    unittest.main()
