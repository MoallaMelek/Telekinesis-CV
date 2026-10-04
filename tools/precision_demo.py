"""Real-model correction workflow on synthetic pixels only; no camera or recording."""
from pathlib import Path
import sys
import time
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import App, parse_args
from segmentation import SegmentationWorker, iou
from test_pipeline import desk_scene


def main(destination):
    frame, truth = desk_scene()
    worker = SegmentationWorker()
    app = App(640, 480, worker, parse_args([]))
    app.started_wall = time.perf_counter()
    def now():
        return time.perf_counter() - app.started_wall
    def settle():
        deadline = time.perf_counter() + 30
        while app.selector.state != 'PREVIEW':
            app.step(frame, now(), .04, [])
            if worker.error:
                raise RuntimeError(worker.error)
            if time.perf_counter() > deadline:
                raise TimeoutError('Correction preview not ready')
            time.sleep(.02)
        return app.step(frame, now(), .04, [])
    try:
        app.step(frame, now(), .04, [])
        app.key(ord('s'), now())
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 290, 175, 0)
        app.on_mouse(cv2.EVENT_MOUSEMOVE, 430, 360, 0)
        app.on_mouse(cv2.EVENT_LBUTTONUP, 430, 360, 0)
        first = settle()
        embedding = app.editor.embedding
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 360, 285, 0)
        app.on_mouse(cv2.EVENT_LBUTTONUP, 360, 285, 0)
        app.on_mouse(cv2.EVENT_RBUTTONDOWN, 420, 340, 0)
        second = settle()
        assert app.editor.embedding is embedding, 'Correction did not reuse image encoding'
        assert iou(app.selector.mask, truth['phone']) > .90, 'Correction lost target'
        app.key(13, now())
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 360, 285, 0)
        app.step(frame, now(), .04, [])
        assert app.counters['locks'] == 1, 'Confirmed correction could not be lifted'
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(destination, np.hstack([first, second])):
            raise OSError('Cannot save demonstration')
        print('Box, corrections, cache reuse, Enter confirmation and extraction verified')
    finally:
        app.close()
        worker.close()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.cache/precision-demo.jpg')
