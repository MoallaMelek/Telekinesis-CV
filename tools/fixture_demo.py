r"""Run the full interaction on the synthetic desk fixture and save a contact sheet.

Real EdgeSAM, real app logic, drawn hand + synthetic landmarks (no webcam imagery).
    .venv\Scripts\python.exe tools\fixture_demo.py sheet.jpg
"""
from pathlib import Path
import sys
import argparse
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import App, parse_args  # noqa: E402
from segmentation import SegmentationWorker  # noqa: E402
from test_core import make_hand  # noqa: E402
from test_pipeline import desk_scene, draw_hand  # noqa: E402

def main(output, debug=False):
    scene, _ = desk_scene()
    worker = SegmentationWorker(threads=4)
    worker.warm_up()
    app = App(640, 480, worker, parse_args(["--debug"] if debug else []))
    state = {"t": 0.0, "out": None}
    shots = []
    deadline = time.monotonic() + 60


    def step(hands=()):
        if worker.error:
            raise RuntimeError(worker.error)
        if time.monotonic() > deadline:
            raise TimeoutError("Fixture demo did not complete within 60 seconds")
        state["t"] += 1 / 25
        frame = scene.copy()
        person = np.zeros((480, 640), np.float32)
        for points, _ in hands:
            person[draw_hand(frame, points)] = 1
        state["out"] = app.step(frame, state["t"], 1 / 25,
                                [{"points": p, "z": np.zeros(21), "label": lab} for p, lab in hands], person)
        time.sleep(.005)


    def shot(title):
        image = state["out"].copy()
        caption = np.full((36, 640, 3), (18, 22, 28), np.uint8)
        cv2.putText(caption, title, (10, 24), 0, .5, (235, 240, 245), 1, cv2.LINE_AA)
        shots.append(np.vstack([caption, image]))


    def hand(tip, pinch=False, label="Right"):
        return (make_hand(tip, palm=70, pinch=pinch), label)


    try:
        while worker.model is None:
            step()
        shot("1 raw scene (nothing selected)")
        while app.selector.state != "PREVIEW":
            step([hand((165, 320))])
        shot("2 touch mug -> outline preview")
        for _ in range(4):
            step([hand((165, 320), True)])
        for i in range(1, 26):
            step([hand((165 + 10 * i, 320 - 4 * i), True)])
        shot("3 pinch + drag: original region reconstructed")
        for _ in range(6):
            step([hand((415, 220), True)])
        for _ in range(4):
            step([hand((415, 220))])
        for _ in range(40):
            step()
        shot("4 released: floats; hand-free refinement")
        obj = app.manip.objects[0]
        tip = obj.position + (0, 60)
        for _ in range(4):
            step([hand(tip)])
        for _ in range(4):
            step([hand(tip, True)])
        b = tip + (110, 0)
        for _ in range(4):
            step([hand(tip, True), hand(b, False, "Left")])
        for _ in range(4):
            step([hand(tip, True), hand(b, True, "Left")])
        for i in range(1, 21):
            step([hand(tip + (-2 * i, 5 * i), True), hand(b + (2 * i, 3 * i), True, "Left")])
        shot("5 two hands: scale + rotate")
        for _ in range(4):
            step([hand(tip + (-40, 100)), hand(b + (40, 60), False, "Left")])
        app.key(ord("h"), state["t"])
        for _ in range(10):
            step()
        shot("6 hidden (H / fist)")
        app.key(ord("h"), state["t"])
        app.key(ord("c"), state["t"])
        for _ in range(10):
            step()
        shot("7 shown + duplicated (C)")
        app.key(ord("r"), state["t"])
        for _ in range(10):
            step()
        shot("8 reset (R / open palm): back at origin")
        rows = [np.hstack(shots[i:i + 4]) for i in range(0, 8, 4)]
        out = str(output)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(out, np.vstack(rows)):
            raise OSError(f"Could not save fixture demo: {out}")
        print("wrote", out, "| background:", obj.group.reconstruction.source, "|", obj.group.refine)


    finally:
        app.close()
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="fixture_demo.jpg")
    parser.add_argument("--debug", action="store_true")
    options = parser.parse_args()
    main(options.output, options.debug)
