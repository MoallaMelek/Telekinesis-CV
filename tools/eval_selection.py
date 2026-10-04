r"""Contact sheet: every EdgeSAM candidate per prompt point, with the chosen one marked.

Uses public SAM demo photographs (downloaded to .cache/fixtures), never webcam images.
    .venv\Scripts\python.exe tools\eval_selection.py out.jpg
"""
from pathlib import Path
import sys
from urllib.request import urlopen

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation import EdgeSAM, rank_candidates  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / ".cache" / "fixtures"
BASE = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/"
# (image, point in 640x480 coords, what a person touching there most likely means)
CASES = [
    ("groceries.jpg", (300, 300), "left bag"), ("groceries.jpg", (345, 250), "second bag"),
    ("groceries.jpg", (450, 300), "right bag"), ("groceries.jpg", (410, 170), "bag contents"),
    ("groceries.jpg", (120, 60), "car edge/tail light"), ("groceries.jpg", (560, 200), "car body"),
    ("truck.jpg", (230, 250), "truck (wheel arch)"), ("truck.jpg", (420, 330), "road"),
    ("truck.jpg", (300, 160), "truck (rear window)"), ("dog.jpg", (300, 240), "dog (head)"),
    ("dog.jpg", (560, 80), "person (leg)"), ("dog.jpg", (390, 230), "metal bowl"),
]


def load(name):
    path = FIXTURES / name
    if not path.is_file():
        FIXTURES.mkdir(parents=True, exist_ok=True)
        with urlopen(BASE + name, timeout=20) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024 or cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            raise RuntimeError(f"Invalid fixture download: {name}")
        path.write_bytes(data)
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unreadable fixture: {name}; remove it and retry")
    return cv2.resize(image, (640, 480))


def main(out_path):
    model = EdgeSAM(threads=4)
    cache, rows = {}, []
    for name, point, meaning in CASES:
        if name not in cache:
            cache[name] = model.encode(load(name))
        embedding = cache[name]
        candidates = model.decode(embedding, [point])
        choice = rank_candidates(candidates)
        tiles = []
        for i, c in enumerate(candidates):
            view = embedding.frame.copy()
            colour = (0, 255, 0) if i == choice else (255, 160, 0)
            view[c.mask] = (view[c.mask] * .45 + np.array(colour) * .55).astype(np.uint8)
            cv2.circle(view, point, 7, (0, 0, 255), -1)
            label = f"{'CHOSEN ' if i == choice else ''}a{c.area:.3f} s{c.score:.2f} st{c.stability:.2f}"
            cv2.putText(view, label, (8, 30), 0, .8, (0, 0, 0), 5)
            cv2.putText(view, label, (8, 30), 0, .8, (255, 255, 255), 2)
            tiles.append(cv2.resize(view, (320, 240)))
        while len(tiles) < 4:
            tiles.append(np.zeros((240, 320, 3), np.uint8))
        head = np.zeros((240, 160, 3), np.uint8)
        cv2.putText(head, meaning, (5, 120), 0, .5, (255, 255, 255), 1)
        rows.append(np.hstack([head] + tiles[:4]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(out_path, np.vstack(rows)):
        raise OSError(f"Could not save evaluation: {out_path}")
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "selection_eval.jpg")
