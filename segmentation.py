"""Point/box-prompted EdgeSAM inference on CPU, plus candidate-mask ranking.

Coordinate chain (all explicit, all tested):
    mirrored camera pixel (x, y)
    -> aspect-preserving resize so the longest side is 1024
    -> zero padding on the right/bottom to 1024x1024 (the encoder input)
    -> the decoder predicts 256x256 low-resolution logits for that padded square
    -> upsample to 1024, CROP the padding away, resize back to the camera size.
"""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
from pathlib import Path
from threading import Event
import time

import cv2
import numpy as np
import onnxruntime as ort

from model_assets import check_cancelled, ensure_asset

MODEL_DIR = Path(__file__).with_name("models") / "edgesam"
REVISION = "e0564124628944e1622973d4e0f68158b46f035a"
MODEL_FILES = {
    "edge_sam_3x_encoder.onnx": "719a498cf5b3fe9be9f01ee513e13d3915f9028aa4f23dfd30eaaa0a17143159",
    "edge_sam_3x_decoder.onnx": "83a2174d54571596913dcb7455d021e713623c3dca30a31c8c41ab98c9fb0863",
}
# SAM-family normalization on 0-255 RGB (the ONNX encoder does not include it).
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], np.float32)
POSITIVE, NEGATIVE, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, PADDING = 1, 0, 2, 3, -1


def ensure_models(directory, cancel=None):
    """Download revision-pinned files once and verify SHA-256; never trust partial files."""
    directory = Path(directory)
    for name, digest in MODEL_FILES.items():
        path = directory / name
        url = f"https://huggingface.co/chongzhou/EdgeSAM/resolve/{REVISION}/{name}"
        ensure_asset(path, url, digest, cancel=cancel)


def preprocess(frame):
    """BGR camera frame -> normalized, padded 1x3x1024x1024 RGB tensor."""
    height, width = frame.shape[:2]
    scale = 1024 / max(height, width)
    new_h, new_w = int(height * scale + 0.5), int(width * scale + 0.5)
    rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (new_w, new_h),
                     interpolation=cv2.INTER_LINEAR).astype(np.float32)
    tensor = np.zeros((1, 3, 1024, 1024), np.float32)
    tensor[0, :, :new_h, :new_w] = ((rgb - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)
    return tensor, (new_h, new_w)


def to_model_coords(points, resized_size, original_size):
    new_h, new_w = resized_size
    height, width = original_size
    points = np.asarray(points, np.float32).reshape(-1, 2)
    return points * np.array([new_w / width, new_h / height], np.float32)


def restore_logits(low_resolution, resized_size, original_size):
    new_h, new_w = resized_size
    height, width = original_size
    square = cv2.resize(low_resolution, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # Undo padding FIRST, then resize. Resizing 256 -> camera directly stretches masks.
    return cv2.resize(square[:new_h, :new_w], (width, height), interpolation=cv2.INTER_LINEAR)


@dataclass
class Embedding:
    """One encoded, immutable camera snapshot plus the hand pose visible in it."""
    snapshot_id: int
    frame: np.ndarray
    features: np.ndarray
    resized_size: tuple
    hand_points: list          # landmark arrays (mirrored pixels) visible in THIS frame
    hand_zone: np.ndarray      # bool mask: approximate hand region in THIS frame
    captured_at: float
    encoder_ms: float
    person: np.ndarray = None  # bool person-segmentation mask of THIS frame (optional)


@dataclass
class Candidate:
    mask: np.ndarray           # bool, camera resolution
    logits: np.ndarray         # float32 logits, camera resolution (soft mask source)
    score: float               # model-predicted IoU
    stability: float           # IoU of masks thresholded at logit -1 and +1
    area: float                # fraction of the frame
    hand_overlap: float        # fraction of the mask inside the hand zone
    borders: int = 0           # image edges the mask runs along (surfaces touch 2+)
    person_overlap: float = 0.0  # fraction of the mask that is the user's body
    rank_reason: str = ""
    selectable: bool = True    # set by ranking; rejected masks are debug-only


@dataclass
class MaskResult:
    snapshot_id: int
    prompt_id: int
    point: tuple
    candidates: list
    choice: int
    encoder_ms: float
    decoder_ms: float
    total_ms: float
    embedding: Embedding = field(repr=False, default=None)


def stability_score(logits, offset=1.0):
    high, low = logits > offset, logits > -offset
    return float(high.sum() / max(low.sum(), 1))


def clean_mask(mask, point=None):
    """Keep the connected component at the prompt (or the largest), fill small holes."""
    mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    keep = None
    if point is not None:
        x, y = int(round(point[0])), int(round(point[1]))
        h, w = mask.shape
        window = labels[max(0, y - 6):min(h, y + 7), max(0, x - 6):min(w, x + 7)]
        ids = window[window > 0]
        if ids.size:
            keep = int(np.bincount(ids).argmax())
    if keep is None:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = labels == keep
    # Also keep sizable pieces (e.g. a mug handle split off by a thin gap).
    for i in range(1, count):
        if i != keep and stats[i, cv2.CC_STAT_AREA] >= .15 * stats[keep, cv2.CC_STAT_AREA]:
            main |= labels == i
    # Fill interior holes smaller than 2% of the object (specular highlights, labels).
    inverse = (~main).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    limit = .02 * main.sum()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]
        if not touches_border and area <= limit:
            main[labels == i] = True
    return main


def rank_candidates(candidates, max_area=.30):
    """Choose the whole object, not the most confident part - nor two merged objects.

    SAM's predicted IoU is highest for small, crisp parts (a label, a truck's door).
    Touch selection almost always means the whole distinct object, so among plausible
    candidates (not huge, not the hand) prefer the LARGEST whose predicted score and
    edge stability are close to the best. Stability (does the mask change when the
    logit threshold moves by +-1?) is what separates a real whole object from a merge of
    two neighbouring objects: the seam between them is uncertain, so merges are less
    stable (measured 0.87 for two touching bags vs 0.98 for one bag).
    """
    def allowed(c):   # hard filters: never the hand, the user's body, or a surface
        return (0 < c.area <= max_area and c.borders <= 1 and c.person_overlap < .6
                and c.hand_overlap < .5 and math.isfinite(c.score)
                and math.isfinite(c.stability))

    for c in candidates:
        c.selectable = allowed(c)

    plausible = [i for i, c in enumerate(candidates) if allowed(c) and c.hand_overlap < .25
                 and c.stability >= .75 and c.score >= .55]
    if not plausible:
        plausible = [i for i, c in enumerate(candidates) if allowed(c) and c.hand_overlap < .5]
    if not plausible:
        for c in candidates:
            c.rank_reason = ("too large" if c.area > max_area else "surface (touches borders)"
                             if c.borders > 1 else "your body (person mask)"
                             if c.person_overlap >= .6 else "covers hand")
        return None
    best = max(candidates[i].score for i in plausible)
    competitive = [i for i in plausible if candidates[i].score >= best - .15]
    steady = max(candidates[i].stability for i in competitive)
    near = [i for i in competitive if candidates[i].stability >= steady - .08]
    choice = max(near, key=lambda i: candidates[i].area)
    for i, c in enumerate(candidates):
        c.rank_reason = ("chosen: largest steady" if i == choice else
                         "too large" if c.area > max_area else
                         "surface (touches borders)" if c.borders > 1 else
                         "your body (person mask)" if c.person_overlap >= .6 else
                         "covers hand" if c.hand_overlap >= .25 else
                         "unstable edges" if c.stability < steady - .08 else
                         "low score" if c.score < best - .15 else "smaller part")
    return choice


class EdgeSAM:
    def __init__(self, directory=MODEL_DIR, threads=4, cancel=None, run_options=None):
        self.cancel = cancel
        self.run_options = run_options or ort.RunOptions()
        ensure_models(directory, cancel)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Without this, idle ORT threads spin and steal CPU from MediaPipe.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        directory = Path(directory)
        self.encoder = ort.InferenceSession(str(directory / "edge_sam_3x_encoder.onnx"),
                                            options, providers=["CPUExecutionProvider"])
        check_cancelled(cancel)
        self.decoder = ort.InferenceSession(str(directory / "edge_sam_3x_decoder.onnx"),
                                            options, providers=["CPUExecutionProvider"])

    def encode(self, frame, snapshot_id=0, hand_points=(), hand_zone=None, captured_at=0.0):
        check_cancelled(self.cancel)
        started = time.perf_counter()
        tensor, resized = preprocess(frame)
        features = self.encoder.run(None, {"image": tensor}, self.run_options)[0]
        if hand_zone is None:
            hand_zone = np.zeros(frame.shape[:2], bool)
        return Embedding(snapshot_id, frame, features, resized, list(hand_points), hand_zone,
                         captured_at, (time.perf_counter() - started) * 1000)

    def decode(self, embedding, positives=(), negatives=(), box=None):
        """Return every candidate for one prompt (camera-resolution masks)."""
        check_cancelled(self.cancel)
        size = embedding.frame.shape[:2]
        coords, labels = [], []
        for p in positives:
            coords.append(p); labels.append(POSITIVE)
        for p in negatives:
            coords.append(p); labels.append(NEGATIVE)
        if box is not None:
            x0, y0, x1, y1 = box
            coords += [(x0, y0), (x1, y1)]
            labels += [BOX_TOP_LEFT, BOX_BOTTOM_RIGHT]
        else:
            # Official SAM ONNX usage: without a box, append one padding point.
            coords.append((0, 0)); labels.append(PADDING)
        model_coords = to_model_coords(coords, embedding.resized_size, size)
        if box is None:
            model_coords[-1] = 0
        feed = {"image_embeddings": embedding.features,
                "point_coords": model_coords[None],
                "point_labels": np.array([labels], np.float32)}
        scores, masks = self.decoder.run(None, feed, self.run_options)
        anchor = positives[0] if positives else None
        candidates = []
        for low, score in zip(masks[0], scores[0]):
            logits = restore_logits(low, embedding.resized_size, size)
            mask = logits > 0          # logit 0 == probability 0.5
            if anchor is not None:
                x, y = int(round(anchor[0])), int(round(anchor[1]))
                near = mask[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
                if not near.any():
                    continue            # The model answered a different question.
            mask = clean_mask(mask, anchor)
            area = float(mask.mean())
            if area * mask.size < 60:
                continue
            overlap = float((mask & embedding.hand_zone).sum() / max(mask.sum(), 1))
            body = (0.0 if embedding.person is None else
                    float((mask & embedding.person).sum() / max(mask.sum(), 1)))
            candidates.append(Candidate(mask, logits, float(score), stability_score(logits),
                                        area, overlap, touched_borders(mask), body))
        # Deduplicate near-identical masks so "alternatives" are really different.
        unique = []
        for c in sorted(candidates, key=lambda c: -c.score):
            if all(iou(c.mask, u.mask) < .92 for u in unique):
                unique.append(c)
        return unique


def touched_borders(mask, fraction=.08):
    """How many image edges the mask covers along >8% of their length."""
    edges = (mask[0], mask[-1], mask[:, 0], mask[:, -1])
    return sum(float(edge.mean()) > fraction for edge in edges)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass
class Job:
    kind: str                   # "select" or "refine"
    snapshot_id: int
    prompt_id: int
    frame: np.ndarray = None    # present when a new encoding is needed
    embedding: Embedding = None # reuse a cached encoding instead
    positives: tuple = ()
    negatives: tuple = ()
    box: tuple = None
    hand_points: tuple = ()
    hand_zone: np.ndarray = None
    person: np.ndarray = None
    captured_at: float = 0.0
    max_area: float = .30


class SegmentationWorker:
    """One running inference plus ONE replaceable pending job: never a growing queue.

    Every job carries (snapshot_id, prompt_id). The main thread bumps prompt_id when the
    user's intent changes, so late results for an abandoned prompt are rejected.
    """
    def __init__(self, directory=MODEL_DIR, threads=4):
        self.directory, self.threads = directory, threads
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmentation")
        self.future = None
        self.pending = None
        self.model = None
        self.error = None
        self.ready = False
        self.cancel = Event()
        self.run_options = ort.RunOptions()
        self.last_encoder_ms = self.last_decoder_ms = 0.0

    @property
    def busy(self):
        return self.future is not None or self.pending is not None

    def warm_up(self):
        """Load (and if needed download) the model in the background at startup."""
        if self.future is None and self.model is None:
            self.future = self.pool.submit(self._load)

    def request(self, job):
        if job.frame is not None:
            job.frame = job.frame.copy()   # the worker owns an immutable copy
        self.pending = job

    def discard_pending(self):
        self.pending = None

    def _load(self):
        check_cancelled(self.cancel)
        if self.model is None:
            self.model = EdgeSAM(self.directory, self.threads, self.cancel, self.run_options)
        return None

    def _run(self, job):
        self._load()
        started = time.perf_counter()
        embedding = job.embedding
        if embedding is None:
            embedding = self.model.encode(job.frame, job.snapshot_id, job.hand_points,
                                          job.hand_zone, job.captured_at)
            embedding.person = job.person
            encoder_ms = embedding.encoder_ms
        else:
            encoder_ms = 0.0
        decoding = time.perf_counter()
        candidates = self.model.decode(embedding, job.positives, job.negatives, job.box)
        decoder_ms = (time.perf_counter() - decoding) * 1000
        choice = rank_candidates(candidates, job.max_area) if candidates else None
        point = tuple(job.positives[0]) if job.positives else None
        return job.kind, MaskResult(job.snapshot_id, job.prompt_id, point, candidates, choice,
                                    encoder_ms, decoder_ms,
                                    (time.perf_counter() - started) * 1000, embedding)

    def poll(self):
        """Return (kind, MaskResult) when a job finished, else None. Never blocks."""
        result = None
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                self.ready = self.model is not None
                self.error = None
            except CancelledError:
                result = None
            except Exception as error:     # surfaced on screen; the camera loop keeps running
                self.error = f"{type(error).__name__}: {error}"
                self.ready = False
                self.pending = None
                print(f"Segmentation unavailable: {self.error}", flush=True)
            self.future = None
            if result is not None:
                self.last_encoder_ms = result[1].encoder_ms or self.last_encoder_ms
                self.last_decoder_ms = result[1].decoder_ms
        if self.future is None and self.pending is not None and self.error is None:
            self.future = self.pool.submit(self._run, self.pending)
            self.pending = None
        return result

    def retry(self):
        if self.future is not None:
            return
        self.error = None
        self.pending = None
        self.ready = False
        self.model = None
        self.warm_up()

    def close(self):
        self.pending = None
        self.cancel.set()
        self.run_options.terminate = True   # interrupts running ONNX work between operators
        self.pool.shutdown(wait=True, cancel_futures=True)
