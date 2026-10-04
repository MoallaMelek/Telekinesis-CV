"""Scene-level CV: person segmentation, locked-object tracking, background reconstruction.

DETECTION asks "is there an object, and where?"  SEGMENTATION asks "which exact pixels
belong to it?" (EdgeSAM, once, when you select).  TRACKING asks "where did THAT same
object go in this new frame?" and is done here cheaply every frame, by matching the
object's own appearance near its last known position. It never searches for a
different object, so it can lose confidence but cannot silently switch targets.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from model_assets import ensure_asset

PERSON_MODEL = Path(__file__).with_name("models") / "mediapipe" / "selfie_segmenter.tflite"
PERSON_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
              "selfie_segmenter/float16/1/selfie_segmenter.tflite")
PERSON_SHA256 = "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b"


def ensure_person_model():
    ensure_asset(PERSON_MODEL, PERSON_URL, PERSON_SHA256, max_bytes=2 * 1024 * 1024)


class PersonSegmenter:
    """MediaPipe selfie segmentation: per-pixel probability that a pixel is a person.

    This is a real segmentation network (about 5 ms per frame here), not a landmark hull.
    It was trained on selfies; hands far from the body or low light reduce its quality.
    """

    def __init__(self):
        ensure_person_model()
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(PERSON_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False)
        self.model = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        self.last_ms = -1

    def segment(self, frame, seconds):
        stamp = max(self.last_ms + 1, int(seconds * 1000))
        self.last_ms = stamp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.model.segment_for_video(image, stamp)
        mask = result.confidence_masks[0].numpy_view()
        # numpy_view borrows MediaPipe's native buffer. Own it before `result` is
        # released (or the segmenter is closed); ascontiguousarray may return a view.
        return np.array(mask.reshape(frame.shape[:2]), dtype=np.float32, order="C", copy=True)

    def close(self):
        self.model.close()


def bbox_of(mask, margin=0, shape=None):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = shape if shape is not None else mask.shape[:2]
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + 1 + margin), min(h, int(ys.max()) + 1 + margin))


class ObjectTracker:
    """Translation-only tracking of ONE physical object by masked template matching.

    Template = the object's pixels plus a thin ring of context (edges carry most of the
    signal for plain objects). Each frame we search a small window around the last pose
    using normalized cross-correlation (TM_CCOEFF_NORMED), which tolerates uniform
    brightness/contrast changes. Pixels covered by the hand/person are removed from the
    comparison; if too much is covered we HOLD the last pose (OCCLUDED) rather than guess.
    """

    def __init__(self, frame, mask, search=28, accept=.55, lost_after=1.5):
        self.search, self.accept, self.lost_after = search, accept, lost_after
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8))
        self.box = bbox_of(ring)
        x0, y0, x1, y1 = self.box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        longest = max(x1 - x0, y1 - y0)
        self.factor = min(1.0, 96 / max(longest, 1))     # search at <= ~96 px for speed
        self.template = self._scale(gray[y0:y1, x0:x1])
        self.template_mask = (self._scale(ring[y0:y1, x0:x1] * 255) > 127).astype(np.uint8) * 255
        self.textured = float(self.template[self.template_mask > 0].std()) > 4
        self.offset = np.zeros(2)          # current translation vs. the lock frame (pixels)
        self.state = "TRACKING"
        self.score = 1.0
        self.occluded_fraction = 0.0
        self.bad_since = None

    def _scale(self, image):
        if self.factor >= 1:
            return image.copy()
        size = (max(8, int(image.shape[1] * self.factor)), max(8, int(image.shape[0] * self.factor)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def update(self, frame, occluder, now):
        x0, y0, x1, y1 = self.box
        h, w = frame.shape[:2]
        dx, dy = np.rint(self.offset).astype(int)
        # Occlusion check at the expected position.
        ex0, ey0, ex1, ey1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
        if ex0 < 0 or ey0 < 0 or ex1 > w or ey1 > h:
            return self._bad(now, "EDGE")
        covered = occluder[ey0:ey1, ex0:ex1] if occluder is not None else None
        tmask = self.template_mask
        if covered is not None and covered.any():
            covered_small = self._scale(covered.astype(np.uint8) * 255) > 127
            self.occluded_fraction = float((covered_small & (tmask > 0)).sum() / max((tmask > 0).sum(), 1))
            tmask = tmask.copy()
            tmask[covered_small] = 0
        else:
            self.occluded_fraction = 0.0
        if self.occluded_fraction > .45 or not self.textured:
            # Holding still is the honest answer when the object is mostly hidden
            # (or when a completely flat object gives nothing to match).
            if self.occluded_fraction > .45:
                return self._bad(now, "OCCLUDED")
            self.state, self.score = "HOLD (no texture)", 0.0
            return self.offset
        r = self.search
        sx0, sy0 = max(0, ex0 - r), max(0, ey0 - r)
        sx1, sy1 = min(w, ex1 + r), min(h, ey1 + r)
        gray = cv2.cvtColor(frame[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2GRAY)
        window = self._scale(gray)
        if window.shape[0] < self.template.shape[0] or window.shape[1] < self.template.shape[1]:
            return self._bad(now, "EDGE")
        result = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED, mask=tmask)
        result[~np.isfinite(result)] = -1
        _, score, _, location = cv2.minMaxLoc(result)
        self.score = float(score)
        if score < self.accept:
            return self._bad(now, "LOW MATCH")
        found = np.array([sx0 + location[0] / self.factor - x0, sy0 + location[1] / self.factor - y0])
        if np.linalg.norm(found - self.offset) > r * 1.2:
            return self._bad(now, "JUMP REJECTED")
        # Sub-pixel jitter makes the hidden region shimmer; ignore < 1.5 px changes.
        if np.linalg.norm(found - self.offset) >= 1.5:
            self.offset = self.offset + .5 * (found - self.offset)
        self.state, self.bad_since = "TRACKING", None
        return self.offset

    def _bad(self, now, reason):
        if self.bad_since is None:
            self.bad_since = now
        lost = now - self.bad_since > self.lost_after
        self.state = ("LOST (holding last pose)" if lost and reason != "OCCLUDED"
                      else reason)
        return self.offset


def feathered_alpha(mask, radius=2):
    """Binary mask (0/1) -> soft alpha (0..1) with a ~radius px transition at the edge.

    Fade edge pixels inside the silhouette to hide aliasing without adding surrounding
    background. Unlike erosion, this preserves thin handles and cords.
    """
    if radius <= 0:
        return mask.astype(np.float32)
    # Fade inside the silhouette. Erosion deleted thin handles and cords entirely;
    # distance-based alpha keeps them visible without importing background pixels.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(distance / radius * 1.15, 0, 1)


class SceneMemory:
    """Timestamped frames (1 per second, ~40 s) + an optional clean background plate.

    A frame showing the object is NOT a background plate. Memory helps only if the camera
    once saw the spot WITHOUT the object (it was placed later, or moved earlier), or if
    you capture a real clean plate (P) with the object physically removed.
    """

    def __init__(self, period=1.0, length=40):
        self.period = period
        self.frames = deque(maxlen=length)
        self.next_at = 0.0
        self.plate = None

    def observe(self, frame, foreground, now):
        if now >= self.next_at:
            self.frames.append((now, frame.copy(), foreground > .5))
            self.next_at = now + self.period

    def capture_plate(self, frame, foreground, now):
        self.plate = (now, frame.copy(), foreground > .5)

    def find_background(self, frame, hole, ring, now):
        """Return (pixels, label) from the best frame where the hole region looks different
        from the object while the surrounding ring matches (same camera pose)."""
        candidates = list(self.frames)[::-1]
        if self.plate is not None:
            candidates.insert(0, self.plate)
        for stamp, old, person in candidates:
            if now - stamp < 1.5:
                continue       # too recent: the object was certainly already there
            visible_ring = ring & ~person
            if visible_ring.sum() < 50 or (hole & person).sum() > .05 * hole.sum():
                continue
            ring_diff = cv2.absdiff(old, frame)[visible_ring].mean()
            hole_diff = cv2.absdiff(old, frame)[hole].mean()
            if ring_diff < 12 and hole_diff > 28:
                label = "clean plate" if old is (self.plate[1] if self.plate else None) else \
                    f"scene memory ({now - stamp:.0f}s ago)"
                return old, label
        return None, None


class Reconstruction:
    """Hide the stationary physical object: paint plausible background over its region.

    Sources, best first: a clean plate / scene memory that genuinely saw the background,
    else inpainting, which INVENTS pixels from the surrounding colours: an instant smooth
    pyramid fill, plus OpenCV-contrib FSR computed in the background (texture guess; B
    switches). Inpainted holes look blurred or smeared; that artifact is expected.
    """

    def __init__(self, frame, mask, memory, now, inpainter=None):
        h, w = mask.shape
        self.hole = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        grown = cv2.dilate(self.hole.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        self.ring = grown & ~self.hole
        self.box = bbox_of(grown, margin=8, shape=(h, w))
        x0, y0, x1, y1 = self.box
        self.alpha = cv2.GaussianBlur(self.hole.astype(np.float32), (7, 7), 0)[y0:y1, x0:x1]
        self.alpha = np.clip(self.alpha * 1.4, 0, 1)[..., None]
        self.ring_crop = self.ring[y0:y1, x0:x1]
        self.ring_mean = frame[y0:y1, x0:x1][self.ring_crop].mean(axis=0) if self.ring_crop.any() else np.zeros(3)
        background, label = memory.find_background(frame, self.hole, self.ring, now) if memory else (None, None)
        self.future = None
        self.patches = {}              # method name -> float32 patch; B cycles between them
        if background is not None:
            self.patches[label] = background[y0:y1, x0:x1].astype(np.float32)
        crop = frame[y0:y1, x0:x1]
        hole = self.hole[y0:y1, x0:x1]
        self.patches["inpaint: smooth fill (blurry)"] = push_pull_fill(crop, hole)
        self.source = next(iter(self.patches))
        if inpainter is not None:
            self.future = inpainter.submit(fsr_inpaint, crop.copy(), hole.astype(np.uint8) * 255)

    @property
    def patch(self):
        return self.patches[self.source]

    def cycle(self):
        names = list(self.patches)
        self.source = names[(names.index(self.source) + 1) % len(names)]
        return self.source

    def poll(self):
        if self.future is not None and self.future.done():
            try:
                self.patches["inpaint: FSR (textured guess)"] = self.future.result().astype(np.float32)
            except Exception as error:   # keep the smooth fill
                print(f"FSR inpainting failed: {error}", flush=True)
            self.future = None

    def render(self, out, live, offset, foreground):
        """Composite the patch at the tracked position; people/hands stay in front."""
        self.poll()
        h, w = out.shape[:2]
        dx, dy = np.rint(offset).astype(int)
        x0, y0, x1, y1 = self.box
        tx0, ty0, tx1, ty1 = max(0, x0 + dx), max(0, y0 + dy), min(w, x1 + dx), min(h, y1 + dy)
        if tx1 <= tx0 or ty1 <= ty0:
            return None
        sx0, sy0 = tx0 - (x0 + dx), ty0 - (y0 + dy)
        sl = (slice(sy0, sy0 + ty1 - ty0), slice(sx0, sx0 + tx1 - tx0))
        patch, alpha = self.patch[sl], self.alpha[sl]
        ring = self.ring_crop[sl]
        live_roi = live[ty0:ty1, tx0:tx1]
        fg = foreground[ty0:ty1, tx0:tx1] if foreground is not None else None
        # Lighting adaptation: shift the patch by how much the visible ring's mean colour
        # changed since the lock frame (auto-exposure drift, a lamp switched on).
        usable = ring if fg is None else ring & (fg < .3)
        if usable.sum() > 30:
            delta = np.clip(live_roi[usable].mean(axis=0) - self.ring_mean, -40, 40)
            patch = patch + delta
        a = alpha if fg is None else alpha * (1 - fg[..., None])
        roi = out[ty0:ty1, tx0:tx1].astype(np.float32)
        out[ty0:ty1, tx0:tx1] = np.clip(roi * (1 - a) + patch * a, 0, 255).astype(np.uint8)
        return (tx0, ty0, tx1, ty1)


def push_pull_fill(image, hole, levels=6):
    """Instant hole fill: 'pull' known colours down an image pyramid with mask-weighted
    averaging, then 'push' them back up. Gives a smooth gradient between the hole's
    borders (no streaks like Telea on big holes), but no texture. ~1-2 ms."""
    known = (~hole).astype(np.float32)
    colour = image.astype(np.float32) * known[..., None]
    pyramid = [(colour, known)]
    for _ in range(levels):
        c, k = pyramid[-1]
        if min(k.shape) < 4:
            break
        pyramid.append((cv2.pyrDown(c), cv2.pyrDown(k)))
    c, k = pyramid[-1]
    filled = c / np.maximum(k, 1e-4)[..., None]
    for c, k in reversed(pyramid[:-1]):
        up = cv2.resize(filled, (k.shape[1], k.shape[0]), interpolation=cv2.INTER_LINEAR)
        weight = np.clip(k * 4, 0, 1)[..., None]      # trust real pixels where present
        filled = np.where(weight > 0, (c / np.maximum(k, 1e-4)[..., None]) * weight + up * (1 - weight), up)
    out = image.astype(np.float32)
    out[hole] = filled[hole]
    return out


def fsr_inpaint(crop, hole):
    h, w = crop.shape[:2]
    factor = min(1.0, 220 / max(h, w))       # bound the cost (~0.3 s at 200 px)
    small = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else crop
    small_hole = (cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                  if factor < 1 else hole)
    out = np.zeros_like(small)
    cv2.xphoto.inpaint(small, 255 - small_hole, out, cv2.xphoto.INPAINT_FSR_FAST)
    if factor < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        # Keep the original pixels outside the hole at full resolution.
        out = np.where(hole[..., None] > 0, out, crop)
    return out


def make_inpainter():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")
