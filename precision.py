"""Frozen-frame, category-independent point/box corrections for ambiguous selection."""
import numpy as np


class SelectionEditor:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.positives, self.negatives = [], []
        self.box = None
        self.drag = self.cursor = None
        self.embedding = None
        self.history = []

    def point(self, x, y):
        h, w = self.frame.shape[:2]
        return (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))

    def save(self):
        self.history.append((self.positives.copy(), self.negatives.copy(), self.box))
        self.history = self.history[-32:]

    def finish_drag(self, point):
        start, self.drag = self.drag, None
        if start is None:
            return False
        self.save()
        if np.linalg.norm(np.asarray(point) - start) < 6:
            self.positives = (self.positives + [point])[-32:]
        else:
            x0, x1 = sorted((start[0], point[0]))
            y0, y1 = sorted((start[1], point[1]))
            if x1 - x0 < 4 or y1 - y0 < 4:
                self.history.pop()
                return False
            self.box = (x0, y0, x1, y1)
            # A new box expresses a new target, not an accumulation of old targets.
            self.positives, self.negatives = [], []
        return True

    def exclude(self, point):
        if not self.positives and self.box is None:
            return False
        self.save()
        self.negatives = (self.negatives + [point])[-32:]
        return True

    def undo(self):
        if not self.history:
            return False
        self.positives, self.negatives, self.box = self.history.pop()
        return True

    @property
    def has_prompt(self):
        return bool(self.positives) or self.box is not None
