"""Conservative low-light preparation for inference, never the displayed camera pixels."""
import cv2
import numpy as np


def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    median = float(np.median(light))
    if median >= 75:
        return frame
    contrast = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(light)
    light = cv2.addWeighted(light, .65, contrast, .35, 0)
    gamma = .70 if median < 45 else .85
    lut = np.clip((np.arange(256) / 255.) ** gamma * 255, 0, 255).astype(np.uint8)
    lab[:, :, 0] = cv2.LUT(light, lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
