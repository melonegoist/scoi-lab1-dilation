from typing import Optional

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray, threshold: Optional[int] = None, invert: bool = False) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, mode)
    return binary
