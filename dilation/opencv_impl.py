import cv2
import numpy as np


def dilate_opencv(binary: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    if not kernel.any():
        raise ValueError("Примитив не содержит ни одного элемента")
    if iterations < 1:
        raise ValueError("iterations должно быть >= 1")
    src = cv2.compare(binary.astype(np.uint8, copy=False), 0, cv2.CMP_GT)
    reflected = cv2.flip(kernel.astype(np.uint8), -1)
    return cv2.dilate(
        src,
        reflected,
        iterations=iterations,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
