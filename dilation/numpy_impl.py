import numpy as np

from .native import kernel_offsets


def dilate_numpy(binary: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations < 1:
        raise ValueError("iterations должно быть >= 1")
    src = binary > 0
    h, w = src.shape
    offsets = kernel_offsets(kernel)
    if not offsets:
        raise ValueError("Примитив не содержит ни одного элемента")
    for _ in range(iterations):
        padded = np.pad(src, 1, constant_values=False)
        out = np.zeros((h, w), dtype=bool)
        for dr, dc in offsets:
            out |= padded[dr:dr + h, dc:dc + w]
        src = out
    return src.astype(np.uint8) * 255
