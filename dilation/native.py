from typing import List, Sequence, Tuple

import numpy as np

Image = List[List[int]]


def kernel_offsets(kernel: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    offsets = []
    for i in range(3):
        for j in range(3):
            if kernel[i][j]:
                offsets.append((2 - i, 2 - j))
    return offsets


def dilate_once(src: Image, height: int, width: int, offsets: List[Tuple[int, int]]) -> Image:
    zero_row = [0] * (width + 2)
    padded = [zero_row] + [[0] + row + [0] for row in src] + [zero_row]

    dst = []
    for y in range(height):
        probes = [(padded[y + dr], dc) for dr, dc in offsets]
        out_row = [0] * width
        for x in range(width):
            for row, dc in probes:
                if row[x + dc]:
                    out_row[x] = 1
                    break
        dst.append(out_row)
    return dst


def dilate_lists(src: Image, kernel: Sequence[Sequence[int]], iterations: int = 1) -> Image:
    if iterations < 1:
        raise ValueError("iterations должно быть >= 1")
    height, width = len(src), len(src[0])
    offsets = kernel_offsets(kernel)
    if not offsets:
        raise ValueError("Примитив не содержит ни одного элемента")
    for _ in range(iterations):
        src = dilate_once(src, height, width, offsets)
    return src


def dilate_native(binary: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    src = [[1 if v else 0 for v in row] for row in binary.tolist()]
    dst = dilate_lists(src, kernel.tolist(), iterations)
    return np.array(dst, dtype=np.uint8) * 255
