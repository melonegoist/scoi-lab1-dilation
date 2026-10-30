import argparse
import os
import time

import cv2
import numpy as np

from dilation import IMPLEMENTATIONS, PRESETS, binarize, make_kernel, to_gray


def added_pixels_overlay(binary: np.ndarray, dilated: np.ndarray) -> np.ndarray:
    out = np.zeros(binary.shape + (3,), np.uint8)
    out[binary > 0] = (255, 255, 255)
    out[(dilated > 0) & (binary == 0)] = (40, 40, 230)
    return out


def main():
    ap = argparse.ArgumentParser(description="Бинаризация и дилатация изображения примитивом 3x3 (native, numpy, opencv)")
    ap.add_argument("image", help="путь к входному изображению")
    ap.add_argument("--kernel", default="square",
                    help=f"примитив 3x3: {', '.join(PRESETS)} или 9 символов 0/1 (по строкам)")
    ap.add_argument("--iterations", type=int, default=1, help="число применений дилатации")
    ap.add_argument("--threshold", type=int, default=None, help="фиксированный порог (по умолчанию — Оцу)")
    ap.add_argument("--invert", action="store_true", help="объект — тёмные пиксели (тёмный текст на светлом)")
    ap.add_argument("--out", default="output", help="папка для результатов")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Не удалось прочитать {args.image}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    kernel = make_kernel(args.kernel)
    gray = to_gray(img)
    binary = binarize(gray, args.threshold, args.invert)

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.image))[0])
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "gray.png"), gray)
    cv2.imwrite(os.path.join(out_dir, "binary.png"), binary)

    print(f"Изображение: {args.image}  {gray.shape[1]}x{gray.shape[0]}")
    print(f"Примитив ({args.kernel}), итераций: {args.iterations}")
    print("\n".join("    " + " ".join("■" if v else "·" for v in row) for row in kernel))

    results = {}
    for name, func in IMPLEMENTATIONS.items():
        t0 = time.perf_counter()
        results[name] = func(binary, kernel, args.iterations)
        dt = time.perf_counter() - t0
        cv2.imwrite(os.path.join(out_dir, f"dilated_{name}.png"), results[name])
        print(f"  {name:6s}: {dt * 1e3:10.2f} мс")

    reference = results["opencv"]
    for name, res in results.items():
        if name != "opencv":
            diff = int(np.count_nonzero(res != reference))
            print(f"  {name} vs opencv: отличающихся пикселей — {diff}")

    cv2.imwrite(os.path.join(out_dir, "added_pixels.png"), added_pixels_overlay(binary, reference))
    before, after = np.count_nonzero(binary), np.count_nonzero(reference)
    print(f"Пикселей объекта: {before} -> {after} (+{after - before})")
    print(f"Результаты сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
