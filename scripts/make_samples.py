import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "images")


def broken_text(rng):
    h, w = 220, 640
    img = np.full((h, w), 235, np.uint8)
    lines = ["Mathematical", "Morphology: A + B"]
    for k, line in enumerate(lines):
        cv2.putText(img, line, (25, 90 + k * 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 4, cv2.LINE_AA)
    holes = rng.random((h, w)) < 0.35
    img[(img < 128) & holes] = 235
    noise = rng.normal(0, 6, (h, w))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def shapes_holes(rng):
    h, w = 300, 420
    img = np.full((h, w), 30, np.uint8)
    cv2.circle(img, (90, 90), 60, 210, -1)
    cv2.rectangle(img, (190, 35), (330, 140), 210, -1)
    cv2.ellipse(img, (110, 220), (80, 45), 20, 0, 360, 210, -1)
    cv2.rectangle(img, (250, 180), (300, 270), 210, -1)
    cv2.rectangle(img, (303, 180), (380, 270), 210, -1)
    pepper = rng.random((h, w)) < 0.04
    img[(img > 128) & pepper] = 30
    cv2.line(img, (190, 90), (330, 95), 30, 1)
    cv2.line(img, (60, 40), (120, 140), 30, 1)
    noise = rng.normal(0, 5, (h, w))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)
    cv2.imwrite(os.path.join(OUT, "broken_text.png"), broken_text(rng))
    cv2.imwrite(os.path.join(OUT, "shapes_holes.png"), shapes_holes(rng))
    try:
        from skimage import data
        cv2.imwrite(os.path.join(OUT, "coins.png"), data.coins())
    except ImportError:
        print("scikit-image не установлен — coins.png пропущен")
    print("Готово:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
