import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dilation import IMPLEMENTATIONS, binarize, dilate_native, dilate_opencv, make_kernel

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(ROOT, "images")
OUT = os.path.join(ROOT, "docs", "img")

SAMPLES = [
    ("broken_text.png", True, "Разорванный текст"),
    ("shapes_holes.png", False, "Фигуры с дырами и трещинами"),
    ("coins.png", False, "Фотография монет (порог Оцу)"),
]


def overlay(binary, dilated):
    rgb = np.zeros(binary.shape + (3,), np.uint8)
    rgb[binary > 0] = (255, 255, 255)
    rgb[(dilated > 0) & (binary == 0)] = (230, 40, 40)
    return rgb


def show(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def load(name, invert):
    gray = cv2.imread(os.path.join(IMAGES, name), cv2.IMREAD_GRAYSCALE)
    return gray, binarize(gray, invert=invert)


def fig_samples():
    k = make_kernel("square")
    report = []
    for name, invert, caption in SAMPLES:
        gray, binary = load(name, invert)
        results = {impl: f(binary, k) for impl, f in IMPLEMENTATIONS.items()}
        same = all(np.array_equal(r, results["opencv"]) for r in results.values())
        dil = results["opencv"]

        h, w = gray.shape
        fig, axes = plt.subplots(2, 2, figsize=(11, 11 * h / w + 1.2))
        axes = axes.ravel()
        show(axes[0], gray, "Исходное (полутоновое)")
        show(axes[1], binary, "Бинаризация")
        show(axes[2], dil, "Дилатация, квадрат 3×3")
        axes[3].imshow(overlay(binary, dil), interpolation="nearest")
        axes[3].set_title("Добавленные пиксели (красным)", fontsize=10)
        axes[3].axis("off")
        fig.suptitle(caption, fontsize=13)
        fig.tight_layout()
        stem = os.path.splitext(name)[0]
        fig.savefig(os.path.join(OUT, f"sample_{stem}.png"), dpi=110)
        plt.close(fig)
        report.append((name, gray.shape, int(np.count_nonzero(binary)), int(np.count_nonzero(dil)), same))
    return report


def pixel_grid(ax, n):
    for t in np.arange(-0.5, n, 1):
        ax.axhline(t, color="#888", linewidth=0.8)
        ax.axvline(t, color="#888", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_iterations():
    _, binary = load("broken_text.png", True)
    k = make_kernel("square")
    fig, axes = plt.subplots(2, 2, figsize=(11, 4.6))
    axes = axes.ravel()
    show(axes[0], binary, "Бинарное изображение")
    for i, it in enumerate([1, 2, 3], start=1):
        show(axes[i], dilate_opencv(binary, k, it), f"Дилатация квадратом 3×3, итераций: {it}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "iterations.png"), dpi=110)
    plt.close(fig)


def fig_kernels():
    _, binary = load("broken_text.png", True)
    specs = ["square", "cross", "hline", "vline"]
    fig, axes = plt.subplots(len(specs), 2, figsize=(9, 2.5 * len(specs)),
                             gridspec_kw={"width_ratios": [1, 2.9]})
    for row, spec in enumerate(specs):
        k = make_kernel(spec)
        ax = axes[row, 0]
        ax.imshow(k * 255, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        pixel_grid(ax, 3)
        ax.set_title(f"{spec}: {int(k.sum())} эл.", fontsize=10)
        show(axes[row, 1], dilate_opencv(binary, k), f"Дилатация примитивом {spec}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kernels.png"), dpi=110)
    plt.close(fig)


def fig_reflection():
    k = make_kernel("100100111")
    a = np.zeros((9, 9), np.uint8)
    a[4, 4] = 255
    ours = dilate_native(a, k)
    raw = cv2.dilate(a, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    panels = [(k * 255, "Примитив B (центр — (1,1))"), (a, "A — одна точка"),
              (ours, "A ⊕ B (определение;\nнаша реализация)"), (raw, "cv2.dilate(A, B)\nбез отражения")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        pixel_grid(ax, img.shape[0])
        ax.set_title(title, fontsize=10)
    axes[0].add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor="#e03030", linewidth=2.5, zorder=5))
    for ax in axes[1:]:
        ax.add_patch(plt.Rectangle((3.5, 3.5), 1, 1, fill=False, edgecolor="#e03030", linewidth=2.5, zorder=5))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.82, wspace=0.08)
    fig.savefig(os.path.join(OUT, "reflection.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = fig_samples()
    fig_iterations()
    fig_kernels()
    fig_reflection()
    print("| изображение | размер | пикселей объекта до | после | реализации совпадают |")
    print("|---|---|---:|---:|---|")
    for name, shape, before, after, same in report:
        print(f"| {name} | {shape[1]}×{shape[0]} | {before} | {after} | {'да' if same else 'НЕТ'} |")
    print(f"Картинки сохранены в {OUT}")


if __name__ == "__main__":
    main()
