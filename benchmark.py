import argparse
import csv
import os
import platform
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from dilation import IMPLEMENTATIONS, make_kernel

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
IMG = os.path.join(DOCS, "img")

LABELS = {"native": "Python (нативно)", "numpy": "NumPy (векторизация)", "opencv": "OpenCV cv2.dilate"}
COLORS = {"native": "#d62728", "numpy": "#1f77b4", "opencv": "#2ca02c"}


def measure(func, *args, min_repeats=5, min_time=0.5, max_repeats=300):
    func(*args)
    times = []
    total = 0.0
    while len(times) < max_repeats and (len(times) < min_repeats or total < min_time):
        t0 = time.perf_counter()
        func(*args)
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
    return min(times)


def run_rounds(configs, rounds):
    best = [[float("inf")] * len(IMPLEMENTATIONS) for _ in configs]
    for r in range(rounds):
        for c, (label, img, kernel) in enumerate(configs):
            for i, func in enumerate(IMPLEMENTATIONS.values()):
                best[c][i] = min(best[c][i], measure(func, img, kernel))
            print(f"  раунд {r + 1}/{rounds}  {label:>12}: "
                  + "  ".join(f"{k}={t * 1e3:.3f} мс" for k, t in zip(IMPLEMENTATIONS, best[c])), flush=True)
    return best


def random_binary(n, density, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) < density).astype(np.uint8) * 255


def expected_checks(p, k=9):
    return sum((1 - p) ** m for m in range(k))


def save_csv(name, header, rows):
    with open(os.path.join(DOCS, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_csv(name, types):
    with open(os.path.join(DOCS, name), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [[t(v) for t, v in zip(types, row)] for row in reader]


def fmt_ms(sec):
    ms = sec * 1e3
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 1:
        return f"{ms:.2f}"
    return f"{ms:.3f}"


def system_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (
        f"CPU: {cpu}, логических ядер: {os.cpu_count()}, потоков OpenCV: {cv2.getNumThreads()}; "
        f"{platform.system()} {platform.release()}; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, OpenCV {cv2.__version__}"
    )


def exp_size(sizes, rounds):
    kernel = make_kernel("square")
    configs = [(f"N={n}", random_binary(n, 0.2, seed=n), kernel) for n in sizes]
    best = run_rounds(configs, rounds)
    rows = [[n] + b for n, b in zip(sizes, best)]
    save_csv("benchmark_size.csv", ["N"] + [f"{k}_sec" for k in IMPLEMENTATIONS], rows)
    return rows


def exp_density(n, densities, rounds):
    kernel = make_kernel("square")
    configs = [(f"p={p:.2f}", random_binary(n, p, seed=1), kernel) for p in densities]
    best = run_rounds(configs, rounds)
    rows = [[p] + b for p, b in zip(densities, best)]
    save_csv("benchmark_density.csv", ["density"] + [f"{k}_sec" for k in IMPLEMENTATIONS], rows)
    return rows


def exp_kernel(n, kernels, rounds):
    img = random_binary(n, 0.2, seed=2)
    configs = [(spec, img, make_kernel(spec)) for spec in kernels]
    best = run_rounds(configs, rounds)
    rows = [[spec, int(make_kernel(spec).sum())] + b for spec, b in zip(kernels, best)]
    save_csv("benchmark_kernel.csv", ["kernel", "elements"] + [f"{k}_sec" for k in IMPLEMENTATIONS], rows)
    return rows


def plot_size(rows):
    sizes = [r[0] for r in rows]
    pixels = [n * n for n in sizes]
    names = list(IMPLEMENTATIONS)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, name in enumerate(names):
        ax.loglog(pixels, [r[1 + i] * 1e3 for r in rows], "o-", color=COLORS[name], label=LABELS[name])
    ref = [rows[-1][1] * 1e3 * px / pixels[-1] for px in pixels]
    ax.loglog(pixels, ref, "--", color="gray", linewidth=1, label="∝ число пикселей (O(n))")
    ax.set_xticks(pixels)
    ax.set_xticklabels([f"{n}²" for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Размер изображения, пикселей")
    ax.set_ylabel("Время, мс (лог. шкала)")
    ax.set_title("Время дилатации от размера изображения (квадрат 3×3, p = 0.2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "bench_size.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    native = [r[1] for r in rows]
    for i, name in enumerate(names[1:], start=1):
        ax.semilogy(range(len(sizes)), [nt / r[1 + i] for nt, r in zip(native, rows)], "o-",
                    color=COLORS[name], label=f"{LABELS[name]} относительно Python")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{n}²" for n in sizes])
    ax.set_xlabel("Размер изображения, пикселей")
    ticks = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ticks if lo <= t <= hi])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Ускорение, раз (лог. шкала)")
    ax.set_title("Во сколько раз быстрее нативной Python-версии")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "bench_speedup.png"), dpi=130)
    plt.close(fig)


def plot_density(rows, n):
    ps = np.array([r[0] for r in rows])
    native = np.array([r[1] for r in rows])
    e = np.array([expected_checks(p) for p in ps])
    b, a = np.polyfit(e, native, 1)
    grid = np.linspace(0, 1, 101)
    model = a + b * np.array([expected_checks(p) for p in grid])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.plot(ps, native * 1e3, "o", color=COLORS["native"], label="измерение")
    ax.plot(grid, model * 1e3, "--", color="gray", label="модель a + b·E[проверок]")
    ax.set_xlabel("Плотность объекта p")
    ax.set_ylabel("Время, мс")
    ax.set_title(f"Python (нативно), {n}×{n}")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    for i, name in enumerate(list(IMPLEMENTATIONS)[1:], start=2):
        ax.plot(ps, [r[i] * 1e3 for r in rows], "o-", color=COLORS[name], label=LABELS[name])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Плотность объекта p")
    ax.set_ylabel("Время, мс")
    ax.set_title(f"NumPy и OpenCV, {n}×{n}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "bench_density.png"), dpi=130)
    plt.close(fig)
    return a, b


def write_markdown(size_rows, dens_rows, kern_rows, n, fit):
    names = list(IMPLEMENTATIONS)
    lines = [f"_Окружение: {system_info()}_", ""]

    lines += ["**Время от размера** (квадрат 3×3, плотность 0.2), мс:", ""]
    lines += ["| N×N | " + " | ".join(LABELS[k] for k in names) + " | ускорение OpenCV / Python |",
              "|---|" + "---:|" * (len(names) + 1)]
    for r in size_rows:
        lines.append(f"| {r[0]}² | " + " | ".join(fmt_ms(t) for t in r[1:]) + f" | ×{r[1] / r[3]:,.0f} |".replace(",", " "))

    lines += ["", f"**Время от плотности объекта** ({n}×{n}, квадрат 3×3), мс:", ""]
    lines += ["| p | E[проверок] | " + " | ".join(LABELS[k] for k in names) + " |",
              "|---:|---:|" + "---:|" * len(names)]
    for r in dens_rows:
        lines.append(f"| {r[0]:.2f} | {expected_checks(r[0]):.2f} | " + " | ".join(fmt_ms(t) for t in r[1:]) + " |")
    a, b = fit
    lines += ["", f"Модель для нативной версии: t ≈ {a * 1e3:.1f} мс + {b * 1e3:.1f} мс · E[проверок]", ""]

    lines += [f"**Время от формы примитива** ({n}×{n}, плотность 0.2), мс:", ""]
    lines += ["| примитив | элементов | " + " | ".join(LABELS[k] for k in names) + " |",
              "|---|---:|" + "---:|" * len(names)]
    for r in kern_rows:
        lines.append(f"| {r[0]} | {r[1]} | " + " | ".join(fmt_ms(t) for t in r[2:]) + " |")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(DOCS, "benchmark_results.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Сравнение быстродействия реализаций дилатации")
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n", type=int, default=512, help="размер для экспериментов 2 и 3")
    ap.add_argument("--rounds", type=int, default=3, help="число раундов замеров (берётся минимум)")
    ap.add_argument("--plot-only", action="store_true", help="только перерисовать графики по сохранённым CSV")
    ap.add_argument("--single-thread", action="store_true", help="запретить OpenCV многопоточность")
    args = ap.parse_args()

    if args.single_thread:
        cv2.setNumThreads(1)
    os.makedirs(IMG, exist_ok=True)
    print(system_info())

    if args.plot_only:
        size_rows = load_csv("benchmark_size.csv", [int] + [float] * 3)
        dens_rows = load_csv("benchmark_density.csv", [float] * 4)
        kern_rows = load_csv("benchmark_kernel.csv", [str, int] + [float] * 3)
    else:
        print("Эксперимент 1: размер изображения")
        size_rows = exp_size(args.sizes, args.rounds)
        print("Эксперимент 2: плотность объекта")
        dens_rows = exp_density(args.n, [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0], args.rounds)
        print("Эксперимент 3: форма примитива")
        kern_rows = exp_kernel(args.n, ["square", "cross", "hline"], args.rounds)

    plot_size(size_rows)
    fit = plot_density(dens_rows, args.n)
    print()
    print(write_markdown(size_rows, dens_rows, kern_rows, args.n, fit))


if __name__ == "__main__":
    main()
