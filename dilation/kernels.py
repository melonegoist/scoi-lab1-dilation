import numpy as np

PRESETS = {
    "square": "111" "111" "111",
    "cross": "010" "111" "010",
    "hline": "000" "111" "000",
    "vline": "010" "010" "010",
}


def make_kernel(spec: str = "square") -> np.ndarray:
    mask = PRESETS.get(spec, spec)
    if len(mask) != 9 or set(mask) - {"0", "1"}:
        raise ValueError(
            f"Ядро должно быть одним из {list(PRESETS)} или строкой из 9 символов 0/1, получено: {spec!r}"
        )
    if "1" not in mask:
        raise ValueError("Примитив должен содержать хотя бы один элемент")
    return np.array([int(c) for c in mask], dtype=np.uint8).reshape(3, 3)
