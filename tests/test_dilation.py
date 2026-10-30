import itertools
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilation import PRESETS, dilate_native, dilate_numpy, dilate_opencv, make_kernel

IMPLS = [dilate_native, dilate_numpy, dilate_opencv]
ALL_KERNELS = [
    np.array(bits, dtype=np.uint8).reshape(3, 3)
    for bits in itertools.product([0, 1], repeat=9)
    if any(bits)
]


def random_binary(rng, h, w, density):
    return (rng.random((h, w)) < density).astype(np.uint8) * 255


@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=lambda k: "".join(map(str, k.ravel())))
def test_all_kernels_match_opencv(kernel):
    rng = np.random.default_rng(int("".join(map(str, kernel.ravel())), 2))
    img = random_binary(rng, 23, 31, 0.15)
    expected = dilate_opencv(img, kernel)
    assert np.array_equal(dilate_native(img, kernel), expected)
    assert np.array_equal(dilate_numpy(img, kernel), expected)


@pytest.mark.parametrize("seed", range(30))
def test_random_images_sizes_iterations(seed):
    rng = np.random.default_rng(seed)
    h, w = rng.integers(1, 60, size=2)
    img = random_binary(rng, h, w, rng.random())
    kernel = make_kernel(list(PRESETS)[seed % len(PRESETS)])
    iterations = int(rng.integers(1, 5))
    expected = dilate_opencv(img, kernel, iterations)
    assert np.array_equal(dilate_native(img, kernel, iterations), expected)
    assert np.array_equal(dilate_numpy(img, kernel, iterations), expected)


@pytest.mark.parametrize("impl", IMPLS)
def test_single_pixel_gives_copy_of_kernel(impl):
    kernel = make_kernel("110110000")
    img = np.zeros((7, 7), np.uint8)
    img[3, 3] = 255
    out = impl(img, kernel) // 255
    expected = np.zeros((7, 7), np.uint8)
    expected[2:5, 2:5] = kernel
    assert np.array_equal(out, expected)


def test_raw_opencv_does_not_reflect_kernel():
    kernel = make_kernel("110110000")
    img = np.zeros((7, 7), np.uint8)
    img[3, 3] = 255
    raw = cv2.dilate(img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) // 255
    assert not np.array_equal(raw, dilate_native(img, kernel) // 255)
    assert np.array_equal(raw, np.flip(dilate_native(img, kernel) // 255))


@pytest.mark.parametrize("impl", IMPLS)
def test_empty_and_full(impl):
    k = make_kernel("square")
    empty = np.zeros((10, 12), np.uint8)
    full = np.full((10, 12), 255, np.uint8)
    assert not impl(empty, k).any()
    assert (impl(full, k) == 255).all()


@pytest.mark.parametrize("impl", IMPLS)
def test_extensive_when_kernel_contains_origin(impl):
    rng = np.random.default_rng(1)
    img = random_binary(rng, 40, 40, 0.2)
    out = impl(img, make_kernel("cross"))
    assert ((img > 0) <= (out > 0)).all()


@pytest.mark.parametrize("impl", IMPLS)
def test_square_decomposes_into_lines(impl):
    rng = np.random.default_rng(2)
    img = random_binary(rng, 50, 45, 0.05)
    direct = impl(img, make_kernel("square"))
    two_step = impl(impl(img, make_kernel("hline")), make_kernel("vline"))
    assert np.array_equal(direct, two_step)


@pytest.mark.parametrize("impl", IMPLS)
def test_iterations_equal_repeated_application(impl):
    rng = np.random.default_rng(3)
    img = random_binary(rng, 30, 30, 0.03)
    k = make_kernel("square")
    assert np.array_equal(impl(img, k, 3), impl(impl(impl(img, k), k), k))


def test_nonbinary_input_is_treated_as_mask():
    k = make_kernel("square")
    img = np.zeros((5, 5), np.uint8)
    img[2, 2] = 7
    for impl in IMPLS:
        assert (impl(img, k)[1:4, 1:4] == 255).all()


def test_bad_kernels():
    with pytest.raises(ValueError):
        make_kernel("000000000")
    with pytest.raises(ValueError):
        make_kernel("11")
    with pytest.raises(ValueError):
        make_kernel("diamond")


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_iterations_of_square_equal_bigger_square(k):
    rng = np.random.default_rng(10 + k)
    img = random_binary(rng, 40, 50, 0.02)
    big = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    expected = cv2.dilate(img, big, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    assert np.array_equal(dilate_native(img, make_kernel("square"), k), expected)
