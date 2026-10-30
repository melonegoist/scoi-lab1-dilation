from .binarize import binarize, to_gray
from .kernels import PRESETS, make_kernel
from .native import dilate_native
from .numpy_impl import dilate_numpy
from .opencv_impl import dilate_opencv

IMPLEMENTATIONS = {
    "native": dilate_native,
    "numpy": dilate_numpy,
    "opencv": dilate_opencv,
}

__all__ = [
    "binarize", "to_gray", "make_kernel", "PRESETS",
    "dilate_native", "dilate_numpy", "dilate_opencv", "IMPLEMENTATIONS",
]
