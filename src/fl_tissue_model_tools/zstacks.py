"""
Z-projection through focus stacking was adapted from the following source:
https://github.com/cmcguinness/focusstack

"""

import re
from typing import Optional, Sequence, Callable, Tuple
import numbers
import dask as d
import numpy.typing as npt
import numpy as np
import cv2

from . import defs
from .helper import get_img_paths


ZPOS_PATTERN = "(z|Z)[0-9]+_"


def _default_get_zpos(z_path: str) -> int:
    """Use `ZPOS_PATTERN` to retrieve z-position from path string.

    Args:
        z_path: The full path or file name of a z-position image

    Returns:
        Image z-position as an integer.

    """
    # Trim the 'Z' from the beginning of the match
    return int(re.search(ZPOS_PATTERN, z_path)[0][1:-1])


def _blur_and_lap(image: npt.NDArray, kernel_size: int=5) -> npt.NDArray:
    """Compute Laplacian of a blurred image.

    Used to perform edge detection of `image`. A larger kernel size
    will contribute to a Laplacian with higher contrast between foreground
    and background, but less resolution within foreground objects.

    Args:
        image: Image on which Laplacian is to be computed.
        kernel_size: Kernel for both the Gaussian blur and the Laplacian.

    Returns:
        Laplacian of blurred image.
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)


def zstack_from_dir(z_stack_dir: str, descending: bool=True,
                    get_zpos: Optional[Callable[[str], int]]=None
                   ) -> Tuple[Sequence[str], npt.NDArray]:
    """Return sorted (by z-position) z-stack image paths and z-stack.

    IMPORTANT: To use the default `get_zpos` function, each z-position image
    in a z-stack directory must include the letter 'Z', followed by the
    z-position, followed by an underscore. That is, the file name must include:
    `Z[position]_`. So, for `position`=10, the file name would include: `Z10_`.

    The z-stack is a sequence of z-position images with a z-index associated
    with them. For example, a 16 image z-stack would have images for indices
    [0, 1, ..., 15]. The `z_stack_dir` should be laid out as follows
    (the file/directory names used below are simply examples):

    zstack_A/
    |--Z1_A.tif
    |--Z2_A.tif
    ...
    |--Z15_A.tif

    Args:
        z_stack_dir: Directory where z-stack images are located.
        descending: Whether z-position index is numbered from top to bottom
            or bottom to top. For example, descending means z-position 3 is
            located _above_ z-position 2.
        get_zpos: A function to sort the z-position images. Must take in a
            z-position image name and return that image's z-position. The
            z-position is used to sort the z-stack.

    Returns:
        A tuple containing (1) a list of the full paths to each z-poistion image
        in the z-stack (sorted by z-position) and (2) the z-stack (a 3-D numpy
        array) containing each z-position image.

    """

    # Get images in z-stack directory
    z_paths = get_img_paths(z_stack_dir)

    if get_zpos is None:
        get_zpos = _default_get_zpos

    # Sort z-stack images by z-position
    z_paths = sorted(z_paths, key = get_zpos, reverse = descending)

    return z_paths, np.array([cv2.imread(img, cv2.IMREAD_ANYDEPTH) for img in z_paths])


def zstack_paths_from_dir(z_stack_dir: str, descending: bool=True,
                        get_zpos: Optional[Callable[[str], int]]=None) -> Sequence[str]:
    """Get sorted z-stack image paths.

    Args:
        z_stack_dir: Directory where z-stack images are located.
        descending: Whether z-position index is numbered from top to bottom
            or bottom to top. For example, descending means z-position 3 is
            located _above_ z-position 2.
        get_zpos: A function to sort the z-position images. Must take in a
            z-position image name and return that image's z-position. The
            z-position is used to sort the z-stack.

    Returns:
        A list of the full paths to each z-poistion image
        in the z-stack (sorted by z-position)

    """
    z_paths = get_img_paths(z_stack_dir)
    if get_zpos is None:
        get_zpos = _default_get_zpos
    sorted_z_paths = sorted(z_paths, key = get_zpos, reverse = descending)
    return sorted_z_paths


def proj_focus_stacking(
    stack: npt.NDArray, axis: int=0, kernel_size:int=5
) -> npt.NDArray:
    """Project image stack along given axis using focus stacking.

    This procedure projects an image stack by retaining the maximum
    sharpness pixels.

    `stack` must be grayscale (8-bit), or will be converted.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        kernel_size: Kernel size to be passed to `_blur_and_lap`.

    Returns:
        Focus stack projection of image stack as grayscale (8-bit) image.

    """
    if stack.dtype != np.uint8:
        stack = stack.round().astype(np.uint8)

    # We do not perform the alignment step which is typically included,
    # since each image in the stack is assumed to be an in-focus cross-section.

    # Compute Laplacian for each slice in stack. This will give sharpness
    # measurement for each pixel in each slice.
    laps = np.array(
        d.compute([
            d.delayed(_blur_and_lap)(pos, kernel_size) for pos in stack
        ])[0]
    )
    output = np.zeros_like(stack[0])

    # For each pixel in output, assign to it the value in the stack with the
    # largest magnitude sharpness measurement at that position.
    abs_laps = np.absolute(laps)
    maxima = np.max(abs_laps, axis=axis)
    mask = (abs_laps == maxima).astype(np.uint8)
    for i in range(len(stack)):
        output = cv2.bitwise_not(stack[i], output, mask=mask[i])
    return defs.GS_MAX - output


def proj_avg(stack: npt.NDArray, axis: int=0, dtype=np.uint8) -> npt.NDArray:
    """Project image stack along given axis using average pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        dtype: The output datatype.

    Returns:
        Average projection of image stack.

    """
    proj = np.mean(stack, axis=axis)
    if issubclass(dtype, numbers.Integral):
        return proj.round().astype(dtype)
    return proj.astype(dtype)


def proj_med(stack: npt.NDArray, axis: int=0, dtype=np.uint8) -> npt.NDArray:
    """Project image stack along given axis using median pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        dtype: The output datatype.

    Returns:
        Median projection of image stack.

    """

    proj = np.median(stack, axis=axis)
    if issubclass(dtype, numbers.Integral):
        return proj.round().astype(dtype)
    return proj.astype(dtype)


def proj_max(stack: npt.NDArray, axis: int=0, dtype=np.uint8) -> npt.NDArray:
    """Project image stack along given axis using maximum pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        dtype: The output datatype.

    Returns:
        Maximum projection of image stack.

    """

    proj = np.max(stack, axis=axis)
    if issubclass(dtype, numbers.Integral):
        return proj.round().astype(dtype)
    return proj.astype(dtype)


def proj_min(stack: npt.NDArray, axis: int=0, dtype=np.uint8) -> npt.NDArray:
    """Project image stack along given axis using minimum pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        dtype: The output datatype.

    Returns:
        Minimum projection of image stack.

    """

    proj = np.min(stack, axis=axis)
    if issubclass(dtype, numbers.Integral):
        return proj.round().astype(dtype)
    return proj.astype(dtype)