"""
Z-projection through focus stacking was adapted from the following source:
https://github.com/cmcguinness/focusstack

"""

import re
import os
import os.path as osp
from pathlib import Path
from typing import Optional, Sequence, Callable
from glob import glob
import numbers
import numpy.typing as npt
import numpy as np
from skimage.exposure import rescale_intensity
import cv2
import tifffile
import imghdr
from pyometiff import OMETIFFReader

from . import defs
from .helper import get_img_paths

# Zpos pattern: flexible, matches substrings like z012., Z34_, z[567]-
ZPOS_PATTERN = "(z|Z)(\[)?[0-9]+(\])?(_|\.|\-)"
TIFF_INTERIM_DIR_SUFFIX = "_TMP_tiff_files"


def default_get_zpos(z_path: str) -> int:
    """Use `ZPOS_PATTERN` to retrieve z-position from path string.

    Args:
        z_path: The full path or file name of a z-position image

    Returns:
        Image z-position as an integer.

    """
    # Trim the 'Z' from the beginning of the match
    return int(re.search(ZPOS_PATTERN, z_path)[0][1:-1])


def _blur_and_lap(image: npt.NDArray, kernel_size: int = 5) -> npt.NDArray:
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


def is_single_file_zstack(fp: str) -> bool:
    """Check whether given file path `fp` is an OME file."""
    basename = osp.basename(fp)
    if not osp.isfile(fp):
        # directory or wrong file path
        return False
    if not any(ext == "ome" for ext in basename.lower().split(".")[1:]):
        if imghdr.what(fp) != "tiff":
            # Not a TIFF file of any kind
            return False
    if re.search(ZPOS_PATTERN, basename) is not None:
        # Looks like a single slice within a z-stack, not a whole z-stack
        return False
    try:
        OMETIFFReader(fp).read()
    except Exception:
        return False
    return True


def is_zstack(fp):
    """Check whether file path given file path `fp` contains a z-stack."""
    return (osp.isdir(fp) and len(glob(f"{fp}/*")) > 0) or is_single_file_zstack(fp)


def save_tiff_zstack(image_stack: npt.NDArray, output_dir: str) -> None:
    """Save a stack of images as individual TIFF files.

    Args:
        image_stack: A numpy array representing a stack of images.
        output_dir: Directory to save the TIFF files.
    """
    for i, image in enumerate(image_stack):
        print(image.shape)
        output_path = osp.join(output_dir, f"Z{str(i+1).zfill(5)}.tiff")
        tifffile.imsave(output_path, image)


def convert_zstack_image_to_tiffs(img_path: str) -> str:
    """Convert a z-stack image file in OME-TIFF format to individual TIFF files.

    Args:
        img_path: Path to the input z-stack image file.

    Returns:
        The directory where the TIFF files are saved.
    """
    output_dir_name = Path(img_path).stem + TIFF_INTERIM_DIR_SUFFIX
    # Clean up output directory name, in case it contains spaces or periods
    output_dir_name = output_dir_name.replace(" ", "_")
    output_dir_name = output_dir_name.replace(".", "_")
    output_dir = osp.join(osp.dirname(img_path), output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    def print_failure_message(e: Exception) -> None:
        print(f"Failed to read {img_path}. Error:\n{e}")

    try:
        with tifffile.TiffFile(img_path) as tif:
            images = tif.asarray()
    except Exception as e:
        print_failure_message(e)

    if images.ndim > 3:
        # Move z dimension to the first axis
        images = np.moveaxis(images, -3, 0)
        # Move channel dimension to the last axis
        images = np.moveaxis(images, 1, -1)
        # Squeeze: Remove axes of length one
        images = np.squeeze(images)

    save_tiff_zstack(images, output_dir)
    return output_dir


def zstack_paths_from_dir(
    z_stack_dir: str,
    descending: bool = True,
    get_zpos: Optional[Callable[[str], int]] = None,
) -> Sequence[str]:
    """Get sorted z-stack image paths.

    Args:
        z_stack_dir: Directory containing z-stack images.
        descending: Whether z-position index is numbered from top to bottom
            or bottom to top. For example, descending means z-position 3 is
            located _above_ z-position 2.
        get_zpos: A function to sort the z-position images. Must take in a
            z-position image name and return that image's z-position. The
            z-position is used to sort the z-stack.

    Returns:
        A list of the full paths to each z-position image
        in the z-stack (sorted by z-position)

    """
    z_paths = get_img_paths(z_stack_dir)
    if get_zpos is None:
        get_zpos = default_get_zpos
    sorted_z_paths = sorted(z_paths, key=get_zpos, reverse=descending)
    return sorted_z_paths


def proj_focus_stacking(
    stack: npt.NDArray, axis: int = 0, kernel_size: int = 5
) -> npt.NDArray:
    """Project image stack along given axis using focus stacking.

    This procedure projects an image stack by retaining the maximum
    sharpness pixels.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        kernel_size: Kernel size to be passed to `_blur_and_lap`.

    Returns:
        Focus stack projection of image stack as grayscale (8-bit) image.

    """

    # We do not perform the alignment step which is typically included,
    # since each image in the stack is assumed to be an in-focus cross-section.

    # Compute Laplacian for each slice in stack to measure the sharpness of each pixel.
    # Assign each output pixel with the value in the stack with the largest magnitude sharpness.

    if axis != 0:
        stack = np.moveaxis(stack, axis, 0)

    maxima = np.full_like(stack[0], -np.inf, dtype=np.float32)
    zproj = stack[0].copy()

    for pos in stack:
        abs_laplacian = np.absolute(_blur_and_lap(pos, kernel_size))
        maxima_mask = abs_laplacian > maxima
        maxima[maxima_mask] = abs_laplacian[maxima_mask]
        zproj[maxima_mask] = pos[maxima_mask]

    return zproj


def proj_avg(stack: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """Project image stack along given axis using average pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)

    Returns:
        Average projection of image stack.

    """
    return np.mean(stack, axis=axis)


def proj_med(stack: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """Project image stack along given axis using median pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)

    Returns:
        Median projection of image stack.

    """

    return np.median(stack, axis=axis)


def proj_max(stack: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """Project image stack along given axis using maximum pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)
        dtype: The output datatype.

    Returns:
        Maximum projection of image stack.

    """

    return np.max(stack, axis=axis)


def proj_min(stack: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """Project image stack along given axis using minimum pixel intensity.

    Args:
        stack: Image stack.
        axis: The axis to project along (defaults to z)

    Returns:
        Minimum projection of image stack.

    """

    return np.min(stack, axis=axis)