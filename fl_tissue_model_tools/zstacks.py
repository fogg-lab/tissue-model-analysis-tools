"""
Z-projection through focus stacking was adapted from the following source:
https://github.com/cmcguinness/focusstack

"""

import re
from string import digits
import os.path as osp
from glob import glob
import numpy.typing as npt
import numpy as np
import cv2


def simplify_zstack_ids(zstack_ids: list[str]):
    """Trim identifiers to remove common prefixes and suffixes.

    Args:
        zstack_ids: List of zstack ids.

    Returns:
        list[str]: List of zstack ids with common prefixes and suffixes removed.

    """
    while zstack_ids[0] and all(zid.startswith(zstack_ids[0][0]) for zid in zstack_ids):
        zstack_ids = [zid[1:] for zid in zstack_ids]
    while zstack_ids[0] and all(zid.endswith(zstack_ids[0][-1]) for zid in zstack_ids):
        zstack_ids = [zid[:-1] for zid in zstack_ids]
    return zstack_ids


def find_zstack_image_sequences(input_dir: str):
    """ Find zstacks from image sequences with numbered filenames.

    Args:
        input_dir: input directory

    Returns:
        dict[str, list[str]]: Dictionary of zstack ids and their file paths.

    """
    img_paths = list(filter(osp.isfile, glob(osp.join(input_dir, "*"))))
    if not img_paths:
        img_paths = list(filter(osp.isfile, glob(osp.join(input_dir, "*", "*"))))
    elif any(osp.isdir(fp) for fp in glob(osp.join(input_dir, "*"))):
        raise ValueError("Found both files and directories in input directory")

    # Get Z stack identifier for each Z slice and parse numbers present in file names
    zslice_stack_ids = []
    zslice_numbers_in_name = []
    for zslice_relpath in [osp.relpath(img_path, input_dir) for img_path in img_paths]:
        name = osp.basename(zslice_relpath)
        dir_name = osp.dirname(zslice_relpath)
        zstack_id = osp.join(dir_name, name.translate(str.maketrans("", "", digits)))
        zslice_stack_ids.append(zstack_id)
        zslice_numbers_in_name.append(list(map(int, re.findall(r"\d+", name)))[::-1])

    zslice_stack_ids = simplify_zstack_ids(zslice_stack_ids)

    # Group Z slices by Z stack identifier
    zstacks = {}
    unique_zslice_stack_ids = set(zslice_stack_ids)
    for zstack_id in unique_zslice_stack_ids:
        zstacks[zstack_id] = []
        zs_inds = [i for i, zid in enumerate(zslice_stack_ids) if zid == zstack_id]
        zs_nums_in_name = [zslice_numbers_in_name[i] for i in zs_inds]
        if not all([len(nums) == len(zs_nums_in_name[0]) for nums in zs_nums_in_name]):
            raise ValueError("Unrecognized Z slice naming convention")
        if len(set([tuple(nums) for nums in zs_nums_in_name])) != len(zs_inds):
            raise ValueError("Unrecognized Z slice numbering convention in image names")
        zs_nums = [nums + [i] for i, nums in zip(zs_inds, zs_nums_in_name)]
        for nums in sorted(zs_nums):
            index = nums[-1]
            zstacks[zstack_id].append(img_paths[index])

    return zstacks


def find_zstack_files(input_dir: str):
    """ Get paths to zstack files.

    Args:
        - input_dir: input directory

    Returns:
        - dict[str, str]: Dictionary of zstack ids and their file paths.

    """
    img_paths = list(filter(osp.isfile, glob(osp.join(input_dir, "*"))))
    if any(osp.isdir(fp) for fp in glob(osp.join(input_dir, "*"))):
        raise ValueError("Found both files and directories in input directory")
    zstack_ids = simplify_zstack_ids([osp.basename(img_path) for img_path in img_paths])
    return {zs_id: fp for zs_id, fp in zip(zstack_ids, img_paths)}


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
