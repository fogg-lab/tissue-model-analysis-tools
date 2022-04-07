"""
Z-projection through focus stacking was adapted from the following source:
https://github.com/cmcguinness/focusstack

"""

import cv2
import re
import numpy as np
from glob import glob
import numpy.typing as npt
from typing import Optional, Sequence, Union, Callable

from . import defs


_zpos_pattern = "(z|Z)[0-9]+_"


def _default_get_zpos(z_path: str) -> int:
    # Trim the 'Z' from the beginning of the match
    return int(re.search(_zpos_pattern, z_path)[0][1:-1])


def z_stack_from_dir(z_stack_dir: str, file_ext: str="tif", descending: bool=True, get_zpos: Optional[Callable[[str], int]]=None) -> tuple[Sequence[str], npt.NDArray]:
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
        file_ext: File extension of z-stack images.
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

    z_paths = [fn.replace("\\", "/") for fn in glob(f"{z_stack_dir}/*.{file_ext}")]
    if file_ext == "tif":
        flag = cv2.IMREAD_ANYDEPTH
    # TODO handle more filetypes
    else:
        flag = cv2.IMREAD_GRAYSCALE
    
    if get_zpos == None:
        get_zpos = _default_get_zpos
    
    sorted_z_paths = sorted(z_paths, key = lambda zp: get_zpos(zp), reverse = descending)
    return sorted_z_paths, np.array([cv2.imread(img_n, flag) for img_n in sorted_z_paths])


def proj_avg(stack: npt.NDArray, axis: Union[int, Sequence[int]]=0) -> npt.NDArray:
    return np.mean(stack, axis=axis)


def proj_max(stack: npt.NDArray, axis: Union[int, Sequence[int]]=0) -> npt.NDArray:
    return np.max(stack, axis=axis)


def proj_min(stack: npt.NDArray, axis: Union[int, Sequence[int]]=0) -> npt.NDArray:
    return np.min(stack, axis=axis)
