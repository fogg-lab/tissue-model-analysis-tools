import os.path as osp
import warnings
from glob import glob
from typing import List, Tuple, Optional, Union
from pathlib import Path
import sys

# Before importing aicsimageio, ignore warning about Java from `bfio.backends`
import logging

logging.getLogger("bfio.backends").setLevel(logging.ERROR)
from aicsimageio import AICSImage
from aicsimageio.dimensions import Dimensions
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.exceptions import UnsupportedFileFormatError
import numpy as np
from numpy.typing import NDArray

from fl_tissue_model_tools.defs import SUPPORTED_IMAGE_FORMATS


def load_image(
    file_path: Union[str, list[str]], T: Optional[int] = None, C: Optional[int] = None
) -> tuple[NDArray, PhysicalPixelSizes]:
    """Load ZYX or YX image from path using AICSImage.

    Args:
        file_path (Union[str, list[str]]): Path to image or paths for image sequence.
        T (int, optional): Index of the time to use (needed if time series).
        C (int, optional): Index of the color channel to use (needed if multi channel).

    Returns:
        np.ndarray: ZYX or YX image for 2D (if there is only 1 Z slice).
        PhysicalPixelSizes: Physical pixel sizes, most likely in microns. If unparsable,
                            returns `PhysicalPixelSizes(Z=None, Y=None, X=None)`.
    """

    if isinstance(file_path, list):
        # Z stack from image sequence
        images, pixel_sizes = zip(*[load_image(fp, T, C) for fp in file_path])
        pixel_sizes = pixel_sizes[0]
        image = np.array(images)
        return image, pixel_sizes

    try:
        img_reader = AICSImage(file_path)
    except UnsupportedFileFormatError as exc:
        print(
            f"\x1b[38;5;1m\x1b[1m[FAILURE]\x1b[0m Unsupported image format: {file_path}\n"
            f"Supported formats: {SUPPORTED_IMAGE_FORMATS}\n"
        )
        sys.exit(1)

    # AICSImage consistently reads images with the same order of dimensions:
    # Time-Channel-Z-Y-X
    if T is None:
        if img_reader.dims.T > 1:
            raise ValueError(
                f"{file_path} is a time series image "
                "but no time index was specified."
            )
        T = 0
    elif T >= img_reader.dims.T or T < 0:
        raise ValueError(
            f"Time {T} is out of range for {file_path} "
            f"with times: 0 - {img_reader.T - 1}"
        )

    if C is None:
        if img_reader.dims.C > 1:
            raise ValueError(
                f"{file_path} is a multi channel image "
                "but no color channel index was specified."
            )
        C = 0
    elif C >= img_reader.dims.C or C < 0:
        raise ValueError(
            f"Color channel {C} is out of range for {file_path} "
            f"with color channels: 0 - {img_reader.C - 1}"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not parse tiff pixel size",
            category=UserWarning,
        )
        pixel_sizes = img_reader.physical_pixel_sizes
    image = img_reader.get_image_data("ZYX", T=T, C=C)

    if len(image) == 1:
        return image[0], pixel_sizes

    return image, pixel_sizes


def get_image_dims(file_path: str) -> Dimensions:
    """Get dimensions of image (Time-Channel-Z-Y-X) from metadata.

    Args:
        file_path: Path to the image file.
    """

    try:
        img_reader = AICSImage(file_path)
    except UnsupportedFileFormatError as exc:
        raise UnsupportedFileFormatError(
            f"Unsupported image format: {file_path}"
            f"Supported formats: {SUPPORTED_IMAGE_FORMATS}"
        )

    return img_reader.dims


def get_img_mask_paths(
    img_dir: str,
    mask_dir=None,
    img_suffix_pattern=".tif",
    label_suffix_pattern="_mask.tif",
) -> Tuple[List[str], List[str]]:
    """Return list of image, label pairs.
    Args:
        img_dir (str): Path to directory containing images.
        mask_dir (str, optional): Path to directory containing masks. Defaults to None.
            If None, mask directory is assumed to be the same as the image directory.
            If images and masks share a directory, either images or masks need a distinct suffix.
        img_suffix_pattern (str, optional): Suffix pattern matching image files. Defaults to '.tif'.
        label_suffix_pattern (str, optional): Suffix pattern matching mask files. Defaults to '_mask.tif'.
    Returns:
        List[Tuple[str, str]]: List of image, mask pairs.
    Raises:
        ValueError: If directories and suffixes for images and masks are identical.
        ValueError: If number of found images and labels do not match.
        ValueError: If an image does not have a corresponding label.
    """

    if mask_dir is None:
        mask_dir = img_dir

    # make sure the search patterns are distinct
    same_dir = img_dir == mask_dir
    if same_dir and img_suffix_pattern == label_suffix_pattern:
        raise ValueError("directories and suffixes for images and labels are identical")
    exclude_mask_suffix_from_img_search = same_dir and label_suffix_pattern.endswith(
        img_suffix_pattern
    )
    exclude_img_suffix_from_mask_search = same_dir and img_suffix_pattern.endswith(
        label_suffix_pattern
    )

    # get image paths
    img_paths = glob(osp.join(img_dir, f"*{img_suffix_pattern}"))
    if exclude_mask_suffix_from_img_search:
        img_paths = [fp for fp in img_paths if not fp.endswith(label_suffix_pattern)]

    # get mask filenames
    mask_filenames = [
        Path(fp).name for fp in glob(osp.join(mask_dir, f"*{label_suffix_pattern}"))
    ]
    if exclude_img_suffix_from_mask_search:
        mask_filenames = [
            fn for fn in mask_filenames if not fn.endswith(img_suffix_pattern)
        ]

    # sort paths and make sure images and masks are paired 1:1
    if len(img_paths) != len(mask_filenames):
        raise ValueError(
            f"number of images ({len(img_paths)}) and labels ({len(mask_filenames)}) is different"
        )
    img_paths = sorted(img_paths)
    mask_paths = []
    for img_path in img_paths:
        sample_name = Path(img_path).name.replace(img_suffix_pattern, "")
        mask_fname = sample_name + label_suffix_pattern
        if mask_fname in mask_filenames:
            mask_paths.append(osp.join(mask_dir, mask_fname))
        else:
            raise ValueError(
                f"label {mask_fname} not found for image {Path(img_path).name}"
            )

    return [*zip(img_paths, mask_paths)]
