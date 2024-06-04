import os.path as osp
from glob import glob
from typing import List, Tuple
from pathlib import Path
import imghdr

def get_img_paths(directory: str) -> List[str]:
    """Get all image paths in a directory.

    Args:
        directory: Path to directory containing images.

    Returns:
        A list of image paths.
    """
    unsupported_img_formats = {None, "rgb", "gif", "xbm"}
    # directory might be a prefix rather than a directory
    if not osp.isdir(directory):
        img_paths = [fp for fp in glob(f"{directory}*") if osp.isfile(fp)]
    else:
        img_paths = [fp for fp in glob(f"{directory}/*") if osp.isfile(fp)]
    img_paths = [fp for fp in img_paths if imghdr.what(fp) not in unsupported_img_formats]
    return img_paths


def get_img_mask_paths(
    img_dir: str, mask_dir=None,
    img_suffix_pattern='.tif', label_suffix_pattern='_mask.tif'
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
    same_dir = img_dir==mask_dir
    if same_dir and img_suffix_pattern==label_suffix_pattern:
        raise ValueError('directories and suffixes for images and labels are identical')
    exclude_mask_suffix_from_img_search = same_dir and label_suffix_pattern.endswith(img_suffix_pattern)
    exclude_img_suffix_from_mask_search = same_dir and img_suffix_pattern.endswith(label_suffix_pattern)

    # get image paths
    img_paths = glob(osp.join(img_dir, f'*{img_suffix_pattern}'))
    if exclude_mask_suffix_from_img_search:
        img_paths = [fp for fp in img_paths if not fp.endswith(label_suffix_pattern)]

    # get mask filenames
    mask_filenames = [Path(fp).name for fp in glob(osp.join(mask_dir, f'*{label_suffix_pattern}'))]
    if exclude_img_suffix_from_mask_search:
        mask_filenames = [fn for fn in mask_filenames if not fn.endswith(img_suffix_pattern)]

    # sort paths and make sure images and masks are paired 1:1
    if len(img_paths) != len(mask_filenames):
        raise ValueError(
            f'number of images ({len(img_paths)}) and labels ({len(mask_filenames)}) is different')
    img_paths = sorted(img_paths)
    mask_paths = []
    for img_path in img_paths:
        sample_name = Path(img_path).name.replace(img_suffix_pattern, '')
        mask_fname = sample_name + label_suffix_pattern
        if mask_fname in mask_filenames:
            mask_paths.append(osp.join(mask_dir, mask_fname))
        else:
            raise ValueError(f'label {mask_fname} not found for image {Path(img_path).name}')

    return [*zip(img_paths, mask_paths)]
