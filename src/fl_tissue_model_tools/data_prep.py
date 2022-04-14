import os
import shutil
import numpy as np
import dask as d
from typing import Sequence


def make_dir(path: str) -> None:
    """Create `path` and all intermediate directories.

    If the provided path is `a/b/c`, all subdirectories or
    files in `c` will be deleted and `c` will be remade.

    Args:
        path: Full path to desired directory to be created.

    Returns:
        None

    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def save_class_imgs(img_paths: Sequence[str], split_list: Sequence[int], split_map: dict[int, str], img_class: str, dset_path: str) -> None:
    """Save images for a particular class according to train/test split.

    Args:
        img_paths: All image names (full path).
        split_list: Array of split labels, for example
            [0, 0, 1, 0, 1, ..., 1, 0]
        split_map: Definition of `split_list` meanings (e.g.,
            0: train, 1: test)
        img_class: The class of the images to be saved. This will
            be used according to the deep learning image classification
            convention of using the last directory level as class label.
            For example, for "Dog", "Cat", and "Bird", classification, dog
            images would be stored as

            Data/
            |--Dog/
            |  |--(dog image files)
        dset_path: Root path for output labels. For classes `a`, `b`, and `c`
            at `dir/subdir/<a,b,c>`, this parameter would be `dir/subdir`.

    Returns:
        None

    """
    for i, img_p, in enumerate(img_paths):
        img_n = img_p.split("/")[-1]
        shutil.copy(img_p, f"{dset_path}/{split_map[split_list[i]]}/{img_class}/{img_n}")
