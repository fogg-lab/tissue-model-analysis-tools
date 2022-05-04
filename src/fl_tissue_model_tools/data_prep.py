import os
import shutil
import numpy as np
import dask as d
import cv2
from typing import Sequence
from copy import deepcopy

from tensorflow.keras import utils
from tensorflow.keras.applications import resnet50

from . import defs
from . import preprocessing as prep


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


class InvasionDataGenerator(utils.Sequence):
    def __init__(self, data_paths, class_labels, batch_size, img_shape, random_state, class_weights=None, shuffle=True, augmentation_function=None):
        self.data_paths = deepcopy(data_paths)
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.class_labels = deepcopy(class_labels)
        self.class_paths = {}
        self.class_counts = {}
        self.img_paths = []
        self.img_labels = []
        self.shuffle = shuffle
        self.rs = random_state
        self.augmentation_function = augmentation_function
        self._get_paths_and_counts(data_paths)
        self.indices = np.arange(len(self.img_paths), dtype=np.uint)
        if class_weights != None:
            self.class_weights = deepcopy(class_weights)
        else:
            self.class_weights = None
        self.shuffle_indices()

    def __len__(self):
        # return len()
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, index):        
        batch_idx_start = index * self.batch_size
        batch_idx_end = batch_idx_start + self.batch_size
        batch_indices = self.indices[batch_idx_start: batch_idx_end]

        img_paths = [self.img_paths[i] for i in batch_indices]
        # Should it be (B,) or (B,1)?
        y = np.array([self.img_labels[i] for i in batch_indices])

        # Generate data
        X = self.prep_images(img_paths)
        
        if self.augmentation_function != None:
            X = self.augmentation_function(X, self.rs, expand_dims=False)
        
        # Setting y to be (m,1) rather than (m,)
        if self.class_weights != None:
            # Weight classes by relative proportions in the training set
            w = np.array([self.class_weights[y_] for y_ in y])
            return X, y[:, np.newaxis], w

        return X, y[:, np.newaxis]

    
    def _get_paths_and_counts(self, data_paths):
        self.class_paths = deepcopy(data_paths)
        self.class_counts = {c: len(pn) for c, pn in self.class_paths.items()}
        for k, v in self.class_paths.items():
            # Paths to each image
            self.img_paths.extend(v)
            # Associate labels with each image path
            self.img_labels.extend(list(np.repeat(k, len(v))))
            
    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        img = prep.min_max_(cv2.resize(img, self.img_shape, cv2.INTER_LANCZOS4).astype(np.float32), defs.GS_MIN, defs.GS_MAX, defs.TIF_MIN, defs.TIF_MAX)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        return img
            
    def shuffle_indices(self):
        # print("shuffling")
        self.rs.shuffle(self.indices)
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_paths), dtype=np.uint)
        if self.shuffle == True:
            self.shuffle_indices()

    def prep_images(self, paths):
        imgs = np.array(d.compute((d.delayed(self._load_img)(p) for p in paths))[0])
        return resnet50.preprocess_input(imgs)
