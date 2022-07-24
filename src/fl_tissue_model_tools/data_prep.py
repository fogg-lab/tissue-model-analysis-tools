import os
import shutil
from typing import Sequence, Callable, Union, Tuple, Dict
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import dask as d
import cv2

from numpy.random import RandomState

from tensorflow.keras import utils
from tensorflow.keras.applications import resnet50

from . import defs
from . import preprocessing as prep


AugmentationFunction = Callable[[npt.NDArray, RandomState], npt.NDArray]


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


def save_class_imgs(
    img_paths: Sequence[str], split_list: Sequence[int],
    split_map: Dict[int, str], img_class: str, dset_path: str
) -> None:
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


def load_inv_depth_img(path: str, img_hw: Tuple[int, int]) -> npt.NDArray:
    """Load an invasion depth image and convert it to grayscale with 3 redundant channels.

    Args:
        path: Path to image. Assumed to be .tif.
        img_hw: Desired height and width for image to be resized to.

    Returns:
        Preprocessed invasion depth image.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    img = prep.min_max_(cv2.resize(img, img_hw, cv2.INTER_LANCZOS4).astype(np.float32),
                        defs.GS_MIN, defs.GS_MAX, defs.TIF_MIN, defs.TIF_MAX)
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img


def prep_inv_depth_imgs(paths: Sequence[str], img_hw: Tuple[int, int]) -> npt.NDArray:
    """Prepare a batch of invasion depth images.

    Args:
        paths: Paths to each image in batch.
        img_hw: Desired height and width for each image to be resized to.

    Returns:
        Preprocessed invasion depth images.
    """
    imgs = np.array(
        d.compute((d.delayed(load_inv_depth_img)(p, img_hw) for p in paths))[0]
    )
    return resnet50.preprocess_input(imgs)


def get_train_val_split(
    tv_class_paths: Dict[int, Sequence[str]], val_split: float=0.2
) -> Tuple[
    Dict[int, Sequence[str]], Dict[int, Sequence[str]]
]:
    """Generate a train/validation split using mapping from labels to image paths.

    Args:
        tv_class_paths: Map of class labels (integer 0...k) to lists of full paths
            for images within each class. This mapping should include all image paths
            desired for inclusion in the training & validation sets.
        val_split: Proportion of data to be used in the validation set. As dataset
            size increases, this number may be smaller.

    Returns:
        Two mappings: (labels -> train paths, labels -> val paths)

    """
    tv_counts = {k: len(v) for k, v in tv_class_paths.items()}
    val_counts = {k: round(v * val_split) for k, v in tv_counts.items()}
    train_data_paths = {k: v[val_counts[k]:] for k, v in tv_class_paths.items()}
    val_data_paths = {k: v[:val_counts[k]] for k, v in tv_class_paths.items()}
    return train_data_paths, val_data_paths


class InvasionDataGenerator(utils.Sequence):
    """Sequence class for handling invasion depth images."""

    def __init__(
        self, class_paths: Sequence[str], class_labels: Dict[str, int],
        batch_size: int, img_shape: Tuple[int, int], random_state: RandomState,
        class_weights: Union[Dict[int, float], bool]=False, shuffle: bool=True,
        augmentation_function: AugmentationFunction=None
    ):
        """Create sequence class for handling invasion depth images.

        Args:
            class_paths: Map of class labels (integer 0...k) to lists of full paths for
                images within each class.
            class_labels (Dict[str, int]): Mapping from class name (directory name) to
                number representing class.
            batch_size: Batch size for training step.
            img_shape: Desired (H, W) of images.
            random_state: RandomState object used to allow for reproducability. Seed of
                RandomState object can be set to `None`.
            class_weights: Either a map from class label (integer 0...k) to weights for
                each class, or a boolean specifying whether to compute balanced weights.
                Used to generate sample weights during training. Useful for imbalanced data.
            shuffle: Whether to shuffle data upon generator creation and after each epoch.
            augmentation_function: Function that can be used to augment the image data.
                This function must take the specified arguments and an `expand_dims` argument
                that allows or prevents images from being given an extra depth axis.
        """

        self.class_paths = deepcopy(class_paths)
        self.class_labels = deepcopy(class_labels)
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.rand_state = random_state
        self.shuffle = shuffle
        self.augmentation_function = augmentation_function

        self.class_counts = {}
        self.img_paths = []
        self.img_labels = []

        self._get_class_counts_and_create_master_image_and_label_lists()
        self.indices = np.arange(len(self.img_paths), dtype=np.uint)
        if isinstance(class_weights, Dict):
            self.class_weights = deepcopy(class_weights)
        elif class_weights == True:
            self.class_weights = prep.balanced_class_weights_from_counts(self.class_counts)
        else:
            self.class_weights = None

        if self.shuffle:
            self.shuffle_indices()

    def __len__(self) -> int:
        """Get length of data (number of batches) 

        Returns:
            Number of batches.
        """
        return len(self.img_paths) // self.batch_size

    def __getitem__(
        self, index
    ) -> Union[
        Tuple[npt.NDArray, npt.NDArray],
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    ]:
        """Retrieve a mini-batch of images, labels, and (optionally) sample weights.

        Args:
            index: Mini-batch index.

        Returns:
            Either (images, labels) or (images, labels, sample weights).
        """
        batch_idx_start = index * self.batch_size
        batch_idx_end = batch_idx_start + self.batch_size
        batch_indices = self.indices[batch_idx_start: batch_idx_end]

        img_paths = [self.img_paths[i] for i in batch_indices]
        y = np.array([self.img_labels[i] for i in batch_indices])

        # Generate data
        X = prep_inv_depth_imgs(img_paths, self.img_shape)

        if self.augmentation_function != None:
            X = self.augmentation_function(X, self.rand_state, expand_dims=False)
        
        # Set y to be (m,1) rather than (m,)
        if self.class_weights != None:
            # Weight classes by relative proportions in the training set
            w = np.array([self.class_weights[y_] for y_ in y])
            # Set y to be (m,1) rather than (m,)
            return X, y[:, np.newaxis], w

        # Set y to be (m,1) rather than (m,)
        return X, y[:, np.newaxis]

    def _get_class_counts_and_create_master_image_and_label_lists(self):
        """Get class counts and create full lists of image paths and labels.

        The ith image path will correspond to the ith image label.
        """
        self.class_counts = {c: len(pn) for c, pn in self.class_paths.items()}
        for k, v in self.class_paths.items():
            # Paths to each image
            self.img_paths.extend(v)
            # Associate labels with each image path
            self.img_labels.extend(list(np.repeat(k, len(v))))

    def shuffle_indices(self) -> None:
        """Shuffle indices used to select image paths for loading into batch."""
        self.rand_state.shuffle(self.indices)

    def on_epoch_end(self) -> None:
        """Perform designated actions at the end of each training epoch."""
        self.indices = np.arange(len(self.img_paths), dtype=np.uint)
        if self.shuffle:
            self.shuffle_indices()
