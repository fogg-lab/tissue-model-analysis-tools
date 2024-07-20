from typing import Sequence, Callable, Union, Tuple, Dict, Optional
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import dask as d
import cv2
from skimage.exposure import rescale_intensity
from numpy.random import RandomState
from fl_tissue_model_tools import helper
import silence_tensorflow.auto  # noqa
from tensorflow.keras import utils
from tensorflow.keras.applications import resnet50

from . import preprocessing as prep


def load_inv_depth_img(
    image: Union[str, npt.NDArray],
    img_hw: Tuple[int, int],
    T: Optional[int] = None,
    C: Optional[int] = None,
) -> npt.NDArray:
    """Load an invasion depth image and convert it to grayscale with 3 redundant channels.

    Args:
        image: Image or path to image.
        img_hw: Desired height and width for image to be resized to.
        T (int, optional): Index of the time to use (needed if time series).
        C (int, optional): Index of the color channel to use (needed if multi channel).

    Returns:
        Preprocessed invasion depth image.
    """
    img = helper.load_image(image, T, C)[0] if isinstance(image, str) else image
    img = cv2.resize(img, img_hw, cv2.INTER_LANCZOS4)
    img = rescale_intensity(img, out_range=(0, 255))
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img


def prep_inv_depth_imgs(
    images: Union[Sequence[str], Sequence[npt.NDArray]],
    img_hw: Tuple[int, int],
    T: Optional[int] = None,
    C: Optional[int] = None,
) -> npt.NDArray:
    """Prepare a batch of invasion depth images.

    Args:
        images: Paths to each image in batch or an already-loaded batch of images.
        img_hw: Desired height and width for each image to be resized to.
        T (int, optional): Index of the time to use (needed if time series).
        C (int, optional): Index of the color channel to use (needed if multi channel).

    Returns:
        Preprocessed invasion depth images.
    """
    imgs = np.array(
        d.compute((d.delayed(load_inv_depth_img)(im, img_hw, T, C) for im in images))[0]
    )
    return resnet50.preprocess_input(imgs)


def get_train_val_split(
    tv_class_paths: Dict[int, Sequence[str]], val_split: float = 0.2
) -> Tuple[Dict[int, Sequence[str]], Dict[int, Sequence[str]]]:
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
    train_data_paths = {k: v[val_counts[k] :] for k, v in tv_class_paths.items()}
    val_data_paths = {k: v[: val_counts[k]] for k, v in tv_class_paths.items()}
    return train_data_paths, val_data_paths


class InvasionDataGenerator(utils.Sequence):
    """Sequence class for handling invasion depth images."""

    def __init__(
        self,
        class_paths: Sequence[str],
        class_labels: Dict[str, int],
        batch_size: int,
        img_shape: Tuple[int, int],
        random_state: RandomState,
        class_weights: Union[Dict[int, float], bool] = False,
        shuffle: bool = True,
        augmentation_function: Callable = None,
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
        elif class_weights:
            self.class_weights = prep.balanced_class_weights_from_counts(
                self.class_counts
            )
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
        Tuple[npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    ]:
        """Retrieve a mini-batch of images, labels, and (optionally) sample weights.

        Args:
            index: Mini-batch index.

        Returns:
            Either (images, labels) or (images, labels, sample weights).
        """
        batch_idx_start = index * self.batch_size
        batch_idx_end = batch_idx_start + self.batch_size
        batch_indices = self.indices[batch_idx_start:batch_idx_end]

        img_paths = [self.img_paths[i] for i in batch_indices]
        img_labels = np.array([self.img_labels[i] for i in batch_indices])

        # Generate data
        preprocessed_imgs = prep_inv_depth_imgs(img_paths, self.img_shape)

        if self.augmentation_function is not None:
            preprocessed_imgs = self.augmentation_function(
                preprocessed_imgs, self.rand_state, expand_dims=False
            )

        # Set img_labels to be (m,1) rather than (m,)
        if self.class_weights is not None:
            # Weight classes by relative proportions in the training set
            weights = np.array([self.class_weights[y_] for y_ in img_labels])
            # Set y to be (m,1) rather than (m,)
            return preprocessed_imgs, img_labels[:, np.newaxis], weights

        # Set y to be (m,1) rather than (m,)
        return preprocessed_imgs, img_labels[:, np.newaxis]

    def _get_class_counts_and_create_master_image_and_label_lists(self):
        """Get class counts and create full lists of image paths and labels.

        The ith image path will correspond to the ith image label.
        """
        self.class_counts = {c: len(pn) for c, pn in self.class_paths.items()}
        for key, path in self.class_paths.items():
            # Paths to each image
            self.img_paths.extend(path)
            # Associate labels with each image path
            self.img_labels.extend(list(np.repeat(key, len(path))))

    def shuffle_indices(self) -> None:
        """Shuffle indices used to select image paths for loading into batch."""
        self.rand_state.shuffle(self.indices)

    def on_epoch_end(self) -> None:
        """Perform designated actions at the end of each training epoch."""
        self.indices = np.arange(len(self.img_paths), dtype=np.uint)
        if self.shuffle:
            self.shuffle_indices()
