import json
from pathlib import Path
from typing import Sequence, Tuple, Union
from numbers import Number
import os
from skimage.exposure import rescale_intensity

import numpy as np
import cv2
import dask as d

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras import Model, optimizers
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from fl_tissue_model_tools import defs


def mean_iou_coef(y: tf.Tensor, yhat: tf.Tensor, smooth: float=1.0,
                  obs_axes: Tuple[int, ...]=(1, 2, 3), thresh: float=0.5):
    """Compute mean intersection over union coefficient.

    Args:
        y: True labels.
        yhat: Predicted labels.
        smooth: Accounts for case of zero union.
        obs_axes: Axes to collapse when computing IoU.
        thresh: Threshold that decides classification. Value of > 0.5
            will yield a more "picky" classification.
    Returns:
        mean IoU coefficient for batch.

    """
    y = K.cast(y, "float32")
    yhat = K.cast(yhat, "float32")
    # Set yhat > thresh to 1, 0 otherwise.
    yhat = K.cast(K.cast(K.greater(K.clip(yhat, 0, 1), thresh), "uint32"), "float32")
    intersection = K.sum(y * yhat, axis=obs_axes)
    union = K.sum(y, axis=obs_axes) + K.sum(yhat, axis=obs_axes) - intersection
    mean_iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return mean_iou


def mean_iou_coef_factory(smooth: int=1, obs_axes: Tuple[int, ...]=(1, 2, 3),
                          thresh: float=0.5):
    """Returns a an instance of the `iou_coef` function with extra parameters filled in.

    Allows user to use `Keras` metrics tracking without defining a lambda function.
    Returns an instance of the `iou_coef` function with the parameters below filled
    in with the values desired by the user.

    Args:
        smooth: Accounts for case of zero union.
        obs_axes: Axes to collapse when computing IoU.
        thresh: Threshold that decides classification. Value of > 0.5
            will yield a more "picky" classification.
    """

    def fn(y, yhat):
        return mean_iou_coef(y, yhat, smooth=smooth, obs_axes=obs_axes,
                             thresh=thresh)
    fn.__name__ = "mean_iou_coef"
    return fn


def save_unet_patch_segmentor_cfg(cfg: dict):
    """Save a UNetXceptionPatchSegmentor config to a json file.
    Args:
        cfg: Config dictionary.

    """
    save_dir = defs.MODEL_TRAINING_DIR / "binary_segmentation" / "configs"

    required_keys = ["patch_size", "checkpoint_file", "filter_counts"]
    optional_keys = ["ds_ratio", "norm_mean", "norm_std", "channels"]

    for key in required_keys:
        if cfg.get(key) is None:
            raise ValueError(f"Missing required config parameter: {key}")

    for key in cfg.keys():
        if key not in required_keys and key not in optional_keys:
            raise ValueError(f"Invalid config parameter: {key}")

    exp_num = get_last_exp_num() + 1

    save_path = save_dir / f"unet_patch_segmentor_{exp_num}.json"

    with open(save_path, "w") as fp:
        json.dump(cfg, fp, indent=4)


def get_last_exp_num() -> int:
    """Get the last experiment number for a UNetXceptionPatchSegmentor config.

    Returns:
        The last experiment number.

    """
    save_dir = defs.MODEL_TRAINING_DIR / "binary_segmentation" / "configs"

    exp_num = 0
    for file in save_dir.glob("*.json"):
        fname = file.name
        if fname.startswith("unet_patch_segmentor_"):
            exp_num = max(exp_num, int(fname.split("_")[-1].split(".")[0]))

    return exp_num


class WarmupSchedule(LearningRateSchedule):
    """Linear warmup before either settling at a constant LR or starting a different schedule."""
    def __init__(self, warmup_steps: int, after_warmup_lr: Union[dict, float]):
        """Create linear warmup schedule.

        Args:
            warmup_steps (int): Number of warmup steps.
            after_warmup_lr (Union[dict, str, float]):
                LR after warmup. Can be a config for a LearningRateSchedule or a constant.
        """

        self.warmup_steps = int(warmup_steps)

        if isinstance(after_warmup_lr, str):
            try:
                after_warmup_lr = float(after_warmup_lr)
            except:
                after_warmup_lr = json.loads(after_warmup_lr)
        if isinstance(after_warmup_lr, Number):
            after_warmup_lr = float(after_warmup_lr)
            self.after_warmup_lr_config = str(after_warmup_lr)
            self.after_warmup_lr = lambda step: after_warmup_lr
            self.after_warmup_init_lr = tf.cast(after_warmup_lr, tf.float32)
        elif isinstance(after_warmup_lr, dict):
            self.after_warmup_lr_config = json.dumps(after_warmup_lr)
            self.after_warmup_lr = optimizers.schedules.deserialize(after_warmup_lr)
            self.after_warmup_init_lr = tf.cast(self.after_warmup_lr(0), tf.float32)
        else:
            raise TypeError("after_warmup_lr must be a float or a serialized LearningRateSchedule"
                            f" (a dict or JSON-serialized string). Got: {type(after_warmup_lr)}")

    def __call__(self, step):
        """Compute LR at a given training step."""
        return self.get_lr(tf.cast(step, tf.float32))

    def get_lr(self, step: tf.float32) -> tf.float32:
        def get_lr_before_warmup():
            return tf.cast(self.after_warmup_init_lr * (step+1) / self.warmup_steps, tf.float32)
        def get_lr_after_warmup():
            return tf.cast(self.after_warmup_lr((step+1) - self.warmup_steps), tf.float32)
        lr = tf.cond(tf.less(step, self.warmup_steps), get_lr_before_warmup, get_lr_after_warmup)
        return lr

    def get_config(self):
        """Get config for serialization."""
        return {
            "warmup_steps": str(self.warmup_steps),
            "after_warmup_lr": self.after_warmup_lr_config
        }


def toggle_TL_freeze(tl_model: Model, base_model_name: str="base_model") -> None:
    """Toggle the `trainable` property of a transfer learning model.

    IMPORTANT: `tl_model` must be recompiled after changing this attribute.

    Args:
        tl_model: The transfer learning model.
        base_model_name: The name of the base model that will have its
            `trainable` property toggled.

    """
    base_model = tl_model.get_layer(base_model_name)
    base_model.trainable = not base_model.trainable


def check_consec_factor(x: Sequence[float], factor: float, reverse: bool=False) -> bool:
    """Check that consecutive elements of `x` increase by a constant factor.

    Args:
        x: An ordered sequence of numbers.
        factor: The factor by which elements of `x` should
            change consecutively.
        reverse: Whether to reverse `x` when assessing consecutive
            factor change.

    Returns:
        Whether elements of `x` change consecutively by a factor of
            `factor`.
    """

    res = True
    if reverse:
        x = list(reversed(x))
    for i in range(1, len(x)):
        res = res and (x[i] == x[i - 1] * factor)
    return res


def load_y(batch_mask_paths):
    # Load the binary segmentation masks
    y = np.array([cv2.imread(mask_path, 0) for mask_path in batch_mask_paths])
    y[y>0] = 1
    return y


def load_x(batch_img_paths):
    # Load the input images
    x = [cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) for img_path in batch_img_paths]
    return np.array(x)


class BinaryMaskSequence(KerasSequence):
    """Helper to iterate over the data"""

    def __init__(self, batch_size, img_paths, seg_paths, random_state,
                load_x, load_y, augmentation_function=None, sample_weights=None,
                repeat_n_times=1, shuffle=True):
        """Initialize the data loader

        Args:
            batch_size (int): Number of samples to load per batch
            img_paths (list): List of paths to the input images
            seg_paths (list): List of paths to the binary segmentation masks
            random_state (np.random.RandomState): Random state to use for shuffling
            load_x (callable): Function to load the input images
            load_y (callable): Function to load the binary segmentation masks
            augmentation_function (callable): Transformation function to apply to the input images
            sample_weights (tuple): Weights to apply to the foreground and background classes
            repeat_n_times (int): Number of times to iterate over the dataset per epoch
            shuffle (bool): Whether to shuffle the dataset after each batch

        """
        self.batch_size = batch_size
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.rs = random_state
        self.load_x = load_x
        self.load_y = load_y
        self.sample_weights = sample_weights
        if sample_weights:
            self.fg_weight = sample_weights[1]
            self.bg_weight = sample_weights[0]
        self.repeat_n_times = repeat_n_times
        self.shuffle = shuffle
        self.augmentation_function = augmentation_function

    def __len__(self):
        return (len(self.seg_paths) * self.repeat_n_times) // self.batch_size

    def __getitem__(self, idx):
        """Returns the batch (input, target) at index `idx`"""
        # Image index, offset by batch
        if self.repeat_n_times > 1:
            i = (idx * self.batch_size) % len(self.img_paths)
        else:
            i = idx * self.batch_size

        batch_img_paths = self.img_paths[i : i + self.batch_size]
        batch_seg_paths = self.seg_paths[i : i + self.batch_size]

        if self.shuffle or self.repeat_n_times > 1:
            remaining_samples_at_i = len(self.img_paths) - i
            if remaining_samples_at_i < self.batch_size:
                batch_img_paths += self.img_paths[0:self.batch_size - remaining_samples_at_i]
                batch_seg_paths += self.seg_paths[0:self.batch_size - remaining_samples_at_i]

        if self.shuffle:
            # Shuffle the data keeping pairs together
            indices = self.rs.permutation(len(self.img_paths))
            self.img_paths = [self.img_paths[i] for i in indices]
            self.seg_paths = [self.seg_paths[i] for i in indices]

        for i, im_path in enumerate(batch_img_paths):
            if Path(im_path).name != Path(batch_seg_paths[i]).name.replace('_mask', ''):
                raise ValueError(f"Image {im_path} and mask {batch_seg_paths[i]} do not match")

        x, y = d.compute([
            d.delayed(load_x)(batch_img_paths),
            d.delayed(load_y)(batch_seg_paths)
        ])[0]

        if self.augmentation_function is not None:
            x, y = self.augmentation_function(x, y)

        x = x[..., np.newaxis]
        y = y[..., np.newaxis]

        # If want to up/down-weight foreground/background pixel loss
        # This is good for images that tend to be imbalanced between foreground
        # and background pixel area.
        if self.sample_weights:
            batch_sample_weights = np.zeros(shape=y.shape)
            batch_sample_weights[y == 1] = self.fg_weight
            batch_sample_weights[y != 1] = self.bg_weight
            return x, y, batch_sample_weights

        return x, y
