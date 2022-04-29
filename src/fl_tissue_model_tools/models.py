import json
import numpy as np
import dask as d
import tensorflow as tf
import keras_tuner as kt
# import keras.backend as K
import tensorflow.keras.backend as K

from operator import lt, gt
from ast import Global, Mod
from copy import deepcopy
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Softmax, Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.applications import resnet50
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy, Loss
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.activations import sigmoid

# Typing
import numpy.typing as npt
from numpy.random import RandomState
from typing import Sequence, Union, Callable, Any, Optional

# Custom
from fl_tissue_model_tools import data_prep


def _check_consec_factor(x: Sequence[float], factor: float, reverse: bool=False) -> bool:
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


def mean_iou_coef(y: tf.Tensor, yhat: tf.Tensor, smooth: float=1.0, obs_axes: tuple[int, ...]=(1, 2, 3), thresh: float=0.5):
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


def mean_iou_coef_factory(smooth: int=1, obs_axes: tuple[int, ...]=(1, 2, 3), thresh: float=0.5):
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
        return mean_iou_coef(y, yhat, smooth=smooth, obs_axes=obs_axes, thresh=thresh)
    fn.__name__ = f"mean_iou_coef"
    return fn


def build_ResNet50_TL(n_outputs: int, img_shape: tuple[int, int], base_init_weights: str="imagenet", base_last_layer: str="conv5_block3_out", output_act: str="sigmoid", base_model_trainable: bool=False, base_model_name: str="base_model") -> Model:
    """Build a ResNet50 model for transfer learning.

    Args:
        n_outputs: Number of output units.
        img_shape: Shape (h, w) of input images.
        base_init_weights: Weights to be loaded for base model.
        base_last_layer: Last layer of base model to keep. Should always be a convolution
            block output layer, due to ResNet50 architecture.
        output_act: Output activation for classification.
        base_model_trainable: Whether base model should be trainable at instantiation. This
            should usually be False so training can occur in two stages (frozen & fine tune).
        base_model_name: Name of the base model. Important to know if want to toggle
            `trainable` property later.

    Returns:
        A ResNet50 model, ready for transfer learning applications.

    """
    resnet50_model = resnet50.ResNet50(weights=base_init_weights, include_top=False, input_shape=img_shape)
    bll_idx = [l.name for l in resnet50_model.layers].index(base_last_layer)
    base_model = Model(inputs=resnet50_model.input, outputs=resnet50_model.layers[bll_idx].output)
    inputs = Input(shape=img_shape)
    # Base model layers should run in inference mode, even after unfreezing base model
    # for fine-tuning. 
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(n_outputs, activation=output_act)(x)
    model = Model(inputs, outputs)

    # Set layer properties
    base_model._name = base_model_name
    base_model.trainable = base_model_trainable

    return model


def build_UNetXception(n_outputs: int, img_shape: tuple[int, int], channels: int=1, filter_counts: tuple[int, int, int, int]=(32, 64, 128, 256), output_act: str="sigmoid") -> Model:
    """Build a UNetXception model for semantic segmentation.

    Adapted from Keras documentation: https://keras.io/examples/vision/oxford_pets_image_segmentation/

    Args:
        n_outputs: Number of output units.
        img_shape: Shape (h, w) of input images.
        channels: Number of channels. Assumed to be grayscale (i.e., 1).
        filter_counts: Number of filters for downsampling (reversed for upsampling). If
            elements are not in ascending order, they will be sorted. Elements must be
            consecutively increasing by a factor of 2 after sorting.
        output_act: Output activation for last layer. Sigmoid is best for binary classificaion.

    Returns:
        UNetXception model.

    """
    inputs = Input(img_shape + (channels,))
    # Validate filter counts
    filter_counts = sorted(filter_counts)
    assert _check_consec_factor(filter_counts, factor=2), f"Filter depths do not increase consecutively by a factor of 2."

    # Downsampling
    x = Conv2D(filter_counts[0], 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x

    # Downsampling Xception blocks
    for i, filters in enumerate(filter_counts[1:]):
        # Don't want a redundant activation layer on first block
        if i != 0:
            x = Activation("relu")(x)

        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)
        
        residual = Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)

        x = add([x, residual])

        previous_block_activation = x
    
    # Upsampling
    for filters in list(reversed(filter_counts)):
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])
        previous_block_activation = x
    
    # Output
    outputs = Conv2D(n_outputs, 3, activation=output_act, padding="same")(x)

    # Define model
    model = Model(inputs, outputs)

    return model


class ResNet50TLHyperModel(kt.HyperModel):
    """ ResNet50 Hypermodel for use with KerasTuner.

    """
    def __init__(self, n_outputs: int, img_shape: tuple[int, int], loss: Loss, metrics: Sequence, name: str=None, tunable: bool=True, base_init_weights: str="image_net", last_layer_options: Sequence[str]=["conv5_block3_out", "conv5_block2_out", "conv5_block1_out", "conv4_block6_out"], output_act: str="sigmoid", adam_beta_1_range: tuple=(0.85, 0.95), adam_beta_2_range: tuple=(0.98, 0.999), frozen_lr_range: tuple=(1e-5, 1e-2), fine_tune_lr_range: tuple=(1e-5, 1e-3), frozen_epochs: int=10, fine_tune_epochs: int=10, base_model_name: str="base_model") -> None:
        """Create ResNet Hypermodel for use with KerasTuner.

        Args:
            n_outputs: Number of output units.
            img_shape: Shape (h, w) of input images.
            loss: Loss used for training.
            metrics: Metrics to track during training.
            name: Name of transfer learning model (see KerasTuner documentation).
            tunable: Whether model is tunable (see KerasTuner documentation).
            base_init_weights: Weights to be loaded for base model.
            last_layer_options: Choices of last layer of base model to keep. All should be a
                convolution block output layer, due to ResNet50 architecture. This will be
                optimized.
            output_act: Output activation for classification.
            adam_beta_1_range: Range of values (min, max) for beta_1. Used for Adam optimizer.
            adam_beta_2_range: Range of values (min, max) for beta_2. Used for Adam optimizer.
            frozen_lr_range: Range of values (min, max) for frozen learning rate.
            fine_tune_lr_range: Range of values (min, max) for fine tune learning rate.
            frozen_epochs: Number of epochs to train frozen model.
            fine_tune_epochs: Number of epochs to fine tune model.
            base_model_name: Name of the base model. Important to know for toggling
                `trainable` property later.
        """
        super().__init__(name, tunable)
        self.n_outputs = n_outputs
        self.img_shape = tuple(deepcopy(img_shape))
        self.base_init_weights = base_init_weights
        self.last_layer_options = last_layer_options
        self.output_act = output_act
        self.frozen_lr_range = deepcopy(frozen_lr_range)
        self.fine_tune_lr_range = deepcopy(fine_tune_lr_range)
        self.frozen_epochs = frozen_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.loss = loss
        self.metrics = metrics
        self.base_model_name = base_model_name
        self.base_model: Model = None
        self.adam_beta_1_range: float = adam_beta_1_range
        self.adam_beta_2_range: float = adam_beta_2_range
        self.adam_beta_1: kt.HyperParameters.Float = None
        self.adam_beta_2: kt.HyperParameters.Float = None

    def build(self, hp: kt.HyperParameters) -> Model:
        ll = hp.Choice("last_resnet_layer", self.last_layer_options)
        model = build_ResNet50_TL(
            self.n_outputs,
            self.img_shape,
            base_last_layer=ll,
            output_act=self.output_act,
            base_model_name=self.base_model_name
        )
        frozen_lr = hp.Float(
            "frozen_lr",
            min_value=self.frozen_lr_range[0],
            max_value=self.frozen_lr_range[1],
            sampling="log"
        )
        self.adam_beta_1 = hp.Float(
            "adam_beta_1",
            min_value=self.adam_beta_1_range[0],
            max_value=self.adam_beta_1_range[1],
            sampling="log"
        )
        self.adam_beta_2 = hp.Float(
            "adam_beta_2",
            min_value=self.adam_beta_2_range[0],
            max_value=self.adam_beta_2_range[1],
            sampling="log"
        )
        frozen_opt = Adam(learning_rate=frozen_lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        model.compile(frozen_opt, self.loss, self.metrics)
        return model

    def fit(self, hp: kt.HyperParameters, model: Model, *args, **kwargs) -> History:
        fine_tune_lr = hp.Float(
            "fine_tune_lr",
            min_value=self.fine_tune_lr_range[0],
            max_value=self.fine_tune_lr_range[1],
            sampling="log"
        )
        fine_tune_opt = Adam(learning_rate=fine_tune_lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        # Fit with frozen base model
        model.fit(*args, **kwargs, epochs=self.frozen_epochs)
        # Fine tune full model
        toggle_TL_freeze(model, self.base_model_name)
        model.compile(fine_tune_opt, self.loss, self.metrics)
        return model.fit(*args, **kwargs, epochs=self.fine_tune_epochs)


class UNetXceptionGridSearch():
    """ Wrapper class for handling grid search for UNet model.

    """
    def __init__(self, save_dir: str, filter_counts_options: Sequence[tuple[int, int, int, int]], n_outputs: int, img_shape: tuple[int, int], optimizer, loss, channels: int=1, output_act: str="sigmoid", callbacks=[], metrics=None) -> None:
        """Create wrapper class for handling grid search for UNet model. 

        Args:
            save_dir: Root directory for saving optimization output.
            filter_counts_options: Choices of filter counts for model. This will
                be optimized.
            n_outputs: Number of output units.
            img_shape: Shape (h, w) of input images.
            optimizer: Optimizer to use for training.
            loss: Loss to use for training.
            channels: Number of channels. Assumed to be grayscale (i.e., 1).
            output_act: Output activation for last layer. Sigmoid is best for binary classificaion.
            callbacks: Callbacks for training. A `ModelCheckpoint` callback will automatically
                be used in addition to any other callbacks.
            metrics: Metrics to track during training. Must include the metric passed to
                `search()` as `objective`. Note: if the search objective will be "val_[metric]", only
                [metric] needs to be included.
    
        """
        self.best_filter_counts = []
        self.best_score = np.NaN
        self.best_score_idx = 0
        self.filter_counts_options = filter_counts_options
        self.save_dir = save_dir
        self.n_outputs = n_outputs
        self.img_shape = img_shape
        self.channels = channels
        self.output_act = output_act
        self.optimizer = optimizer
        self.loss = loss
        self.callbacks = callbacks
        self.metrics = metrics
        self.histories = []

        data_prep.make_dir(self.save_dir)
    
    def search(self, objective: str, comparison: str, *args, search_verbose: bool=True, **kwargs) -> None:
        """Execute grid search using `objective` to assess performance.

        Args:
            objective: Objective function used to assess model performance. Should
                be one of the metrics passed to __init__().
            comparison: One of ["min", "max"]. Used to assess hyperparameter performance.
                If higher score of `objective` is good, use `max`, otherwise, use `min`. 
            search_verbose: Print out updates during grid search.

        """
        assert comparison == "min" or comparison == "max", f"comparison operator must be either {min} or {max}"
        if comparison == "min":
            self.best_score = np.inf
            compare = lt
            get_best = np.min
            get_best_idx = np.argmin
        else:
            self.best_score = -np.inf
            compare = gt
            get_best = np.max
            get_best_idx = np.argmax
        

        for i, fc in enumerate(self.filter_counts_options):
            if search_verbose:
                print(f"Testing filter counts: {fc}")
            best_weights_file = f"{self.save_dir}/best_weights_config_{i}.h5"
            cp_callback = ModelCheckpoint(best_weights_file, save_best_only=True, save_weights_only=True)
            model = build_UNetXception(self.n_outputs, self.img_shape, channels=self.channels, filter_counts=fc, output_act=self.output_act)
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            h = model.fit(callbacks=self.callbacks + [cp_callback],*args, **kwargs)
            self.histories.append(h)
            
            cur_best_score = get_best(h.history[objective])
            cur_best_score_idx = get_best_idx(h.history[objective])
            if search_verbose:
                print(f"Best objective value observed: {cur_best_score}")
                print(f"Best objective value observed on epoch: {cur_best_score_idx + 1}")
                print(f"Previous best score: {self.best_score}")
            
            # If model is an improvement
            if compare(cur_best_score, self.best_score):
                if search_verbose:
                    print(f"Current best ({cur_best_score}) is an improvement over previous best ({self.best_score}).")
                self.best_score = cur_best_score
                self.best_filter_counts = deepcopy(fc)
                self.best_score_idx = i
                if search_verbose:
                    print(f"Current best hyper parameters: {self.best_filter_counts}")
                    print(f"Current best objective value observed: {self.best_score}")
                # TODO: save hyperparams for best model
                with open(f"{self.save_dir}/best_model_hps.json", "w") as fp:
                    hp_meta = {
                        "search_objective": objective,
                        "best_score": self.best_score,
                        "best_hps": {"filter_counts": self.best_filter_counts},
                        "best_weights_file": best_weights_file}
                    json.dump(hp_meta, fp)
            else:
                if search_verbose:
                    print(f"Current best ({cur_best_score}) is not an improvement over previous best ({self.best_score}).")

    def get_best_model(self) -> None:
        """Get the best model obtained during the hyperparameter search.

        Returns:
            The best model found during the hyperparameter search.

        """
        model = build_UNetXception(self.n_outputs, self.img_shape, channels=self.channels, filter_counts=self.best_filter_counts, output_act=self.output_act)
        model.load_weights(f"{self.save_dir}/best_weights_config_{self.best_score_idx}.h5")
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model
