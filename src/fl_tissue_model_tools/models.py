import json
from operator import lt, gt
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Tuple, Optional
from itertools import product
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from numpy import typing as npt
import keras_tuner as kt
from PIL import Image
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image

import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.applications import resnet50
from tensorflow.keras.callbacks import History, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import Loss

from fl_tissue_model_tools.smooth_tiled_predictions import predict_img_with_smooth_windowing
from fl_tissue_model_tools import defs
from fl_tissue_model_tools import models_util as mu

def build_ResNet50_TL(
    n_outputs: int, img_shape: Tuple[int, int],
    base_init_weights: str="imagenet", base_last_layer: str="conv5_block3_out",
    output_act: str="sigmoid", base_model_trainable: bool=False,
    base_model_name: str="base_model"
) -> Model:
    """Build a ResNet50 model for transfer learning.

    Args:
        n_outputs: Number of output units.
        img_shape: Shape (h, w) of input images.
        base_init_weights: Weights to be loaded for base model.
        base_last_layer: Last layer of base model to keep. Should always be a convolution
            block output layer, due to ResNet50 architecture.
        output_act: Output activation for classification.
        base_model_trainable: Whether base model should be trainable at instantiation.
            This should usually be False so training can occur in two stages
            (frozen & fine tune).
        base_model_name: Name of the base model. Important to know if want to toggle
            `trainable` property later.

    Returns:
        A ResNet50 model, ready for transfer learning applications.

    """
    resnet50_model = resnet50.ResNet50(
        weights=base_init_weights, include_top=False, input_shape=img_shape
    )
    bll_idx = [l.name for l in resnet50_model.layers].index(base_last_layer)
    base_model = Model(
        inputs=resnet50_model.input, outputs=resnet50_model.layers[bll_idx].output
    )
    inputs = Input(shape=img_shape)
    # Base model layers should run in inference mode, even after unfreezing base model
    # for fine-tuning.
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    # outputs = Dense(n_outputs, activation=output_act)(x)
    x = Dense(n_outputs)(x)
    outputs = Activation(activation=output_act)(x)
    model = Model(inputs, outputs)

    # Set layer properties
    base_model._name = base_model_name
    base_model.trainable = base_model_trainable

    return model


def build_UNetXception(
    n_outputs: int, img_shape: Tuple[int, int], channels: int = 1,
    filter_counts: Tuple[int, int, int, int] = (32, 64, 128, 256),
    output_act: str = "sigmoid"
) -> Model:
    """Build a UNetXception model for semantic segmentation.

    Adapted from Keras documentation:
        https://keras.io/examples/vision/oxford_pets_image_segmentation/

    Args:
        n_outputs: Number of output units.
        img_shape: Shape (h, w) of input images.
        channels: Number of channels. Assumed to be grayscale (i.e., 1).
        filter_counts: Number of filters for downsampling (reversed for upsampling). If
            elements are not in ascending order, they will be sorted. Elements must be
            consecutively increasing by a factor of 2 after sorting.
        output_act: Output activation for last layer.
            Sigmoid is best for binary classificaion.

    Returns:
        UNetXception model.
    """
    inputs = Input(img_shape + (channels,))
    # Validate filter counts
    filter_counts = sorted(filter_counts)

    assert mu.check_consec_factor(filter_counts, factor=2), ("Filter depths do not "
                                                           "increase consecutively "
                                                           "by a factor of 2.")

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

        residual = Conv2D(
            filters, 1, strides=2, padding="same"
        )(previous_block_activation)

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
    """ ResNet50 Hypermodel for use with KerasTuner."""
    def __init__(
        self,
        n_outputs: int,
        img_shape: Tuple[int, int],
        loss: Loss,
        weighted_metrics: Sequence,
        name: str=None,
        tunable: bool=True,
        base_init_weights: str="image_net",
        last_layer_options: Sequence[str] = [
            "conv5_block3_out", "conv5_block2_out",
            "conv5_block1_out", "conv4_block6_out"],
        output_act: str="sigmoid",
        adam_beta_1_range: tuple=(0.85, 0.95),
        adam_beta_2_range: tuple=(0.98, 0.999),
        frozen_lr_range: tuple=(1e-5, 1e-2),
        fine_tune_lr_range: tuple=(1e-5, 1e-3),
        frozen_epochs: int=50,
        fine_tune_epochs: int=50,
        base_model_name: str="base_model",
        es_criterion: str="val_loss",
        es_mode: str="min",
        es_patience: int=5,
        es_min_delta: float=0.0001,
        mcp_criterion: str="val_loss",
        mcp_mode: str="min",
        mcp_best_frozen_weights_path: str="best_frozen_weights"
    ) -> None:
        """Create ResNet Hypermodel for use with KerasTuner.

        Args:
            n_outputs: Number of output units.
            img_shape: Shape (h, w) of input images.
            loss: Loss used for training.
            weighted_metrics: Metrics to track during training. Metrics will be weighted
                by sample weights, if the dataset outputs them.
            name: Name of transfer learning model (see KerasTuner documentation).
            tunable: Whether model is tunable (see KerasTuner documentation).
            base_init_weights: Weights to be loaded for base model.
            last_layer_options: Choices of last layer of base model to keep. All should be a
                convolution block output layer, due to ResNet50 architecture. Part of
                hyperparameter search space.
            output_act: Output activation for classification.
            adam_beta_1_range: Range of Adam optimizer values (min, max) for beta_1.
                Part of hyperparameter search space.
            adam_beta_2_range: Range of Adam optimizer values (min, max) for beta_2.
                Part of hyperparameter search space.
            frozen_lr_range: Range of values (min, max) for frozen learning rate. Part
                of hyperparameter search space.
            fine_tune_lr_range: Range of values (min, max) for fine tune learning rate.
                Part of hyperparameter search space.
            frozen_epochs: Number of epochs to train frozen model.
            fine_tune_epochs: Number of epochs to fine tune model.
            base_model_name: Name of the base model. Important to know for toggling
                `trainable` property later.
            es_criterion: Criterion for early stopping callback. See Keras documentation.
            es_mode: Mode for early stopping callback. See Keras documentation.
            es_patience: Earl stopping callback patience. See Keras documentation.
            es_min_delta: Early stopping callback minimum delta. See Keras documentation.
            mcp_criterion: Model checkpoint callback criterion. See Keras documentation.
            mcp_mode: Model checkpoint callback mode. See Keras documentation.
            mcp_best_frozen_weights_path: Model checkpoint callback path for saved (best)
                weights. See Keras documentation.

        """
        super().__init__(name, tunable)
        self.n_outputs = n_outputs
        self.img_shape = tuple(deepcopy(img_shape))
        self.loss = loss
        self.weighted_metrics = weighted_metrics
        self.base_init_weights = base_init_weights
        self.last_layer_options = deepcopy(last_layer_options)
        self.output_act = output_act
        self.adam_beta_1_range: float = adam_beta_1_range
        self.adam_beta_2_range: float = adam_beta_2_range
        self.frozen_lr_range = deepcopy(frozen_lr_range)
        self.fine_tune_lr_range = deepcopy(fine_tune_lr_range)
        self.frozen_epochs = frozen_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.base_model_name = base_model_name
        self.es_criterion = es_criterion
        self.es_mode = es_mode
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.mcp_criterion = mcp_criterion
        self.mcp_mode = mcp_mode
        self.mcp_best_frozen_weights_path = mcp_best_frozen_weights_path
        self.base_model: Model = None

        self.adam_beta_1: kt.HyperParameters.Float = None
        self.adam_beta_2: kt.HyperParameters.Float = None

    def build(self, hp: kt.HyperParameters) -> Model:
        """Build hypermodel using tunable hyperparameters.

        Args:
            hp: Hyperparameters to be tuned during model building. These
                are passed automatically during search.

        Returns:
            Model: Compiled ResNet50 transfer learning model with frozen base
                model.
        """
        ### Hyperparameters ###
        ll = hp.Choice("last_resnet_layer", self.last_layer_options)
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

        ### Build model ###
        model = build_ResNet50_TL(
            self.n_outputs,
            self.img_shape,
            base_last_layer=ll,
            output_act=self.output_act,
            base_model_name=self.base_model_name
        )

        ### Optimizer (frozen) ###
        frozen_opt = optimizers.Adam(
            learning_rate=frozen_lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2
        )

        model.compile(frozen_opt, self.loss, weighted_metrics=self.weighted_metrics)
        return model

    def fit(self, hp: kt.HyperParameters, model: Model, *args, **kwargs) -> History:
        """Fit a ResNet50 hypermodel.

        Args:
            hp: Hyperparameters to be tuned during model training. These
                are passed automatically during search.
            model: Model for which hyperparameters are being tuned. 

        Returns:
            History: Model training history for the model trained using a given
                hyperparameter configuration.
        """

        ### Callbacks ###
        frozen_es_callback = EarlyStopping(
            monitor=self.es_criterion,
            mode=self.es_mode,
            min_delta=self.es_min_delta,
            patience=self.es_patience
        )

        frozen_mcp_callback = ModelCheckpoint(
            filepath=self.mcp_best_frozen_weights_path,
            monitor=self.mcp_criterion,
            mode=self.mcp_mode,
            save_best_only=True,
            save_weights_only=True
        )

        fine_tune_es_callback = EarlyStopping(
            monitor=self.es_criterion,
            mode=self.es_mode,
            min_delta=self.es_min_delta,
            patience=self.es_patience
        )

        ### Hyperparameters (fine tune) ###
        fine_tune_lr = hp.Float(
            "fine_tune_lr",
            min_value=self.fine_tune_lr_range[0],
            max_value=self.fine_tune_lr_range[1],
            sampling="log"
        )

        # Keras Tuner passes a callbacks argument, pop to remove from
        # Kwargs (will add back later)
        kt_callbacks = kwargs.pop("callbacks")

        ### Optimizer (fine tune) ###
        fine_tune_opt = optimizers.Adam(learning_rate=fine_tune_lr, beta_1=self.adam_beta_1,
                                        beta_2=self.adam_beta_2)

        ### Fitting ###
        # Fit with frozen base model
        model.fit(*args, **kwargs, epochs=self.frozen_epochs,
                callbacks=kt_callbacks + [frozen_es_callback, frozen_mcp_callback])

        # Load best weights from training with frozen base model
        model.load_weights(self.mcp_best_frozen_weights_path)

        # Fit fine tuned full model
        mu.toggle_TL_freeze(model, self.base_model_name)
        model.compile(fine_tune_opt, self.loss, weighted_metrics=self.weighted_metrics)

        return model.fit(
            *args, **kwargs, epochs=self.fine_tune_epochs,
            callbacks=kt_callbacks + [fine_tune_es_callback]
        )


class UNetXceptionGridSearch():
    """ Wrapper class for handling grid search for UNet model."""
    def __init__(
        self,
        save_dir: str,
        filter_counts_options: Sequence[Tuple[int, int, int, int]],
        optimizer_cfg_options: Sequence[dict],
        n_outputs: int,
        img_shape: Tuple[int, int],
        loss,
        channels: int=1,
        output_act: str="sigmoid",
        callbacks=None,
        metrics=None,
    ) -> None:
        """Create wrapper class for handling grid search for UNet model.

        Args:
            save_dir: Root directory for saving optimization output.
            filter_counts_options: Choices of filter counts for model. This will
                be optimized.
            n_outputs: Number of output units.
            img_shape: Shape (h, w) of input images.
            optimizer_cfg: Optimizer configuration serialized with tf.keras.optimizers.serialize.
            loss: Loss to use for training.
            channels: Number of channels. Assumed to be grayscale (i.e., 1).
            output_act: Output activation for last layer. Sigmoid is best for
                binary classificaion.
            callbacks: Callbacks for training. A `ModelCheckpoint` callback will
                automatically be used in addition to any other callbacks.
            metrics: Metrics to track during training. Must include the metric passed to
                `search()` as `objective`. Note: if the search objective will be
                "val_[metric]", only [metric] needs to be included.
        """
        self.best_filter_counts = []
        self.best_optimizer_cfg = {}
        self.best_score = np.NaN
        self.best_score_idx = 0
        self.filter_counts_options = filter_counts_options
        self.save_dir = save_dir
        self.n_outputs = n_outputs
        self.img_shape = img_shape
        self.channels = channels
        self.output_act = output_act
        self.optimizer_cfg_options = optimizer_cfg_options
        self.loss = loss
        self.callbacks = callbacks if callbacks is not None else []
        self.metrics = metrics
        self.histories = []

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


    def search(self, objective: str, comparison: str, *args, search_verbose: bool=True,
                **kwargs) -> None:
        """Execute grid search using `objective` to assess performance.

        Args:
            objective: Objective function used to assess model performance. Should
                be one of the metrics passed to __init__().
            comparison: One of ["min", "max"]. Used to assess hyperparameter performance.
                If higher score of `objective` is good, use `max`, otherwise, use `min`.
            search_verbose: Print out updates during grid search.
        """
        assert comparison == "min" or comparison == "max", (
            "comparison operator must be either \"min\" or \"max\""
        )
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

        # iterate through the input list of filter counts,
        # fit a unet with each, and save the best
        hp_gen = product(self.filter_counts_options, range(len(self.optimizer_cfg_options)))
        for i, (fc, optim_cfg_idx) in enumerate(hp_gen):
            optimizer_cfg = self.optimizer_cfg_options[optim_cfg_idx]
            if search_verbose:
                print(f"Testing filter counts: {fc}")
                print(f"Optimizer index: {optim_cfg_idx}")
            K.clear_session()
            best_weights_file = f"{self.save_dir}/best_weights_config_{i}.weights.h5"
            cp_callback = ModelCheckpoint(
                best_weights_file, save_best_only=True, save_weights_only=True, monitor="loss"
            )
            model = build_UNetXception(
                self.n_outputs, self.img_shape, channels=self.channels,
                filter_counts=fc, output_act=self.output_act
            )

            # get optimizer
            optimizer = optimizers.deserialize(optimizer_cfg)

            # compile model
            model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
            callbacks = self.callbacks + [cp_callback]
            h = model.fit(callbacks=callbacks, *args, **kwargs)
            self.histories.append(h)

            cur_best_score = get_best(h.history[objective])
            cur_best_score_idx = get_best_idx(h.history[objective])
            if search_verbose:
                print(f"Best objective value observed: {cur_best_score}")
                print(f"Best objective value observed on epoch: {cur_best_score_idx + 1}")
                print(f"Previous best score: {self.best_score}")

            # If model is an improvement, notify user and save model
            if compare(cur_best_score, self.best_score):
                if search_verbose:
                    print(
                        f"Current best ({cur_best_score}) is an improvement over "
                        f"previous best ({self.best_score})."
                    )
                self.best_score = cur_best_score
                self.best_filter_counts = deepcopy(fc)

                self.best_optimizer_cfg = deepcopy(optimizer_cfg)

                self.best_score_idx = i
                if search_verbose:
                    print("Current best hyper parameters:\n")
                    print(f"  Best filter counts: {self.best_filter_counts}")
                    print(f"  Best optimizer: {self.best_optimizer_cfg}")
                    print(f"Current best objective value observed: {self.best_score}")
                # TODO: save hyperparams for best model

                with open(f"{self.save_dir}/best_model_hps.json", "w") as fp:
                    best_optim_cfg = deepcopy(self.best_optimizer_cfg)
                    try:
                        best_optim_cfg_str = json.dumps(best_optim_cfg)
                    except TypeError:  # learning rate schedule may need to be serialized
                        serialized_lr = optimizers.serialize(best_optim_cfg['config']['learning_rate'])
                        best_optim_cfg['config']['learning_rate'] = serialized_lr
                        best_optim_cfg_str = json.dumps(best_optim_cfg)
                    hp_meta = {
                        "search_objective": objective,
                        "best_score": self.best_score,
                        "best_hps": {
                            "filter_counts": self.best_filter_counts,
                            "optimizer_cfg": best_optim_cfg_str
                        },
                        "best_weights_file": best_weights_file
                    }
                    json.dump(hp_meta, fp)
            else:
                if search_verbose:
                    print(
                        f"Current best ({cur_best_score}) is not an improvement "
                        f"over previous best ({self.best_score})."
                    )

    def get_best_model(self) -> None:
        """Get the best model obtained during the hyperparameter search.

        Returns:
            The best model found during the hyperparameter search.

        """
        model = build_UNetXception(
            self.n_outputs, self.img_shape, channels=self.channels,
            filter_counts=self.best_filter_counts, output_act=self.output_act
        )
        model.load_weights(f"{self.save_dir}/best_weights_config_{self.best_score_idx}.weights.h5")
        optimizer = optimizers.deserialize(self.best_optimizer_cfg)
        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        return model


class UNetXceptionPatchSegmentor():
    """ Class for binary segmentation inference on images in patches with UNetXception model."""
    def __init__(self, patch_size: int, checkpoint_file: str, filter_counts: Tuple[int],
                 ds_ratio: float=0.5, norm_mean: Optional[float]=None, norm_std: Optional[float]=None, channels: int=1):
        self.patch_size = patch_size
        self.channels = channels
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.ds_ratio = ds_ratio
        self.model = build_UNetXception(
            1, (patch_size, patch_size), channels=channels, filter_counts=filter_counts, output_act="sigmoid"
        )
        self.model.load_weights(checkpoint_file)

    def predict(self, x: npt.NDArray, auto_resample=True) -> npt.NDArray:
        x = x.astype(np.float32)
        original_shape = x.shape
        target_shape = tuple(np.round(np.multiply(original_shape[:2], self.ds_ratio)).astype(int))
        do_resampling = original_shape != target_shape and auto_resample
        if do_resampling:
            x = Image.fromarray(x).resize(target_shape, resample=Image.Resampling.LANCZOS)
            x = np.array(x)

        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / self.norm_std

        pred = predict_img_with_smooth_windowing(
            x,
            window_size=self.patch_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            pred_func=self.model.predict
        )

        if do_resampling:
            pred = Image.fromarray(pred).resize(original_shape, resample=Image.Resampling.NEAREST)
            pred = np.array(pred)

        return pred


def get_unet_patch_segmentor_from_cfg(cfg_json: str) -> UNetXceptionPatchSegmentor:
    """Get a UNetXceptionPatchSegmentor object from a config json file.

    Args:
        cfg_json: Path to config json file.

    Returns:
        A UNetXceptionPatchSegmentor.

    """
    with open(cfg_json, "r") as fp:
        cfg = json.load(fp)

    models_dir = Path(defs.MODEL_TRAINING_DIR)
    checkpoint_file = models_dir / "binary_segmentation" / "checkpoints" / cfg["checkpoint_file"]

    segmentor = UNetXceptionPatchSegmentor(
        cfg["patch_size"],
        checkpoint_file,
        cfg["filter_counts"],
        ds_ratio=cfg.get("ds_ratio", 1),
        norm_mean=cfg.get("norm_mean", None),
        norm_std=cfg.get("norm_std", None),
        channels=cfg.get("channels", 1)
    )

    return segmentor
