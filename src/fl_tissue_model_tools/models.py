from ast import Global
from copy import deepcopy
from typing import Sequence, Union, Callable, Any
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Softmax
from tensorflow.keras.applications import resnet50
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import BinaryCrossentropy, Loss
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.activations import sigmoid
import keras_tuner as kt


def build_ResNet50_TL(n_outputs: int, img_shape: tuple[int, int], base_init_weights: str="imagenet", base_last_layer: str="conv5_block3_out", output_act: str="sigmoid", base_model_trainable: bool=False, base_model_name: str="base_model") -> Model:
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


def build_UNetXception(n_outputs: int, img_shape: tuple[int, int]) -> Model:
    pass


def toggle_TL_freeze(tl_model: Model, base_model_name: str="base_model") -> None:
    base_model = tl_model.get_layer(base_model_name)
    base_model.trainable = not base_model.trainable


class ResNet50TLHyperModel(kt.HyperModel):
    def __init__(self, n_outputs: int, img_shape: tuple[int, int], frozen_optimizer: Optimizer, fine_tune_optimizer: Callable[[Any], Optimizer], loss: Loss, metrics: Sequence, name: str=None, tunable: bool=True, base_init_weights: str="image_net", last_layer_options: Sequence[str]=["conv5_block3_out", "conv5_block2_out", "conv5_block1_out", "conv4_block6_out"], output_act: str="sigmoid", min_fine_tune_lr: float=1e-5, frozen_epochs: int=10, fine_tune_epochs: int=10, base_model_name: str="base_model") -> None:
        super().__init__(name, tunable)
        self.n_outputs = n_outputs
        self.img_shape = tuple(deepcopy(img_shape))
        self.base_init_weights = base_init_weights
        self.last_layer_options = last_layer_options
        self.output_act = output_act
        self.min_fine_tune_lr = min_fine_tune_lr
        self.frozen_epochs = frozen_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.frozen_optimizer = frozen_optimizer
        self.fine_tune_optimizer = fine_tune_optimizer
        self.loss = loss
        self.metrics = metrics
        self.base_model_name = base_model_name
        self.base_model: Model = None

    def build(self, hp: kt.HyperParameters) -> Model:
        ll = hp.Choice("last_resnet_layer", self.last_layer_options)
        model = build_ResNet50_TL(
            self.n_outputs,
            self.img_shape,
            base_last_layer=ll,
            output_act=self.output_act,
            base_model_name=self.base_model_name
        )
        model.compile(self.frozen_optimizer, self.loss, self.metrics)
        return model

    def fit(self, hp: kt.HyperParameters, model: Model, *args, **kwargs) -> History:
        fine_tune_lr = hp.Float("fine_tune_lr", min_value=self.min_fine_tune_lr, max_value=1e-3, sampling="log")
        # Fit with frozen base model
        model.fit(*args, **kwargs, epochs=self.frozen_epochs)
        # Fine tune full model
        toggle_TL_freeze(model, self.base_model_name)
        model.compile(self.fine_tune_optimizer(fine_tune_lr), self.loss, self.metrics)
        return model.fit(*args, **kwargs, epochs=self.fine_tune_epochs)
