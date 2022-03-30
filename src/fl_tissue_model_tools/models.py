from typing import Sequence
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Softmax
from tensorflow.keras.applications import resnet50
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt



def build_ResNet50_TL(base_model: Model, ll: int, img_shape: tuple) -> Model:
    base_model = Model(inputs=base_model.input, outputs=base_model.layers[ll].output)
    base_model.trainable = False
    inputs = Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model_1 = Model(inputs, outputs)
    return model_1


class ResNet50TLHyperModel(kt.HyperModel):
    def __init__(self, inp_shape: tuple[int, int, int], name: str=None, tunable: bool=True, last_layer_options: Sequence[str]=["conv5_block3_out"], min_fine_tune_lr: float=1e-5, max_epochs_1: int=10, max_epochs_2: int=10) -> None:
        super().__init__(name, tunable)
        self.inp_shape = inp_shape
        self.last_layer_options = last_layer_options
        self.min_fine_tune_lr = min_fine_tune_lr
        self.max_epochs_1 = max_epochs_1
        self.max_epochs_2 = max_epochs_2
        self.base_model: Model = None

    def build(self, hp: kt.HyperParameters) -> Model:
        self.base_model = resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=self.inp_shape)
        # Make sure to only choose from full convolution stages
        ll = hp.Choice("last_resnet_layer", self.last_layer_options)
        ll_idx = [l.name for l in self.base_model.layers].index(ll)
        self.base_model = Model(inputs=self.base_model.input, outputs=self.base_model.layers[ll_idx].output)
        self.base_model.trainable = False
        inputs = Input(shape=self.inp_shape)
        x = self.base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])
        return model


    def fit(self, hp: kt.HyperParameters, model: Model, *args, **kwargs) -> None:
        epochs_1 = hp.Int("epochs_1", min_value=2, max_value=self.max_epochs_1)
        epochs_2 = hp.Int("epochs_2", min_value=2, max_value=self.max_epochs_2)
        fine_tune_lr = hp.Float("fine_tune_lr", min_value=self.min_fine_tune_lr, max_value=1e-3, sampling="log")
        # Fit with frozen base model
        model.fit(*args, **kwargs, epochs=epochs_1)
        # Fine tune full model
        model.trainable = True
        model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy()]
        )
        return model.fit(*args, **kwargs, epochs=epochs_2)
