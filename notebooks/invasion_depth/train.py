import os
from glob import glob
import json
import multiprocessing

# Set this to false if you are not running on the Oregon State University HPC cluster
RUNNING_ON_HPC_CLUSTER = True

if RUNNING_ON_HPC_CLUSTER:
    '''
    Before running this script, run these commands to load the necessary modules with slurm:
    module load cuda
    module load gcc
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    CONDA_PREFIX = os.environ['CONDA_PREFIX']
    CUDA_PATH=os.environ['CUDA_PATH']
    LD_LIBRARY_PATHS = f':{CONDA_PREFIX}/lib:{CUDA_PATH}/lib64:{CUDA_PATH}/extras/CUPTI/lib64'
    if LD_LIBRARY_PATHS not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH']+=LD_LIBRARY_PATHS
    os.environ['XLA_FLAGS']=f'--xla_gpu_cuda_data_dir={CUDA_PATH}'

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from fl_tissue_model_tools import data_prep, dev_config, models
import fl_tissue_model_tools.preprocessing as prep


dirs = dev_config.get_dev_directories("../../dev_paths.txt")

with open("../../model_training/invasion_depth_training_values.json", 'r') as fp:
    training_values = json.load(fp)
training_values["rs_seed"] = None if (training_values["rs_seed"] == "None") else training_values["rs_seed"]

with open("../../model_training/invasion_depth_best_hp.json", 'r') as fp:
    best_hp = json.load(fp)

### Data paths ###
root_data_path = f"{dirs.data_dir}/invasion_data/"
model_training_path = f"{dirs.analysis_dir}/resnet50_invasion_model"
demo_model_training_path = f"{model_training_path}/demo"
mcp_best_frozen_weights_file = f"{demo_model_training_path}/best_frozen_weights.h5"
mcp_best_finetune_weights_file = f"{demo_model_training_path}/best_finetune_weights.h5"

### General training parameters ###
resnet_inp_shape = tuple(training_values["resnet_inp_shape"])
class_labels = training_values["class_labels"]
# Binary classification -> only need 1 output unit
n_outputs = 1

seed = training_values["rs_seed"]
val_split = training_values["val_split"]
batch_size = training_values["batch_size"]
frozen_epochs = training_values["frozen_epochs"]
fine_tune_epochs = training_values["fine_tune_epochs"]
# frozen_epochs = 5
# fine_tune_epochs = 5
adam_beta_1 = best_hp["adam_beta_1"]
adam_beta_2 = best_hp["adam_beta_2"]
frozen_lr = best_hp["frozen_lr"]
fine_tune_lr = best_hp["fine_tune_lr"]
last_resnet_layer = best_hp["last_resnet_layer"]

### Early stopping ###
es_criterion = "val_loss"
es_mode = "min"
# Update these depending on seriousness of experiment
es_patience = training_values["early_stopping_patience"]
es_min_delta = training_values["early_stopping_min_delta"]

### Model saving ###
mcp_criterion = "val_loss"
mcp_mode = "min"
mcp_best_only = True
# Need to set to True otherwise base model "layer" won't save/load properly
mcp_weights_only = True

data_prep.make_dir(demo_model_training_path)

rs = np.random.RandomState(seed)

# Training & validation data (drawn from same image set & randomly assigned)
tv_class_paths = {v: glob(f"{root_data_path}/train/{k}/*.tif") for k, v in class_labels.items()}
for k, v in tv_class_paths.items():
    rs.shuffle(v)

train_data_paths, val_data_paths = data_prep.get_train_val_split(tv_class_paths, val_split=val_split)

train_datagen = data_prep.InvasionDataGenerator(
    train_data_paths,
    class_labels,
    batch_size,
    resnet_inp_shape[:2],
    rs,
    class_weights=True,
    shuffle=True,
    augmentation_function=prep.augment_invasion_imgs
)

val_datagen = data_prep.InvasionDataGenerator(
    val_data_paths,
    class_labels,
    batch_size,
    resnet_inp_shape[:2],
    rs,
    class_weights=train_datagen.class_weights,
    shuffle=True,
    augmentation_function=train_datagen.augmentation_function
)

K.clear_session()

tl_model = models.build_ResNet50_TL(
    n_outputs,
    resnet_inp_shape,
    base_last_layer=last_resnet_layer,
    # Switch to softmax once n_outputs > 1
    output_act="sigmoid",
    base_model_trainable=False
)

tl_model.compile(
    optimizer=Adam(learning_rate=frozen_lr, beta_1=adam_beta_1, beta_2=adam_beta_2),
    loss=BinaryCrossentropy(),
    weighted_metrics=[BinaryAccuracy()]
)

try:
    print(tl_model.summary())
except Exception as e:
    print(e)

es_callback = EarlyStopping(monitor=es_criterion, mode=es_mode, min_delta=es_min_delta, patience=es_patience)
mcp_callback = ModelCheckpoint(mcp_best_frozen_weights_file, monitor=mcp_criterion, mode=mcp_mode, save_best_only=mcp_best_only, save_weights_only=mcp_weights_only)

n_cores = multiprocessing.cpu_count()
n_workers = min(16, n_cores-1)

h1 = tl_model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=frozen_epochs,
    callbacks=[es_callback, mcp_callback],
    workers=n_workers
)

print((np.argmin(h1.history["val_loss"]), np.min(h1.history["val_loss"])))
print((np.argmax(h1.history["val_binary_accuracy"]), np.max(h1.history["val_binary_accuracy"])))

### Fine tune training ###

tl_model.load_weights(mcp_best_frozen_weights_file)

# Make base model trainable (leave layers in inference mode)
models.toggle_TL_freeze(tl_model)

# tl_model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])
tl_model.compile(
    optimizer=Adam(learning_rate=fine_tune_lr, beta_1=adam_beta_1, beta_2=adam_beta_2),
    loss=BinaryCrossentropy(),
    weighted_metrics=[BinaryAccuracy()]
)

try:
    print(tl_model.summary())
except Exception as e:
    print(e)

es_callback = EarlyStopping(monitor=es_criterion, mode=es_mode, min_delta=es_min_delta, patience=es_patience)
mcp_callback = ModelCheckpoint(mcp_best_finetune_weights_file, monitor=mcp_criterion, mode=mcp_mode, save_best_only=mcp_best_only, save_weights_only=mcp_weights_only)

h2 = tl_model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=fine_tune_epochs,
    callbacks=[es_callback, mcp_callback],
    workers=n_workers
)

print((np.argmin(h2.history["val_loss"]), np.min(h2.history["val_loss"])))
print((np.argmax(h2.history["val_binary_accuracy"]), np.max(h2.history["val_binary_accuracy"])))
