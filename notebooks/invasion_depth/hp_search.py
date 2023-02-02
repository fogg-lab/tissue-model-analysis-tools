import os
from glob import glob
import json

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
import keras_tuner as kt
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.config import threading

threading.set_inter_op_parallelism_threads(1)
threading.set_intra_op_parallelism_threads(1)

from fl_tissue_model_tools import data_prep, dev_config, models
import fl_tissue_model_tools.preprocessing as prep

dirs = dev_config.get_dev_directories("../../dev_paths.txt")

with open("../../model_training/invasion_depth_training_values.json", 'r') as fp:
    training_values = json.load(fp)
training_values["rs_seed"] = None if (training_values["rs_seed"] == "None") else training_values["rs_seed"]

with open("../../model_training/invasion_depth_hp_space.json", 'r') as fp:
       hp_search_space = json.load(fp)

### Data paths ###
root_data_path = f"{dirs.data_dir}/invasion_data/"
model_training_path = f"{dirs.analysis_dir}/resnet50_invasion_model"
project_name = "invasion_hp_trials"
hypermodel_name = "invasion_depth_hypermodel"
hp_search_hp_path = f"{model_training_path}/hyperparameter_search_hps"
hp_search_weights_path = f"{model_training_path}/hyperparameter_search_weights"
best_hp_file = f"{hp_search_hp_path}/best_hyperparams_v1.json"
mcp_best_frozen_weights_file = f"{hp_search_weights_path}/best_frozen_weights.h5"


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


### Early stopping ###
es_criterion = "val_loss"
es_mode = "min"
# Update these depending on seriousness of experiment
es_patience = training_values["early_stopping_patience"]
es_min_delta = training_values["early_stopping_min_delta"]


### Frozen model saving (for transitioning from frozen model to fine-tuned model) ###
mcp_criterion = "val_loss"
mcp_mode = "min"


### Hyperparameter search ###
adam_beta_1_range = tuple(hp_search_space["adam_beta_1_range"])
adam_beta_2_range = tuple(hp_search_space["adam_beta_2_range"])
frozen_lr_range = tuple(hp_search_space["frozen_lr_range"])
fine_tune_lr_range = tuple(hp_search_space["fine_tune_lr_range"])
last_layer_options = hp_search_space["last_layer_options"]
num_initial_points = hp_search_space["num_initial_points"]
max_opt_trials = hp_search_space["max_opt_trials"]
# num_initial_points = 3
# max_opt_trials = 5

data_prep.make_dir(hp_search_hp_path)
data_prep.make_dir(hp_search_weights_path)

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

hypermodel = models.ResNet50TLHyperModel(
    n_outputs=n_outputs,
    img_shape=resnet_inp_shape,
    loss=BinaryCrossentropy(),
    weighted_metrics=[BinaryAccuracy()],
    name=hypermodel_name,
    output_act="sigmoid",
    adam_beta_1_range=adam_beta_1_range,
    adam_beta_2_range=adam_beta_2_range,
    frozen_lr_range=frozen_lr_range,
    fine_tune_lr_range=fine_tune_lr_range,
    frozen_epochs=frozen_epochs,
    fine_tune_epochs=fine_tune_epochs,
    base_model_name="base_model",
    # EarlyStopping callback parameters
    es_criterion=es_criterion,
    es_mode=es_mode,
    es_patience=es_patience,
    es_min_delta=es_min_delta,
    # Frozen ModelCheckpoint callback parameters
    mcp_criterion=mcp_criterion,
    mcp_mode=mcp_mode,
    mcp_best_frozen_weights_path=mcp_best_frozen_weights_file
)

tuner = kt.BayesianOptimization(
    hypermodel=hypermodel,
    objective="val_loss",
    num_initial_points=num_initial_points,
    max_trials=max_opt_trials,
    seed=seed,
    # directory="../model_training/",
    directory=model_training_path,
    project_name=project_name
)

# Cannot use external callbacks. Callbacks are defined inside the hypermodel's fit function
tuner.search(
    train_datagen,
    validation_data=val_datagen,
    workers=1
)
