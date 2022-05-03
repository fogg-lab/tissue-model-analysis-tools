#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import dask as d
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
import json
import multiprocessing
import os

from glob import glob
from copy import deepcopy
from tensorflow import data
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import resnet50

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


from fl_tissue_model_tools import data_prep, dev_config, models, defs
import fl_tissue_model_tools.preprocessing as prep


# In[3]:


n_cores = multiprocessing.cpu_count()
n_cores


# In[4]:


dirs = dev_config.get_dev_directories("../dev_paths.txt")


# # Set up model training parameters

# In[5]:


with open("../model_training/invasion_depth_training_values.json", 'r') as fp:
       training_values = json.load(fp)
training_values["rs_seed"] = None if (training_values["rs_seed"] == "None") else training_values["rs_seed"]


# In[6]:


training_values


# In[7]:


with open("../model_training/invasion_depth_hp_space.json", 'r') as fp:
       hp_search_space = json.load(fp)


# In[8]:


hp_search_space


# In[10]:


root_data_path = f"{dirs.data_dir}/invasion_data/development"
model_training_path = f"{dirs.analysis_dir}/resnet50_invasion_model"
project_name = "invasion_hp_trials"
hypermodel_name = "invasion_depth_hypermodel"
hp_search_hp_path = f"{model_training_path}/hyperparameter_search_hps"
hp_search_weights_path = f"{model_training_path}/hyperparameter_search_weights"
best_hp_file = f"{hp_search_hp_path}/best_hyperparams_v1.json"
mcp_best_frozen_weights_file = f"{hp_search_weights_path}/best_frozen_weights.h5"


### General training parameters ###
resnet_inp_shape = (128, 128, 3)
# Binary classification -> only need 1 output unit
n_outputs = 1
seed = training_values["rs_seed"]
val_split = training_values["val_split"]
batch_size = training_values["batch_size"]
frozen_epochs = training_values["frozen_epochs"]
fine_tune_epochs = training_values["fine_tune_epochs"]


### Early stopping ###
es_criterion = "val_loss"
es_mode = "min"
# Update these depending on seriousness of experiment
es_patience = 25
es_min_delta = 0.0001


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
class_labels = {"no_invasion": 0, "invasion": 1}


# In[11]:


data_prep.make_dir(hp_search_hp_path)
data_prep.make_dir(hp_search_weights_path)


# # Prep for loading data

# In[12]:


rs = np.random.RandomState(seed)


# In[13]:


data_paths = {v: glob(f"{root_data_path}/train/{k}/*.tif") for k, v in class_labels.items()}
for k, v in data_paths.items():
    rs.shuffle(v)


# In[14]:


data_counts = {k: len(v) for k, v in data_paths.items()}
val_counts = {k: round(v * val_split) for k, v in data_counts.items()}
train_counts = {k: v - val_counts[k] for k, v in data_counts.items()}
train_class_weights = prep.balanced_class_weights_from_counts(train_counts)


# In[15]:


data_counts


# In[16]:


train_class_weights


# In[17]:


train_data_paths = {k: v[val_counts[k]:] for k, v in data_paths.items()}
val_data_paths = {k: v[:val_counts[k]] for k, v in data_paths.items()}


# # Datasets

# In[18]:


# With class weights
train_datagen = data_prep.InvasionDataGenerator(train_data_paths, class_labels, batch_size, resnet_inp_shape[:2], rs, class_weights=train_class_weights, augmentation_function=prep.augment_imgs)
val_datagen = data_prep.InvasionDataGenerator(val_data_paths, class_labels, batch_size, resnet_inp_shape[:2], rs, class_weights=train_class_weights, augmentation_function=prep.augment_imgs)
# # Without class weights
# train_datagen = InvasionDataGenerator(train_data_paths, class_labels, batch_size, resnet_inp_shape[:2], rs, augmentation_function=prep.augment_imgs)
# val_datagen = InvasionDataGenerator(val_data_paths, class_labels, batch_size, resnet_inp_shape[:2], rs, augmentation_function=prep.augment_imgs)


# # Build hyper model

# In[19]:


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


# In[21]:


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


# In[ ]:


# Cannot use external callbacks. Callbacks are defined inside the hypermodel's fit function
tuner.search(
    train_datagen,
    validation_data=val_datagen,
    workers=n_cores
)


# In[ ]:


tuner.results_summary()


# In[ ]:


best_hp = tuner.get_best_hyperparameters()[0]


# In[ ]:


best_hp.values


# In[ ]:


# with open("../model_training/best_hp_values.json", "w") as fp:
with open(best_hp_file, "w") as fp:
    json.dump(best_hp.values, fp, sort_keys=True)


# In[ ]:


best_tl_model = tuner.get_best_models()[0]


# In[ ]:


best_tl_model.summary()


# In[ ]:




