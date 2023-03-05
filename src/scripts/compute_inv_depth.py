import os
import sys
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import dask as d
import pandas as pd
import tensorflow.keras.backend as K
import tensorflow as tf

from fl_tissue_model_tools import models, data_prep, defs
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import zstacks as zs

DEFAULT_CONFIG_PATH = str(defs.SCRIPT_CONFIG_DIR) / "default_invasion_depth_computation.json"


def main():
    args = su.parse_inv_depth_args({"default_config_path": DEFAULT_CONFIG_PATH})
    verbose = args.verbose

    ### Verify input source ###
    try:
        su.inv_depth_verify_input_dir(args.in_root, verbose=verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()

    ### Verify output destination ###
    try:
        su.inv_depth_verify_output_dir(args.out_root, verbose=verbose)
    except PermissionError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()

    ### Load best hyperparameters ###
    if verbose:
        su.verbose_header("Loading Classifier")

    best_hp_path = defs.MODEL_TRAINING_DIR / "invasion_depth_best_hp.json"
    with open(best_hp_path, 'r') as fp:
        best_hp = json.load(fp)

    ### Load model training parameters ###
    training_params_path = defs.MODEL_TRAINING_DIR / "invasion_depth_training_params.json"
    with open(training_params_path, 'r') as fp:
       training_values = json.load(fp)
    if training_values["rs_seed"] == "None":
        training_values["rs_seed"] = None

    ### Set model variables ###
    cls_thresh = training_values["cls_thresh"]
    resnet_inp_shape = tuple(training_values["resnet_inp_shape"])
    n_models = training_values["n_models"]
    n_outputs = 1
    last_resnet_layer = best_hp["last_resnet_layer"]
    descending = bool(args.order)

    ### Load config ###
    config_path = args.config
    try:
        config = su.inv_depth_verify_config_file(config_path, n_models, verbose=verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()
    n_pred_models = config["n_pred_models"]

    best_ensemble_dir = defs.MODEL_TRAINING_DIR / "best_ensemble"

    ### Create models ###
    best_val_losses = np.zeros(n_models)
    for i in range(n_models):
        out_csv = str(best_ensemble_dir / f"best_model_history_{i}.csv")
        h_df = pd.read_csv(out_csv)
        ft_h_df = h_df.query("training_stage=='finetune'")
        best_val_losses[i] = ft_h_df.val_loss.min()

    sorted_best_model_idx = best_val_losses.argsort()

    # Create the (n_pred_models) best models from the saved weights
    K.clear_session()
    inv_depth_models = [
        models.build_ResNet50_TL(n_outputs, resnet_inp_shape,
            base_last_layer=last_resnet_layer, base_model_trainable=False)
            for _ in range(n_pred_models)
    ]

    for i, m in enumerate(inv_depth_models):
        if verbose:
            print(f"Loading classifier {i}...")
        # Weights don't load properly in trainable set to False for model
        # Set trainable to True, load weights, then set back to False
        ith_best_idx = sorted_best_model_idx[i]
        m.trainable = True
        weights_path = str(best_ensemble_dir / f"best_finetune_weights_{ith_best_idx}.h5")
        m.load_weights(weights_path)
        m.trainable = False
        if verbose:
            print(f"... Classifier {i} loaded.")

    if verbose:
        print("All classifiers loaded.")
        print(su.SFM.success)
        su.verbose_footer()

    ### Generate predictions ###
    if verbose:
        su.verbose_header("Making predictions")

    try:
        zstack_dir = args.in_root
        # Load data
        zpaths = zs.zstack_paths_from_dir(zstack_dir, descending=descending)
        x = data_prep.prep_inv_depth_imgs(zpaths, resnet_inp_shape[:-1])
        # Convert to tensor before calling predict() to speed up execution
        x = tf.convert_to_tensor(x, dtype="float")

        # Make predictions
        # Probability predictions of each model
        yhatp_m = np.array(
            d.compute([d.delayed(m.predict)(x).squeeze()
                        for m in inv_depth_models])[0]
        ).T
        # Mean probability predictions (ensemble predictions)
        yhatp = np.mean(yhatp_m, axis=1, keepdims=True)
        # Threshold probability predictions
        yhat = (yhatp > cls_thresh).astype(np.int32)
        if verbose:
            print("... Predictions finished.")

        # Save outputs
        if verbose:
            print("Saving results...")
        output_file = pd.DataFrame({"img_name": [Path(zp).name for zp in zpaths],
                        "inv_prob": yhatp.squeeze(), "inv_label": yhat.squeeze()})
        out_csv_path = str(args.out_root / "invasion_depth_predictions.csv")
        output_file.to_csv(out_csv_path, index=False)
        if verbose:
            print("... Results saved.")

        if verbose:
            print(su.SFM.success)
            su.verbose_footer()

    except Exception as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


if __name__ == "__main__":
    main()
