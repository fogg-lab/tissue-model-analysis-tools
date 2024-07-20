import os
import sys
import json
from glob import glob

import numpy as np
import pandas as pd
import silence_tensorflow.auto  # noqa
import tensorflow as tf
import tensorflow.keras.backend as K

from fl_tissue_model_tools import models, data_prep, defs, helper
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools.success_fail_messages import SFM
from fl_tissue_model_tools import zstacks as zs

DEFAULT_CONFIG_PATH = str(
    defs.SCRIPT_CONFIG_DIR / "default_invasion_depth_computation.json"
)


def main(args=None):
    if args is None:
        args = su.parse_inv_depth_args({"default_config_path": DEFAULT_CONFIG_PATH})
        args_prespecified = False
    else:
        args_prespecified = True

    ### Verify input source ###
    if os.path.isfile(args.in_root):
        print(f"{SFM.failure} Input directory is a file: {args.in_root}", flush=True)
        sys.exit(1)

    if not os.path.isdir(args.in_root):
        print(
            f"{SFM.failure} Input directory does not exist: {args.in_root}",
            flush=True,
        )
        sys.exit(1)

    zstack_paths = glob(os.path.join(args.in_root, "*"))

    if len(zstack_paths) == 0:
        print(f"{SFM.failure} Input directory is empty: {args.in_root}", flush=True)
        sys.exit(1)

    ### Verify output destination ###
    try:
        su.inv_depth_verify_output_dir(args.out_root)
    except PermissionError as e:
        print(f"{SFM.failure} {e}", flush=True)
        sys.exit(1)

    ### Load best hyperparameters ###
    su.section_header("Loading Classifier")

    best_hp_path = defs.MODEL_TRAINING_DIR / "invasion_depth_best_hp.json"
    with open(best_hp_path, "r") as fp:
        best_hp = json.load(fp)

    ### Load model training parameters ###
    training_params_path = (
        defs.MODEL_TRAINING_DIR / "invasion_depth_training_values.json"
    )
    with open(training_params_path, "r") as fp:
        training_values = json.load(fp)
    if training_values["rs_seed"] == "None":
        training_values["rs_seed"] = None

    ### Set model variables ###
    cls_thresh = training_values["cls_thresh"]
    resnet_inp_shape = tuple(training_values["resnet_inp_shape"])
    n_models = training_values["n_models"]
    n_outputs = 1
    last_resnet_layer = best_hp["last_resnet_layer"]

    ### Load config ###
    config_path = DEFAULT_CONFIG_PATH if args_prespecified else args.config
    try:
        config = su.inv_depth_verify_config_file(config_path, n_models)
    except FileNotFoundError as e:
        print(f"{SFM.failure} {e}", flush=True)
        sys.exit(1)
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
        models.build_ResNet50_TL(
            n_outputs,
            resnet_inp_shape,
            base_init_weights=None,
            base_last_layer=last_resnet_layer,
            base_model_trainable=False,
        )
        for _ in range(n_pred_models)
    ]

    for i, m in enumerate(inv_depth_models):
        print(f"Loading classifier {i}...", flush=True)
        # Weights don't load properly in trainable set to False for model
        # Set trainable to True, load weights, then set back to False
        ith_best_idx = sorted_best_model_idx[i]
        m.trainable = True
        weights_path = str(
            best_ensemble_dir / f"best_finetune_weights_{ith_best_idx}.h5"
        )
        m.load_weights(weights_path)
        m.trainable = False
        print(f"... Classifier {i} loaded.", flush=True)

    print("All classifiers loaded.", flush=True)
    print(SFM.success, flush=True)
    su.section_footer()

    ### Generate predictions ###
    su.section_header("Making predictions")

    # Load data

    test_path = zstack_paths[0]
    if os.path.isdir(test_path) or helper.get_image_dims(test_path).Z == 1:
        zstack_paths = zs.find_zstack_image_sequences(args.in_root)
    else:
        zstack_paths = zs.find_zstack_files(args.in_root)

    inv_id_col = "Z Slice ID"
    inv_prob_col = "Invasion Probability"
    inv_pred_col = "Invasion Prediction (0=no 1=yes)"

    predictions = {
        inv_prob_col: [],
        inv_pred_col: [],
    }
    z_slice_ids = []

    for zstack_id, zstack_path in zstack_paths.items():
        # zstack_path might also be a list of paths (z stack from image sequence)
        print(f"Processing {zstack_id}...", flush=True)
        try:
            img, _ = helper.load_image(zstack_path, args.time, args.channel)
        except OSError as error:
            print(f"{SFM.failure}{error}", flush=True)
            sys.exit(1)
        x = data_prep.prep_inv_depth_imgs(img, resnet_inp_shape[:-1])
        x = tf.convert_to_tensor(x, dtype="float")
        # Make predictions
        yhatp_m = np.array([m.predict(x).squeeze() for m in inv_depth_models]).T
        # Mean probability predictions (ensemble predictions)
        yhatp = np.mean(yhatp_m, axis=1, keepdims=True)
        for z in range(len(yhatp)):
            slice_id = f"{zstack_id}_z{z}"
            # yhatp[z] is either a float or a numpy array with one element
            inv_prob = np.atleast_1d(yhatp[z])[0]
            inv_prob = round(inv_prob, 4)
            inv_label = int(inv_prob > cls_thresh)
            z_slice_ids.append(slice_id)
            predictions[inv_prob_col].append(inv_prob)
            predictions[inv_pred_col].append(inv_label)

    results = pd.DataFrame(predictions, index=pd.Index(z_slice_ids, name=inv_id_col))

    # Save outputs
    print("Saving results...", flush=True)
    out_csv_path = os.path.join(args.out_root, "invasion_depth_predictions.csv")
    out_csv_path = helper.get_unique_output_filepath(out_csv_path)
    results.to_csv(out_csv_path)
    print("... Results saved.", flush=True)

    print(SFM.success, flush=True)
    su.section_footer()


if __name__ == "__main__":
    main()
