import os
import sys
import json
from pathlib import Path
from glob import glob

import numpy as np
import dask as d
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)
import tensorflow.keras.backend as K

from fl_tissue_model_tools import models, data_prep, defs, helper
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import zstacks as zs

DEFAULT_CONFIG_PATH = str(defs.SCRIPT_CONFIG_DIR / "default_invasion_depth_computation.json")


def main(args=None):
    if args is None:
        args = su.parse_inv_depth_args({"default_config_path": DEFAULT_CONFIG_PATH})
        args_prespecified = False
    else:
        args_prespecified = True

    ### Verify input source ###
    if os.path.isfile(args.in_root):
        print(f"{su.SFM.failure} Input directory is a file: {args.in_root}", flush=True)
        sys.exit(1)

    if not os.path.isdir(args.in_root):
        print(f"{su.SFM.failure} Input directory does not exist: {args.in_root}", flush=True)
        sys.exit(1)

    zstack_paths = glob(os.path.join(args.in_root, "*"))

    if len(zstack_paths) == 0:
        print(f"{su.SFM.failure} Input directory is empty: {args.in_root}", flush=True)
        sys.exit(1)

    ### Verify output destination ###
    try:
        su.inv_depth_verify_output_dir(args.out_root)
    except PermissionError as e:
        print(f"{su.SFM.failure} {e}", flush=True)
        sys.exit()

    ### Load best hyperparameters ###
    su.section_header("Loading Classifier")

    best_hp_path = defs.MODEL_TRAINING_DIR / "invasion_depth_best_hp.json"
    with open(best_hp_path, 'r') as fp:
        best_hp = json.load(fp)

    ### Load model training parameters ###
    training_params_path = defs.MODEL_TRAINING_DIR / "invasion_depth_training_values.json"
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

    ### Load config ###
    config_path = DEFAULT_CONFIG_PATH if args_prespecified else args.config
    try:
        config = su.inv_depth_verify_config_file(config_path, n_models)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}", flush=True)
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
        models.build_ResNet50_TL(n_outputs, resnet_inp_shape, base_init_weights=None,
            base_last_layer=last_resnet_layer, base_model_trainable=False)
            for _ in range(n_pred_models)
    ]

    for i, m in enumerate(inv_depth_models):
        print(f"Loading classifier {i}...", flush=True)
        # Weights don't load properly in trainable set to False for model
        # Set trainable to True, load weights, then set back to False
        ith_best_idx = sorted_best_model_idx[i]
        m.trainable = True
        weights_path = str(best_ensemble_dir / f"best_finetune_weights_{ith_best_idx}.h5")
        m.load_weights(weights_path)
        m.trainable = False
        print(f"... Classifier {i} loaded.", flush=True)

    print("All classifiers loaded.", flush=True)
    print(su.SFM.success, flush=True)
    su.section_footer()

    ### Generate predictions ###
    su.section_header("Making predictions")

    # Load data

    test_path = zstack_paths[0]
    if os.path.isdir(test_path) or helper.get_image_dims(test_path).Z > 1:
        zstack_paths = zs.find_zstack_image_sequences(args.in_root)
    else:
        zstack_paths = zs.find_zstack_files(args.in_root)

    images = []
    image_names = []
    for zsp in zstack_paths.values():
        if isinstance(zsp, list):
            for img_path in zsp:
                images.append(helper.load_image(img_path, args.time, args.channel))
                image_names.append(Path(img_path).stem)
        else:
            image = helper.load_image(zsp, args.time, args.channel)
            if image.ndim == 2:
                images.append(image)
                image_names.append(Path(zsp).stem)
            else:
                for z in len(images):
                    images.append(images[z])
                    image_names.append(f"{Path(zsp).stem}_z{z}")

    x = data_prep.prep_inv_depth_imgs(images, resnet_inp_shape[:-1])
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
    print("... Predictions finished.", flush=True)

    # Save outputs
    print("Saving results...", flush=True)
    output_file = pd.DataFrame({"img_name": image_names,
                    "inv_prob": yhatp.squeeze(), "inv_label": yhat.squeeze()})
    out_csv_path = os.path.join(args.out_root, "invasion_depth_predictions.csv")
    output_file.to_csv(out_csv_path, index=False)
    print("... Results saved.", flush=True)

    print(su.SFM.success, flush=True)
    su.section_footer()


if __name__ == "__main__":
    main()
