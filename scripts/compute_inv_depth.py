import os
import sys
import json
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import dask as d
import pandas as pd

from fl_tissue_model_tools import models, data_prep
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import zstacks as zs


def main():
    args = su.parse_inv_depth_args({})
    verbose = args.verbose


    ### Tidy up paths ###
    in_root = args.in_root.replace("\\", "/")
    out_root = args.out_root.replace("\\", "/")


    ### Verify input source ###
    try:
        zstack_paths = su.inv_depth_verify_input_dir(in_root, verbose=verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


    ### Verify output destination ###
    try:
        su.inv_depth_verify_output_dir(out_root, verbose=verbose)
    except PermissionError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()

    
    ### Load best hyperparameters ###
    if verbose:
        su.verbose_header("Loading Classifier")

    with open("../model_training/invasion_depth_best_hp.json", 'r') as fp:
        best_hp = json.load(fp)
    

    ### Load model config ###
    with open("../model_training/invasion_depth_training_values.json", 'r') as fp:
       training_values = json.load(fp)
    training_values["rs_seed"] = None if (training_values["rs_seed"] == "None") else training_values["rs_seed"]

    
    ### Set model variables ###
    cls_thresh = training_values["cls_thresh"]
    resnet_inp_shape = tuple(training_values["resnet_inp_shape"])
    class_labels = training_values["class_labels"]
    n_models = training_values["n_models"]
    n_outputs = 1
    last_resnet_layer = best_hp["last_resnet_layer"]
    descending = bool(args.order)


    # Create models
    K.clear_session()
    inv_depth_models = [
        models.build_ResNet50_TL(n_outputs, resnet_inp_shape, base_last_layer=last_resnet_layer, base_model_trainable=False) for _ in range(n_models)
    ]
    for i, m in enumerate(inv_depth_models):
        if verbose:
            print(f"Loading classifier {i}...")
        # Weights don't load properly in trainable set to False for model
        # Set trainable to True, load weights, then set back to False
        m.trainable = True
        m.load_weights(f"../model_training/best_ensemble/best_finetune_weights_{i}.h5")
        m.trainable = False
        if verbose:
            print(f"... Classifier {i} loaded.")

    if verbose:
        print(f"All classifiers loaded.")
        print(su.SFM.success)
        su.verbose_footer()

    # Generate predictions
    if verbose:
        su.verbose_header("Making predictions")

    try:
        for zstack_dir in zstack_paths:
            # Load data
            zstack_id = zstack_dir.split("/")[-1]
            zpaths = zs.zstack_paths_from_dir(zstack_dir, descending=descending)

            if verbose:
                print(f"Making predictions for Z stack:{os.linesep}\t{zstack_id}{os.linesep}...")

            x = data_prep.prep_inv_depth_imgs(zpaths, resnet_inp_shape[:-1])
            # Convert to tensor before calling predict() to speed up execution
            x = tf.convert_to_tensor(x, dtype="float")

            # Make predictions
            # Probability predictions of each model
            yhatp_m = np.array(
                d.compute([d.delayed(m.predict)(x).squeeze() for m in inv_depth_models])[0]
            ).T
            # Mean probability predictions (ensemble predictions)
            yhatp = np.mean(yhatp_m, axis=1, keepdims=True)
            # Threshold probability predictions
            yhat = (yhatp > cls_thresh).astype(np.int32)
            if verbose:
                print(f"... Predictions finished.")
            
            # Save outputs
            if verbose:
                print(f"Saving results...")
            output_file = pd.DataFrame({"img_name": [zp.split("/")[-1] for zp in zpaths], "inv_prob": yhatp.squeeze(), "inv_label": yhat.squeeze()})
            output_file.to_csv(f"{out_root}/{zstack_id}.csv", index=False)
            if verbose:
                print(f"... Results saved.")

        if verbose:
            print(su.SFM.success)
            su.verbose_footer()
    except Exception as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


if __name__ == "__main__":
    main()