import sys
from pathlib import Path
import json

import numpy as np
from skimage import measure, morphology, feature, exposure, filters
import cv2

from fl_tissue_model_tools import helper
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import models
from fl_tissue_model_tools import defs
from fl_tissue_model_tools.topology import (
    compute_nx_graph,
    compute_barcode,
    compute_colored_tree_and_barcode,
    plot_colored_tree,
    filter_graph,
    smooth_graph
)

DEFAULT_CONFIG_PATH = "../config/default_branching_computation.json"


def analyze_img(img_path: Path, model: models.UNetXceptionPatchSegmentor, output_dir: Path,
                well_width_microns: int, save_intermediates: bool) -> None:
    '''Measure branches in image and save results to output directory.

    Args:
        img_path: Path to image.
        model: Model to use for segmentation.
        output_dir: Directory to save results to.
    '''

    img_base_name, img_ext = img_path.stem, img_path.suffix

    img = cv2.imread(str(img_path), 0)
    pred, mask = model.predict(img)     # pred is unthresholded
    img = exposure.equalize_adapthist(img)  # equalize

    if save_intermediates:
        cv2.imwrite(str(output_dir / f"{img_base_name}_equalized.{img_ext}"), img*255)

    img[mask==0]=0  # mask out background
    # img = pred * img    # weight by prediction

    if save_intermediates:
        cv2.imwrite(str(output_dir / f"{img_base_name}_masked.{img_ext}"), img*255)

    img = cv2.bilateralFilter(img, 15, 75, 75)  # bilateral filter
    # img = cv2.GaussianBlur(img, (5, 5), 0)  # gaussian blur

    if save_intermediates:
        cv2.imwrite(str(output_dir / f"{img_base_name}_blurred.{img_ext}"), img*255)

    img = (img*256).astype(np.double)   # convert to double


def main():
    '''Computes cell area and saves to output directory.'''

    ### Parse arguments ###
    arg_defaults = {
        "default_config_path": DEFAULT_CONFIG_PATH,
    }

    args = su.parse_cell_area_args(arg_defaults)

    ### Load/validate config ###

    if not Path(args.config).is_file():
        print(f"{su.SFM.failure}Config file {args.config} does not exist.")
        sys.exit()

    config = json.load(args.config)
    well_width_microns = config.get("well_width_microns", 1000.0)
    model_cfg_path = config.get("model_cfg_path", "")
    use_latest_model_cfg = config.get("use_latest_model_cfg", True)

    if use_latest_model_cfg and model_cfg_path:
        print(f"{su.SFM.failure}Cannot use both use_latest_model_cfg and model_cfg_path.")
        sys.exit()

    if not use_latest_model_cfg and not model_cfg_path:
        print(f"{su.SFM.failure}Must specify either use_latest_model_cfg or model_cfg_path.")
        sys.exit()

    if use_latest_model_cfg:
        last_exp = models.get_last_exp_num()
        models_dir = Path(defs.SCRIPT_REL_MODEL_TRAINING_PATH)
        model_cfg_dir = models_dir / "binary_segmentation" / "configs"
        model_cfg_path = str(model_cfg_dir / f"unet_patch_segmentor_{last_exp}.json")

    if not model_cfg_path.is_file():
        print(f"{su.SFM.failure}Model config file {model_cfg_path} does not exist.")
        sys.exit()

    ### Verify input and output directories ###
    input_dir = Path(args.in_root)
    if not input_dir.exists():
        print(f"{su.SFM.failure}Input directory {args.input_dir} does not exist.")
        sys.exit()

    output_dir = Path(args.out_root)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    ### Get image paths ###
    img_paths = helper.get_img_paths(args.input_dir)

    if not img_paths:
        print(f"{su.SFM.failure}No images found in {args.input_dir}")
        sys.exit()

    ### Load model ###
    model = models.get_unet_patch_segmentor_from_cfg(args.model_cfg)

    ### Analyze images ###
    for img_path in img_paths:
        analyze_img(Path(img_path), model, output_dir, well_width_microns, args.save_intermediates)


if __name__ == "__main__":
    main()
