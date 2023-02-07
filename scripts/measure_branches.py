import sys
from pathlib import Path
from typing import Tuple
import json
import csv

import numpy as np
from skimage import exposure
import cv2
from matplotlib import pyplot as plt
import networkx as nx

from fl_tissue_model_tools import helper
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import models
from fl_tissue_model_tools import defs
from fl_tissue_model_tools.transforms import filter_branch_seg_mask
from fl_tissue_model_tools.topology import MorseGraph
from fl_tissue_model_tools.analysis import pixels_to_microns

DEFAULT_CONFIG_PATH = "../config/default_branching_computation.json"


def analyze_img(img_path: Path, model: models.UNetXceptionPatchSegmentor, output_dir: Path,
                well_width_microns: int, save_intermediates: bool,
                morse_thresholds: Tuple[float, float],
                graph_smoothing_window: int,
                min_branch_length: int) -> None:
    '''Measure branches in image and save results to output directory.

    Args:
        img_path: Path to image.
        model: Model to use for segmentation.
        output_dir: Directory to save results to.
    '''

    print('')
    print("=========================================")
    print(f"Analyzing {img_path.stem}...")
    print("=========================================")

    if save_intermediates:
        vis_dir = output_dir / "visualizations" / img_path.stem
        vis_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), 0)
    img_orig = img.copy()

    print("")
    print("Segmenting image...")

    pred, mask = model.predict(img)     # pred is unthresholded
    img = exposure.equalize_adapthist(img)  # equalize

    if save_intermediates:
        save_path = str(vis_dir / "equalized.png")
        cv2.imwrite(save_path, np.round(img*255).astype(np.uint8))
        save_path = str(vis_dir / "prediction.png")
        cv2.imwrite(save_path, (pred*256).astype(np.uint8))
        save_path = str(vis_dir / "mask.png")
        cv2.imwrite(save_path, np.round(mask*255).astype(np.uint8))

    # filter out non-branching structures from mask
    mask = filter_branch_seg_mask(mask)

    if save_intermediates:
        save_path = str(vis_dir / "filtered_mask.png")
        cv2.imwrite(save_path, np.round(mask*255).astype(np.uint8))

    img[mask==0]=0      # mask out background

    if save_intermediates:
        save_path = str(vis_dir / "masked_image.png")
        cv2.imwrite(save_path, np.round(img*255).astype(np.uint8))

    img = pred * img    # weight by prediction

    if save_intermediates:
        save_path = str(vis_dir / "masked_weighted_image.png")
        cv2.imwrite(save_path, np.round(img*255).astype(np.uint8))

    img = img.astype(np.float32)

    #img = cv2.bilateralFilter(img, 15, 75, 75)  # bilateral filter
    for _ in range(10):
        img = cv2.medianBlur(img, 5)

    if save_intermediates:
        save_path = str(vis_dir / "blurred_image.png")
        cv2.imwrite(save_path, np.round(img*255).astype(np.uint8))

    img = np.round(img*255).astype(np.double)   # convert to double

    print("")
    print("Computing graph and barcode...")

    try:
        morse_graph = MorseGraph(img, thresholds=morse_thresholds,
                                 smoothing_window=graph_smoothing_window,
                                 min_branch_length=min_branch_length)
    except nx.exception.NetworkXPointlessConcept:
        print(f"No branches found for {img_path.stem}.")
        return

    if save_intermediates:
        save_path = str(vis_dir / "barcode.png")
        plt.figure(figsize=(6, 6))
        plt.margins(0)
        ax = plt.gca()
        morse_graph.plot_colored_barcode(ax=ax)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        save_path = str(vis_dir / "morse_tree.png")
        plt.figure(figsize=(10, 10))
        plt.margins(0)
        ax = plt.gca()
        ax.imshow(img_orig, cmap='gray')
        morse_graph.plot_colored_tree(ax=ax)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    print("")
    print("Computing branch statistics...")

    # Get total and average branch length
    total_branch_length = morse_graph.get_total_branch_length()
    avg_branch_length = morse_graph.get_average_branch_length()
    total_num_branches = len(morse_graph.barcode)

    total_branch_length = pixels_to_microns(total_branch_length, img.shape[0], well_width_microns)
    avg_branch_length = pixels_to_microns(avg_branch_length, img.shape[0], well_width_microns)

    # Write results to csv file
    fields = [img_path.stem, total_num_branches, total_branch_length, avg_branch_length]
    with open(output_dir / "branching_analysis.csv", "a", encoding="utf-16") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    print(f"Results saved to {output_dir / 'branching_analysis.csv'}.")


def main():
    '''Computes cell area and saves to output directory.'''

    ### Parse arguments ###
    arg_defaults = {
        "default_config_path": DEFAULT_CONFIG_PATH,
    }

    args = su.parse_branching_args(arg_defaults)

    ### Load/validate config ###

    if not Path(args.config).is_file():
        print(f"{su.SFM.failure}Config file {args.config} does not exist.")
        sys.exit()

    with open(args.config, 'r', encoding="utf8") as config_fp:
        config = json.load(config_fp)

    well_width_microns = config.get("well_width_microns", 1000.0)
    morse_thresholds = config.get("graph_thresh_1", 1), config.get("graph_thresh_2", 4)
    graph_smoothing_window = config.get("graph_smoothing_window", 15)
    min_branch_length = config.get("min_branch_length", 15)
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

    if not Path(model_cfg_path).is_file():
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
    img_paths = helper.get_img_paths(input_dir)

    if not img_paths:
        print(f"{su.SFM.failure}No images found in {input_dir}")
        sys.exit()

    ### Load model ###
    model = models.get_unet_patch_segmentor_from_cfg(model_cfg_path)

    ### Create output csv ###
    fields = ["Image", "Total # of branches", "Total branch length (µm)", "Average branch length (µm)"]
    with open(output_dir / "branching_analysis.csv", "w", encoding="utf-16") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    ### Analyze images ###
    for img_path in img_paths:
        analyze_img(Path(img_path), model, output_dir, well_width_microns, args.save_intermediates,
                    morse_thresholds, graph_smoothing_window, min_branch_length)


if __name__ == "__main__":
    main()
