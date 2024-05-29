import os
import sys
from pathlib import Path
import json
import csv

import numpy as np
import cv2
from matplotlib import pyplot as plt
from networkx.exception import NetworkXPointlessConcept as nxPointlessConceptException
from skimage.exposure import rescale_intensity
from skimage.morphology import medial_axis, disk, binary_erosion
from scipy.ndimage import distance_transform_edt

from fl_tissue_model_tools import helper, models, models_util, defs
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools.transforms import filter_branch_seg_mask
from fl_tissue_model_tools.topology import MorseGraph
from fl_tissue_model_tools.analysis import pixels_to_microns
from fl_tissue_model_tools.well_mask_generation import (
    generate_well_mask,
    gen_superellipse_mask,
)


DEFAULT_CONFIG_PATH = str(defs.SCRIPT_CONFIG_DIR / "default_branching_computation.json")


def save_vis(img, save_dir, filename):
    img = rescale_intensity(img, out_range=(0, 255))
    cv2.imwrite(os.path.join(save_dir, filename), img)


def analyze_img(
    img_path: Path,
    model: models.UNetXceptionPatchSegmentor,
    output_dir: Path,
    config: dict,
    use_well_mask: bool = False,
) -> None:
    """Measure branches in image and save results to output directory.

    Args:
        img_path (pathlib.Path): Path to image.
        model (models.UNetXceptionPatchSegmentor): Model to use for segmentation.
        output_dir (pathlib.Path): Directory to save results to.
    """

    well_width_microns = config.get("well_width_microns", 1000.0)
    morse_thresholds = config.get("graph_thresh_1", 2), config.get("graph_thresh_2", 4)
    graph_smoothing_window = config.get("graph_smoothing_window", 10)
    min_branch_length = config.get("min_branch_length", 10)

    print("")
    print("==================================   =======")
    print(f"Analyzing {img_path.stem}...")
    print("=========================================")

    img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
    img = rescale_intensity(img, out_range=(0, 1)).astype(np.float32)

    # downsample image with Lanczos interpolation
    target_shape = tuple(
        np.round(np.multiply(img.shape[:2], model.ds_ratio)).astype(int)
    )
    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_LANCZOS4)

    # Create directory for intermediate outputs and save original image
    vis_dir = output_dir / "visualizations" / img_path.stem
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_vis(img, vis_dir, "original_image.png")

    # Create a mask over the well and a smaller, inverted mask for pruning
    # The pruning mask is used to remove spurious branches detected near the well edge
    if use_well_mask:
        print("Applying mask to image...")
        well_mask = generate_well_mask(img, return_superellipse_params=True)
    else:
        well_mask = np.full_like(img, fill_value=True, dtype=np.bool_)

    if isinstance(well_mask, tuple):
        well_mask, t, d, s_a, s_b, c_x, c_y, n = well_mask
        well_mask = well_mask > 0
        # shrink superellipse for the pruning mask
        d *= 0.9
        shrunken_well_mask = gen_superellipse_mask(
            t, d, s_a, s_b, c_x, c_y, n, img.shape[:2]
        )
    else:
        # `generate_well_mask` failed to fit a superellipse
        # convert `well_mask` to a boolean mask
        well_mask = well_mask > 0
        # shrink superellipse for the pruning mask using binary erosion
        shrunken_well_mask = binary_erosion(well_mask, disk(5))
    pruning_mask = np.logical_not(shrunken_well_mask)

    print("")
    print("Segmenting image...")

    pred = model.predict(img * well_mask, auto_resample=False)
    seg_mask = pred > 0.5

    # Save pred and save well mask if needed
    save_vis(pred, vis_dir, "prediction.png")
    if use_well_mask:
        # save_vis(well_mask.astype(np.float32), vis_dir, "well_mask.png")
        # cv2.imwrite(os.path.join(save_dir, filename), img)
        cv2.imwrite(os.path.join(vis_dir, "well_mask.png"), well_mask * 255)

    # filter out non-branching structures from segmentation mask
    seg_mask = filter_branch_seg_mask(seg_mask * well_mask).astype(float)

    # Enhance centerlines
    skel, dist = medial_axis(seg_mask, return_distance=True)
    centerline_dt = distance_transform_edt(np.logical_not(skel))
    relative_dt = dist / (dist + centerline_dt)

    pred = pred * relative_dt

    # Save seg mask and distance transform result
    save_vis(seg_mask, vis_dir, "segmentation_mask.png")
    save_vis(pred, vis_dir, "distance_transform.png")

    pred = rescale_intensity(pred, out_range=(0, 255))
    pred = pred.astype(np.double)

    print("\nComputing graph and barcode...")

    try:
        morse_graph = MorseGraph(
            pred,
            thresholds=morse_thresholds,
            smoothing_window=graph_smoothing_window,
            min_branch_length=min_branch_length,
            pruning_mask=pruning_mask,
        )
    except nxPointlessConceptException:
        print(f"No branches found for {img_path.stem}.")
        return

    # Save barcode and Morse tree visualization
    save_path = str(vis_dir / "barcode.png")
    plt.figure(figsize=(6, 6))
    plt.margins(0)
    ax = plt.gca()
    morse_graph.plot_colored_barcode(ax=ax)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    save_path = str(vis_dir / "morse_tree.png")
    plt.figure(figsize=(10, 10))
    plt.margins(0)
    ax = plt.gca()
    ax.imshow(rescale_intensity(img, out_range=(0, 255)), cmap="gray")
    morse_graph.plot_colored_tree(ax=ax)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)

    print("\nComputing branch statistics...")

    # Get total and average branch length
    total_branch_length = morse_graph.get_total_branch_length()
    avg_branch_length = morse_graph.get_average_branch_length()
    total_num_branches = len(morse_graph.barcode)

    total_branch_length = pixels_to_microns(
        total_branch_length, img.shape[1], well_width_microns
    )
    avg_branch_length = pixels_to_microns(
        avg_branch_length, img.shape[1], well_width_microns
    )

    # Write results to csv file
    fields = [img_path.stem, total_num_branches, total_branch_length, avg_branch_length]
    with open(output_dir / "branching_analysis.csv", "a", encoding="utf-16") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)

    print(f"Results saved to {output_dir / 'branching_analysis.csv'}.")


def main():
    ### Parse arguments ###
    arg_defaults = {
        "default_config_path": DEFAULT_CONFIG_PATH,
    }

    args = su.parse_branching_args(arg_defaults)

    ### Load/validate config ###

    if not Path(args.config).is_file():
        print(f"{su.SFM.failure}Config file {args.config} does not exist.")
        sys.exit()

    with open(args.config, "r", encoding="utf8") as config_fp:
        config = json.load(config_fp)

    model_cfg_path = config.get("model_cfg_path", "")
    use_latest_model_cfg = config.get("use_latest_model_cfg", True)

    if use_latest_model_cfg and model_cfg_path:
        print(
            f"{su.SFM.failure}Cannot use both use_latest_model_cfg and model_cfg_path."
        )
        sys.exit()

    if not use_latest_model_cfg and not model_cfg_path:
        print(
            f"{su.SFM.failure}Must specify either use_latest_model_cfg or model_cfg_path."
        )
        sys.exit()

    if use_latest_model_cfg:
        last_exp = models_util.get_last_exp_num()
        model_cfg_dir = defs.MODEL_TRAINING_DIR / "binary_segmentation" / "configs"
        model_cfg_path = str(model_cfg_dir / f"unet_patch_segmentor_{last_exp}.json")

    if not Path(model_cfg_path).is_file():
        print(f"{su.SFM.failure}Model config file {model_cfg_path} does not exist.")
        sys.exit()

    ### Verify input and output directories ###
    input_dir = Path(args.in_root)
    if not input_dir.exists():
        print(f"{su.SFM.failure}Input directory {args.in_root} does not exist.")
        sys.exit()

    output_dir = Path(args.out_root)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Save config to output directory
    with open(output_dir / "config.json", "w", encoding="utf8") as f:
        json.dump(config, f, indent=4)

    ### Get image paths ###
    img_paths = helper.get_img_paths(input_dir)

    if not img_paths:
        print(f"{su.SFM.failure}No images found in {input_dir}")
        sys.exit()

    ### Load model ###
    model = models.get_unet_patch_segmentor_from_cfg(model_cfg_path)

    ### Create output csv ###
    fields = [
        "Image",
        "Total # of branches",
        "Total branch length (µm)",
        "Average branch length (µm)",
    ]
    with open(output_dir / "branching_analysis.csv", "w", encoding="utf-16") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)

    ### Analyze images ###
    for img_path in img_paths:
        analyze_img(Path(img_path), model, output_dir, config, args.detect_well)


if __name__ == "__main__":
    main()
