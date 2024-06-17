import os
import sys
from pathlib import Path
from glob import glob
import json
import csv
from itertools import product

import numpy as np
import cv2
from matplotlib import pyplot as plt
from networkx.exception import NetworkXPointlessConcept as nxPointlessConceptException
from skimage.exposure import rescale_intensity
from skimage.feature import canny
from skimage.filters import gaussian, sato, unsharp_mask
from skimage.transform import resize
from skimage.morphology import medial_axis, disk, binary_erosion, dilation, square, closing
from scipy.ndimage import distance_transform_edt

from fl_tissue_model_tools import helper, models, models_util, defs
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools.transforms import filter_branch_seg_mask, regionprops_image
from fl_tissue_model_tools.topology import MorseGraph
from fl_tissue_model_tools.well_mask_generation import (
    generate_well_mask,
    gen_superellipse_mask,
)


DEFAULT_CONFIG_PATH = str(defs.SCRIPT_CONFIG_DIR / "default_branching_computation.json")
DOWNSAMPLE_WIDTH = 384


def create_output_csv(output_file: Path):
    """Create output csv file for branch analysis results.

    Args:
        output_file (pathlib.Path): Path to output csv file.
    """

    fields = [
        "Image",
        "Total # of branches",
        "Total branch length (µm)",
        "Average branch length (µm)",
    ]
    with open(output_file, "w", encoding="utf-16") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)


def save_vis(img, save_dir, filename):
    img = rescale_intensity(img, out_range=(0, 255))
    cv2.imwrite(os.path.join(save_dir, filename), img)


def pixels_to_microns(
    num_pixels: float, im_width_px: int, im_width_microns: float
) -> float:
    """Convert pixels to microns in specified resolution.

    Args:
        num_pixels: Number of pixels to convert to microns.
        im_width_px: Width of image in pixels.
        im_width_microns: Physical width of image region in microns.
    """

    return (im_width_microns / im_width_px) * num_pixels


def microns_to_pixels(
    num_microns: float, im_width_px: int, im_width_microns: float
) -> float:
    """Convert microns to pixels in specified resolution.

    Args:
        num_microns: Number of microns to convert to pixels.
        im_width_px: Width of image in pixels.
        im_width_microns: Physical width of image region in microns.
    """

    return (im_width_px / im_width_microns) * num_microns


def make_well_mask(img: np.ndarray, use_well_mask: bool):
    """
    Create a mask over the well and a smaller, inverted mask for pruning
    The pruning mask is used to remove spurious branches detected near the well edge
    """

    if use_well_mask:
        print("Applying mask to image...", flush=True)
        well_mask = generate_well_mask(img, return_superellipse_params=True)
    else:
        well_mask = np.full_like(img, fill_value=True, dtype=bool)

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

    return well_mask, shrunken_well_mask


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

    image_width_microns = config.get("image_width_microns")
    graph_thresh_1 = config.get("graph_thresh_1", 5)
    graph_thresh_2 = config.get("graph_thresh_2", 10)
    graph_smoothing_window = config.get("graph_smoothing_window", 12)
    min_branch_length = config.get("min_branch_length", 12)
    max_branch_length = config.get("max_branch_length")
    remove_isolated_branches = config.get("remove_isolated_branches", False)
    time_index = config.get("time")
    channel_index = config.get("channel")

    print("", flush=True)
    print("=========================================", flush=True)
    print(f"Analyzing {img_path.stem}...", flush=True)
    print("=========================================", flush=True)

    img, pix_sizes = helper.load_image(img_path, time_index, channel_index)
    n_dims = img.ndim

    if image_width_microns is None:
        # Use pixel size from image metadata if available
        if pix_sizes.X is None:
            print(f"{su.SFM.warning} image_width_microns not provided in the config, "
                  "and could not be inferred from the image metadata. "
                  "using arbitrary value of 1000 microns.")
            image_width_microns = 1000
        else:
            image_width_microns = img.shape[-1] * pix_sizes.X

    # Create directory for intermediate outputs
    vis_dir = output_dir / "visualizations" / img_path.stem
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_dsamp_res = (
        np.multiply(img.shape[-2:], DOWNSAMPLE_WIDTH / img.shape[-1])
        .round()
        .astype(int)
    )

    if n_dims == 3:
        ### Z stack. Apply Frangi vesselness filter and post-process it.

        # Store max-projection of original image in visualizations directory
        original_image = img.max(0)
        save_vis(original_image, vis_dir, "original_image.png")

        # Apply mild gaussian filter and downsample
        for i in range(len(img)):
            img[i, :] = gaussian(img[i], sigma=1, preserve_range=True)
        img = resize(
            img,
            (len(img), *img_dsamp_res),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        img = rescale_intensity(img, out_range=(0, 1)).astype(np.float32)

        # Vesselness filter with the Sato method
        img_vess = np.zeros_like(img, shape=(len(img) - 1, *img.shape[1:]))
        print("Processing slices...", flush=True)
        for z in range(len(img) - 1):
            im = np.maximum(img[z], img[z + 1])
            img_vess[z] = sato(
                im, sigmas=[1, 2, 3, 4, 5, 7, 9, 11, 13, 15], black_ridges=False
            )
            print(".", end="", flush=True)
        print("", flush=True)

        # Sharpen, max-project and find edges
        img_vess_sharp = unsharp_mask(img_vess, 2, 2)
        vessels = img_vess_sharp.max(0)
        edges = canny(vessels, sigma=0)

        # Mask vessels by flooding in from edges towards positive intensity gradient
        mask = medial_axis(edges)

        eccentricity = regionprops_image(mask, "eccentricity")
        circ_diam = regionprops_image(mask, "equivalent_diameter_area")

        mask = np.where(eccentricity * circ_diam > 3.5, mask, 0)

        edge_blur_iters = 3
        for _ in range(edge_blur_iters):
            vessels_blur = gaussian(vessels)
            vessels = np.where(mask, vessels_blur, vessels)

        region_expansion_iters = 10
        slice_by_offset = {-1: slice(1, None), 0: slice(None, None), 1: slice(None, -1)}
        for _ in range(region_expansion_iters):
            mask_lo = np.zeros_like(mask)
            mask_hi = np.zeros_like(mask)
            for r, c in (p for p in product((-1, 0, 1), repeat=2) if p != (0, 0)):
                src = slice_by_offset[r], slice_by_offset[c]
                dst = slice_by_offset[-r], slice_by_offset[-c]
                dst_lt_src = vessels[dst] < vessels[src]
                mask_lo[dst] = np.where(mask[src] & dst_lt_src, 1, mask_lo[dst])
                mask_hi[dst] = np.where(mask[src] & ~dst_lt_src, 1, mask_hi[dst])
            mask |= (vessels > 0.01) & mask_hi & ~mask_lo

        mask &= ~edges
        vessels_mask = closing(mask, disk(2))

        vessels_mask = filter_branch_seg_mask(vessels_mask, None, False)

        vessels = np.where(dilation(vessels_mask, square(3)), img_vess_sharp.max(0), 0)
        img = gaussian(vessels)

        well_mask, shrunken_well_mask = make_well_mask(img, use_well_mask)
        pruning_mask = np.logical_not(shrunken_well_mask)

    else:
        ### 2D image. Mask vessels using binary segmentation model and post-process.
        target_shape = tuple(
            np.round(np.multiply(img.shape[:2], model.ds_ratio)).astype(int)
        )
        img = cv2.resize(img, target_shape, interpolation=cv2.INTER_LANCZOS4)
        # Store original image in visualizations directory
        original_image = img.copy()
        save_vis(original_image, vis_dir, "original_image.png")
        img = rescale_intensity(img, out_range=(0, 1)).astype(np.float32)

        well_mask, shrunken_well_mask = make_well_mask(img, use_well_mask)
        pruning_mask = np.logical_not(shrunken_well_mask)

        print("", flush=True)
        print("Segmenting image...", flush=True)

        pred = model.predict(img * well_mask, auto_resample=False)

        # Save pred and save well mask if needed
        save_vis(pred, vis_dir, "prediction.png")

        # Apply vessel probability threshold to get segmentation mask
        seg_mask = pred > 0.5

        # Filter out non-branching structures from segmentation mask
        seg_mask = filter_branch_seg_mask(seg_mask * well_mask).astype(float)

        # Enhance centerlines
        skel, dist = medial_axis(seg_mask, return_distance=True)
        centerline_dt = distance_transform_edt(np.logical_not(skel))
        relative_dt = dist / (dist + centerline_dt)

        pred = pred * relative_dt

        # Save seg mask and distance transform result
        save_vis(seg_mask, vis_dir, f"segmentation_mask.png")
        save_vis(pred, vis_dir, f"distance_transform.png")

        # Downsample vessel prediction for Morse skeletonization
        pred = resize(
            pred,
            img_dsamp_res,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        pruning_mask = resize(
            pruning_mask, img_dsamp_res, order=0, preserve_range=True
        ).astype(bool)

    if use_well_mask:
        cv2.imwrite(os.path.join(vis_dir, "well_mask.png"), well_mask * 255)

    embed_graph_params = {
        "thresh1": np.atleast_1d(graph_thresh_1).tolist(),
        "thresh2": np.atleast_1d(graph_thresh_2).tolist(),
    }
    param_names, param_vals = zip(*embed_graph_params.items())
    param_combinations = product(*param_vals)
    cfgs = [dict(zip(param_names, comb)) for comb in param_combinations]
    tuned = [k for k, v in embed_graph_params.items() if len(v) > 1]

    # Define zero padding for numeric parameters for file naming
    param_str_fmts = {}
    for k, v in embed_graph_params.items():
        if all(isinstance(x, (int, float)) for x in v):
            if all(isinstance(x, int) for x in v):
                fixed_width = max(len(str(x)) for x in v)
                param_str_fmts[k] = f"{{:0{fixed_width}d}}"
            else:
                width_left = max(str(float(x)).find(".") for x in v)
                width_right = max(len(str(float(x)).split(".")[1]) for x in v)
                fixed_width = width_left + 1 + width_right  # +1 for decimal point
                param_str_fmts[k] = f"{{:0{fixed_width}.{width_right}f}}"
        else:
            param_str_fmts[k] = "{}"

    # For each configuration, construct and analyze embedded graph
    for cfg in cfgs:
        tuned_str = "".join(
            f"_{k}_{param_str_fmts[k].format(v)}" for k, v in cfg.items() if k in tuned
        )
        tuned_str = f"_CONFIG{tuned_str}" if tuned_str else ""

        if n_dims == 2:
            img = pred
            print("\nComputing graph and barcode...", flush=True)

        min_branch_length_px = microns_to_pixels(
            min_branch_length, img.shape[1], image_width_microns
        )
        min_branch_length_px = round(min_branch_length_px)
        max_branch_length_px = (
            None
            if max_branch_length is None
            else microns_to_pixels(max_branch_length, img.shape[1], image_width_microns)
        )
        if max_branch_length_px is not None:
            max_branch_length_px = round(max(1, max_branch_length_px))
        smoothing_window_px = microns_to_pixels(
            graph_smoothing_window, img.shape[1], image_width_microns
        )
        smoothing_window_px = round(max(1, smoothing_window_px))

        try:
            morse_graph = MorseGraph(
                rescale_intensity(img, out_range=(0, 255)),
                thresholds=(cfg["thresh1"], cfg["thresh2"]),
                smoothing_window=smoothing_window_px,
                min_branch_length=min_branch_length_px,
                max_branch_length=max_branch_length_px,
                remove_isolated_branches=remove_isolated_branches,
                pruning_mask=pruning_mask,
            )
        except nxPointlessConceptException:
            print(f"No branches found for {img_path.stem}.", flush=True)
            return

        # Save barcode and Morse tree visualization
        save_path = str(vis_dir / f"barcode{tuned_str}.png")
        plt.figure(figsize=(6, 6))
        plt.margins(0)
        ax = plt.gca()
        scaling_factor = original_image.shape[1] / img_dsamp_res[1]
        morse_graph.plot_colored_barcode(scaling_factor=scaling_factor, ax=ax)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        save_path = str(vis_dir / f"morse_tree{tuned_str}.png")
        fig_width = 10
        fig_height = fig_width * (original_image.shape[0] / original_image.shape[1])
        plt.figure(figsize=(fig_width, fig_height))
        plt.margins(0)
        ax = plt.gca()
        ax.imshow(rescale_intensity(original_image, out_range=(0, 255)), cmap="gray")
        morse_graph.plot_colored_tree(scaling_factor=scaling_factor, ax=ax)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close("all")

        print("\nComputing branch statistics...", flush=True)

        # Get total and average branch length
        total_branch_length = morse_graph.get_total_branch_length()
        avg_branch_length = morse_graph.get_average_branch_length()
        total_num_branches = len(morse_graph.barcode)

        total_branch_length = pixels_to_microns(
            total_branch_length, img.shape[1], image_width_microns
        )
        avg_branch_length = pixels_to_microns(
            avg_branch_length, img.shape[1], image_width_microns
        )

        # Write results to csv file
        fields = [
            img_path.stem,
            total_num_branches,
            total_branch_length,
            avg_branch_length,
        ]
        output_file = output_dir / f"branching_analysis{tuned_str}.csv"
        if not output_file.is_file():
            create_output_csv(output_file)
        with open(output_file, "a", encoding="utf-16") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(fields)

        print(f"Results saved to {output_file}.", flush=True)


def main(args=None):
    ### Parse arguments ###
    arg_defaults = {
        "default_config_path": DEFAULT_CONFIG_PATH,
    }

    if args is None:
        args = su.parse_branching_args(arg_defaults)
        ### Load/validate config ###
        if not Path(args.config).is_file():
            print(
                f"{su.SFM.failure}Config file {args.config} does not exist.", flush=True
            )
            sys.exit()
        with open(args.config, "r", encoding="utf8") as config_fp:
            config = json.load(config_fp)
    else:
        config = {}
        config["image_width_microns"] = args.image_width_microns
        config["graph_thresh_1"] = args.graph_thresh_1
        config["graph_thresh_2"] = args.graph_thresh_2
        config["graph_smoothing_window"] = args.graph_smoothing_window
        config["min_branch_length"] = args.min_branch_length
        config["remove_isolated_branches"] = args.remove_isolated_branches

    model_cfg_path = config.get("model_cfg_path")

    if not model_cfg_path:
        last_exp = models_util.get_last_exp_num()
        model_cfg_dir = defs.MODEL_TRAINING_DIR / "binary_segmentation" / "configs"
        model_cfg_path = str(model_cfg_dir / f"unet_patch_segmentor_{last_exp}.json")

    if not Path(model_cfg_path).is_file():
        print(
            f"{su.SFM.failure}Model config file {model_cfg_path} does not exist.",
            flush=True,
        )
        sys.exit()

    ### Verify input and output directories ###
    input_dir = Path(args.in_root)
    if not input_dir.exists():
        print(
            f"{su.SFM.failure}Input directory {args.in_root} does not exist.",
            flush=True,
        )
        sys.exit()

    output_dir = Path(args.out_root)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Save config to output directory
    with open(output_dir / "config.json", "w", encoding="utf8") as f:
        json.dump(config, f, indent=4)

    ### Get image paths ###
    img_paths = glob(str(input_dir / "*"))

    if not img_paths:
        print(f"{su.SFM.failure}No images found in {input_dir}", flush=True)
        sys.exit()

    ### Load model ###
    model = models.get_unet_patch_segmentor_from_cfg(model_cfg_path)

    config["time"] = args.time
    config["channel"] = args.channel

    ### Analyze images ###
    for img_path in img_paths:
        analyze_img(Path(img_path), model, output_dir, config, args.detect_well)


if __name__ == "__main__":
    main()
