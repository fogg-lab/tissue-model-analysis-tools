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
from skimage.filters import gaussian, frangi, median
from skimage.transform import resize
from skimage.morphology import medial_axis, disk, binary_erosion, dilation
from skimage import measure
from scipy.ndimage import distance_transform_edt

from fl_tissue_model_tools import helper, models, models_util, defs
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools.transforms import filter_branch_seg_mask
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

        # Apply gaussian filter with a small sigma
        for i in range(len(img)):
            img[i, :] = gaussian(img[i], sigma=1, preserve_range=True)

        img = resize(
            img,
            (len(img), *img_dsamp_res),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        img_vess = np.zeros_like(img)

        for i, im in enumerate(img):
            im_vess = frangi(im, sigmas=range(1, 16, 2), beta=1.5, black_ridges=False)
            img_vess[i] = rescale_intensity(im_vess, out_range=(0, 1))
            img[i] = rescale_intensity(im, out_range=(0, 1))

        maxima = np.ones_like(img, dtype=bool)
        maxima[1:] = img_vess[1:] > img_vess[:-1]
        maxima[:-1] &= img_vess[:-1] >= img_vess[1:]
        img_vess[~maxima] = 0
        background = img_vess < 0.01
        img_vess[background] = 0
        maxima[background] = 0

        ### Handle overlapping regions prior to creating 2D projection:
        # Find and label regions for each z slice in the vesselness maxima array;
        # calculate regions for slice as maxima[z-1] | maxima[z] | maxima[z+1],
        # calculate bbox diagonal length for regions on this mask,
        # and, in case of overlapping regions at other depths,
        # use the bbox diagonal length as a priority metric for what to project.
        # Finally, add a buffer around objects on the projection that exist at
        # contiguous xy coordinates but >1 difference in z position.
        # This procedure cleanly separates different-depth objects for the 2D projection
        # and prioritizes regions with a larger bounding box diagonal in cases of overlap.

        # bbox_diag: Max bbox diagonal of potential vessel (Frangi >.01) at each xy position
        bbox_diag = np.zeros_like(maxima[0], dtype=np.float32)

        # best_vess_zpos: z position of voxel with max bbox diagonal at each xy position
        best_vess_zpos = np.full_like(maxima[0], fill_value=-1, dtype=np.int32)

        for z in range(len(maxima)):
            zslice_regions = maxima[z]
            # Union mask with the slice above and the slice below the current slice
            if z != 0:
                zslice_regions |= maxima[z - 1]
            if z != len(maxima) - 1:
                zslice_regions |= maxima[z + 1]
            slice_region_labels = measure.label(zslice_regions)
            bbox = measure.regionprops_table(
                label_image=slice_region_labels,
                properties=["bbox"],
            )
            x0, y0 = bbox["bbox-0"], bbox["bbox-1"]
            x1, y1 = bbox["bbox-2"], bbox["bbox-3"]
            # Calculate (squared) length of regions bbox diagonals
            diag_lengths = (x0 - x1) ** 2 + (y0 - y1) ** 2
            # The labels start at 1 (0 is background). Prepend 0 to index by label
            diag_lengths = np.insert(diag_lengths, 0, 0)
            slice_region_labels[~maxima[z]] = 0
            cur_bbox_diag = diag_lengths[slice_region_labels]
            is_max = cur_bbox_diag > bbox_diag
            best_vess_zpos[is_max] = z
            bbox_diag[is_max] = cur_bbox_diag[is_max]

        # Calculate buffer region of separation between objects at different depths
        combined_buffer_mask = np.zeros_like(bbox_diag, dtype=bool)
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                if row_shift == col_shift == 0:
                    continue
                # Compare source array elements to shifted (in 1 of 8 directions) array
                src_rows_slice = [None, None]
                src_cols_slice = [None, None]
                dst_rows_slice = [None, None]
                dst_cols_slice = [None, None]
                if row_shift == -1:  # Up
                    src_rows_slice[0] = 1
                    dst_rows_slice[1] = -1
                elif row_shift == 1:  # Down
                    src_rows_slice[1] = -1
                    dst_rows_slice[0] = 1
                if col_shift == -1:  # Left
                    src_cols_slice[0] = 1
                    dst_cols_slice[1] = -1
                elif col_shift == 1:  # Right
                    src_cols_slice[1] = -1
                    dst_cols_slice[0] = 1

                src_rows_slice = slice(src_rows_slice[0], src_rows_slice[1])
                dst_rows_slice = slice(dst_rows_slice[0], dst_rows_slice[1])
                src_cols_slice = slice(src_cols_slice[0], src_cols_slice[1])
                dst_cols_slice = slice(dst_cols_slice[0], dst_cols_slice[1])

                src_slice = src_rows_slice, src_cols_slice
                dst_slice = dst_rows_slice, dst_cols_slice
                z_diff = abs(best_vess_zpos[src_slice] - best_vess_zpos[dst_slice])
                buffer_mask = (z_diff > 1) & (
                    bbox_diag[src_slice] < bbox_diag[dst_slice]
                )
                combined_buffer_mask[src_slice][buffer_mask] = 1
                best_vess_zpos[src_slice][buffer_mask] = -1
                bbox_diag[src_slice][buffer_mask] = 0

        # Max-project img_vess and zero-out the buffer region to separate objects
        img_vess_zprojection = img_vess.max(axis=0)
        img_vess_zprojection[combined_buffer_mask] = 0

        # Clean up projection by removing objects that won't qualify as vessels
        # (objects whose medial axis has a bbox diagonal length below min_branch_length)
        labeled_vess_img = measure.label(img_vess_zprojection > 0)
        skel_vess_img = medial_axis(labeled_vess_img > 0).astype(int)
        skel_vess_img = dilation(skel_vess_img, disk(1))
        skel_vess_img[labeled_vess_img == 0] = 0
        skel_vess_img[skel_vess_img > 0] = labeled_vess_img[skel_vess_img > 0]
        bbox = measure.regionprops_table(
            label_image=skel_vess_img,
            properties=["bbox"],
        )
        x0, y0 = bbox["bbox-0"], bbox["bbox-1"]
        x1, y1 = bbox["bbox-2"], bbox["bbox-3"]
        diag_lengths = (x0 - x1) ** 2 + (y0 - y1) ** 2
        # The labels start at 1 (0 is background). Prepend 0 to index by label
        diag_lengths = np.insert(diag_lengths, 0, 0)
        thresh = microns_to_pixels(
            min_branch_length, skel_vess_img.shape[1], image_width_microns
        )
        # Square the threshold (we skipped sqrt in the calculation of diag_lengths)
        thresh **= 2
        exclude_mask = diag_lengths[labeled_vess_img] < thresh
        img_vess_zprojection[exclude_mask] = 0

        # Enhance centerlines and reduce variance
        vess_skel = medial_axis(img_vess_zprojection > 0)
        vess_skel_dil = dilation(vess_skel, disk(1))
        vess_skel_dil_med = median(vess_skel_dil, disk(1))
        vess_skel_refined = medial_axis(vess_skel_dil_med)
        vess_skel_blur = gaussian(vess_skel_refined, 1)
        img_vess_zprojection = np.sqrt(img_vess_zprojection * vess_skel_blur)
        img = img_vess_zprojection

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
        plt.figure(figsize=(10, 10))
        plt.margins(0)
        ax = plt.gca()
        ax.imshow(rescale_intensity(original_image, out_range=(0, 255)), cmap="gray")
        morse_graph.plot_colored_tree(scaling_factor=scaling_factor, ax=ax)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
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
