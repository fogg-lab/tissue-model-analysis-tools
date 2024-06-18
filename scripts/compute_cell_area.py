import os
import sys
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
import cv2
import numpy as np
import numpy.typing as npt
import dask as d
import pandas as pd
from skimage.exposure import rescale_intensity

from fl_tissue_model_tools import defs
from fl_tissue_model_tools import helper
from fl_tissue_model_tools import preprocessing as prep
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools.well_mask_generation import generate_well_mask

DEFAULT_CONFIG_PATH = str(defs.SCRIPT_CONFIG_DIR / "default_cell_area_computation.json")
THRESH_SUBDIR = "thresholded"
CALC_SUBDIR = "calculations"


def load_img(
    img_path: str,
    dsamp_size: Optional[int] = None,
    T: Optional[int] = None,
    C: Optional[int] = None,
) -> npt.NDArray:
    """Load and optionally downsample image.

    Args:
        img_path: Path to image.
        dsize: If downsampling image, size to downsample to.
        T (int, optional): Index of the time to use (needed if time series).
        C (int, optional): Index of the color channel to use (needed if multi channel).

    Returns:
        Loaded and (optionally) downsampled image.

    """

    img = helper.load_image(img_path, T, C)[0]
    if img.ndim == 3:
        # Max projection
        img = img.max(0)
    if dsamp_size is not None:
        dsamp_ratio = dsamp_size / max(img.shape)
        dsize = tuple(np.round(np.multiply(img.shape, dsamp_ratio)).astype(int))
        img = cv2.resize(img, dsize, cv2.INTER_AREA)
    return img


def mask_and_threshold(
    img: npt.NDArray,
    sd_coef: float,
    rand_state: np.random.RandomState,
    well_mask: Optional[npt.NDArray] = None,
    well_idx: Optional[Tuple[npt.NDArray, npt.NDArray]] = None,
) -> npt.NDArray:
    """Apply well mask to image and perform foreground thresholding on the masked image.

    Args:
        img: Original image.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rand_state: RandomState object to allow for reproducability.
        well_mask: Mask of well.
        well_idx: Indices within mask.

    Returns:
        Thresholded images.

    """
    img = rescale_intensity(img, out_range=(0, 1)).astype(np.float32)
    if well_mask is not None:
        masked = prep.apply_mask(img, well_mask)
    else:
        masked = img
    masked_and_thresholded = prep.exec_threshold(masked, well_idx, sd_coef, rand_state)

    return (masked_and_thresholded > 0).astype(np.uint8) * defs.MAX_UINT8


def prep_images(
    img_paths: Sequence[str],
    dsamp_size: int,
    T: Optional[int] = None,
    C: Optional[int] = None,
) -> List[npt.NDArray]:
    """Create grayscale, downsampled versions of original images.

    Args:
        img_paths: Paths to each image.
        dsamp_size: Image size after downsampling.
        T (int, optional): Index of the time to use (needed if time series).
        C (int, optional): Index of the color channel to use (needed if multi channel).

    Returns:
        Downsampled, grayscale versions of initial images.

    """
    gs_ds_imgs = d.compute(
        [
            d.delayed(load_img)(img_p, dsamp_size=dsamp_size, T=T, C=C)
            for img_p in img_paths
        ]
    )[0]
    return gs_ds_imgs


def well_mask_setup(img: npt.NDArray):
    """Generate a well mask for an image.

    Args:
        img: Image to generate mask for.

    Returns:
        (Mask, mask coordinates, mask area).

    """
    well_mask = generate_well_mask(img, mask_val=defs.MAX_UINT8)
    well_coords = np.where(well_mask > 0)
    well_pix_area = len(well_coords[0])
    return well_mask, well_coords, well_pix_area


def threshold_images(
    imgs: List[npt.NDArray],
    sd_coef: float,
    rand_state: np.random.RandomState,
    well_masks: List[npt.NDArray],
    well_idx: List[Tuple[npt.NDArray, npt.NDArray]],
) -> List[npt.NDArray]:
    """Apply mask & threshold to all images.

    Args:
        imgs: Original images.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rand_state: RandomState object to allow for reproducibility.
        well_masks: Well masks.
        well_idx: Indices within each well mask.

    Returns:
        List of masked/thresholded images.

    """
    gmm_thresh_all = d.compute(
        [
            d.delayed(mask_and_threshold)(
                img, sd_coef, rand_state, well_masks[i], well_idx[i]
            )
            for i, img in enumerate(imgs)
        ]
    )[0]
    return gmm_thresh_all


def compute_area_prop(img: npt.NDArray, ref_area: Optional[int] = None) -> float:
    """Computes the proportion of pixels that are thresholded in circular area.

    Args:
        img: A masked and thresholded image. Background pixels are 0.
        ref_area: Number of pixels in the circular mask area applied to the image.
            Default is None, equivalent to `ref_area=img.size` (all pixels).
        min_val: This parameter is currently unused. Defaults to 0.

    Returns:
        Proportion of pixels in circular mask area that are thresholded.
    """
    if ref_area is None:
        ref_area = img.size
    return np.sum(img > 0) / ref_area


def compute_areas(
    imgs: List[npt.NDArray], well_pix_area: List[Optional[int]]
) -> npt.NDArray:
    """Compute non-zero pixel area of thresholded images.

    Args:
        imgs: Thresholded images.
        well_pix_area: Pixel area of well mask.

    Returns:
        Non-zero pixel areas of thresholded images.

    """
    area_prop = d.compute(
        [
            d.delayed(compute_area_prop)(img, well_pix_area[i])
            for i, img in enumerate(imgs)
        ]
    )[0]
    area_prop = np.array(area_prop)
    return area_prop


def main(args=None):
    """Computes cell area and saves to output directory."""

    if args is None:
        args = su.parse_cell_area_args(
            {
                "thresh_subdir": THRESH_SUBDIR,
                "calc_subdir": CALC_SUBDIR,
                "default_config_path": DEFAULT_CONFIG_PATH,
            }
        )
        args_prespecified = False
    else:
        args_prespecified = True

    ### Verify input source ###
    try:
        all_img_paths = su.cell_area_verify_input_dir(args.in_root)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}", flush=True)
        sys.exit()

    ### Verify output destination ###
    try:
        su.cell_area_verify_output_dir(args.out_root, THRESH_SUBDIR, CALC_SUBDIR)
    except PermissionError as error:
        print(f"{su.SFM.failure} {error}", flush=True)
        sys.exit()

    ### Load config ###
    config_path = DEFAULT_CONFIG_PATH if args_prespecified else args.config
    try:
        config = su.cell_area_verify_config_file(config_path)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}", flush=True)
        sys.exit()

    su.section_header("Performing Analysis")

    ### Prep images ###
    dsamp_size = config["dsamp_size"]
    sd_coef = config["sd_coef"]
    rs_seed = None if (config["rs_seed"] == "None") else config["rs_seed"]
    batch_size = config["batch_size"]

    area_prop = []
    gmm_thresh_all = []
    rand_state = np.random.RandomState(rs_seed)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # Keep a list of all well masks for saving intermediate outputs later
    all_well_masks = []

    for img_paths in chunks(all_img_paths, batch_size):
        try:
            gs_ds_imgs = prep_images(img_paths, dsamp_size, T=args.time, C=args.channel)
        except OSError as error:
            print(f"{su.SFM.failure}{error}", flush=True)
            sys.exit()

        # Well masking
        if args.detect_well:
            well_masks_with_info = [well_mask_setup(img) for img in gs_ds_imgs]
            well_masks, well_idx, well_pix_areas = zip(*well_masks_with_info)
        else:
            well_masks, well_idx, well_pix_areas = [[None] * len(gs_ds_imgs)] * 3
        all_well_masks.extend(well_masks)

        # Threshold images
        gmm_thresh = threshold_images(
            gs_ds_imgs, sd_coef, rand_state, well_masks, well_idx
        )
        gmm_thresh_all.extend(gmm_thresh)
        ### Compute areas ###
        area_prop.extend(compute_areas(gmm_thresh, well_pix_areas))

    area_prop = np.array(area_prop)

    print("... Areas computed successfully.", flush=True)

    ### Save results ###
    print(f"{os.linesep}Saving results...", flush=True)

    img_ids = [Path(img_n).stem for img_n in all_img_paths]
    area_df = pd.DataFrame(data={"image_id": img_ids, "area_pct": area_prop * 100})

    # Save intermediate output: thresholded images
    for i, img_id in enumerate(img_ids):
        if args.detect_well:
            # Save masked image
            out_img = all_well_masks[i]
            out_path = os.path.join(
                args.out_root, THRESH_SUBDIR, f"{img_id}_well_mask.png"
            )
            cv2.imwrite(out_path, out_img)
        # Save thresholded image
        out_img = gmm_thresh_all[i].astype(np.uint8)
        out_path = os.path.join(
            args.out_root, THRESH_SUBDIR, f"{img_id}_thresholded.png"
        )
        cv2.imwrite(out_path, out_img)

    if args.detect_well:
        print(
            f"... Well masks saved to:{os.linesep}\t{args.out_root}/{THRESH_SUBDIR}",
            flush=True,
        )
    print(
        f"... Thresholded images saved to:{os.linesep}\t{args.out_root}/{THRESH_SUBDIR}",
        flush=True,
    )

    area_out_path = os.path.join(args.out_root, CALC_SUBDIR, "cell_area.csv")
    area_df.to_csv(area_out_path, index=False)

    print(f"... Area calculations saved to:{os.linesep}\t{area_out_path}", flush=True)
    print(su.SFM.success, flush=True)
    print(su.END_SEPARATOR, flush=True)


if __name__ == "__main__":
    main()