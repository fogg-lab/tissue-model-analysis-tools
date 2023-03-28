import os
import sys
from pathlib import Path
from typing import Sequence, Tuple, List
import cv2
import numpy as np
import numpy.typing as npt
import dask as d
import pandas as pd

from fl_tissue_model_tools import defs
from fl_tissue_model_tools import preprocessing as prep
from fl_tissue_model_tools import analysis as an
from fl_tissue_model_tools import script_util as su

DEFAULT_CONFIG_PATH = str(defs.SCRIPT_CONFIG_DIR / "default_cell_area_computation.json")
THRESH_SUBDIR = "thresholded"
CALC_SUBDIR = "calculations"


def load_img(img_path: str, dsamp: bool=True, dsize: int=250) -> npt.NDArray:
    """Load and downsample image.

    Args:
        img_path: Path to image.
        dsamp: Whether to downsample the image.
        dsize: If downsampling image, size to downsample to.

    Returns:
        Loaded and (optionally) downsampled image.

    """

    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    if dsamp:
        img = cv2.resize(img, dsize, cv2.INTER_AREA)

    return img


def load_and_norm(img_path: str, dsize: int=250) -> npt.NDArray:
    """Load and normalize image to new range.

    Args:
        img_path: Path to image.
        dsize: If downsampling image, size to downsample to.

    Returns:
        Normalized image located at img_path.

    """
    new_min, new_max = 0, defs.MAX_UINT8
    old_min, old_max = 0, defs.MAX_UINT16
    img = load_img(img_path, dsamp=True, dsize=dsize)

    return prep.min_max_(img, new_min, new_max, old_min, old_max)


def mask_and_threshold(
    img: npt.NDArray, circ_mask: npt.NDArray, pinhole_idx: Tuple[npt.NDArray, npt.NDArray],
    sd_coef: float, rand_state: np.random.RandomState,
) -> npt.NDArray:
    """Apply circular mask to image and perform foreground thresholding on the
    masked image.

    Args:
        img: Original image.
        circ_mask: Circular mask.
        pinhole_idx: Indices within circular mask.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rand_state: RandomState object to allow for reproducability.

    Returns:
        Thresholded images.

    """
    masked = prep.apply_mask(img, circ_mask).astype(float)
    masked_and_thresholded = prep.exec_threshold(masked, pinhole_idx, sd_coef, rand_state)

    return masked_and_thresholded


def prep_images(img_paths: Sequence[str], dsamp_size: int) -> List[npt.NDArray]:
    """Create grayscale, downsampled versions of original images.

    Args:
        img_paths: Paths to each image.
        dsamp_size: Image size after downsampling.
        extension: File extension for image.

    Returns:
        Downsampled, grayscale versions of initial images.

    """
    gs_ds_imgs = d.compute([
        d.delayed(load_and_norm)(img_p, dsize=(dsamp_size, dsamp_size))
        for img_p in img_paths
    ])[0]
    return gs_ds_imgs


def circ_mask_setup(img_shape: Tuple[int, int], pinhole_cut: int) -> \
                    Tuple[npt.NDArray, Tuple[npt.NDArray, npt.NDArray], int]:
    """Compute values needed to apply circular mask to images.

    Args:
        img_shape: (H,W) of images.
        pinhole_cut: Number of pixels shorter than image height/width that
            circular mask radius will be. Larger pinhole_cut -> more of the
            initial image will be trimmed.

    Returns:
        (Circular mask, indices within pinhole, area of circular mask).

    """
    img_center = (img_shape[1] // 2, img_shape[0] // 2)
    circ_rad = img_center[0] - (pinhole_cut)
    circ_mask = prep.gen_circ_mask(img_center, circ_rad, img_shape, defs.MAX_UINT8)
    pinhole_idx = np.where(circ_mask > 0)
    circ_pix_area = np.sum(circ_mask > 0)
    return circ_mask, pinhole_idx, circ_pix_area


def circ_mask_setup_auto(img: npt.NDArray, pinhole_buffer=0.04):
    """Generate circular mask for image based on detected well boundary."""
    circ_mask = prep.gen_circ_mask_auto(img, pinhole_buffer, mask_val=defs.MAX_UINT8)
    pinhole_idx = np.where(circ_mask > 0)
    circ_pix_area = len(pinhole_idx[0])
    return circ_mask, pinhole_idx, circ_pix_area


def threshold_images(
    imgs: List[npt.NDArray], circ_masks: List[npt.NDArray],
    pinhole_idx: List[Tuple[npt.NDArray, npt.NDArray]], sd_coef: float,
    rand_state: np.random.RandomState
) -> List[npt.NDArray]:
    """Apply mask & threshold to all images.

    Args:
        imgs: Original images.
        circ_mask: Circular masks.
        pinhole_idx: Indices within each circular mask.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rand_state: RandomState object to allow for reproducability.

    Returns:
        List of masked/thresholded images.

    """
    gmm_thresh_all = d.compute(
        [d.delayed(mask_and_threshold)(img, circ_masks[i], pinhole_idx[i], sd_coef, rand_state)
            for i, img in enumerate(imgs)]
    )[0]
    return gmm_thresh_all


def compute_areas(
    imgs: List[npt.NDArray], circ_pix_area: List[int]
) -> npt.NDArray[float]:
    """Compute non-zero pixel area of thresholded images.

    Args:
        imgs: Thresholded images.
        circ_pix_area: Pixel area of circular mask.

    Returns:
        Non-zero pixel areas of thresholded images.

    """
    area_prop = d.compute(
        [d.delayed(an.compute_area_prop)(img, circ_pix_area[i]) for i, img in enumerate(imgs)]
    )[0]
    area_prop = np.array(area_prop)
    return area_prop


def main():
    '''Computes cell area and saves to output directory.'''

    args = su.parse_cell_area_args({
        "thresh_subdir": THRESH_SUBDIR,
        "calc_subdir": CALC_SUBDIR,
        "default_config_path": DEFAULT_CONFIG_PATH}
    )
    verbose = args.verbose


    ### Verify input source ###
    try:
        all_img_paths = su.cell_area_verify_input_dir(args.in_root, verbose=verbose)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()


    ### Verify output destination ###
    try:
        su.cell_area_verify_output_dir(args.out_root, THRESH_SUBDIR, CALC_SUBDIR, verbose=verbose)
    except PermissionError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()


    ### Load config ###
    config_path = args.config
    try:
        config = su.cell_area_verify_config_file(config_path, verbose)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()

    if verbose:
        su.verbose_header("Performing Analysis")
        print("Computing areas...")


    ### Prep images ###
    dsamp_size = config["dsamp_size"]
    sd_coef = config["sd_coef"]
    rs_seed = None if (config["rs_seed"] == "None") else config["rs_seed"]
    detect_well_edge = config["detect_well_edge"]
    batch_size = config["batch_size"]
    pinhole_buffer = config["pinhole_buffer"]
    pinhole_cut = int(round(dsamp_size * pinhole_buffer))

    area_prop = []
    gmm_thresh_all = []
    rand_state = np.random.RandomState(rs_seed)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for img_paths in chunks(all_img_paths, batch_size):
        try:
            gs_ds_imgs = prep_images(img_paths, dsamp_size)
        except OSError as error:
            print(f"{su.SFM.failure}{error}")
            sys.exit()

        # Variables for image masking
        if detect_well_edge:
            circ_masks_with_info = [circ_mask_setup_auto(img, pinhole_buffer) for img in gs_ds_imgs]
            circ_masks, pinhole_idx, circ_pix_areas = zip(*circ_masks_with_info)
        else:
            circ_mask, pinhole_idx, circ_pix_area = circ_mask_setup(gs_ds_imgs[0].shape, pinhole_cut)
            circ_masks = [circ_mask] * len(gs_ds_imgs)
            pinhole_idx = [pinhole_idx] * len(gs_ds_imgs)
            circ_pix_areas = [circ_pix_area] * len(gs_ds_imgs)

        # Threshold images
        gmm_thresh = threshold_images(gs_ds_imgs, circ_masks, pinhole_idx, sd_coef, rand_state)
        gmm_thresh_all.extend(gmm_thresh)
        ### Compute areas ###
        area_prop.extend(compute_areas(gmm_thresh, circ_pix_areas))

    area_prop = np.array(area_prop)

    if verbose:
        print("... Areas computed successfully.")

    ### Save results ###
    if verbose:
        print(f"{os.linesep}Saving results...")

    img_ids = [Path(img_n).stem for img_n in all_img_paths]
    area_df = pd.DataFrame(
        data = {"image_id": img_ids, "area_pct": area_prop * 100}
    )
    for i, img_id in enumerate(img_ids):
        out_img = gmm_thresh_all[i].astype(np.uint8)
        out_path = os.path.join(args.out_root, THRESH_SUBDIR, f"{img_id}_thresholded.png")
        cv2.imwrite(out_path, out_img)

    if verbose:
        print(f"... Thresholded images saved to:{os.linesep}\t{args.out_root}/{THRESH_SUBDIR}")

    area_out_path = os.path.join(args.out_root, CALC_SUBDIR, "cell_area.csv")
    area_df.to_csv(area_out_path, index=False)

    if verbose:
        print(f"... Area calculations saved to:{os.linesep}\t{area_out_path}")
        print(su.SFM.success)
        print(su.verbose_end)


if __name__ == "__main__":
    main()
