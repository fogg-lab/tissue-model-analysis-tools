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
from fl_tissue_model_tools.well_mask_generation import generate_well_mask

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
    img: npt.NDArray, well_mask: npt.NDArray, well_idx: Tuple[npt.NDArray, npt.NDArray],
    sd_coef: float, rand_state: np.random.RandomState,
) -> npt.NDArray:
    """Apply well mask to image and perform foreground thresholding on the masked image.

    Args:
        img: Original image.
        well_mask: Mask of well.
        well_idx: Indices within mask.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rand_state: RandomState object to allow for reproducability.

    Returns:
        Thresholded images.

    """
    masked = prep.apply_mask(img, well_mask).astype(float)
    masked_and_thresholded = prep.exec_threshold(masked, well_idx, sd_coef, rand_state)

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


def well_mask_setup(img: npt.NDArray, well_buffer=0.05):
    """Generate a well mask for an image.

    Args:
        img: Image to generate mask for.
        well_buffer: Buffer to add around the well mask, as a fraction of
            the well diameter. Defaults to 0.05.

    Returns:
        (Mask, mask coordinates, mask area).

    """
    well_mask = generate_well_mask(img, well_buffer, mask_val=defs.MAX_UINT8)
    well_coords = np.where(well_mask > 0)
    well_pix_area = len(well_coords[0])
    return well_mask, well_coords, well_pix_area


def threshold_images(
    imgs: List[npt.NDArray], well_masks: List[npt.NDArray],
    well_idx: List[Tuple[npt.NDArray, npt.NDArray]], sd_coef: float,
    rand_state: np.random.RandomState
) -> List[npt.NDArray]:
    """Apply mask & threshold to all images.

    Args:
        imgs: Original images.
        well_masks: Well masks.
        well_idx: Indices within each well mask.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rand_state: RandomState object to allow for reproducability.

    Returns:
        List of masked/thresholded images.

    """
    gmm_thresh_all = d.compute(
        [d.delayed(mask_and_threshold)(img, well_masks[i], well_idx[i], sd_coef, rand_state)
            for i, img in enumerate(imgs)]
    )[0]
    return gmm_thresh_all


def compute_areas(
    imgs: List[npt.NDArray], well_pix_area: List[int]
) -> npt.NDArray[float]:
    """Compute non-zero pixel area of thresholded images.

    Args:
        imgs: Thresholded images.
        well_pix_area: Pixel area of well mask.

    Returns:
        Non-zero pixel areas of thresholded images.

    """
    area_prop = d.compute(
        [d.delayed(an.compute_area_prop)(img, well_pix_area[i]) for i, img in enumerate(imgs)]
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
    batch_size = config["batch_size"]
    well_buffer = config["well_buffer"]

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

        # Well masking
        well_masks_with_info = [well_mask_setup(img, well_buffer) for img in gs_ds_imgs]
        well_masks, well_idx, well_pix_areas = zip(*well_masks_with_info)

        # Threshold images
        gmm_thresh = threshold_images(gs_ds_imgs, well_masks, well_idx, sd_coef, rand_state)
        gmm_thresh_all.extend(gmm_thresh)
        ### Compute areas ###
        area_prop.extend(compute_areas(gmm_thresh, well_pix_areas))

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
