import os
import sys
import cv2
import numpy as np
import dask as d
import pandas as pd
import json
import numpy.typing as npt

from glob import glob
from typing import Sequence


from fl_tissue_model_tools import data_prep, defs
from fl_tissue_model_tools import preprocessing as prep
from fl_tissue_model_tools import analysis as an
from fl_tissue_model_tools import script_util as su


linesep = os.linesep
default_config_path = f"../config/default_cell_area_computation.json"
thresh_subdir = "thresholded"
calc_subdir = "calculations"


def load_img(img_path: str, dsamp: bool=True, dsize: int=250, extension: str="tif") -> npt.NDArray:
    """Load and downsample image.

    Args:
        img_path: Path to image.
        dsamp: Whether to downsample the image.
        dsize: If downsampling image, size to downsample to.
        extension: File extension for image. Only tif & png
            currently supported.

    Returns:
        Loaded and (optionally) downsampled image.

    """
    if extension == "tif":
        # cv.IMREAD_ANYDEPTH loads the image as a 16 bit grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    elif extension == "png":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if dsamp:
        img = cv2.resize(img, dsize, cv2.INTER_AREA)
    return img


def load_and_norm(img_path: str, extension: str, dsamp: bool=True, dsize: int=250) -> npt.NDArray:
    """Load and normalize image to new range.

    Args:
        img_path: Path to image.
        extension: File extension for image.
        dsamp: Whether to downsample the image. Only tif & png
            currently supported.
        dsize: If downsampling image, size to downsample to.

    Raises:
        OSError: Unsupported file extension supplied.

    Returns:
        Normalized image located at img_path.

    """
    a = defs.GS_MIN
    b = defs.GS_MAX

    if extension == "png":
        mn = a
        mx = b
    elif extension == "tif":
        mn = defs.TIF_MIN
        mx = defs.TIF_MAX
    else:
        raise OSError(f"Unsupported file type for analysis: {extension}")
    img = load_img(img_path, dsamp=True, dsize=dsize)
    return prep.min_max_(img, a, b, mn, mx)


def mask_and_threshold(img: npt.NDArray, circ_mask: npt.NDArray, pinhole_idx: tuple[npt.NDArray, npt.NDArray], sd_coef: float, rs: np.random.RandomState) -> npt.NDArray:
    """Apply circle mask to image and perform foreground thresholding on the masked image.

    Args:
        img: Original image.
        circ_mask: Circular mask.
        pinhole_idx: Indices within circular mask.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rs: RandomState object to allow for reproducability.

    Returns:
        Thresholded images.

    """
    masked = prep.apply_mask(img, circ_mask).astype(float)
    return prep.exec_threshold(masked, pinhole_idx, sd_coef, rs)


def prep_images(img_paths: Sequence[str], dsamp_size: int, extension: str) -> list[npt.NDArray]:
    """Create grayscale, downsampled versions of original images.

    Args:
        img_paths: Paths to each image.
        dsamp_size: Image size after downsampling.
        extension: File extension for image.

    Returns:
        Downsampled, grayscale versions of initial images.

    """
    gs_ds_imgs = d.compute(
        [d.delayed(load_and_norm)(img_p, extension, dsamp=True, dsize=(dsamp_size, dsamp_size)) for img_p in img_paths]
    )[0]
    return gs_ds_imgs


def circ_mask_setup(img_shape: tuple[int, int], pinhole_cut: int) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray], int]:
    """Compute values needed to apply circular mask to images.

    Args:
        img_shape: (H,W) of images.
        pinhole_cut: Number of pixels shorter than image height/width that
            circular mask radius will be. Larger pinhole_cut -> more of the
            initial image will be trimmed.

    Returns:
        (Circular mask, indices within pinhole, area of circular mask).

    """
    img_center = (img_shape[0] // 2, img_shape[1] // 2)
    circ_rad = img_center[0] - (pinhole_cut)
    circ_mask = prep.gen_circ_mask(img_center, circ_rad, img_shape, defs.GS_MAX)
    pinhole_idx = np.where(circ_mask > 0)
    circ_pix_area = np.sum(circ_mask > 0)
    return circ_mask, pinhole_idx, circ_pix_area


def threshold_images(imgs: list[npt.NDArray], circ_mask: npt.NDArray, pinhole_idx: tuple[npt.NDArray, npt.NDArray], sd_coef: float, rs: np.random.RandomState) -> list[npt.NDArray]:
    """Apply mask & threshold to all images.

    Args:
        imgs: Original images.
        circ_mask: Circular mask.
        pinhole_idx: Indices within circular mask.
        sd_coef: Threshold pixels with less than sd_coef * foreground_mean.
        rs: RandomState object to allow for reproducability.

    Returns:
        List of masked/thresholded images.

    """
    gmm_thresh_all = d.compute(
        [d.delayed(mask_and_threshold)(img, circ_mask, pinhole_idx, sd_coef, rs) for img in imgs]
    )[0]
    return gmm_thresh_all


def compute_areas(imgs: list[npt.NDArray], circ_pix_area: int) -> npt.NDArray[np.float_]:
    """Compute non-zero pixel area of thresholded images.

    Args:
        imgs: Thresholded images.
        circ_pix_area: Pixel area of circular mask.

    Returns:
        Non-zero pixel areas of thresholded images.

    """
    area_prop = d.compute(
        [d.delayed(an.compute_area_prop)(img, circ_pix_area) for img in imgs]
    )[0]
    area_prop = np.array(area_prop)
    return area_prop


def main():
    args = su.parse_cell_area_args({"thresh_subdir": thresh_subdir, "calc_subdir": calc_subdir, "default_config_path": default_config_path})
    verbose = args.verbose


    ### Tidy up paths ###
    in_root = args.in_root.replace("\\", "/")
    out_root = args.out_root.replace("\\", "/")


    ### Verify input source ###
    extension = args.extension.replace(".", "")
    try:
        img_paths = su.cell_area_verify_input_dir(in_root, extension, verbose=verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


    ### Verify output destination ###
    try:
        su.cell_area_verify_output_dir(out_root, thresh_subdir, calc_subdir, verbose=verbose)
    except PermissionError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


    ### Load config ###
    config_path = args.config
    try:
        config = su.cell_area_verify_config_file(config_path, verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()

    if verbose:
        su.verbose_header("Performing Analysis")
        print("Computing areas...")


    ### Prep images ###
    dsamp_size = config["dsamp_size"]
    sd_coef = config["sd_coef"]
    rs_seed = None if (config["rs_seed"] == "None") else config["rs_seed"]
    pinhole_buffer = config["pinhole_buffer"]
    rs = np.random.RandomState(rs_seed)
    pinhole_cut = int(round(dsamp_size * pinhole_buffer))
    try:
        gs_ds_imgs = prep_images(img_paths, dsamp_size, extension)
    except OSError as e:
        print(f"{su.SFM.failure}{e}")
        sys.exit()

    # variables for image masking
    circ_mask, pinhole_idx, circ_pix_area = circ_mask_setup(gs_ds_imgs[0].shape, pinhole_cut)
    # Threshold images
    gmm_thresh_all = threshold_images(gs_ds_imgs, circ_mask, pinhole_idx, sd_coef, rs)


    ### Compute areas ###
    area_prop = compute_areas(gmm_thresh_all, circ_pix_area)

    if verbose:
        print("... Areas computed successfully.")


    ### Save results ###
    if verbose:
        print(f"{linesep}Saving results...")

    img_ids = [img_n.split("/")[-1] for img_n in img_paths]
    area_df = pd.DataFrame(
        data = {"image_id": img_ids, "area_pct": area_prop * 100}
    )
    for i in range(len(img_ids)):
        out_img = gmm_thresh_all[i].astype(np.uint8)
        img_id = img_ids[i]
        out_path = f"{out_root}/{thresh_subdir}/{img_id}_thresholded.png"
        cv2.imwrite(
            out_path,
            out_img
        )
    if verbose:
        print(f"... Thresholded images saved to:{os.linesep}\t{out_root}/{thresh_subdir}")
    area_df.to_csv(f"{out_root}/{calc_subdir}/area_results.csv", index=False)
    if verbose:
        print(f"... Area calculations saved to:{os.linesep}\t{out_root}/{calc_subdir}")
        print(su.SFM.success)
        print(su.verbose_end)


if __name__ == "__main__":
    main()
