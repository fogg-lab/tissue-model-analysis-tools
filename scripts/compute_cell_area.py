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


def load_img(img_name, dsamp=True, dsize=250):
    """Load and downsample image.
    
    """
    # cv.IMREAD_ANYDEPTH loads the image as a 16 bit grayscale image
    img = cv2.imread(img_name, cv2.IMREAD_ANYDEPTH)
    if dsamp:
        img = cv2.resize(img, dsize, cv2.INTER_AREA)
    return img


def load_and_norm(img_name, a, b, mn, mx, dsamp=True, dsize=250):
    """ Load and normalize image to new range.
    
    """
    img = load_img(img_name, dsamp=True, dsize=dsize)
    return prep.min_max_(img, a, b, mn, mx)


def mask_and_threshold(img, circ_mask, pinhole_idx, sd_coef, rs):
    """Apply circle mask to image and perform foreground thresholding on the masked image.
    
    """
    masked = prep.apply_mask(img, circ_mask).astype(float)
    return prep.exec_threshold(masked, pinhole_idx, sd_coef, rs)


def prep_images(img_names: Sequence[str], dsamp_size: int) -> list[npt.NDArray]:
    gs_ds_imgs = d.compute(
        [d.delayed(load_and_norm)(img_n, defs.GS_MIN, defs.GS_MAX, defs.TIF_MIN, defs.TIF_MAX, dsamp=True, dsize=(dsamp_size, dsamp_size)) for img_n in img_names]
    )[0]
    return gs_ds_imgs


def threshold_images(imgs: list[npt.NDArray], circ_mask: npt.NDArray, pinhole_idx: tuple[npt.NDArray, npt.NDArray], sd_coef: float, rs: np.random.RandomState) -> list[npt.NDArray]:
    gmm_thresh_all = d.compute(
        [d.delayed(mask_and_threshold)(img, circ_mask, pinhole_idx, sd_coef, rs) for img in imgs]
    )[0]
    return gmm_thresh_all


def compute_areas(imgs: list[npt.NDArray], circ_pix_area: int) -> npt.NDArray[np.float_]:
    area_prop = d.compute(
        [d.delayed(an.compute_area_prop)(img, circ_pix_area) for img in imgs]
    )[0]
    area_prop = np.array(area_prop)
    return area_prop


def main():
    try:
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
        except Exception as e:
            print(f"{su.SFM.failure}{e}")
            sys.exit()

        ### Load config ###
        config_path = args.config
        try:
            config = su.cell_area_verify_config_file(config_path, verbose)
        except FileNotFoundError as e:
            print(f"{su.SFM.failure} {e}")
            sys.exit()

        ### Prep images ###
        dsamp_size = config["dsamp_size"]
        sd_coef = config["sd_coef"]
        rs_seed = None if (config["rs_seed"] == "None") else config["rs_seed"]
        pinhole_buffer = config["pinhole_buffer"]
        pinhole_cut = int(round(dsamp_size * pinhole_buffer))
        gs_ds_imgs = prep_images(img_paths, dsamp_size)

        # variables for image masking
        img_shape = gs_ds_imgs[0].shape
        img_center = (img_shape[0] // 2, img_shape[1] // 2)
        circ_rad = img_center[0] - (pinhole_cut)
        circ_mask = prep.gen_circ_mask(img_center, circ_rad, img_shape, defs.GS_MAX)
        pinhole_idx = np.where(circ_mask > 0)
        circ_pix_area = np.sum(circ_mask > 0)

        rs = np.random.RandomState(rs_seed)

        ### Threshold images ###
        gmm_thresh_all = threshold_images(gs_ds_imgs, circ_mask, pinhole_idx, sd_coef, rs)

        ### Compute areas ###
        area_prop = compute_areas(gmm_thresh_all, circ_pix_area)

        ### Save results ###
        img_ids = [img_n.split("/")[-1] for img_n in img_paths]
        area_df = pd.DataFrame(
            data = {"image_id": img_ids, "area_pct": area_prop * 100}
        )
        if verbose:
            print(f"{linesep}Saving results...")
        for i in range(len(img_ids)):
            out_img = gmm_thresh_all[i].astype(np.uint8)
            img_id = img_ids[i]
            out_path = f"{out_root}/{thresh_subdir}/{img_id}_thresholded.png"
            cv2.imwrite(
                out_path,
                out_img
            )
        if verbose:
            print(f"{linesep}\tThresholded images saved for reference.")
        area_df.to_csv(f"{out_root}/{calc_subdir}/area_results.csv", index=False)
        if verbose:
            print(f"{linesep}\tArea calculations saved.")
            print(f"{linesep}All analysis results saved to: {linesep}\t{out_root}")
    except PermissionError:
        raise PermissionError(f"{linesep*2}Must ensure that no previous analysis output files located in {linesep}\t{out_root}/{thresh_subdir} or {linesep}\t{out_root}/{calc_subdir} {linesep}are open in another application. {linesep*2}Close any such files and try again.{linesep}")


if __name__ == "__main__":
    main()
