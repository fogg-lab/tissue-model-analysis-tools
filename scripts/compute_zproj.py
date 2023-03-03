import os
import sys
from pathlib import Path
import numpy as np
import numpy.typing as npt
import cv2

import compute_cell_area as area

from fl_tissue_model_tools import defs
from fl_tissue_model_tools import preprocessing as prep
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import zstacks as zs
from fl_tissue_model_tools import helper


proj_methods = {
    "min": zs.proj_min,
    "max": zs.proj_max,
    "med": zs.proj_med,
    "avg": zs.proj_avg,
    "fs": zs.proj_focus_stacking
}


def get_zstack(zs_path: str, descending: bool) -> npt.NDArray:
    """Given path to Z stack, return Z stack as array of images.

    Args:
        zs_path: Path to Z stack.
        descending: Whether to consider 0 to be the bottom or top. If
            descending=True, 0 is considered to be the bottom.

    Returns:
        Z stack as array of images.
    """

    z_paths = sorted(helper.get_img_paths(zs_path), key=zs.default_get_zpos, reverse=descending)
    z_stack = []

    for z_path in z_paths:
        z_img = cv2.imread(z_path, cv2.IMREAD_ANYDEPTH)
        z_img_norm = prep.min_max_(z_img, 0, defs.MAX_UINT8, 0, defs.MAX_UINT16)
        z_stack.append(z_img_norm.round().astype(np.uint8))

    return np.array(z_stack)


def save_zproj(zproj: npt.NDArray, out_root: str, zid: str, zproj_type: str, ext: str) -> None:
    """Save Z projected image.

    Args:
        zproj: Z projected image.
        out_root: Path to root output directory.
        zid: Base filename of Z projected image.
        zproj_type: Projection type, appended to zid.

    Raises:
        OSError: Unsupported file extension supplied.
    """
    old_min = 0
    old_max = defs.MAX_UINT8

    cast_type = np.uint16
    new_min = 0
    new_max = defs.MAX_UINT16

    zproj_out_path = os.path.join(out_root, f"{zid}_{zproj_type}{ext}")
    zproj_out_img = prep.min_max_(zproj, new_min, new_max, old_min, old_max).astype(cast_type)
    cv2.imwrite(zproj_out_path, zproj_out_img)


def main():
    '''Computes z projections and saves to output directory.'''

    args = su.parse_zproj_args()
    verbose = args.verbose
    compute_cell_area = args.area


    ### Verify input source ###
    try:
        zstack_paths = su.zproj_verify_input_dir(args.in_root, verbose=verbose)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()


    ### Verify output destination ###
    try:
        su.zproj_verify_output_dir(args.out_root, verbose=verbose)
    except PermissionError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()


    ### Compute Z projections ###
    if verbose:
        su.verbose_header("Constructing Z projections")

    proj_method = proj_methods[args.method]
    descending = bool(args.order)
    if verbose:
        print("Loading and computing Z stacks...")
    try:
        # zprojs: A dictionary of Z projections, keyed by Z stack ID.
        zprojs = {Path(zsp).name: proj_method(get_zstack(zsp, descending)) for zsp in zstack_paths}
    except OSError as error:
        print(f"{su.SFM.failure}{error}")
        sys.exit()

    if verbose:
        print("... Projections computed.")

    ### Save Z projections ###

    # Use first extension from the input directory as the output extension
    out_ext = Path(zstack_paths[0]).suffix
    if out_ext not in (".tif", ".tiff", ".png"):
        out_ext = ".tif"

    if verbose:
        print(f"{os.linesep}Saving projections...")
    for z_id, zproj in zprojs.items():
        save_zproj(zproj, args.out_root, z_id, args.method, out_ext)

    if verbose:
        print("... Projections saved.")
        print(su.SFM.success)
        print(su.verbose_end)

    if compute_cell_area:
        if "-a" in sys.argv:
            sys.argv.remove("-a")
        elif "--area" in sys.argv:
            sys.argv.remove("--area")
        # Set in_root to out_root (where z projections are saved)
        sys.argv[-2] = sys.argv[-1]
        area.main()


if __name__ == "__main__":
    main()