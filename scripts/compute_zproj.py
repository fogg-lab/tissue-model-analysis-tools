import os
import sys
from pathlib import Path
import dask as d
import numpy as np
import numpy.typing as npt
import cv2

import compute_cell_area as area

from fl_tissue_model_tools import defs
from fl_tissue_model_tools import preprocessing as prep
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import zstacks as zs


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

    new_min, new_max = defs.GS_MIN, defs.GS_MAX
    old_min, old_max = defs.TIF_MIN, defs.TIF_MAX

    zstack = zs.zstack_from_dir(zs_path, descending)[1]
    zstack = prep.min_max_(zstack, new_min, new_max, old_min, old_max)
    return zstack


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
    old_min = defs.GS_MIN
    old_max = defs.GS_MAX

    cast_type = np.uint16
    new_min = defs.TIF_MIN
    new_max = defs.TIF_MAX

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

    zp_method = args.method
    descending = bool(args.order)
    z_ids = [Path(zsp).name for zsp in zstack_paths]
    if verbose:
        print("Loading Z stacks...")
    try:
        zstacks = d.compute(
            d.delayed(
                {z_ids[i]: get_zstack(zsp, descending) for i, zsp in enumerate(zstack_paths)}
            )
        )[0]
    except OSError as error:
        print(f"{su.SFM.failure}{error}")
        sys.exit()

    if verbose:
        print("... Z stacks loaded.")

    proj_method = proj_methods[zp_method]

    if verbose:
        print(f"{os.linesep}Computing projections...")

    zprojs = d.compute(
        d.delayed(
            {z_id: proj_method(zstacks[z_id]) for z_id in z_ids}
        )
    )[0]

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
        save_zproj(zproj, args.out_root, z_id, zp_method, out_ext)

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