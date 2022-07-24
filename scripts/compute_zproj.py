import os
import sys
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


def get_zstack(zs_path: str, extension: str, descending: bool) -> npt.NDArray:
    """Given path to Z stack, return Z stack as array of images.

    Args:
        zs_path: Path to Z stack.
        extension: File extension for image. Only tif & png
            currently supported.
        descending: Whether to consider 0 to be the bottom or top. If
            descending=True, 0 is considered to be the bottom.

    Raises:
        OSError: Unsupported file extension supplied.

    Returns:
        Z stack as array of images.
    """
    new_min = defs.GS_MIN
    new_max = defs.GS_MAX

    if extension == "png":
        old_min = new_min
        old_max = new_max
    elif extension == "tif":
        old_min = defs.TIF_MIN
        old_max = defs.TIF_MAX
    else:
        raise OSError(f"Unsupported file type for analysis: {extension}")
    zstack = zs.zstack_from_dir(zs_path, extension, descending)[1]
    zstack = prep.min_max_(zstack, new_min, new_max, old_min, old_max)
    return zstack


def save_zproj(
    zproj: npt.NDArray, out_root: str, zid: str, zproj_type: str, extension: str
) -> None:
    """Save Z projected image.

    Args:
        zproj: Z projected image.
        out_root: Path to root output directory.
        zid: Base filename of Z projected image.
        zproj_type: Projection type, appended to zid.
        extension: File extension for image. Only tif & png
            currently supported.

    Raises:
        OSError: Unsupported file extension supplied.
    """
    old_min = defs.GS_MIN
    old_max = defs.GS_MAX
    if extension == "tif":
        cast_type = np.uint16
        new_min = defs.TIF_MIN
        new_max = defs.TIF_MAX
    elif extension == "png":
        cast_type = np.uint8
        new_min = old_min
        new_max = old_max
    else:
        raise OSError(f"Unsupported file type for analysis: {extension}")
    cv2.imwrite(
        f"{out_root}/{zid}_{zproj_type}.{extension}",
        prep.min_max_(zproj, new_min, new_max, old_min, old_max).astype(cast_type)
    )


def main():
    '''Computes z projections and saves to output directory.'''

    args = su.parse_zproj_args()
    verbose = args.verbose
    compute_cell_area = args.area

    ### Tidy up paths ###
    in_root = args.in_root.replace("\\", "/")
    out_root = args.out_root.replace("\\", "/")

    ### Verify input source ###
    extension = args.extension.replace(".", "")
    try:
        zstack_paths = su.zproj_verify_input_dir(in_root, extension,
                                                 verbose=verbose)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()

    ### Verify output destination ###
    try:
        su.zproj_verify_output_dir(out_root, verbose=verbose)
    except PermissionError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()

    ### Compute Z projections ###
    if verbose:
        su.verbose_header("Constructing Z projections")

    zp_method = args.method
    descending = bool(args.order)
    z_ids = [zsp.split("/")[-1] for zsp in zstack_paths]
    if verbose:
        print("Loading Z stacks...")
    try:
        zstacks = d.compute(
            d.delayed({z_ids[i]: get_zstack(zsp, extension, descending)
                        for i, zsp in enumerate(zstack_paths)})
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
        d.delayed({z_id: proj_method(zstacks[z_id]) for z_id in z_ids})
    )[0]
    if verbose:
        print("... Projections computed.")

    ### Save Z projections ###
    if verbose:
        print(f"{os.linesep}Saving projections...")
    for z_id, zproj in zprojs.items():
        save_zproj(zproj, out_root, z_id, zp_method, extension)

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
