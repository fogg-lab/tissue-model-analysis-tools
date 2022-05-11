import os
import sys
import dask as d
import numpy as np
import cv2
import numpy.typing as npt

from fl_tissue_model_tools import defs
from fl_tissue_model_tools import preprocessing as prep
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import zstacks as zs


proj_method = {
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
    zpaths, zstack = zs.zstack_from_dir(zs_path, extension, descending)
    zstack = prep.min_max_(zstack, a, b, mn, mx)
    return zstack


def save_zproj(zproj: npt.NDArray, out_root: str, zid: str, zproj_type: str, extension: str) -> None:
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
    mn = defs.GS_MIN
    mx = defs.GS_MAX
    if extension == "tif":
        cast_type = np.uint16
        a = defs.TIF_MIN
        b = defs.TIF_MAX
    elif extension == "png":
        cast_type = np.uint8
        a = mn
        b = mx
    else:
        raise OSError(f"Unsupported file type for analysis: {extension}")
    cv2.imwrite(
        f"{out_root}/{zid}_{zproj_type}.{extension}",
        prep.min_max_(zproj, a, b, mn, mx).astype(cast_type)
    )


def main():
    args = su.parse_zproj_args({})
    verbose = args.verbose


    ### Tidy up paths ###
    in_root = args.in_root.replace("\\", "/")
    out_root = args.out_root.replace("\\", "/")


    ### Verify input source ###
    extension = args.extension.replace(".", "")
    try:
        zstack_paths = su.zproj_verify_input_dir(in_root, extension, verbose=verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


    ### Verify output destination ###
    try:
        su.zproj_verify_output_dir(out_root, verbose=verbose)
    except PermissionError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


    ### Compute Z projections ###
    if verbose:
        su.verbose_header("Constructing Z projections")

    zp_method = args.method
    descending = bool(args.order)
    z_ids = [zsp.split("/")[-1] for zsp in zstack_paths]
    if verbose:
        print(f"Loading Z stacks...")
    try:
        zstacks = d.compute(
            d.delayed({z_ids[i]: get_zstack(zsp, extension, descending) for i, zsp in enumerate(zstack_paths)})
        )[0]
    except OSError as e:
        print(f"{su.SFM.failure}{e}")
        sys.exit()

    if verbose:
        print(f"... Z stacks loaded.")

    pm = proj_method[zp_method]

    if verbose:
        print(f"{os.linesep}Computing projections...")

    zprojs = d.compute(
        d.delayed({z_id: pm(zstacks[z_id]) for z_id in z_ids})
    )[0]
    if verbose:
            print(f"... Projections computed.")
    
    ### Save Z projections ###
    if verbose:
        print(f"{os.linesep}Saving projections...")
    for z_id, zproj in zprojs.items():
        save_zproj(zproj, out_root, z_id, zp_method, extension, verbose=verbose)
    
    if verbose:
        print(f"... Projections saved.")
        print(su.SFM.success)
        print(su.verbose_end)


if __name__ == "__main__":
    main()
