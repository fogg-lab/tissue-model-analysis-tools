import os
import sys
import dask as d
import numpy as np
import cv2

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


def get_zstack(zs_path, extension, descending):
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


def save_zproj(zproj, out_root, zid, zp_type, extension, verbose=False):
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
        f"{out_root}/{zid}_{zp_type}.{extension}",
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
    zp_method = args.method
    z_ids = [zsp.split("/")[-1] for zsp in zstack_paths]
    try:
        zstacks = d.compute(
            d.delayed({z_ids[i]: get_zstack(zsp, "tif", True) for i, zsp in enumerate(zstack_paths)})
        )[0]
    except OSError as e:
        print(f"{su.SFM.failure}{e}")
        sys.exit()
    pm = proj_method[zp_method]
    zprojs = d.compute(
        d.delayed({z_id: pm(zstacks[z_id]) for z_id in z_ids})
    )[0]

    
    ### Save Z projections ###
    for z_id, zproj in zprojs.items():
        save_zproj(zproj, out_root, z_id, zp_method, extension, verbose=verbose)


if __name__ == "__main__":
    main()
