import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import numpy.typing as npt
import cv2

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
    "fs": zs.proj_focus_stacking,
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

    z_paths = sorted(
        helper.get_img_paths(zs_path), key=zs.default_get_zpos, reverse=descending
    )
    return np.array([cv2.imread(z_path, cv2.IMREAD_ANYDEPTH) for z_path in z_paths])


def main():
    """Computes z projections and saves to output directory."""

    print(sys.argv[1:])

    args = su.parse_zproj_args()
    compute_cell_area = args.area

    ### Verify input source ###
    try:
        zstack_paths = su.zproj_verify_input_dir(args.in_root)
    except FileNotFoundError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()

    ### Verify output destination ###
    try:
        su.zproj_verify_output_dir(args.out_root)
    except PermissionError as error:
        print(f"{su.SFM.failure} {error}")
        sys.exit()

    ### Compute Z projections ###
    su.section_header("Constructing Z projections")

    proj_method = proj_methods[args.method]
    descending = bool(args.order)
    print("Loading and computing Z stacks...")
    try:
        # zprojs: A dictionary of Z projections, keyed by Z stack ID.
        zprojs = {
            Path(zsp).name: proj_method(get_zstack(zsp, descending))
            for zsp in zstack_paths
        }
    except OSError as error:
        print(f"{su.SFM.failure}{error}")
        sys.exit()

    print("... Projections computed.")

    ### Save Z projections ###

    # Use first extension from the input directory as the output extension
    out_ext = Path(zstack_paths[0]).suffix
    if out_ext not in (".tif", ".tiff", ".png"):
        out_ext = ".tiff"

    print(f"{os.linesep}Saving projections...")
    for z_id, zproj in zprojs.items():
        if z_id.endswith(zs.TIFF_INTERIM_DIR_SUFFIX):
            shutil.rmtree(os.path.join(args.in_root, z_id))
            z_id = z_id[: -len(zs.TIFF_INTERIM_DIR_SUFFIX)]
        cv2.imwrite(
            os.path.join(args.out_root, f"{z_id}_{args.method}_{out_ext}"), zproj
        )

    print("... Projections saved.")
    print(su.SFM.success)
    print(su.END_SEPARATOR)

    if compute_cell_area:
        if "-a" in sys.argv:
            sys.argv.remove("-a")
        elif "--area" in sys.argv:
            sys.argv.remove("--area")

        script_path = defs.SCRIPT_DIR / "compute_cell_area.py"

        options = [
            arg for arg in sys.argv[1:] if arg != args.in_root and arg != args.out_root
        ]

        # use out_root as both in_root and out_root
        subprocess.run(
            [sys.executable, script_path, *options, args.out_root, args.out_root],
            check=True,
        )


if __name__ == "__main__":
    main()
