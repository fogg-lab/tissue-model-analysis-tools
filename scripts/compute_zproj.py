import os
from glob import glob
from pathlib import Path
import subprocess
import sys
import numpy as np
import cv2

from fl_tissue_model_tools import defs
from fl_tissue_model_tools import script_util as su
from fl_tissue_model_tools import success_fail_messages as SFM
from fl_tissue_model_tools import zstacks as zs
from fl_tissue_model_tools import helper
from fl_tissue_model_tools.scripts import compute_cell_area


proj_methods = {
    "min": zs.proj_min,
    "max": zs.proj_max,
    "med": zs.proj_med,
    "avg": zs.proj_avg,
    "fs": zs.proj_focus_stacking,
}


def main(args=None):
    """Computes z projections and saves to output directory."""
    if args is None:
        args = su.parse_zproj_args()
        args_prespecified = False
    else:
        args_prespecified = True
    try:
        compute_area_after_zproj = args.area
    except AttributeError:
        compute_area_after_zproj = False

    ### Verify input source ###
    if os.path.isfile(args.in_root):
        print(f"{SFM.failure} Input directory is a file: {args.in_root}", flush=True)
        sys.exit(1)

    if not os.path.isdir(args.in_root):
        print(
            f"{SFM.failure} Input directory does not exist: {args.in_root}",
            flush=True,
        )
        sys.exit(1)

    zstack_paths = glob(os.path.join(args.in_root, "*"))

    if len(zstack_paths) == 0:
        print(f"{SFM.failure} Input directory is empty: {args.in_root}", flush=True)
        sys.exit(1)

    test_path = zstack_paths[0]
    if os.path.isdir(test_path) or helper.get_image_dims(test_path).Z == 1:
        zstack_paths = zs.find_zstack_image_sequences(args.in_root)
    else:
        zstack_paths = zs.find_zstack_files(args.in_root)

    ### Verify output destination ###
    try:
        su.zproj_verify_output_dir(args.out_root)
    except PermissionError as error:
        print(f"{SFM.failure} {error}", flush=True)
        sys.exit(1)

    ### Compute Z projections ###
    su.section_header("Constructing Z projections")

    proj_method = proj_methods[args.method]
    print("Loading and computing Z stacks...", flush=True)
    try:
        # zprojs: A dictionary of Z projections, keyed by Z stack ID.
        zprojs = {
            zs_id: proj_method(helper.load_image(zsp, args.time, args.channel)[0])
            for (zs_id, zsp) in zstack_paths.items()
        }
    except OSError as error:
        print(f"{SFM.failure}{error}", flush=True)
        sys.exit(1)

    print("... Projections computed.", flush=True)

    ### Save Z projections ###

    # Use first extension from the input directory as the output extension
    out_ext = Path(np.atleast_1d(list(zstack_paths.values())[0])[0]).suffix
    if out_ext not in (".tif", ".tiff", ".png"):
        out_ext = ".tiff"

    print(f"{os.linesep}Saving projections...", flush=True)
    for z_id, zproj in zprojs.items():
        img_id = z_id.replace("/", "_").replace("\\", "_")
        filename = f"{img_id}_{args.method}{out_ext}"
        save_path = os.path.join(args.out_root, filename)
        save_path = helper.get_unique_output_filepath(save_path)
        cv2.imwrite(save_path, zproj)

    print("... Projections saved.", flush=True)
    print(SFM.success, flush=True)
    print(su.END_SEPARATOR, flush=True)

    if compute_area_after_zproj:
        if args_prespecified:
            compute_cell_area.main(args)
        else:
            if "-a" in sys.argv:
                sys.argv.remove("-a")
            elif "--area" in sys.argv:
                sys.argv.remove("--area")

            script_path = defs.SCRIPT_DIR / "compute_cell_area.py"

            options = [
                arg
                for arg in sys.argv[1:]
                if arg != args.in_root and arg != args.out_root
            ]

            # use out_root as both in_root and out_root
            subprocess.run(
                [sys.executable, script_path, *options, args.out_root, args.out_root],
                check=True,
            )


if __name__ == "__main__":
    main()
