import argparse
import sys
import os
import os.path as osp
from pathlib import Path
from glob import glob
import shutil
import json
from typing import Any, Dict, Union, List

from fl_tissue_model_tools import helper
from fl_tissue_model_tools.colored_messages import SFM
from fl_tissue_model_tools import zstacks as zs
from fl_tissue_model_tools.exceptions import ZStackInputException


DASH = "="
CHUNK_WIDTH = shutil.get_terminal_size((10, 10)).columns
END_SEPARATOR = f"{SFM.cyan}{DASH * CHUNK_WIDTH}{SFM.reset}{os.linesep}"


def section_header(title: str) -> None:
    """Print a section header.

    Args:
        title: Text to display in the header.

    """
    print(
        f"{os.linesep}{SFM.cyan}{f'[{title}]':{DASH}<{CHUNK_WIDTH}}{SFM.reset}",
        flush=True,
    )


def section_footer() -> None:
    """Print a section footer."""
    print(END_SEPARATOR, flush=True)


def parse_branching_args(arg_defaults: Dict[str, Any]) -> argparse.Namespace:
    """Parse commandline arguments to the branching script.

    Args:
        arg_defaults: Default values for the commandline arguments that have
            default values.

    Returns:
        Parsed commandline arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "in_root",
        type=str,
        help=(
            "Full path to root directory of input images. "
            "Ex: [...]/my_data/images/experiment_1_yyyy_mm_dd/"
        ),
    )

    parser.add_argument(
        "out_root",
        type=str,
        help=(
            "Full path to root directory where output will be stored. "
            "Ex: [...]/my_data/analysis_output/experiment_1_yyyy_mm_dd/. "
            "In this example, experiment_1_yyyy_mm_dd/ will be created if it "
            "does not already exist. If it does exist then the contents of "
            "experiment_1_yyyy_mm_dd/ will be overwritten."
        ),
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help=(
            "Index of color channel (starting from 0) to read from images. "
            "If no argument is supplied, images must be single channel."
        ),
    )

    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help=(
            "Index of time (starting from 0) to read from images. "
            "If no argument is supplied, images must not be time series."
        ),
    )

    parser.add_argument(
        "-w",
        "--detect-well",
        action="store_true",
        help=(
            "Auto detect the well boundary and exclude regions outside the well. "
            "This feature is only enabled when the flag is provided."
        ),
    )

    parser.add_argument(
        "--image-width-microns",
        type=float,
        default=None,
        help=(
            "Physical width in microns of the region captured by each image. "
            "If not specified, the script will try to find the pixel to micron "
            "conversion factor from the image metadata, but if it cannot be found, "
            "you must provide this value. It is equal to the the horizontal resolution "
            "of the image multiplied by the pixel size in microns."
        ),
    )

    parser.add_argument(
        "--graph-thresh-1",
        nargs="+",
        type=float,
        default=None,  # None => config file takes precedence
        help=(
            "This threshold controls how much of the morse graph is used to compute the number of branches. "
            "Lower values include more of the graph, and more branches are detected. "
            "Higher values include less of the graph, and fewer branches are detected."
            "You can provide multiple values (separated by space characters) to test multiple thresholds."
            "\nDEFAULT: If no value is specified and no config file is passed, the min branch length will be 5."
        ),
    )

    parser.add_argument(
        "--graph-thresh-2",
        nargs="+",
        type=float,
        default=None,  # None => config file takes precedence
        help=(
            "This is the threshold for connecting branches, e.g. where it is "
            "ambiguous whether two branches are part of the same component. Lower "
            "values result in more connected branches, and higher values result in "
            "more disconnections.\n"
            "You can provide multiple values (separated by space characters) to test multiple thresholds."
            "\nDEFAULT: If no value is specified and no config file is passed, the min branch length will be 10."
        ),
    )

    parser.add_argument(
        "--min-branch-length",
        type=float,
        default=None,  # None => config file takes precedence
        help=(
            "The minimum branch length (in microns) to consider.\nDEFAULT: If no value is specified "
            "and no config file is passed, the min branch length will be 12."
        ),
    )

    parser.add_argument(
        "--max-branch-length",
        type=float,
        default=None,  # None => config file takes precedence
        help=(
            "This is the maximum branch length (in microns) to consider. By default, "
            "this parameter is not included. If it is not specified, no maximum branch "
            "will be enforced."
        ),
    )

    parser.add_argument(
        "--remove-isolated-branches",
        action="store_true",
        help=(
            "Whether to remove branches that are not connected to any other branches "
            "after the network is trimmed per the branch length constraints "
            "(enforcing minimum and maximum branch lengths might isolate some "
            "branches, which may or may not be desired). This behavior is only enabled "
            "when the flag is provided."
        ),
    )

    parser.add_argument(
        "--graph-smoothing-window",
        type=float,
        default=None,
        help=(
            "This is the window size (in microns) for smoothing the branch paths. Default=12"
        ),
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=arg_defaults["default_config_path"],
        help=(
            "Full path to branching configuration file. Ex: "
            "[...]/my_data/analysis_output/experiment_1_yyyy_mm_dd/branching_config.json"
        ),
    )

    args = parser.parse_args()

    if not args.remove_isolated_branches:
        args.remove_isolated_branches = None  # None => config file takes precedence

    return _strip_quotes(args)


### Parse Arguments ###
def parse_cell_area_args(arg_defaults: Dict[str, Any]) -> argparse.Namespace:
    """Parse commandline arguments to the cell area computation script.

    Args:
        arg_defaults: Default values for the commandline arguments that have
            default values.

    Returns:
        Parsed commandline arguments.

    """
    thresh_subdir = arg_defaults["thresh_subdir"]
    calc_subdir = arg_defaults["calc_subdir"]
    default_config_path = arg_defaults["default_config_path"]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "in_root",
        type=str,
        help=(
            "Full path to root directory of input images. "
            "Ex: [...]/my_data/images/experiment_1_yyyy_mm_dd/"
        ),
    )

    parser.add_argument(
        "out_root",
        type=str,
        help=(
            "Full path to root directory where output will be stored. "
            "Ex: [...]/my_data/analysis_output/experiment_1_yyyy_mm_dd/. "
            "In this example, experiment_1_yyyy_mm_dd/ will be created if it "
            "does not already exist. If it does exist then the contents of "
            f"experiment_1_yyyy_mm_dd/{thresh_subdir}/ and "
            f"experiment_1_yyyy_mm_dd/{calc_subdir}/ will be overwritten."
        ),
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help=(
            "Index of color channel (starting from 0) to read from images. "
            "If no argument is supplied, images must be single channel."
        ),
    )

    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help=(
            "Index of time (starting from 0) to read from images. "
            "If no argument is supplied, images must not be time series."
        ),
    )

    parser.add_argument(
        "-w",
        "--detect-well",
        action="store_true",
        help=(
            "Auto detect the well boundary and exclude regions outside the well. "
            "This feature is only enabled when the flag is provided."
        ),
    )

    parser.add_argument(
        "--sd-coef",
        type=float,
        default=None,
        help="A multiplier of the foreground standard deviation used to help "
        "determine the threshold. See the capabilities notebook for details. Default=0",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=default_config_path,
        help=(
            "Full path to cell area computation configuration file. Ex: "
            "C:/my_config/cell_area_comp_config.json. If no argument supplied, "
            "default configuration will be used."
        ),
    )

    args = parser.parse_args()
    return _strip_quotes(args)


def parse_zproj_args() -> argparse.Namespace:
    """Parse commandline arguments to the Z projection script.

    Args:
        arg_defaults: Default values for the commandline arguments that have
            default values.
    Returns:
        Parsed commandline arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "in_root",
        type=str,
        help=(
            "Full path to root directory of input zstacks. Ex: "
            "[...]/my_data/z_stacks/experiment_1_yyyy_mm_dd/"
        ),
    )

    parser.add_argument(
        "out_root",
        type=str,
        help=(
            "Full path to root directory where output will be stored. Ex: "
            "[...]/my_data/z_projections/experiment_1_yyyy_mm_dd/. In this "
            "example, experiment_1_yyyy_mm_dd/ will be created if it does not "
            "already exist."
        ),
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help=(
            "Index of color channel (starting from 0) to read from images. "
            "If no argument is supplied, images must be single channel."
        ),
    )

    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help=(
            "Index of time (starting from 0) to read from images. "
            "If no argument is supplied, images must not be time series."
        ),
    )

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="max",
        choices=["min", "max", "med", "avg", "fs"],
        help=(
            "Z projection method. If no argument supplied, defaults to 'max'.\n"
            "min = Minimum intensity projection\n"
            "max = Maximum intensity projection\n"
            "med = Median intensity projection\n"
            "avg = Average intensity projection\n"
            "fs = Focus stacking"
        ),
    )

    parser.add_argument(
        "-a",
        "--area",
        action="store_true",
        help="Compute cell area after computing Z projection.",
    )

    args = parser.parse_args()
    return _strip_quotes(args)


def parse_inv_depth_args(arg_defaults: Dict[str, Any]) -> argparse.Namespace:
    """Parse commandline arguments to the invasion depth script.

    Args:
        arg_defaults: Default values for the commandline arguments that have
            default values.

    Returns:
        Parsed commandline arguments.

    """
    default_config_path = arg_defaults["default_config_path"]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "in_root",
        type=str,
        help=(
            "Full path to root directory of input zstacks. "
            "Ex: [...]/my_data/z_stacks/experiment_1_yyyy_mm_dd/. "
        ),
    )

    parser.add_argument(
        "out_root",
        type=str,
        help=(
            "Full path to root directory where output will be stored. "
            "Ex: [...]/my_data/z_projections/experiment_1_yyyy_mm_dd/. "
            "In this example, experiment_1_yyyy_mm_dd/ will be created "
            "if it does not already exist."
        ),
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help=(
            "Index of color channel (starting from 0) to read from images. "
            "If no argument is supplied, images must be single channel."
        ),
    )

    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help=(
            "Index of time (starting from 0) to read from images. "
            "If no argument is supplied, images must not be time series."
        ),
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=default_config_path,
        help=(
            "Full path to invasion depth computation configuration file. "
            "Ex: C:/my_config/inv_depth_comp_config.json. If no argument "
            "supplied, default configuration will be used."
        ),
    )

    args = parser.parse_args()
    return _strip_quotes(args)


### File/Directory Validation ###


def print_input_dir_help():
    """Print help message for input directory structure."""
    print(
        f"{SFM.info} Documentation for input directory structure can be found at the following link:\n"
        "https://github.com/fogg-lab/tissue-model-analysis-tools/tree/main?tab=readme-ov-file#image-input-directory-structure",
        flush=True,
    )


def check_input_dir_structure(input_path: str):
    """Verify input directory contains either files or subfolders of files.

    Args:
        input_path: Path to input images.

    """
    if not osp.isdir(input_path):
        print(
            f"{SFM.failure} Input data directory not found:{os.linesep}\t{input_path}",
            print_input_dir_help(),
            flush=True,
        )
        sys.exit(1)

    files = list(filter(osp.isfile, glob(osp.join(input_path, "*"))))
    dirs = list(filter(osp.isdir, glob(osp.join(input_path, "*"))))

    if not files and not dirs:
        print(f"{SFM.failure} Input directory is empty: {input_path}", flush=True)
        print_input_dir_help()
        sys.exit(1)
    if files and dirs:
        print(
            f"{SFM.failure} Input directory contains both files and subfolders: {input_path}",
            flush=True,
        )
        print_input_dir_help()
        sys.exit(1)

    # Check for nested directories
    nested_dirs = list(filter(osp.isdir, glob(osp.join(input_path, "*", "*"))))
    if nested_dirs:
        nested_dirs_str = "  \n".join(nested_dirs)
        print(
            f"{SFM.failure} Input directory contains nested subfolders:\n"
            f"{nested_dirs_str}\n",
            flush=True,
        )
        print_input_dir_help()
        sys.exit(1)


def cell_area_verify_input_dir(input_path: str) -> Dict[str, Union[str, List[str]]]:
    """Verify appropriate contents of input data directory.

    Args:
        input_path: Path to input images.

    Returns:
        A dictionary containing image names mapped to their paths.
        Z stacks stored as image sequences are represented by a list of paths.

    """
    section_header("Verifying Input Directory")

    check_input_dir_structure(input_path)

    test_path = glob(osp.join(input_path, "*"))[0]
    if os.path.isdir(test_path) or helper.get_image_dims(test_path).Z == 1:
        try:
            img_paths = zs.find_zstack_image_sequences(input_path)
            if any(len(img_seq) == 1 for img_seq in img_paths.values()):
                img_paths = {}  # not z stacks. probably projections.
        except ZStackInputException:
            img_paths = {}
    else:
        try:
            img_paths = zs.find_zstack_files(input_path)
        except ZStackInputException as exc:
            print(f"{SFM.failure} {exc}", flush=True)
            print_input_dir_help()
            sys.exit(1)

    if len(img_paths) == 0:
        img_paths = {
            Path(fp).stem: fp
            for fp in glob(osp.join(input_path, "*"))
            if helper.get_image_dims(fp).Z == 1
        }
        if len(img_paths) == 0:
            print(f"{SFM.failure}No images found in {input_path}", flush=True)
            print_input_dir_help()
            sys.exit(1)

    print(f"Found {len(img_paths)} images in:{os.linesep}\t{input_path}", flush=True)
    print(SFM.success, flush=True)
    section_footer()

    return img_paths


def cell_area_verify_output_dir(
    output_path: str, thresh_subdir: str, calc_subdir: str
) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.
        thresh_subdir: Name of subdirectory where thresholded images will be
            stored: output_path/thresh_subdir/
        calc_subdir: Name of subdirectory where computation outputs will be
            stored: output_path/calc_subdir/

    """
    section_header("Verifying Output Directory")

    if not osp.isdir(output_path):
        if osp.isfile(output_path):
            print(f"{SFM.failure} Output path is a file: {output_path}")
            sys.exit(1)
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)
        os.makedirs(output_path, exist_ok=True)
        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)
    elif len(glob(osp.join(output_path, "*"))) > 0:
        print(
            f"{SFM.warning}Output directory is not empty:{os.linesep}\t{output_path}\n"
            f"{SFM.warning}This will add to the existing contents, which might not be desired.",
            flush=True,
        )
    else:
        print(f"Found dir:{os.linesep}\t{output_path}", flush=True)

    print("Creating subdirs...", flush=True)
    print(f"\t{output_path}/{thresh_subdir}", flush=True)
    print(f"\t{output_path}/{calc_subdir}", flush=True)

    os.makedirs(f"{output_path}/{thresh_subdir}", exist_ok=True)
    os.makedirs(f"{output_path}/{calc_subdir}", exist_ok=True)

    print("... Created dirs:", flush=True)
    print(f"\t{output_path}/{thresh_subdir}", flush=True)
    print(f"\t{output_path}/{calc_subdir}", flush=True)
    print(SFM.success, flush=True)
    section_footer()


def cell_area_verify_config_file(config_path: str) -> Dict[str, Any]:
    """Verify config script for performing area computations.

    Args:
        config_path: Path to config file.

    Raises:
        FileNotFoundError: Config file not found.

    Returns:
        Dictionary containing configuration values.

    """
    section_header("Verifying Config File")

    if not osp.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf8") as config_fp:
        config = json.load(config_fp)

    print(f"Using config file: {os.linesep}\t{config_path}", flush=True)
    print(f"{os.linesep}Parameter values:", flush=True)
    for key, val in config.items():
        print(f"{key:<20}{val:>20}", flush=True)
    print(SFM.success, flush=True)
    section_footer()

    return config


def zproj_verify_output_dir(output_path: str) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.

    """
    section_header("Verifying Output Directory")

    if not osp.isdir(output_path):
        if osp.isfile(output_path):
            print(f"{SFM.failure} Output path is a file: {output_path}")
            sys.exit(1)
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)
        os.makedirs(output_path, exist_ok=True)
        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)
    elif len(glob(osp.join(output_path, "*"))) > 0:
        print(
            f"{SFM.warning}Output directory is not empty:{os.linesep}\t{output_path}\n"
            f"{SFM.warning}This will add to the existing contents, which might not be desired.",
            flush=True,
        )
    else:
        print(f"Found dir:{os.linesep}\t{output_path}", flush=True)

    print(SFM.success, flush=True)
    section_footer()


def branching_verify_output_dir(output_path: str) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.

    """
    section_header("Verifying Output Directory")

    if not osp.isdir(output_path):
        if osp.isfile(output_path):
            print(f"{SFM.failure} Output path is a file: {output_path}")
            sys.exit(1)
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)
        os.makedirs(output_path, exist_ok=True)
        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)
    elif len(glob(osp.join(output_path, "*"))) > 0:
        print(
            f"{SFM.warning}Output directory is not empty:{os.linesep}\t{output_path}\n"
            f"{SFM.warning}This will add to the existing contents, which might not be desired.",
            flush=True,
        )
    else:
        print(f"Found dir:{os.linesep}\t{output_path}", flush=True)

    print(SFM.success, flush=True)
    section_footer()


def inv_depth_verify_output_dir(output_path: str) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.

    """
    section_header("Verifying Output Directory")

    if not osp.isdir(output_path):
        if osp.isfile(output_path):
            print(f"{SFM.failure} Output path is a file: {output_path}")
            sys.exit(1)
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)
        os.makedirs(output_path, exist_ok=True)
        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)
    elif len(glob(osp.join(output_path, "*"))) > 0:
        print(
            f"{SFM.warning}Output directory is not empty:{os.linesep}\t{output_path}\n"
            f"{SFM.warning}This will add to the existing contents, which might not be desired.",
            flush=True,
        )
    else:
        print(f"Found dir:{os.linesep}\t{output_path}", flush=True)

    print(SFM.success, flush=True)
    section_footer()


def inv_depth_verify_config_file(config_path: str, n_models: int) -> Dict[str, Any]:
    """Verifies that the config file exists and is valid.

    Args:
        config_path: Path to config file.
        n_models: The number of saved models.

    Raises:
        FileNotFoundError: Raised when the config file is not found.
        AssertionError: Raised when the desired number of ensemble models is greater
                        than the number of saved models.

    Returns:
        A dictionary of config parameters.
    """
    if not osp.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config file: {os.linesep}\t{config_path}", flush=True)

    with open(config_path, "r", encoding="utf8") as config_fp:
        config = json.load(config_fp)
    n_pred_models = config["n_pred_models"]
    if not n_pred_models <= n_models:
        raise AssertionError(
            f"Desired number of ensemble members ({n_pred_models}) "
            "is greater than number of saved models."
        )

    print(f"{os.linesep}Parameter values:", flush=True)
    for key, val in config.items():
        print(f"{key:<20}{val:>20}", flush=True)
    print(SFM.success, flush=True)
    section_footer()

    return config


def _strip_quotes(args: argparse.Namespace) -> argparse.Namespace:
    """Strips quotes around in_root and out_root arguments (escaped by the interactive prompt)."""
    args.in_root = args.in_root.strip("'").strip('"')
    args.out_root = args.out_root.strip("'").strip('"')
    return args
