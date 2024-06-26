import argparse
import os
import os.path as osp
from pathlib import Path
from glob import glob
import shutil
import json
import imghdr
from dataclasses import dataclass
from typing import Sequence, Any

from typing import Dict

from . import data_prep, helper


@dataclass
class SFM:
    """Colorized messages for success/failure output using Ansi escape sequences."""

    red = "\x1b[38;5;1m\x1b[1m"
    green = "\x1b[38;5;2m\x1b[1m"
    cyan = "\x1b[38;5;6m\x1b[1m"
    yellow = "\x1b[38;5;3m\x1b[1m"
    reset = "\x1b[0m"
    success = f"{green}[SUCCESS]{reset}"
    failure = f"{red}[FAILURE]{reset}"
    warning = f"{yellow}[WARNING]{reset}"
    all_succeeded = f"{green}[ALL SUCCEEDED]{reset}"
    failures_present = f"{red}[FAILURES PRESENT]{reset}"


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
        help="Auto detect the well boundary and exclude regions outside the well.",
    )

    parser.add_argument(
        "--image-width-microns",
        type=float,
        default=None,
        help=(
            "Physical width in microns of the region captured by each image. "
            "For instance, if 1 pixel in the image corresponds to 0.8 microns, "
            "this value should equal to 0.8x the horizontal resolution of the image. "
            "If not specified, the script will attempt to infer this value from the "
            "image metadata."
        ),
    )

    parser.add_argument(
        "--graph-thresh-1",
        nargs="+",
        type=float,
        default=None,   # None => config file takes precedence
        help=(
            "This threshold controls how much of the morse graph is used to compute the number of branches. "
            "Lower values include more of the graph, and more branches are detected. "
            "Higher values include less of the graph, and fewer branches are detected."
            "You can provide multiple values (separated by space characters) to test multiple thresholds."
        ),
    )

    parser.add_argument(
        "--graph-thresh-2",
        nargs="+",
        type=float,
        default=None,   # None => config file takes precedence
        help=(
            "This is the threshold for connecting branches, e.g. where it is "
            "ambiguous whether two branches are part of the same component. Lower "
            "values result in more connected branches, and higher values result in "
            "more disconnections.\n"
            "You can provide multiple values (separated by space characters) to test multiple thresholds."
        ),
    )

    parser.add_argument(
        "--min-branch-length",
        type=float,
        default=None,   # None => config file takes precedence
        help=("The minimum branch length (in microns) to consider."),
    )

    parser.add_argument(
        "--max-branch-length",
        type=float,
        default=None,   # None => config file takes precedence
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
            "branches, which may or may not be desired)."
        ),
    )

    parser.add_argument(
        "--graph-smoothing-window",
        type=float,
        default=None,
        help=("This is the window size (in microns) for smoothing the branch paths."),
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
        args.remove_isolated_branches = None    # None => config file takes precedence

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
        help="Auto detect the well boundary and exclude regions outside the well.",
    )

    parser.add_argument(
        "--sd-coef",
        type=float,
        default=None,
        help="A multiplier of the foreground standard deviation used to help "
             "determine the threshold. See the capabilities notebook for details.",
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
def cell_area_verify_input_dir(input_path: str) -> Sequence[str]:
    """Verify appropriate contents of input data directory.

    Args:
        input_path: Path to input images.

    Raises:
        FileNotFoundError: Input data directory not found.
        FileNotFoundError: Input data directory has no images.

    Returns:
        The full paths to each relevant image in the directory.

    """
    section_header("Verifying Input Directory")

    if not osp.isdir(input_path):
        raise FileNotFoundError(
            f"Input data directory not found:{os.linesep}\t{input_path}"
        )

    # Make sure all files are valid image files
    img_paths = []
    for fp in glob(osp.join(input_path, "*")):
        if osp.isfile(fp):
            # Reading image dimensions should not raise an error
            # If image is in an unsupported format, `get_image_dims` will raise an error
            # and print out what the supported formats are.
            helper.get_image_dims(fp)
            img_paths.append(fp)

    if len(img_paths) == 0:
        raise FileNotFoundError(
            f"No files found in input directory:{os.linesep}\t{input_path}"
        )

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
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)
        data_prep.make_dir(output_path)
        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)

    print("Creating subdirs (overwriting if previously existed)...", flush=True)
    print(f"\t{output_path}/{thresh_subdir}", flush=True)
    print(f"\t{output_path}/{calc_subdir}", flush=True)

    data_prep.make_dir(f"{output_path}/{thresh_subdir}")
    data_prep.make_dir(f"{output_path}/{calc_subdir}")

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
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)

        data_prep.make_dir(output_path)

        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)

    else:
        print(f"Found dir:{os.linesep}\t{output_path}", flush=True)
        print("Clearing...", flush=True)

        # Remove previous zproj images in output directory
        for prev_output_fp in [Path(output_path) / f for f in os.listdir(output_path)]:
            if osp.isfile(prev_output_fp) and imghdr.what(prev_output_fp) is not None:
                os.remove(prev_output_fp)

        print(f"... Cleared dir:{os.linesep}\t{output_path}", flush=True)

    print(SFM.success, flush=True)
    section_footer()


def inv_depth_verify_output_dir(output_path: str) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.

    """
    section_header("Verifying Output Directory")

    if not osp.isdir(output_path):
        print(f"Did not find output dir:{os.linesep}\t{output_path}", flush=True)
        print("Creating...", flush=True)

        data_prep.make_dir(output_path)

        print(f"... Created dir:{os.linesep}\t{output_path}", flush=True)
    else:
        print(f"Found dir:{os.linesep}\t{output_path}", flush=True)
        print("Clearing...", flush=True)

        # Remove previous zproj output files
        for prev_output_filepath in [
            Path(output_path) / f for f in os.listdir(output_path)
        ]:
            if (
                osp.isfile(prev_output_filepath)
                and prev_output_filepath.suffix == ".csv"
            ):
                os.remove(prev_output_filepath)

        print(f"... Cleared dir:{os.linesep}\t{output_path}", flush=True)

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
