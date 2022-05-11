import argparse
import os
import shutil
import json
import re
from glob import glob
from dataclasses import dataclass
from typing import Sequence, Any

from . import defs, data_prep, zstacks


@dataclass
class SFM:
    red = "\u001b[31m"
    green = "\u001b[32m"
    cyan = "\u001b[36m"
    reset = "\u001b[0m"
    success = f"{green}[SUCCESS]{reset}"
    failure = f"{red}[FAILURE]{reset}"
    all_succeeded = f"{green}[ALL SUCCEEDED]{reset}"
    failures_present = f"{red}[FAILURES PRESENT]{reset}"

dash = "="
chunk_width = shutil.get_terminal_size((10, 10)).columns
verbose_end = f"{SFM.cyan}{dash * chunk_width}{SFM.reset}{os.linesep}"


### Verbose Output ###
def verbose_header(title: str) -> None:
    """Print a section header during verbose output.

    Args:
        title: Text to display in the header.

    """
    print(f"{os.linesep}{SFM.cyan}{f'[{title}]':{dash}<{chunk_width}}{SFM.reset}")


def verbose_footer() -> None:
    """Print a section footer during verbose output.
    """
    print(verbose_end)


### Parse Arguments ###
def parse_cell_area_args(arg_defaults: dict[str, Any]) -> argparse.Namespace:
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

    parser.add_argument("in_root", type=str, help="Full path to root directory of input images. Ex: [...]/my_data/images/experiment_1_yyyy_mm_dd/")

    parser.add_argument("out_root", type=str, help=f"Full path to root directory where output will be stored. Ex: [...]/my_data/analysis_output/experiment_1_yyyy_mm_dd/. In this example, experiment_1_yyyy_mm_dd/ will be created if it does not already exist. If it does exist then the contents of experiment_1_yyyy_mm_dd/{thresh_subdir}/ and experiment_1_yyyy_mm_dd/{calc_subdir}/ will be overwritten.")

    parser.add_argument("-c", "--config", type=str, default=default_config_path, help="Full path to cell area computation configuration file. Ex: C:/my_config/cell_area_comp_config.json. If no argument supplied, default configuration will be used.")

    parser.add_argument("-e", "--extension", type=str, default="tif", help="File extension type of images. If no argument supplied, defaults to tif.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during script execution.")

    args = parser.parse_args()
    return args


def parse_zproj_args(arg_defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse commandline arguments to the Z projection script.

    Args:
        arg_defaults: Default values for the commandline arguments that have
            default values.
    Returns:
        Parsed commandline arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("in_root", type=str, help="Full path to root directory of input zstacks. Ex: [...]/my_data/z_stacks/experiment_1_yyyy_mm_dd/")

    parser.add_argument("out_root", type=str, help="Full path to root directory where output will be stored. Ex: [...]/my_data/z_projections/experiment_1_yyyy_mm_dd/. In this example, experiment_1_yyyy_mm_dd/ will be created if it does not already exist.")

    parser.add_argument("-e", "--extension", type=str, default="tif", help="File extension type of images. If no argument supplied, defaults to tif.")

    parser.add_argument("-m", "--method", type=str, default="fs", choices=["min", "max", "med", "avg", "fs"], help="Z projection method. If no argument supplied, defaults to 'fs' (focus stacking).")

    parser.add_argument("-o", "--order", type=int, default=1, choices=[0, 1], help=f"Interpretation of Z stack order. 0=Ascending, 1=Descending. For Z stack of size k: -o 0 means (TOP -> BOTTOM) = (Z0 -> Zk) while -o 1 means (TOP -> BOTTOM) = (Zk -> Z0).")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during script execution.")

    args = parser.parse_args()
    return args


def parse_inv_depth_args(arg_defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse commandline arguments to the invasion depth script.

    Args:
        arg_defaults: Default values for the commandline arguments that have
            default values.
    Returns:
        Parsed commandline arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("in_root", type=str, help="Full path to root directory of input zstacks. Ex: [...]/my_data/z_stacks/experiment_1_yyyy_mm_dd/. Images must be .tif.")

    parser.add_argument("out_root", type=str, help="Full path to root directory where output will be stored. Ex: [...]/my_data/z_projections/experiment_1_yyyy_mm_dd/. In this example, experiment_1_yyyy_mm_dd/ will be created if it does not already exist.")

    parser.add_argument("-o", "--order", type=int, default=1, choices=[0, 1], help=f"Interpretation of Z stack order. 0=Ascending, 1=Descending. For Z stack of size k: -o 0 means (TOP -> BOTTOM) = (Z0 -> Zk) while -o 1 means (TOP -> BOTTOM) = (Zk -> Z0).")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during script execution.")

    args = parser.parse_args()
    return args


### File/Directory Validation ###
def cell_area_verify_input_dir(input_path: str, extension: str, verbose: bool=False) -> Sequence[str]:
    """Verify appropriate contents of input data directory.

    Directory should contain images with '.extension' suffix.

    Args:
        input_path: Path to input images.
        extension: Image file extension (e.g., 'tif').
        verbose: Whether to print verbose output.

    Raises:
        FileNotFoundError: Input data directory not found.
        FileNotFoundError: Input data directory does not contain any files
            with the appropriate file extension.

    Returns:
        The full paths to each relevant image in the directory.

    """
    if verbose:
        verbose_header("Verifying Input Directory")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input data directory not found:{os.linesep}\t{input_path}")
    img_paths = [fp.replace("\\", "/") for fp in glob(f"{input_path}/*.{extension}")]
    if len(img_paths) == 0:
        raise FileNotFoundError(f"Input data directory contains no files with extension:{os.linesep}\t{extension}")

    if verbose:
        print(f"Found {len(img_paths)} .{extension} files in:{os.linesep}\t{input_path}")
        print(SFM.success)
        verbose_footer()

    return img_paths


def cell_area_verify_output_dir(output_path: str, thresh_subdir: str, calc_subdir: str, verbose: bool=False) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.
        thresh_subdir: Name of subdirectory where thresholded images will be
            stored: output_path/thresh_subdir/
        calc_subdir: Name of subdirectory where computation outputs will be
            stored: output_path/calc_subdir/
        verbose: Whether to print verbose output.

    """
    if verbose:
        verbose_header("Verifying Output Directory")

    if not os.path.isdir(output_path):
        if verbose:
            print(f"Did not find output dir:{os.linesep}\t{output_path}")
            print(f"Creating...")
        data_prep.make_dir(output_path)
        if verbose:
            print(f"... Created dir:{os.linesep}\t{output_path}")

    if verbose:
        print(f"Creating subdirs (overwriting if previously existed)...")
        print(f"\t{output_path}/{thresh_subdir}")
        print(f"\t{output_path}/{calc_subdir}")

    data_prep.make_dir(f"{output_path}/{thresh_subdir}")
    data_prep.make_dir(f"{output_path}/{calc_subdir}")

    if verbose:
            print(f"... Created dirs:")
            print(f"\t{output_path}/{thresh_subdir}")
            print(f"\t{output_path}/{calc_subdir}")
            print(SFM.success)
            verbose_footer()


def cell_area_verify_config_file(config_path: str, verbose: bool=False) -> dict[str, Any]:
    """Verify config script for performing area computations.

    Args:
        config_path: Path to config file.
        verbose: Whether to print verbose output.

    Raises:
        FileNotFoundError: Config file not found.

    Returns:
        Dictionary containing configuration values.

    """
    if verbose:
        verbose_header("Verifying Config File")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as fp:
        config = json.load(fp)

    if verbose:
        print(f"Using config file: {os.linesep}\t{config_path}")
        print(f"{os.linesep}Parameter values:")
        for k, v in config.items():
            print(f"{k:<20}{v:>20}")
        print(SFM.success)
        verbose_footer()

    return config


def zproj_verify_input_dir(input_path: str, extension: str, verbose: bool=False) -> Sequence[str]:
    """Verify appropriate contents of input data directory.

    Input directory should contain 1 subdirectory for each Z stack with
    each of these subdirectories (Z stacks) contining all Z position images
    for that stack. These images should have a '.extension' suffix. Each
    Z stack image should have the pattern ...Z[pos]_... in its name,
    where [pos] is a number denoting the Z stack position for that image.

    Args:
        input_path: Path to input Z stacks.
        extension: Image file extension (e.g., 'tif').
        verbose: Whether to print verbose output.

    Raises:
        FileNotFoundError: Input data directory not found.
        FileNotFoundError: One or more Z stack subdirectories does not
            contain any images with '.extension' suffix.
        FileNotFoundError: One or more image files does not follow the
            specified naming convention.

    Returns:
        List of full paths for Z stack subdirectories.

    """
    if verbose:
        verbose_header("Verifying Input Directory")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input data directory not found:{os.linesep}\t{input_path}")
    zstack_paths = [fp.replace("\\", "/") for fp in glob(f"{input_path}/*")]

    if verbose:
        print(f"{'Z Stack ID':<40}{'No. Z Positions':>20}")

    for zsp in zstack_paths:
        img_paths = [fp.replace("\\", "/") for fp in glob(f"{zsp}/*.{extension}")]
        n_imgs = len(img_paths)
        if n_imgs == 0:
            raise FileNotFoundError(f"Input data directory contains Z stack subdirectory holding no files with extension:{os.linesep}\t{extension}{os.linesep}Offending subdirectory:{os.linesep}\t{zsp}")
        for ip in img_paths:
            iname = ip.split("/")[-1]
            pattern = re.search(zstacks._zpos_pattern, ip)
            if pattern is None:
                raise FileNotFoundError(f"Image file{os.linesep}\t{iname}{os.linesep}does not contain the expected pattern to denote Z position. Files must have ...Z[pos]_... in their name, where [pos] is a number denoting Z stack position.")

        if verbose:
            zsp_id = zsp.split("/")[-1]
            print(f"{zsp_id:.<40}{n_imgs:.>20}")
            
    if verbose:
        print(SFM.success)
        verbose_footer()

    return zstack_paths


def zproj_verify_output_dir(output_path: str, verbose: bool=True) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.
        verbose: Whether to print verbose output.

    """
    if verbose:
        verbose_header("Verifying Output Directory")

    if not os.path.isdir(output_path):
        if verbose:
            print(f"Did not find output dir:{os.linesep}\t{output_path}")
            print(f"Creating...")

        data_prep.make_dir(output_path)

        if verbose:
            print(f"... Created dir:{os.linesep}\t{output_path}")

    else:
        if verbose:
            print(f"Found dir:{os.linesep}\t{output_path}")
            print("Clearing...")

        data_prep.make_dir(output_path)

        if verbose:
            print(f"... Cleared dir:{os.linesep}\t{output_path}")

    if verbose:
        print(SFM.success)
        verbose_footer()


def inv_depth_verify_input_dir(input_path: str, verbose: bool=False) -> Sequence[str]:
    """Verify appropriate contents of input data directory.

    Input directory should contain 1 subdirectory for each Z stack with
    each of these subdirectories (Z stacks) contining all Z position images
    for that stack. These images should have a '.extension' suffix. Each
    Z stack image should have the pattern ...Z[pos]_... in its name,
    where [pos] is a number denoting the Z stack position for that image.

    Args:
        input_path: Path to input Z stacks.
        verbose: Whether to print verbose output.

    Raises:
        FileNotFoundError: Input data directory not found.
        FileNotFoundError: One or more Z stack subdirectories does not
            contain any images with '.extension' suffix.
        FileNotFoundError: One or more image files does not follow the
            specified naming convention.

    Returns:
        List of full paths for Z stack subdirectories.

    """
    # Only supports .tif files, currently
    extension = "tif"

    if verbose:
        verbose_header("Verifying Input Directory")
    
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input data directory not found:{os.linesep}\t{input_path}")

    zstack_paths = [fp.replace("\\", "/") for fp in glob(f"{input_path}/*")]

    if verbose:
        print(f"{'Z Stack ID':<60}{'No. Z Positions':>20}")
    
    for zsp in zstack_paths:
        img_paths = [fp.replace("\\", "/") for fp in glob(f"{zsp}/*.{extension}")]
        n_imgs = len(img_paths)
        if n_imgs == 0:
            raise FileNotFoundError(f"Input data directory contains Z stack subdirectory holding no files with extension:{os.linesep}\t{extension}{os.linesep}Offending subdirectory:{os.linesep}\t{zsp}")
        for ip in img_paths:
            iname = ip.split("/")[-1]
            pattern = re.search(zstacks._zpos_pattern, ip)
            if pattern is None:
                raise FileNotFoundError(f"Image file{os.linesep}\t{iname}{os.linesep}does not contain the expected pattern to denote Z position. Files must have ...Z[pos]_... in their name, where [pos] is a number denoting Z stack position.")

        if verbose:
            zsp_id = zsp.split("/")[-1]
            print(f"{zsp_id:.<60}{n_imgs:.>20}")

    if verbose:
        print(SFM.success)
        verbose_footer()

    return zstack_paths


def inv_depth_verify_output_dir(output_path: str, verbose: bool=True) -> None:
    """Verify output directory is either created or wiped.

    Args:
        output_path: Path to root output directory.
        verbose: Whether to print verbose output.

    """
    if verbose:
        verbose_header("Verifying Output Directory")

    if not os.path.isdir(output_path):
            if verbose:
                print(f"Did not find output dir:{os.linesep}\t{output_path}")
                print(f"Creating...")

            data_prep.make_dir(output_path)

            if verbose:
                print(f"... Created dir:{os.linesep}\t{output_path}")
    else:
        if verbose:
            print(f"Found dir:{os.linesep}\t{output_path}")
            print("Clearing...")

        data_prep.make_dir(output_path)

        if verbose:
            print(f"... Cleared dir:{os.linesep}\t{output_path}")

    if verbose:
        print(SFM.success)
        verbose_footer()
