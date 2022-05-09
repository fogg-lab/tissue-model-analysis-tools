import argparse
import os
import shutil
import json
from glob import glob
from dataclasses import dataclass
from typing import Sequence

from . import defs, data_prep


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


def verbose_header(title: str) -> None:
    print(f"{os.linesep}{SFM.cyan}{f'[{title}]':{dash}<{chunk_width}}{SFM.reset}")


def verbose_footer() -> None:
    pass


def parse_cell_area_args(arg_defaults: dict[str, str]) -> argparse.Namespace:
    thresh_subdir = arg_defaults["thresh_subdir"]
    calc_subdir = arg_defaults["calc_subdir"]
    default_config_path = arg_defaults["default_config_path"]

    ### Parse commandline arguments ###
    parser = argparse.ArgumentParser()

    parser.add_argument("in_root", type=str, help="Full path to root directory of input images. Ex: [...]/my_data/images/experiment_1_yyyy_mm_dd/")

    parser.add_argument("out_root", type=str, help=f"Full path to root directory where output will be stored. Ex: [...]/my_data/analysis_output/experiment_1_yyyy_mm_dd/. In this example, experiment_1_yyyy_mm_dd/ will be created if it does not already exist. If it does exist then the contents of experiment_1_yyyy_mm_dd/{thresh_subdir}/ and experiment_1_yyyy_mm_dd/{calc_subdir}/ will be overwritten.")

    parser.add_argument("-c", "--config", type=str, default=default_config_path, help="Full path to cell area computation configuration file. Ex: C:/my_config/cell_area_comp_config.json. If no argument supplied, default configuration will be used.")

    parser.add_argument("-e", "--extension", type=str, default="tif", help="File extension type of images. If no argument supplied, defaults to tif.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during script execution.")

    args = parser.parse_args()
    return args


def cell_area_verify_input_dir(path: str, extension: str, verbose: bool=False) -> Sequence[str]:
    if verbose:
        verbose_header("Verifying Input Directory")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input data directory not found:{os.linesep}\t{path}")
    img_paths = [fp.replace("\\", "/") for fp in glob(f"{path}/*.{extension}")]
    if len(img_paths) == 0:
        raise FileNotFoundError(f"Input data directory contains no files with extension:{os.linesep}\t{extension}")
    if verbose:
        print(f"Found {len(img_paths)} .{extension} files in:{os.linesep}\t{path}")
        print(SFM.success)
        print(verbose_end)
    return img_paths


def cell_area_verify_output_dir(path: str, thresh_subdir: str, calc_subdir: str, verbose: bool=False) -> None:
    if verbose:
        verbose_header("Verifying Output Directory")
    if not os.path.isdir(path):
        if verbose:
            print(f"Did not find output dir:{os.linesep}\t{path}")
            print(f"Creating...")
        data_prep.make_dir(path)
        if verbose:
            print(f"... Created dir:{os.linesep}\t{path}")
    if verbose:
        print(f"Creating subdirs (overwriting if previously existed)...")
        print(f"\t{path}/{thresh_subdir}")
        print(f"\t{path}/{calc_subdir}")
    data_prep.make_dir(f"{path}/{thresh_subdir}")
    data_prep.make_dir(f"{path}/{calc_subdir}")
    if verbose:
            print(f"... Created dirs:")
            print(f"\t{path}/{thresh_subdir}")
            print(f"\t{path}/{calc_subdir}")
            print(SFM.success)
            print(verbose_end)


def cell_area_verify_config_file(path: str, verbose: bool=False):
    if verbose:
        verbose_header("Verifying Config File")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as fp:
        config = json.load(fp)
    if verbose:
        print(f"Using config file: {os.linesep}\t{path}")
        print(f"{os.linesep}Parameter values:")
        for k, v in config.items():
            print(f"{k:<20}{v:>20}")
        print(SFM.success)
        print(verbose_end)
    return config

