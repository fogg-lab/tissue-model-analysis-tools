"""
Create or move the base directory to store config files, custom scripts, data, and output.
Creates the following directory structure:
    fl_tissue_model_tools/
        config/
            default_branching_computation.json
            default_cell_area_computation.json
            default_invasion_depth_computation.json
        model_training/
        output/
        scripts/
            compute_branches.py
            compute_cell_area.py
            compute_inv_depth.py
            compute_zproj.py
"""

from pathlib import Path
import shutil
import sys
import re
import configparser
from urllib.request import urlretrieve

from fl_tissue_model_tools import defs

CFG_SUBDIR = "config"
SCRIPTS_SUBDIR = "scripts"
MODEL_SUBDIR = "model_training"

USER_HOME = Path.home().resolve()


def configure(target_base_dir: str = ""):
    """Create or move the base directory for config files, scripts, and data."""

    if (
        re.search("^[A-Z]:", target_base_dir)
        and ("\\" not in target_base_dir)
        and ("/" not in target_base_dir)
    ):
        # Contains a drive letter but no slashes - likely got stripped by the shell
        print(
            f"\nWARNING: Path received from the command line may be invalid: {target_base_dir}"
        )
        print(
            "If you are using a unix-style shell on Windows like Git Bash,"
            " you should type your path in one of these ways:"
        )
        print(r"""  1. Enclose the path in quotes (' or ")""")
        print(r"""  2. Use forward slashes (/) instead of single bash slashes (\)""")
        print(
            r"""  3. Use double bash slashes (\\) instead of single bash slashes (\)"""
        )
        print(
            "In any case, enclosing the path in quotes at the command line is recommended."
        )
        print("")
        # Verify the path with the user, with y/n prompt and 20 second timeout
        is_valid = input(f"Use the path '{target_base_dir}'? [y/n]: ")
        if is_valid.lower() != "y":
            print("Exiting...")
            sys.exit(1)

    default_base_dir = str(defs.BASE_DIR)
    if not default_base_dir:
        default_base_dir = str(USER_HOME / defs.PKG_NAME)

    if target_base_dir == "":
        print(f"\nEnter the preferred base directory location for {defs.PKG_NAME}.")
        print(
            "If it does not exist, it will be created. Leave empty to use the default."
        )
        target_base_dir = (
            input(f"Base directory [{default_base_dir}]: ") or default_base_dir
        )
        print("")

    prev_base_dir = str(defs.BASE_DIR)

    # Ensure that base_dir.parent exists
    if not Path(target_base_dir).parent.is_dir():
        print(
            f"Error - Parent directory does not exist: {Path(target_base_dir).parent}"
        )
        sys.exit(1)

    # Create or move the base directory
    if Path(target_base_dir).exists():
        print(f"Writing to base directory: {target_base_dir}")
    elif Path(prev_base_dir).exists():
        print(f"Moving base directory from {prev_base_dir} to {target_base_dir}")
        try:
            Path(prev_base_dir).rename(target_base_dir)
        except PermissionError:
            print(
                f"Cannot move directory {prev_base_dir} to {target_base_dir}: Permission denied"
            )
            sys.exit(1)
    else:
        print(f"Creating base directory: {target_base_dir}")
        try:
            Path(target_base_dir).mkdir(parents=True)
        except PermissionError:
            print(f"Cannot create directory {target_base_dir}: Permission denied")
            sys.exit(1)

    if not defs.PKG_CONFIG_DIR.exists():
        repo_owner = "fogg-lab"
        repo_name = "tissue-model-analysis-tools"
        base_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/"
        print(f"...Retrieving scripts and script config files from github.com/{repo_owner}/{repo_name}")
        # Must be using Pyinstaller executable. Download from GitHub
        config_dir = Path(target_base_dir) / CFG_SUBDIR
        config_dir.mkdir(exist_ok=True, parents=True)
        scripts_dir = Path(target_base_dir) / SCRIPTS_SUBDIR
        scripts_dir.mkdir(exist_ok=True)
        model_dir = Path(target_base_dir) / MODEL_SUBDIR
        model_dir.mkdir(exist_ok=True)
        best_ensemble_dir = model_dir / "best_ensemble"
        best_ensemble_dir.mkdir(exist_ok=True)
        binary_seg_checkpoints_dir = model_dir / "binary_segmentation" / "checkpoints"
        binary_seg_checkpoints_dir.mkdir(exist_ok=True, parents=True)
        binary_seg_configs_dir = model_dir / "binary_segmentation" / "configs"
        binary_seg_configs_dir.mkdir(exist_ok=True)
        cfg_files = [
            "config/default_branching_computation.json",
            "config/default_cell_area_computation.json",
            "config/default_invasion_depth_computation.json",
        ]
        for cfg_file in cfg_files:
            urlretrieve(base_url + cfg_file, config_dir / Path(cfg_file).name)
        script_files = [
            "scripts/compute_branches.py",
            "scripts/compute_cell_area.py",
            "scripts/compute_inv_depth.py",
            "scripts/compute_zproj.py",
        ]
        for script_file in script_files:
            urlretrieve(base_url + script_file, scripts_dir / Path(script_file).name)
        model_training_files = [
            "model_training/invasion_depth_best_hp.json",
            "model_training/invasion_depth_hp_space.json",
            "model_training/invasion_depth_training_values.json",
        ]
        for model_training_file in model_training_files:
            urlretrieve(
                base_url + model_training_file,
                model_dir / Path(model_training_file).name,
            )
        best_ensemble_files = [
            "model_training/best_ensemble/best_finetune_weights_0.h5",
            "model_training/best_ensemble/best_finetune_weights_1.h5",
            "model_training/best_ensemble/best_finetune_weights_2.h5",
            "model_training/best_ensemble/best_finetune_weights_3.h5",
            "model_training/best_ensemble/best_finetune_weights_4.h5",
            "model_training/best_ensemble/best_model_history_0.csv",
            "model_training/best_ensemble/best_model_history_1.csv",
            "model_training/best_ensemble/best_model_history_2.csv",
            "model_training/best_ensemble/best_model_history_3.csv",
            "model_training/best_ensemble/best_model_history_4.csv",
        ]
        for best_ensemble_file in best_ensemble_files:
            urlretrieve(
                base_url + best_ensemble_file,
                best_ensemble_dir / Path(best_ensemble_file).name,
            )
        binary_seg_checkpoints_files = [
            "model_training/binary_segmentation/checkpoints/checkpoint_1.h5",
        ]
        for binary_seg_checkpoints_file in binary_seg_checkpoints_files:
            urlretrieve(
                base_url + binary_seg_checkpoints_file,
                binary_seg_checkpoints_dir / Path(binary_seg_checkpoints_file).name,
            )
        binary_seg_configs_files = [
            "model_training/binary_segmentation/configs/unet_patch_segmentor_1.json",
        ]
        for binary_seg_configs_file in binary_seg_configs_files:
            urlretrieve(
                base_url + binary_seg_configs_file,
                binary_seg_configs_dir / Path(binary_seg_configs_file).name,
            )
    else:
        # Copy subdirectories into the base directory
        for src_dir, dest_dir in [
            (defs.PKG_CONFIG_DIR, CFG_SUBDIR),
            (defs.PKG_SCRIPTS_DIR, SCRIPTS_SUBDIR),
            (defs.PKG_MODEL_DIR, MODEL_SUBDIR),
        ]:
            dest_dir_path = Path(target_base_dir) / dest_dir
            shutil.copytree(src_dir, dest_dir_path, dirs_exist_ok=True)

    if target_base_dir != prev_base_dir:
        # Update package config file with new base_dir

        config = configparser.ConfigParser()
        config.read(defs.PKG_CFG_PATH)

        # Keep the base directory relative to current user's home directory
        config.set(
            defs.PKG_NAME, "base_dir", target_base_dir.replace(str(USER_HOME), "~")
        )
        with open(defs.PKG_CFG_PATH, "w", encoding="utf-8") as config_file:
            config.write(config_file)

    print(f"\nThe base directory for {defs.PKG_NAME} is now: {target_base_dir}")

    if str(USER_HOME) in target_base_dir and str(USER_HOME) not in str(
        defs.PKG_BASE_DIR
    ):
        # Make note about multiple users if package is installed system-wide
        print(
            f'Note: For other users, "{USER_HOME}" will be replaced with their home directory.'
        )
