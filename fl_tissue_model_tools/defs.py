from pathlib import Path
import sys
from platform import system
import configparser
import numpy as np

SUPPORTED_IMAGE_FORMATS = ("ND2", "TIF", "TIFF", "OME-TIFF")

# Max values for each integer type, placed here for convenience
MAX_UINT16 = np.iinfo(np.uint16).max
MAX_UINT8 = np.iinfo(np.uint8).max

# Epsilon value for floating point operations
EPSILON = np.finfo(np.float32).eps

# Base directory of this package (where it is installed) and its config file path
PKG_BASE_DIR = Path(__file__).resolve().parent
PKG_CFG_PATH = PKG_BASE_DIR / "package.cfg"

# Name of this package
try:
    _pkg_config = configparser.ConfigParser()
    _pkg_config.read(PKG_CFG_PATH)
    PKG_NAME = _pkg_config["metadata"]["name"]
except KeyError:
    PKG_NAME = "fl_tissue_model_tools"

# Paths to scripts and config files from this package
PKG_SCRIPTS_DIR = PKG_BASE_DIR / "scripts"
PKG_CONFIG_DIR = PKG_BASE_DIR / "config"
PKG_MODEL_DIR = PKG_BASE_DIR / "model_training"

# Get the user-specified base directory to store scripts, config, and output
try:
    _user_base_dir = _pkg_config[PKG_NAME]["base_dir"]
except KeyError as e:
    # Looks like the user is running the PyInstaller executable.
    _exec_dir = Path(sys.executable).parent
    is_macos = system() == "Darwin"
    if is_macos and str(_exec_dir).endswith(".app/Contents/MacOS"):
        _user_base_dir = str(_exec_dir.parent / "Resources")
    elif (_exec_dir / "_internal").is_dir():
        _user_base_dir = str(_exec_dir.parent / "_internal")
    else:
        raise e

# Expand a user-relative base directory to an absolute path for current user
if _user_base_dir.startswith("~"):
    BASE_DIR = Path.home().resolve() / _user_base_dir[2:]
else:
    BASE_DIR = Path(_user_base_dir)

### Subdirectories in the user-specified base directory

# Directory to store model config files, checkpoints, and logs
MODEL_TRAINING_DIR = BASE_DIR / "model_training"

# Config files for each script
SCRIPT_CONFIG_DIR = BASE_DIR / "config"

# Scripts from this package plus any custom scripts
SCRIPT_DIR = BASE_DIR / "scripts"

# Output directory for scripts (each script will create its own subdirectory here)
OUTPUT_DIR = BASE_DIR / "output"
