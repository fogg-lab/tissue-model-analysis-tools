from pathlib import Path
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
PKG_CFG_PATH = PKG_BASE_DIR / 'package.cfg'

# Name of this package
_pkg_config = configparser.ConfigParser()
_pkg_config.read(PKG_CFG_PATH)
PKG_NAME = _pkg_config['metadata']['name']

# Paths to scripts and config files from this package
PKG_SCRIPTS_DIR = PKG_BASE_DIR / 'scripts'
PKG_CONFIG_DIR = PKG_BASE_DIR / 'config'
PKG_MODEL_DIR = PKG_BASE_DIR / 'model_training'

# Get the user-specified base directory to store scripts, config, and output
_user_base_dir = _pkg_config[PKG_NAME]['base_dir']

# Expand a user-relative base directory to an absolute path for current user
if _user_base_dir.startswith('~'):
    BASE_DIR = Path.home().resolve() / _user_base_dir[2:]
else:
    BASE_DIR = Path(_user_base_dir)

### Subdirectories in the user-specified base directory

# Directory to store model config files, checkpoints, and logs
MODEL_TRAINING_DIR = BASE_DIR / 'model_training'

# Config files for each script
SCRIPT_CONFIG_DIR = BASE_DIR / 'config'

# Scripts from this package plus any custom scripts
SCRIPT_DIR = BASE_DIR / 'scripts'

# Output directory for scripts (each script will create its own subdirectory here)
OUTPUT_DIR = BASE_DIR / 'output'
