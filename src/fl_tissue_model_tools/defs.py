from pathlib import Path
import configparser
import pkg_resources
import numpy as np

MAX_UINT16 = np.iinfo(np.uint16).max
MAX_UINT8 = np.iinfo(np.uint8).max

PKG_BASE_DIR = Path(__file__).resolve().parent.parent
_setup_cfg = configparser.ConfigParser()
_setup_cfg.read(PKG_BASE_DIR / 'setup.cfg')
PKG_NAME = _setup_cfg['metadata']['name']

PKG_CFG_PATH = pkg_resources.resource_filename(PKG_NAME, f'{PKG_NAME}.cfg')
_pkg_config = configparser.ConfigParser()
_pkg_config.read(PKG_CFG_PATH)

PKG_SCRIPTS_DIR = PKG_BASE_DIR / 'scripts'
PKG_CONFIG_DIR = PKG_BASE_DIR / 'config'

_base_dir = _pkg_config[PKG_NAME]['base_dir']
if _base_dir.startswith('~'):
    BASE_DIR = Path.home().resolve() / _base_dir[2:]
else:
    BASE_DIR = Path(_base_dir)

MODEL_TRAINING_DIR = BASE_DIR / 'model_training'
SCRIPT_CONFIG_DIR = BASE_DIR / 'config'
SCRIPT_DIR = BASE_DIR / 'scripts'   # Contains scripts from this package plus any custom scripts
OUTPUT_DIR = BASE_DIR / 'output'
