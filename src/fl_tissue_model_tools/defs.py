from pathlib import Path
import configparser
import pkg_resources
import numpy as np

MAX_UINT16 = np.iinfo(np.uint16).max
MAX_UINT8 = np.iinfo(np.uint8).max

_PKG_BASE_DIR = Path(__file__).resolve().parent.parent
_SETUP_CFG = configparser.ConfigParser()
_SETUP_CFG.read(_PKG_BASE_DIR / 'setup.cfg')
PKG_NAME = _SETUP_CFG['metadata']['name']

_PKG_CFG_PATH = pkg_resources.resource_filename(PKG_NAME, f'{PKG_NAME}.cfg')
_PKG_CONFIG = configparser.ConfigParser()
_PKG_CONFIG.read(_PKG_CFG_PATH)

PKG_SCRIPTS_DIR = _PKG_BASE_DIR / 'scripts'
PKG_CONFIG_DIR = _PKG_BASE_DIR / 'config'

BASE_DIR = _PKG_CONFIG[PKG_NAME]['base_dir']
if BASE_DIR.startswith('~'):
    BASE_DIR = Path.home().resolve() / BASE_DIR[2:]
else:
    BASE_DIR = Path(BASE_DIR)

MODEL_TRAINING_DIR = BASE_DIR / 'model_training'
SCRIPT_CONFIG_DIR = BASE_DIR / 'config'
SCRIPT_DIR = BASE_DIR / 'scripts'   # Contains scripts from this package plus any custom scripts
OUTPUT_DIR = BASE_DIR / 'output'
