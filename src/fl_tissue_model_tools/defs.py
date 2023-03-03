import numpy as np
from pathlib import Path

MAX_UINT16 = np.iinfo(np.uint16).max
MAX_UINT8 = np.iinfo(np.uint8).max

PKG_ROOT = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = PKG_ROOT / '..' / '..' / 'model_training'
SCRIPT_CONFIG_DIR = PKG_ROOT / '..' / '..' / 'config'
