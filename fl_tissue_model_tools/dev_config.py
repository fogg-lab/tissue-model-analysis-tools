from dataclasses import dataclass
from fl_tissue_model_tools import defs

@dataclass
class DevDirs:
    """Container class for developer directory information.

    Attributes:
        data_dir: The root directory below which all other data
            are nested.
        analysis_dir: The root directory below which all analysis
            output will be saved.
        figures_dir: The root directory below which all figure
            output will be saved.

    """
    data_dir: str
    analysis_dir: str
    figures_dir: str


def get_dev_directories() -> DevDirs:
    """Get directories for creating a `DevDirs` object."""
    dev_dirs = DevDirs(
        data_dir=defs.BASE_DIR / 'data',
        analysis_dir=defs.BASE_DIR / 'analysis',
        figures_dir=defs.BASE_DIR / 'figures'
    )
    return dev_dirs
