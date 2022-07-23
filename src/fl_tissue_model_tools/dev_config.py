from dataclasses import dataclass


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


def get_dev_directories(dev_paths_file: str) -> DevDirs:
    """Get directories for creating a `DevDirs` object.

    Assumes that a file of the correct format will be located
    at `dev_paths_file`. The `dev_paths_file` should be formatted
    according as follows:

    [line 1] full/path/to/root/data/dir

    [line 2] full/path/to/root/analysis/dir

    [line 3] full/path/to/root/figures/dir

    [line 4] blank line

    Args:
        dev_paths_file: Full path to a file which defines the location
            of each `DevDir` field.

    Returns:
        A `DevDirs` object with each field filled in using the values
            in `dev_paths_file`.

    """
    with open(dev_paths_file) as fp:
        dirs = [l.strip() for l in fp.readlines()]
    return DevDirs(data_dir=dirs[0], analysis_dir=dirs[1], figures_dir=dirs[2])
