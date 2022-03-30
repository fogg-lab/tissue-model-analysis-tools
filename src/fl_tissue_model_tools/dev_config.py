from dataclasses import dataclass
from matplotlib import docstring


@dataclass
class DevDirs:
    data_dir: str
    analysis_dir: str
    figures_dir: str


def get_dev_directories(dev_paths_file: str) -> DevDirs:
    with open(dev_paths_file) as fp:
        dirs = [l.strip() for l in fp.readlines()]
    return DevDirs(data_dir=dirs[0], analysis_dir=dirs[1], figures_dir=dirs[2])
