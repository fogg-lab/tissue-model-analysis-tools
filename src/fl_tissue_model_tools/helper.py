import os
from glob import glob
import imghdr
from typing import List

def get_img_paths(directory: str) -> List[str]:
    """Get all image paths in a directory.

    Args:
        directory: Path to directory containing images.

    Returns:
        A list of image paths.
    """
    unsupported_img_formats = {None, "rgb", "gif", "xbm"}
    img_paths = [fp.replace("\\", "/") for fp in glob(f"{directory}/*") if os.path.isfile(fp)]
    img_paths = [fp for fp in img_paths if imghdr.what(fp) not in unsupported_img_formats]
    return img_paths
