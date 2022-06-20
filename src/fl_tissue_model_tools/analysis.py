import numpy as np
import numpy.typing as npt


def compute_area_prop(img: npt.NDArray, ref_area: int, min_val: float=0) -> float:
    """Computes the proportion of pixels that are thresholded in circular area.

    Args:
        img: A masked and thresholded image. Background pixels are 0.
        ref_area: Number of pixels in the circular mask area applied to the image.
        min_val: This parameter is currently unused. Defaults to 0.

    Returns:
        Proportion of pixels in circular mask area that are thresholded.
    """    
    return np.sum(img > 0) / ref_area
