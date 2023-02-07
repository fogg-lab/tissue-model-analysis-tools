import numpy as np
import numpy.typing as npt


def compute_area_prop(img: npt.NDArray, ref_area: int) -> float:
    """Computes the proportion of pixels that are thresholded in circular area.

    Args:
        img: A masked and thresholded image. Background pixels are 0.
        ref_area: Number of pixels in the circular mask area applied to the image.
        min_val: This parameter is currently unused. Defaults to 0.

    Returns:
        Proportion of pixels in circular mask area that are thresholded.
    """

    return np.sum(img > 0) / ref_area


def pixels_to_microns(num_pixels: float, im_width_px: int, im_width_microns: float) -> float:
    """Convert pixels to microns in specified resolution.

    Args:
        num_pixels: Number of pixels to convert to microns.
        im_width_px: Width of image in pixels.
        im_width_microns: Physical width of image region in microns.
    """

    return (im_width_microns / im_width_px) * num_pixels
