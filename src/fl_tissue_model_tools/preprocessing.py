import cv2
import numpy as np
import dask as d
from sklearn.mixture import GaussianMixture

# Typing
import numpy.typing as npt
from numpy.random import RandomState
from typing import Union, Sequence, Any

# Custom
from . import defs


def min_max_(x: npt.NDArray[Any], a: float, b: float, mn: float, mx: float) -> npt.NDArray[np.float64]:
    """Normalize the `x` from the range [`mn`, `mx`] to the range [`a`, `b`]

    Args:
        x: The array to be normalized.
        a: The new min value.
        b: The new max value.
        mn: The old min value.
        mx: The old max value.

    Returns:
        The normalized array.

    """
    x = x.astype(np.float64)
    return a + ( (x - mn) * (b - a) ) / (mx - mn)


def gen_circ_mask(center: npt.ArrayLike, rad: float, shape: npt.ArrayLike, mask_val: np.uint8) -> npt.NDArray[np.uint8]:
    """Generate a 2D circular mask.

    The circle mask is a size `shape` array of uint8, where an element
    has value `mask_val` if it is in the circle and is 0 otherwise.
    The circle is centered at the pixel `center` and has radius `rad`.

    Args:
        center: Coordinates of the center pixel of the circle.
        rad: Radius of the circle.
        shape: The shape, (H,W), of the 2D array.
        mask_val: The value to give pixels within the circle.

    Returns:
        The masked image.

    """
    circ_mask = np.zeros(shape, dtype="uint8")
    return cv2.circle(circ_mask, center, rad, mask_val, cv2.FILLED)


def apply_mask(img: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """Apply a mask to an image.

    Args:
        img: Image to be masked.
        mask: Mask to be applied.

    Returns:
        Masked image.

    """
    return cv2.bitwise_and(img, img, mask=mask)


def bin_thresh(img: npt.NDArray, img_max: npt.NDArray, threshold: float=0) -> npt.NDArray:
    """Threshold an image by setting all pixels with value above `threshold` to `img_max`.

    Args:
        img: Image to be thresholded.
        img_max: Value to replace `True`.
        threshold: Min value for which `img_max` is substituted.

    Returns:
        Copy of original array with `img_max` replacing pixels
        with values > `threshold` and all other pixels set to 0.

    """
    img = np.copy(img)
    img = np.where(img>threshold, img_max, 0)
    return img


def exec_threshold(masked: npt.NDArray, mask_idx: npt.ArrayLike, sd_coef: float, rs: RandomState) -> npt.NDArray:
    """Apply threshold to obtain the foreground (cell content) of a plate image.

    A 2-component Gaussian mixture model is fit to pixel the intensities of
    a masked image. Intent is that one Gaussian fits background pixels and
    the other fits foreground pixels. Foreground Gaussian has the larger mean.
    Foreground pixels are defined as all pixels with intensities greater than the
    mean of the foreground Gaussian plus `sd_coef` times the standard deviation
    of the foreground Gaussian.

    Args:
        masked: Masked image.
        mask_idx: Indices of pixels within the mask boundary.
        sd_coef: Coefficient of foreground Gaussian standard deviation
            that determines foreground threshold strictness. A negative value
            means that intensities to the left of the
            foreground Gaussian's mean will be retained. 
        rs: A NumPy random state object to allow for reproducability. 

    Returns:
        Copy of original image with background pixels set to 0.

    """
    # Select pixels within the mask. Exclude masked-out pixels since they
    # will alter the shape of the background distribution.
    X = masked[mask_idx][:, np.newaxis]
    gm = GaussianMixture(n_components=2, random_state=rs).fit(X)
    # Get GMM components
    means = gm.means_.squeeze()
    sds = np.sqrt(gm.covariances_.squeeze())
    # Get mean foreground mean & threshold value
    fg_dist_idx = np.argmax(means)
    fg_thresh = min(defs.GS_MAX, means[fg_dist_idx] + sds[fg_dist_idx] * sd_coef)
    # Apply threshold
    gmm_masked = np.copy(masked)
    gmm_masked = np.where(gmm_masked <= fg_thresh, 0, gmm_masked)
    return gmm_masked


def dt_blur(img: npt.NDArray, blur_itr: int, dist_metric: int=cv2.DIST_L2, k_size: int=3) -> npt.NDArray[np.uint8]:
    """Apply distance transform and blur the image for `blur_itr` iterations.

    Args:
        img: Image to be transformed.
        blur_itr: Number of iterations to apply Gaussian blur.
        dist_metric: Distance computation to use in distance transform.
            Defaults to Euclidean.
        k_size: Kernel size for iterative Gaussian blur.

    Returns:
        Distance transformed and blurred image in grayscale format.
    """

    proc_img = cv2.distanceTransform(bin_thresh(img, defs.GS_MAX).round().astype(np.uint8), dist_metric, 5)
    return blur(proc_img, blur_itr, k_size)


def sdt_blur(img: npt.NDArray, blur_itr: int, dist_metric: int=cv2.DIST_L2, k_size: int=3) -> npt.NDArray[np.uint8]:
    """Apply the signed distance and blur the image for `blur_itr` iterations.

    The signed distance transform assigns each pixel in a binary mask
    its distance to the boundary and assigns each pixel not in the mask
    -1 times its distance to the boundary

    Args:
        img: Image to be transformed.
        blur_itr: Number of iterations to apply Gaussian blur.
        dist_metric: Distance computation to use in distance transform.
            Defaults to Euclidean.
        k_size: Kernel size for iterative Gaussian blur.

    Returns:
        Signed distance transformed and blurred image.

    """
    # all pixels of value greater than 0
    mask = bin_thresh(img, defs.GS_MAX).round().astype(np.uint8)
    # distance of pixels in the mask
    proc_img = cv2.distanceTransform(mask, dist_metric, 5)
    # distance of pixels not in the mask
    proc_img = proc_img - cv2.distanceTransform(np.logical_not(mask).astype(np.uint8), dist_metric, 5)
    return blur(proc_img, blur_itr, k_size, gs=False)


def blur(img: npt.NDArray, blur_itr: int, k_size: int=3, gs: bool=True) -> npt.NDArray[np.uint8]:
    """Blur an image for `blur_itr` iterations.

    Args:
        img: Image to be transformed.
        blur_itr: Number of iterations to apply Gaussian blur.
        k_size: Kernel size for iterative Gaussian blur.
        gs: Whether to convert resulting image to grayscale.

    Returns:
        Blurred image.

    """
    proc_img = img.copy()
    for i in range(blur_itr):
        proc_img = cv2.GaussianBlur(proc_img, (k_size, k_size), 0)

    if gs == False:
        return proc_img.round()
    return proc_img.round().astype(np.uint8)


def augment_img(img: npt.NDArray[Union[np.float_, np.int_]], rot: int, hflip: bool, vflip: bool, expand_dims: bool=True) -> npt.NDArray[Union[np.float_, np.int_]]:
    hw = img.shape[:2]
    # Horizontal flip
    if hflip:
        img = cv2.flip(img, 1)
    # Vertical flip
    if vflip:
        img = cv2.flip(img, 0)
    # Rotation
    rot_mat = cv2.getRotationMatrix2D((hw[1] // 2, hw[0] // 2), rot, 1.0)
    
    if expand_dims:
        img = np.expand_dims(cv2.warpAffine(img, rot_mat, hw), 2)
    
    return img


def augment_img_mask_pairs(x: npt.NDArray[np.float_], y: npt.NDArray[np.int_], rs: RandomState) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    assert len(x) == len(y), f"x and y must have the same shape, x: {x.shape} != y: {y.shape}"
    m = len(x)
    # Cannot parallelize (random state ensures reproducibility)
    rots = rs.choice([0, 90, 180, 270], size=m)
    hflips = rs.choice([True, False], size=m)
    vflips = rs.choice([True, False], size=m)

    def aug_imgs(imgs):
        return np.array([augment_img(imgs[i], rots[i], hflips[i], vflips[i]) for i in range(m)])

    x, y = d.compute((d.delayed(aug_imgs)(x), d.delayed(aug_imgs)(y)))[0]
    return x, y


def map2bin(lab: npt.NDArray[np.int_], fg_vals: Sequence[int], bg_vals: Sequence[int], fg: int=1, bg: int=0) -> npt.NDArray[np.int_]:
    fg_mask = np.isin(lab, fg_vals)
    bg_mask = np.isin(lab, bg_vals)
    lab_c = lab.copy()
    lab_c[fg_mask] = fg
    lab_c[bg_mask] = bg
    return lab_c
