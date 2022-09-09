from typing import Union, Sequence, Any
from typing import Tuple, Dict
import cv2
import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
import dask as d
from sklearn.mixture import GaussianMixture

from . import defs


def min_max_(
    img: npt.NDArray, new_min: float, new_max: float, old_min: float, old_max: float
) -> npt.NDArray[np.float64]:
    """Normalize the array `img` from the range [`old_min`, `old_max`]
       to the range [`new_min`, `new_max`]

    Args:
        img: The array to be normalized.
        new_min: The new min value.
        new_max: The new max value.
        old_min: The old min value.
        old_max: The old max value.

    Returns:
        The normalized array.
    """

    img = img.astype(np.float64)
    return new_min + ( (img - old_min) * (new_max - new_min) ) / (old_max - old_min)


def gen_circ_mask(
    center: Tuple[int, int], rad: float, shape: Tuple[int, int], mask_val: np.uint8
) -> npt.NDArray[np.uint8]:
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


def bin_thresh(
    img: npt.NDArray, img_max: npt.NDArray, threshold: float=0
) -> npt.NDArray:
    """Threshold an image by setting all pixels with value above `threshold` to `img_max`
    and all other pixels to 0.

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


def exec_threshold(
    masked: npt.NDArray, mask_idx: Sequence, sd_coef: float, rand_state: RandomState
) -> npt.NDArray:
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
        rand_state: A NumPy random state object to allow for reproducability.

    Returns:
        Copy of original image with background pixels set to 0.
    """
    # Select pixels within the mask. Exclude masked-out pixels since they
    # will alter the shape of the background distribution.
    X = masked[mask_idx][:, np.newaxis]
    gm = GaussianMixture(n_components=2, random_state=rand_state).fit(X)
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


def dt_blur(
    img: npt.NDArray, blur_itr: int, dist_metric: int=cv2.DIST_L2, k_size: int=3
) -> npt.NDArray[np.uint8]:
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

    proc_img = cv2.distanceTransform(
        bin_thresh(img, defs.GS_MAX).round().astype(np.uint8), dist_metric, 5
    )
    return blur(proc_img, blur_itr, k_size)


def sdt_blur(
    img: npt.NDArray, blur_itr: int, dist_metric: int=cv2.DIST_L2, k_size: int=3
) -> npt.NDArray[np.uint8]:
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
    proc_img -= cv2.distanceTransform(np.logical_not(mask).astype(np.uint8),
                                      dist_metric, 5)
    return blur(proc_img, blur_itr, k_size, gs=False)


def blur(
    img: npt.NDArray, blur_itr: int, k_size: int=3, gs: bool=True
) -> npt.NDArray[np.uint8]:
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


def augment_img(
    img: npt.NDArray[Union[np.float_, np.int_]], rot: int, hflip: bool,
    vflip: bool, expand_dims: bool=True
) -> npt.NDArray[Union[np.float_, np.int_]]:
    """Augment an image using rotations and horizontal/vertical flips

    Args:
        img: Original image.
        rot: Rotation angle for image.
        hflip: Whether to horizontally flip image.
        vflip: Whether to vertically flip image.
        expand_dims: Whether to add a depth axis to the image after
            augmentation steps.

    Returns:
        Augmented image.
    """
    hw = img.shape[:2]
    # Horizontal flip
    if hflip:
        img = cv2.flip(img, 1)
    # Vertical flip
    if vflip:
        img = cv2.flip(img, 0)
    # Rotation
    rot_mat = cv2.getRotationMatrix2D((hw[1] // 2, hw[0] // 2), rot, 1.0)
    img = cv2.warpAffine(img, rot_mat, hw)

    if expand_dims:
        img = np.expand_dims(img, 2)
    
    return img


def augment_img_mask_pairs(
    original_imgs: npt.NDArray[np.float_],
    original_masks: npt.NDArray[np.int_],
    rand_state: RandomState
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    """Augment a set of image/mask pairs using rotations and horizontal/vertical flips

    Args:
        original_imgs: Original images.
        original_masks: Original masks.
        rand_state: RandomState object to allow for reproducability.

    Returns:
        (Augmented images, matched augmented masks)
    """
    assert len(original_imgs) == len(original_masks), (
        f"x and y must have the same shape, x: {original_imgs.shape} != y: {original_masks.shape}")
    m = len(original_imgs)
    # Cannot parallelize (random state ensures reproducibility)
    rots = rand_state.choice([0, 90, 180, 270], size=m)
    hflips = rand_state.choice([True, False], size=m)
    vflips = rand_state.choice([True, False], size=m)

    def aug_imgs(imgs):
        return np.array(
            [
                augment_img(
                    imgs[i], rots[i], hflips[i], vflips[i]
                ) for i in range(m)
            ]
        )

    original_imgs, original_masks = d.compute((d.delayed(aug_imgs)(original_imgs),
                                               d.delayed(aug_imgs)(original_masks)))[0]
    return original_imgs, original_masks


def augment_imgs(
    x: npt.NDArray[np.float_],  rand_state: RandomState, rot_options=(0, 90, 180, 270),
    expand_dims: bool=False
) -> npt.NDArray[np.float_]:
    """Augment a set of images using rotations and horizontal/vertical flips.

    Args:
        x: Original images.
        rand_state: RandomState object to allow for reproducability.
        rot_options: Random rotation angle choices.
        expand_dims: Whether to add a depth axis to each image after
            augmentation steps.

    Returns:
        Augmented image set.
    """
    m = len(x)
    rots = rand_state.choice(rot_options, size=m)
    hflips = rand_state.choice([True, False], size=m)
    vflips = rand_state.choice([True, False], size=m)

    x = d.compute(
        [d.delayed(augment_img)(x[i], rots[i], hflips[i], vflips[i], expand_dims)
            for i in range(m)]
    )[0]
    return np.array(x)


def map2bin(
    lab: npt.NDArray[np.int_], fg_vals: Sequence[int], bg_vals: Sequence[int],
    fg: int=1, bg: int=0
) -> npt.NDArray[np.int_]:
    """Convert a mask n-map (e.g., trimap) to a simple binary map.

    Args:
        lab: Label mask.
        fg_vals: Foreground mask values.
        bg_vals: Background mask values.
        fg: Desired mask foreground value.
        bg: Desired mask background value.

    Returns:
        Binary mask of same dimension as input mask.
    """
    fg_mask = np.isin(lab, fg_vals)
    bg_mask = np.isin(lab, bg_vals)
    lab_c = lab.copy()
    lab_c[fg_mask] = fg
    lab_c[bg_mask] = bg
    return lab_c


def balanced_class_weights_from_counts(class_counts) -> Dict[Any, float]:
    """Create balanced weights using class counts.

    Args:
        class_counts: Counts of number of items in each class. Example:
            {c1: n_c1, c2: n_c2, ..., ck: n_ck}.

    Returns:
        dict[Any, float]: Weights for each class. Example:
            {c1: w_c1, c2: w_c2, ..., ck: w_ck}.
    """
    n = np.sum(list(class_counts.values()))
    n_c = len(class_counts.keys())
    weights = {}
    for ci, n_ci in class_counts.items():
        weights[ci] = n / (n_c * n_ci)
    return weights
