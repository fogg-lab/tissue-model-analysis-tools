from typing import Sequence, Any, Tuple, Dict, List, Callable, Optional
from numbers import Integral
import cv2
import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
import dask as d
from sklearn.mixture import GaussianMixture

from . import defs


def apply_mask(img: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """Apply a mask to an image.
    Args:
        img: Image to be masked.
        mask: Mask to be applied.
    Returns:
        Masked image.
    """
    masked_img = np.copy(img)
    masked_img[mask == 0] = 0
    return masked_img


def bin_thresh(
    img: npt.NDArray, img_max: npt.NDArray, threshold: float = 0
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
    img = np.where(img > threshold, img_max, 0)
    return img


def exec_threshold(
    masked: npt.NDArray,
    mask_idx: Optional[Sequence],
    sd_coef: float,
    rand_state: RandomState,
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
    if mask_idx is None:
        mask_idx = tuple(np.indices(masked.shape).reshape(2, -1))
    pixels = masked[mask_idx][:, np.newaxis]

    # print(pixels.shape, pixels.min(), pixels.max(), pixels.mean(), pixels.std())
    # print(masked.shape, masked.min(), masked.max(), masked.mean(), masked.std())

    gm = GaussianMixture(n_components=2, random_state=rand_state)
    gm = gm.fit(pixels)

    # Get GMM components
    means = gm.means_.squeeze()
    sds = np.sqrt(gm.covariances_.squeeze())

    # Get mean foreground mean & threshold value
    foreground_dist_idx = np.argmax(means)
    foreground_thresh = min(
        defs.MAX_UINT8, means[foreground_dist_idx] + sds[foreground_dist_idx] * sd_coef
    )

    # Apply threshold
    gmm_masked = np.copy(masked)
    gmm_masked = np.where(gmm_masked <= foreground_thresh, 0, gmm_masked)

    return gmm_masked


def gen_circ_mask(
    center: Tuple[int, int],
    radius: float,
    shape: Tuple[int, int],
    mask_val: Integral = 1,
) -> npt.NDArray:
    """Generate a circular mask.
    Args:
        center: Center coordinates (column, row) of the circle.
        radius: Radius of the circle.
        shape: The shape, (height, width), of the mask.
        mask_val (int): Value in the range [0,255] to give pixels in the circle.
    Returns:
        npt.NDArray: The circular mask.
    """

    circ_mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(circ_mask, center, radius, mask_val, -1)

    return circ_mask


def dt_blur(
    img: npt.NDArray, blur_itr: int, dist_metric: int = cv2.DIST_L2, k_size: int = 3
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
        bin_thresh(img, defs.MAX_UINT8).round().astype(np.uint8), dist_metric, 5
    )
    return blur(proc_img, blur_itr, k_size)


def sdt_blur(
    img: npt.NDArray, blur_itr: int, dist_metric: int = cv2.DIST_L2, k_size: int = 3
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
    mask = bin_thresh(img, defs.MAX_UINT8).round().astype(np.uint8)
    # distance of pixels in the mask
    proc_img = cv2.distanceTransform(mask, dist_metric, 5)
    # distance of pixels not in the mask
    proc_img -= cv2.distanceTransform(
        np.logical_not(mask).astype(np.uint8), dist_metric, 5
    )
    return blur(proc_img, blur_itr, k_size, gs=False)


def blur(
    img: npt.NDArray, blur_itr: int, k_size: int = 3, gs: bool = True
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
    for _ in range(blur_itr):
        proc_img = cv2.GaussianBlur(proc_img, (k_size, k_size), 0)

    if gs == False:
        return proc_img.round()
    return proc_img.round().astype(np.uint8)


def get_augmentor(augmentations: List[Callable]) -> Callable:
    """Returns a function that applies a list of augmentations to an image/mask pair."""

    def augmentor(
        image: npt.NDArray, mask: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        assert image.shape == mask.shape, "Image and mask must have the same shape."

        for aug in augmentations:
            transformed = aug(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask

    return augmentor


def get_batch_augmentor(augmentations: List[Callable]) -> Callable:
    """Returns a function that applies a list of augmentations to a batch of image/mask pairs."""

    augmentor = get_augmentor(augmentations)

    def batch_augmentor(
        images: npt.NDArray, masks: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        assert images.shape == masks.shape, "Images and masks must have the same shape."
        num_samples = images.shape[0]

        # image_mask_pairs = d.compute([
        #    d.delayed(augmentor)(images[i], masks[i])
        #    for i in range(num_samples)])[0]
        image_mask_pairs = [augmentor(images[i], masks[i]) for i in range(num_samples)]

        transformed_images, transformed_masks = zip(*image_mask_pairs)

        return np.array(transformed_images), np.array(transformed_masks)

    return batch_augmentor


def augment_invasion_imgs(
    images: npt.NDArray,
    rand_state: RandomState,
    rot_options=(0, 90, 180, 270),
    expand_dims: bool = False,
) -> npt.NDArray:
    """Transform a list of images with random flips and rotations.
    Args:
        images: Original images.
        rand_state: RandomState object to allow for reproducability.
        rot_options: Random rotation angle choices.
        expand_dims: Whether to add a depth axis to each image after augmentation steps.
    Returns:
        Transformed images.
    """

    num_images = len(images)
    rots = rand_state.choice(rot_options, size=num_images)
    hflips = rand_state.choice([True, False], size=num_images)
    vflips = rand_state.choice([True, False], size=num_images)

    def augment_img(img, idx):
        # Horizontal flip
        if hflips[idx]:
            img = cv2.flip(img, 1)

        # Vertical flip
        if vflips[idx]:
            img = cv2.flip(img, 0)

        # Rotation
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rots[idx], 1.0)
        img = cv2.warpAffine(img, rotation_matrix, img.shape[:2])

        if expand_dims:
            img = np.expand_dims(img, 2)

        return img

    images = d.compute(
        [
            d.delayed(augment_img)(
                [images[i]], rots[i], hflips[i], vflips[i], expand_dims
            )[0]
            for i in range(num_images)
        ]
    )[0]

    return np.array(images)


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
