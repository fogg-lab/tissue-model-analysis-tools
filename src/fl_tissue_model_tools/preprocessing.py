from typing import Sequence, Any, Tuple, Dict, List, Callable
import random
import cv2
import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
import dask as d
from sklearn.mixture import GaussianMixture
from skimage import exposure, filters, morphology
from scipy.spatial import KDTree

from . import defs
from . import transforms


def min_max_(
    img: npt.NDArray, new_min: float, new_max: float, old_min: float, old_max: float
) -> npt.NDArray[float]:
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
    img_norm = new_min + ( (img - old_min) * (new_max - new_min) ) / (old_max - old_min)

    return img_norm


def combine_im_with_mask_dist_transform(
    img: npt.NDArray, mask: npt.NDArray, blend_exponent: float = 1
) -> npt.NDArray[float]:
    """Highlight centerlines of mask components in image using distance transform.
    Args:
        img: The image.
        mask: The binary mask.
        blend_exponent: The exponent applied to transformed mask before blending with the image.
                        For example, a value of 1.5 will highlight centerlines more prominently,
                        while a value of 0.5 will retain more detail of the original image.
    Returns:
        The image blended with the transformed mask.
    """
    img = np.copy(img)
    mask = np.copy(mask)
    mask = (mask / np.max(mask)).astype(np.uint8)
    dist_to_border = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = morphology.skeletonize(mask).astype(np.uint8)
    skel_coords = np.argwhere(skeleton)
    skeleton = KDTree(skel_coords)
    mask_coords = np.argwhere(mask)
    mask_distances_to_skeleton, _ = skeleton.query(mask_coords)
    dist_to_skeleton = np.zeros(mask.shape)
    dist_to_skeleton[mask_coords[:, 0], mask_coords[:, 1]] = mask_distances_to_skeleton
    dist_transformed = 1 - (dist_to_skeleton / (dist_to_skeleton + dist_to_border))
    dist_transformed = np.nan_to_num(dist_transformed)
    dist_transformed = np.power(dist_transformed, blend_exponent)

    return dist_transformed * img


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


def gen_circ_mask_auto(
    img: npt.NDArray, pinhole_buffer=0.04, mask_val: np.uint8 = 1
) -> npt.NDArray[np.uint8]:
    """Generate a 2D circular well mask automatically from an image."""
    img = np.copy(img)
    if img.dtype == float and (img.min() < -1 or img.max() > 1):
        mult_factor = 255 if img.max() <= 255 else 1
        img *= mult_factor
        img = np.round(img).astype(np.uint16)
    img = exposure.equalize_adapthist(img)
    img = np.round(img*255).astype(np.uint8)
    img = cv2.bilateralFilter(img, 20, 30, 30)
    im_thresh = np.zeros_like(img)
    thresh_val = round(img.max() * 0.05)
    im_thresh[img>thresh_val] = 1
    min_area = round(img.shape[0] * img.shape[1] * 0.0004)
    im_thresh = transforms.remove_small_islands(im_thresh, min_area0=min_area, min_area1=min_area)
    kernel=np.ones((3, 3), np.uint8)
    im_thresh = cv2.dilate(im_thresh,kernel, iterations=1)
    im_thresh = cv2.erode(im_thresh,kernel, iterations=1)
    edgefilter = filters.roberts(im_thresh)
    edge_rows, edge_cols = np.where(edgefilter>0)
    edge_xy = [*zip(edge_cols.tolist(), edge_rows.tolist())]
    circles = []
    radius_reduction_factor = 0.825
    inner_diameter_proportion_est = 1 - pinhole_buffer * 2
    outer_diameter_proportion_est = inner_diameter_proportion_est / radius_reduction_factor
    outer_radius_prop_est = outer_diameter_proportion_est / 2
    target_radius_proportion_range = outer_radius_prop_est - .05, outer_radius_prop_est + .05
    radius_min = img.shape[0] * target_radius_proportion_range[0]
    radius_max = img.shape[0] * target_radius_proportion_range[1]
    center_x_min, center_x_max = img.shape[1] * 0.43, img.shape[1] * 0.57
    center_y_min, center_y_max = center_x_min, center_x_max

    def get_circle(points_3: Sequence[Tuple[float, float]]):
        """Get circle parameters from 3 points: [(x,y),(x,y),(x,y)]"""
        x, y, z = [xcoord+ycoord*1j for (xcoord,ycoord) in points_3]
        w = (z-x) / (y-x)
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
        return {
            'radius': abs(c+x),
            'center_x': c.real * -1,
            'center_y': c.imag * -1
        }

    for _ in range(150):
        circle_points_3 = random.sample(edge_xy, k=3)
        try:
            circle = get_circle(circle_points_3)
        except ZeroDivisionError:
            continue
        # Make sure radius and center are in range
        if not (radius_min < circle['radius'] < radius_max):
            continue
        if not (center_x_min < circle['center_x'] < center_x_max):
            continue
        if not (center_y_min < circle['center_y'] < center_y_max):
            continue
        circles.append(circle)
    if len(circles) == 0:
        circle = {
            'radius': img.shape[0] * inner_diameter_proportion_est,
            'center_x': img.shape[1] * 0.5,
            'center_y': img.shape[0] * 0.5
        }
        circles.append(circle)
    mask_radius = np.median([circ['radius'] for circ in circles])
    mask_radius = round(mask_radius*radius_reduction_factor)
    mask_center_x = np.median([circ['center_x'] for circ in circles])
    mask_center_y = np.median([circ['center_y'] for circ in circles])
    mask_center = round(mask_center_x), round(mask_center_y)

    return gen_circ_mask(mask_center, mask_radius, edgefilter.shape, mask_val)


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
    pixels = masked[mask_idx][:, np.newaxis]
    gm = GaussianMixture(n_components=2, random_state=rand_state)
    gm = gm.fit(pixels)

    # Get GMM components
    means = gm.means_.squeeze()
    sds = np.sqrt(gm.covariances_.squeeze())

    # Get mean foreground mean & threshold value
    foreground_dist_idx = np.argmax(means)
    foreground_thresh = min(defs.GS_MAX, means[foreground_dist_idx] + sds[foreground_dist_idx] * sd_coef)

    # Apply threshold
    gmm_masked = np.copy(masked)
    gmm_masked = np.where(gmm_masked <= foreground_thresh, 0, gmm_masked)

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


def get_augmentor(augmentations: List[Callable]) -> Callable:
    """Returns a function that applies a list of augmentations to an image/mask pair."""

    def augmentor(image: npt.NDArray, mask: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        assert image.shape == mask.shape, 'Image and mask must have the same shape.'

        for aug in augmentations:
            transformed = aug(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask

    return augmentor


def get_batch_augmentor(augmentations: List[Callable]) -> Callable:
    """Returns a function that applies a list of augmentations to a batch of image/mask pairs."""

    augmentor = get_augmentor(augmentations)

    def batch_augmentor(images: npt.NDArray, masks: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        assert images.shape == masks.shape, "Images and masks must have the same shape."
        num_samples = images.shape[0]

        image_mask_pairs = d.compute([
            d.delayed(augmentor)(images[i], masks[i])
            for i in range(num_samples)])[0]

        transformed_images, transformed_masks = zip(*image_mask_pairs)

        return np.array(transformed_images), np.array(transformed_masks)

    return batch_augmentor


def augment_invasion_imgs(
    images: npt.NDArray[float],
    rand_state: RandomState,
    rot_options=(0, 90, 180, 270),
    expand_dims: bool=False
) -> npt.NDArray[float]:
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

    images = d.compute([d.delayed(augment_img)([images[i]], rots[i], hflips[i],
                  vflips[i], expand_dims)[0] for i in range(num_images)])[0]

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
