from typing import Sequence, Any, Tuple, Dict, List, Callable, Optional
from numbers import Number, Integral
from itertools import product
import cv2
import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
import dask as d
from sklearn.mixture import GaussianMixture
from skimage.filters import median as median_filter
from skimage.morphology import disk
from skimage.feature import canny
from scipy.spatial import ConvexHull, Delaunay
from skimage.exposure import rescale_intensity, equalize_adapthist

from . import defs


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


def gen_circ_mask(center: Tuple[int, int], radius: float, shape: Tuple[int, int],
                  mask_val: Integral=1) -> npt.NDArray:
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


def create_convex_hull_mask(array_shape: Tuple[int, int], hull: ConvexHull) -> np.ndarray:
    """Create a mask for a convex hull.
    Args:
        array_shape: Shape of the array for which the mask is created.
        hull: Convex hull.
    """
    # Create an empty mask with the same shape as the input array
    mask = np.zeros(array_shape, dtype=np.uint8)

    # Get the Delaunay triangulation of the convex hull points
    delaunay = Delaunay(hull.points[hull.vertices])

    # Iterate over all coordinates in the array
    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            # Check if the current coordinate is inside the convex hull
            if delaunay.find_simplex((i, j)) >= 0:
                mask[i, j] = 1

    return mask


def score_circle(edges: npt.NDArray, hull: npt.NDArray, center: Tuple, radius: int):
    """Score a circle based on how well it fits to the outer edges.
    Args:
        edges (np.ndarray): Mask of the canny edges of the image.
        hull (npt.NDArray): Mask of the convex hull around the edges.
        center (Tuple): Center of the circle (x, y).
        radius (int): Radius of the circle.
    """

    circle_mask = gen_circ_mask(center, radius, hull.shape)

    # Mask must be completely inside the convex hull
    if np.sum(circle_mask & ~hull) > 0:
        return 0

    score = np.sum(edges & circle_mask)

    return score


def find_circle(edges: npt.NDArray, hull: npt.NDArray, radius_bounds: Tuple[int,int],
                center_x_bounds: Tuple[int,int], center_y_bounds: Tuple[int,int]
               ) -> Tuple[Tuple[int, int], int]:
    """Find the best circle that fits to the convex hull.
    Args:
        edges: Mask of the canny edges of the image.
        hull: Mask of the convex hull to fit the circle to.
        radius_bounds: min, max of the circle radius.
        center_x_bounds: min, max of the circle center on the horizontal axis.
        center_y_bounds: min, max of the circle center on the vertical axis.
        values_per_search_iteration: Number of values of each parameter to try in each iteration.
    Returns:
        The center (col, row) coordinates and the radius of the best circle.
    """

    best_circle_score = 0

    hull = hull.astype(bool)

    # starting parameters for the search for the best circle
    best_center_x = np.round(np.mean(center_x_bounds)).astype(int)
    best_center_y = np.round(np.mean(center_y_bounds)).astype(int)
    best_radius = round(np.mean(radius_bounds))

    min_center_x, max_center_x = center_x_bounds
    min_center_y, max_center_y = center_y_bounds
    min_rad, max_rad = radius_bounds

    tried_params = set()    # avoid re-evaluating the same circle parameters

    # values to try for each parameter in each iteration
    n_vals = 15

    while max_center_x != min_center_x or max_center_y != min_center_y or max_rad != min_rad:
        # try up to n_vals evenly spaced values from the search space for each parameter
        center_x_vals = np.unique(np.linspace(min_center_x, max_center_x, n_vals, dtype=int))
        center_y_vals = np.unique(np.linspace(min_center_y, max_center_y, n_vals, dtype=int))
        radius_vals = np.unique(np.linspace(min_rad, max_rad, n_vals, dtype=int))
        for center_x, center_y, radius in product(center_x_vals, center_y_vals, radius_vals):
            circle_params = (center_x, center_y, radius)
            if circle_params in tried_params:
                continue
            tried_params.add(circle_params)
            circle_score = score_circle(edges, hull, (center_x, center_y), radius)
            if circle_score > best_circle_score:
                best_circle_score = circle_score
                best_center_x, best_center_y = center_x, center_y
                best_radius = radius

        # shrink the search space
        min_center_x = round(np.mean([min_center_x, min_center_x + 1, best_center_x]))
        max_center_x = round(np.mean([max_center_x, max_center_x - 1, best_center_x]))
        min_center_y = round(np.mean([min_center_y, min_center_y + 1, best_center_y]))
        max_center_y = round(np.mean([max_center_y, max_center_y - 1, best_center_y]))
        min_rad = round(np.mean([min_rad, min_rad + 1, best_radius]))
        max_rad = round(np.mean([max_rad, max_rad - 1, best_radius]))

    return (best_center_x, best_center_y), best_radius


def get_well_mask_circle_params(img: npt.NDArray, pinhole_buffer: float=0.04,
                                auto=True) -> npt.NDArray:
    """Get the parameters of a circular mask to use for the image.
    Args:
        img (np.ndarray): Image to generate the mask for.
        pinhole_buffer (float): Buffer around the pinhole as a proportion of the image width.
        auto (bool): Whether to automatically find the best circle to fit to the outer edges.
    Returns:
        tuple: The center center coordinates (col, row) and the radius of the circle.
    """

    # Return fixed circular mask parameters if not fitting a circle to the outer edges
    def get_fixed_circ_mask():
        avg_side_len = np.mean(img.shape[:2])
        radius = (1 - pinhole_buffer * 2) * avg_side_len / 2
        center = img.shape[1] / 2, img.shape[0] / 2
        return (*center, radius, radius, 0)
    if not auto:
        return get_fixed_circ_mask()

    # Downsample image while maintaining aspect ratio (the smallest side is resized to 256)
    ds_factor = 256 / min(img.shape[:2])
    ds_hw = np.round(np.multiply(img.shape, ds_factor)).astype(int)
    img_ds = cv2.resize(img, [*ds_hw, *img.shape[2:]], interpolation=cv2.INTER_LANCZOS4)

    # Median filter and get edges
    img_ds = median_filter(img_ds, disk(3))
    img_ds = rescale_intensity(img_ds, out_range=(0, 1))
    img_ds = equalize_adapthist(img_ds)
    edges = canny(img_ds, sigma=1)

    # If there are no edges, return fixed mask
    if np.sum(edges) == 0:
        return get_fixed_circ_mask()

    # Get convex hull of edges
    edge_coords = np.argwhere(edges)
    hull = ConvexHull(edge_coords)
    well_mask = create_convex_hull_mask(edges.shape, hull)
    for _ in range(3):
        well_mask = cv2.dilate(well_mask, np.ones((3, 3)))
        well_mask = median_filter(well_mask, footprint=np.ones((3, 3)))

    # Find a circular mask that fits inside the convex hull
    # To save time we'll downsample the hull mask, fit a circle in it, rescale the parameters,
    # then look for a circle at the full size in a smaller initial search space

    hull_ds_shape = np.round(np.multiply(well_mask.shape, 0.25)).astype(int)
    hull = cv2.resize(well_mask, hull_ds_shape, interpolation=cv2.INTER_NEAREST)
    edges_ds = cv2.resize(edges.astype(np.uint8), hull.shape, interpolation=cv2.INTER_NEAREST)

    min_radius = round(np.mean(hull.shape) * 0.45)
    max_radius = round(np.mean(hull.shape) * 0.68)
    min_center_y, min_center_x = np.round(np.multiply(hull.shape, 0.25)).astype(int)
    max_center_y, max_center_x = np.round(np.multiply(hull.shape, 0.75)).astype(int)

    hull_center, hull_radius = find_circle(edges_ds, hull, (min_radius, max_radius),
                                           (min_center_x, max_center_x),
                                           (min_center_y, max_center_y))

    hull_center = np.multiply(hull_center, 4)
    hull_radius = hull_radius * 4

    min_radius = max(min_radius * 4 - 1, hull_radius - 3)
    max_radius = min(max_radius * 4 + 1, hull_radius + 3)
    min_center_x = max(min_center_x * 4 - 1, hull_center[0] - 3)
    max_center_x = min(max_center_x * 4 + 1, hull_center[0] + 3)
    min_center_y = max(min_center_y * 4 - 1, hull_center[1] - 3)
    max_center_y = min(max_center_y * 4 + 1, hull_center[1] + 3)

    mask_center, mask_radius = find_circle(edges, well_mask, (min_radius, max_radius),
                                           (min_center_x, max_center_x),
                                           (min_center_y, max_center_y))

    # rescale the mask center and radius to the full size image
    mask_center = np.round(np.divide(mask_center, ds_factor)).astype(int)
    mask_radius = mask_radius / ds_factor
    # reduce the mask radius by 5% to correct for earlier dilation
    mask_radius = mask_radius * 0.95
    # reduce the mask radius further to account for pinhole buffer
    mask_radius = round(mask_radius * (1 - pinhole_buffer * 2))

    return mask_center, mask_radius


def gen_well_mask_auto(img: npt.NDArray, pinhole_buffer: float=0.04,
                       mask_val: Optional[Integral]=None) -> npt.NDArray:
    """Generate a circular well mask for an image.
    Args:
        img: Image to generate the mask for.
        pinhole_buffer: Buffer around the pinhole as a proportion of the image width.
        mask_val (int): Value in the range [0,255] to give pixels in the circle.
    Returns:
        npt.NDArray: Mask.
    """

    # Get circle parameters
    center, radius = get_well_mask_circle_params(img, pinhole_buffer)

    # Generate mask
    well_mask = gen_circ_mask(center, radius, img.shape, mask_val)

    return well_mask


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
    foreground_thresh = min(defs.MAX_UINT8, means[foreground_dist_idx] + sds[foreground_dist_idx] * sd_coef)

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
        bin_thresh(img, defs.MAX_UINT8).round().astype(np.uint8), dist_metric, 5
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
    mask = bin_thresh(img, defs.MAX_UINT8).round().astype(np.uint8)
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
    for _ in range(blur_itr):
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

        #image_mask_pairs = d.compute([
        #    d.delayed(augmentor)(images[i], masks[i])
        #    for i in range(num_samples)])[0]
        image_mask_pairs = [augmentor(images[i], masks[i]) for i in range(num_samples)]

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
