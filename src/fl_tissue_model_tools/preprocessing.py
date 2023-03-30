from typing import Sequence, Any, Tuple, Dict, List, Callable, Optional
from numbers import Number
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
from skimage.draw import ellipse
from scipy.ndimage import generate_binary_structure
from scipy.spatial import ConvexHull, Delaunay
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.measure import EllipseModel

from . import defs
from .gwdt_impl import gwdt_impl


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


def gen_circ_mask(
    center: Tuple[int, int], radius: float, shape: Tuple[int, int], mask_val: Number=True
) -> npt.NDArray:
    """Generate a circular mask.
    Args:
        center: Center (x, y) coordinates of the circle.
        radius: Radius of the circle.
        shape: The shape, (height, width), of the mask.
        mask_val: The value to give pixels in the circle. By default, the mask is a boolean array.
    Returns:
        npt.NDArray: The circular mask.
    """

    # Create coordinate grids for the mask
    y_grid, x_grid = np.ogrid[:shape[0], :shape[1]]

    # Calculate euclidean distance from center for each pixel
    distance_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid-center[1])**2)

    # Create the mask
    circ_mask = distance_from_center <= radius

    # Apply the mask value
    if isinstance(mask_val, bool):
        circ_mask = circ_mask if mask_val else ~circ_mask
    else:
        circ_mask = circ_mask * mask_val

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


def score_circle(hull: npt.NDArray, center: Tuple, radius: int):
    """Score a circle based on how well it fits to the convex hull.
    Args:
        hull (npt.NDArray): Mask of the convex hull.
        center (Tuple): Center of the circle (x, y).
        radius (int): Radius of the circle.
    """

    # Calculate IOU between the circle and the convex hull
    circle_mask = gen_circ_mask(center, radius, hull.shape)
    iou = np.sum(circle_mask & hull) / np.sum(circle_mask | hull)

    return iou


def find_circle(hull: npt.NDArray, radius_bounds: Tuple[int,int], center_x_bounds: Tuple[int,int],
                center_y_bounds: Tuple[int,int], values_per_search_iteration: int = 12
) -> Tuple[Tuple[int, int], int]:
    """Find the best circle that fits to the convex hull.
    Args:
        hull: Mask of the convex hull to fit the circle to.
        radius_bounds: min, max of the circle radius.
        center_x_bounds: min, max of the circle center on the horizontal axis.
        center_y_bounds: min, max of the circle center on the vertical axis.
        values_per_search_iteration: Number of values of each parameter to try in each iteration.
    Returns:
        The center (x, y) coordinates and the radius of the best circle.
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
    n_vals = values_per_search_iteration

    while not (max_center_x - min_center_x == 0
               and max_center_y - min_center_y == 0
               and max_rad - min_rad == 0):
        # try up to n_vals evenly spaced values from the search space for each parameter
        center_x_vals = np.unique(np.linspace(min_center_x, max_center_x, n_vals, dtype=int))
        center_y_vals = np.unique(np.linspace(min_center_y, max_center_y, n_vals, dtype=int))
        radius_vals = np.unique(np.linspace(min_rad, max_rad, n_vals, dtype=int))
        for center_x, center_y, radius in product(center_x_vals, center_y_vals, radius_vals):
            circle_params = (center_x, center_y, radius)
            if circle_params in tried_params:
                continue
            tried_params.add(circle_params)
            circle_score = score_circle(hull, (center_x, center_y), radius)
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


def gen_circ_mask_auto(
    img: npt.NDArray, pinhole_buffer=0.04, mask_val: np.uint8=1
) -> npt.NDArray[np.uint8]:
    """Generate a 2D circular well mask automatically from an image.
    Args:
        img: Image to generate the mask from.
        pinhole_buffer: Buffer around the pinhole as a proportion of the image width.
        mask_val: Value to use for the mask.
    """

    # Downsample image while maintaining aspect ratio (the smallest side is resized to 256)
    ds_ratio = 256 / min(img.shape[:2])
    ds_hw = np.round(np.multiply(img.shape, ds_ratio)).astype(int)
    img_ds = cv2.resize(img, [*ds_hw, *img.shape[2:]], interpolation=cv2.INTER_LANCZOS4)

    # Median filter and get edges
    img_ds = median_filter(img_ds, disk(3))
    img_ds = rescale_intensity(img_ds.astype(float), out_range=(0, 1))
    img_ds = equalize_adapthist(img_ds)
    edges = canny(img_ds, sigma=1)

    # Get convex hull of edges
    edge_coords = np.argwhere(edges)
    hull = ConvexHull(edge_coords)
    well_mask = create_convex_hull_mask(edges.shape, hull)
    for _ in range(3):
        well_mask = cv2.dilate(well_mask, np.ones((3, 3)))
        well_mask = median_filter(well_mask, footprint=np.ones((3, 3)))

    # Search for a circle mask that fits to the convex hull with maximum IOU
    # To save time we'll downsample the hull mask (e.g. to 64x64), fit a circle to it,
    # upsample, and then look for a circle mask in a smaller initial search space

    hull_ds_shape = np.round(np.multiply(well_mask.shape, 0.25)).astype(int)
    hull_ds = cv2.resize(well_mask, hull_ds_shape, interpolation=cv2.INTER_NEAREST)

    min_radius = round(np.mean(hull_ds.shape) * 0.48)
    max_radius = round(np.mean(hull_ds.shape) * 0.68)
    min_center_x = round(hull_ds.shape[1] * 0.25)
    max_center_x = round(hull_ds.shape[1] * 0.75)
    min_center_y = round(hull_ds.shape[0] * 0.25)
    max_center_y = round(hull_ds.shape[0] * 0.75)

    hull_ds_center, hull_ds_radius = find_circle(hull_ds, (min_radius, max_radius),
                                                 (min_center_x, max_center_x),
                                                 (min_center_y, max_center_y))

    hull_center = np.multiply(hull_ds_center, 4)
    hull_radius = hull_ds_radius * 4

    min_radius = max(min_radius * 4 - 1, hull_radius - 3)
    max_radius = min(max_radius * 4 + 1, hull_radius + 3)
    min_center_x = max(min_center_x * 4 - 1, hull_center[0] - 3)
    max_center_x = min(max_center_x * 4 + 1, hull_center[0] + 3)
    min_center_y = max(min_center_y * 4 - 1, hull_center[1] - 3)
    max_center_y = min(max_center_y * 4 + 1, hull_center[1] + 3)

    mask_center, mask_radius = find_circle(well_mask, (min_radius, max_radius),
                                            (min_center_x, max_center_x),
                                            (min_center_y, max_center_y))

    # reduce the mask radius by 5% to correct for earlier dilation
    mask_radius = round(mask_radius * 0.95)
    # reduce the mask radius further to account for pinhole buffer
    mask_radius = round(mask_radius * (1 - pinhole_buffer * 2))

    # adjust well mask
    well_mask = gen_circ_mask(mask_center, mask_radius, well_mask.shape, mask_val)

    # remove edges outside of the well mask
    edges[~well_mask] = False

    # remove edges that are within 7/8 of the radius to the center
    rr, cc = np.nonzero(edges)
    dists = np.sqrt((rr - mask_center[1])**2 + (cc - mask_center[0])**2)
    edges[rr[dists < mask_radius * 0.875], cc[dists < mask_radius * 0.875]] = 0

    # fit ellipse to remaining edges
    rr, cc = np.nonzero(edges)
    xy = np.array([cc, rr]).T
    well_ellipse = EllipseModel()
    well_ellipse.estimate(xy)

    # Replace the well mask with the ellipse at the original resolution
    xc, yc, a, b, _ = np.multiply(well_ellipse.params, 1/ds_ratio)
    rr, cc = ellipse(round(yc), round(xc), round(a), round(b), shape=img.shape[:2])
    well_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    well_mask[rr, cc] = mask_val

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


def dt_gray_weighted(img: npt.NDArray, threshold: int, structure: Optional[npt.NDArray] = None):
    """Gray-weighted distance transform
    From https://github.com/chunglabmit/gwdt/blob/master/gwdt/gwdt.py (MIT License)

    This algorithm finds the weighted manhattan distance from the background
    to every foreground point. The distance is the smallest sum of image values
    along a path. Path steps are taken in directions indicated by the structure.

    Args:
        img: Image to be transformed.
        threshold: Threshold value for foreground.
        structure: Structuring element used for the distance transform.
    Returns:
        npt.NDArray: Gray-weighted distance transform of the image.
    """

    if structure is None:
        structure = generate_binary_structure(img.ndim, 1)

    foreground_img = img - threshold
    foreground_img[foreground_img < 0] = 0

    pad_size = [(_//2, _//2) for _ in structure.shape]
    padded_img = np.pad(img, pad_size).astype(np.float32)
    d = np.mgrid[tuple([slice(-ps[0], ps[1]+1) for ps in pad_size])]
    d = d[:, structure]
    stride = []
    for idx in range(d.shape[1]):
        accumulator = 0
        for idx2 in range(d.shape[0]):
            accumulator += padded_img.strides[idx2] * d[idx2, idx] / padded_img.dtype.itemsize
        if accumulator != 0:
            stride.append(accumulator)
    strides = np.array(stride, np.int64)
    marks = np.zeros(padded_img.shape, np.uint8)
    # MARK_ALIVE = 1
    # MARK_FAR = 3
    # so False * 2 + 1 = MARK_ALIVE and
    #    True * 2 + 1 = MARK_FAR
    mark_slices = [slice(ps[0], s-ps[1])
                   for ps, s in zip(pad_size, padded_img.shape)]
    marks[tuple(mark_slices)] = (img > 0) * 2 + 1
    output = np.zeros(padded_img.shape, np.float32)
    gwdt_impl(padded_img.ravel(), output.ravel(), strides, marks.ravel())
    return output[tuple(mark_slices)]


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
