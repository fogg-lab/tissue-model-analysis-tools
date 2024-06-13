from typing import Tuple
import traceback
from numbers import Integral
import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, binary_erosion
from skimage.feature import canny
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import gamma
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, resize
from skimage.draw import disk as circle


def get_superellipse_hull(x, y, n, num_iters=25000):
    """Find a superellipse that encloses the given points.
    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        n (float): Exponent of the superellipse.
        num_iters (int): Number of random parameter sets to try. Defaults to 25000.
    Returns:
        tuple: Tuple containing:
            t (float): Rotation angle of the superellipse in radians.
            d (float): Diameter of the superellipse.
            s_a (float): Scale factor for the x-axis.
            s_b (float): Scale factor for the y-axis.
            c_x (float): x-coordinate of the center of the superellipse.
            c_y (float): y-coordinate of the center of the superellipse.
    """

    # Generate array of num_iters random values for each parameter
    # Interpolate random values for the parameterss within their bounds
    linear_weights = np.random.rand(num_iters, 6)

    # Get the parameters to use for each iteration
    bounds = np.array([
        (-np.pi/20, np.pi/20),   # theta
        (0.67, 1.33),            # d
        (0.9, 1.1),              # s_a
        (0.9, 1.1),              # s_b
        (-0.3, 0.3),             # c_x
        (-0.3, 0.3)              # c_y
    ])

    param_values = (bounds[:, 1] - bounds[:, 0]) * linear_weights + bounds[:, 0]
    t, d, s_a, s_b, c_x, c_y = param_values.T[..., np.newaxis]

    # Using a general equation for a superellipse, find parameters that enclose the points
    # If the exponent (n) is even, absolute value is not needed
    # If n is 2, theta doesn't affect anything, so it can be ignored
    if n == 2:
        candidate_indices = np.where(
            np.max(
                ((x - c_x) / (d * s_a)) ** 2 + ((y - c_y) / (d * s_b)) ** 2,
            axis=1) < 1)[0]
    elif n % 2 == 0:
        candidate_indices = np.where(
            np.max(
                (((x - c_x) * np.cos(t) - ((y - c_y) * np.sin(t))) / (d * s_a)) ** n
                + (((x - c_x) * np.sin(t) + (y - c_y) * np.cos(t)) / (d * s_b)) ** n,
            axis=1) < 1)[0]
    else:
        candidate_indices = np.where(
            np.max(
                np.abs(((x - c_x) * np.cos(t) - ((y - c_y) * np.sin(t))) / (d * s_a)) ** n
                + np.abs(((x - c_x) * np.sin(t) + (y - c_y) * np.cos(t)) / (d * s_b)) ** n,
            axis=1) < 1)[0]

    t = t[candidate_indices]
    d = d[candidate_indices]
    s_a = s_a[candidate_indices]
    s_b = s_b[candidate_indices]
    c_x = c_x[candidate_indices]
    c_y = c_y[candidate_indices]

    # Find the candidate index with (roughly) the smallest area
    smallest_area_idx = np.argmin(
        4 * d**2 * s_a * s_b
        * gamma(1 + 1 / n)**2 / gamma(1 + 2 / n)
    )

    t = t[smallest_area_idx][0]
    d = d[smallest_area_idx][0]
    s_a = s_a[smallest_area_idx][0]
    s_b = s_b[smallest_area_idx][0]
    c_x = c_x[smallest_area_idx][0]
    c_y = c_y[smallest_area_idx][0]

    return t, d, s_a, s_b, c_x, c_y


def gen_superellipse_mask(t, d, s_a, s_b, c_x, c_y, n, shape) -> np.ndarray[np.bool_]:
    """Generate a superellipse mask with the given parameters.
    Args:
        t (float): Rotation angle of the superellipse in radians.
        d (float): Diameter of the superellipse.
        s_a (float): Scale factor for the x-axis.
        s_b (float): Scale factor for the y-axis.
        c_x (float): x-coordinate of the center of the superellipse.
        c_y (float): y-coordinate of the center of the superellipse.
        n (float): Exponent of the superellipse.
        shape (tuple): Shape of the mask to generate.

    """
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    X, Y = np.meshgrid(x, y)

    mask = ((np.abs(((X - c_x) * np.cos(t) - (Y - c_y) * np.sin(t)) / (d*s_a))) ** n
            + (np.abs(((X - c_x) * np.sin(t) + (Y - c_y) * np.cos(t)) / (d*s_b))) ** n
            < 1)

    # Flip so x is y and y is x
    mask = np.swapaxes(mask, 0, 1)

    return mask


def create_convex_hull_mask(array_shape: Tuple[int, int], hull_vertices: npt.NDArray) -> npt.NDArray:
    """Create a mask for a convex hull.
    Args:
        array_shape: Shape of the array for which the mask is created.
        hull_vertices (np.ndarray): Vertices of the convex hull.
    """
    # Create an empty mask with the same shape as the input array
    mask = np.zeros(array_shape, dtype=np.uint8)

    # Get the Delaunay triangulation of the convex hull points
    delaunay = Delaunay(hull_vertices)

    # Generate a meshgrid of coordinates
    coord_arrays = np.indices(array_shape).reshape(2, -1).T

    # Check if each coordinate is inside the convex hull
    mask = (delaunay.find_simplex(coord_arrays) >= 0).reshape(array_shape)

    return mask


def generate_well_mask(image: npt.NDArray, mask_val: Integral=1,
                       return_superellipse_params: bool=False) -> npt.NDArray:
    """Generate a binary mask over the well in an image.
    Args:
        image (np.ndarray): Image to create the mask for.
        mask_val (int): Value in the range [0,255] to give pixels in the well. Defaults to 1.
        return_superellipse_params (bool): Whether to return the parameters of the superellipse.
                                           Defaults to False.
    Returns:
        Union[np.ndarray, tuple]: Mask over the well. If return_superellipse_params is True, returns
                                  a tuple containing the mask and the superellipse parameters.

    """
    ### Threshold the image
    im_thresh = auto_threshold_well(image)

    # Downsample the thresholded image
    downsamp_ratio = min(1, 200 / np.max(im_thresh.shape))
    im_thresh = rescale(im_thresh, downsamp_ratio, order=0, preserve_range=True)

    ### Get the convex hull of the thresholded image

    # Get the mask's border
    im_thresh_border = canny(im_thresh)
    # Include points on the edge of the image
    im_thresh_border[0, :] += im_thresh[0, :]
    im_thresh_border[-1, :] += im_thresh[-1, :]
    im_thresh_border[:, 0] += im_thresh[:, 0]
    im_thresh_border[:, -1] += im_thresh[:, -1]

    def get_circ_mask():
        # Fall back to a circular mask if the convex hull fails (e.g. blank thresholded image)

        circ_mask = np.zeros(image.shape, dtype=np.uint8)
        center = image.shape[0] // 2, image.shape[1] // 2
        radius = int(image.shape[0] * 0.5 * (1 - 0.95))
        rr, cc = circle((center[0], center[1]), radius, shape=image.shape)
        circ_mask[rr, cc] = mask_val

        return circ_mask

    border_points = np.argwhere(im_thresh_border)
    try:
        hull = ConvexHull(border_points)
    except ValueError:
        return get_circ_mask()
    hull_vertices = border_points[hull.vertices]

    hull_verts_mask = np.zeros_like(im_thresh)
    hull_verts_mask[hull_vertices[:, 0], hull_vertices[:, 1]] = 1

    # Create a mask for the convex hull
    well_mask = create_convex_hull_mask(im_thresh.shape, hull_vertices)
    well_mask_border = canny(well_mask)
    well_mask_border[0, :] += well_mask[0, :]
    well_mask_border[-1, :] += well_mask[-1, :]
    well_mask_border[:, 0] += well_mask[:, 0]
    well_mask_border[:, -1] += well_mask[:, -1]

    ### Try fitting a superellipse to the convex hull

    # See if well is more circular or rectangular to determine the exponent
    area = np.sum(well_mask)
    perimeter = np.sum(well_mask_border)
    if perimeter / area > .027:
        # Well is more rectangular
        n = 8
    else:
        # Well is more circular
        n = 2

    # Get the parameters for the superellipse
    x = hull_vertices[:, 0] / im_thresh.shape[0] * 2 - 1
    y = hull_vertices[:, 1] / im_thresh.shape[1] * 2 - 1
    found_superellipse = False
    try:
        t, d, s_a, s_b, c_x, c_y = get_superellipse_hull(x, y, n)
        d *= 0.9
        well_mask = gen_superellipse_mask(t, d, s_a, s_b, c_x, c_y, n, im_thresh.shape)
        found_superellipse = True
    except Exception:
        traceback.print_exc()
        print("Falling back to convex hull well mask.", flush=True)

    # Prepare the mask
    well_mask = well_mask.astype(np.uint8) * mask_val
    well_mask = resize(well_mask, image.shape, order=0, preserve_range=True)

    if found_superellipse and return_superellipse_params:
        return well_mask, t, d, s_a, s_b, c_x, c_y, n

    return well_mask


def auto_threshold_well(image: npt.NDArray) -> npt.NDArray:
    """Threshold an image to get a rough mask of the well.
    Args:
        img (np.ndarray): Image to threshold.
    Returns:
        npt.ndarray: Thresholded image.

    """
    # Blur the image and get the min and max values
    im_blur = gaussian(image, sigma=1)
    im_blur = rescale_intensity(im_blur, out_range=(0, 255)).astype(np.uint8)
    im_extrema = im_blur.min(), im_blur.max()

    # Get median value of each corner of the image
    # To see if the outside of the well is lighter or darker than the inside
    corner_medians = []

    x_stop_left = int(image.shape[0] * 0.05)
    x_start_right = int(image.shape[0] * 0.95)
    y_stop_top = int(image.shape[1] * 0.05)
    y_start_bottom = int(image.shape[1] * 0.95)

    top_left_med = np.median(im_blur[:x_stop_left, :y_stop_top])
    top_right_med = np.median(im_blur[:x_stop_left, y_start_bottom:])
    bottom_left_med = np.median(im_blur[x_start_right:, :y_stop_top])
    bottom_right_med = np.median(im_blur[x_start_right:, y_start_bottom:])
    corner_medians = [top_left_med, top_right_med, bottom_left_med, bottom_right_med]
    corners_extrema = min(corner_medians), max(corner_medians)

    corners_diff_from_min = np.abs(im_extrema[0] - corners_extrema[0])
    corners_diff_from_max = np.abs(im_extrema[1] - corners_extrema[1])

    if corners_diff_from_min > corners_diff_from_max:
        # Invert the image
        im_blur = 255 - im_blur

    # Threshold the image
    thresh = threshold_otsu(im_blur)
    im_thresh = im_blur >= thresh
    im_thresh = binary_erosion(im_thresh, footprint=disk(5))

    return im_thresh
