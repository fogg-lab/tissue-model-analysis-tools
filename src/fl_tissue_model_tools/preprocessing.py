import numpy as np
import cv2
from sklearn.mixture import GaussianMixture

# Typing
import numpy.typing as npt
from numpy.random import RandomState

from . import defs


def min_max_(x: npt.NDArray, a: float, b: float, mn: float, mx: float) -> npt.NDArray[np.float64]:
    '''
        Normalize the array {x} from the range [{mn}, {mx}] to the range [{a}, {b}]
    '''
    x = x.astype(np.float64)
    return a + ( (x - mn) * (b - a) ) / (mx - mn)


def gen_circ_mask(center: npt.ArrayLike, rad: float, shape: npt.ArrayLike, mask_val: np.uint8) -> npt.NDArray[np.uint8]:
    '''
        Generate a circle mask.
        The circle mask is a size {shape} array of uint8, where an element
        has value {mask_val} if it is in the circle and is 0 otherwise.
        The circle is centered at the pixel {center} and has radius {rad}.
    '''
    circ_mask = np.zeros(shape, dtype="uint8")
    return cv2.circle(circ_mask, center, rad, mask_val, cv2.FILLED)



def apply_mask(img: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    '''
        Apply the mask {mask} to an image {img}.
        Applying the mask will zero out all pixels in
        {img} where mask is false and leave all other pixels
        the same.
    '''
    return cv2.bitwise_and(img, img, mask=mask)


def exec_threshold(masked: npt.NDArray, pinhole_idx: npt.ArrayLike, sd_coef: float, rs: RandomState) -> npt.NDArray:
    '''
        Create a mask of the foreground of an image of cells in a plate.
        A two component Gaussian mixture model is fit to the intensities of the pixels,
        with the intent of one Gaussian fitting the intensities of the background pixels
        and the other fitting the intensities of the foreground pixels.
        The foreground Gaussian is the Gaussian with the larger mean.
        The foreground is determined to be all pixels of intensity greater than
        ({sd_coef} x the standard deviation of the foreground Gaussian)
        + the mean of the foreground Gaussian.
        args:
            masked (numpy array of floats) : the image to be masked
            pinhole_idx (numpy array of bools) : the indices of the all pixels in the plate
            sd_coef (float) : (see above)
            rs (RandomState) : random state for calculating Gaussian mixture
        retval:
            (numpy array of floats) : original image with all background pixels set to 0
    '''
    # Select pinhole pixels
    X = masked[pinhole_idx][:, np.newaxis]
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
    # rescale the foreground to the range [0,1]
    # gmm_masked = defs.GS_MAX*(gmm_masked - fg_thresh)/(defs.GS_MAX - fg_thresh)
    return gmm_masked

