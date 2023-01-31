import warnings
from math import floor
from PIL import Image
import numpy as np
from typing import Tuple, List, Union, Callable
from skimage import measure, morphology


def get_elastic_dual_transform(
    grid_width_range: Tuple[int, int],
    grid_height_range: Tuple[int, int],
    magnitude_range: Tuple[int, int],
    rs: np.random.RandomState=None
) -> Callable:
    """Return a function that performs elastic distortion on an image and mask"""

    if rs is None:
        rs = np.random.RandomState()

    def elastic_dual_transform(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform elastic distortion on an image and mask"""
        if image.ndim == 2:
            image = image[..., None]
        if mask.ndim == 2:
            mask = mask[..., None]
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape")

        grid_width = rs.randint(grid_width_range[0], grid_width_range[1]+1)
        grid_height = rs.randint(*grid_height_range[0], grid_height_range[1]+1)
        magnitude = rs.randint(*magnitude_range[0], magnitude_range[1]+1)

        image, mask = elastic_distortion([image, mask], grid_width, grid_height, magnitude, rs)

        return {'image': image, 'mask': mask}

    return elastic_dual_transform


def elastic_distortion(images, grid_width=None, grid_height=None, magnitude=8, rs=None):
    """
    Elastic distortion operation from the Augmentor library

    Source:
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py

    Distorts the passed image(s) according to the parameters supplied during
    instantiation, returning the newly distorted image.

    :param images: The image(s) to be distorted.
    :type images: List containing.
    :return: List of transformed images.
    """

    extra_dim = [False] * len(images)
    redundant_dims = [False] * len(images)
    dtypes = [img.dtype for img in images]

    # Convert numpy arrays to PIL images
    for i, img in enumerate(images):
        mode = "L" if img.dtype == np.uint8 or np.max(img) <= 255 else "I"
        if img.ndim == 3 and img.shape[2] > 1:
            redundant_dims[i] = True
            img = img[:, :, 0]
        elif img.ndim == 3:
            extra_dim[i] = True
        if dtypes[i] != np.uint8:
            dtype = np.uint8 if np.max(img) <= 255 else np.uint16
            img = img.astype(dtype)
        images[i] = Image.fromarray(np.squeeze(img), mode=mode)

    width, height = images[0].size

    horizontal_tiles = grid_width
    vertical_tiles = grid_height

    width_of_square = int(floor(width / float(horizontal_tiles)))
    height_of_square = int(floor(height / float(vertical_tiles)))

    width_of_last_square = width - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = height - (height_of_square * (vertical_tiles - 1))

    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])

    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

    last_row = range(horizontal_tiles * vertical_tiles - horizontal_tiles,
                     horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for a, b, c, d in polygon_indices:
        dx = rs.randint(-magnitude, magnitude)
        dy = rs.randint(-magnitude, magnitude)

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1,
                        x2, y2,
                        x3 + dx, y3 + dy,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1,
                        x2 + dx, y2 + dy,
                        x3, y3,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1,
                        x2, y2,
                        x3, y3,
                        x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy,
                        x2, y2,
                        x3, y3,
                        x4, y4]

    generated_mesh = []
    for i, dim in enumerate(dimensions):
        generated_mesh.append([dim, polygons[i]])

    def do_transform(image):
        return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

    augmented_images = []

    for image in images:
        augmented_images.append(do_transform(image))

    for i, augmented_img in enumerate(augmented_images):
        # Convert PIL image back to numpy array
        augmented_images[i] = np.asarray(augmented_img).astype(dtypes[i])
        if extra_dim[i]:
            augmented_images[i] = np.expand_dims(augmented_images[i], axis=2)
        elif redundant_dims[i]:
            augmented_images[i] = np.repeat(augmented_images[i][:, :, np.newaxis], 3, axis=2)

    return augmented_images

def remove_small_islands(mask, min_area0=100, min_area1=100, connectivity0=1, connectivity1=1):
    '''Remove small islands from a binary mask.

    Args:
        mask (np.ndarray): binary mask of values 0 and 1
        min_area0 (int): minimum area of islands with value 0 to change to 1
        min_area1 (int): minimum area of islands with value 1 to change to 0
        connectivity0 (int): pixel connectivity to include in islands with value 0
        connectivity1 (int): pixel connectivity to include in islands with value 1

    '''
    if np.min(mask) != 0 or np.max(mask) > 1:
        raise ValueError('this function expects a binary mask of values 0 and 1')

    mask = mask.copy()
    inverse_mask = 1 - mask
    labeled_regions_inverse = measure.label(inverse_mask, connectivity=connectivity0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labeled_inverse_regions = morphology.remove_small_objects(labeled_regions_inverse, min_size=min_area0)

    mask[labeled_inverse_regions == 0] = 1
    labeled_regions = measure.label(mask, connectivity=connectivity1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labeled_regions = morphology.remove_small_objects(labeled_regions, min_size=min_area1)

    mask[labeled_regions == 0] = 0
    return mask
