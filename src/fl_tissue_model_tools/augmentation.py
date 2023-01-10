from math import floor
from PIL import Image
import numpy as np


def elastic_distortion(images, grid_width, grid_height, magnitude, rs):
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
        images[i] = Image.fromarray(np.squeeze(img), mode = mode)

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
