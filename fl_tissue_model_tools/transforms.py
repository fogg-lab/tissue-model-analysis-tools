import warnings
from math import floor
from typing import Tuple, Callable, Optional
from PIL import Image
import numpy as np
import numpy.typing as npt
from skimage.filters import median as median_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label as measure_label
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects, disk, skeletonize
import networkx as nx
import cv2


def get_elastic_dual_transform(
    grid_width_range: Tuple[int, int] = [4, 8],
    grid_height_range: Tuple[int, int] = [4, 8],
    magnitude_range: Tuple[int, int] = [7, 9],
    rs: np.random.RandomState = None,
    p: float = 0.9,
) -> Callable:
    """Return a function that performs elastic distortion on an image and mask"""

    if rs is None:
        rs = np.random.RandomState()

    def elastic_dual_transform(
        image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform elastic distortion on an image and mask"""
        if rs.rand() > p:
            return {"image": image, "mask": mask}
        grid_width = rs.randint(grid_width_range[0], grid_width_range[1] + 1)
        grid_height = rs.randint(grid_height_range[0], grid_height_range[1] + 1)
        magnitude = rs.randint(magnitude_range[0], magnitude_range[1] + 1)

        image, mask = elastic_distortion(
            [image, mask], grid_width, grid_height, magnitude, rs
        )

        # apply median blur to mask
        mask = cv2.medianBlur(mask, 5)

        return {"image": image, "mask": mask}

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
    max_vals = [img.max() for img in images]

    # Convert numpy arrays to PIL images
    for i, img in enumerate(images):
        if img.ndim == 3 and img.shape[2] > 1:
            redundant_dims[i] = True
            img = img[:, :, 0]
        elif img.ndim == 3:
            extra_dim[i] = True

        img = img.astype(np.float32)
        images[i] = Image.fromarray(np.squeeze(img), mode="F")

    width, height = images[0].size

    horizontal_tiles = grid_width
    vertical_tiles = grid_height

    width_of_square = floor(width / float(horizontal_tiles))
    height_of_square = floor(height / float(vertical_tiles))

    width_of_last_square = width - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = height - (height_of_square * (vertical_tiles - 1))

    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            dimensions_1 = horizontal_tile * width_of_square
            dimensions_2 = vertical_tile * height_of_square
            if horizontal_tile == (horizontal_tiles - 1):
                dimensions_3 = width_of_last_square + dimensions_1
            else:
                dimensions_3 = width_of_square + dimensions_1
            if vertical_tile == (vertical_tiles - 1):
                dimensions_4 = height_of_last_square + (
                    height_of_square * vertical_tile
                )
            else:
                dimensions_4 = height_of_square + (height_of_square * vertical_tile)
            dimensions.append([dimensions_1, dimensions_2, dimensions_3, dimensions_4])

    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

    last_row = range(
        horizontal_tiles * vertical_tiles - horizontal_tiles,
        horizontal_tiles * vertical_tiles,
    )

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append(
                [i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles]
            )

    polygons = np.array(polygons)

    for a, b, c, d in polygon_indices:
        dx = rs.randint(-magnitude, magnitude)
        dy = rs.randint(-magnitude, magnitude)
        polygons[a][4:6] += (dx, dy)
        polygons[b][2:4] += (dx, dy)
        polygons[c][6:8] += (dx, dy)
        polygons[d][0:2] += (dx, dy)

    generated_mesh = []
    for i, dim in enumerate(dimensions):
        generated_mesh.append([dim, polygons[i]])

    def do_transform(image):
        return image.transform(
            image.size, Image.MESH, generated_mesh, resample=Image.Resampling.BICUBIC
        )

    augmented_images = []

    for image in images:
        augmented_images.append(do_transform(image))

    for i, augmented_img in enumerate(augmented_images):
        # Convert PIL image back to numpy array
        augmented_img = np.asarray(augmented_img)
        if extra_dim[i]:
            augmented_img = np.expand_dims(augmented_img, axis=2)
        elif redundant_dims[i]:
            augmented_img = np.repeat(augmented_img[:, :, np.newaxis], 3, axis=2)
        # clip to original range and restore dtype
        augmented_img = np.clip(augmented_img, 0, max_vals[i])
        if np.issubdtype(dtypes[i], np.integer):
            augmented_img = np.round(augmented_img)
        augmented_images[i] = augmented_img.astype(dtypes[i])

    return augmented_images


def remove_small_islands(
    mask, min_area0=100, min_area1=100, connectivity0=1, connectivity1=1
):
    """Remove small islands from a binary mask.

    Args:
        mask (np.ndarray): binary mask of values 0 and 1
        min_area0 (int): minimum area of islands with value 0 to change to 1
        min_area1 (int): minimum area of islands with value 1 to change to 0
        connectivity0 (int): pixel connectivity to include in islands with value 0
        connectivity1 (int): pixel connectivity to include in islands with value 1

    """
    if np.min(mask) != 0 or np.max(mask) > 1:
        raise ValueError("this function expects a binary mask of values 0 and 1")

    mask = mask.copy()
    inverse_mask = 1 - mask
    labeled_regions_inverse = measure_label(inverse_mask, connectivity=connectivity0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labeled_inverse_regions = remove_small_objects(
            labeled_regions_inverse, min_size=min_area0
        )

    mask[labeled_inverse_regions == 0] = 1
    labeled_regions = measure_label(mask, connectivity=connectivity1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labeled_regions = remove_small_objects(
            labeled_regions, min_size=min_area1
        )

    mask[labeled_regions == 0] = 0
    return mask


def nx_graph_from_binary_skeleton(skeleton: npt.NDArray) -> nx.Graph:
    """
    Create a weighted, undirected networkx graph from a binary skeleton image.
    The graph will have a 'physical_pos' attribute that maps node ids to their
    physical coordinates in the skeleton image.

    Args:
        skeleton (npt.NDArray): binary skeleton image
    Returns:
        nx.Graph: weighted, undirected graph
    """
    skeleton = skeleton.astype(bool)
    g = nx.Graph()

    # get physical positions of nodes and add them to the graph
    node_pos = np.argwhere(skeleton)
    g.graph["physical_pos"] = node_pos

    # if skeleton is empty, return empty graph
    if len(node_pos) == 0:
        return g

    # get node labels
    node_labels = np.full(skeleton.shape, -1)
    node_labels[node_pos[:, 0], node_pos[:, 1]] = np.arange(node_pos.shape[0])
    edge_connected_nodes = np.zeros(skeleton.shape, dtype=bool)
    weighted_edges = []

    # function to shift the skeleton (pad sides, crop opposite sides from the padding)
    def shift_2d(arr: npt.NDArray, pad_vals: npt.NDArray) -> npt.NDArray:
        padded = np.pad(arr, pad_vals)
        pad_bottom, pad_right = pad_vals[0, 1], pad_vals[1, 1]
        h, w = arr.shape
        shifted = padded[pad_bottom : (h + pad_bottom), pad_right : (w + pad_right)]
        return shifted

    # shift skeleton in each possible edge direction to find connected nodes
    # intersection of shifted skeleton and original skeleton gives dest nodes
    # dest nodes shifted back to original position gives src nodes
    for shift_rows, shift_cols in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        ## find skeleton nodes connected by an edge for the current shift direction

        # pad and crop to shift skeleton 1 pixel down, right, down-right, or down-left
        pad_top, pad_bottom = (shift_rows == 1), 0
        pad_left, pad_right = (shift_cols == 1), (shift_cols == -1)
        pad_vals = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
        shifted_skel = shift_2d(skeleton, pad_vals)

        # dest nodes: intersection of the shifted skeleton and the original skeleton
        dest_nodes = skeleton * shifted_skel
        if not np.any(dest_nodes):
            continue

        # src nodes: dest nodes shifted back to their original position
        pad_vals = np.flip(pad_vals, axis=1)
        src_nodes = shift_2d(dest_nodes, pad_vals)

        # add src and dest nodes to edge_connected_nodes
        edge_connected_nodes += src_nodes + dest_nodes

        # get node ids
        src_node_ids = node_labels[(node_labels > -1) & src_nodes]
        dest_node_ids = node_labels[(node_labels > -1) & dest_nodes]

        # create list of weighted edges: [(node_id_1, node_id_2, weight), ...]
        weight = np.linalg.norm((shift_rows, shift_cols))
        weights = np.full(src_node_ids.shape, weight)
        new_weighted_edges = zip(src_node_ids, dest_node_ids, weights)
        weighted_edges.extend(new_weighted_edges)

    # add weighted edges to graph
    g.add_weighted_edges_from(weighted_edges)

    # add disconnected nodes to graph
    isolated_nodes = skeleton * np.logical_not(edge_connected_nodes)
    if np.any(isolated_nodes):
        isolated_node_ids = node_labels[(node_labels > -1) & isolated_nodes].tolist()
        g.add_nodes_from(isolated_node_ids)

    return g


def regionprops_image(mask: npt.NDArray[np.bool_], prop: str) -> npt.NDArray:
    """Measure a scalar property on the mask return property image with the same shape.

    Args:
        mask: Boolean mask.
        prop: Property to measure, e.g. 'area'.
    Returns:
        npt.NDArray: Property values corresponding to mask regions, background=0.

    """
    labels = measure_label(mask)
    props = np.insert(regionprops_table(labels, properties=[prop])[prop], 0, 0)
    return props[labels]


def filter_branch_seg_mask(
    mask: npt.NDArray,
    footprint: Optional[npt.NDArray] = disk(2),
    remove_isolated=True,
) -> npt.NDArray:
    """
    Remove components from the segmentation mask that do not contain branches.
    Args:
        mask (npt.NDArray): Segmentation mask to filter
        footprint (npt.NDArray): Structuring element for median filter.
                                 Defaults to disk(2). Set to None to skip median filter.
    Returns:
        npt.NDArray: Filtered segmentation mask
    """

    if footprint is not None:
        # median filter unless explicitly disabled
        mask = median_filter(mask, footprint=footprint)

    labeled_components = measure_label(mask, connectivity=2)
    region_props = regionprops(labeled_components)
    region_circularities = [
        4 * np.pi * r.area / (r.perimeter**2 + 1e-7) for r in region_props
    ]

    seg_skel = skeletonize(mask)
    G = nx_graph_from_binary_skeleton(seg_skel)

    fork_nodes = {n for n in G.nodes() if G.degree[n] > 2}
    components = [*nx.connected_components(G)]

    remove_nodes_1_per_cc = []
    remove_nodes_all = set()

    def get_node_cc_label(node):
        node_coords = G.graph["physical_pos"][node]
        return labeled_components[node_coords[0]][node_coords[1]]

    for cc in components:
        cc_node_sample = next(iter(cc))
        node_cc_label = get_node_cc_label(cc_node_sample)
        circularity = region_circularities[node_cc_label - 1]
        if (remove_isolated and not cc.intersection(fork_nodes)) or circularity > 0.8:
            remove_nodes_1_per_cc.append(cc_node_sample)
            remove_nodes_all.update(cc)

    removed_components = set()

    for node in remove_nodes_1_per_cc:
        node_cc_label = get_node_cc_label(node)
        mask[labeled_components == node_cc_label] = 0
        removed_components.add(node_cc_label)

    G.remove_nodes_from(remove_nodes_all)

    return mask


def enhance_vessels_3d(img_vess: npt.NDArray) -> npt.NDArray:
    """Post-process Frangi filter to enhance main vessels and reduce noise."""

    vessels_z = img_vess.argmax(axis=0)

    maxima = np.ones_like(img_vess, dtype=bool)
    maxima[1:] = img_vess[1:] > img_vess[:-1]
    maxima[:-1] &= img_vess[:-1] >= img_vess[1:]
    img_vess[~maxima] = 0
    background = np.ones_like(img_vess, dtype=bool)
    # thresholds: [0.1, 0.072, 0.052, 0.037, 0.027, 0.019, 0.014, 0.01]
    thresholds = 0.1 * 0.72 ** np.arange(8)
    for z in range(len(img_vess)):
        zslice = img_vess[z].copy()
        for i in range(len(thresholds) - 1):
            t0 = thresholds[i + 1]  # lower threshold
            t1 = thresholds[i]  # upper threshold
            mask = apply_hysteresis_threshold(zslice, t0, t1)
            background[z, mask] = 0
            # Prepare for next iteration
            zslice[mask] = 1
            zslice[~mask] = np.minimum(zslice[~mask], t0 - 1e6)
    img_vess[background] = 0
    maxima[background] = 0

    ### Handle overlapping regions prior to creating 2D projection:
    # Find and label regions for each z slice in the vesselness maxima array;
    # calculate regions for slice as maxima[z-1] | maxima[z] | maxima[z+1],
    # calculate bbox diagonal length for regions on this mask,
    # and, in case of overlapping regions at other depths,
    # use the bbox diagonal length as a priority metric for what to project.
    # Finally, add a buffer around objects on the projection that exist at
    # contiguous xy coordinates but >1 difference in z position.
    # This procedure cleanly separates different-depth objects for the 2D projection
    # and prioritizes regions with a larger bounding box diagonal in cases of overlap.

    # bbox_diag: Max bbox diagonal of potential vessel (Frangi >.01) at each xy position
    bbox_diag = np.zeros_like(maxima[0], dtype=np.float32)

    # best_vess_zpos: z position of voxel with max bbox diagonal at each xy position
    best_vess_zpos = np.full_like(maxima[0], fill_value=-1, dtype=np.int32)

    for z in range(len(maxima)):
        zslice_regions = maxima[z]
        # Union mask with the slice above and the slice below the current slice
        if z != 0:
            zslice_regions |= maxima[z - 1]
        if z != len(maxima) - 1:
            zslice_regions |= maxima[z + 1]
        slice_region_labels = measure_label(zslice_regions)
        bbox = regionprops_table(
            label_image=slice_region_labels,
            properties=["bbox"],
        )
        x0, y0 = bbox["bbox-0"], bbox["bbox-1"]
        x1, y1 = bbox["bbox-2"], bbox["bbox-3"]
        # Calculate (squared) length of regions bbox diagonals
        diag_lengths = (x0 - x1) ** 2 + (y0 - y1) ** 2
        # The labels start at 1 (0 is background). Prepend 0 to index by label
        diag_lengths = np.insert(diag_lengths, 0, 0)
        slice_region_labels[~maxima[z]] = 0
        cur_bbox_diag = diag_lengths[slice_region_labels]
        is_max = cur_bbox_diag > bbox_diag
        best_vess_zpos[is_max] = z
        bbox_diag[is_max] = cur_bbox_diag[is_max]

    # Calculate buffer region of separation between objects at different depths
    combined_buffer_mask = np.zeros_like(vessels_z, dtype=bool)
    for row_shift in (-1, 0, 1):
        for col_shift in (-1, 0, 1):
            if row_shift == col_shift == 0:
                continue
            # Compare source array elements to shifted (in 1 of 8 directions) array
            src_rows_slice = [None, None]
            src_cols_slice = [None, None]
            dst_rows_slice = [None, None]
            dst_cols_slice = [None, None]
            if row_shift == -1:  # Up
                src_rows_slice[0] = 1
                dst_rows_slice[1] = -1
            elif row_shift == 1:  # Down
                src_rows_slice[1] = -1
                dst_rows_slice[0] = 1
            if col_shift == -1:  # Left
                src_cols_slice[0] = 1
                dst_cols_slice[1] = -1
            elif col_shift == 1:  # Right
                src_cols_slice[1] = -1
                dst_cols_slice[0] = 1

            src_rows_slice = slice(src_rows_slice[0], src_rows_slice[1])
            dst_rows_slice = slice(dst_rows_slice[0], dst_rows_slice[1])
            src_cols_slice = slice(src_cols_slice[0], src_cols_slice[1])
            dst_cols_slice = slice(dst_cols_slice[0], dst_cols_slice[1])

            src_slice = src_rows_slice, src_cols_slice
            dst_slice = dst_rows_slice, dst_cols_slice
            z_diff = abs(vessels_z[src_slice] - vessels_z[dst_slice])
            buffer_mask = (z_diff > 1) & (bbox_diag[src_slice] < bbox_diag[dst_slice])
            combined_buffer_mask[src_slice][buffer_mask] = 1
            vessels_z[src_slice][buffer_mask] = -1
            bbox_diag[src_slice][buffer_mask] = 0

    vessels_z[combined_buffer_mask] = -1

    for z in range(len(img_vess)):
        zslice = np.where(vessels_z == z, img_vess[z], 0)
        mask = np.zeros_like(zslice, dtype=bool)
        for i in range(len(thresholds) - 1):
            t0 = thresholds[i + 1]  # lower threshold
            t1 = thresholds[i]  # upper threshold
            mask |= apply_hysteresis_threshold(zslice, t0, t1)
            # Prepare for next iteration
            zslice[mask] = 1
            zslice[~mask] = np.minimum(zslice[~mask], t0 - 1e6)
        img_vess[z] = np.where(mask, img_vess[z], 0)

    mask_filt = filter_branch_seg_mask(
        img_vess.max(0)>0,
        footprint=None,
        remove_isolated=False)

    return np.where(mask_filt, img_vess, 0)
