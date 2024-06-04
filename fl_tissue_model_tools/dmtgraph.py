"""A direct Python implementation of `computeDMTGraph` from the `pydmtgraph` extension.

The original source code by Mitchell Black can be found at this URL:
https://github.com/blackmit/pydmtgraph/tree/main

"""

from numba import njit
import numpy as np
from numpy.typing import NDArray


# Names for data members of Vertex stored as an ndarray
X = 0
Y = 1
VALUE = 2
PARENT = 3
MORSE_PARENT = 4
NEIGHBOR_1 = 5
NEIGHBOR_2 = 6
NEIGHBOR_3 = 7
NEIGHBOR_4 = 8

# Names for data members of Edge stored as an ndarray
INDEX = 0
V1_IDX = 1
V2_IDX = 2
DV1_IDX = 3
DV2_IDX = 4
PAIR_TYPE = 5
PERSISTENCE = 6

# Pair types (names that map to values stored in an ndarray)
UNKNOWN_PAIR_TYPE = 0
VERTEX_EDGE_PAIR = 1
EDGE_TRIANGLE_PAIR = 2


def compute_dmt_graph(img: NDArray[np.float32], delta1: float, delta2: float = 0.0):
    """
    Computes the Discrete Morse Theory (DMT) graph for a given grayscale image.

    This function creates simplices, computes persistence, and constructs the DMT graph.

    The original C++ code by Mitchell Black can be found at this URL:
    https://github.com/blackmit/pydmtgraph/tree/main

    Args:
        img (NDArray[np.float32]): A 2D array representing a grayscale image.
        delta1 (float): The persistence threshold for edges in the graph.
        delta2 (float, optional): The persistence threshold for vertices in the graph. Default is 0.0.

    Returns:
        tuple: A tuple containing:
            - vertices (NDArray[np.int32]): An array of vertex positions.
            - edges (NDArray[np.int32]): An array of edge indices.
    """
    deltas = np.array([delta1, delta2], dtype=np.float32)

    ### Create simplices
    nrows, ncols = img.shape
    n_verts = nrows * ncols
    n_dual_verts = ((nrows - 1) * (ncols - 1) * 2) + 1
    n_edges = (nrows - 1) * ncols + nrows * (ncols - 1) + (nrows - 1) * (ncols - 1)
    V = np.full((n_verts, 9), -1, dtype=np.float32)
    DV = np.full((n_dual_verts, 9), -1, dtype=np.float32)
    E = np.full((n_edges, 7), -1, dtype=np.float32)
    img = -img
    create_vertices(img, V)
    n_dual_verts = create_dual_vertices(img, DV)
    create_edges(img, E, n_dual_verts)

    ### Compute persistence
    edge_v1_indices = E[:, V1_IDX].astype(np.int32)
    edge_v2_indices = E[:, V2_IDX].astype(np.int32)
    edge_v1_values = V[edge_v1_indices, VALUE]
    edge_v2_values = V[edge_v2_indices, VALUE]
    edge_max_val = np.maximum(edge_v1_values, edge_v2_values)
    edges_index = np.arange(len(E))
    # Sort edges by edge_max_val, then index to break ties
    sort_perm = np.lexsort((edges_index, edge_max_val))
    edges_index = edges_index[sort_perm]
    edge_max_val = edge_max_val[sort_perm]
    E = E[sort_perm]
    compute_persistence_1(E, V, edge_max_val)
    # Sort edges by -edge_val, then -index to break ties
    sort_perm = np.lexsort((-edges_index, -edge_max_val))
    edge_max_val = edge_max_val[sort_perm]
    E = E[sort_perm]
    compute_persistence_2(E, DV, edge_max_val)

    ### Construct the graph
    vertices, edges = collect(deltas, E, V)

    return vertices, edges


@njit(cache=True)
def find(v_index: np.int32, V: NDArray[np.float32]) -> np.int32:
    while V[v_index][PARENT] != v_index:
        v_index = np.int32(V[v_index][PARENT])

    root = v_index

    v_index = np.int32(V[v_index][PARENT])
    while V[v_index][PARENT] != v_index:
        parent = np.int32(V[v_index][PARENT])
        V[v_index][PARENT] = np.float32(root)
        v_index = parent

    return root


@njit(cache=True)
def merge(
    v1_index: np.int32,
    v2_index: np.int32,
    V: NDArray[np.float32],
    flip_comparison: bool = False,
) -> np.float32:
    p1_index: np.int32 = find(v1_index, V)
    p2_index: np.int32 = find(v2_index, V)

    if p1_index == p2_index:
        return np.float32(np.nan)
    elif flip_comparison and (
        V[p1_index][VALUE] > V[p2_index][VALUE]
        or (V[p1_index][VALUE] == V[p2_index][VALUE] and p1_index > p2_index)
    ):
        V[p2_index][PARENT] = np.float32(p1_index)
        return V[p2_index][VALUE]
    elif not flip_comparison and (
        V[p1_index][VALUE] < V[p2_index][VALUE]
        or (V[p1_index][VALUE] == V[p2_index][VALUE] and p1_index < p2_index)
    ):
        V[p2_index][PARENT] = np.float32(p1_index)
        return V[p2_index][VALUE]
    else:
        V[p1_index][PARENT] = np.float32(p2_index)
        return V[p1_index][VALUE]


def create_vertices(img: NDArray[np.float32], V: NDArray[np.float32]):
    n_rows, n_cols = img.shape
    n_verts = n_rows * n_cols

    # Create arrays for coordinates
    x_coords, y_coords = np.meshgrid(
        np.arange(n_rows), np.arange(n_cols), indexing="ij"
    )

    # Flatten arrays to get one-dimensional coordinates
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    img_flat = img.flatten()

    # Fill in the V array
    V[:n_verts, X] = x_coords_flat
    V[:n_verts, Y] = y_coords_flat
    V[:n_verts, VALUE] = img_flat
    V[:n_verts, PARENT] = np.arange(n_verts)


def create_dual_vertices(img: NDArray[np.float32], DV: NDArray[np.float32]) -> int:
    n_rows, n_cols = img.shape
    n_dual_verts = (n_rows - 1) * (n_cols - 1) * 2

    # Create arrays for coordinates
    r_coords, c_coords = np.meshgrid(
        np.arange(n_rows - 1), np.arange(n_cols - 1), indexing="ij"
    )

    # Flatten arrays to get one-dimensional coordinates
    r_coords_flat = r_coords.flatten()
    c_coords_flat = c_coords.flatten()

    # Generate dual vertices coordinates and values
    x_coords = np.repeat(r_coords_flat, 2)
    y_coords = np.repeat(c_coords_flat * 2, 2) + np.tile([0, 1], len(r_coords_flat))

    max_vals_0 = np.maximum.reduce([img[:-1, :-1], img[:-1, 1:], img[1:, :-1]])
    max_vals_1 = np.maximum.reduce([img[:-1, 1:], img[1:, :-1], img[1:, 1:]])

    values = np.empty(n_dual_verts, dtype=np.float32)
    values[0::2] = max_vals_0.flatten()
    values[1::2] = max_vals_1.flatten()

    # Fill in the DV array
    DV[:n_dual_verts, X] = x_coords
    DV[:n_dual_verts, Y] = y_coords
    DV[:n_dual_verts, VALUE] = values
    DV[:n_dual_verts, PARENT] = np.arange(n_dual_verts)

    DV[n_dual_verts][VALUE] = np.inf
    return n_dual_verts


def create_edges(img: NDArray[np.float32], E: NDArray[np.float32], n_dual_verts: int):
    n_rows, n_cols = img.shape
    n_edges = 0

    # Vertical edges
    v_r_coords, v_c_coords = np.meshgrid(
        np.arange(n_rows - 1), np.arange(n_cols), indexing="ij"
    )
    v_r_coords_flat = v_r_coords.flatten()
    v_c_coords_flat = v_c_coords.flatten()

    v_indices = np.arange(len(v_r_coords_flat))
    v1_indices = v_r_coords_flat * n_cols + v_c_coords_flat
    v2_indices = v1_indices + n_cols

    dual_indices = v_r_coords_flat * 2 * (n_cols - 1) + v_c_coords_flat * 2

    E[v_indices, INDEX] = v_indices
    E[v_indices, V1_IDX] = v1_indices
    E[v_indices, V2_IDX] = v2_indices
    E[v_indices, DV1_IDX] = np.where(
        v_c_coords_flat == 0, n_dual_verts, dual_indices - 1
    )
    E[v_indices, DV2_IDX] = np.where(
        v_c_coords_flat == n_cols - 1, n_dual_verts, dual_indices
    )
    E[v_indices, PAIR_TYPE] = 0
    E[v_indices, PERSISTENCE] = np.inf
    n_edges += len(v_indices)

    # Horizontal edges
    h_r_coords, h_c_coords = np.meshgrid(
        np.arange(n_rows), np.arange(n_cols - 1), indexing="ij"
    )
    h_r_coords_flat = h_r_coords.flatten()
    h_c_coords_flat = h_c_coords.flatten()

    h_indices = np.arange(len(h_r_coords_flat)) + n_edges
    h1_indices = h_r_coords_flat * n_cols + h_c_coords_flat
    h2_indices = h1_indices + 1

    dual_indices = h_r_coords_flat * 2 * (n_cols - 1) + h_c_coords_flat * 2

    E[h_indices, INDEX] = h_indices
    E[h_indices, V1_IDX] = h1_indices
    E[h_indices, V2_IDX] = h2_indices
    E[h_indices, DV1_IDX] = np.where(
        h_r_coords_flat == 0, n_dual_verts, dual_indices - 2 * (n_cols - 1) + 1
    )
    E[h_indices, DV2_IDX] = np.where(
        h_r_coords_flat == n_rows - 1, n_dual_verts, dual_indices
    )
    E[h_indices, PAIR_TYPE] = 0
    E[h_indices, PERSISTENCE] = np.inf
    n_edges += len(h_indices)

    # Diagonal edges
    d_r_coords, d_c_coords = np.meshgrid(
        np.arange(n_rows - 1), np.arange(n_cols - 1), indexing="ij"
    )
    d_r_coords_flat = d_r_coords.flatten()
    d_c_coords_flat = d_c_coords.flatten()

    d_indices = np.arange(len(d_r_coords_flat)) + n_edges
    d1_indices = d_r_coords_flat * n_cols + d_c_coords_flat + 1
    d2_indices = d1_indices + n_cols - 1

    dual_indices = d_r_coords_flat * 2 * (n_cols - 1) + d_c_coords_flat * 2

    E[d_indices, INDEX] = d_indices
    E[d_indices, V1_IDX] = d1_indices
    E[d_indices, V2_IDX] = d2_indices
    E[d_indices, DV1_IDX] = dual_indices
    E[d_indices, DV2_IDX] = dual_indices + 1
    E[d_indices, PAIR_TYPE] = 0
    E[d_indices, PERSISTENCE] = np.inf


@njit(cache=True)
def compute_persistence_1(
    E: NDArray[np.float32],
    V: NDArray[np.float32],
    edge_values: NDArray[np.float32],
):
    for i, e in enumerate(E):
        death = edge_values[i]
        birth = merge(
            np.int32(e[V1_IDX]),
            np.int32(e[V2_IDX]),
            V,
        )
        if not np.isnan(birth):
            e[PERSISTENCE] = death - birth
            e[PAIR_TYPE] = np.float32(VERTEX_EDGE_PAIR)


@njit(cache=True)
def compute_persistence_2(
    E: NDArray[np.float32],
    DV: NDArray[np.float32],
    edge_values: NDArray[np.float32],
):
    for i, e in enumerate(E):
        birth = edge_values[i]
        if e[PAIR_TYPE] == np.float32(UNKNOWN_PAIR_TYPE):
            death = merge(
                np.int32(e[DV1_IDX]),
                np.int32(e[DV2_IDX]),
                DV,
                True,
            )
        else:
            death = np.float32(np.nan)
        if not np.isnan(death):
            e[PERSISTENCE] = death - birth
            e[PAIR_TYPE] = np.float32(EDGE_TRIANGLE_PAIR)


@njit(cache=True)
def collect(
    deltas: NDArray[np.float32],
    E: NDArray[np.float32],
    V: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    ### Collect tree
    for e in E:
        if e[PAIR_TYPE] == np.float32(VERTEX_EDGE_PAIR) and e[PERSISTENCE] < deltas[0]:
            # Link up the vertices in `e` (commit edge)
            for v1_index, v2_index in ((e[V1_IDX], e[V2_IDX]), (e[V2_IDX], e[V1_IDX])):
                v1 = V[int(v1_index)]
                for n in range(NEIGHBOR_1, NEIGHBOR_4 + 1):
                    if v1[n] == -1:
                        v1[n] = v2_index
                        break

    ### Cancel Morse pairs
    explored = np.full(len(V), -1, dtype=np.int32)
    queue = np.zeros(len(V), dtype=np.int32)
    for i in range(len(V)):
        v = V[i]
        if v[MORSE_PARENT] == np.float32(-1):
            queue_start = 0
            queue_end = 1
            queue[0] = i
            min_v = i

            while queue_start < queue_end:
                curr_index = queue[queue_start]
                queue_start += 1
                curr = V[curr_index]
                explored[curr_index] = i
                min_val = V[min_v][VALUE]
                cur_val = curr[VALUE]
                if cur_val < min_val or ((cur_val == min_val) and (curr_index < min_v)):
                    min_v = curr_index
                for n in range(NEIGHBOR_1, NEIGHBOR_4 + 1):
                    neighbor = np.int32(curr[n])
                    if neighbor == -1:
                        break
                    if explored[neighbor] != i:
                        queue[queue_end] = neighbor
                        queue_end += 1

            V[min_v][MORSE_PARENT] = np.float32(min_v)
            queue_start = 0
            queue_end = 1
            queue[0] = np.int32(min_v)
            while queue_start < queue_end:
                curr_index = queue[queue_start]
                queue_start += 1
                curr = V[curr_index]
                for n in range(NEIGHBOR_1, NEIGHBOR_4 + 1):
                    neighbor = np.int32(curr[n])
                    if neighbor == -1:
                        break
                    if V[neighbor][MORSE_PARENT] == np.float32(-1):
                        V[neighbor][MORSE_PARENT] = np.float32(curr_index)
                        queue[queue_end] = neighbor
                        queue_end += 1

    ### Collect unstable manifold (M_un)
    M_un_verts = np.full(len(V), False, dtype=np.bool_)
    M_un_edges = np.full((len(E), 2), -1, dtype=np.int32)
    M_un_edges_idx = 0

    def collect_path_to_min(
        v_index: np.int32,
        V: NDArray[np.float32],
        M_un_verts: NDArray[np.bool_],
        M_un_edges: NDArray[np.int32],
        M_un_edges_idx: np.int32,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], np.int32]:
        curr_index = v_index
        curr = V[curr_index]
        while (
            not M_un_verts[curr_index]
            and np.int32(curr[MORSE_PARENT]) != curr_index
            and curr[MORSE_PARENT] != np.float32(-1)
        ):
            M_un_verts[curr_index] = True
            if M_un_edges_idx == len(M_un_edges):
                # Increase array capacity
                M_un_edges = np.concatenate((M_un_edges, np.full_like(M_un_edges, -1)))
            M_un_edges[M_un_edges_idx][0] = curr_index
            M_un_edges[M_un_edges_idx][1] = np.int32(curr[MORSE_PARENT])
            M_un_edges_idx += 1
            curr_index = np.int32(curr[MORSE_PARENT])
            curr = V[curr_index]

        return M_un_edges, M_un_verts, M_un_edges_idx

    for e in E:
        edge_val = max(V[int(e[V1_IDX])][VALUE], V[int(e[V2_IDX])][VALUE])
        if e[PERSISTENCE] > deltas[0] and edge_val < -deltas[1]:
            v1_idx = np.int32(e[V1_IDX])
            v2_idx = np.int32(e[V2_IDX])
            M_un_edges, M_un_verts, M_un_edges_idx = collect_path_to_min(
                v1_idx, V, M_un_verts, M_un_edges, M_un_edges_idx
            )
            M_un_edges, M_un_verts, M_un_edges_idx = collect_path_to_min(
                v2_idx, V, M_un_verts, M_un_edges, M_un_edges_idx
            )

            if M_un_edges_idx == len(M_un_edges):
                # Increase array capacity
                M_un_edges = np.concatenate((M_un_edges, np.full_like(M_un_edges, -1)))
            M_un_edges[M_un_edges_idx][0] = v1_idx
            M_un_edges[M_un_edges_idx][1] = v2_idx
            M_un_edges_idx += 1

    ### Return unstable manifold
    max_num_vertex_positions = M_un_edges_idx * 2
    vertex_positions = np.full((max_num_vertex_positions, 2), -1, dtype=np.int32)
    vertex_positions_idx = 0
    edge_indices = np.full((M_un_edges_idx, 2), -1, dtype=np.int32)
    index_map = np.full(len(V), -1, dtype=np.int32)

    for i in range(M_un_edges_idx):
        e = M_un_edges[i]
        if index_map[e[0]] == np.int32(-1):
            index_map[e[0]] = vertex_positions_idx
            vertex_positions[vertex_positions_idx][0] = np.int32(V[e[0]][X])
            vertex_positions[vertex_positions_idx][1] = np.int32(V[e[0]][Y])
            vertex_positions_idx += 1
        if index_map[e[1]] == np.int32(-1):
            index_map[e[1]] = vertex_positions_idx
            vertex_positions[vertex_positions_idx][0] = np.int32(V[e[1]][X])
            vertex_positions[vertex_positions_idx][1] = np.int32(V[e[1]][Y])
            vertex_positions_idx += 1
        edge_indices[i][0] = index_map[e[0]]
        edge_indices[i][1] = index_map[e[1]]

    vertex_positions = vertex_positions[:vertex_positions_idx]

    return vertex_positions[:vertex_positions_idx], edge_indices
