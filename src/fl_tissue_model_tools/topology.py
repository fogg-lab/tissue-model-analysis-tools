import numpy as np
import numpy.typing as npt
import gudhi as gd
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from pydmtgraph.dmtgraph import DMTGraph

def __random_color(i : int):
    """ Convert an int to a random color """
    from cv2 import cvtColor, COLOR_HSV2BGR

    phi = 0.618033988749895
    step = 180*phi

    return cvtColor(np.array([[[step*i, 220, 255]]], np.uint8), COLOR_HSV2BGR)[0][0] / 255


def plot_colored_barcode(barcode_and_colors, ax=None, **kwargs):
    """ Plot a colored barcode computed by `compute_colored_tree_and_barcode`

        Args:
            barcode_and_colors (list): list of bars and colors in the format returned
                by `compute_colored_tree_and_barcode`. Each item in the list is
                a tuple of a persistence pair and a color.
            ax (matplotlib.axes.Axes): axis on which to plot barcode. defaults to None.
                If no axis is provided, the tree is just plotted on the current axis of plt.
            kwargs (dict): Additional keyword arguments.
                These are forwarded to the ax.barh call.
    """
    # if no axis is provided, fetch the current axis
    import matplotlib.pyplot as plt
    ax_provided = ax is not None
    ax = ax if ax_provided else plt.gca()
    # sort bars in ascending order by birth time
    barcode_and_colors.sort(reverse=True)
    # prepare args for bar plot
    heights = [i for i in range(len(barcode_and_colors))]
    births = [bar[0] for bar, color in barcode_and_colors]
    widths = [bar[1] - bar[0] for bar, color in barcode_and_colors]
    colors = [color for bar, color in barcode_and_colors]
    # plot the barcode
    ax.barh(heights, widths, left=births, color=colors, **kwargs)
    ax.set_yticks([])
    ax.set_xlabel("Barcode")
    if not ax_provided:
        plt.show()

def plot_colored_tree(edges_and_colors, ax=None, **kwargs):
    """ Plot a colored tree computed by `compute_colored_tree_and_barcode`

        Args:
            edges_and_colors (list): List of edges and colors.
                Each item in the list is a tuple of a line and an rgb color.
                Each line in a tuple with the 2d endpoints of an edge in the tree
            ax (matplotlib.axes.Axes): axis on which to plot barcode. defaults to None.
                If no axis is provided, the tree is just plotted on the current axis of plt.
            kwargs (dict): Additional keyword arguments.
                These are forwarded to the LineCollection constructor.
                For example, kwargs could contain linewidth.
    """
    # if no axis is provided, fetch the current axis
    ax_provided = ax is not None
    ax = ax if ax_provided else plt.gca()
    # prepare the edges to be plotted
    edges = [line for line, color in edges_and_colors]
    colors = [color for line, color in edges_and_colors]
    edges_collection = LineCollection(edges, colors=colors, **kwargs)
    # plot the tree
    ax.add_collection(edges_collection)
    ax.set_axis_off()
    if not ax_provided:
        plt.show()


def nx_graph_from_binary_skeleton(skeleton: npt.NDArray) -> nx.Graph:
    skeleton = skeleton.astype(bool)
    g = nx.Graph()

    if not np.any(skeleton):
        g.graph['physical_pos'] = np.empty((0, 2), dtype=int)
        return g

    node_labels = np.full(skeleton.shape, -1)
    weighted_edges = []
    nodes = np.zeros_like(skeleton, dtype=bool)
    node_pos = np.empty((0, 2), dtype=int)

    edges_added = set()

    def shift_2d(arr: npt.NDArray, pad_vals: npt.NDArray) -> npt.NDArray:
        # add zero-padding to sides then crop the opposite sides from the padding
        padded = np.pad(arr, pad_vals)
        pad_bottom, pad_right = pad_vals[0,1], pad_vals[1,1]
        h, w = arr.shape
        return padded[pad_bottom:(h + pad_bottom), pad_right:(w + pad_right)]

    for (shift_rows, shift_cols) in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        ## find skeleton nodes connected by an edge for the current shift direction

        # shift the skeleton 1 pixel down, right, down-right, or down-left
        # use pad and crop method
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

        # find new nodes not already added (they will need to be assigned an id)
        new_nodes = (src_nodes + dest_nodes) * np.logical_not(nodes)
        nodes += new_nodes
        new_nodes_pos = np.argwhere(new_nodes)

        # record previous node count and add new nodes coordinates to node_pos
        prev_node_count = node_pos.shape[0]
        node_pos = np.vstack((node_pos, new_nodes_pos))

        # assign numerical ids to new nodes
        new_node_ids = np.arange(prev_node_count, node_pos.shape[0])
        node_labels[new_nodes_pos[:, 0], new_nodes_pos[:, 1]] = new_node_ids

        # get node ids for all current src and dest nodes
        src_node_ids = node_labels[(node_labels > -1) & src_nodes]
        dest_node_ids = node_labels[(node_labels > -1) & dest_nodes]

        # create list of weighted edges: [(node_id_1, node_id_2, weight), ...]
        weight = np.linalg.norm((shift_rows, shift_cols))
        weights = np.full(src_node_ids.shape, weight)
        new_weighted_edges = zip(src_node_ids, dest_node_ids, weights)
        weighted_edges.extend(new_weighted_edges)

        new_edges_added = [tuple(sorted((src, dest))) for src, dest in zip(src_node_ids, dest_node_ids)]    

    # add weighted edges to graph
    g.add_weighted_edges_from(weighted_edges)

    # add edges of weight 0 to graph (isolated nodes)
    isolated_nodes = skeleton * np.logical_not(nodes)
    if np.any(isolated_nodes):
        isolated_nodes_pos = np.argwhere(isolated_nodes)
        prev_node_count = node_pos.shape[0]
        node_pos = np.vstack((node_pos, isolated_nodes_pos))
        isolated_node_ids = np.arange(prev_node_count, node_pos.shape[0])
        node_labels[isolated_nodes_pos[:, 0], isolated_nodes_pos[:, 1]] = isolated_node_ids
        isolated_edges = zip(isolated_node_ids, isolated_node_ids)
        g.add_edges_from(isolated_edges, weight=0)

    # add physical node positions to graph
    g.graph['physical_pos'] = node_pos

    return g


def compute_colored_tree_and_barcode(vertices, edges):
    """ Compute a tree and barcode colored according to branches.

    Args:
        vertices (V x 2 numpy array of ints):
            Array where ith row stores 2d coordinate of ith vertex of a graph
        edges (E x 2 numpy array of ints):
            array where kth row [i, j] storing the indices i and j of
            the kth edge's endpoints in `vertices`

    Returns:
        edges_and_colors (list): List of edges and colors.
            Each item in the list is a tuple of a line and an rgb color.
            Each line in a tuple with the 2d endpoints of an edge in the tree
        barcode_and_colors (list): list of bars and colors.
            Each item in the list is a tuple of a persistence pair and an rgb color.

    Raises:
        ValueError: The input graph must be a forest
    """
    G = __convert_to_networkx_graph(vertices, edges)

    if not nx.is_forest(G):
        raise ValueError("Input graph must be a forest")

    # function for computing the length of the edge {u,v}
    edge_length = lambda u, v : np.linalg.norm(vertices[u]-vertices[v])
    # list of edges and barcodes to be returned
    edges_and_colors = []
    barcode_and_colors = []
    # compute the path between each vertex and the root of its connected component
    dist_to_center = { }
    parent = { v : None for v in G.nodes }
    graph_components = [ G.subgraph(c).copy() for c in nx.connected_components(G) ]
    for g in graph_components:
        # we treat the center of the graph as the root
        center = np.random.choice([n for n in g])
        # create a dict where each vertex points to its parent in the tree.
        # we set parents with a bfs starting at the center.
        # we also use the bfs to compute distance to center
        parent[center] = center
        dist_to_center[center] = 0
        unvisited_vertices = [ center ]
        while len(unvisited_vertices) > 0:
            v = unvisited_vertices.pop(0)
            for n in G.neighbors(v):
                # if the parent of a node has not been assigned,
                # it has not been visited
                if parent[n] is None:
                    parent[n] = v
                    dist_to_center[n] = dist_to_center[v] + edge_length(n,v)
                    unvisited_vertices.append(n)
    # check that the parents were set properly
    assert all([parent[v] is not None for v in G.nodes])
    # Each vertex in the tree belongs to a unique branch
    # corresponding to a leaf. Specifically, a vertex is
    # in the longest branch from a leaf to the center
    # of all its descendant leaves.
    #
    # Label each vertex with its branch.
    # We do this by labelling all vertices on the path
    # between the leaf and the center,
    # unless we encounter vertex has already been labelled
    # with a more distant leaf, which means all other vertices on
    # the path are apart of this branch.
    leaves = [ v for v in G.nodes if G.degree[v] == 1 ]
    max_dist_to_leaf = { v : -np.inf for v in G.nodes }
    branch_label = { }
    for leaf in leaves:
        current_vertex = leaf
        current_parent = parent[current_vertex]
        max_dist_to_leaf[leaf] = current_distance = 0
        branch_label[leaf] = leaf
        # This while loop follows the unique path from
        # a leaf to the root.
        while current_parent != current_vertex:
            current_distance += edge_length(current_parent, current_vertex)
            if current_distance < max_dist_to_leaf[current_parent]:
                # We've reached a vertex that has a descendant leaf that is
                # further away, so it is part of another branch.
                # Thus, we quit our traversal.
                break
            current_vertex = current_parent
            current_parent = parent[current_vertex]
            max_dist_to_leaf[current_vertex] = current_distance
            branch_label[current_vertex] = leaf
    # now that we have labelled each vertex with its branch,
    # we fill our list of edges and colors
    # where each branch is a different color.
    for i, leaf in enumerate(leaves):
        current_vertex = leaf
        current_label = leaf
        current_color = __random_color(i)
        current_parent = parent[leaf]
        current_distance = 0
        # Follow the path from the leaf
        # until we encounter another branch or reach the root.
        # Add each edge along the way with the color of the branch.
        while current_label == leaf and current_vertex != current_parent:
            # update distance from `leaf`.
            # this is used after the loop finishes to compute the barcode
            current_distance += edge_length(current_parent, current_vertex)
            # v1 and v2 are the coordinates of the vertex
            v1 = vertices[current_vertex]
            v2 = vertices[current_parent]
            # reverse v1 and v2 as mpl uses image coordinates
            c1 = (v1[1], v1[0])
            c2 = (v2[1], v2[0])
            edges_and_colors.append(([c1, c2], current_color))
            # update pointers for next iteration of loop
            current_vertex = current_parent
            current_parent = parent[current_vertex]
            current_label = branch_label[current_vertex]
        # add the branch of the current leaf to the barcode
        # its birth is the (negative) distance of the leaf to the center
        # the death is the distance where we encounter a longer branch
        birth = -dist_to_center[leaf]
        death = birth + current_distance
        barcode_and_colors.append(((birth, death), current_color))

    return edges_and_colors, barcode_and_colors


def __convert_to_networkx_graph(vertices, edges):
    """ Convert a dmtgraph to a Networkx graph """
    G = nx.Graph()
    for vertex0, vertex1 in edges:
        # Add each edge to the graph with weight = Euclidean distance between endpoints
        # Each row in `edges` is an array [i, j],
        #   where i and j are ints representing the index of the endpoints.
        # Each row in `verts` is an array [x, y],
        #   where x and y are the 2d-coordinates of the vertex.
        edge_length = np.linalg.norm(vertices[vertex0]-vertices[vertex1])
        G.add_edge(vertex0, vertex1, weight=edge_length)
    return G


def __shortest_path_tree(
    G: nx.Graph, distances: dict
):
    """ Return the shortest path tree of a networkx graph with respect to some root.

        Args:
            G (networkx.Graph):
            distances (dict): dictionary of distances to the root
    """

    C = nx.Graph()
    for vertex0, vertex1 in G.edges():
        C.add_edge(vertex0, vertex1, weight=max(distances[vertex0], distances[vertex1]))
    return nx.minimum_spanning_tree(C)


def compute_morse_skeleton_and_barcode(im: npt.NDArray[np.double],
                                       threshold1: float=0.5,
                                       threshold2: float=0.0):
    """Fit a Morse skeleton to the image `im`.
        Also, compute the lower star filtration of the Morse skeleton where each
        vertex's filtration value is the negative distance from the graph center.
        Args:
            im (npt.NDArray[np.float64]): Grayscale image
            threshold1 (float): threshold for the Morse skeleton simplification step. See paper.
            threshold2 (float): threshold for the edges. We only take the 1-unstable manifold of edges
                                with value > threshold2. The higher the value, the more disconnected
                                the graph will be
        Returns:
            verts_total (V x 2 numpy array of ints): Array where ith row stores
                                                       2d coordinate of ith vertex in Morse skeleton
            edges_total (E x 2 numpy array of ints): array where kth row [i, j]
                                                       storing the indices i and j of the kth edge's
                                                       endpoints in `verts_total`
            bc_total (n x 2 numpy array of floats): array where each row is a bar in the barcode
                                                   of the filtration on the Morse skeleton
    """

    # These are initialized to being empty
    # however, we set the number of columns so we can use np.concatenate later.
    verts_total = np.zeros((0,2))
    edges_total = np.zeros((0,2), dtype=int)
    bc_total = np.zeros((0,2))

    # Compute the Morse skeleton
    # Slice a bounding box of the connected component to speed up computation
    dmtG = DMTGraph(im)
    verts, edges = dmtG.computeGraph(threshold1, threshold2)

    # If the graph is not empty, we compute and plot the barcode of the Morse skeleton.
    # The filtration adds the vertices and edges in decreasing order of distance from the center.
    if(len(edges) > 0):
        # create a networkx graph of Morse skeleton
        # we use {G} to compute the distance of each vertex from the "center" of the graph
        # The center is a graph-theoretic notion defined here:
        #   https://en.wikipedia.org/wiki/Graph_center
        G = nx.Graph()
        for v0, v1 in edges:
            # add each graph to the graph with weight = Euclidean distance between endpoints
            # each edge in edges is an array [i, j, _],
            #   where i and j are ints representing the index of the endpoints.
            # each vertex in verts is an array [x, y],
            #   where x and y are the 2d-coordinates of the vertex.
            edge_length = np.linalg.norm(verts[v0]-verts[v1])
            G.add_edge(v0, v1, weight=edge_length)

        graph_components = [ G.subgraph(c).copy() for c in nx.connected_components(G) ]
        for g in graph_components:
            # Choose an arbitrary vertex as the center and compute the distance of each vertex
            center = np.random.choice([n for n in g])
            distances = nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length(
                g, center)

            # compute the shortest path tree of the graph component
            spt = __shortest_path_tree(g, distances)

            # Now, we use the distances to compute the barcode of the Morse skeleton
            #
            # Build a Gudhi complex to compute the filtration
            # `K` is just another representation of the Morse skeleton
            K = gd.SimplexTree()

            # Add the vertices to the complex with their filtration values
            # `distances` is a dict with entries of the form {vertex_idx : distance to `center`}
            for key in distances:
                K.insert([key], filtration=-distances[key])

            # Add the edges to the complex.
            # The filtation value of an edge is the max filtation value of its vertices.
            for v0, v1 in spt.edges():
                K.insert([v0, v1], filtration = max(K.filtration([v0]), K.filtration([v1])))

            # compute the barcode of the Morse skeleton
            K.compute_persistence()

            # retrieve the 0-dimensional barcode
            bc = K.persistence_intervals_in_dimension(0)

            bc_total = np.concatenate((bc_total, bc), axis=0)
            verts_total = np.concatenate((verts_total, verts), axis=0)
            edges_total = np.concatenate((edges_total, np.array(spt.edges)), axis=0)

    return verts_total, edges_total, bc_total
