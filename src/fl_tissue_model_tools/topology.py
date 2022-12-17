import cv2 as cv
import dask as d
import numpy as np
import numpy.typing as npt

import gudhi as gd
import networkx as nx

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
    return


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
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt
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
    return


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
        centers = nx.algorithms.distance_measures.center(g)
        center = centers[0]
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
    for v0, v1 in edges:
        # Add each edge to the graph with weight = Euclidean distance between endpoints
        # Each row in `edges` is an array [i, j],
        #   where i and j are ints representing the index of the endpoints.
        # Each row in `verts` is an array [x, y],
        #   where x and y are the 2d-coordinates of the vertex.
        edge_length = np.linalg.norm(vertices[v0]-vertices[v1])
        G.add_edge(v0, v1, weight=edge_length)
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
    for v0, v1 in G.edges():
        C.add_edge(v0, v1, weight=max(distances[v0], distances[v1]))
    return nx.minimum_spanning_tree(C)


def compute_morse_skeleton_and_barcode(im: npt.ArrayLike, mask: npt.ArrayLike,
                                       threshold1: float=0.5, threshold2: float=0.0,
                                       component_min_size: int=1000):
    """Fit a Morse skeleton to all sufficiently large components of the image `im`.

        Fits a Morse Skeleton to all components of `im` with area larger than `component_min_size`.
        The components are specified by the binary image `mask`,
         which is a mask of the foreground pixels.
        Each component is fit in parallel using dask.
        Also, compute the lower star filtration of the Morse skeleton where each
        vertex's filtration value is the negative distance from the graph center.

        Args:
            im (numpy array of uint8): Grayscale image
            mask (numpy array of bools): Foreground mask of `im`
            graph_recon_path (str): path to the graph_recon_DM directory.
                                    By default, the function assumes we are running the code
                                    in fogg_lab_tissue_model_analysis_tools/notebooks
            threshold1 (float): threshold for the Morse skeleton simplification step. See paper.
            threshold2 (float): threshold for the edges. We only take the 1-unstable manifold of edges
                                with value > threshold2. The higher the value, the more disconnected
                                the graph will be
            component_min_size (int): Minimum area of components to be fit with a Morse skeleton.


        Returns:
            verts_total (V x 2 numpy array of ints): Array where ith row stores
                                                       2d coordinate of ith vertex in Morse skeleton
            edges_total (E x 2 numpy array of ints): array where kth row [i, j]
                                                       storing the indices i and j of the kth edge's
                                                       endpoints in `verts_total`
            bc_total (n x 2 numpy array of floats): array where each row is a bar in the barcode
                                                   of the filtration on the Morse skeleton

    """
    # compute the connected components of im
    _, im_components, component_stats, _ = cv.connectedComponentsWithStats(
        mask.astype('uint8')
    )

    # indices of connected components sorted in increasing order by area
    # component_idx_sorted = np.argsort(component_stats[:, 4])

    # {component_stats[1:,4]} contains the area of each connected component,
    #   exluding the background which is row 0
    # {components_idx} contains indices of all components larger than {component_min_size}
    #   we add 1 to the idx because we removed the first row of {component_stats}
    components_idx = [
        idx+1
        for idx, area
        in enumerate(component_stats[1:,4])
        if area > component_min_size
    ]
    print(f"{len(components_idx)} connected components found\n\n")

    # Arrays that will store all vertices, edges, and barcodes of the disconnected Morse skeletons
    # These are initialized to being empty
    # however, we set the number of columns so we can use np.concatenate later.
    verts_total = np.zeros((0,2))
    edges_total = np.zeros((0,2), dtype=int)
    bc_total = np.zeros((0,2))

    # Compute the Morse skeleton and bar code for all
    # connected components larger than {component_min_size}
    for idx in components_idx:
        # pixels of the ith largest component of the image
        component = (im_components == idx)

        # find original pixel values in the component
        component = component.astype('uint8') * im

        # rescale the image to the range [0,1]
        component = component / 255

        # stats contain the top left corner (x_min, y_min) and weight and height
        # of bounding box around the connected component
        y_min, x_min, h, w, _ = component_stats[idx]

        # compute the Morse skeleton
        # we slice a bounding box of the connected component to speed up computation
        #
        # NOTICE: This function will return a warning about opening an empty file
        #         if the Morse skeleton is empty. Don't be alarmed. This is safe.
        dmtG = DMTGraph(component[x_min:x_min+w, y_min:y_min+h])
        verts, edges = dmtG.computeGraph(threshold1, threshold2)

        # If the graph is not empty, we compute and plot the barcode of the Morse skeleton.
        # The filtration adds the vertices and edges in decreasing order of
        # distance from the center.
        if(len(edges) > 0):
            # each row of verts are 2d coordinates of the vertex
            # the vertex coordinates are relative to the bounding box
            # so we translate them to be relative to the original image
            verts = np.array([[x + x_min, y + y_min] for x, y in verts])

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

            # draw the graph with networkx
            # create a dictionary with vertex positions
            # elements of vertex are in image coordinates, so we negate y-coordinate
            # positions = {v : [x, -y] for v, [x, y] in enumerate(verts)}
            # plot graph with vertex positions
            # nx.draw(G, pos=positions, width=0.1, node_size=5)
            # plt.draw()

            graph_components = [ G.subgraph(c).copy() for c in nx.connected_components(G) ]
            for g in graph_components:
                # center of the graph
                centers = nx.algorithms.distance_measures.center(g)
                center = centers[0]
                # Use a random vertex instead of the actual center,
                #  as centers can take a while to compute
                # center = np.random.randint(len(verts))
                # Distances of each vertex to the center
                distances = nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length(
                    g, center)

                # compute the shortest path tree of the graph component
                spt = __shortest_path_tree(g, distances)
                # add edges in shortest path tree to total list of edges
                spt_edges = np.array(spt.edges)
                edges_total = np.concatenate((edges_total, spt_edges), axis=0)

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

            # concatenate vertices and edges with vertices and edges of other connected components
            #
            # the indices of `edges` are relative to `verts`, not `verts_total`
            # we need to add the number vertics previously in the Morse skeletons
            # to each of these indices so they are relative to `verts_total`
            # (we also ignore the last column of edges as this contains a mysterious variable
            # used by the graph_recon package.)
            num_prev_verts = verts_total.shape[0]
            edges += num_prev_verts
            verts_total = np.concatenate((verts_total, verts), axis=0)
            # edges_total = np.concatenate((edges_total, edges), axis=0)

    return verts_total, edges_total, bc_total
