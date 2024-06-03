from typing import Tuple
import math
from numbers import Number

import numpy as np
import numpy.typing as npt
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_HSV2BGR

from fl_tissue_model_tools.dmtgraph import compute_dmt_graph


class MorseGraph:
    """Morse skeleton of an image represented as a forest. Each tree is a connected component."""

    def __init__(
        self,
        img: npt.NDArray,
        thresholds: Tuple[Number, Number] = (1, 4),
        min_branch_length: int = 15,
        smoothing_window: int = 15,
        pruning_mask: npt.NDArray = None,
        method=0,
    ):
        self.method = method
        self.smoothing_window = smoothing_window
        self.thresholds = thresholds
        self.min_branch_length = min_branch_length
        self.pruning_mask = pruning_mask
        self._shape = img.shape[:2]
        self.barcode = None
        self._leaves = None
        self._branches = None
        self._parent = None
        self._dist_to_root = None
        self._edges_and_colors = None
        self._barcode_and_colors = None
        self._G = None
        self._branch_label = None
        self._vertices = None
        self.__compute_graph(img, thresholds)
        self.__get_branch_labels()
        self.__compute_branches_and_barcode()
        self.__filter_graph()

    ### Public methods ###

    def get_total_branch_length(self) -> float:
        """Get the sum of persistence interval lengths of a barcode."""
        bar_lengths = self.__barcode_interval_lengths()
        return np.sum(bar_lengths)

    def get_average_branch_length(self) -> float:
        """Return the average bar length of a barcode."""
        bar_lengths = self.__barcode_interval_lengths()
        bar_sum = np.sum(bar_lengths)
        if bar_sum == 0:
            return 0
        return bar_sum / len(bar_lengths)

    def plot_colored_barcode(self, ax=None, **kwargs):
        """Plot a colored barcode computed by `compute_colored_tree_and_barcode`.

        Args:
            barcode_and_colors (list): list of bars and colors in the format returned
                by `compute_colored_tree_and_barcode`. Each item in the list is
                a tuple of a persistence pair and a color.
            ax (matplotlib.axes.Axes): axis on which to plot barcode. defaults to None.
                If no axis is provided, the tree is just plotted on the current axis of plt.
            kwargs (dict): Additional keyword arguments.
                These are forwarded to the ax.barh call.

        Initializes:
            Calls self.__compute_colored_tree_and_barcode() to initialize tree & barcode as needed.
        """

        if not self._barcode_and_colors:
            self.__compute_colored_tree_and_barcode()

        # if no axis is provided, fetch the current axis
        ax_provided = ax is not None
        ax = ax if ax_provided else plt.gca()
        if self._barcode_and_colors:
            # sort bars in ascending order by birth time
            return_first_element = lambda pair: pair[
                0
            ]  # only use first element of tuple to sort
            self._barcode_and_colors.sort(reverse=True, key=return_first_element)
            # prepare args for bar plot
            heights = [*range(len(self._barcode_and_colors))]
            barcode, colors = zip(*self._barcode_and_colors)
            births, widths = zip(*[(bar[0], bar[1] - bar[0]) for bar in barcode])
        else:
            heights = []
            widths = []
            births = []
            colors = []
        # plot the barcode
        ax.barh(heights, widths, left=births, color=colors, **kwargs)
        ax.set_yticks([])
        ax.set_xlabel("Barcode")
        if not ax_provided:
            plt.show()

    def plot_colored_tree(self, ax=None, **kwargs):
        """Plot a colored tree computed by `compute_colored_tree_and_barcode`

        Args:
            edges_and_colors (list): List of edges and colors.
                Each item in the list is a tuple of a line and an rgb color.
                Each line in a tuple with the 2d endpoints of an edge in the tree
            ax (matplotlib.axes.Axes): axis on which to plot barcode. defaults to None.
                If no axis is provided, the tree is just plotted on the current axis of plt.
            kwargs (dict): Additional keyword arguments.
                These are forwarded to the LineCollection constructor.
                For example, kwargs could contain linewidth.

        Side effects:
            Calls self.__compute_colored_tree_and_barcode() to initialize tree & barcode as needed.
        """

        if not self._edges_and_colors:
            self.__compute_colored_tree_and_barcode()

        # if no axis is provided, fetch the current axis
        ax_provided = ax is not None
        ax = ax if ax_provided else plt.gca()

        if self._edges_and_colors:
            # prepare the edges to be plotted
            edges, colors = zip(*self._edges_and_colors)
            # add alpha channel to colors (fixed at 1.0)
            colors = [(*c, 1.0) for c in colors]

            edges_collection = LineCollection(edges, colors=colors, **kwargs)
            # plot the tree
            ax.add_collection(edges_collection)

        ax.set_axis_off()
        ax.autoscale()

        if not ax_provided:
            plt.show()

    ### Private methods ###

    def __compute_graph(self, img, thresholds) -> Tuple[nx.Graph, np.ndarray]:
        """Get morse skeleton graph and initialize attributes for the graph.
        Initializes:
            self._parent (dict): Maps each vertex to its parent in the tree.
            self._dist_to_root (dict): Maps each vertex to its distance to the root of its tree.
            self._G (nx.Graph): Graph of the morse skeleton (a forest).
            self._vertices (V x 2 numpy array of floats):
                Array where ith row stores 2d coordinate of ith vertex of a graph.
        """

        G, vertices = self.__compute_nx_graph(img, *thresholds, method=self.method)

        # Smooth vertices in the graph
        vertices = self.__smooth_graph(G, vertices, self.smoothing_window)

        # Remove segments of vertices inside the pruning mask
        G = self.__trim_graph(
            G, vertices, self.min_branch_length, self._shape, self.pruning_mask
        )

        # Compute minimum spanning forest of the graph
        self._G, self._parent, self._dist_to_root = self.__get_forest(G, vertices)

        self._vertices = vertices

    def __get_branch_labels(self):
        """Label each vertex with its branch.
            Each vertex in the tree belongs to a unique branch corresponding to a leaf.
            Specifically, a vertex is in the longest branch from a leaf to the center
            of all its descendant leaves.
            We do this by labelling all vertices on the path between the leaf and the center,
            unless we encounter vertex has already been labeled with a more distant leaf,
            which means all other vertices on the path are apart of this branch.

        Initializes:
            self._leaves (list): List of vertices that are leaves in the forest.
            self._branch_label (dict): maps each vertex to its branch.
        """

        parent = self._parent
        verts = self._vertices
        leaves = [n for n in self._G.nodes if self._G.degree[n] == 1]
        max_dist_to_leaf = {v: -np.inf for v in self._G.nodes}
        branch_label = {}
        for leaf in leaves:
            current_vertex = leaf
            current_parent = parent[current_vertex]
            max_dist_to_leaf[leaf] = current_distance = 0
            branch_label[leaf] = leaf
            # This while loop follows the unique path from
            # a leaf to the root.
            while current_parent != current_vertex:
                current_distance += np.linalg.norm(
                    verts[current_parent] - verts[current_vertex]
                )
                if current_distance < max_dist_to_leaf[current_parent]:
                    # We've reached a vertex that has a descendant leaf that is
                    # further away, so it is part of another branch.
                    # Thus, we quit our traversal.
                    break
                current_vertex = current_parent
                current_parent = parent[current_vertex]
                max_dist_to_leaf[current_vertex] = current_distance
                branch_label[current_vertex] = leaf

        self._leaves = leaves
        self._branch_label = branch_label

    def __compute_branches_and_barcode(self) -> None:
        """Compute the branches and barcodes of the forest.

        Initializes:
            self._branches (list): A list of the branches of the tree.
                Each entry of branches is an E x 2 numpy array of edges
            self.barcode (list): Barcode of the tree with respect to a root.
                The bar barcode[i] corresponds to the branch branches[i].

        Raises:
            ValueError: If the graph is not a forest.
        """

        branches = []
        barcode = []
        verts = self._vertices

        for leaf in self._leaves:
            current_vertex = leaf
            current_label = leaf
            current_parent = self._parent[leaf]
            current_distance = 0
            current_branch = []
            # Follow the path from the leaf
            # until we encounter another branch or reach the root.
            # Add each edge along the way with the color of the branch.
            while current_label == leaf and current_vertex != current_parent:
                # update distance from `leaf`.
                # this is used after the loop finishes to compute the barcode
                current_distance += self.__edge_len(
                    verts, current_parent, current_vertex
                )
                # add current edge to list of returned edges
                current_branch.append((current_vertex, current_parent))
                # update pointers for next iteration of loop
                current_vertex = current_parent
                current_parent = self._parent[current_vertex]
                current_label = self._branch_label[current_vertex]
            branches.append(np.array(current_branch))
            # add the branch of the current leaf to the barcode
            # its birth is the (negative) distance of the leaf to the center
            # the death is the distance where we encounter a longer branch
            birth = -self._dist_to_root[leaf]
            death = birth + current_distance
            barcode.append((birth, death))

        self._branches = branches
        self.barcode = barcode

    def __smooth_graph(self, G, vertices, window_size):
        """Smooth a graph's vertex positions using sliding window smoothing.
        Args:
            G (nx.Graph): Graph to smooth vertices of.
            vertices (np.ndarray): Array of vertex positions.
            window_size (int): Size of the window to use for smoothing.
        Returns:
            np.ndarray: Smoothed vertex positions.
        """

        if window_size <= 1:
            return

        vertices = vertices.copy()

        # Fix the position of all leaves and junctions
        fixed_verts = {v for v in G.nodes if G.degree[v] != 2}
        visited = set()  # visited vertices

        for fixed_vert_start in fixed_verts:
            for segment_base_vert in G.neighbors(fixed_vert_start):
                branch_vert = segment_base_vert
                if branch_vert in visited:
                    continue
                segment_vertices = [fixed_vert_start, branch_vert]
                branch_verts_visited = set()
                while G.degree[branch_vert] == 2:
                    neighbors = list(G.neighbors(branch_vert))
                    next_vert = (
                        neighbors[0] if neighbors[0] != branch_vert else neighbors[1]
                    )
                    if next_vert in branch_verts_visited:
                        break
                    branch_vert = next_vert
                    branch_verts_visited.add(branch_vert)
                    segment_vertices.append(branch_vert)
                segment_vertices_pos = vertices[segment_vertices]
                smoothed_verts = self.__moving_average_fixed_ends(
                    segment_vertices_pos, window_size
                )
                vertices[segment_vertices] = smoothed_verts
                visited.update([segment_vertices[0], segment_vertices[-1]])

        return vertices

    def __filter_graph(self):
        """Remove all branches from the graph that are shorted min_branch_length.

        Args:
            min_branch_length (int):
                threshold for branch length.
                Any branch of shorter than min_branch_length is removed

        Updates:
            filtered_branches (E' x 2 numpy array of ints):
                updated to array of all edges in branches longer than min_branch_length
            self.barcode: updated to the barcode of filtered graph
        """

        filtered_branches = []
        filtered_barcode = []
        edges_to_remove = []

        for branch, bar in zip(self._branches, self.barcode):
            birth, death = bar
            if death - birth >= self.min_branch_length:
                filtered_branches.append(branch)
                filtered_barcode.append(bar)
            else:
                edges_to_remove.extend(branch)

        self._branches = filtered_branches
        self.barcode = filtered_barcode
        self._G.remove_edges_from(edges_to_remove)
        self._G.remove_nodes_from(list(nx.isolates(self._G)))

    def __barcode_interval_lengths(self) -> np.ndarray:
        """Get persistence interval lengths in the barcode."""
        if not self.barcode:
            return np.array([])
        barcode = np.array(self.barcode)
        bar_lengths = barcode[:, 1] - barcode[:, 0]
        bar_lengths = bar_lengths[~np.isinf(bar_lengths)]
        return bar_lengths

    def __compute_colored_tree_and_barcode(self):
        """Compute a tree and barcode colored according to branches.

        Initialize the following attributes:
            edges_and_colors (list): List of edges and colors.
                Each item in the list is a tuple of a edge and an rgb color.
                Each edge in a tuple with the 2d endpoints of an edge in the tree
            barcode_and_colors (list): list of bars and colors.
                Each item in the list is a tuple of a persistence pair and an rgb color.
        """

        edges_and_colors = []
        barcode_and_colors = []

        for i, (branch, bar) in enumerate(zip(self._branches, self.barcode)):
            color = self.__random_color(i)
            barcode_and_colors.append((bar, color))
            for v1idx, v2idx in branch:
                v1 = self._vertices[v1idx]
                v2 = self._vertices[v2idx]
                # reverse v1 and v2 as mpl uses image coordinates
                c1 = (v1[1], v1[0])
                c2 = (v2[1], v2[0])
                edges_and_colors.append(([c1, c2], color))

        self._edges_and_colors = edges_and_colors
        self._barcode_and_colors = barcode_and_colors

    ### Utilities ###

    @staticmethod
    def __compute_nx_graph(
        im: npt.NDArray,
        threshold1: Number = 0.5,
        threshold2: Number = 0.0,
        method=0,
    ):
        """Fit a Morse skeleton to the image `im`.
        Args:
            im (npt.NDArray[np.float64]): Grayscale image
            threshold1 (float): threshold for the Morse skeleton simplification step. See paper.
            threshold2 (float): threshold for the edges. We only take the 1-unstable manifold of edges
                                with value > threshold2. The higher the value, the more disconnected
                                the graph will be
        Returns:
            G (nx.Graph): Graph of the Morse skeleton.
            verts (V x 2 numpy array of floats): vertices of DMTGraph generated from image
        """

        # Compute the Morse skeleton
        V, E = compute_dmt_graph(im.astype(np.float32), threshold1, threshold2)

        print(f"{V.shape=}, {V.dtype=}")
        print(f"{E.shape=}, {E.dtype=}")

        G = MorseGraph.__convert_to_networkx_graph(E)

        V = V.astype(np.float32)

        return G, V

    @staticmethod
    def __prep_moving_avg_fixed_endpoints(A: npt.ArrayLike, n: int) -> npt.NDArray:
        """Return array prepared for moving average fit fixed at original endpoints of A.
        Args:
            A: Array to be transformed.
            n: Window size for the moving average fit.
        """
        # Repeat elements in the first and last windows n - (indices from endpoint) times.
        # With this method, the moving average is fixed to the original endpoints in a natural way.
        # Examples:
        #   n=3, A=[0,1,2,3,4,5,6,7], A_transformed=[0,0,0,1,1,2,3,4,5,6,6,7,7,7]
        #   n=4, A=[0,1,2,3,4,5,6,7], A_transformed=[0,0,0,0,1,1,1,2,2,3,4,5,5,6,6,6,7,7,7,7]

        # make sure window size is at least 2
        assert n >= 2

        # make sure first and last window don't overlap more than 1 element at most
        assert min(n, math.ceil(len(A) / 2)) == n

        # Elements less than window size from either end will not be repeated
        A_transformed = A[n - 1 : -(n - 1)]
        for i in reversed(range(n - 1)):
            idx1, idx2 = i, -i - 1
            repeat = n - i
            A_transformed = np.concatenate(
                ([A[idx1]] * repeat, A_transformed, [A[idx2]] * repeat)
            )

        return A_transformed

    @staticmethod
    def __moving_average(A, n=3):
        """Computes moving average of array `A` with window size `n`"""
        ret = np.cumsum(A, axis=0, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    @staticmethod
    def __moving_average_fixed_ends(A: npt.ArrayLike, n: int) -> npt.NDArray:
        """Computes moving average of array `A` with window size `n` fixed at original endpoints of A.
        Args:
            A: Array to be transformed.
            n: Window size for the moving average fit.
        """

        # make sure first and last window don't overlap more than 1 element at most
        n = min(n, math.ceil(len(A) / 2))

        assert n != 0

        if n == 1:
            return A

        A_transformed = MorseGraph.__prep_moving_avg_fixed_endpoints(A, n)
        moving_avg = MorseGraph.__moving_average(A_transformed, n)

        return MorseGraph.__interp_n_verts_uniform_spacing(moving_avg, len(A))

    @staticmethod
    def __interp_n_verts_uniform_spacing(verts: npt.ArrayLike, n: int) -> npt.NDArray:
        """Interpolate n points along the polygonal chain defined by `verts`.
            The returned sequence is fixed to start and end at the original endpoints of `verts`.
        Args:
            verts: Original sequence of vertices.
            n: Number of interpolated vertices to return.
        Returns:
            npt.NDArray: Interpolated vertices, uniformly spaced by euclidean distance.
        """

        assert len(verts) >= 2
        assert n >= 2

        # Get the spacing to use between interpolated vertices
        dists = np.linalg.norm(verts[1:] - verts[:-1], axis=1)
        total_dist = np.sum(dists)
        accum_dists = np.cumsum(np.concatenate(([0], dists)))
        interp_step = total_dist / (n - 1)

        # The first vertex is fixed to the the first vertex of the original sequence
        interp_verts = [verts[0]]

        for i in range(1, n - 1):
            interp_dist = i * interp_step
            idx = np.searchsorted(accum_dists, interp_dist, side="right") - 1
            # Interpolate between vertices[idx] and vertices[idx+1]
            # The new vertex is a weighted average of the two vertices around it
            lineseg_travel_prop = (interp_dist - accum_dists[idx]) / (
                accum_dists[idx + 1] - accum_dists[idx]
            )
            new_vert = verts[idx] + (verts[idx + 1] - verts[idx]) * lineseg_travel_prop
            interp_verts.append(new_vert)

        # The last vertex is fixed to the the last vertex of the original sequence
        interp_verts.append(verts[-1])

        return np.array(interp_verts)

    @staticmethod
    def __random_color(i: int):
        """Convert an int to a random color"""

        phi = 0.618033988749895
        step = 180 * phi

        return (
            cvtColor(np.array([[[step * i, 220, 255]]], np.uint8), COLOR_HSV2BGR)[0][0]
            / 255
        )

    @staticmethod
    def __convert_to_networkx_graph(edges) -> nx.Graph:
        """Convert a dmtgraph to a Networkx graph"""
        G = nx.Graph()
        for vertex0, vertex1 in edges:
            # Each row in `edges` is an array [i, j],
            #   where i and j are ints representing the index of the endpoints.
            # Each row in `verts` is an array [x, y],
            #   where x and y are the 2d-coordinates of the vertex.
            G.add_edge(vertex0, vertex1)
        return G

    @staticmethod
    def __get_forest(G: nx.Graph, verts: np.ndarray) -> Tuple[nx.Graph, dict, dict]:
        """Get a minimum spanning forest of the graph `G`, node parents,
            and node distances to the root node of each tree.
        Args:
            G: Graph to get the minimum spanning forest of.
            verts: Vertices of the graph.
        Returns:
            Tuple[nx.Graph, dict, dict]: The minimum spanning forest of `G`,
                node parents, and node distances to the root node of each tree.
        """

        forest = nx.Graph()
        parent = {n: None for n in G.nodes()}
        dist_to_root = {}  # each node's distance to the root of its tree

        skipped_vertices = set()

        for g in [G.subgraph(c) for c in nx.connected_components(G)]:
            root, max_degree = max(g.degree, key=lambda x: x[1])
            if max_degree <= 2:
                skipped_vertices.update(g.nodes())
                continue  # skip isolated branches
            # create a dict where each vertex points to its parent in the tree.
            # we set parents with a bfs starting at the root.
            # we also use the bfs to compute distance to root.
            parent[root] = root
            dist_to_root[root] = 0
            unvisited_vertices = [root]
            while unvisited_vertices:
                v = unvisited_vertices.pop(0)
                for n in g.neighbors(v):
                    if parent[n] is None:
                        forest.add_edge(v, n)
                        parent[n] = v
                        dist_to_root[n] = dist_to_root[v] + MorseGraph.__edge_len(
                            verts, v, n
                        )
                        unvisited_vertices.append(n)

        # check that the parents were set properly
        # assert all([parent[n] is not None for n in G.nodes if n not in skipped_vertices])

        # check that the resulting graph is a forest
        # assert nx.is_forest(forest)

        return forest, parent, dist_to_root

    @staticmethod
    def __edge_len(verts, v1_idx, v2_idx):
        """Compute the Euclidean distance between two vertices"""
        return np.linalg.norm(verts[v1_idx] - verts[v2_idx])

    @staticmethod
    def __trim_graph(
        G: nx.Graph,
        vertices: npt.NDArray,
        min_branch_length: int,
        shape: Tuple[int, int],
        pruning_mask: npt.NDArray = None,
    ) -> nx.Graph:
        """Remove branches that are too short or are positioned inside the pruning mask.
        Args:
            G: Graph to trim.
            vertices: Physical positions of the graph nodes.
            pruning_mask: Mask of regions to prune.
        Returns:
            nx.Graph: Pruned graph.
        """

        G = G.copy()

        if pruning_mask is None:
            pruning_mask = np.zeros(shape, dtype=bool)
        elif pruning_mask.dtype != bool:
            pruning_mask = pruning_mask > 0

        def get_segment_length(segment):
            # segment is an edge-connected sequence of nodes that aren't junctions
            edge_lengths = [
                MorseGraph.__edge_len(vertices, n1, n2) for n1, n2 in G.edges(segment)
            ]
            return sum(edge_lengths)

        # Remove segments that are too short or are positioned inside the mask.
        # Pass 1: Prune offshoots and isolated segments, starting from leaves.
        # Pass 2: Prune other segments, starting from junctions.
        # After each pass, also remove isolated nodes in the graph.
        # These two passes are repeated until there are no more segments to prune.

        pass_num = 1
        pruning_complete = False

        while not pruning_complete:

            junctions = {n for n in G.nodes if G.degree[n] > 2}
            base_nodes = (
                {n for n in G.nodes if G.degree[n] == 1} if pass_num == 1 else junctions
            )
            unmarked_nodes = {n for n in G.nodes if n not in junctions}
            segments = []
            short_segments = []

            while base_nodes:
                starting_node = base_nodes.pop()
                neighbors = {
                    n for n in G.neighbors(starting_node) if n in unmarked_nodes
                }
                while neighbors:
                    node = neighbors.pop()
                    segment = [node]
                    while neighbor := [
                        n for n in G.neighbors(node) if n in unmarked_nodes
                    ]:
                        node = neighbor[0]
                        segment.append(node)
                        unmarked_nodes.remove(node)
                    # if one of the segment's endpoints is a leaf, consider removal based on length
                    segment_has_leaf = (
                        G.degree[segment[0]] == 1 or G.degree[segment[-1]] == 1
                    )
                    if (
                        segment_has_leaf
                        and get_segment_length(segment) < min_branch_length
                    ):
                        short_segments.append(segment)
                    else:
                        segments.append(segment)

            if segments:
                segment_pos = [
                    np.round(np.median(vertices[s], axis=0)).astype(int)
                    for s in segments
                ]
                segments_remove_idx = np.argwhere(
                    pruning_mask[tuple(zip(*segment_pos))]
                ).flatten()
                segments_to_remove = [segments[i] for i in segments_remove_idx]
            else:
                segments_to_remove = []
            segments_to_remove.extend(short_segments)

            for segment in segments_to_remove:
                G.remove_edges_from(set(G.edges(segment)))
                G.remove_nodes_from(segment)

            G.remove_nodes_from(list(nx.isolates(G)))

            pruning_complete = pass_num == 2 and not segments_to_remove
            pass_num = 2 if pass_num == 1 else 1

        # end = perf_counter_ns()
        # print(f"Pruning took {round((end - start) / 1e6)} ms")

        return G
