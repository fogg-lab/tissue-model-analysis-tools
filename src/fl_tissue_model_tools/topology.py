from typing import Tuple
import math
import subprocess
import pickle
import os

import numpy as np
import numpy.typing as npt
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import cv2
from cv2 import cvtColor, COLOR_HSV2BGR

from pydmtgraph.dmtgraph import DMTGraph


class MorseGraph:
    """Morse skeleton of an image represented as a forest. Each tree is a connected component."""

    def __init__(self, img: npt.NDArray, thresholds: Tuple[float, float]=(1,4),
                 min_branch_length: int=15, smoothing_window: int=15):
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
        self.__smooth_graph(smoothing_window)
        self.__filter_graph(min_branch_length)


    ### Public methods ###


    def get_total_branch_length(self) -> float:
        """Get the sum of persistence interval lengths of a barcode."""
        bar_lengths = self.__barcode_interval_lengths()
        return np.sum(bar_lengths)


    def get_average_branch_length(self) -> float:
        """Return the average bar length of a barcode."""
        bar_lengths = self.__barcode_interval_lengths()
        bar_sum = np.sum(bar_lengths)
        return bar_sum / len(bar_lengths)


    def plot_colored_barcode(self, ax=None, **kwargs):
        """ Plot a colored barcode computed by `compute_colored_tree_and_barcode`.

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
        # sort bars in ascending order by birth time
        return_first_element = lambda pair : pair[0] # only use first element of tuple to sort
        self._barcode_and_colors.sort(reverse=True, key=return_first_element)
        # prepare args for bar plot
        heights = [*range(len(self._barcode_and_colors))]
        barcode, colors = zip(*self._barcode_and_colors)
        births, widths = zip(*[(bar[0], bar[1] - bar[0]) for bar in barcode])
        # plot the barcode
        ax.barh(heights, widths, left=births, color=colors, **kwargs)
        ax.set_yticks([])
        ax.set_xlabel("Barcode")
        if not ax_provided:
            plt.show()


    def plot_colored_tree(self, ax=None, **kwargs):
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

        Side effects:
            Calls self.__compute_colored_tree_and_barcode() to initialize tree & barcode as needed.
        """

        if not self._edges_and_colors:
            self.__compute_colored_tree_and_barcode()

        # if no axis is provided, fetch the current axis
        ax_provided = ax is not None
        ax = ax if ax_provided else plt.gca()
        # prepare the edges to be plotted
        edges, colors = zip(*self._edges_and_colors)
        # add alpha channel to colors (fixed at 1.0)
        colors = [(*c, 1.0) for c in colors]

        #if 'linewidth' not in kwargs and 'linewidths' not in kwargs:
            #kwargs['linewidth'] = np.max([xlim, ylim]) / 100.0

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

        G, vertices = self.__compute_nx_graph(img, *thresholds)

        # Make G a forest

        forest = nx.Graph()

        parent = {n: None for n in G.nodes()}
        dist_to_root = {}   # each node's distance to the root of its tree

        graph_components = [G.subgraph(c) for c in nx.connected_components(G)]

        skipped_vertices = set()

        for g in graph_components:
            root, max_degree = max(g.degree, key=lambda x: x[1])
            if max_degree <= 2:
                skipped_vertices.update(g.nodes())
                continue    # skip connected components that are just one branch
            # create a dict where each vertex points to its parent in the tree.
            # we set parents with a bfs starting at the root.
            # we also use the bfs to compute distance to root.
            parent[root] = root
            dist_to_root[root] = 0
            unvisited_vertices = [root]
            while unvisited_vertices:
                v = unvisited_vertices.pop(0)
                for n in G.neighbors(v):
                    if parent[n] is None:
                        forest.add_edge(v, n)
                        parent[n] = v
                        dist_to_root[n] = dist_to_root[v] + self.__edge_len(vertices, v, n)
                        unvisited_vertices.append(n)

        # check that the parents were set properly
        assert all([parent[n] is not None for n in G.nodes if n not in skipped_vertices])

        # check that the forest is a forest
        assert nx.is_forest(forest)

        self._parent = parent
        self._dist_to_root = dist_to_root
        self._G = forest
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
        max_dist_to_leaf = { v : -np.inf for v in self._G.nodes }
        branch_label = { }
        for leaf in leaves:
            current_vertex = leaf
            current_parent = parent[current_vertex]
            max_dist_to_leaf[leaf] = current_distance = 0
            branch_label[leaf] = leaf
            # This while loop follows the unique path from
            # a leaf to the root.
            while current_parent != current_vertex:
                current_distance += np.linalg.norm(verts[current_parent] - verts[current_vertex])
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
        """ Compute the branches and barcodes of the forest.

        Initializes:
            self._branches (list): A list of the branches of the tree.
                Each entry of branches is an E x 2 numpy array of edges
            self.barcode (list): Barcode of the tree with respect to a root.
                The bar barcode[i] corresponds to the branch branches[i].

        Raises:
            ValueError: If the graph is not a forest.
        """

        if not nx.is_forest(self._G):
            raise ValueError("Graph must be a forest")

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
                current_distance += self.__edge_len(verts, current_parent, current_vertex)
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


    def __smooth_graph(self, window_size):
        """ Smooth a graph using sliding window smoothing.
            Args:
                window_size (int): Size of the window to use for smoothing.
            Side effects:
                self._vertices is updated with the smoothed vertex positions.
        """

        if window_size <= 1:
            return

        vertices = self._vertices

        # We fix the position of all leaves and merge points between two branches
        fixed_verts = {v for v in self._G.nodes if self._G.degree[v] != 2}
        # the smoothed graph will have the same edges as the input graph
        # but different vertex positions
        new_vertices = vertices.copy()
        for branch in self._branches:
            # verify that edges in branch are consecutive
            # this should be true as this is how compute_branches_and_barcode works
            assert all([ branch[i][1] == branch[i+1][0] for i in range(len(branch)-1) ])
            branch_vertices = np.array( [branch[i][0] for i in range(len(branch))]+[branch[-1][1]] )
            # A branch may have fixed points besides its endpoints,
            # for example, when another branch is attached to the middle of the branch.
            # We therefore decompose the branch into the segments
            # connecting fixed points and smooth each of these segments.
            branch_fixed_vertices = [ i for i, vertex in enumerate(branch_vertices) if vertex in fixed_verts ]
            for i in range(len(branch_fixed_vertices)-1):
                segment_start, segment_end = branch_fixed_vertices[i], branch_fixed_vertices[i+1]
                segment_vertices = branch_vertices[segment_start:segment_end+1]
                smoothed_verts = self.__moving_average_fixed_ends(vertices[segment_vertices], window_size)
                new_vertices[segment_vertices] = smoothed_verts

        self._vertices = new_vertices


    def __filter_graph(self, min_branch_length):
        """ Remove all branches from a tree that are less than length min_branch_lengths.

        Args:
            min_branch_length (int):
                threshold for branch length.
                Any branch of shorter than min_branch_length is removed

        Updates:
            filtered_branches (E' x 2 numpy array of ints):
                updated to array of all edges in branches longer than min_branch_length
            self.barcode: updated to the barcode of filtered graph
            self._G: updated to the filtered graph (removes edges not in filtered_branches)
        """

        filtered_branches = []
        filtered_barcode = []
        filtered_edges = set()

        for branch, bar in zip(self._branches, self.barcode):
            birth, death = bar
            if death - birth > min_branch_length:
                filtered_branches.append(branch)
                filtered_barcode.append(bar)
                filtered_edges.update([tuple(e) for e in branch])

        self._branches = filtered_branches

        edges_to_remove = [e for e in self._G.edges if e not in filtered_edges]
        self._G.remove_edges_from(edges_to_remove)
        self.barcode = filtered_barcode


    def __barcode_interval_lengths(self) -> float:
        """Get persistence interval lengths in the barcode."""
        barcode = np.array(self.barcode)
        bar_lengths = barcode[:, 1] - barcode[:, 0]
        bar_lengths = bar_lengths[~np.isinf(bar_lengths)]
        return bar_lengths


    def __compute_colored_tree_and_barcode(self):
        """ Compute a tree and barcode colored according to branches.

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
    def __compute_nx_graph(im: npt.NDArray[np.double],
                         threshold1: float=0.5, threshold2: float=0.0):
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
        # Slice a bounding box of the connected component to speed up computation
        #dmtG = DMTGraph(im)
        #verts, edges = dmtG.computeGraph(threshold1, threshold2)

        # spawn subprocess to compute the graph as temporary workaround for memory leak in DMTGraph
        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        subprocess_path = os.path.join(cur_file_path, 'compute_graph_proc.py')
        rand_id=np.random.randint(0,1000000)
        fname=cur_file_path + '/tmp'+str(rand_id)
        #with tempfile.NamedTemporaryFile() as f:
        cv2.imwrite(fname+'.png', im)
        cmd = ['python', subprocess_path, fname+'.png', str(threshold1), str(threshold2), fname + '.pkl']
        subprocess.check_call(cmd)
        with open(fname + '.pkl', 'rb') as f2:
            verts, edges = pickle.load(f2)
        os.remove(fname+'.png')
        os.remove(fname+'.pkl')

        G = MorseGraph.__convert_to_networkx_graph(edges)

        verts = verts.astype(float)

        return G, verts


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
        assert min(n, math.ceil(len(A)/2)) == n

        # Elements less than window size from either end will not be repeated
        A_transformed = A[n-1:-(n-1)]
        for i in reversed(range(n-1)):
            idx1, idx2 = i, -i-1
            repeat = n - i
            A_transformed = np.concatenate(([A[idx1]]*repeat, A_transformed, [A[idx2]]*repeat))

        return A_transformed


    @staticmethod
    def __moving_average(A, n=3):
        """ Computes moving average of array `A` with window size `n` """
        ret = np.cumsum(A, axis=0, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:] / n


    @staticmethod
    def __moving_average_fixed_ends(A: npt.ArrayLike, n: int) -> npt.NDArray:
        """Computes moving average of array `A` with window size `n` fixed at original endpoints of A.
        Args:
            A: Array to be transformed.
            n: Window size for the moving average fit.
        """

        # make sure first and last window don't overlap more than 1 element at most
        n = min(n, math.ceil(len(A)/2))

        assert n != 0

        if n==1:
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

        for i in range(1, n-1):
            interp_dist = i * interp_step
            idx = np.searchsorted(accum_dists, interp_dist, side='right') - 1
            # Interpolate between vertices[idx] and vertices[idx+1]
            # The new vertex is a weighted average of the two vertices around it
            lineseg_travel_prop = (interp_dist - accum_dists[idx]) / (accum_dists[idx+1] - accum_dists[idx])
            new_vert = verts[idx] + (verts[idx+1] - verts[idx]) * lineseg_travel_prop
            interp_verts.append(new_vert)

        # The last vertex is fixed to the the last vertex of the original sequence
        interp_verts.append(verts[-1])

        return np.array(interp_verts)


    @staticmethod
    def __random_color(i: int):
        """ Convert an int to a random color """

        phi = 0.618033988749895
        step = 180*phi

        return cvtColor(np.array([[[step*i, 220, 255]]], np.uint8), COLOR_HSV2BGR)[0][0] / 255


    @staticmethod
    def __convert_to_networkx_graph(edges) -> nx.Graph:
        """ Convert a dmtgraph to a Networkx graph """
        G = nx.Graph()
        for vertex0, vertex1 in edges:
            # Each row in `edges` is an array [i, j],
            #   where i and j are ints representing the index of the endpoints.
            # Each row in `verts` is an array [x, y],
            #   where x and y are the 2d-coordinates of the vertex.
            G.add_edge(vertex0, vertex1)
        return G


    @staticmethod
    def __edge_len(verts, v1_idx, v2_idx):
        """ Compute the Euclidean distance between two vertices """
        return np.linalg.norm(verts[v1_idx] - verts[v2_idx])
