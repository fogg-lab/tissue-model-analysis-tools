import numpy as np
import numpy.typing as npt
import networkx as nx
import gudhi as gd



def compute_peristence_barcode_from_skeleton(
    edges: npt.ArrayLike, verts: npt.ArrayLike
):
    """
    Given a morse skeleton represented by its vertices and edges,
    this function computes the persistence barcode.
    """

    edges = edges.astype(int)
    # Create a networkx graph of Morse skeleton
    # We use the networkx graph to compute the distance of each vertex from
    #   the "center" of the graph
    # The center is a graph-theoretic notion defined here:
    #   https://en.wikipedia.org/wiki/Graph_center
    G = nx.Graph()
    for v0, v1, in edges:
        # Add each edge to the graph with weight = Euclidean distance between endpoints
        # Each row in `edges` is an array [i, j, _],
        #   where i and j are ints representing the index of the endpoints.
        # Each row in `verts` is an array [x, y],
        #   where x and y are the 2d-coordinates of the vertex.
        edge_length = np.linalg.norm(verts[v0]-verts[v1])
        G.add_edge(v0, v1, weight=edge_length)

    # center of the graph
    centers = nx.algorithms.distance_measures.center(G)
    center = centers[0]
    # Use a random vertex instead of the actual center,
    #  as centers can take a while to compute
    # Distances of each vertex to the center
    distances = nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length(
        G, center)
    
    spt = __shortest_path_tree(G, distances)
    spt_edges = np.array(spt.edges)

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
    for v0, v1, in spt_edges:
        K.insert([v0, v1], filtration = max(K.filtration([v0]), K.filtration([v1])))

    # compute the barcode of the Morse skeleton
    K.compute_persistence()
    # retrieve the 0-dimensional barcode
    bc = K.persistence_intervals_in_dimension(0)

    return bc

def __shortest_path_tree(
    G: nx.Graph, distances: dict
):
    """ Return the shortest path tree of a networkx graph.
        Args:
            G (networkx graph):
            distances (dictionary): dictionary of distances to the root
    """

    C = nx.Graph()
    for v0, v1 in G.edges():
        C.add_edge(v0, v1, weight=max(distances[v0], distances[v1]))
    return nx.minimum_spanning_tree(C)