from typing import Tuple, Dict, List, Union, Set
import numpy as np
import numpy.typing as npt


class Vertex:
    '''A vertex in a branching graph.'''
    def __init__(self, y_val: float, x_val: float, vert_id: int):
        self.y_val, self.x_val = y_val, x_val
        self.vert_id = vert_id
        self.edge_ids = set()

    def add_edge(self, edge_id: int):
        '''Add edge to vertex.'''
        self.edge_ids.add(edge_id)

    def remove_edge(self, edge_id: int):
        '''Remove edge from vertex.'''
        self.edge_ids.remove(edge_id)

class Edge:
    '''An edge connecting two vertices in a branching graph.'''
    def __init__(self, vert0: Vertex, vert1: Vertex):
        self.vertices = tuple([vert0, vert1].sort(key=lambda vertex: vertex.vert_id))

    def get_neighbour(self, vert_id: int) -> Union[Vertex, None]:
        '''Get neighbour of vertex.'''
        if vert_id == self.vertices[0].vert_id:
            return self.vertices[1]
        elif vert_id == self.vertices[1].vert_id:
            return self.vertices[0]
        else:
            return None

class Branch:
    '''A branch in a branching graph.'''
    def __init__(self, starting_vertex: Vertex, edges: Dict[int, Edge]):
        if len(starting_vertex.edge_ids) not in [1, 2]:
            raise ValueError('Starting vertex must have 1 or 2 edges.')
        self.vertices = self._get_vertices(starting_vertex, edges)
        self.endpoints = set((self.vertices[0], self.vertices[-1]))
        self.terminals = {v for v in self.endpoints if len(v.edge_ids) == 1}
        self.forks = self._get_forks(edges)

    def _get_next_neighbor(self, vert0, vert1, edges):
        '''Get next neighbor of vertex (vert0->vert1->next_neighbor).
        Return None if the number of next neighbors does not equal 1'''
        vert1_edges = [edges[edge_id] for edge_id in vert1.edge_ids
                       if edge_id not in vert0.edge_ids]
        if len(vert1_edges) != 1:
            return None
        return vert1_edges[0].get_neighbour(vert1.vert_id)

    def _get_vertices(self, initial_vertex: Vertex, edges: Dict[int, Edge]) -> List[Vertex]:
        '''Get vertices in branch starting from an initial vertex at an endpoint.'''
        if len(initial_vertex.edge_ids) not in [1, 2]:
            raise ValueError('Initial vertex must have 1 or 2 edges.')
        initial_vertex_edges = [edges[edge_id] for edge_id in initial_vertex.edge_ids]
        neighbor_verts = set()
        for edge in initial_vertex_edges:
            neighbor_verts.add(edge.get_neighbour(initial_vertex.vert_id))

        if len(neighbor_verts == 2):
            neighbor1, neighbor2 = neighbor_verts
            neighbor1_edges = [edges[edge_id] for edge_id in neighbor1.edge_ids]
            neighbor2_edges = [edges[edge_id] for edge_id in neighbor2.edge_ids]
            if len(neighbor1_edges) > 2 and len(neighbor2_edges) <= 2:
                next_vert = neighbor2
            elif len(neighbor2_edges) > 2 and len(neighbor1_edges) <= 2:
                next_vert = neighbor1
            elif len(neighbor1_edges) > 2 and len(neighbor2_edges) > 2:
                return [initial_vertex]
            else:
                raise ValueError('Initial vertex must be an endpoint on a branch.')
        elif len(neighbor_verts) == 1:
            next_vert = neighbor_verts.pop()
            if len(next_vert.edge_ids) > 2:
                return [initial_vertex]

        branch_verts = [initial_vertex, next_vert]

        branch_complete = False
        while not branch_complete:
            next_vert = self._get_next_neighbor(branch_verts[-2], branch_verts[-1], edges)
            if next_vert is None:
                branch_complete = True
            else:
                branch_verts.append(next_vert)

        return branch_verts

    def _get_forks(self, edges: Dict[int, Edge]) -> Set[Vertex]:
        '''Get forks in branch.'''

        forks = set()

        for vert in self.endpoints:
            for edge_id in vert.edge_ids:
                neighbor = edges[edge_id].get_neighbour(vert.vert_id)
                if neighbor not in self.vertices and len(neighbor.edge_ids) > 2:
                    forks.add(neighbor)

        return forks

class BranchingGraph:
    '''A branching graph.

    Parameters
    ----------
    vertices (npt.NDArray): Array of vertices [y, x].
    edges (npt.NDArray): Array of edges [vert1, v2]. vert1 and v2 are indices of vertices.
    '''

    def __init__(self, vertices_in: npt.NDArray, edges_in: npt.NDArray[np.int32]):
        self.vertices = {i: Vertex(y, x, i) for i, (y, x) in enumerate(vertices_in)}
        self.edges = self._get_edges(edges_in)
        self._add_edges_to_vertices()
        self.branches = self._get_branches()

    def _get_edges(self, edges: npt.NDArray[np.int32]) -> Dict[int, Edge]:
        '''Get edges from array of edges.'''
        edges = {edge_id: Edge(vert0, vert1) for edge_id, (vert0, vert1) in enumerate(edges)}
        return edges

    def _remove_vertex(self, vert_id: int):
        '''Remove vertex from graph.'''
        removed_vertex = self.vertices.pop(vert_id)
        for edge_id in removed_vertex.edge_ids:
            other_vert = self.edges[edge_id].get_neighbour(vert_id)
            other_vert.remove_edge(edge_id)
            del self.edges[edge_id]

    def _add_edges_to_vertices(self):
        '''Add edges to vertices.'''
        for edge_id, edge in self.edges.items():
            vert0, vert1 = edge.vertices
            self.vertices[vert0].add_edge(edge_id)
            self.vertices[vert1].add_edge(edge_id)

    def _get_branches(self) -> Dict[int, Branch]:
        '''Get branches from graph.'''

        branch_endpoints = set()
        forks = set()

        for vertex in self.vertices.values():
            if len(vertex.edge_ids) > 2:
                forks.add(vertex.vert_id)
            elif len(vertex.edge_ids) == 1:
                branch_endpoints.add(vertex.vert_id)

        for fork in forks:
            for edge_id in fork.edge_ids:
                neighbor = self.edges[edge_id].get_neighbour(fork.vert_id)
                #TEMP - making sure neighbors of forks cannot also be forks
                if len(neighbor.edge_ids) > 2:
                    raise ValueError('A neighbor of a fork is also a fork.')
                branch_endpoints.add(neighbor)

        seen_vertices = set()
        current_branch_id = 0
        branches = {}

        for vert_id in branch_endpoints:
            if vert_id in seen_vertices:
                continue
            new_branch = Branch(vert_id, self.edges)
            seen_vertices.update(new_branch.vertices)
            branches[current_branch_id] = new_branch
            current_branch_id += 1

        return branches

    def get_verts_edges(self) -> Tuple(npt.NDArray, npt.NDArray[np.int32]):
        '''Get vertices and edges.'''

        # Reassign vertex ids
        for i, vertex in self.vertices.items():
            vertex.vert_id = i

        edges = np.array([[v.vert_id for v in edge] for edge in self.edges])
        vertices = np.array([[vertex.y, vertex.x] for vertex in self.vertices.values()])

        return vertices, edges
