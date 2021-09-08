import itertools

import sys
from functools import total_ordering
import heapq

class MatrixNode():
    def __init__(self, has_edge: int, weight: float):
        self.has_edge = has_edge
        self.weight = weight
    
    def __repr__(self):
        rep = f"MatrixNode(has_edge: {self.has_edge}, weight: {self.weight})"
        return rep

@total_ordering
class Edge():
    def __init__(self, fromV, toV, weigth):
        self.fromV = fromV
        self.toV = toV
        self.weight = weigth

    def __eq__(self, other) -> bool:
        if not isinstance(other, Edge):
            return TypeError
        return self.weight == other.weight
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, Edge):
            return TypeError
        return self.weight > other.weight
    
    def __repr__(self) -> str:
        return f"Edge ({self.fromV, self.toV}; weight={self.weight})"


class Graph:
    def __init__(self):
        self.graph_repr = None

    def __repr__(self) -> str:
        return str(self.graph_repr)

    def read_graph(self, path: str=None, represent_as="matrix"):
        fp = sys.stdin
        self.V, self.E = self.__get_V_E(fp)
        lines = fp.readlines()
        self.isDigraph = False
        self.graph_repr = self.AdjMatrix(self.V, self.isDigraph, lines)

    def __get_V_E(self, fp):
        while line := fp.readline():
            return map(lambda x: int(x), line.split(" "))

    def write_file(self, path="data/output/"):
        print(self.graph_repr)

    def get_adj_list(self):
            edge_list = []
            for fromV, line in enumerate(self.graph_repr.graph_repr):
                adj_list = []
                for toV, vertex in enumerate(line):
                    if vertex.has_edge:
                        adj_list.append(Edge(fromV, toV, vertex.weight))
            return edge_list

    class AdjMatrix:
        def __init__(self, V: int, isDigraph, lines: str):
            self.header = "digraph G" if isDigraph else "graph G"
            self.isDigraph = isDigraph
            self.isValued = False
            self.graph_repr = self.__get_adj_matrix(lines, V, isDigraph)

        def __get_adj_matrix(self, lines: str, V: int, isDigraph):
            adj_matrix = [[MatrixNode(0, None) for _ in range(V)] for _ in range(V)]
            for line in lines:
                if line.strip()[0] == "c":
                    continue
                row, column, weight = map(lambda x: int(x), line.strip().split(" "))
                (adj_matrix[row - 1][column - 1]).has_edge = 1
                (adj_matrix[row - 1][column - 1]).weight = weight
                if weight != 0:
                    self.isValued = True
                if not isDigraph:
                    self.__add_simetric_edge(adj_matrix, row, column, weight)
            return adj_matrix

        def __add_simetric_edge(self, adj_matrix, row, column, weight):
            adj_matrix[column - 1][row - 1] = MatrixNode(1, weight)

        def __repr__(self) -> str:
            body = self.__get_formatted_graph()
            formated_string = self.header + "\n" + "{\n" + body + "}\n"
            return formated_string

        def __get_formatted_graph(self):
            separator = " -> " if self.isDigraph else " -- "
            get_weigth_str = (lambda weight: f" [label = {weight}]") if self.isValued else lambda _: ""
            get_col_range = lambda idx, matrix_dimension: range(0, matrix_dimension) if self.isDigraph else range(idx, matrix_dimension)
            
            body = ""
            matrix_dimension = len(self.graph_repr)
            for row_idx in range(matrix_dimension):
                for col_idx in get_col_range(row_idx, matrix_dimension):
                    if self.graph_repr[row_idx][col_idx].has_edge:
                        _from = str(row_idx + 1)
                        _to = str(col_idx + 1)
                        weight = self.graph_repr[row_idx][col_idx].weight
                        body = body + _from + separator + _to + get_weigth_str(weight) +";\n"
            return body


if __name__ == "__main__":
    class HeapNodeVertex():
        def __init__(self, vertex_name, key_wt, pred_vertex, adj_list):
            self.vertex_name = vertex_name
            self.key_wt = key_wt
            self.pred_vertex = pred_vertex
            self.adj_list = adj_list

        def __eq__(self, other) -> bool:
            if not isinstance(other, Edge):
                return TypeError
            return self.key_wt == other.key_wt
    
        def __gt__(self, other) -> bool:
            if not isinstance(other, Edge):
                return TypeError
            return self.key_wt > other.key_wt
    
        def __repr__(self) -> str:
            return f"Vertex {self.vertex_name + 1}"
    
    
    def matrix_to_adj_list(adj_matrix):
        vertex_list = [HeapNodeVertex(vertex, float('inf'), None, []) for vertex in range(len(adj_matrix))]

        for fromV, line in enumerate(adj_matrix):
            for toV, vertex in enumerate(line):
                if vertex.has_edge:
                    vertex_list[fromV].adj_list.append(vertex_list[toV])
        return vertex_list


    import itertools
    REMOVED = '<removed-vertex>'      # placeholder for a removed vertex

    def add_vertex(vertex, queue, entry_dict, counter, priority=0):
        'Add a new vertex or update the priority of an existing vertex'
        if vertex.vertex_name in entry_dict:
            remove_vertex(vertex, entry_dict)
        count = next(counter)
        entry = [priority, count, vertex]
        entry_dict[vertex.vertex_name] = entry
        heapq.heappush(queue, entry)

    def remove_vertex(vertex, entry_dict):
        'Mark an existing vertex as REMOVED.  Raise KeyError if not found.'
        entry = entry_dict.pop(vertex.vertex_name)
        entry[-1] = REMOVED

    def pop_vertex(queue, entry_dict):
        'Remove and return the lowest priority vertex. Raise KeyError if empty.'
        while queue:
            _, _, vertex = heapq.heappop(queue)
            if vertex is not REMOVED:
                del entry_dict[vertex.vertex_name]
                return vertex
        raise KeyError('pop from an empty priority queue')


    my_graph = Graph()
    my_graph.read_graph()
    
    adj_matrix = my_graph.graph_repr.graph_repr

    vertex_list = matrix_to_adj_list(adj_matrix)
    vertex_set = [True] * len(adj_matrix)

    entry_finder = {}               # mapping of vertexs to entries
    counter = itertools.count()     # unique sequence count
    vertex_pq = []
    for vertex in vertex_list:
        add_vertex(vertex, vertex_pq, entry_finder, counter, vertex.key_wt)
    MST = []
    while(vertex_pq and any(vertex_set)):
        vertex_from = pop_vertex(vertex_pq, entry_finder)
        vertex_from.key_wt = 0
        vertex_set[vertex_from.vertex_name] = False
        for adj_vertex in vertex_from.adj_list:
            edge_wt = adj_matrix[vertex_from.vertex_name][adj_vertex.vertex_name].weight
            if vertex_set[adj_vertex.vertex_name] and (edge_wt < adj_vertex.key_wt):
                adj_vertex.pred_vertex = vertex_from.vertex_name
                adj_vertex.key_wt = edge_wt
                add_vertex(adj_vertex, vertex_pq, entry_finder, counter, adj_vertex.key_wt)
        MST.append(vertex_from)

    total_sum = 0
    for vertex in reversed(MST[1:]):
        total_sum = total_sum + adj_matrix[vertex.vertex_name][vertex.pred_vertex].weight
    print(total_sum)