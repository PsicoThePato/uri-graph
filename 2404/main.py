import itertools

import sys
from functools import total_ordering
from typing import List

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

    def get_edge_list(self) -> List[Edge]:
            edge_list = []
            adj_matrix = self.graph_repr.graph_repr
            number_of_vertex = len(adj_matrix)
            for line_idx in range(number_of_vertex):
                for col_idx in range(line_idx, number_of_vertex):
                    if adj_matrix[line_idx][col_idx].has_edge:
                        fromV = line_idx
                        toV = col_idx
                        edge_weight = adj_matrix[line_idx][col_idx].weight
                        edge_list.append(Edge(fromV, toV, edge_weight))
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

    class UfArrayNode():
        def __init__(self, rank, parent):
            self.rank = rank
            self.parent = parent


    def initialize_uf_array(number_of_vertex):
        uf_array = [UfArrayNode(0, -1) for _ in range(number_of_vertex)]
        return uf_array

    def find(vertex, uf_array):
        if uf_array[vertex].parent == -1:
            return vertex
        uf_array[vertex].parent = find(uf_array[vertex].parent, uf_array)
        return uf_array[vertex].parent

    def union_by_rank(fromVertex, toVertex, uf_array):
        if uf_array[fromVertex].rank < uf_array[toVertex].parent:
            uf_array[fromVertex].parent = toVertex
        elif uf_array[toVertex].parent < uf_array[fromVertex].rank:
            uf_array[toVertex].parent = fromVertex
        else:
            uf_array[fromVertex].parent = toVertex
            uf_array[toVertex].rank = uf_array[toVertex].rank + 1
    
    def forms_cycle(fromVertex, toVertex, uf_array) -> bool:
        return (find(fromVertex, uf_array) == find(toVertex, uf_array))



    my_graph = Graph()
    my_graph.read_graph()
    adj_matrix = my_graph.graph_repr.graph_repr
    number_of_vertex = len(adj_matrix)
    edge_list = my_graph.get_edge_list()
    edge_list.sort(key= lambda edge: edge.weight)
    MST = []
    uf_array = initialize_uf_array(number_of_vertex)
    current_MST_size = 0
    max_MST_size = number_of_vertex - 1
    for edge in edge_list:
        if not forms_cycle(edge.fromV, edge.toV, uf_array):
            MST.append(edge)
            union_by_rank(edge.fromV, edge.toV, uf_array)
            current_MST_size = current_MST_size + 1
            if current_MST_size == max_MST_size:
                break
    total_sum = 0
    for edge in MST:
        total_sum = total_sum + edge.weight
    print(total_sum)

