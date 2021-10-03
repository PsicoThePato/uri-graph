import itertools

import sys
from functools import total_ordering
from typing import List

import pprint
import copy

@total_ordering
class MatrixNode():
    def __init__(self, has_edge: int, weight: float):
        self.has_edge = has_edge
        self.weight = weight
    
    def __repr__(self):
        rep = f"{self.weight}"
        return rep

    def __eq__(self, other) -> bool:
        if not isinstance(other, MatrixNode):
            return TypeError
        return self.weight == other.weight

    def __gt__(self, other) -> bool:
        if not isinstance(other, MatrixNode):
            return TypeError
        return self.weight > other.weight


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


SHIRT_SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
SHIRT_MAPPING = {}
for idx, size in enumerate(SHIRT_SIZES):
    SHIRT_MAPPING[size] = idx


class Graph:
    def __init__(self):
        self.graph_repr = None
        self.nShirts = None
        self.nPersons = None

    def __repr__(self) -> str:
        return str(self.graph_repr)

    def read_graph(self, path: str=None, represent_as="matrix"):
        fp = sys.stdin
        self.nShirts, self.nPersons = self.__get_V_E(fp)
        lines = [next(fp) for _ in range(self.nPersons)]
        #_ = fp.readlines()
        self.isDigraph = False
        self.graph_repr = self.AdjMatrix(self.nPersons, self.nShirts, self.isDigraph, lines)

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
        def __init__(self, nPersons: int, nShirts: int, isDigraph, lines: str):
            self.header = "digraph G" if isDigraph else "graph G"
            self.isDigraph = isDigraph
            self.isValued = False
            self.graph_repr = self.__get_adj_matrix(lines, nPersons, nShirts, isDigraph)

        def __get_adj_matrix(self, lines: str, nPersons: int, nShirts: int, isDigraph):
            adj_matrix = [[MatrixNode(0, 0) for _ in range(nPersons + len(SHIRT_SIZES) + 2)] for _ in range(nPersons + len(SHIRT_SIZES) + 2)]
            #print(f"Minha matrix é de {len(adj_matrix)}")
            self.isValued = True
            for idx, line in enumerate(lines):
                #  first vertex is reserved to flow source
                person_idx = idx + 1
                #  connects source to person
                (adj_matrix[0][person_idx]).has_edge = 1
                (adj_matrix[0][person_idx]).weight = 1

                shirt1, shirt2 = line.strip().split(" ")
                shirt1_idx = nPersons + SHIRT_MAPPING[shirt1] + 1
                shirt2_idx = nPersons + SHIRT_MAPPING[shirt2] + 1
                
                #  connects person to shirts
                (adj_matrix[person_idx][shirt1_idx]).has_edge = 1
                (adj_matrix[person_idx][shirt1_idx]).weight = 1
                (adj_matrix[person_idx][shirt2_idx]).has_edge = 1
                (adj_matrix[person_idx][shirt2_idx]).weight = 1
                
                #  since it's always a multiple of len(SHIRT_SIZES), this gives the quantity of each size
                shirtsPerSize = int(nShirts/len(SHIRT_SIZES))
                #  connects shirts to target
                (adj_matrix[shirt1_idx][-1]).has_edge = 1
                (adj_matrix[shirt1_idx][-1]).weight = shirtsPerSize
                (adj_matrix[shirt2_idx][-1]).has_edge = 1
                (adj_matrix[shirt2_idx][-1]).weight = shirtsPerSize

            return adj_matrix

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

    def dijkstra(adj_matrix) -> List[Edge]:
        def matrix_to_adj_list(adj_matrix):
            vertex_list = [HeapNodeVertex(vertex, float('inf'), None, []) for vertex in range(len(adj_matrix))]

            for fromV, line in enumerate(adj_matrix):
                for toV, vertex in enumerate(line):
                    if vertex.weight > 0:
                        vertex_list[fromV].adj_list.append(vertex_list[toV])
            return vertex_list


        import itertools
        import heapq
        REMOVED = '<removed-vertex>'      # placeholder for a removed vertex

        def add_vertex(vertex: HeapNodeVertex, queue, entry_dict, counter, priority=0):
            'Add a new vertex or update the priority of an existing vertex'
            if vertex.vertex_name in entry_dict:
                remove_vertex(vertex, entry_dict)
            count = next(counter)
            entry = [priority, count, vertex]
            entry_dict[vertex.vertex_name] = entry
            heapq.heappush(queue, entry)

        def remove_vertex(vertex: HeapNodeVertex, entry_dict):
            'Mark an existing vertex as REMOVED.  Raise KeyError if not found.'
            entry = entry_dict.pop(vertex.vertex_name)
            entry[-1] = REMOVED

        def pop_vertex(queue, entry_dict) -> HeapNodeVertex:
            'Remove and return the lowest priority vertex. Raise KeyError if empty.'
            while queue:
                _, _, vertex = heapq.heappop(queue)
                if vertex is not REMOVED:
                    del entry_dict[vertex.vertex_name]
                    return vertex
            raise KeyError('pop from an empty priority queue')

        #  start of ACTUAL dijikstra algorithm
        vertex_list = matrix_to_adj_list(adj_matrix)
        vertex_set = [True] * len(adj_matrix)

        entry_finder = {}               # mapping of vertexs to entries
        counter = itertools.count()     # unique sequence count
        vertex_pq = []
        vertex_list[0].key_wt = 0
        for vertex in vertex_list:
            add_vertex(vertex, vertex_pq, entry_finder, counter, vertex.key_wt)
        vertex_path = [None] * len(vertex_set)
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
            vertex_path[vertex_from.vertex_name] = vertex_from
        
        path = []
        target = -1
        #print(f"Masoq {vertex_path[-1].vertex_name}")
        #print("V PATH")
        #print(vertex_path)
        while (target := vertex_path[target].vertex_name) != 0:
            pred = vertex_path[target].pred_vertex
            if pred is None:
                return []
            current_vertex = vertex_path[target].vertex_name
            # print(pred)
            # print(current_vertex)
            path.append(Edge(pred, current_vertex, adj_matrix[pred][current_vertex].weight))
            target = pred
        #print([*reversed(path)])
        return [*reversed(path)]

    def print_matrix(matrix):
        for line in matrix:
            print(line)


    def ford_fulkerson(adj_matrix: List[List[MatrixNode]]):
        pp = pprint.PrettyPrinter(4)
        max_flow = 0
        aug_path = dijkstra(adj_matrix)
        while aug_path:
            min_edge = min(aug_path)
            residual_capacity = min_edge.weight
            #print(f"residual capacity: {residual_capacity}")
            #print(min_edge)
            if residual_capacity == 0:
                print(f"FROM: {min_edge.fromV}")
                print(f"TO: {min_edge.toV}")
                print(f"Peso é {min_edge.weight}")
                print(max_flow)
                exit(0)
            #breakpoint()
            max_flow = max_flow + residual_capacity
            for edge in aug_path:
                fromV = edge.fromV
                toV = edge.toV
                current_residue = copy.copy(adj_matrix[fromV][toV].weight)
                adj_matrix[fromV][toV].weight = current_residue - residual_capacity
                adj_matrix[toV][fromV].weight = adj_matrix[toV][fromV].weight + residual_capacity
            
            aug_path = dijkstra(adj_matrix)           
            #print("###")
            #pp.pprint(adj_matrix)
            
            #print("###")
        return max_flow


    nTests = int(sys.stdin.readline())
    #print(nTests)
    for _ in range(nTests):
        myG = Graph()
        myG.read_graph()
        pp = pprint.PrettyPrinter(4)
        #pp.pprint(myG.graph_repr.graph_repr)
        adj_matrix = myG.graph_repr.graph_repr
        max_flow = ford_fulkerson(adj_matrix)
        if max_flow >= myG.nPersons:
            print("YES")
        else:
            print("NO")

