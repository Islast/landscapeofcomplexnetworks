import numpy as np
import networkx as nx

def perturb_function(f, node_list, epsilon=0.00000001):
    f_perturbed = {}
    for x in node_list:
        f_perturbed[x] = f(x) + np.random.uniform(low=0, high=epsilon)
    return lambda x: f_perturbed.get(x)

def Fast_search_of_nondegenerate_critical_nodes(G, h):
    graph = G.copy()
    C = {}
    basin = {}
    blank_nodes = {}
    for k in range(len(G)):
        C[k] = {}
        vertex_set = sorted(graph.nodes(),key=h)
        for j, v in enumerate(vertex_set):
            N = {y for y in graph.neighbors(v) if y in vertex_set[0:j]}
            if not N:
                C[k].update({v:{v}})
                basin.update({v:v})
            elif len({basin.get(y) for y in N if y in basin}) == 1:
                basin[v] = {basin.get(y) for y in N}.pop()
                C[k][basin[v]].add(v)
        blank_nodes[k+1] = [x for x in vertex_set if x not in basin]
        graph = graph.subgraph(blank_nodes[k+1])
    return C, basin, blank_nodes
