import numpy as np
import networkx as nx
import networkx.algorithms.dag as dag
import functools
import mogutda
import copy


def perturb_function(f, node_list, epsilon=0.00001):
    f_perturbed = {}
    for x in node_list:
        f_perturbed[x] = f(x) + np.random.uniform(low=0, high=epsilon)
    return lambda x: f_perturbed.get(x)


def Fast_search_of_nondegenerate_critical_nodes(nl):
    '''
    An algorithm to find the non-degenerate critical nodes of networklandscape
    based off of [cite]

    Returns nondegenerate_critical_nodes, attraction_basin_map where
    nondegenerate_critical_nodes is a dictionary mapping an integer k to the
    nondegenarate critical nodes of index k
    attraction_basin_map is a dictionary mapping a node x to the unique
    non-degenarate critical node which...

    nl should be a NetworkLandscape object or a tuple (G,h) where G is a
    networkx graph and h is an injective function from the nodes of G into an
    ordered set
    '''
    if not isinstance(nl, NetworkLandscape):
        nl = NetworkLandscape(*nl)
    G = nl.graph.copy()
    nondegenerate_critical_nodes = {}
    attraction_basin_map = {}
    for k in range(len(G)):
        nondegenerate_critical_nodes[k] = set()
        V = sorted(G.nodes(), key=nl.h)
        for v in V:
            downhill_neighbours = {y for y in G.neighbors(v)
                                   if nl.h(y) < nl.h(v)}
            if not downhill_neighbours:
                # The kth order nondegenerate critical nodes are precisely the
                # local minima in the subgraph in the kth pass through this for
                # loop
                nondegenerate_critical_nodes[k].add(v)
                attraction_basin_map.update({v: v})
            else:
                downhill_basins = {attraction_basin_map.get(y)
                                   for y in downhill_neighbours}
                if (len(downhill_basins) == 1 and downhill_basins != {None}):
                    attraction_basin_map[v] = downhill_basins.pop()
        # Restrict G to the separatrix
        G = G.subgraph(set(V).difference(set(attraction_basin_map.keys())))
    return nondegenerate_critical_nodes, attraction_basin_map


def recompose(f, n):
    '''compose function f with itself n times'''
    def r(s):
        return functools.reduce(lambda x, _: f(x), range(n), s)
    return r


class NetworkLandscape:
    '''
    A NetworkLandscape is a network with values assigned to its vertices by a
    function h. h must be injective (no two nodes can have the same value).

    NetworkLandscape attributes:
    - self.graph       the underlying network
    - self.h           the function defining weights on the nodes of self.graph
    - self.directed    each edge in self.graph is given a direction u-->v where
                       h(u) > h(v). This defines the directed graph
                       self.directed

    NetworkLandscape methods:
    - self.direct_graph()               used to define self.directed
    - self.downhill_path()              not particularly useful
    '''
    def __init__(self, G, h):
        '''
        initialise a NetworkLandscape object.
        G should be a networkx.Graph object
        h should be an injective function on the nodes of G
        '''
        # First verify that h is injective
        if len({h(x) for x in G.nodes()}) != len(G.nodes()):
            raise ValueError("h is not injective on the vertices of G")

        self.graph = G
        self.h = h
        self.directed = self.direct_graph()

    def verify_vertex_set(self, V):
        '''
        raise an error if V is not a subset of the nodes of self.graph
        '''
        if not V.issubset(set(self.graph.nodes())):
            raise ValueError("input must be a subset of the nodes of\
                                self.graph")

    def h_set(self, V):
        '''
        returns max(h(x) for x in V) where V is a list, tuple or set of nodes
        '''
        V = set(V)
        self.verify_vertex_set(V)
        return max([self.h(x) for x in V])

    def downhill_neighbours(self, x):
        '''
        Takes a node x to the set of downhill neighbours of x (e.g
        neighbours y s.t h(y)<h(x)).
        If this set is empty, the function instead returns the set
        containing only x.
        '''
        if {y for y in self.graph.neighbors(x)
                if self.h(y) < self.h(x)}:
            return {y for y in self.graph.neighbors(x)
                    if self.h(y) < self.h(x)}
        else:
            return {x}

    def discrete_gradient_flow(self, V):
        '''
        A function from the power set of NetworkLandscape.graph.nodes() to
        itself which maps a subset of vertices to its immediate neighbors
        with lower h values.
        '''
        self.verify_vertex_set(V)
        return functools.reduce(set.union,
                                [self.downhill_neighbours(x) for x in V])

    def inf_dgf(self, V):
        '''
        Returns a set V' such that V'=discrete_gradient_flow^k(V) for some k
        and V' is a fixed point of discrete_gradient_flow.
        Equivalently, returns the set of local minima whose attraction basins
        intersect with V.
        '''
        self.verify_vertex_set(V)
        while self.discrete_gradient_flow(V) != V:
            V = self.discrete_gradient_flow(V)
        return self.discrete_gradient_flow(V)

    def direct_graph(self):
        '''
        This returns a directed graph on the nodes of G
        an edge (u,v) exists in the new graph if (u,v) is
        an edge in G and h(u) > h(v)
        '''
        G_directed = nx.DiGraph(self.graph)
        for u, v in self.graph.edges:
            if self.h(u) < self.h(v):
                G_directed.remove_edge(u, v)
            else:
                G_directed.remove_edge(v, u)
        return G_directed

    def downhill_path(self, u, v):
        '''
        Returns True if a downhill path from u to v exists, False otherwise.

        Equivalently, it evaluates the truth value of the statement
        "u is reachable from v"
        '''
        if self.h(u) <= self.h(v):
            return False
        else:
            if not dag.is_directed_acyclic_graph(self.directed):
                raise TypeError('NetworkLandscape.directed must be an acyclic\
                 directed graph')
            else:
                return (u in dag.ancestors(self.directed, v))

    def local_minima(self):
        '''
        Returns the set of local minima of the NetworkLandscape.

        The local minima are those vertices whose h value is no larger than the
        values of its neighbors. Note that this set is the maximal fixed point
        of the discrete_gradient_flow
        '''
        return {x for x in self.graph.nodes()
                if self.downhill_neighbours(x) == {x}}

    def attraction_basin(self, x):
        '''
        Returns the attraction basin of a local minimum x.
        If x is not a local minimum
        '''
        return {y for y in self.graph.nodes()
                if self.inf_dgf({y}) == {x}}

    def separatrix(self):
        '''
        Returns the set of nodes from which multiple local minima are reachable
        '''
        return {y for y in self.graph.nodes()
                if len(self.inf_dgf({y})) > 1}

    def sub_landscape(self, V):
        '''
        Returns a new NetworkLandscape with h inherited and graph restricted to
        vertex set V.
        '''
        self.verify_vertex_set(V)
        return NetworkLandscape(self.graph.subgraph(V), self.h)

    def deformable(self, path1, path2):
        '''
        Returns true if path1 is deformable to path2
        path1 and path2 are lists of vertices representing paths in self.graph
        '''
        def F(x):
            return {y for y in path2 if self.downhill_path(y, x)}

        return (all([len(F(x)) > 0 for x in path1])
                and functools.reduce(set.union, [F(x) for x in path1])
                == set(path2))
    # index 1 critical nodes are maximal nodes on paths between index 0
    # critical points (local minima) that are not deformable (minimum energy)
    # paths.

    def nought_simplices(self):
        '''
        Returns a list of the nought simplices of the NetworkLandscape
        '''
        return [(node,)
                for node in self.graph.nodes()]

    def one_simplices(self):

        return [(x, y)
                for x in self.graph.nodes()
                for y in self.graph.nodes()
                if self.downhill_path(y, x)]

    def two_simplices(self):
        '''
        Returns a list of the two-simplices of the NetworkLandscape
        '''
        return [(x, y, z)
                for x in self.graph.nodes()
                for y in self.graph.nodes()
                for z in self.graph.nodes()
                if (self.downhill_path(y, x)
                and self.downhill_path(z, y))]

    def get_simplices(self):
        '''
        return a list of dionysus.Simplex objects containing the simplices of
        the NetworkLandscape
        '''
        simplices = [*self.two_simplices(),
                     *self.one_simplices(),
                     *self.nought_simplices()]
        return simplices


def detect_critical_nodes(nl, k):
    '''
    '''
    if k == 0:
        critical_nodes = []
        simplices = nl.get_simplices()
        F = mogutda.SimplicialComplex()
        nodes = list(nl.graph.nodes())
        for i, node in enumerate(nodes[0:-1]):
            Fplus = mogutda.SimplicialComplex(
                    [s for s in simplices if nl.h_set(s) <= nl.h(nodes[i+1])]
                    )
            if Fplus.betti_number(0) > F.betti_number(0):
                critical_nodes.append(node)
            F = copy.copy(Fplus)

    elif k >= 1:
        critical_nodes = []
        f = recompose(lambda landscape:
                      landscape.sub_landscape(landscape.separatrix()), k-1)
        nlk = f(nl)
        simplices = nlk.get_simplices()
        F = mogutda.SimplicialComplex()
        nodes = list(nl.graph.nodes())
        for i, node in enumerate(nodes[0:-1]):
            Fplus = mogutda.SimplicialComplex(
                    [s for s in simplices if nlk.h_set(s) <= nlk.h(nodes[i+1])]
                    )
            if (Fplus.betti_number(0) < F.betti_number(0)
               or Fplus.betti_number(1) > F.betti_number(1)):
                critical_nodes.append(node)
            F = copy.copy(Fplus)
    return critical_nodes
