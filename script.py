import numpy as np
from scipy import sparse
import random
from itertools import product
from networkx import draw_networkx, spring_layout, from_numpy_array
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def initialise_network(n=0, edges=[]):
    # Returns, from a number of nodes and a list of edges
    # (nodes indexed by numbers 0 to n-1), a network as a
    # sparse matrix.
    i = np.array([a[0] for a in edges] + [a[1] for a in edges])
    j = np.array([a[1] for a in edges] + [a[0] for a in edges])
    V = np.ones(2*len(edges))
    return sparse.csr_matrix((V, (i, j)), shape=(n, n))


def import_zachary():
    # Reads and returns the adjacency matrix of Zachary's Karate
    # Club as defined in zachary.txt
    return np.loadtxt("zachary.txt")


def deg_sequence(M):
    # return the degree sequence of a graph represented by sparse
    # matrix M
    n = M.shape[0]
    return [M.getrow(k).sum() for k in range(n)]
    

def neighbours(k, M):
    # get neighbours of a node in the network represented by
    # sparse matrix M
    return M.getrow(k).nonzero()[1]

  
def downhill_neighbours(k, M, elevation, nodes=None):
    # Find the lower elevation neighbours of k in the network
    # represented by adjacency matrix M,
    # If nodes is passed, the neighbours will be restricted to
    # this set
    
    if not nodes:
        nodes = {x for x in range(0, size)}

    # get neighbours of node k with lower elevations
    Nminus = {x for x in neighbours(k, M)
              if elevation[k] > elevation[x]
              if x in nodes}
    if len(Nminus) == 0:
        return {k}
    else:
        return Nminus


def FastSearchOfNonDegenerateCriticalNodes(M, elevation):
    # return the critical nodes and attraction basins of a network
    # defined by sparse matrix M
    size = M.shape[0]
    # initialise the attraction_basin and critical_nodes dictionaries
    attraction_basin, critical_nodes = {}, {}
    nodeset = {x for x in range(size)}
    # 
    for i in range(size):
        # at each stage in the iteration we want to work over only the
        # nodes that have not already been classified (e.g assigned an
        # attraction basin, e.g added to the attraction_basin dictionary)
        nodeset = sorted(set(nodeset).difference(attraction_basin.keys()),
                         key=lambda k: elevation[k])

        # if there are no nodes left to consider we are done and we
        # return the dictionary of critical nodes, and of attraction
        # basins
        if len(nodeset) == 0:
            return critical_nodes, attraction_basin

        # Otherwise we iterate over the remaining nodes
        for x in nodeset:
            N = downhill_neighbours(x, M, elevation, nodes=nodeset)
            if N == {x}:
                critical_nodes[x] = i
                attraction_basin[x] = x
            else:
                T = {attraction_basin.get(y) for y in N}
                if len(T) == 1 and None not in T:
                    attraction_basin[x] = T.pop()


# ==== The following section deals with generating networks for protein lattices =====


def is_self_avoiding(walk):
    seen = set()
    for i in walk:
        if i in seen:
            return False
        seen.add(i)
    return True


def transform_SAW(walk):
    # Transform a length n string comprised of the letters
    # U, D, L and R into a n+1 lenth list of coordinates
    # starting with (0,0).
    coords = [(0,0)]
    direction = {"U": (1,0), "D": (-1, 0), "L": (0, -1), "R": (0, 1)}
    for i in walk:
        coords.append(tuple(np.add(coords[-1], direction[i])))
    return coords

            
def generate_SAWs(n):
    # Generate a list of self all self avoiding walks of length n beginning at (0, 0)
    walk_list = [transform_SAW(i) for i in product("UDLR", repeat=n-1)]
    return [walk for walk in walk_list if is_self_avoiding(walk)]


def protein_energy(walk, protein_string):
    # return the energy of the protein defined by protein_string in the
    # configuration defined by walk.
    if len(walk) != len(protein_string):
        raise ValueError("both inputs to protein_energy should have the same length")
    # restrict our list of coordinates to those points in the lattice that
    # contain a "H" molecule.
    H_coords = [walk[i] for i in range(len(walk)) if protein_string[i] == "H"]
    energy_value = 0
    for i in range(len(H_coords)):
        for j in range(i+1, len(H_coords)):
            if np.sum(np.absolute(np.subtract(H_coords[i], H_coords[j]))) <= 1:
                energy_value += 1
    return -energy_value + random.uniform(0, 0.5)
    
                

def generate_protein_network(protein_string):
    # returns the network matrix and elevation dictionary of
    # the protein defined by protein_string
    n = len(protein_string)
    # generate all the self avoiding walks of length n
    coord_seqs = generate_SAWs(n)
    # N will be the number of nodes in the network
    N = len(coord_seqs)
    elevation = {i: protein_energy(coord_seqs[i], protein_string) for i in range(N)}
    edges=[]
    for i in range(N):
        for j in range(i+1, N):
            if sum([coord_seqs[i][x] != coord_seqs[j][x] for x in range(n)]) == 1:
                edges.append((i, j))
    network = initialise_network(n=N, edges=edges)
    return network, elevation


def print_summary_statistics(critical_nodes, attraction_basin):
    # prints a series of summary statistics on the number
    # of ith order critical nodes and the mean size of their
    # attraction basins.
    print("size of network: {}".format(len(attraction_basin)))
    basin_size = {}
    for i in critical_nodes.keys():
        basin_size[i] = sum([ i == a for a in attraction_basin.values()])
    print("mean basin size: {}".format(np.mean(list(basin_size.values()))))
    for i in set(critical_nodes.values()):
        print("mean basin size for {}th order nodes: {}".format(i, np.mean([basin_size[v] for v in critical_nodes.keys() if critical_nodes[v]==i])))
        print("number of {}th order nodes: {}".format(i, sum([1 for x in critical_nodes.values() if x==i])))



# ============= output =========================
        
              
if __name__ == "__main__":

    zachary = import_zachary()
    sparse_zachary = sparse.csr_matrix(zachary)
    elevation = {i: -1*sparse_zachary.getrow(i).sum() + random.uniform(0, 0.5)
                 for i in range(zachary.shape[0])}
    critical_nodes, attraction_basin = FastSearchOfNonDegenerateCriticalNodes(sparse_zachary, elevation)
    print(critical_nodes)
    print(attraction_basin)

    # Plotting Zachary's Karate Club
    # fix a layout for this network
    nx_zachary = from_numpy_array(zachary)
    pos = spring_layout(nx_zachary)

    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    # plot the attraction basins
    # starting with a complicated process to get the right colours -_-
    colours = list(set(attraction_basin.values()))
    remap_node_colour = {colours[i]: i for i in range(len(colours))}
    node_color = [remap_node_colour[attraction_basin[x]] for x in range(zachary.shape[0])]
    axes[0].axison=False
    axes[0].set_title("(a)")
    draw_networkx(
        nx_zachary,
        pos=pos,
        ax=axes[0],
        node_color=node_color,
        labels = critical_nodes,
        cmap="Set1")
    axes[1].axison = False
    axes[1].set_title("(b)")
    # plot the network along with ground truth communities
    ground_truth = [1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0]    
    draw_networkx(
        nx_zachary,
        pos=pos,
        ax=axes[1],
        node_color=ground_truth)
    fig.savefig("zachary.png",bbox_inches='tight')

    
    # ================== plot the network of HPPH =========================
    # We start with a very small protein network, HPPH
    protein_network, elevation = generate_protein_network("HPPH")
    C, A = FastSearchOfNonDegenerateCriticalNodes(protein_network, elevation)
    # plot the subgraph on the critical nodes of the protein network
    PF_network = from_numpy_array(protein_network.toarray())
    pos = graphviz_layout(PF_network, prog="neato")
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    # plot the attraction basins
    # starting with a complicated process to get the right colours -_-
    colours = list(set(A.values()))
    remap_node_colour = {colours[i]: i for i in range(len(colours))}
    node_color = [remap_node_colour[A[x]] for x in range(protein_network.shape[0])]
    axes[0].axison=False
    axes[0].set_title("nodal elevation")
    draw_networkx(
        PF_network,
        pos=pos,
        ax=axes[0],
        node_color=[elevation[i] for i in range(protein_network.shape[0])],
        with_labels=False,
        node_size=100)
    axes[1].axison = False
    axes[1].set_title("critical nodes and attraction basins")
    # plot the attraction basins
    draw_networkx(
        PF_network,
        pos=pos,
        ax=axes[1],
        node_color=node_color,
        labels=C,
        node_size=100,
        cmap="tab20")
    fig.savefig("HPPH.png",bbox_inches='tight')
    print_summary_statistics(C, A)

    # ================= The protein network of HPHHPH ======================
    protein_network, elevation = generate_protein_network("HPHHPH")
    C, A = FastSearchOfNonDegenerateCriticalNodes(protein_network, elevation)
    # plot the subgraph on the critical nodes of the protein network
    PF_network = from_numpy_array(protein_network.toarray())
    pos = graphviz_layout(PF_network, prog="neato")
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    # plot the attraction basins
    # starting with a complicated process to get the right colours -_-
    colours = list(set(A.values()))
    remap_node_colour = {colours[i]: i for i in range(len(colours))}
    node_color = [remap_node_colour[A[x]] for x in range(protein_network.shape[0])]
    axes[0].axison=False
    axes[0].set_title("nodal elevation")
    draw_networkx(
        PF_network,
        pos=pos,
        ax=axes[0],
        node_color=[elevation[i] for i in range(protein_network.shape[0])],
        with_labels=False,
        node_size=100)
    axes[1].axison = False
    axes[1].set_title("critical nodes and attraction basins")
    # plot the attraction basins
    draw_networkx(
        PF_network,
        pos=pos,
        ax=axes[1],
        node_color=node_color,
        labels=C,
        node_size=100,
        cmap="tab20")
    fig.savefig("HPHHPH.png",bbox_inches='tight')

    print_summary_statistics(C, A)

    # ======================= plot the network of HPPHPHPH =================
    protein_network, elevation = generate_protein_network("HPPHPHPH")
    C, A = FastSearchOfNonDegenerateCriticalNodes(protein_network, elevation)
    # plot the subgraph on the critical nodes of the protein network
    critical_nodes = C.keys()
    mask = np.array([i in critical_nodes for i in range(protein_network.shape[0])])
    PF = protein_network[mask][:, mask].toarray()
    PF_network = from_numpy_array(PF)
    pos = graphviz_layout(PF_network, prog="neato")
    fig, axes = plt.subplots(1, 1, figsize=(8.27, 11.69))
    draw_networkx(PF_network,
                  pos=pos,
                  node_color=[C[i] for i in sorted(critical_nodes)],
                  with_labels=False,
                  node_size=100)
    axes.axison = False
    fig.savefig("protein_folding.png",bbox_inches='tight')

    print_summary_statistics(C, A)
