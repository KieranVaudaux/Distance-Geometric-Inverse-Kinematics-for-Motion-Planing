

import networkx as nx
import scipy as sp
import numpy as np


def incidence_matrix_(
    G, nodelist=None, edgelist=None, oriented=False, weight=None, *, dtype=None
):
    """Returns incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the source (tail) of an edge and
    1 for the destination (head) of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    dtype : a NumPy dtype or None (default=None)
        The dtype of the output sparse array. This type should be a compatible
        type of the weight argument, eg. if weight would return a float this
        argument should also be a float.
        If None, then the default for SciPy is used.

    Returns
    -------
    A : SciPy sparse array
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://videolectures.net/mit18085f07_strang_lec03/
    """

    if nodelist is None:
        nodelist = list(G)
    if edgelist is None:
        if G.is_multigraph():
            edgelist = list(G.edges(keys=True))
        else:
            edgelist = list(G.edges())
    A = sp.sparse.lil_matrix((len(nodelist), len(edgelist)), dtype=dtype)
    node_index = {node: i for i, node in enumerate(nodelist)}
    for ei, e in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError as err:
            raise nx.NetworkXError(
                f"node {u} or {v} in edgelist but not in nodelist"
            ) from err
        if weight is None:
            wt = 1
        else:
            if G.is_multigraph():
                ekey = e[2]
                wt = G[u][v][ekey].get(weight, 1)
            else:
                wt = G[u][v].get(weight, 1)
        if oriented:
            A[ui, ei] = -wt
            A[vi, ei] = wt
        else:
            A[ui, ei] = wt
            A[vi, ei] = wt
    return A.asformat("csc")

Rot_z = lambda a: np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a), np.cos(a), 0],
                 [0, 0, 1]])

Rot_y = lambda a: np.array([[np.cos(a), 0, -np.sin(a)],
                  [0, 1, 0],
                 [np.sin(a), 0, np.cos(a)]])

Rot_x = lambda a: np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                 [0, np.sin(a), np.cos(a)]])

def rot_axis(axis):
    if 'x':
        return Rot_x
    if 'y':
        return Rot_y
    if 'z':
        return Rot_z

def wraptopi(e):
    return np.mod(e + np.pi, 2 * np.pi) - np.pi

def max_min_distance_revolute(r, P, C, N):
    delta = P-C
    d_min_s = N.dot(delta)**2 + (np.linalg.norm(np.cross(N, delta)) - r)**2
    if d_min_s > 0:
        d_min = np.sqrt(d_min_s)
    else:
        d_min = 0
    d_max_s = N.dot(delta)**2 + (np.linalg.norm(np.cross(N, delta)) + r)**2
    if d_max_s > 0:
        d_max = np.sqrt(d_max_s)
    else:
        d_max = 0

    return d_max, d_min