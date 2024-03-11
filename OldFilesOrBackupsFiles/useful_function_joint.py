import numpy as np
import networkx as nx


# +

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
    import scipy as sp

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



# -

R = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
e1 = np.array([1,0,0])

# +
Rot_x = lambda a4: np.array([[1, 0, 0],
                  [0, np.cos(a4), np.sin(a4)],
                 [0, -np.sin(a4), np.cos(a4)]])

T = lambda x: np.array([[1, 0, 0,x[0]],
                  [0, 1, 0,x[1]],
                 [0, 0, 1,x[2]],
                [0,0,0,1]])
# -

Rot_z = lambda a: np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a), np.cos(a), 0],
                 [0, 0, 1]])

dRot_x = lambda a4: np.array([[0, 0, 0],
                  [0, -np.sin(a4), np.cos(a4)],
                 [0, -np.cos(a4), -np.sin(a4)]])

Rz = lambda a1: np.array([[np.cos(a1), np.sin(a1), 0,0],
                  [-np.sin(a1), np.cos(a1), 0,0],
                 [0, 0, 1,0],
                 [0,0,0,1]])

# +
Ry1 = lambda a2: np.array([[np.cos(a2), 0, np.sin(a2),25],
                  [0, 1, 0,0],
                 [-np.sin(a2), 0, np.cos(a2),0],
                 [0,0,0,1]])

Ry = lambda a2: np.array([[np.cos(a2), 0, np.sin(a2),0],
                  [0, 1, 0,0],
                 [-np.sin(a2), 0, np.cos(a2),0],
                 [0,0,0,1]])

Rx = lambda a4: np.array([[1, 0, 0,0],
                  [0, np.cos(a4), np.sin(a4),0],
                 [0, -np.sin(a4), np.cos(a4),0],
                 [0,0,0,1]])
# -

Ry1_cam = lambda a2: np.array([[np.cos(a2), 0, np.sin(a2),25],
                  [0, 1, 0,0],
                 [-np.sin(a2), 0, np.cos(a2),0],
                 [0,0,0,1]])

Ry2 = lambda a3: np.array([[np.cos(a3), 0, np.sin(a3),560],
                  [0, 1, 0,0],
                 [-np.sin(a3), 0, np.cos(a3),0],
                 [0,0,0,1]])

Rx1 = lambda a4: np.array([[1, 0, 0,515],
                  [0, np.cos(a4), np.sin(a4),0],
                 [0, -np.sin(a4), np.cos(a4),25],
                 [0,0,0,1]])

Ry3 = lambda a5: np.array([[np.cos(a5), 0, np.sin(a5),0],
                  [0, 1, 0,0],
                 [-np.sin(a5), 0, np.cos(a5),0],
                 [0,0,0,1]])

Rx2 = lambda a6: np.array([[1, 0, 0,0],
                  [0, np.cos(a6), np.sin(a6),0],
                 [0, -np.sin(a6), np.cos(a6),0],
                 [0,0,0,1]])

Rx2_cam = lambda a6: np.array([[1, 0, 0,0],
                  [0, np.cos(a6), np.sin(a6),0],
                 [0, -np.sin(a6), np.cos(a6),0],
                 [0,0,0,1]])

Rx2_ = np.diag((1,1,1,1.))
Rx2_[0,3] = 90 + 220


transform_end_effector = np.diag([1,1,1.,1])
transform_end_effector[:3,3] = np.array([90.,0.,0])



############## Seuls paramètres à changer/toucher #################
transform_end_effector_cam = np.diag([1,1,1.,1])
transform_end_effector_cam[:3,3] = np.array([0,0.,0])

l_tige = 220 
transform_end_effector_vespa_front = np.diag([1,1,1,1.])
transform_end_effector_vespa_front[:3,3] = np.array([l_tige,0.,0])

transform_end_effector_vespa_back = np.diag([1,1.,1,1])
transform_end_effector_vespa_back[:3,3] = np.array([l_tige,0.,0])
######################################################################

def R_end_effector(config):
    if config == 'camera':
        return transform_end_effector@transform_end_effector_cam
    elif config == 'vespa_front':
        return transform_end_effector@transform_end_effector_vespa_front
    elif config == 'vespa_back':
        return transform_end_effector@transform_end_effector_vespa_back
    

def pose_axes(v):
    t1 = np.array((25,0,0))
    t2 = np.array((560,0,0))
    t3 = np.array((515,0,25))
    t4 = np.array((0,0,0))
    t5 = np.array((90,0,0))

    R1 = Rz
    R2 = Ry
    R3 = Ry
    R4 = Rx
    R5 = Ry
    R6 = Rx

    T1 = T(t1)
    T2 = T(t2)
    T3 = T(t3)
    T4 = T(t4)
    T5 = T(t5)

    #transformation = [np.eye(4),Rz(v[0]),Ry1(v[1]),Ry2(v[2]),Rx1(v[3]),Ry3(v[4]),Rx2(v[5]),transform_end_effector]
    transformation = [np.eye(4),R1(v[0]),T1@R2(v[1]),T2@R3(v[2]),T3@R4(v[3]),T4@R5(v[4]),T5@R6(v[5])]

    pose_ = [np.linalg.multi_dot(transformation[:i+2]) for i in range(6)]

    return pose_


pose_robot_ = lambda v, config: Rz(v[0])@Ry1(v[1])@Ry2(v[2])@Rx1(v[3])@Ry3(v[4])@Rx2(v[5])@R_end_effector(config)
pose_robot_cam_ = lambda v, config: Rz(v[0])@Ry1_cam(v[1])@Ry2(v[2])@Rx1(v[3])@Ry3(v[4])@Rx2_cam(v[5])@R_end_effector(config)

pose_robot_a1_a5 = lambda v: Rz(v[0])@Ry1(v[1])@Ry2(v[2])@Rx1(v[3])@Ry3(v[4])

pose_wrist = lambda v: Rz(v[0])@Ry1(v[1])@Ry2(v[2])@Rx1(v[3])

dRz = lambda a1: np.array([[-np.sin(a1), np.cos(a1), 0,0],
                  [-np.cos(a1), -np.sin(a1), 0,0],
                 [0, 0, 0,0],
                 [0,0,0,0]])

dRy1 = lambda a2: np.array([[-np.sin(a2), 0, np.cos(a2),0],
                  [0, 0, 0,0],
                 [-np.cos(a2), 0, -np.sin(a2),0],
                 [0,0,0,0]])

dRy1_cam = lambda a2: np.array([[-np.sin(a2), 0, np.cos(a2),0],
                  [0, 0, 0,0],
                 [-np.cos(a2), 0, -np.sin(a2),0],
                 [0,0,0,0]])

dRy2 = lambda a3: np.array([[-np.sin(a3), 0, np.cos(a3),0],
                  [0, 0, 0,0],
                 [-np.cos(a3), 0, -np.sin(a3),0],
                 [0,0,0,0]])

dRx1 = lambda a4: np.array([[0, 0, 0,0],
                  [0, -np.sin(a4), np.cos(a4),0],
                 [0, -np.cos(a4), -np.sin(a4),0],
                 [0,0,0,0]])

dRy3 = lambda a5: np.array([[-np.sin(a5), 0, np.cos(a5),0],
                  [0, 0, 0,0],
                 [-np.cos(a5), 0, -np.sin(a5),0],
                 [0,0,0,0]])

dRx2 = lambda a6: np.array([[0, 0, 0,0],
                  [0, -np.sin(a6), np.cos(a6),0],
                 [0, -np.cos(a6), -np.sin(a6),0],
                 [0,0,0,0]])

dRx2_cam = lambda a6: np.array([[0, 0, 0,0],
                  [0, -np.sin(a6), np.cos(a6),0],
                 [0, -np.cos(a6), -np.sin(a6),0],
                 [0,0,0,0]])


def d_pose_axes(v):
    t1 = np.array((25,0,0))
    t2 = np.array((560,0,0))
    t3 = np.array((515,0,25))
    t4 = np.array((0,0,0))
    t5 = np.array((90,0,0))

    R1 = Rz
    R2 = Ry
    R3 = Ry
    R4 = Rx
    R5 = Ry
    R6 = Rx

    dR1 = dRz
    dR2 = dRy1
    dR3 = dRy1
    dR4 = dRx1
    dR5 = dRy1
    dR6 = dRx1

    T1 = T(t1)
    T2 = T(t2)
    T3 = T(t3)
    T4 = T(t4)
    T5 = T(t5)

    #transformation = [np.eye(4),Rz(v[0]),Ry1(v[1]),Ry2(v[2]),Rx1(v[3]),Ry3(v[4]),Rx2(v[5]),transform_end_effector]
    transformation = [np.eye(4),R1(v[0]),T1@R2(v[1]),T2@R3(v[2]),T3@R4(v[3]),T4@R5(v[4]),T5@R6(v[5])]
    d_transformation = [dR1(v[0]),T1@dR2(v[1]),T2@dR3(v[2]),T3@dR4(v[3]),T4@dR5(v[4]),T5@dR6(v[5])]
    pose_ = [R1(v[0]),T1@R2(v[1]),T2@R3(v[2]),T3@R4(v[3]),T4@R5(v[4]),T5@R6(v[5])]

    return pose_


dd_My = np.diag((-1,0,-1,0))
dd_Mx = np.diag((0,-1,-1,0))
dd_Mz = np.diag((-1,-1,0.,0))
M = np.zeros((4,4))
M[:3,:3] = np.ones((3,3))
M[3,3] = 1

def d_pose_robot_(v,config):
    Rz_ = Rz(v[0])
    Ry1_ = Ry1(v[1])
    Ry2_ = Ry2(v[2])
    Rx1_ = Rx1(v[3])
    Ry3_ = Ry3(v[4])
    Rx2_ = Rx2(v[5])
    
    end_effector = R_end_effector(config)
    
    start = Rz_@Ry1_
    end = Ry3_@Rx2_@end_effector
    
    return np.array([dRz(v[0])@Ry1_@Ry2_@Rx1_@end,
                                Rz_@dRy1(v[1])@Ry2_@Rx1_@end,
                                start@dRy2(v[2])@Rx1_@end,
                                start@Ry2_@dRx1(v[3])@end,
                                start@Ry2_@Rx1_@dRy3(v[4])@Rx2_@end_effector,
                                start@Ry2_@Rx1_@Ry3_@dRx2(v[5])@end_effector])

def d_pose_robot_a1_a5(v):
    Rz_ = Rz(v[0])
    Ry1_ = Ry1(v[1])
    Ry2_ = Ry2(v[2])
    Rx1_ = Rx1(v[3])
    Ry3_ = Ry3(v[4])
    
    start = Rz_@Ry1_
    end = Ry3_
    
    return np.array([dRz(v[0])@Ry1_@Ry2_@Rx1_@end,
                                Rz_@dRy1(v[1])@Ry2_@Rx1_@end,
                                start@dRy2(v[2])@Rx1_@end,
                                start@Ry2_@dRx1(v[3])@end,
                                start@Ry2_@Rx1_@dRy3(v[4])])

def d_pose_robot_A4_to_A6(v):
    Ry3_ = Ry3(v[4])
    start = Rz(v[0])@Ry1_cam(v[1])@Ry2(v[2])
    
    return np.array([start@dRx1(v[3])@Ry3_@Rx2_cam(v[5]),start@Rx1(v[3])@dRy3(v[4])@Rx2(v[5]),start@Rx1(v[3])@Ry3_@dRx2(v[5])])

def d_pose_robot_cam_A4_to_A6(v):
    Ry3_ = Ry3(v[4])
    start = Rz(v[0])@Ry1(v[1])@Ry2(v[2])
    
    return np.array([start@dRx1(v[3])@Ry3_@Rx2(v[5]),start@Rx1(v[3])@dRy3(v[4])@Rx2(v[5]),start@Rx1(v[3])@Ry3_@dRx2_cam(v[5])])

def d_pose_robot_cam(v):
    Rz_ = Rz(v[0])
    Ry1_ = Ry1_cam(v[1])
    Ry2_ = Ry2(v[2])
    Rx1_ = Rx1(v[3])
    Ry3_ = Ry3(v[4])
    Rx2_ = Rx2_cam(v[5])
    
    start = Rz_@Ry1_
    end = Ry3_@Rx2_
    
    return np.array([M*dRz(v[0])@Ry1_@Ry2_@Rx1_@end,
                                M*(Rz_@dRy1_cam(v[1]))@Ry2_@Rx1_@end,
                                M*(start@dRy2(v[2]))@Rx1_@end,
                                M*(start@Ry2_@dRx1(v[3]))@end,
                                M*(start@Ry2_@Rx1_@dRy3(v[4]))@Rx2_,
                                M*(start@Ry2_@Rx1_@Ry3_@dRx2_cam(v[5]))])

def dd_pose_robot_(v,config):
    Rz_ = Rz(v[0])
    Ry1_ = Ry1(v[1])
    Ry2_ = Ry2(v[2])
    Rx1_ = Rx1(v[3])
    Ry3_ = Ry3(v[4])
    Rx2_ = Rx2(v[5])
    
    dRz_ = dRz(v[0])
    dRy1_ = dRy1(v[1])
    dRy2_ = dRy2(v[2])
    dRx1_ = dRx1(v[3])
    dRy3_ = dRy3(v[4])
    dRx2_ = dRx2(v[5])
    
    end_effector = R_end_effector(config)
    
    start = Rz_@Ry1_
    end = Ry3_@Rx2_@end_effector
    
    return (np.array([M*dRz(v[0])@Ry1_@Ry2_@Rx1_@end,
                                M*(Rz_@dRy1(v[1]))@Ry2_@Rx1_@end,
                                M*(start@dRy2(v[2]))@Rx1_@end,
                                M*(start@Ry2_@dRx1(v[3]))@end,
                                M*(start@Ry2_@Rx1_@dRy3(v[4]))@Rx2_@end_effector,
                                M*(start@Ry2_@Rx1_@Ry3_@dRx2(v[5]))@end_effector]),
           np.array([[M*(dd_Mz)@Rz_@Ry1_@Ry2_@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(dRz_@dRy1_)@Ry2_@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(dRz_@Ry1_@dRy2_)@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(dRz_@Ry1_@Ry2_@dRx1_)@Ry3_@Rx2_@end_effector,
                                M*(dRz_@Ry1_@Ry2_@Rx1_@dRy3_)@Rx2_@end_effector,
                                M*(dRz_@Ry1_@Ry2_@Rx1_@Ry3_@dRx2_)@end_effector],
                    [M*(dRz_@dRy1_)@Ry2_@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@dd_My@Ry1_)@Ry2_@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@dRy1_@dRy2_)@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@dRy1_@Ry2_@dRx1_)@Ry3_@Rx2_@end_effector,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@Ry3_@dRx2_)@end_effector],
                    [M*(dRz_@Ry1_@dRy2_)@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@dRy1_@dRy2_)@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@Ry1_@dd_My@Ry2_)@Rx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@Ry1_@dRy2_@dRx1_)@Ry3_@Rx2_@end_effector,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@Ry3_@dRx2_)@end_effector],
                    [M*(dRz_@Ry1_@Ry2_@dRx1_)@Ry3_@Rx2_@end_effector,
                                M*(Rz_@dRy1_)@Ry2_@dRx1_@Ry3_@Rx2_@end_effector,
                                M*(Rz_@Ry1_@dRy2_@dRx1_)@Ry3_@Rx2_@end_effector,
                                M*(Rz_@Ry1_@Ry2_@dd_Mx@Rx1_)@Ry3_@Rx2_@end_effector,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@Ry3_@dRx2_)@end_effector],
                    [M*(dRz_@Ry1_@Ry2_@Rx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@dRy3_)@Rx2_@end_effector,
                                M*(Rz_@Ry1_@Ry2_@Rx1_@dd_My@Ry3_)@Rx2_@end_effector,
                                M*(Rz_@Ry1_@Ry2_@Rx1_@dRy3_@dRx2_)@end_effector],
                    [M*(dRz_@Ry1_@Ry2_@Rx1_@Ry3_@dRx2_)@end_effector,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@Ry3_@dRx2_)@end_effector,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@Ry3_@dRx2_)@end_effector,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@Ry3_@dRx2_)@end_effector,
                                M*(Rz_@Ry1_@Ry2_@Rx1_@dRy3_@dRx2_)@end_effector,
                                M*(Rz_@Ry1_@Ry2_@Rx1_@Ry3_@dd_Mx@Rx2_)@end_effector]]))



def dd_pose_robot_cam(v):
    Rz_ = Rz(v[0])
    Ry1_ = Ry1_cam(v[1])
    Ry2_ = Ry2(v[2])
    Rx1_ = Rx1(v[3])
    Ry3_ = Ry3(v[4])
    Rx2_ = Rx2_cam(v[5])
    
    dRz_ = dRz(v[0])
    dRy1_ = dRy1_cam(v[1])
    dRy2_ = dRy2(v[2])
    dRx1_ = dRx1(v[3])
    dRy3_ = dRy3(v[4])
    dRx2_ = dRx2_cam(v[5])
    
    start = Rz_@Ry1_
    end = Ry3_@Rx2_
    
    
    return (np.array([M*dRz(v[0])@Ry1_@Ry2_@Rx1_@end,
                                M*(Rz_@dRy1(v[1]))@Ry2_@Rx1_@end,
                                M*(start@dRy2(v[2]))@Rx1_@end,
                                M*(start@Ry2_@dRx1(v[3]))@end,
                                M*(start@Ry2_@Rx1_@dRy3(v[4]))@Rx2_,
                                M*(start@Ry2_@Rx1_@Ry3_@dRx2(v[5]))]),
           np.array([[M*(dd_Mz)@Rz_@Ry1_@Ry2_@Rx1_@Ry3_@Rx2_,
                                M*(dRz_@dRy1_)@Ry2_@Rx1_@Ry3_@Rx2_,
                                M*(dRz_@Ry1_@dRy2_)@Rx1_@Ry3_@Rx2_,
                                M*(dRz_@Ry1_@Ry2_@dRx1_)@Ry3_@Rx2_,
                                M*(dRz_@Ry1_@Ry2_@Rx1_@dRy3_)@Rx2_,
                                M*(dRz_@Ry1_@Ry2_@Rx1_@Ry3_@dRx2_)],
                    [M*(dRz_@dRy1_)@Ry2_@Rx1_@Ry3_@Rx2_,
                                M*(Rz_@dd_My@Ry1_)@Ry2_@Rx1_@Ry3_@Rx2_,
                                M*(Rz_@dRy1_@dRy2_)@Rx1_@Ry3_@Rx2_,
                                M*(Rz_@dRy1_@Ry2_@dRx1_)@Ry3_@Rx2_,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@dRy3_)@Rx2_,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@Ry3_@dRx2_)],
                    [M*(dRz_@Ry1_@dRy2_)@Rx1_@Ry3_@Rx2_,
                                M*(Rz_@dRy1_@dRy2_)@Rx1_@Ry3_@Rx2_,
                                M*(Rz_@Ry1_@dd_My@Ry2_)@Rx1_@Ry3_@Rx2_,
                                M*(Rz_@Ry1_@dRy2_@dRx1_)@Ry3_@Rx2_,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@dRy3_)@Rx2_,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@Ry3_@dRx2_)],
                    [M*(dRz_@Ry1_@Ry2_@dRx1_)@Ry3_@Rx2_,
                                M*(Rz_@dRy1_)@Ry2_@dRx1_@Ry3_@Rx2_,
                                M*(Rz_@Ry1_@dRy2_@dRx1_)@Ry3_@Rx2_,
                                M*(Rz_@Ry1_@Ry2_@dd_Mx@Rx1_)@Ry3_@Rx2_,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@dRy3_)@Rx2_,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@Ry3_@dRx2_)],
                    [M*(dRz_@Ry1_@Ry2_@Rx1_@dRy3_)@Rx2_,
                                M*(Rz_@dRy1_@Ry2_@Rx1_@dRy3_)@Rx2_,
                                M*(Rz_@Ry1_@dRy2_@Rx1_@dRy3_)@Rx2_,
                                M*(Rz_@Ry1_@Ry2_@dRx1_@dRy3_)@Rx2_,
                                M*(Rz_@Ry1_@Ry2_@Rx1_@dd_My@Ry3_)@Rx2_,
                                M*(Rz_@Ry1_@Ry2_@Rx1_@dRy3_@dRx2_)],
                    [M*(dRz_@Ry1_@Ry2_@Rx1_@Ry3_@dRx2_),
                                M*(Rz_@dRy1_@Ry2_@Rx1_@Ry3_@dRx2_),
                                M*(Rz_@Ry1_@dRy2_@Rx1_@Ry3_@dRx2_),
                                M*(Rz_@Ry1_@Ry2_@dRx1_@Ry3_@dRx2_),
                                M*(Rz_@Ry1_@Ry2_@Rx1_@dRy3_@dRx2_),
                                M*(Rz_@Ry1_@Ry2_@Rx1_@Ry3_@dd_Mx@Rx2_)]]))
