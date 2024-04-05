

import networkx as nx
import scipy as sp
import numpy as np
import pinocchio

deg2rad = np.pi/180.
rad2deg = 180./np.pi


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

Rot_y = lambda a: np.array([[np.cos(a), 0, np.sin(a)],
                  [0, 1, 0],
                 [-np.sin(a), 0, np.cos(a)]])

Rot_x = lambda a: np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                 [0, np.sin(a), np.cos(a)]])

def rot_axis(axis):
    if 'x' == axis:
        return Rot_x
    if 'y' == axis:
        return Rot_y
    if 'z' == axis:
        return Rot_z
    if '-x' == axis:
        return lambda x: Rot_x(-x)
    if '-y' == axis:
        return lambda x: Rot_y(-x)
    if '-z' == axis:
        return lambda x: Rot_z(-x)

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

def compute_rotation_axis_from_model(model,data):
    axis = []
    q0 = pinocchio.neutral(model)
    for i in range(q0.shape[0]):
        pinocchio.forwardKinematics(model,data,pinocchio.neutral(model))
        R_ref_0 = data.oMi[i+1].rotation

        q = np.zeros((q0.shape[0]))
        q[i] = np.pi/4
        pinocchio.forwardKinematics(model,data,q)
        R_ref = data.oMi[i+1].rotation
        
        if (R_ref_0.T@R_ref)[1,2] < 0:
            axis.append('x')
        elif (R_ref_0.T@R_ref)[1,2] > 0:
            axis.append('-x')
        elif(R_ref_0.T@R_ref)[0,2] > 0:
            axis.append('y')
        elif (R_ref_0.T@R_ref)[0,2] < 0:
            axis.append('-y')
        elif (R_ref_0.T@R_ref)[0,1] < 0:
            axis.append('z')
        elif (R_ref_0.T@R_ref)[0,1] > 0:
            axis.append('z')
        else:
            print(2)
    return axis

def trans_axis(axis,axis_length):
    if axis=='x':
        return pinocchio.SE3(np.eye(3),np.array([axis_length,0.,0]))
    elif axis=='y':
        return pinocchio.SE3(np.eye(3),np.array([0,axis_length,0.]))
    elif axis=='z':
        return pinocchio.SE3(np.eye(3),np.array([0,0.,axis_length]))
    elif axis=='-x':
        return pinocchio.SE3(np.eye(3),np.array([-axis_length,0.,0]))
    elif axis=='-y':
        return pinocchio.SE3(np.eye(3),np.array([0,-axis_length,0.]))
    elif axis=='-z':
        return pinocchio.SE3(np.eye(3),np.array([0,0.,-axis_length]))


def norm(x):
    return x/max(np.linalg.norm(x),10**-9)

def direction_to_position(Y,C,D,data,axis_length):
    X = np.linalg.pinv(C@C.transpose())@C@D@Y.transpose()
    
    X = X - X[0,:] #+ data.oMi[1].translation
    R = axis_length*np.linalg.inv(X[1:4,:3])@np.diag((1,-1,1))
    #print(X[:4,:]@np.linalg.inv(X[1:4,:3]))
    #print(X[:4,:])
    X = X@R + data.oMi[1].translation
    #print(X@R.T)
    R_ = np.diag((1,-1,1))@np.linalg.inv(Y[:3,:3])
    return X, R_@Y

def skew(x):
    """
    Creates a skew symmetric matrix from vector x
    """
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return X

def cost_(Q1 : np.ndarray, Q2: np.ndarray,C_joints_limit:np.ndarray,Q1_: np.ndarray, Q2_: np.ndarray, *Y):
    u = [y for y in Y[:2]]
    u = np.concatenate(u,axis=1)
    #u = Y[0]
    Y = np.concatenate(Y,axis=1)
    #try:
        #print(np.trace(C_joints_limit.transpose()@C_joints_limit@Y.transpose()@Y),np.trace(Q1_ + Q2_@u.transpose()@u))
    #except:
        #print('')
        #r = 0
    return np.trace(Q1 + Q2@Y.transpose()@Y)

def euclidean_gradient_(Q2: np.ndarray,*Y,ind = None):
    Y = np.concatenate(Y,axis=1)
    grad = 2*Q2@Y.transpose()
    
    return [grad.transpose()[:,start:end] for start, end in zip(ind[:-1],ind[1:])]
                            
def euclidean_hessian_(Q2: np.ndarray,*U,ind=None,n=None):

    #Y = np.concatenate(U[:n],axis=1)
    U = np.concatenate(U[n:],axis=1)

    hess = (2*Q2 )@U.transpose()

    return [hess.transpose()[:,start:end] for start, end in zip(ind[:-1],ind[1:])]

def dimension_initialisation(Y_init, d, constraints, n_anchor, n_below, C_joints_limit, n_D, Anchor, max_constr):

    if n_anchor > 0:
        if Y_init[1].shape[0] == d:
            Y_init_ = Y_init[:2]
        elif Y_init[1].shape[0] < d:
            Y_init_ = np.zeros((d,Y_init[1].shape[1]))
            Y_init_[:Y_init[1].shape[0],:] = Y_init[1]
            Y_init_ = [Anchor, Y_init_]
        else:
            U,S,V = np.linalg.svd(Y_init[1],full_matrices=False)
            Y_init_ = [Anchor, U[:d,:d]@np.diag(S)[:d,:]@V/np.maximum(np.linalg.norm(U[:d,:d]@np.diag(S)[:d,:]@V,axis=0),10**-9)]
            
        if constraints and not max_constr:
            #print(Y_init_[0].shape,Y_init_[1].shape,d)
            Y_init_.append((C_joints_limit[:n_below,:n_D]@np.concatenate(Y_init_,axis=1).transpose()).transpose())

            if C_joints_limit.shape[0] != n_below:
                Y_init_.append((C_joints_limit[n_below:,:n_D]@np.concatenate(Y_init_[:2],axis=1).transpose()).transpose())
    else:
        #print([y.shape for y in Y_init])
        if Y_init[0].shape[0] == d:
            Y_init_ = Y_init   
        elif Y_init[0].shape[0] < d:
            Y_init_ = np.zeros((d,Y_init[0].shape[1]))
            Y_init_[:Y_init[0].shape[0],:] = Y_init[1]
            Y_init_ = [Y_init_]
        else:
            U,S,V = np.linalg.svd(Y_init,full_matrices=False) 
            Y_init_ = [U[:d,:d]@np.diag(S)[:d,:]@V/np.maximum(np.linalg.norm(U[:d,:d]@np.diag(S)[:d,:]@V,axis=0),10**-9)]

        if constraints and not max_constr:
            Y_init_.append((C_joints_limit[:n_below,:n_D]@Y_init_[0].transpose()).transpose())

            if C_joints_limit.shape[0] != n_below:
                Y_init_.append((C_joints_limit[n_below:,:n_D]@Y_init_[0].transpose()).transpose())
    
    return Y_init_


def cost_max(Q1 : np.ndarray, Q2: np.ndarray,Q_below: np.ndarray, Q_above: np.ndarray,
             D_below: np.ndarray, D_above:np.ndarray, *Y,lambda_cons=None):
        
        Y = np.concatenate(Y,axis=1)
        
        constr_below = - (np.linalg.norm(Q_below@Y.transpose(),axis=1)**2 - D_below)
        constr_above = (np.linalg.norm(Q_above@Y.transpose(),axis=1)**2 - D_above)

        constr = (np.sum(np.maximum(constr_below, 0)) + np.sum(np.maximum(constr_above, 0))) 
        #print(constr_below)
        #print(np.linalg.norm(Q_below@Y.transpose(),axis=1)**2)
        #print(D_below)
        #print(constr_below)
        #print(np.sum(np.maximum(constr_below, 0)),np.sum(np.maximum(constr_above, 0)),np.trace(Q1 + Q2@Y.transpose()@Y))
        return np.trace(Q1 + Q2@Y.transpose()@Y) + lambda_cons*constr

def euclidean_gradient_max(Q2: np.ndarray,zip_below, zip_above, *Y,ind=None,lambda_cons=None):
        
        Y = np.concatenate(Y,axis=1)
        grad = 2*Q2@Y.transpose()

        #grad_below = - sum([4*max((d-np.linalg.norm(q[:,np.newaxis].transpose()@Y.transpose(),'fro')**2,0))*qq@Y.transpose() for d,q,qq in zip_below])
        grad_below = - sum([2*((d-np.linalg.norm(q[:,np.newaxis].transpose()@Y.transpose(),'fro')**2>0))*qq@Y.transpose() for d,q,qq in zip_below])
       
        grad_above=  sum([2*((-d+np.linalg.norm(q[:,np.newaxis].transpose()@Y.transpose(),'fro')**2>0))*qq@Y.transpose() for d,q,qq in zip_above])
        
        #print('grad',np.linalg.norm(grad_above))
        grad = grad + lambda_cons*(grad_below + grad_above)
        return [grad.transpose()[:,start:end] for start, end in zip(ind[:-1],ind[1:])]

def euclidean_hessian_max(Q2: np.ndarray,zip_below, zip_above, *U,ind=None,n = None,lambda_cons=None):
        
        Y = np.concatenate(U[:n],axis=1)
        U = np.concatenate(U[n:],axis=1)

        hess_below = sum([2*((d-np.linalg.norm(q[:,np.newaxis].transpose()@Y.transpose(),'fro')**2>0))*qq@U.transpose() for d,q,qq in zip_below])
        hess_above = sum([2*((-d+np.linalg.norm(q[:,np.newaxis].transpose()@Y.transpose(),'fro')**2>0))*qq@U.transpose() for d,q,qq in zip_above])
        
        hess =  (hess_above + hess_below)
        
        hess = (2*Q2 )@U.transpose() + lambda_cons*hess

        return [hess.transpose()[:,start:end] for start, end in zip(ind[:-1],ind[1:])]