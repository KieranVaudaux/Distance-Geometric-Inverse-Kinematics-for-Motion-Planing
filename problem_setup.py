import pinocchio
import numpy as np
import time 
import pymanopt 
from pymanopt import manifolds, optimizers, tools, core
from pymanopt.core.problem  import Problem
from pymanopt.tools.diagnostics import *
from utils import *
import networkx as nx
import matplotlib.pyplot as plt

import tqdm

LOWER = "lower_limit"
UPPER = "upper_limit"
BOUNDED = "bounded"
BELOW = "below"
ABOVE = "above"
TYPE = "type"
OBSTACLE = "obstacle"
ROBOT = "robot"
END_EFFECTOR = "end_effector"
RADIUS = "radius"
DIST = "weight"
POS = "pos"
BASE = "base"
ROOT = None
ANCHOR = "anchor"
SUFFIX = "_tilde"
BASE_GRAPH = "base_graph"
UNDEFINED = None



def create_base_graph(model, data, axis_length, q_init = 'neutral', base_anchor=True, model_axis = None):

    trans_z = [trans_axis(axis,axis_length) for axis in model_axis]
    
    if str(q_init) == 'neutral':
        q_init = pinocchio.neutral(model)
    
    pinocchio.forwardKinematics(model,data,q_init)
    
    robot_name = model.names[1]
    ROOT = robot_name

    pos_ref = {}

    base = nx.empty_graph()

    for idx, (name, oMi) in enumerate(zip(model.names, data.oMi)):
        if idx > 1:
            cur, aux_cur = (name, name+SUFFIX)
            cur_pos, aux_cur_pos = (
                oMi.translation,
                pinocchio.SE3.act(pinocchio.SE3(oMi.rotation,oMi.translation),trans_z[idx-1]).translation,
            )
            
            dist = np.linalg.norm(cur_pos - aux_cur_pos)
            # Add nodes for joint and edge between them
            base.add_nodes_from(
                [(cur, {POS: cur_pos}), (aux_cur, {POS: aux_cur_pos})]
            )
            base.add_edge(
                cur, aux_cur, **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: [], ANCHOR: False}
            )
            
            # If there exists a preceeding joint, connect it to new
            if idx != 0:
                pred, aux_pred = (model.names[idx-1], model.names[idx-1]+SUFFIX)
                for u in [pred, aux_pred]:
                    for v in [cur, aux_cur]:
                        dist = np.linalg.norm(
                            base.nodes[u][POS] - base.nodes[v][POS]
                        )
                        base.add_edge(
                            u,
                            v,
                            **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: [], ANCHOR: False},
                        )
        elif idx == 1:
            #print(name,oMi.translation)
            base = nx.DiGraph(
                [
                    (robot_name, "x"),
                    (robot_name, "y"),
                    (robot_name, robot_name+SUFFIX),
                    ("x", "y"),
                    ("y", robot_name+SUFFIX),
                    (robot_name+SUFFIX ,"x"),
                ]
            )
            base.add_nodes_from(
                [
                    ("x", {POS: np.array([axis_length, 0, 0]) + oMi.translation, TYPE: [BASE]}),
                    ("y", {POS: np.array([0, -axis_length, 0]) + oMi.translation, TYPE: [BASE]}),
                    (robot_name, {POS: oMi.translation, TYPE: [ROBOT, BASE]}),
                    (robot_name+SUFFIX, {POS: trans_z[idx-1].translation + oMi.translation, TYPE: [ROBOT, BASE]}),
                ]
            )

            for u, v in base.edges():
                base[u][v][DIST] = np.linalg.norm(base.nodes[u][POS] - base.nodes[v][POS])
                base[u][v][LOWER] = base[u][v][DIST]
                base[u][v][UPPER] = base[u][v][DIST]
                base[u][v][ANCHOR] = base_anchor
                base[u][v][BOUNDED] = []

    # Set node type to robot
    nx.set_node_attributes(base, [ROBOT], TYPE)
    base.nodes[ROOT][TYPE] = [ROBOT, BASE]
    base.nodes[ROOT + SUFFIX][TYPE] = [ROBOT, BASE]

    #print(nx.get_node_attributes(base,POS))
    
    for u in base.nodes():
        pos_ref[u] = base.nodes[u][POS]

    return base, pos_ref, list(base.nodes())



def goal_graph(model, data, G2, axis_length, position, direction = None, anchor = True,dict_alias = None,pos_ref=None):
    
    end_effector = [dict_alias[model.names[-1]]]
    if direction is not None:
        end_effector.append(model.names[-1]+SUFFIX)

    list_nodes_base = ['x','y',dict_alias[model.names[1]],model.names[1]+SUFFIX]

    for i, cur_end_effector in enumerate(end_effector):
        pos_end = position

        if direction is not None and i == 1:
            pos_end = pos_end + direction*axis_length
        
        G2.nodes[cur_end_effector][POS] = pos_end

        for cur_base in list_nodes_base:
            
            if not((i == 1) and (len(end_effector)==1)):
                dist = np.linalg.norm(pos_end-G2.nodes[cur_base][POS])

                G2.add_edge(
                            cur_base, cur_end_effector, **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: [], ANCHOR: anchor}
                        ) 
    if direction is not None:
        dist = axis_length
        G2.add_edge(
                            dict_alias[model.names[-1]], model.names[-1]+SUFFIX, **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: [], ANCHOR: anchor}
                        )

    list_edges_anchor = []
    list_other_edges = []
    new_edge_order  = []
    A_anchor = []

    for key ,anch_ in nx.get_edge_attributes(G2,ANCHOR).items():

        if anch_:
            list_edges_anchor.append(key)
            A_anchor.append(norm(G2.nodes[key[1]][POS]- G2.nodes[key[0]][POS]))
        else:
            list_other_edges.append(key)
    new_edge_order = list_edges_anchor.copy() 

    for e in list_other_edges:
        new_edge_order.append(e)
    
    if len(A_anchor)>0:
        A_anchor = np.array(A_anchor).transpose()
    else:
        A_anchor = None
    
    n = len(list_other_edges)
    n_anchor = len(list_edges_anchor)

    weight = nx.get_edge_attributes(G2, DIST)
    D = np.diag([weight[i] for i in new_edge_order])

    C = incidence_matrix_(G2,oriented=True, edgelist=new_edge_order).toarray()
    C = np.array(C)

    Y_init = np.zeros((3,n))

    for i, e in enumerate(list_other_edges):
        i = i 
        p1 = pos_ref[e[0]] #G2.nodes[e[0]][POS]
        p2 = pos_ref[e[1]] #G2.nodes[e[1]][POS]
        #print(e,p1,p2)
        Y_init[:,i] = (p2-p1)/max(np.linalg.norm(p2-p1),10**-9)
    if not (A_anchor is None):
        Y_init = [A_anchor,Y_init]

    return G2, list_edges_anchor, list_edges_anchor, new_edge_order, n,n_anchor, A_anchor, D,C, Y_init


def root_angle_limits(G,model,data,axis_length,dict_alias,model_axis):
    upper_limits = np.minimum(-model.lowerPositionLimit,model.upperPositionLimit)
    limited_joints = [] 
    i = 2
    T1 = data.oMi[1]
    base_names = ["x", "y"]
    names = [dict_alias[model.names[i]],model.names[i]+SUFFIX]
    pinocchio.forwardKinematics(model,data,pinocchio.neutral(model))
    #T_axis = pinocchio.SE3(np.eye(3),np.array([0,0,axis_length]))
    T_axis = [trans_axis(axis,axis_length) for axis in model_axis]

    for base_node in base_names:
        for node in names:
            T0 = pinocchio.SE3.Identity()
            T0.translation = G.nodes[base_node][POS]

            if node == dict_alias[model.names[i]]:
                T2 = data.oMi[i]
            else:
                T2 = pinocchio.SE3.act(data.oMi[i],T_axis[i])

            N = T1.rotation[0:3, 2]
            C = T1.translation + (N.dot(T2.translation - T1.translation)) * N
            r = np.linalg.norm(T2.translation - C)
            P = T0.translation
            d_max, d_min = max_min_distance_revolute(r, P, C, N)
            d = np.linalg.norm(T2.translation - T0.translation)

            if d_max == d_min:
                limit = False
            elif d == d_max:
                limit = BELOW
            elif d == d_min:
                limit = ABOVE
            else:
                limit = None

            if limit:
                T_rel = pinocchio.SE3.act(pinocchio.SE3.inverse(T1),data.oMi[i])
                if node != dict_alias[model.names[i]]:
                    T_rel = pinocchio.SE3.act(T_rel,T_axis[i])

                d_limit = np.linalg.norm(
                    pinocchio.SE3.act(pinocchio.SE3.act(T1,pinocchio.SE3(Rot_z(upper_limits[0]),np.zeros((3,)))),T_rel).translation
                    - T0.translation
                )

                q0 = np.zeros((upper_limits.shape[0]))
                q0[0] = upper_limits[0]
                
                pinocchio.framesForwardKinematics(model,data,q0)

                if SUFFIX not in node:
                    d_limit = np.linalg.norm(data.oMi[1].translation-T0.translation)
                if SUFFIX in node:
                    d_limit = np.linalg.norm(pinocchio.SE3.act(data.oMi[1],T_axis[i]).translation-T0.translation)

                if limit == ABOVE:
                    d_max = d_limit
                else:
                    d_min = d_limit
                limited_joints += [dict_alias[model.names[i]]]  # joint at p0 is limited
            
            G.add_edge(base_node, node)
            if d_max == d_min:
                G[base_node][node][DIST] = d_max
            G[base_node][node][BOUNDED] = [limit]
            G[base_node][node][UPPER] = d_max
            G[base_node][node][LOWER] = d_min

            

    return G


def set_limits(G,model,data,axis_length,dict_alias,model_axis):
    """
    Sets known bounds on the distances between joints.
    This is induced by link length and joint limits.
    """
    #T_axis = pinocchio.SE3(np.eye(3),np.array([0,0,axis_length]))
    T_axis = [trans_axis(axis,axis_length) for axis in model_axis]
    upper_limits = np.minimum(-model.lowerPositionLimit,model.upperPositionLimit)#-5*deg2rad

    limited_joints = []  # joint limits that can be enforced

    pinocchio.forwardKinematics(model,data,pinocchio.neutral(model))
    T_zero = {name: oMi for idx, (name, oMi) in enumerate(zip(model.names, data.oMi))}
    
    for idx, (name, oMi) in enumerate(zip(model.names, data.oMi)):
        if idx > 2:
            cur, prev = name, model.names[idx - 2]
            names = [
                (dict_alias[model.names[idx - 2]], dict_alias[name]),
                (dict_alias[model.names[idx - 2]],name+SUFFIX),
                (prev+SUFFIX, dict_alias[name]),
                (prev+SUFFIX, name+SUFFIX),
            ]


            for ids in names:
                q0 = np.zeros((upper_limits.shape[0]))
                q0[idx-2] = upper_limits[idx-2]
                #print(prev,cur,idx)
                pinocchio.framesForwardKinematics(model,data,q0)
                if SUFFIX not in ids[0] and SUFFIX not in ids[1]:
                    TT1 = pinocchio.SE3.Identity()
                    TT2 = pinocchio.SE3.Identity()
                elif SUFFIX in ids[1] and SUFFIX not in ids[0]:
                    TT1 = pinocchio.SE3.Identity()
                    TT2 = T_axis[idx-1]
                elif SUFFIX not in ids[1] and SUFFIX in ids[0]:
                    TT1 = T_axis[idx-3]
                    TT2 = pinocchio.SE3.Identity()
                else:
                    TT1 = T_axis[idx-3]
                    TT2 = T_axis[idx-1]

                q0 = np.zeros((upper_limits.shape[0]))
                q0[idx-2] = upper_limits[idx-2]
                pinocchio.framesForwardKinematics(model,data,q0)
                d_limit = np.linalg.norm(pinocchio.SE3.act(data.oMi[idx-2],TT1).translation - pinocchio.SE3.act(data.oMi[idx],TT2).translation)
                
                list_dist = []
                list_dist_complement = []
                for theta in np.linspace(0,np.pi,283):
                    q0 = np.zeros((upper_limits.shape[0]))
                    q0[idx-2] = theta
                    pinocchio.framesForwardKinematics(model,data,q0)
                    if theta <= upper_limits[idx-2]:
                        list_dist.append(np.linalg.norm(pinocchio.SE3.act(data.oMi[idx-2],TT1).translation - pinocchio.SE3.act(data.oMi[idx],TT2).translation))
                    else:
                        list_dist_complement.append(np.linalg.norm(pinocchio.SE3.act(data.oMi[idx-2],TT1).translation - pinocchio.SE3.act(data.oMi[idx],TT2).translation))
                    
                list_dist = np.array(list_dist)
                diff_dist = list_dist[:-1] - list_dist[1:]
                #print(len(list_dist),len(list_dist_complement))
                #if len(list_dist_complement)>1:
                list_dist_complement = np.array(list_dist_complement)
                diff_dist_complement = list_dist_complement[:-1] - list_dist_complement[1:]
                if len(list_dist)>0:
                    d_max = max(list_dist)
                    d_min = min(list_dist)
                    if np.max(np.abs(diff_dist))<10**-7:
                        limit = False
                        d_max = d_limit
                        d_min = d_limit
                    elif (False not in (diff_dist > 0)) and (False not in (diff_dist_complement > 0)) and (False not in (list_dist > d_limit)) and (False not in (list_dist_complement < d_limit)):
                        limit = BELOW
                    elif (False not in(diff_dist <= 0)) and (False not in(diff_dist_complement < 0)) and (False not in (list_dist < d_limit)) and (False not in (list_dist_complement > d_limit)):
                        limit = ABOVE
                    else:
                        limit=False
                        print('#############################################')


                    limited_joints += [cur]

                    G.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        G[ids[0]][ids[1]][DIST] = d_max
                    G[ids[0]][ids[1]][BOUNDED] = [limit]
                    G[ids[0]][ids[1]][UPPER] = d_max
                    G[ids[0]][ids[1]][LOWER] = d_min
                else:
                    if len(list_dist_complement)<0:
                        print('Two list equal zero')

                if limit:
                    break
    return G, limited_joints


def constraints_graph(model, data, G, D, new_edge_order, axis_length,dict_alias,model_axis,original_nodes):
    D_tilde = {name: [val, ind] for ind, (name, val) in enumerate(zip(new_edge_order,D))}

    list_edges_below = []
    list_edges_above = []

    D_below = []
    D_above = []

    path = {}

    for n1 in original_nodes:
        for n2 in original_nodes:

            path[(n1,n2)] = []
            if n1 != n2 and ((n1,n2) not in new_edge_order):
                #if (n1,n2) == ('x', 'lwa4d_2_joint_tilde'):
                  #     print(nx.is_path(G,['x','lwa4d_1_joint_tilde']))
                    #    print('path',[p in nx.all_simple_edge_paths(G, dict_alias[n1], dict_alias[n2], 5)])
                for p in nx.all_simple_edge_paths(G.to_undirected(), dict_alias[n1], dict_alias[n2], 2):
                    
                    if len(p)>1:
                        path[(n1,n2)].append(p)

    G = root_angle_limits(G,model,data,axis_length,dict_alias,model_axis)
    G, lim_joints = set_limits(G,model,data,axis_length,dict_alias,model_axis)

    for edge, val in nx.get_edge_attributes(G,BOUNDED).items():

        if BELOW in val:
            D_below.append(G[edge[0]][edge[1]][LOWER])
            list_edges_below.append(edge)
        if ABOVE in val:
            D_above.append(G[edge[0]][edge[1]][UPPER])
            list_edges_above.append(edge)

    list_edges_joints_limit = list_edges_below.copy()
    list_edges_joints_limit.extend(list_edges_above)
    
    D_joints_limit = D_below.copy()
    D_joints_limit.extend(D_above)
    
    C_joints_limit = np.zeros((len(list_edges_joints_limit),len(list_edges_joints_limit)+len(new_edge_order)))
    #print(D_tilde)
    for idx, e in enumerate(list_edges_joints_limit):
        
        p = path[e][0]

        if p[0] in new_edge_order:
            p0 = p[0]
        else:
            p0 = (p[0][1],p[0][0])

        if p[1] in new_edge_order:
            p1 = p[1]
        else:
            p1 = (p[1][1],p[1][0])
        
        C_joints_limit[idx,D_tilde[p0][1]] = D_tilde[p0][0]
        C_joints_limit[idx,D_tilde[p1][1]] = D_tilde[p1][0]

        C_joints_limit[idx,len(new_edge_order)+idx] = -1
    
    return G, D_joints_limit, list_edges_joints_limit, C_joints_limit, list_edges_below, list_edges_above

def joint_variables(model,data, Y, new_edge_order, T_final, axis_length,dict_alias,model_axis):   
    
    T = {}
    T[1] = data.oMi[1] 

    pinocchio.forwardKinematics(model,data,pinocchio.neutral(model))
    T_zero = {idx: oMi for idx, (name, oMi) in enumerate(zip(model.names, data.oMi))}
    #print(T_zero)
    trans_z = [trans_axis(axis,axis_length) for axis in model_axis]
    omega_z = [skew(trans_axis(axis,1).translation) for axis in model_axis]
    
    theta = np.zeros((len(model.names)-1,))
    
    for idx, (name, oMi) in enumerate(zip(model.names, data.oMi)):
        if idx > 1:
            cur, aux_cur = (dict_alias[name], dict_alias[name+SUFFIX])
            pred, aux_pred = (dict_alias[model.names[idx-1]], dict_alias[model.names[idx-1]+SUFFIX])

            T_prev = T[idx-1]
            
            T_prev_0 = T_zero[idx-1] # previous p xf at 0
            T_0 = T_zero[idx] # cur p xf at 0
            T_rel = pinocchio.SE3.act(pinocchio.SE3.inverse(T_prev_0),T_0) # relative xf
            T_0_q = pinocchio.SE3.act(T_zero[idx],trans_z[idx-1]) # cur q xf at 0
            ps_0 = pinocchio.SE3.act(pinocchio.SE3.inverse(T_prev_0),T_0_q).translation # relative xf
            ps_0 = norm(ps_0)
            
            ind_ps = new_edge_order.index((pred,aux_cur))
            ps = T[idx-1].rotation.transpose()@Y[:,ind_ps]
            
            theta[idx-2] = np.arctan2(-ps_0.dot(omega_z[idx-2]).dot(ps), ps_0.dot(omega_z[idx-2].dot(omega_z[idx-2].T)).dot(ps))
            rot = rot_axis(model_axis[idx-2])
            rot_axis_z = pinocchio.SE3(rot(theta[idx-2]),np.zeros((3,)))
            T[idx] = pinocchio.SE3.act(pinocchio.SE3.act(T_prev,rot_axis_z),T_rel)

    # if the rotation axis of final joint is aligned with ee frame z axis,
    # get angle from EE pose if available
    #if ((T_final is not None) and (la.norm(cross(T_rel.trans, np.asarray([0, 0, 1]))) < tol)):
     #   T_th = (T[cur]).inv().dot(T_final[ee]).as_matrix()
     #   theta[ee] = wraptopi(theta[ee] +  arctan2(T_th[1, 0], T_th[0, 0]))

    return theta

def contract_graph(G):
    list_contraction = []
    
    dict_alias = {node: node for i,node in enumerate(G.nodes())}

    for i, e in enumerate(G.edges()):
        dist = np.linalg.norm(G.nodes[e[0]][POS] - G.nodes[e[1]][POS])
        if dist == 0:
            list_contraction.append(e)
            dict_alias[e[1]] = e[0]
    
    for e in list_contraction:
        G = nx.contracted_edge(G,e,self_loops=False)

    return G, dict_alias

def graph_motion_planning(G, position, direction = None, base_anchor = True ):



    return


