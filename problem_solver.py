import numpy as np
import time 
import pymanopt 
from pymanopt import manifolds, optimizers, tools, core
from pymanopt.core.problem  import Problem
from pymanopt.tools.diagnostics import *
from utils import *



def simple_IK(d ,C , C_joints_limit ,D ,D_joints ,n_below, n_above, Anchor_ ,M ,max_iter, W=None ,Y_init=None ,
                use_rand=False, min_grad_norm = 10**-10,verbosity = 2,lambda_cons=1, constraints=True,max_constr = False, Delta_bar = 30,Delta0=0.001,
                check_grad = False):
    if  W is None:
        W = np.eye(D.shape[0])

    if Anchor_ is not None:
        #print('ANCHOR')
        n_anchor = Anchor_.shape[1]
        Anchor = np.zeros((d,n_anchor))
        Anchor[:3,:] = Anchor_

        if constraints and not max_constr:
            if n_below != C_joints_limit.shape[0]:
                print('With ABOVE2')
                M = manifolds.Product([manifolds.ConstantFactory(Anchor),
                        manifolds.Oblique(d,D.shape[1]-Anchor.shape[1]),
                        manifolds.ComplementBall(d, n_below, D_joints[:n_below], 10**-8),
                        manifolds.Ball(d, C_joints_limit.shape[0]-n_below, D_joints[n_below:], 10**-8)])
            else:
                print('No above')
                M = manifolds.Product([manifolds.ConstantFactory(Anchor),
                        manifolds.Oblique(d,D.shape[1]-Anchor.shape[1]),
                        manifolds.ComplementBall(d, n_below, D_joints[:n_below], 10**-8)])
        else:
            M = manifolds.Product([manifolds.ConstantFactory(Anchor),
                    manifolds.Oblique(d,D.shape[1]-Anchor.shape[1])])
                
    else:
         n_anchor = 0
         Anchor = Anchor_
         if constraints and not max_constr:
            if n_below != C_joints_limit.shape[0]:
                M = manifolds.Product([manifolds.Oblique(d,D.shape[1]),
                        manifolds.ComplementBall(d, n_below, D_joints[:n_below], 10**-8),
                        manifolds.Ball(d, n_below, D_joints[:n_below], 10**-8)])
            else:
                M = manifolds.Product([manifolds.Oblique(d,D.shape[1]),
                        manifolds.ComplementBall(d, n_below, D_joints[:n_below], 10**-8)])

         else:
            M = manifolds.Product([manifolds.Oblique(d,D.shape[1])])

    Q2_ = -D@(W@C.transpose()@np.linalg.pinv(C@W@C.transpose()))@C@W@D
    Q1_ = D@W@D

    if constraints:
        if not max_constr:
            Q2 = lambda_cons**2*C_joints_limit.transpose()@C_joints_limit
            Q1 = np.zeros((C_joints_limit.shape[1],C_joints_limit.shape[1]))
            Q2[:Q2_.shape[0],:Q2_.shape[1]] = Q2[:Q2_.shape[0],:Q2_.shape[1]] + Q2_
            Q1[:Q1_.shape[0],:Q1_.shape[1]] = Q1_
        else:
            Q2 = Q2_
            Q1 = Q1_

            D_below = D_joints[:n_below]**2
            D_above = D_joints[n_below:]**2

            Q_below = C_joints_limit[:n_below,:D.shape[0]]
            Q_above = C_joints_limit[n_below:,:D.shape[0]]

            QQ_below = np.array([q[:,np.newaxis]@q[:,np.newaxis].transpose() for q in Q_below])
            QQ_above = np.array([q[:,np.newaxis]@q[:,np.newaxis].transpose() for q in Q_above])
            
            zip_below = zip(D_below,Q_below,QQ_below)
            zip_above = zip(D_above,Q_above,QQ_above)
    else:
        Q2 = Q2_
        Q1 = Q1_

    Y = M.random_point()
    ind = [0]
    ind.extend([y.shape[1] for y in Y])
    ind = np.array(ind)
    ind = np.cumsum(ind)
    n = len(Y)

    #cost_max(Q1 : np.ndarray, Q2: np.ndarray,Q_below: np.ndarray, Q_above: np.ndarray,
    #         D_below: np.ndarray, D_above:np.ndarray, *Y)
    #euclidean_gradient_max(Q2y,zip_below, zip_above, *Y,ind=None)
    #euclidean_hessian_max(Q2,zip_below, zip_above, *U,ind=None,n = None)

    @pymanopt.function.numpy(M)
    def cost(*Y):
        if max_constr:
            return cost_max(Q1,Q2,Q_below,Q_above,D_below, D_above,*Y,lambda_cons=lambda_cons)
        else:
            return cost_(Q1,Q2,C_joints_limit,Q1_,Q2_,*Y)
    
    @pymanopt.function.numpy(M)
    def euclidean_gradient(*Y):
        if max_constr:
            return euclidean_gradient_max(Q2,zip_below, zip_above, *Y,ind=ind,lambda_cons=lambda_cons)
        else:
            return euclidean_gradient_(Q2,*Y,ind=ind)
                            
    @pymanopt.function.numpy(M)
    def euclidean_hessian(*U):
        if max_constr:
            return euclidean_hessian_max(Q2,zip_below, zip_above, *U,ind=ind,n = n,lambda_cons=lambda_cons)
        else:
            return euclidean_hessian_(Q2,*U,ind=ind,n=n)
    

    problem = Problem(manifold=M, 
                        cost=cost,
                        euclidean_gradient=euclidean_gradient, 
                        euclidean_hessian=euclidean_hessian,
                        )
    
    if check_grad:
        check_gradient(problem)
    optimizer = optimizers.TrustRegions(max_iterations=max_iter,use_rand=use_rand,min_gradient_norm=min_grad_norm,verbosity=verbosity)

    if  Y_init is None:
        Y_init_ = M.random_point()
    else:
        Y_init_ =  dimension_initialisation(Y_init, d, constraints, n_anchor, n_below, C_joints_limit, D.shape[0], Anchor, max_constr)
    #Y_init_ = M.retraction(Y_init_,M.zero_vector(Y_init_))
    #Y_init_ = M.random_point() 
    #print(cost_max(Q1,Q2,Q_below,Q_above,D_below, D_above,*Y_init))
    Y_star = optimizer.run(problem,initial_point=Y_init_, Delta_bar = Delta_bar,Delta0=Delta0).point
    
    return Y_star, Q2+Q1