"""
Synergistic cmi in Ising systems. 


Tiago Martinelli, 2021
"""

import numpy as np
import pandas as pd
from itertools import combinations
import dit
from dit.multivariate import total_correlation as I
from numpy.random import randn, uniform

def HighOrder_Ising(N, order):
#     """
#     Generates a joint distribution where the first n-1 variables are maxent
#     iid, and the last variable is related to them via a high-order Ising-style
#     Hamiltonian.

#     Parameters
#     ----------
#     N : int
#         number of binary variables of the system
#     orderx : array
#         variance of the x-order terms in the Hamiltonian. (If zero in the coordinate then there
#         are no such terms) 

#     Returns
#     -------
#     p : np.ndarray
#         array with pmf of resulting Ising model
#     """
    K= 2**N

    binary_str = [np.binary_repr(i, N) for i in range(K)]
    binary_map = np.vstack([np.array([s[i] for i in range(N)]).astype(int) for s in binary_str])
    binary_map = 2*binary_map - 1;

    H = np.zeros(K);
    
    if np.count_nonzero(order) == 1:
        
        k=np.nonzero(order)[0][0]
        for c in combinations(range(N), k+1):
            l = np.vstack([binary_map[:, c[i]] for i in range(len(c))])
            coupling = order[k]*randn();
            H = H + coupling*np.multiply.reduce(l, axis=0)

    else:
    
        for k in range(1, N):
            for c in combinations(range(N), k+1):
                l = np.vstack([binary_map[:, c[i]] for i in range(len(c))])
                coupling = order[k-1]*randn();
                H = H + coupling*np.multiply.reduce(l, axis=0)
            
    p = np.exp(H)
    p = p/p.sum()

    outcomes = [format(i, '0%ib'%N) for i in range(K)]
    dist = dit.Distribution(outcomes, p)
    return dist

def HighOrder_DirectedIsing(N, order):
#     """
#     Generates a joint distribution where the first n-1 variables are maxent
#     iid, and the last variable is related to them via a high-order Ising-style
#     Hamiltonian.

#     Parameters
#     ----------
#     N : int
#         number of binary variables of the system
#     orderx : array
#         variance of the x-order terms in the Hamiltonian. (If zero in the coordinate then there
#         are no such terms) 

#     Returns
#     -------
#     p : np.ndarray
#         array with pmf of resulting Ising model
#     """
    K= 2**N

    binary_str = [np.binary_repr(i, N) for i in range(K)]
    binary_map = np.vstack([np.array([s[i] for i in range(N)]).astype(int) for s in binary_str])
    binary_map = 2*binary_map - 1;

    H = np.zeros(K);
    
    if np.count_nonzero(order) == 1:     
        k=np.nonzero(order)[0][0]
        for c in combinations(range(N-1), k+1):
            l = np.vstack(([binary_map[:, c[i]] for i in range(len(c))], [binary_map[:,-1]]))
            coupling = order[k]*uniform();
            H = H + coupling*np.multiply.reduce(l, axis=0)

    else:
    
        for k in range(1, N):
            for c in combinations(range(N), k):
                l = np.vstack(([binary_map[:, c[i]] for i in range(len(c))], [binary_map[:,-1]]))
                coupling = order[k-1]*uniform();
                H = H + coupling*np.multiply.reduce(l, axis=0)
            
    p = np.exp(H)
    p = p/p.sum()

    outcomes = [format(i, '0%ib'%N) for i in range(K)]
    dist = dit.Distribution(outcomes, p)
    return dist

def Simple_HO_DirectedIsing(order, std2=0, std3=0, std4=0, std5=0):
    
    N = order + 1
    K = 2**N

    binary_str = [np.binary_repr(i, N) for i in range(K)]
    binary_map = np.vstack([np.array([s[i] for i in range(N)]).astype(int) for s in binary_str])
    binary_map = 2*binary_map - 1;

    H = np.zeros(K);    

    ## Pairwise interactions
    coupling2 = std2*uniform();
    H = H + coupling2*(binary_map[:,0]*binary_map[:,-1]);

    # 3-rd order interactions
    coupling3 = std3*uniform();
    H = H + coupling3*(binary_map[:,1]*binary_map[:,2]*binary_map[:,-1]);
                    
    # 4-th order interactions
    coupling4 = std4*uniform();
    H = H + coupling4*(binary_map[:,3]*binary_map[:,4]*binary_map[:,5]*binary_map[:,-1]);
                
    ## 5-th order interactions
    coupling5 = std5*uniform();
    H = H + coupling5*(binary_map[:,6]*binary_map[:,7]*binary_map[:,8]*binary_map[:,9]*binary_map[:,-1]);

    p = np.exp(H)
    p = p/p.sum()

    outcomes = [format(i, '0%ib'%N) for i in range(K)]
    dist = dit.Distribution(outcomes, p)
    return dist

    outcomes = [format(i, '0%ib'%N) for i in range(K)]
    dist = dit.Distribution(outcomes, p)
    return dist

def cmi_synergy(size, std):#, atoms=None):
    """Utility function to compute synergy based on cmi of a random PDF given by an Ising
    model with given size, model order, and coefficient std."""   
#     kwargs = {'order%i'%i : std for i in range(2, order+1)} if order > 1 else {} 
#     d = My_DirectedIsing(size, **kwargs)

    #if atoms is not None:
    #order=0.1*np.eye(1, size-1, atoms)
        
    order=std*np.array([1]*(size-1))
    d = HighOrder_DirectedIsing(size, order)
    #d= Simple_HO_DirectedIsing(size, std)    
    #size = size+1
    mi = I(d, [list(range(size-1)), [size-1]])
    syn_order = []
    cmi = []        
    for source in range(0, size-1):
        for k in combinations(range(size-1), source+1):
            for index in k:
                cond_set = [x for x in k if x != index]
                syn_order.append(len(cond_set)+1)
                CMI = I(d, [[index], [size-1]], cond_set)
                cmi.append(CMI)

    df=pd.DataFrame()
    df['cmi_synergy']=cmi
    df['order']=syn_order
    table = df.groupby('order').mean()
    
    #if atoms is not None:
    #    return table.cumsum()['cmi_synergy'].tolist()
    return mi, table['cmi_synergy'].tolist()

def cmi_synergy_order(tpl):
    print(tpl)
    size, std = tpl
    mi, cmi = cmi_synergy(size, std)
    return cmi[size-2]/mi

