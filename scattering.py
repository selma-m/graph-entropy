import torch
import numpy as np
import networkx as nx
from numpy import linalg as LA
from itertools import product 

torch.manual_seed(29)
np.random.seed(29)

def lazy_random_walk(A):
    # input is adjacency matrix
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    d = A.sum(0) # sum along columns
    P_t = A/d 
    P_t[torch.isnan(P_t)] = 0
    identity_matrix = torch.eye(P_t.shape[0], dtype=P_t.dtype)
    P = 0.5 * (identity_matrix + P_t)
    return P

def graph_wavelet(P):
    psi = []
    for d1 in [1,2,4,8,16]: # these are the scales
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        W_d1_tensor = torch.tensor(W_d1, dtype=torch.float32)
        psi.append(W_d1_tensor)
    psi.append(torch.tensor(LA.matrix_power(P,2*16), dtype=torch.float32)) # add low-pass operator
    return psi

def node_level(x,A):
    P = lazy_random_walk(A)
    psi = graph_wavelet(P)

    new_x = x
    for i in range(len(psi)):
        new_x = psi[i]@new_x # multiply with wavelet
        if i < len(psi): # don't apply linearity the last time
            new_x = np.abs(new_x) # non-linearity

    return new_x

def node_level_blis(x,A, num_layers = 10):
    P = lazy_random_walk(A)
    psi = graph_wavelet(P)
    new_x = x
    for i in range(len(psi)):
        new_x = psi[i]@new_x # multiply with wavelet
        new_x = np.maximum(new_x, 0) + np.maximum(-new_x,0)# non-linearity

    return new_x


