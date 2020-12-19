import numpy as np
from scipy.sparse import dok_matrix 

signature = np.array([-1,1,1,1])
metric = np.diag(signature)
space_mask = np.array([0,1,1,1])

def mdot(x,y,**kwargs):
    return (x*y*signature).sum(axis=-1,**kwargs)
def sq_mnorm(x,**kwargs):
    return mdot(x,x,**kwargs)
def mnorm(x,**kwargs):
    return np.sqrt(sq_mnorm(x,**kwargs))
def edot(x,y,**kwargs):
    return (x*y*space_mask).sum(axis=-1,**kwargs)
def sq_enorm(x,**kwargs):
    return edot(x,x,**kwargs)
def enorm(x,**kwargs):
    return np.sqrt(sq_enorm(x,**kwargs))
def is_on_cone(Dx_grids,**kwargs):
    return np.isclose(sq_mnorm(Dx_grids,**kwargs),0,atol=1)

def antisym2(T):
    return T - T.T
def outerprod(u,v):
    return np.tensordot(u, v, axes=0)

square_lc_dict = {
    (1, 11) : 1, (1, 14) : -1, (2, 7) : -1, (2, 13) : 1,
    (3, 6) : 1, (3, 9) : -1, (4, 11) : -1, (4, 14) : 1,
    (6, 3) : 1, (6, 12) : -1, (7, 2) : -1, (7, 8) : 1,
    (8, 7) : 1, (8, 13) : -1, (9, 3) : -1, (9, 12) : 1,
    (11, 1) : 1, (11, 4) : -1, (12, 6) : -1, (12, 9) : 1,
    (13, 2) : 1, (13, 8) : -1, (14, 1) : -1, (14, 4) : 1
}

square_levi_civita = dok_matrix((16,16),dtype=np.float32)
for (i,j), v in square_lc_dict.items():
    square_levi_civita[i,j] = v
levi_civita = square_levi_civita.toarray().reshape(4,4,4,4)

def get_E_field(F,out_dim=3):
    if out_dim == 3:
        return F[...,0,1:]
    else:
        return F[...,0]

def get_B_field(F,out_dim=3):
    Fcov = np.einsum('ki,...ij,jl->...kl',metric,F,metric)
    Fdual = np.einsum('...ij,ijkl->...kl',Fcov,levi_civita/2) 
    if out_dim == 3:
        return Fdual[...,0,1:]
    else:
        return Fdual[...,0]