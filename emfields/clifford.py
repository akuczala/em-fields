import numpy as np
from itertools import combinations

from emfields.em_fields import *


def wedge_vec(v1,v2):
    return (np.dot(v1,v2) - np.dot(v2,v1))/2
def scalar_part(A):
    return np.trace(A)/A.shape[0]
#Pauli
id2 = np.eye(2)
X = np.array([[0,1],[1,0]],dtype=np.float64)
Z = np.array([[1,0],[0,-1]],dtype=np.float64)
T = np.dot(X,Z) # = -iY, squares to -1

signature = (-1, 1, 1, 1)
metric = np.diag(signature).astype(np.float64)
gammas = np.stack([
    outerprod(T, id2),
    outerprod(X, X),
    outerprod(Z, id2),
    outerprod(X, Z)
]).transpose(0,1,3,2,4).reshape(4,4,4)

#sigma_ij = gamma[i]^gamma[j]
sigmas = np.array([
    [wedge_vec(gammas[i],gammas[j]) for j in range(4)] for i in range(4)
])
pseudoscalar = np.dot(sigmas[0,1],sigmas[2,3])

def get_basis(indices):
    if indices == 'pseudo':
        return pseudoscalar # = reverse
    elif len(indices) == 0:
        return 1
    elif len(indices) == 1:
        return gammas[indices[0]]
    elif len(indices) == 2:
        return sigmas[indices[0],indices[1]]
    elif len(indices) == 3:
        return np.dot(sigmas[indices[0],indices[1]],gammas[indices[2]])
    elif len(indices) == 4:
        return np.dot(sigmas[indices[0],indices[1]],sigmas[indices[2],indices[3]])
    else:
        raise ValueError(f'Invalid indices {indices}')
    
def get_component(A,indices):
    i_rev = indices if indices == 'pseudo' else indices[::-1] 
    comp = get_basis(i_rev)
    signature_sign = -1 if indices == 'pseudo' or 0 in indices else 1
    return scalar_part(np.dot(comp,A))*signature_sign


clif_basis = [[c for c in combinations(range(4),m)] for m in range(5)]

def flatten_nested_dict(dct):
    return {k: v for _, d in dct.items() for k, v in d.items()}

def get_components(A, return_zero = False):
    return flatten_nested_dict(get_graded_components(A, return_zero = return_zero))

def get_graded_components(A, return_zero = False):
    return dict(filter(
        lambda kv: len(kv[1])>0,
        {
            i: dict(filter(
                lambda kv: return_zero or not np.isclose(kv[1],0),
                {
                    indices: get_component(A, indices) for indices in rank_bases

                }.items()
            )) for i, rank_bases in enumerate(clif_basis) 
        }.items()
    ))

def from_components(components):
    return sum(coeff*get_basis(indices) for indices, coeff in components.items())

def get_vector(A):
    return np.array([get_component(A,[i]) for i in range(4)])

def twoform_to_clif(F):
    return sum(F[i,j]*sigmas[i,j] for i,j in combinations(range(4),2))

def bivec_get_ortho(B):
    Bsq = np.dot(B,B)
    Bsq0 = scalar_part(Bsq)
    BsqI = get_component(Bsq,'pseudo')
    r = np.abs(Bsq0 + 1j*BsqI)
    th = np.angle(Bsq0 + 1j*BsqI)

    Bhat = np.dot((np.cos(-th/2)*np.eye(4) + np.sin(-th/2)*pseudoscalar),B)/np.sqrt(r)
    #print(get_components(np.dot(Bhat,Bhat)))
    B1 = np.sqrt(r)*np.cos(th/2)*Bhat
    B2 = np.sqrt(r)*np.sin(th/2)*np.dot(pseudoscalar,Bhat)
    return B1, B2

def decompose_bivec_blade(b):
    if np.isclose(b,0).all():
        return (np.zeros([4,4]),np.zeros([4,4]))
    v1 = None
    for i in range(4)[::-1]: #tends to make smaller index appear first in output
        dot_comps = get_graded_components(np.dot(gammas[i],b))
        if 1 in dot_comps and len(dot_comps) > 0:
            v1 = gammas[i]
            break
    if v1 is None:
        raise Exception('wat')
    v1 = -(np.dot(v1,b) - np.dot(b,v1))/2 #
    v2 = np.dot(v1,b)/scalar_part(np.dot(v1,v1))
    try:
        assert np.isclose(wedge_vec(v1,v2), b).all()
    except AssertionError:
        raise ValueError('Input is not a blade')
    return (v1, v2)

# def blade_exp(blade):
#     bsq_0 = scalar_part(np.dot(blade,blade))
#     if np.isclose(bsq_0, 0):
#         return np.eye(4)
#     elif bsq_0 > 0:
#         return np.eye(4)*np.cosh(bsq_0) + blade*np.sinh(bsq_0)/bsq_0
#     else:
#         return np.eye(4)*np.cos(bsq_0) + blade*np.sin(-bsq_0)/bsq_0
def rotate(M, B, angle):
    #spacelike blade B
    normed_B = B/np.sqrt(np.abs(scalar_part(np.dot(B,B))))
    R0 = np.cos(-angle/2)*np.eye(4)
    R2 = B*np.sin(-angle/2)
    R = R0 + R2; R_rev = R0 - R2
    return np.dot(np.dot(R,M),R_rev)
def boost(M, B, angle):
    #timelike blade B
    normed_B = B/np.sqrt(np.abs(scalar_part(np.dot(B,B))))
    R0 = np.cosh(-angle/2)*np.eye(4)
    R2 = B*np.sinh(-angle/2)
    R = R0 + R2; R_rev = R0 - R2
    return np.dot(np.dot(R,M),R_rev)


for i in range(4):
    assert (np.dot(gammas[i],gammas[i]) == signature[i]*np.eye(4)).all()
    assert get_component(gammas[i],[i]) == 1.0

for i in range(4):
    for j in range(4):
        if i != j:
            assert (sigmas[i,j] == -sigmas[j,i]).all()
            assert get_component(sigmas[i,j],[i,j]) == 1.0
            
assert get_component(pseudoscalar,'pseudo') == 1.0

test_multi = gammas[2] + 2*gammas[3] + 4*sigmas[0,1] + sigmas[1,0] + 4*pseudoscalar
assert (test_multi == from_components(get_components(test_multi))).all()