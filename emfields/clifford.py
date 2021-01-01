import numpy as np
import itertools
from collections.abc import Iterable
from functools import singledispatch
from numbers import Number
from emfields.em_fields import *

zero_mat = np.zeros([4,4])
id_mat = np.eye(4)

class Clif:

    def __init__(self, arr = zero_mat):
        self._arr = _init_arr(arr)

    def __add__(self, other):
        return _add(other, self)
    def __radd__(self,other):
        return _add(other, self)

    def __sub__(self, other):
        return _rev_sub(other, self)
    def __rsub__(self, other):
        return _sub(other, self)
    def __neg__(self):
        return Clif(-self._arr)

    def __mul__(self, other):
        return _rev_mul(other, self)
    def __rmul__(self, other):
        return _mul(other, self)

    def __truediv__(self, other):
        return _rev_div(other, self)

    def __eq__(self, other):
        return _eq(other, self)

    def __getitem__(self, index):
        return self.get_component(index)*get_basis(index)

    def __repr__(self):
        if self == 0:
            return 'Clif(0)'
        print_coef = lambda coef: '' if coef == 1 else f'{coef}*'
        print_scalar = lambda idx, coef: f'Clif({coef}) + '
        print_blade = lambda idx, coef: f'{print_coef(coef)}e{list(idx)} + ' 
        print_I = lambda idx, coef: f'{print_coef(coef)}I + '
        print_map = {0: print_scalar, 1: print_blade, 2: print_blade, 3: print_blade, 4: print_I}
        return ''.join(print_map[len(idx)](idx, coef) for idx, coef in self.get_components().items())[:-3]

    def isclose(self, other, **kwargs):
        return np.isclose(self._arr, Clif(other)._arr, **kwargs).all()

    def scalar_part(self):
        return np.trace(self._arr)/self._arr.shape[0]

    def get_component(self,indices):
        if not isinstance(indices,Iterable):
            indices = [indices]
        i_rev = indices if indices == 'I' else indices[::-1] 
        comp = get_basis(i_rev)
        signature_sign = -1 if indices == 'I' or 0 in indices else 1
        return (comp * self).scalar_part()*signature_sign

    def get_components(self, return_zero = False):
        return flatten_nested_dict(self.get_graded_components(return_zero = return_zero)) 

    def get_graded_components(self, return_zero = False):
        return dict(filter(
            lambda kv: len(kv[1])>0,
            {
                i: dict(filter(
                    lambda kv: return_zero or not np.isclose(kv[1],0),
                    {
                        indices: self.get_component(indices) for indices in rank_bases

                    }.items()
                )) for i, rank_bases in enumerate(clif_basis) 
            }.items()
        ))
    def get_vector(self):
        return np.array([self.get_component([i]) for i in range(4)])

    def get_grade(self, k):
        return sum(self.get_component(indices)*e[indices] for indices in clif_basis[k])
    def wedge(self, other):
        return sum(
            self.get_grade(k)._wedge_homogeneous(other.get_grade(l),k,l)
            for k,l in itertools.product(range(top_grade+1),range(top_grade+1))
        )
    def _wedge_homogeneous(self, other, self_grade, other_grade):
        if self_grade + other_grade > top_grade:
            return Clif(0)
        else:
            return (self*other).get_grade(self_grade + other_grade)

#use singledispatch to support operators with python / numpy numerical types
#would use singledispatchmethod but this is only supported in python 3.8+
#and blender uses 3.7

@singledispatch
def _init_arr(x):
    raise TypeError(f'Cannot instantiate Clif with {x} of type {type(x)}')
@_init_arr.register
def _(x: np.ndarray):
    return x
@_init_arr.register
def _(x: list):
    if len(x) != 4:
        raise ValueError(f'Clif can only be instantiated with lists of length 4; got {x}')
    return from_components({(i): xi for i, xi in enumerate(x)})._arr
@_init_arr.register
def _(x: Number):
    return x*id_mat
@_init_arr.register
def _(x: Clif):
    return x._arr

#some of these could be eliminated using Clif(x), but could slow execution
@singledispatch
def _eq(x, y):
    raise TypeError(f'{type(x)}=={type(y)} not supported')
@_eq.register
def _(x: Number, y: Clif):
    return Clif(x*id_mat) == y
@_eq.register
def _(x: Clif, y: Clif):
    return np.all(x._arr == y._arr)

@singledispatch
def _add(x, y):
    raise TypeError(f'{type(x)}+{type(y)} not supported')
@_add.register
def _(x: Number, y: Clif):
    return Clif(x*id_mat) + y
@_add.register
def _(x: Clif, y: Clif):
    return Clif(x._arr + y._arr)

@singledispatch
def _rev_sub(x, y):
    return sub(y, x)
@_rev_sub.register
def _(x: Number, y: Clif):
    return y - Clif(x*id_mat)
@_rev_sub.register
def _(x: Clif, y: Clif):
    return _sub(y, x)

@singledispatch
def _sub(x, y):
    return TypeError(f'{type(y)}-{type(x)} not supported')
@_sub.register
def _(x: Number, y: Clif):
    return Clif(x*id_mat) - y
@_sub.register
def _(x: Clif, y: Clif):
    return Clif(x._arr - y._arr)

@singledispatch
def _rev_mul(x, y):
    return _mul(y, x) 
@_rev_mul.register
def _(x: Number, y: Clif):
    return _mul(x, y)
@_rev_mul.register
def _(x: Clif, y: Clif):
    return _mul(y, x)

@singledispatch
def _mul(x, y):
    return TypeError(f'{type(y)}*{type(x)} not supported')
@_mul.register
def _(x: Number, y: Clif):
    return Clif(x*y._arr)
@_mul.register
def _(x: Clif, y: Clif):
    return Clif(np.dot(x._arr,y._arr))

@singledispatch
def _rev_div(x, y):
    return TypeError(f'{type(y)}/{type(x)} not supported')
@_rev_div.register
def _(x: Number, y: Clif):
    return Clif(y._arr/x)

def wedge_vec(v1,v2):
    return (v1*v2 - v2*v1)/2

signature = (-1, 1, 1, 1)
metric = np.diag(signature).astype(np.float64)

def build_gammas():
    #Pauli
    id2 = np.eye(2)
    X = np.array([[0,1],[1,0]],dtype=np.float64)
    Z = np.array([[1,0],[0,-1]],dtype=np.float64)
    T = np.dot(X,Z) # = -iY, squares to -1
    
    return np.stack([
        outerprod(T, id2),
        outerprod(X, X),
        outerprod(Z, id2),
        outerprod(X, Z)
    ]).transpose(0,1,3,2,4).reshape(4,4,4)

gammas = [Clif(mat) for mat in build_gammas()]
#sigma_ij = gamma[i]^gamma[j]
sigmas = np.array([
    [wedge_vec(gammas[i],gammas[j]) for j in range(4)] for i in range(4)
])
I = sigmas[0,1]*sigmas[2,3]




def get_basis(indices):

    if not isinstance(indices,Iterable):
        indices = [indices]

    if indices == 'I':
        return I # = reverse
    elif len(indices) == 0:
        return Clif(1)
    elif len(indices) == 1:
        return gammas[indices[0]]
    elif len(indices) == 2:
        return sigmas[indices[0],indices[1]]
    elif len(indices) == 3:
        return sigmas[indices[0],indices[1]] * gammas[indices[2]]
    elif len(indices) == 4:
        return sigmas[indices[0],indices[1]] * sigmas[indices[2],indices[3]]
    else:
        raise ValueError(f'Invalid indices {indices}')
    


def flatten_nested_dict(dct):
    return {k: v for _, d in dct.items() for k, v in d.items()}


def from_components(components):
    return sum(coeff*get_basis(indices) for indices, coeff in components.items())

def twoform_to_clif(F):
    return sum(F[i,j]*sigmas[i,j] for i,j in itertools.combinations(range(4),2))

def bivec_get_ortho(B):
    if B.isclose(0):
        raise Exception('Got 0')
    Bsq = B*B
    Bsq0 = Bsq.scalar_part()
    BsqI = Bsq.get_component('I')
    r = np.abs(Bsq0 + 1j*BsqI)
    th = np.angle(Bsq0 + 1j*BsqI)

    Bhat = (np.cos(-th/2) + np.sin(-th/2)*I)*B/np.sqrt(r)
    B1 = np.sqrt(r)*np.cos(th/2)*Bhat
    B2 = np.sqrt(r)*np.sin(th/2)*I*Bhat
    return B1, B2

def decompose_bivec_blade(b, v0 = None):
    if b.isclose(0).all():
        return (0.,0.)
    
    if v0 is None:
        for i in range(4): #tends to make smaller index appear first in output
            dot_comps = (gammas[i]*b).get_graded_components()
            if 1 in dot_comps and len(dot_comps) > 0:
                v0 = gammas[i]
                break
        if v0 is None:
            raise ValueError(f'{b} might not be a blade')
    
    v1 = (v0*b - b*v0)/2 #
    v2 = -(v1*b)/(v1*v1).scalar_part()
    try:
        assert wedge_vec(v2,v1).isclose(b)
    except AssertionError:
        raise ValueError(f'{b} is not a blade')
    try:
        assert np.isclose((v1*v2).scalar_part(),0)
    except AssertionError:
        raise ValueError(f'Resulting vectors {v1}, {v2} not perpendicular')
    return (v2, v1)

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
    normed_B = B/np.sqrt(np.abs((B*B).scalar_part()))
    R0 = np.cos(-angle/2)
    R2 = B*np.sin(-angle/2)
    R = R0 + R2; R_rev = R0 - R2
    return R*M*R_rev
def boost(M, B, angle):
    #timelike blade B
    normed_B = B/np.sqrt(np.abs((B*B).scalar_part()))
    R0 = np.cosh(-angle/2)
    R2 = B*np.sinh(-angle/2)
    R = R0 + R2; R_rev = R0 - R2
    return R*M*R_rev

top_grade = 4
clif_basis = [[c for c in itertools.combinations(range(4),m)] for m in range(top_grade+1)]
e = from_components({c: 1.0 for c in itertools.chain.from_iterable(clif_basis)})

for i in range(4):
    assert gammas[i]*gammas[i] == signature[i]
    assert gammas[i].get_component([i]) == 1.0

for i in range(4):
    for j in range(4):
        if i != j:
            assert sigmas[i,j] == -sigmas[j,i]
            assert sigmas[i,j].get_component([i,j]) == 1.0
            
assert I.get_component('I') == 1.0

test_multi = gammas[2] + 2*gammas[3] + 4*sigmas[0,1] + sigmas[1,0] + 4*I
assert test_multi == from_components(test_multi.get_components())
assert eval(repr(test_multi)) == test_multi

assert (lambda x: x[0]*x[1])(bivec_get_ortho(sigmas[0,1]+sigmas[2,3] + sigmas[0,2])) == I