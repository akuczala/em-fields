import numpy as np
from scipy.linalg import norm
from functools import reduce

from emfields import *
from emfields.point_charges import *

def F_to_Edir(F):
    Efield = get_E_field(F, out_dim=4)
    Enorm = enorm(Efield)
    if np.isclose(Enorm,0):
        return np.full_like(Efield,np.nan)
    else:
        return Efield/Enorm
def F_to_Bdir(F):
    Bfield = get_B_field(F, out_dim=4)
    Bnorm = enorm(Bfield)
    if np.isclose(Bnorm,0):
        return np.full_like(Bfield,np.nan)
    else:
        return Bfield/Bnorm
def make_field_line(F_fn, x0, n_steps, dx, F_to_dir):
    def field_line_step(x_old,F_old,d):
        x = x_old + d*dx
        F = F_fn(x)
        d = np.zeros_like(d) if np.isnan(x).all() else F_to_dir(F)
        return (x, F, d)
    out = [(x0, np.nan, np.zeros_like(x0))]
    for i in range(n_steps):
        out.append(field_line_step(*(out[-1])))
    return out[1:]

def test_Efield_lines(t, charge_dict, n_steps=20, dx=0.1):
    lines = []
    F_fn = lambda x: calc_F_traj(1, x, **charge_dict)
    for th in np.arange(20)*2*np.pi/20:
        line = make_field_line(
            F_fn,
            x0=np.array([t,0.5*np.cos(th) + np.cosh(t) - 1,0.5*np.sin(th),0.]),
            dx=dx,
            n_steps=n_steps,
            F_to_dir=F_to_dir
        )
        line_points = np.array([tup[0] for tup in line])
        lines.append(line_points)
    return lines
def test_Bfield_lines(t, charge_dict, n_steps=20, dx=0.1):
    lines = []
    F_fn = lambda x: calc_F_traj(1, x, **charge_dict)
    for r in np.linspace(0.5,2,5):
        line = make_field_line(
            F_fn,
            x0=np.array([t,np.cosh(t) - 2,r,0.]),
            n_steps=n_steps,
            dx=dx,
            F_to_dir=F_to_Bdir
        )
        line_points = np.array([tup[0] for tup in line])
        lines.append(line_points)
    return lines