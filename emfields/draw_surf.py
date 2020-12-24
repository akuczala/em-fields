import numpy.linalg as lin
import numpy as np
from scipy.integrate import solve_ivp
import emfields.clifford as clif
from emfields.clifford import I
from emfields.em_fields import mnorm, mdot

import bmesh 

def get_norm(F):
    return (F*F).scalar_part()
def get_form_path_tangent(x, F_fun, project_out):
    F = F_fun(x)
    if np.isnan(F).any():
        return np.zeros_like(x)
    F_clif = clif.twoform_to_clif(F) #assume F is a blade for now
    Fsq = (F_clif*F_clif).scalar_part()
    tangent = (F_clif*I*project_out).get_grade(1).get_vector()
    if np.isclose(tangent,0).all():
        return tangent
    #return 0.5*tangent/lin.norm(tangent)*(absFsq_min/np.abs(Fsq))**0.25
    return tangent/lin.norm(tangent)
def ivp_dx(t,x, *args):
    return get_form_path_tangent(x, *args)

def calc_closed_curve(
        x0, F_fun, project_out,
        res = 0.1, max_s= 10, solver_kwargs = dict(atol = 1e-8, rtol = 1e-4)
    ):
    s_range = np.arange(0,max_s,res)
    s_span = (s_range.min(),s_range.max())
    init_tangent = ivp_dx(0,x0,F_fun,project_out)
    orbit_dir = lambda t, y, *args: np.dot(y-x0,init_tangent) + 1e-16 #to exclude event at t=0
    orbit_dir.direction = 1
    orbit_dir.terminal = True
    sol = solve_ivp(
        ivp_dx,
        t_span = s_span,
        t_eval=s_range,
        y0=x0, args=(F_fun,project_out),
        events = orbit_dir,
        **solver_kwargs
    )
    if len(sol.t_events) == 0:
        print(f'Could not find closed orbit starting at {x0}')
    if not sol.success:
        raise Exception(f'Could not find solution starting at {x0}')
    x_sol = sol.y
    return x_sol.T

def calc_surface(x0, F_fun, project_outs, **kwargs):
    init_curve = calc_closed_curve(x0, F_fun, project_outs[0], **kwargs)
    perp_curve = calc_closed_curve(x0, F_fun, project_outs[1], **kwargs)
    curves = [init_curve]
    for x in perp_curve[:5]:
        curves.append(calc_closed_curve(x, F_fun, project_outs[0], **kwargs))
    return curves

def draw_closed_curve(x_sol, to_3d_pos, bm = None, return_geom = False, remove_doubles=True):
    if bm is None:
        bm = bmesh.new()
    verts = [bm.verts.new(to_3d_pos(p)) for p in x_sol]
    edges = []
    viter = zip(verts,verts[1:]+ [verts[0]])
    for v1, v2 in viter:
        edges += bmesh.ops.contextual_create(bm, geom=[v1, v2])['edges']

    if remove_doubles:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    if return_geom:
        return {'verts': verts, 'edges': edges}
    else:
        return bm

def draw_surface(curves, to_3d_pos, bm = None):
    if bm is None:
        bm = bmesh.new()
    curve_geoms = [draw_closed_curve(curve, to_3d_pos, bm, return_geom=True, remove_doubles=False) for curve in curves]
    curve_edges = [g['edges'] for g in curve_geoms]
    for edges1, edges2 in zip(curve_edges,curve_edges[1:-1]):
        for edge1, edge2 in zip(edges1, edges2):
            bmesh.ops.contextual_create(bm, geom=[edge1,edge2])
    return bm
