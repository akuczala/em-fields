import numpy.linalg as lin
import numpy as np
from scipy.integrate import solve_ivp
import emfields.clifford as clif
from emfields.clifford import I
from emfields.em_fields import mnorm, mdot

import bmesh 

def draw_vec(v: np.ndarray, x: np.ndarray, to_3d_pos, bm=None, scale_fun = lambda v: 1., normalize = True, center = True):
    if bm is None:
        bm = bmesh.new()

    if np.isnan(v).any():
        return bm

    if normalize and not np.isclose(v,0).all():
        v = v/lin.norm(v)

    vec_points = scale_fun(v)*np.array([np.zeros_like(v), v])
    origin_point = x - np.mean(vec_points,axis=0) if center else x
    vec_verts = [bm.verts.new(to_3d_pos(p + origin_point)) for p in vec_points]

    bmesh.ops.contextual_create(bm, geom=vec_verts)

    return bm

def draw_bivec(B: clif.Clif,x: np.ndarray, to_3d_pos, bm=None, **kwargs):
    if bm is None:
        bm = bmesh.new()

    #print(B._arr)
    if np.isnan(B._arr).any() or B.isclose(0):
        return bm

    b1, b2 = clif.bivec_get_ortho(B)

    bm = draw_bivec_blade(b1, x, to_3d_pos, bm=bm, **kwargs)
    bm = draw_bivec_blade(b2, x, to_3d_pos, bm=bm, **kwargs)

    return bm

def draw_bivec_blade(B: clif.Clif, x, to_3d_pos, bm=None, scale_fun = lambda B: 1., center = True):
    if bm is None:
        bm = bmesh.new()

    if B.isclose(0):
        return bm

    v1, v2 = clif.decompose_bivec_blade(B)
    v1, v2 = (v.get_vector() for v in (v1,v2))

    euclid_normed_vecs =  [v/lin.norm(v) for v in (v1,v2)]

    # take a sqrt here so that isclose uses the correct magnitude
    mdots = [(lambda vdot: np.sqrt(np.abs(vdot))*np.sign(vdot))(mdot(v,v)) for v in (v1, v2)]
    mdot_signs = tuple(0 if np.isclose(v_sq,0) else int(np.sign(v_sq)) for v_sq in mdots)
    #abs_mnorm = lambda v: v/np.sqrt(np.abs(mdot(v,v)))
    abs_mnorm = lambda v : v #don't do minkowski normalization: we can get some absurdly large numbers
    space_time_vecs = lambda timelike, spacelike: [abs_mnorm(v)*np.sign(timelike[0]) for v in (timelike, spacelike)]
    null_space_vecs = lambda null, spacelike: [null*np.sign(null[0]), abs_mnorm(spacelike)*np.sign(null[0])]
    null_time_vecs = lambda null, timelike: [null*np.sign(timelike[0]), abs_mnorm(timelike)*np.sign(timelike[0])]

    sign_map = {
        (-1,-1): space_time_vecs(euclid_normed_vecs[0],euclid_normed_vecs[1]),
        (1,1): space_time_vecs(euclid_normed_vecs[0],euclid_normed_vecs[1]),
        (-1,1): space_time_vecs(euclid_normed_vecs[0],euclid_normed_vecs[1]),
        (1,-1): space_time_vecs(euclid_normed_vecs[1],euclid_normed_vecs[0]),
        (0, 1): null_space_vecs(euclid_normed_vecs[0],euclid_normed_vecs[1]),
        (1, 0): null_space_vecs(euclid_normed_vecs[1],euclid_normed_vecs[0]),
        (0,-1): null_time_vecs(euclid_normed_vecs[0],euclid_normed_vecs[1]),
        (-1,0): null_time_vecs(euclid_normed_vecs[1],euclid_normed_vecs[0])
    } 
    try:
        v1, v2 = sign_map[mdot_signs]
    except:
        raise ValueError(f'Got two null vectors? Got signs {mdot_signs} ')

    bivec_points = scale_fun(B)*np.array([np.zeros_like(v1), v1, v1 + v2, v2])
    origin_point = x - np.mean(bivec_points,axis=0) if center else x

    bivec_verts = [bm.verts.new(to_3d_pos(p + origin_point)) for p in bivec_points]

    bmesh.ops.contextual_create(bm, geom=bivec_verts)

    return bm

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
        n_pts = 20, res=None, max_s= 10, frac_curve = 1,
        solver_kwargs = dict(atol = 1e-8, rtol = 1e-4)
    ):
    if n_pts is None and res is None:
        raise Exception('Either n_pts or res must not equal None')

    s_span = (0,max_s)
    init_tangent = ivp_dx(0,x0,F_fun,project_out)
    orbit_dir = lambda t, y, *args: np.dot(y-x0,init_tangent) + 1e-16 #to exclude event at t=0
    orbit_dir.direction = 1
    orbit_dir.terminal = True
    if res is None:
        sol = solve_ivp(
            ivp_dx,
            t_span = s_span,
            y0=x0, args=(F_fun,project_out),
            events = orbit_dir,
            **solver_kwargs
        )
        if not sol.success:
            raise Exception(f'Could not find solution starting at {x0}')
        if len(sol.t_events) == 0:
            raise Exception(f'Could not find closed orbit starting at {x0}')
        
        #don't know what to make of this case
        if len(sol.t_events[0]) == 0:
            s_end = max_s
        else:
            s_end = sol.t_events[0][0]*frac_curve

        s_range = np.linspace(0,s_end,n_pts)
        events = None
    else:
        s_end = max_s
        events = orbit_dir
        s_range = np.arange(0,s_end,res)
    
    sol = sol = solve_ivp(
        ivp_dx,
        t_span = s_span,
        t_eval = s_range,
        events = events,
        y0=x0, args=(F_fun,project_out),
        **solver_kwargs
    )
    if not sol.success:
        raise Exception(f'Could not find solution starting at {x0}')

    x_sol = sol.y
    return x_sol.T

def calc_surface(x0, F_fun, project_outs, **kwargs):
    init_curve = calc_closed_curve(x0, F_fun, project_outs[0], **kwargs)
    perp_curve = calc_closed_curve(x0, F_fun, project_outs[1], **kwargs)
    curves = [init_curve]
    for x in np.concatenate([perp_curve[-3:],perp_curve[:5]],axis=0): #debug
        curves.append(calc_closed_curve(x, F_fun, project_outs[0], **kwargs))
    return curves

def draw_curve(x_sol, to_3d_pos, bm = None, closed=True, return_geom = False, remove_doubles=True):
    if bm is None:
        bm = bmesh.new()
    verts = [bm.verts.new(to_3d_pos(p)) for p in x_sol]
    edges = []
    viter = zip(verts,verts[1:]+ [verts[0]]) if closed else zip(verts,verts[1:])
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
    curve_geoms = [draw_curve(curve, to_3d_pos, bm, return_geom=True, remove_doubles=False) for curve in curves]
    curve_edges = [g['edges'] for g in curve_geoms]
    for edges1, edges2 in zip(curve_edges,curve_edges[1:]):
        for edge1, edge2 in zip(edges1, edges2):
            bmesh.ops.contextual_create(bm, geom=[edge1,edge2])
    return bm
