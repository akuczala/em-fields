import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from emfields.clifford import e, Clif

from emfields.em_fields import *

def calc_F_point(q,x,r,u,a, exclude_radius=0.1):
    R = x - r
    if exclude_radius is not None and enorm(R) < exclude_radius:
        print('exclude')
        return np.full([4,4],np.nan)
    const = q/(4*np.pi)
    p = mdot(u,R)
    term1 = (1-mdot(a,R))*outerprod(R,u)/p**3
    term2 = (outerprod(R,a)-outerprod(u,u))/p**2
    F = const*antisym2(term1 + term2)
    return F
def calc_F_point_static_only(q,x,r,u,a, exclude_radius=0.1):
    R = x - r
    if exclude_radius is not None and enorm(R) < exclude_radius:
        print('exclude')
        return np.full([4,4],np.nan)
    const = q/(4*np.pi)
    p = mdot(u,R)
    term1 = outerprod(R,u)/p**3
    F = const*antisym2(term1)
    return F
def calc_F_point_radiation_only(q,x,r,u,a, exclude_radius=0.1):
    R = x - r
    if exclude_radius is not None and enorm(R) < exclude_radius:
        print('exclude')
        return np.full([4,4],np.nan)
    const = q/(4*np.pi)
    p = mdot(u,R)
    term1 = (-mdot(a,R))*outerprod(R,u)/p**3
    term2 = (outerprod(R,a)-outerprod(u,u))/p**2
    F = const*antisym2(term1 + term2)
    return F

def calc_F_traj(q,x,r_tau,u_tau,a_tau,tau_r,F_point_fn=calc_F_point,fkwargs={}, **kwargs):
    tau = tau_r(x,**fkwargs)
    if np.isnan(tau):
        return np.zeros([4,4])
    return F_point_fn(q,x,r_tau(tau,**fkwargs),u_tau(tau,**fkwargs),a_tau(tau,**fkwargs), **kwargs)

def get_stationary_dict(x_init=np.zeros(4)):
    def r_tau(tau):
        return np.array([tau,0,0,0]) + x_init
    def u_tau(tau):
        return np.array([1,0,0,0])
    def a_tau(tau):
        return np.array([0,0,0,0])
    def tau_r(x):
        x_shift = x - x_init
        return x_shift[0] - enorm(x_shift)
    return dict(r_tau=r_tau,u_tau=u_tau,a_tau=a_tau,tau_r=tau_r)

def get_constant_velocity_dict(v,x_init=np.zeros(4)):
    def r_tau(tau):
        return np.array([tau,tau*v,0,0])/np.sqrt(1-v*v) + x_init
    def u_tau(tau):
        return np.array([1,v,0,0])/np.sqrt(1-v*v)
    def a_tau(tau):
        return np.array([0,0,0,0])
    def tau_r(x):
        x_shift = x - x_init
        gamma = 1/np.sqrt(1-v*v)
        discr = (v*x_shift[0]-x_shift[1])**2 + gamma**2 * (x_shift[2]**2+x_shift[3]**2)
        return ((x_shift[0]-v*x_shift[1]) - np.sqrt(discr))*gamma
    return dict(r_tau=r_tau,u_tau=u_tau,a_tau=a_tau,tau_r=tau_r)

def get_accelerating_dict(x_init=np.array([0,1.,0,0])):
    x_offset = x_init - np.array([0,1.,0,0])
    def r_tau(tau):
        return np.array([np.sinh(tau),np.cosh(tau),0,0]) + x_offset
    def u_tau(tau):
        return np.array([np.cosh(tau),np.sinh(tau),0,0])
    def a_tau(tau):
        return np.array([np.sinh(tau),np.cosh(tau),0,0])
#     def tau_r(x,x_offset):
#         Rnorm = lambda tau: sq_mnorm(x - r_tau(tau,x_offset))
#         sol = root_scalar(Rnorm,x0=-5,x1=-1)
#         print(sol.root)
#         return sol.root
    def tau_r(x):
        x_shift = x - x_offset
        xsq= sq_mnorm(x_shift)
        discr = (1 + xsq)**2 + 4*(x_shift[0]**2 - x_shift[1]**2)
        if discr < 0:
            return np.nan
        else:
            log_arg = (-1 - xsq + np.sqrt(discr))/(2*(x_shift[0]-x_shift[1]))
            if log_arg > 0:
                return np.log(log_arg)
            else:
                return np.nan
    return dict(r_tau=r_tau,u_tau=u_tau,a_tau=a_tau,tau_r=tau_r)

def get_blastoff_dict(x_init=np.zeros([4])):
    rest_dict = get_constant_velocity_dict(0.,x_init=x_init)
    accel_dict = get_accelerating_dict(x_init=x_init)
    def r_tau(tau):
        if tau > 0:
            return accel_dict['r_tau'](tau)
        else:
            return rest_dict['r_tau'](tau)
    def u_tau(tau):
        if tau > 0:
            return accel_dict['u_tau'](tau)
        else:
            return rest_dict['u_tau'](tau)
    def a_tau(tau):
        if tau > 0:
            return accel_dict['a_tau'](tau)
        else:
            return rest_dict['a_tau'](tau)
    def tau_r(x):
        tau = rest_dict['tau_r'](x)
        if tau > 0:
            return accel_dict['tau_r'](x)
        else:
            return tau
    return dict(r_tau=r_tau,u_tau=u_tau,a_tau=a_tau,tau_r=tau_r)

def accel_quad(t, a=1.):
    if t < 0:
        return Clif(0)
    elif t < 1:
        return 4*t*(1-t)*e[0]*e[1]*a
    elif t < 2:
        return 4*(t-1)*(t-2)*e[0]*e[1]*a
    else:
        return Clif(0)
def dxdu(t,y,accel, accel_kwargs={}):
    x, u = [Clif(list(v)) for v in y.reshape(2,4)]
    du = (accel(t,**accel_kwargs)*u).get_grade(1)
    dx = u
    return np.concatenate([dx.get_vector(),du.get_vector()])

def interp_path(accel,t_range,accel_kwargs={},x0=np.zeros([4]),u0=np.eye(4)[0]):
    y0 = np.concatenate([x0,u0])
    sol = solve_ivp(dxdu,y0=y0,t_span=(t_range[0],t_range[-1]),t_eval=t_range,
                    args=(accel_quad,accel_kwargs),
                    atol=1e-16,rtol=1e-15
    )
    if not sol.success:
        raise Exception('Solution not found')
    dy_t = np.array([dxdu(t,y,accel,accel_kwargs) for t,y in zip(t_range,sol.y.T)])
    x_t = sol.y[:4].T
    u_t = sol.y[4:].T
    a_t = dy_t[:,4:]
    
    out_dict =  {
        fname: interp1d(t_range,v_t, axis=0, fill_value='extrapolate', bounds_error=False)
        for fname, v_t in zip(['r_tau','u_tau','a_tau'],[x_t,u_t,a_t])
    }
    #time component is always invertible
    tau_r0 = interp1d(x_t[:,0],t_range, axis=0, fill_value='extrapolate', bounds_error=False)
    
    def tau_r(x):
        tr = x[0] - enorm(x)
    #     if tr < 0:
    #         return tr
    #     elif tr > 2:
    #         tr = x[0] - enorm(x-accel_quad_dict['r_tau'](3))
    #         return accel_quad_dict['tau_r0'](tr)
        Rnorm = lambda tau: sq_mnorm(x - out_dict['r_tau'](tau))
        fprime = lambda tau: -2*mdot(out_dict['u_tau'](tau),x - out_dict['r_tau'](tau))

        #sol = root_scalar(Rnorm,x0=0.0, x1 =2.0,rtol=1e-8,xtol=1e-8)
        sol = root(Rnorm,tr-1)
        #print(sol)
        #print(Rnorm(sol.root))
        if not sol.success:
            print(sol)
            #raise Exception(f'Could not find root. {x}')
        return sol.x.item()
    out_dict['tau_r'] = tau_r
    return out_dict
def get_accel_quad_dict(a):
    print('interpolating...')
    out_dict = interp_path(accel_quad,accel_kwargs={'a':a},t_range=np.linspace(-1,3,50),x0=np.array([-1.,0,0,0]))
    print('done')
    return out_dict
