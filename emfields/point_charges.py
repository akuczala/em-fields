import numpy as np

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

def calc_F_traj(q,x,r_tau,u_tau,a_tau,tau_r,fkwargs={}, **kwargs):
    tau = tau_r(x,**fkwargs)
    if np.isnan(tau):
        return np.zeros([4,4])
    return calc_F_point(q,x,r_tau(tau,**fkwargs),u_tau(tau,**fkwargs),a_tau(tau,**fkwargs), **kwargs)

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