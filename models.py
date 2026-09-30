import numpy as np

def sir_v(lamb,gamma,t,x):
    """Vectorized sir with coefficients lambda and gamma"""
    s = x[0]
    i = x[1]
    return np.array([-lamb*s*i, lamb*s*i - gamma*i,gamma*i])

def sirs_bv(b,d,K,delta,lamb,gamma,t,x):
    """vectorized sirs with birth rate b and death rate d and population capacity K"""
    n = np.sum(x)
    s = x[0]
    i = x[1]
    r = x[2]
    s_p = b*n - n**2 / K - lamb*s-d*s + delta*r
    i_p = lamb*i*s - (gamma+d)*i
    r_p = gamma*i - d*r - delta*r
    return np.array([s_p, i_p, r_p])
