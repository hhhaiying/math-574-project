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


def s_alpha(alpha,lamb,gamma,s,i,r):
    """S compartment with age transfer alpha and infection transfer lambda
    s (3,)     : s[0] first age group s[1] second age group
    alpha (n,) : transfer rate from s_1 to s_2
    lamb (n+1,): infection rate for s_1 and s_2"""

    s_1 = s[0]
    s_2 = s[1]
    s_3 = s[2]
    s_1_p = -(alpha[0]+lamb[0])*s_1
    s_2_p =  alpha[0]*s_1-alpha[1]*s_2 -lamb[1]*s_2
    s_3_p = alpha[1]*s_2 - lamb[2]*s_3 
    i_1 = i[0]
    i_2 = i[1]
    i_3 = i[2]
    i_1_p = lamb[0]*s_1 - gamma[0]*i_1 -alpha[0]*i_1
    i_2_p = lamb[1]*s_2 - gamma[1]*i_2 + alpha[0]*i_1 - alpha[1]*i_2
    i_3_p = lamb[2]*s_3 - gamma[2]*i_3 + alpha[1]*i_2
    r_1 = r[0]
    r_2 = r[1]
    r_3 = r[2]
    r_1_p = gamma[0]*i_1 - alpha[0]*r_1
    r_2_p = gamma[1]*i_2 + alpha[0]*r_1 - alpha[1]*r_2
    r_3_p = gamma[2]*i_3 + alpha[1]*r_2 - alpha[2]*r_3
    state = np.array([s_1_p,s_2_p,s_3_p,i_1_p,i_2_p,i_3_p,r_1_p,r_2_p,r_3_p])
    return state
