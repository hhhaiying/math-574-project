import numpy as np

def SSPEV(b,alpha,lamb,gamma,delta,nu,s,i,r):
    """S compartment with age transfer alpha and infection transfer lambda
    b (1,)     : Birth rate for population 1
    s (2,)     : Susceptible groups
    i (2,)     : Infected groups
    r (2,)     : Recovered groups
    B (1,)     : SSPE group
    alpha (2,) : Age rate from s_1 to s_2
    lamb (4,) : Effective contact number between s_i*i_j
    gamma (2,) : Recovery rate 
    nu    (1,) : Vaccination rate 
    """

    
    s_1 = s[0]
    s_2 = s[1]
    i_1 = i[0]
    i_2 = i[1]
    r_1 = r[0]
    r_2 = r[1] 
    s_1_p = b-(alpha[0]+lamb[0]*i_1+lamb[1]*i_2)*s_1
    s_2_p =  -lamb[2]*s_2*i_1 - lamb[3]*s_2*i_2 + alpha[0]*s_1 - (nu+b)*s_2
    i_1_p = (lamb[0]*i_1+lamb[1]*i_2)*s_1 - gamma[0]*i_1
    i_2_p = lamb[2]*s_2*i_1 + lamb[3]*s_2*i_2  - gamma[1]*i_2 
    r_1_p = gamma[0]*i_1  - (delta[0]+b)*r_1
    r_2_p = gamma[1]*i_2 - (delta[1]+b)*r_2
    b_0_p = delta[0]*r_1 + delta[1]*r_2

    state = np.array([s_1_p,s_2_p,i_1_p,i_2_p,r_1_p,r_2_p,b_0_p])
    return state

def stability_system(b,d,beta,gamma,nu,s,i):
    """
    s,i system of reduced model
    """
    s_p = b*(1-s-i)-beta*i*s-nu*s
    i_p = b*i*s - gamma*i
    return s_p, i_p
