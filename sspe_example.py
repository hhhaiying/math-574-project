import numpy as np
import matplotlib.pyplot as plt
from models import s_alpha_SSPE


def taylor_system_11(f,initial,alpha,lamb,gamma,p,t0,t1,n):
    """Naive 2 S compartment connection. Forward 2nd order. SSPE leaks at proportion p : (2,)"""
    t = t0
    h = (t1 - t0) / n
    points = np.zeros([9,n])

    s = initial[:,0]
    i = initial[:,1]
    r = initial[:,2]
    b = initial[:,3]

    for k in range(n):
        points[0,k] = t
        points[1,k] = s[0]
        points[2,k] = s[1]
        points[3,k] = i[0]
        points[4,k] = i[1]
        points[5,k] = r[0]
        points[6,k] = r[1]
        points[7,k]= b[0]
        points[8,k]= b[1]
        s_1_p,s_2_p,i_1_p,i_2_p,r_1_p,r_2_p,B_1_p,B_2_p = f(alpha,lamb,gamma,p,s,i,r,b)
        s_1_pp = -alpha[0]*s_1_p - lamb[0]*s_1_p
        s_2_pp = alpha[0]*s_1_p - lamb[1]*s_2_p # - alpha[1]*s_2_p can use for death rate
        i_1_pp = lamb[0]*s_1_p - gamma[0]*i_1_p -alpha[0]*i_1_p
        i_2_pp = lamb[1]*s_2_p - gamma[1]*i_2_p + alpha[0]*i_1_p # - alpha[1]*i_2_p can use for death rate
        r_1_pp = gamma[0]*i_1_p - alpha[0]*r_1_p - p[0]*r_1_p
        r_2_pp = gamma[1]*i_2_p + alpha[0]*r_1_p - p[1]*r_2_p# - alpha[1]*r_2_p can use for death rate
        B_1_pp = p[0]*r_1_p
        B_2_pp = p[1]*r_2_p

        s[0] += h*(s_1_p + 0.5*h*s_1_pp)
        s[1] += h*(s_2_p + 0.5*h*s_2_pp)
        i[0] += h*(i_1_p+ 0.5*h*i_1_pp)
        i[1] += h*(i_2_p+ 0.5*h*i_2_pp)
        r[0] += h*(r_1_p + 0.5*h*r_1_pp)
        r[1] += h*(r_2_p + 0.5*h*r_2_pp)
        b[0] += h*(B_1_p + 0.5*h*B_1_pp)
        b[1] += h*(B_2_p + 0.5*h*B_2_pp)
        t += h

    return points


alpha = 0.5*np.ones(2)
lamb = np.array([0.1,0.3])
gamma = np.array([0.8,0.8])
p = np.array([0.001,0.001])
initial_s = np.array([0.3,0.64])
initial_i = np.array([0.03,0.03])
initial_r = np.zeros(2)
initial_B = np.zeros(2)
initial = np.column_stack([initial_s,initial_i,initial_r,initial_B])
points = taylor_system_11(s_alpha_SSPE,initial,alpha,lamb,gamma,p,0,15,100)
t = points[0,:]
s_1 = points[1,:]
s_2 = points[2,:]
i_1 = points[3,:]
i_2 = points[4,:]
r_1 = points[5,:]
r_2 = points[6,:]
B_1 = points[7,:]
B_2 = points[8,:]

plt.plot(t,s_1, label='Infants < 1')
plt.plot(t,s_2, label='Everyone Else')
plt.plot(t,i_1, label='Infected Infants < 1')
plt.plot(t,i_2, label='Infected Everyone Else')
plt.plot(t,r_1, label='Recovered infants < 1')
plt.plot(t,r_2, label='Recovered Everyone Else')
plt.plot(t,B_1, label='SSPE Infants < 1')
plt.plot(t,B_2, label='SSPE Everyone Else')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title(f"Second order approx \n Two age compartments alpha = {alpha} \n lambda = {lamb}\n gamma = {gamma}\n sspe_rate = {p}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/sir_alpha_SSPE_system.jpg',dpi=200)
plt.show()
plt.close()


plt.plot(t,B_1, label='SSPE Infants < 1')
plt.plot(t,B_2, label='SSPE Everyone Else')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title(f"Second order approx \n Two age compartments alpha = {alpha} \n lambda = {lamb}\n gamma = {gamma}\n sspe_rate = {p}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/sir_alpha_SSPE_system_SSPE_curves.jpg',dpi=200)
plt.show()
plt.close()
