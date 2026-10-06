import numpy as np
import matplotlib.pyplot as plt
from models import s_alpha



def taylor_system_10(f,initial,alpha,lamb,gamma,t0,t1,n):
    """Naive 2 S compartment connection. Forward 2nd order"""
    t = t0
    h = (t1 - t0) / n
    points = np.zeros([10,n])

    s = initial[:,0]
    i = initial[:,1]
    r = initial[:,2]

    for k in range(n):
        points[0,k] = t
        points[1,k] = s[0]
        points[2,k] = s[1]
        points[3,k] = s[2]
        points[4,k] = i[0]
        points[5,k] = i[1]
        points[6,k] = i[2]
        points[7,k] = r[0]
        points[8,k] = r[1]
        points[9,k] = r[2]
        s_1_p,s_2_p,s_3_p,i_1_p,i_2_p,i_3_p,r_1_p,r_2_p,r_3_p = f(alpha,lamb,gamma,s,i,r)
        s_1_pp = -alpha[0]*s_1_p - lamb[0]*s_1_p
        s_2_pp = alpha[0]*s_1_p - lamb[1]*s_2_p - alpha[1]*s_2_p
        s_3_pp = alpha[1]*s_2_p - lamb[2]*s_3_p
        t += h
        s[0] += h*(s_1_p + 0.5*s_1_pp)
        s[1] += h*(s_2_p + 0.5*s_2_pp)
        s[2] += h*(s_3_p + 0.5*s_3_pp)
        i[0] += h*(i_1_p)
        i[1] += h*(i_2_p)
        i[2] += h*(i_3_p)
        r[0] += h*r_1_p
        r[1] += h*r_2_p
        r[2] += h*r_3_p
    return points


alpha = 0.5*np.ones(3)
lamb = np.array([0.1,0.3,0.7])
gamma = np.array([0.8,0.8,0.2])
initial_s = np.array([0.1,0.4,0.5])
initial_i = np.array([0.03,0.03,0.03])
initial_r = np.zeros(3)
initial = np.column_stack([initial_s,initial_i,initial_r])
points = taylor_system_10(s_alpha,initial,alpha,lamb,gamma,0,15,100)
t = points[0,:]
s_1 = points[1,:]
s_2 = points[2,:]
s_3 = points[3,:]
i_1 = points[4,:]
i_2 = points[5,:]
i_3 = points[6,:]
r_1 = points[7,:]
r_2 = points[8,:]
r_3 = points[9,:]

plt.plot(t,s_1, label='population 1')
plt.plot(t,s_2, label='population 2')
plt.plot(t,s_3, label='population 3')
plt.plot(t,i_1, label='infected population 1')
plt.plot(t,i_2, label='infected population 2')
plt.plot(t,i_3, label='infected population 3')
plt.plot(t,r_1, label='recovered population 3')
plt.plot(t,r_2, label='recovered population 3')
plt.plot(t,r_3, label='recovered population 3')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title(f"Second order approx \n Three age compartments alpha = {alpha} \n lambda = {lamb}\n gamma = {gamma}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/sir_alpha_system.jpg',dpi=200)
plt.show()
plt.close()
