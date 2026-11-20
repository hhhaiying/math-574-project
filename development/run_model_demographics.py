import numpy as np
import matplotlib.pyplot as plt
from model import SSPEV


def taylor_system_13(f,initial,t0,t1,n):
    """First order forward iteration"""
    t = t0
    h = (t1 - t0) / n
    points = np.zeros([8,n])

    s = initial[:,0]
    i = initial[:,1]
    r = initial[:,2]
    b_0 = initial[:,3]

    for k in range(n):
        points[0,k] = t
        points[1,k] = s[0]
        points[2,k] = s[1]
        points[3,k] = i[0]
        points[4,k] = i[1]
        points[5,k] = r[0]
        points[6,k] = r[1]
        points[7,k]= b_0[0]
        s_1_p,s_2_p,i_1_p,i_2_p,r_1_p,r_2_p,b_0_p = f(b,alpha,lamb,gamma,delta,nu,s,i,r)

        s[0] += h*s_1_p
        s[1] += h*s_2_p
        i[0] += h*i_1_p
        i[1] += h*i_2_p
        r[0] += h*r_1_p
        r[1] += h*r_2_p
        b_0 += h*b_0_p
        t += h

    return points


b = 0.2
alpha = 0.5*np.ones(2)
lamb = np.array([[0.2,0.3],[0.3,0.2]])
lamb = lamb.flatten()
gamma = np.array([0.8,0.8])
delta = np.random.uniform(0,1)
delta = np.array([delta,1-delta])
nu = 0.4
initial_s = np.array([0.25,0.5])
initial_i = np.array([0.25,0.])
initial_r = np.zeros(2)
initial_b = np.zeros(2)
initial = np.column_stack([initial_s,initial_i,initial_r,initial_b])
points = taylor_system_13(SSPEV,initial,0,10,100)
t = points[0,:]
s_1 = points[1,:]
s_2 = points[2,:]
i_1 = points[3,:]
i_2 = points[4,:]
r_1 = points[5,:]
r_2 = points[6,:]
b_1 = points[7,:]


plt.figure(figsize=(13,8))
plt.plot(t,s_1, label='Children < 5')
plt.plot(t,s_2, label='Everyone Else')
plt.plot(t,i_1, label='Infected Children < 5')
plt.plot(t,i_2, label='Infected Everyone Else')
plt.plot(t,r_1, label='Recovered Children < 5')
plt.plot(t,r_2, label='Recovered Everyone Else')
plt.plot(t,b_1, label='SSPE Incidence')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title(f"First order approx \n Two age compartments alpha = {alpha} \n lambda = {lamb}\n gamma = {gamma}\n sspe_rate = {np.round(delta,3)} Vaccination rate {nu}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/SSPEV_system.jpg',dpi=200)
plt.show()
plt.close()


fig, ax = plt.subplots(1,2,figsize=(13,8))
ax[0].plot(t,b_1, label='SSPE Incidence Children < 5')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Percentage')
ax[0].set_title(f"SSPE Incidence\n sspe_rate = {np.round(delta,3)} vaccination_rate {nu}")
ax[0].grid()
ax[0].legend()
ax[1].plot(t,r_1, label='Recovered Children < 5')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Percentage')
ax[1].grid()
ax[1].set_title(f'Recovered population over time rate = {nu}')
plt.legend()
plt.tight_layout()
fig.suptitle(f"First order approx \n Two age compartments alpha = {alpha} \n lambda = {lamb}\n gamma = {gamma}\n sspe_rate = {np.round(delta,3)} vaccination_rate {nu}")
fig.tight_layout()
plt.savefig(f'figures/SSPEV_curves.jpg',dpi=200)
plt.show()
plt.close()
