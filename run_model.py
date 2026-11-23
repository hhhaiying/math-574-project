import numpy as np
import matplotlib.pyplot as plt
from model import SSPE
from data_processing import data_processing,give_I, give_nu

def taylor_system_14(f,initial,t0,t1,n):
    """First order forward iteration"""
    t = t0
    h = (t1 - t0) / n
    points = np.zeros([9,n])

    s = initial[:,0]
    i = initial[:,1]
    r = initial[:,2]
    b_0 = initial[:,3]
    v_0 = initial[:,4]

    for k in range(n):
        points[0,k] = t
        points[1,k] = s[0]
        points[2,k] = s[1]
        points[3,k] = i[0]
        points[4,k] = i[1]
        points[5,k] = r[0]
        points[6,k] = r[1]
        points[7,k] = b_0[0]
        points[8,k] = v_0[0]
        s_1_p,s_2_p,i_1_p,i_2_p,r_1_p,r_2_p,b_0_p,v_0_p = f(beta,gamma,delta,nu,s,i,r)

        s[0] += h*s_1_p
        s[1] += h*s_2_p
        i[0] += h*i_1_p
        i[1] += h*i_2_p
        r[0] += h*r_1_p
        r[1] += h*r_2_p
        b_0 += h*b_0_p
        v_0 += h*v_0_p
        t += h

    return points


beta = np.array([[1.7280,6.686],[0.3688,9.9163]])
beta = beta.flatten()
gamma = np.array([1/8,1/8])
delta = (1 / 1336) * (1 / 8)
delta = np.array([delta,0])
nu_ = give_nu()
pop_1,pop_2 = data_processing()
vpop_1, vpop_2 = 0.72*pop_1,0.91*pop_2
infpop_1, infpop_2 = give_I()
suspop_1,suspop_2 = pop_1 - infpop_1 - vpop_1, pop_2 - infpop_2 - vpop_2
nu = np.array([0.9*nu_/suspop_1,0.1*nu_/suspop_2])
print(nu)
initial_s = np.array([suspop_1,suspop_2])
initial_i = np.array([infpop_1,infpop_2])
initial_r = np.zeros(2)
initial_b = np.zeros(2)
initial_v = np.array([vpop_1,vpop_2])
initial = np.column_stack([initial_s,initial_i,initial_r,initial_b,initial_v])
initial_d = {'s_0' : initial_s, 'i_0' : initial_i, 'r_0': initial_r, 'b_0' : initial_b, 'v_0' : initial_v}
print('Initial Conds', initial_d)
points = taylor_system_14(SSPE,initial,0,60,360)
t = points[0,:]
s_1 = points[1,:]
s_2 = points[2,:]
i_1 = points[3,:]
i_2 = points[4,:]
r_1 = points[5,:]
r_2 = points[6,:]
b_1 = points[7,:]
v_1 = points[8,:]

plt.figure(figsize=(13,8))
plt.plot(t,s_1, label='Susceptible Children < 5')
plt.plot(t,s_2, label='Susceptible Everyone Else')
plt.plot(t,i_1, label='Infected Children < 5')
plt.plot(t,i_2, label='Infected Everyone Else')
plt.plot(t,r_1, label='Recovered Children < 5')
plt.plot(t,r_2, label='Recovered Everyone Else')
plt.plot(t,b_1, label='SSPE Incidence')
#plt.plot(t,v_1, label='Vaccinated population')
plt.xlabel('Time (days)')
plt.ylabel('Percentage')
plt.title(rf"First order approx {'\n'} $\beta$ = {beta} {'\n'} $\gamma$ = {gamma} {'\n'} sspe_rate = {np.round(delta[0],3)} Vaccination rate {np.round(nu,4)}/day")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/SSPE_system.jpg',dpi=200)
plt.show()
plt.close()


fig, ax = plt.subplots(2,2,figsize=(13,8))
ax[0,0].plot(t,b_1, label='SSPE Incidence Children < 5')
ax[0,0].set_xlabel('Time (days)')
ax[0,0].set_ylabel('Percentage')
ax[0,0].set_title(f"SSPE Incidence \n sspe_rate = {np.round(delta[0],4)} vaccination_rate {np.round(nu,4)} / day")
ax[0,0].grid()
ax[0,0].legend()
ax[0,1].plot(t,r_1, label='Recovered Children < 5')
ax[0,1].plot(t,r_2, label='Recovered People > 5')
ax[0,1].set_xlabel('Time (days)')
ax[0,1].set_ylabel('Percentage')
ax[0,1].legend()
ax[0,1].grid()
ax[0,1].set_title(f'Recovered population over time rate')
ax[1,1].plot(t,v_1, label='Vaccinated Population')
ax[1,1].set_xlabel('Time (days)')
ax[1,1].set_ylabel('Percentage')
ax[1,1].legend()
ax[1,1].grid()
ax[1,1].set_title(rf'Vaccinated population over time rate = $\nu$ {nu}')
ax[1,0].set_visible(False)
plt.legend()
plt.tight_layout()
fig.suptitle(rf"First order approx {'\n'} $\beta$ = {beta} {'\n'} $\gamma$ = {gamma} {'\n'} sspe_rate = {np.round(delta[0],4)} vaccination_rate {np.round(nu,4)}")
fig.tight_layout()
plt.savefig(f'figures/SSPE_curves.jpg',dpi=200)
plt.show()
plt.close()
