import numpy as np
import matplotlib.pyplot as plt
from numerics import rk_system
from models import sir_v,sirs_bv


# susceptible proportion
lamb = 0.9
# infected proportion
gamma = 0.1
# initial rates
initial = np.array([0.9,0.1,0])

points,t = rk_system(lambda t,x : sir_v(0.9,0.1,t,x),initial,0,100,100)

s = points[:,0]
i = points[:,1]
r = points[:,2]

plt.plot(t,s, label='susceptible')
plt.plot(t,i, label='infected')
plt.plot(t,r, label='recovered')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title(f"RK4 approx lambda {lamb} gamma {gamma}")
plt.legend()
plt.savefig('figures/sir_system_rk.jpg',dpi=200)
plt.show()
plt.close()


##### 
### Example of sir with logistic birth and death
## birth rate
b = 1
## death rate
d = 1 - b
## delta rate of becoming suceptible again
lamb = 0.75
gamma = 0.25
delta = 0.5
delta = 0.2
K = 1.5


points,t = rk_system(lambda t,x :sirs_bv(b,d,K,delta,lamb,gamma,t,x),np.array([0.9,0.1,0]),0.,10.,100)
#                t = np.linspace(0,10,100)
s = points[:,0]
i = points[:,1]
r = points[:,2]
plt.plot(t,s, label='susceptible')
plt.plot(t,i, label='infected')
plt.plot(t,r, label='recovered')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title(f"RK4 approx \n Logistic Growth lambda {lamb} gamma {gamma} delta {delta} \n birth rate {np.round(b,2)} and death rate {np.round(d,2)} capacity K {K}")
plt.legend()
plt.savefig(f'figures/sirs_bvrk4_system_{delta}_{gamma}_{lamb}.jpg',dpi=200)
plt.tight_layout()
plt.show()
plt.close()



#deltas = np.linspace(0.25,0.75,3)
#gammas = np.linspace(0.25,0.75,3)
#lambdas = np.linspace(0.25,0.75,3)
#bs = np.linspace(0,1,4)
#for b in bs:
#    d = 1-b
#    for delta in deltas:
#        for gamma in gammas:
#            for lamb in lambdas:
#                points,t = rk_system(lambda t,x :sirs_bv(b,d,K,delta,lamb,gamma,t,x),np.array([0.9,0.1,0]),0.,10.,100)
##                t = np.linspace(0,10,100)
#                s = points[:,0]
#                i = points[:,1]
#                r = points[:,2]
#                plt.plot(t,s, label='susceptible')
#                plt.plot(t,i, label='infected')
#                plt.plot(t,r, label='recovered')
#                plt.xlabel('Time')
#                plt.ylabel('Percentage')
#                plt.title(f"RK4 approx \n Logistic Growth lambda {lamb} gamma {gamma} delta {delta} \n birth rate {np.round(b,2)} and death rate {np.round(d,2)} capacity K {K}")
#                plt.legend()
##                plt.savefig(f'figures/sirs_bvrk4_system_{delta}_{gamma}_{lamb}.jpg',dpi=200)
#                plt.tight_layout()
#                plt.show()
#                plt.close()
 
