import numpy as np
import matplotlib.pyplot as plt
from models import sirs_bv

### parameters between 0 and 1
test = np.linspace(0,1,25)
x = np.meshgrid(test,test,test)

lamb = 0.75
gamma = 0.25
delta = 0.5
b = 1
d = 1 -b
K = 1.5

points = sirs_bv(b,d,K,delta,lamb,gamma,test,x)
u = points[:,0]
v = points[:,1]
w = points[:,2]

ax = plt.figure().add_subplot(projection='3d')
ax.quiver(test,test,test,u,v,w ,length=0.1,normalize=True)
ax.set_title(f"Vector field for sirs with birth {np.round(b,2)} death {np.round(d,2)} delta {np.round(delta,2)}\n lambda {np.round(lamb,2)} gamma {np.round(gamma,2)}")

#                ax.scatter(y[0,:],y[1,:],y[2,:])
ax.set_xlabel("s")
ax.set_ylabel("i")
ax.set_zlabel("r")
plt.tight_layout()
#                plt.savefig(f'figures/vector_field_sirs_b_{delta}_{gamma}_{lamb}.jpg')
plt.show()

#deltas = np.linspace(0.25,0.75,2)
#gammas = np.linspace(0.25,0.75,2)
#lambdas = np.linspace(0.25,0.75,2)
#bs = np.linspace(0,1,4)
#b = bs[1]
#d = 1 - b
#delta = deltas[1]
#lamb = lambdas[1]
#gamma = gammas[1]
#K = 1.5
#
#for b in bs:
#    for delta in deltas:
#        for gamma in gammas:
#            for lamb in lambdas:
#                d = 1 - b
#                points = sirs_bv(b,d,K,delta,lamb,gamma,test,x)
#                u = points[:,0]
#                v = points[:,1]
#                w = points[:,2]
#
#                ax = plt.figure().add_subplot(projection='3d')
#                ax.quiver(test,test,test,u,v,w ,length=0.1,normalize=True)
#                ax.set_title(f"Vector field for sirs with birth {np.round(b,2)} death {np.round(d,2)} delta {np.round(delta,2)}\n lambda {np.round(lamb,2)} gamma {np.round(gamma,2)}")
#
#                #                ax.scatter(y[0,:],y[1,:],y[2,:])
#                ax.set_xlabel("s")
#                ax.set_ylabel("i")
#                ax.set_zlabel("r")
#                plt.tight_layout()
##                plt.savefig(f'figures/vector_field_sirs_b_{delta}_{gamma}_{lamb}.jpg')
#                plt.show()
#
