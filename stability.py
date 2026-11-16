import numpy as np
import matplotlib.pyplot as plt
from model import stability_system
from scipy.integrate import solve_ivp
from scipy.optimize import root
b = 0.3
d = 0.1
beta = 2
gamma = 1
nu = 0.1

def sys(t,v):
    val = stability_system(b,d,beta,gamma,nu,v[0],v[1])
    return val

R_0 = (beta/gamma)*(b/(b+nu))
print(f"R_0 Value: {R_0}")

J_P0 = np.array([
    [-b-nu,-b-(beta*b)/(b+gamma)],
    [0, (beta*b)/(b+gamma)-gamma]
])

P_star = np.array([
    gamma/beta,(b*beta-gamma*(b+nu))/(beta*(b+gamma))
])

rt = root(lambda v : stability_system(b,d,beta,gamma,nu,v[0],v[1]),x0=(0.5,0.5))

points = np.linspace(0,1,10)
ini = np.column_stack([np.cos((np.pi/2)*points),np.sin((np.pi/2)*points)])
X,Y = np.meshgrid(points,points)
U,V = stability_system(b,d,beta,gamma,nu,X,Y)
L = np.hypot(U,V)
L[L==0]=1
plt.quiver(X,Y,U/L,V/L)
for y0 in ini:
    sol = solve_ivp(sys,(0,10),y0,t_eval=np.linspace(0,10))
    x,y = sol.y
    plt.plot(x,y)


plt.plot(P_star[0],P_star[1],'o',label='Analytical Equilibrium')
plt.plot(rt.x[0],rt.x[1],'o',label='Numerical Equilibrium')
plt.title(f'Field for b={b} d={d} beta={beta} \n gamma={gamma} nu ={nu}')
plt.xlabel('s')
plt.ylabel('i')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/stability_field.jpg',dpi=200)
plt.show()
