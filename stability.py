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

P_0 = np.array([
    b/(b+nu),0
])

P_star = np.array([
    gamma/beta,(b*beta-gamma*(b+nu))/(beta*(b+gamma))
])

rt = root(lambda v : stability_system(b,d,beta,gamma,nu,v[0],v[1]),x0=(0.5,0.5))
#print("Is the root near P_0?", np.allclose(rt.x,P_0))
#print("Is the root near P_star?",np.allclose(rt.x,P_star))
points = np.linspace(0,1,10)
ini = np.column_stack([np.cos((np.pi/2)*points),np.sin((np.pi/2)*points)])
fig, ax = plt.subplots(2)
X,Y = np.meshgrid(points,points)
U,V = stability_system(b,d,beta,gamma,nu,X,Y)
L = np.hypot(U,V)
L[L==0]=1
ax[0].quiver(X,Y,0.1*(U/L),0.1*(V/L))
for y0 in ini:
    sol = solve_ivp(sys,(0,10),y0,t_eval=np.linspace(0,10))
    x,y = sol.y
    ax[0].plot(x,y)

ax[0].plot(P_0[0],P_0[1],'o',label="Disease free")
ax[0].plot(P_star[0],P_star[1],'o',label='Analytical Equilibrium')
ax[0].plot(rt.x[0],rt.x[1],'o',label='Numerical Equilibrium')
ax[0].set_title(rf'Field for $R_0$ = {np.round(R_0,2)} {'\n'} b={b} d={d} $\beta$={beta} $\gamma$={gamma} $\nu$ ={nu}')
ax[0].set_xlabel('s')
ax[0].set_ylabel('i')
ax[0].legend()


b = 0.3
d = 0.1
beta = 1
gamma = 2
nu = 0.1

R_0 = (beta/gamma)*(b/(b+nu))
print(f"R_0 Value: {R_0}")
J_P0 = np.array([
    [-b-nu,-b-(beta*b)/(b+gamma)],
    [0, (beta*b)/(b+gamma)-gamma]
])

P_0 = np.array([
    b/(b+nu),0
])

P_star = np.array([
    gamma/beta,(b*beta-gamma*(b+nu))/(beta*(b+gamma))
])

X,Y = np.meshgrid(points,points)
U,V = stability_system(b,d,beta,gamma,nu,X,Y)
L = np.hypot(U,V)
L[L==0]=1
ax[1].quiver(X,Y,0.1*(U/L),0.1*(V/L))
for y0 in ini:
    sol = solve_ivp(sys,(0,10),y0,t_eval=np.linspace(0,10))
    x,y = sol.y
    ax[1].plot(x,y)
rt = root(lambda v : stability_system(b,d,beta,gamma,nu,v[0],v[1]),x0=(0.5,0.))
ax[1].plot(P_0[0],P_0[1],'o',label="Disease free")
ax[1].plot(rt.x[0],rt.x[1],'o',label='Numerical Equilibrium')
ax[1].set_title(rf'Field for $R_0$ = {np.round(R_0,2)} {'\n'} b={b} d={d} $\beta=${beta} $\gamma=${gamma} $\nu =${nu}')
ax[1].set_xlabel('s')
ax[1].set_ylabel('i')
ax[1].legend()
plt.tight_layout()
plt.savefig(f'figures/stability_field.jpg',dpi=200)
plt.show()
