import numpy as np
import maptlotlib.pyplot as plt


b = 0.3
d = 0.1
beta = 2
gamma = 1
nu = 0.1

def indicator(beta):
  return np.where(beta<1,0,1)
def r_0(b,beta,gamma,nu):
  r = (beta/gamma)*(b/(b+nu))
  return r
beta = np.linspace(0,5,200)
R_0 = r_0(b,beta,gamma,nu)
mask = (R_0 >1) 
i_star = lambda x : (b+delta)*(1-1/x)/(b+gamma+delta)
plt.plot(R_0[mask],indicator(R_0)[mask]*i_star(R_0)[mask],c='blue',label=r'$P^\star$')
plt.plot(R_0[R_0>=2],indicator(R_0)[R_0>=2]*i_star(R_0)[R_0>=2],c='blue',linestyle='--',dashes=(3,2))
plt.plot(R_0[R_0<1],i_star(R_0)[R_0<1]*indicator(R_0)[R_0<1],c='black',label=r'$P_0$')
plt.hlines(y=0,xmin=1,xmax=np.max(R_0),color='black',linestyles='dashed')
plt.scatter([1.],[0.],facecolors='none',linewidths=2,s=80,zorder=4,edgecolors="tab:red",label = r"$R_0=1$")
plt.scatter([2+delta/(b+gamma)],[i_star(2+delta/(b+gamma))],facecolors='none',linewidths=2,s=80,zorder=4,edgecolors="tab:purple",label = r"$R_0=2+\frac{\delta}{b+\gamma}$")
plt.title(rf"Bifurcation diagram for {'\n'} $b$ = {b}  $\beta$ = {gamma}, $\gamma = $ {delta} $\nu = {nu}$")
plt.ylabel(r'$I^\star$')
plt.xlabel(r'$R_0$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
