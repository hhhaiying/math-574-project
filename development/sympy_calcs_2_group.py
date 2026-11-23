import sympy as sp
import numpy as np 
from data_processing import give_nu,give_birth,give_params_2

#### For 2 group model analysis

beta_1,beta_2,beta_3,beta_4,gamma_1,gamma_2,nu_1,nu_2,delta = sp.symbols('beta_1 beta_2 beta_3 beta_4 gamma_1 gamma_2 nu_1 nu_2 delta', positive=True)
s_1,s_2,i_1,i_2,r_1,r_2,b,v = sp.symbols('S_1 S_2 I_1 I_2 R_1 R_2 B V')
state = sp.Matrix([s_1,s_2,i_1,i_2,r_1,r_2,b,v])
params_2 = give_params_2
print(params_2())

s_1_p = -(beta_1*i_1+beta_2*i_2)*s_1 - nu_1*s_1
s_2_p =  -beta_3*s_2*i_1 - beta_4*s_2*i_2 - nu_2*s_2
i_1_p = (beta_1*i_1+beta_2*i_2)*s_1 - gamma_1*i_1
i_2_p = beta_3*s_2*i_1 + beta_4*s_2*i_2  - gamma_2*i_2 
r_1_p = gamma_1*i_1  - delta*r_1
r_2_p = gamma_2*i_2 
b_0_p = delta*r_1 
v_0_p = nu_1*s_1 + nu_2*s_2

F_2 = sp.Matrix([s_1_p,s_2_p,i_1_p,i_2_p,r_1_p,r_2_p,b_0_p,v_0_p])


eqs_2 = sp.solve(F_2,(s_1,s_2,i_1,i_2,r_1,r_2,b,v),dict=True)
print("Equilibrium")
sp.pprint(eqs_2)
J_2 = F_2.jacobian(state)
params_2 = give_params_2()
sym_lookup = {sym.name: sym for sym in J_2.free_symbols}
sym_params = {sym_lookup[k]: v for k, v in params_2.items() if k in sym_lookup}
print("Jacobian at Equilibrium")
J_0 = J_2.subs(sym_params).subs({s_1:0,s_2:0,i_1:0,i_2:0,r_1:0,r_2:0,b:0,v:0})

