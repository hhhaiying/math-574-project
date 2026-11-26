import itertools
import sympy as sp
import numpy as np 
from data_processing import give_nu,give_birth,give_params_2,data_processing

#### For 2 group model analysis

beta_1,beta_2,beta_3,beta_4,gamma_1,gamma_2,nu_1,nu_2,delta = sp.symbols('beta_1 beta_2 beta_3 beta_4 gamma_1 gamma_2 nu_1 nu_2 delta', positive=True)
s_1,s_2,i_1,i_2,r_1,r_2,b,v = sp.symbols('S_1 S_2 I_1 I_2 R_1 R_2 B V')
state = sp.Matrix([s_1,s_2,i_1,i_2,r_1,r_2,b,v])
state_core = sp.Matrix([s_1,s_2,i_1,i_2])
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
F_core = sp.Matrix([s_1,s_2,i_1,i_2])
J_core = F_core.jacobian(state_core)
print(J_core)
eqs_2 = sp.solve(F_2,(s_1,s_2,i_1,i_2,r_1,r_2,b,v),dict=True)
print("Equilibrium")
sp.pprint(eqs_2)
J_2 = F_2.jacobian(state)
params_2 = give_params_2()
sym_lookup = {sym.name: sym for sym in J_2.free_symbols}
sym_params = {sym_lookup[k]: v for k, v in params_2.items() if k in sym_lookup}
print("Jacobian at Equilibrium")
J_0 = J_2.subs(sym_params).subs({s_1:0,s_2:0,i_1:0,i_2:0,r_1:0,r_2:0,b:0,v:0})



pop_1,pop_2 = data_processing()


### Gemini thinking generated code
def sample_simplex_2d(scale, n_steps=20):
    """Returns points (s, i) with s >= 0, i >= 0, s + i <= scale."""
    pts = []
    for a in range(n_steps + 1):
        for b in range(n_steps + 1 - a):
            pts.append((scale * a / n_steps, scale * b / n_steps))
    return np.array(pts)

# Build 2D simplices scaled by pop_1 and pop_2, then take their Cartesian product
tri_1 = sample_simplex_2d(pop_1, n_steps=20)  # shape (231, 2)
tri_2 = sample_simplex_2d(pop_2, n_steps=20)  # shape (231, 2)

# Cartesian product -> shape (53361, 4) ordered as (s1, i1, s2, i2)
grid_4d = np.array([
    [s_1, i_1, s_2, i_2]
    for (s_1, i_1), (s_2, i_2) in itertools.product(tri_1, tri_2)
])

# Compile symbolic expressions (with parameters already substituted) for fast evaluation
J_func = sp.lambdify((s_1, s_2, i_1, i_2), J_core.subs(params_2), modules="numpy")
JF_func = sp.lambdify((s_1, s_2, i_1, i_2), (J_core * F_core).subs(params_2), modules="numpy")

# Evaluate norms and condition numbers across all 4D grid points
max_jf_inf = 0.0
max_cond_2 = 0.0
worst_jf_pt = None

for s1_v, i1_v, s2_v, i2_v in grid_4d:
    jf_val = JF_func(s1_v, s2_v, i1_v, i2_v).flatten()
    jf_norm = np.linalg.norm(jf_val, np.inf)
    if jf_norm > max_jf_inf:
        max_jf_inf = jf_norm
        worst_jf_pt = (s1_v, s2_v, i1_v, i2_v)

    j_val = J_func(s1_v, s2_v, i1_v, i2_v)
    cond = np.linalg.cond(j_val, 2)
    if cond > max_cond_2:
        max_cond_2 = cond

print(f"Max ||J*f||_inf = {max_jf_inf:.6e} at (s1, s2, i1, i2) = {worst_jf_pt}")
print(f"Max cond_2(J_core) = {max_cond_2:.2f}")
