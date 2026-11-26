import numpy as np
import sympy as sp
from data_processing import data_processing, give_nu,give_birth,give_params_2

# Independent variable and positive parameters 
t = sp.symbols('t', real=True, nonnegative=True)
h = sp.symbols('h',real=True,positive=True)
b, beta, gamma, nu = sp.symbols('b beta gamma nu', positive=True)

# State variables as functions of time
s = sp.Function('s')(t)
i = sp.Function('i')(t)

# Reduced 2D ODEs
ds_dt = b*(1-s-i)-beta*s*i-nu*s
di_dt = beta*s*i - gamma*i

ode_S = sp.Eq(sp.diff(s, t), ds_dt)
ode_I = sp.Eq(sp.diff(i, t), di_dt)


f = sp.Matrix([

     b*(1-s-i)-beta*s*i-nu*s,
     beta*s*i - gamma*i
])


y = sp.Matrix([s,i])
# Display system
#print("First Order Local Truncation Error of f")
s_sym, i_sym = sp.symbols('s i',nonnegative=True)
F = f.subs({s:s_sym,i:i_sym})
y_sym = sp.Matrix([s_sym,i_sym])
J = F.jacobian(y_sym)
#sp.pprint(sp.simplify((h**2/2)*J*f))
#error = sp.Abs((h**2/2)*J*F)

### Equilibria
eqs = sp.solve(F,(s_sym,i_sym),dict=True)
#sp.pprint(sp.simplify(eqs[1][s_sym]))
#error = sp.Abs(sp.simplify(J*F))
error = sp.Abs(sp.simplify(J))
#error = sp.Abs(sp.simplify(F))
#sp.pprint(J.row)
#sp.pprint(F)

params = {
    b: give_birth(),
    beta: sp.Rational(9, 10),
    gamma: sp.Rational(1, 8),
    nu: give_nu(),
}



def empirical_r0():
    R_09 = (beta/gamma)*(b/(b+nu))
    R_0 = ((18*b/(b+nu)).subs(params))
    return R_0,R_09

funcs = [J,F,J*F]
labels = ["J", "F", "J*F"]
for func in funcs: 
    i = funcs.index(func)
    error = sp.simplify(func)
    error_10 = error.subs({s_sym:1,i_sym:0} | params)
    error_01 = error.subs({s_sym:0,i_sym:1} | params)
    error_h = error.subs({s_sym:sp.Rational(1,2),i_sym:sp.Rational(1,2)} | params)
    error_p0 = error.subs({s_sym:(eqs[0][s_sym]).subs(params),i_sym:0} | params)
    error_pstar = error.subs({s_sym:(eqs[1][s_sym]).subs(params),i_sym:(eqs[1][i_sym]).subs(params)} | params)
    print(f"FUNC {labels[i]} 1 Norm at {1,0}", error_10.norm(1))
    print(f"FUNC {labels[i]} Infty Norm at {1,0}", error_10.norm(sp.oo))
    print(f"FUNC {labels[i]} 1 Norm  at {0,1}", error_01.norm(1))
    print(f"FUNC {labels[i]} Infty Norm  at {0,1}", error_01.norm(sp.oo))
    print(f"FUNC {labels[i]} 1 Norm  at {0.5,0.5}", error_h.norm(1))
    print(f"FUNC {labels[i]} Infty Norm  at {0.5,0.5}", error_h.norm(sp.oo))
    print(f"FUNC {labels[i]} 1 Norm  at p_0", error_p0.norm(1))
    print(f"FUNC {labels[i]} Infty Norm  at p_0", error_p0.norm(sp.oo))
    print(f"FUNC {labels[i]} 1 Norm  at p_star", error_pstar.norm(1))
    print(f"FUNC {labels[i]} Infty Norm  at p_star", error_pstar.norm(sp.oo))


### for the condition number and LTE
h = 60 / 360
error_jf = sp.simplify(J*F)
error_jf_pstar = error_jf.subs({s_sym:(eqs[1][s_sym]).subs(params),i_sym:(eqs[1][i_sym]).subs(params)} | params)
error_jf_p0 = error_jf.subs({s_sym:(eqs[0][s_sym]).subs(params),i_sym:(eqs[0][i_sym]).subs(params)} | params)
jf_star = sp.matrix2numpy(error_jf_pstar,dtype=float)
jf_0 = sp.matrix2numpy(error_jf_p0,dtype=float)
jf_10 = sp.matrix2numpy(error_jf.subs({s_sym:1,i_sym:0} | params),dtype=float)
jf_01 = sp.matrix2numpy(error_jf.subs({s_sym:0,i_sym:1} | params),dtype=float)
jf_h = sp.matrix2numpy(error_jf.subs({s_sym:0.5,i_sym:0.5} | params),dtype=float)


error = sp.simplify(J)
error_pstar = error.subs({s_sym:(eqs[1][s_sym]).subs(params),i_sym:(eqs[1][i_sym]).subs(params)} | params)
error_p0 = error.subs({s_sym:(eqs[0][s_sym]).subs(params),i_sym:0} | params)
J_star = sp.matrix2numpy(error_pstar,dtype=float)
J_0 = sp.matrix2numpy(error_p0,dtype=float)
J_10 = sp.matrix2numpy(J.subs({s_sym:1,i_sym:0} | params),dtype=float)
J_01 = sp.matrix2numpy(J.subs({s_sym:0,i_sym:1} | params),dtype=float)
J_h = sp.matrix2numpy(J.subs({s_sym:0.5,i_sym:0.5} | params),dtype=float)
print("Condition Number and LTE")
print("J_10", np.linalg.cond(J_10),0.5*h**2*np.linalg.norm(jf_10,np.inf))
print("J_01", np.linalg.cond(J_01),0.5*h**2*np.linalg.norm(jf_01,np.inf))
print("J_h", np.linalg.cond(J_h),0.5*h**2*np.linalg.norm(jf_h,np.inf))
print("J_star", np.linalg.cond(J_star), 0.5*h**2*np.linalg.norm(jf_star,np.inf))
print("J_0", np.linalg.cond(J_star), 0.5*h**2*np.linalg.norm(jf_star,np.inf))

### for the LTE

#sp.pprint(R_0)
#print("nu")
#sp.pprint(nu.subs(params))
#print("R_0")
#sp.pprint(R_0.subs(params))


