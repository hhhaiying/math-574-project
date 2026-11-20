import sympy as sp
from data_processing import give_nu
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
error = sp.Abs(J)
#sp.pprint(J.row)
#sp.pprint(F)
error_10 = error.subs({s_sym:1,i_sym:0})
error_01 = error.subs({s_sym:0,i_sym:1})
error_11 = error.subs({s_sym:1,i_sym:1})
error_h = error.subs({s_sym:0.5,i_sym:0.5})
error_p0 = error.subs({s_sym:eqs[0][s_sym],i_sym:0})
error_pstar = error.subs({s_sym:eqs[1][s_sym],i_sym:eqs[1][i_sym]})

new = give_nu([0.72,0.91])
sp.pprint(error_10.norm(1).subs({b:1,beta:0.9,gamma:1/8,nu:new}))
sp.pprint(error_10.norm(sp.oo).subs({b:1,beta:0.9,gamma:1/8,nu:new}))
sp.pprint(error_10.norm(2).subs({b:1,beta:0.9,gamma:1/8,nu:new}))
sp.pprint(error_p0.norm(1))
sp.pprint(error_pstar.norm(1))
sp.pprint(error_h.norm(1))
#sp.pprint(sp.Max([sum(error.row(r)) for r in range(error.rows)])
#sp.pprint(error_p0.norm(1))
