import sympy as sp
from data_processing import give_nu,give_birth
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
error = sp.Abs(sp.simplify(J*F))
#sp.pprint(J.row)
#sp.pprint(F)

params = {
    b: give_birth(),
    beta: sp.Rational(9, 10),
    gamma: sp.Rational(1, 8),
    nu: give_nu(),
}

error_10 = error.subs({s_sym:1,i_sym:0})
error_01 = error.subs({s_sym:0,i_sym:1})
error_h = error.subs({s_sym:sp.Rational(1,2),i_sym:sp.Rational(1,2)})
error_p0 = error.subs({s_sym:eqs[0][s_sym],i_sym:0})
error_pstar = error.subs({s_sym:eqs[1][s_sym],i_sym:eqs[1][i_sym]})
sp.pprint(params)

R_0 = (beta/gamma)*(b/(b+nu))
#print("nu")
#sp.pprint(nu.subs(params))
#print("R_0")
#sp.pprint(R_0.subs(params))
