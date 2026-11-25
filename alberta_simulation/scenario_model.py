"""
Two-age-group measles model used for the vaccination scenarios
Group 1 = children under five, group 2 = aged five and older
Same data as run_model.py; the contact matrix is scaled so that R0 = 18
and SSPE cases = (1/1367) x infections in children under five
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

pop = pd.read_csv("data/population.csv")
N = pop["Population"].sum()
n1 = pop["Population"][0:2].sum() / N  # children under five
n = np.array([n1, 1 - n1])

doses = pd.read_csv("data/alberta_measles_doses_2025.csv")
nu_total = doses["doses_administered"].mean() / 7 / N

gamma = 1 / 8
contacts = np.array([[1.728, 6.686],
                     [0.3688, 9.9163]])
sspe_risk = 1 / 1367
    

def simulate(cov1, cov2, R0=18):
    q = R0 * gamma / max(abs(np.linalg.eigvals(contacts)))
    beta = q * contacts

    v = np.array([cov1, cov2]) * n
    i0 = np.array([0, 10 / N]) if cov2 < 1 else np.array([0, 0])
    s0 = n - v - i0
    nu = np.array([0.9, 0.1]) * nu_total / np.maximum(s0, 1e-12)

    def model(t, x):
        s, i = x[0:2], x[2:4]
        new = beta @ (i / n) * s  # new infections
        return np.concatenate([-new - nu * s, new - gamma * i, new])

    sol = solve_ivp(model, (0, 365), np.concatenate([s0, i0, [0, 0]]),
                    rtol=1e-8, atol=1e-14)
    infections = sol.y[4:6, -1] * N
    R_e = max(abs(np.linalg.eigvals(beta * (s0[:, None] / n) / gamma)))
    return R_e, infections, sspe_risk * infections[0]
