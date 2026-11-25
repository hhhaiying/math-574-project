import numpy as np
from scenario_model import simulate, n, N

# Compare age groups
R_e, inf, sspe = simulate(0.72, 0.91)
print(f"Baseline: R_e = {R_e:.2f}, SSPE = {sspe:.1f}")
print(f"  children: {n[0]:.1%} of population, {inf[0] / inf.sum():.1%} of infections")
print(f"  infected: children {inf[0] / (n[0] * N):.1%}, older {inf[1] / (n[1] * N):.1%}")

# Vaccination scenarios 
print("\nCoverage (under 5, 5+) | R_e | child infections | SSPE")
for cov1, cov2 in [(0.72, 0.91), (0.72, 0.94), (0, 0), (0.5, 0.5)]:
    R_e, inf, sspe = simulate(cov1, cov2)
    print(f"({cov1:.0%}, {cov2:.0%}) | {R_e:.2f} | {inf[0]:,.0f} | {sspe:.1f}")

# Sensitivity to R0 
print("\nR0 | R_e | SSPE | older coverage needed for SSPE < 1")
for R0 in [12, 15, 18]:
    R_e, inf, sspe = simulate(0.72, 0.91, R0)
    needed = next(c2 for c2 in np.linspace(0.80, 0.99, 96) if simulate(0.72, c2, R0)[2] < 1)
    print(f"{R0} | {R_e:.2f} | {sspe:.1f} | {needed:.0%}")
