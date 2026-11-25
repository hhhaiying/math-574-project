import numpy as np
import matplotlib.pyplot as plt
from scenario_model import simulate, n, N

extra = np.linspace(0, 150_000, 61)  # additional people vaccinated
max_children = 0.28 * n[0] * N - 52_471  # unvaccinated children aged 1-4 (infants cannot be vaccinated)
direct = [simulate(0.72 + x / (n[0] * N), 0.91)[2] if x <= max_children else np.nan for x in extra]
indirect = [simulate(0.72, 0.91 + x / (n[1] * N))[2] for x in extra]

plt.figure(figsize=(7, 4.2))
plt.plot(extra / 1000, direct, color="#2a78d6", lw=2, label="Vaccinate ages < 5")
plt.plot(extra / 1000, indirect, color="#eb6834", lw=2, label="Vaccinate ages $\geq$ 5")
plt.xlim(0, None)
plt.ylim(0, None)
plt.xlabel("Additional people vaccinated (thousands)")
plt.ylabel("Expected SSPE cases (Ages < 5)")
plt.legend(frameon=False)
end_sspe = simulate(0.72 + max_children / (n[0] * N), 0.91)[2]
plt.plot(max_children / 1000, end_sspe, "o", color="#2a78d6", ms=7)
plt.annotate(f"({max_children / 1000:.0f}k, {end_sspe:.0f})", (max_children / 1000, end_sspe),
             xytext=(8, 4), textcoords="offset points")
plt.tight_layout()
plt.savefig("figures/direct_vs_indirect.png", dpi=200)
plt.show()

R_e, inf, sspe = simulate(1.0, 0.91)
print(f"All children vaccinated: R_e = {R_e:.2f}, total infections = {inf.sum():,.0f}")
