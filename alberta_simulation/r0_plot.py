import numpy as np
import matplotlib.pyplot as plt
from scenario_model import simulate

cov2_list = np.linspace(0.80, 0.98, 73)
colors = {12: "#1baf7a", 15: "#eb6834", 18: "#2a78d6"}

plt.figure(figsize=(7, 4.2))
for R0, color in colors.items():
    sspe_list = [simulate(0.72, c2, R0)[2] for c2 in cov2_list]
    plt.plot(cov2_list * 100, sspe_list, color=color, lw=2, label=f"$R_0$ = {R0}")
plt.axvline(91, color="gray", ls="--", lw=1)
plt.text(91.15, plt.ylim()[1] * 0.93, "current 91%", color="gray", fontsize=9)
plt.xlabel("Vaccination coverage (Ages $\geq$ 5, %)")
plt.ylabel("Expected SSPE cases (Ages < 5)")
plt.legend(frameon=False)
plt.xlim(80, 98)
plt.ylim(0, None)
plt.tight_layout()
plt.savefig("figures/r0_sensitivity.png", dpi=200)
plt.show()

