import numpy as np
import matplotlib.pyplot as plt
from scenario_model import simulate

cov2_list = np.linspace(0.88, 0.98, 51)
sspe_list = np.array([simulate(0.72, c2)[2] for c2 in cov2_list])

now = simulate(0.72, 0.91)[2]
k = np.argmax(sspe_list < 1)
below1 = np.interp(1, [sspe_list[k], sspe_list[k - 1]], [cov2_list[k], cov2_list[k - 1]]) * 100

x0, y0 = 88, 0

plt.figure(figsize=(7, 4.2))
plt.plot(cov2_list * 100, sspe_list, color="#2a78d6", lw=2)

# current coverage
plt.plot([91, 91], [y0, now], color="gray", ls="--", lw=1)
plt.plot([x0, 91], [now, now], color="gray", ls="--", lw=1)
plt.plot(91, now, "o", color="#2a78d6", ms=7)
plt.annotate(f"(91%, {now:.0f})", (91, now), xytext=(8, 4), textcoords="offset points")

# SSPE = 1
plt.plot([x0, below1], [1, 1], color="gray", ls="--", lw=1)
plt.plot([below1, below1], [y0, 1], color="gray", ls="--", lw=1)
plt.plot(below1, 1, "o", color="#2a78d6", ms=7)
plt.annotate(f"({below1:.1f}%, 1)", (below1, 1), xytext=(8, 6), textcoords="offset points")

plt.xlim(x0, 98)
plt.ylim(y0, None)
plt.xlabel(r"Vaccination coverage (Ages $\geq$ 5, %)")
plt.ylabel("Expected SSPE cases (Ages < 5)")
plt.tight_layout()
plt.savefig("figures/coverage.png", dpi=200)
plt.show()