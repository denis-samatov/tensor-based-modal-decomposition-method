"""Graphical abstract (ACG: >= 1328 x 531 px, readable at 13 x 5 cm), drawn from outputs/key_numbers.json.
Output: figures/graphical_abstract.pdf and .png (2656 x 1062 px)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bss import FIG, OUT  # noqa: E402

K = json.loads((OUT / "key_numbers.json").read_text())
def k(label, s, N):
    return K[f"P2|{label}|{s}|{N}|rmse_p|mean"]
bars = [("No measurements\n(ensemble mean)", k("No measurements (prior)", "wells-joint", 30), "#6B6B6B"),
        ("Prior + IDW\n(30 wells)", k("Prior + IDW residual", "wells-joint", 30), "#AA3377"),
        ("TBMD (48,48,2) $\\ell_1$\n(30 wells)", k("TBMD (48,48,2), l1 [original]", "wells-joint", 30), "#D55E00"),
        ("POD-E $\\ell_1$\n(30 wells)", k("POD-E, l1", "wells-joint", 30), "#009E73"),
        ("POD-E LS\n(30 QR channels)", k("POD-E, LS", "grid", 30), "#0072B2")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(8.853, 3.541), dpi=300)
ax0 = fig.add_axes([0.02, 0.04, 0.33, 0.92]); ax0.axis("off")
ax0.text(0, 0.97, "Brugge: 10 control scenarios", fontsize=12, weight="bold", va="top")
lines = ["Leave-one-scenario-out evaluation",
         "Bases: POD, energy-weighted POD, TBMD",
         "Sensors: QR/greedy, wells, random",
         "Estimators: least squares, $\\ell_1$",
         "",
         "Untruncated TBMD = rotated POD-E",
         "Spatial truncation: error floor",
         "Energy weighting: stable for N < r"]
for i, t in enumerate(lines):
    ax0.text(0, 0.83 - i * 0.105, t, fontsize=10, va="top", weight="bold" if i >= 5 else "normal")
ax = fig.add_axes([0.60, 0.14, 0.37, 0.72])
y = list(range(len(bars)))[::-1]
ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
for yy, b in zip(y, bars):
    ax.text(b[1] + 0.02, yy, f"{b[1]:.2f}", va="center", fontsize=10)
ax.set_yticks(y); ax.set_yticklabels([b[0].replace("\n", " ") for b in bars], fontsize=9)
ax.set_xlabel("Pressure RMSE (bar)")
ax.set_xlim(0, max(b[1] for b in bars) * 1.2)
ax.set_title("Held-out scenario, mean of 10 folds", fontsize=10.5)
fig.savefig(FIG / "graphical_abstract.pdf"); fig.savefig(FIG / "graphical_abstract.png", dpi=300)
print("graphical abstract written")
