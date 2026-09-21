"""ACG graphical abstract generated from the E9 summary; no generated imagery."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bss import FIG, OUT
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(8.853, 3.541), dpi=300)
flow = fig.add_axes([0.02, 0.69, 0.96, 0.28])
flow.axis("off")
boxes = [
    "Brugge ensemble\n10 control scenarios",
    "Nested TBMD\nand POD-E",
    "Sparse point\nobservations",
    "Pressure / saturation\nreconstruction",
]
for i, label in enumerate(boxes):
    x = i * 0.255
    flow.add_patch(
        FancyBboxPatch(
            (x, 0.16),
            0.225,
            0.66,
            boxstyle="round,pad=0.01",
            facecolor="#eef3f7",
            edgecolor="#37556c",
            linewidth=1,
            clip_on=False,
        )
    )
    flow.text(x + 0.1125, 0.49, label, ha="center", va="center", fontsize=11)
    if i < 3:
        flow.annotate(
            "", (x + 0.25, 0.49), (x + 0.229, 0.49), arrowprops={"arrowstyle": "->", "lw": 1.3}
        )
ax = fig.add_axes([0.10, 0.20, 0.40, 0.43])
df = pd.read_csv(OUT / "e9_property_mode/outer_summary.csv")
subset = df[
    (df.capacity_regime == "equal_total") & (df.geometry == "grid") & (df.budget == 30)
].set_index("model_label")
for offset, method, color, hatch, label in [
    (-0.18, "3D optimized", "#4477aa", "//", "Independent 3D"),
    (0.18, "4D optimized", "#228833", "", "Selected 4D"),
]:
    values = [subset.loc[method, "anomaly_relative_frobenius_" + p + "_mean"] for p in ["p", "so"]]
    ax.bar(
        [offset, 1 + offset],
        values,
        width=0.34,
        color=color,
        hatch=hatch,
        edgecolor="black",
        linewidth=0.6,
        label=label,
    )
    for x, v in zip([offset, 1 + offset], values):
        ax.text(x, v + 0.035, f"{v:.2f}", ha="center", fontsize=10)
ax.set_xticks([0, 1], ["Pressure", "Oil saturation"])
ax.set_ylim(0, 1.45)
ax.set_ylabel("Anomaly-relative error", fontsize=10)
ax.set_title("Matched E9: 30 grid channels", fontsize=11)
ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.19))
text = fig.add_axes([0.55, 0.16, 0.43, 0.48])
text.axis("off")
text.text(0, 0.97, "Partial sharing helps sparse recovery", weight="bold", fontsize=11, va="top")
text.text(0, 0.72, "Shared spatial factors;\nproperty-specific coefficients", fontsize=11, va="top")
text.text(
    0,
    0.36,
    "POD-E retains the lowest errors\nat all existing joint-property wells",
    fontsize=11,
    va="top",
)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "graphical_abstract.pdf")
fig.savefig(FIG / "graphical_abstract.png", dpi=300)
plt.close(fig)
print(
    "Graphical abstract: deterministic Matplotlib vector PDF and PNG", fig.canvas.get_width_height()
)
