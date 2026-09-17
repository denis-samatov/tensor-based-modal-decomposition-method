"""LaTeX tables for the manuscript and supplement, generated from outputs/key_numbers.json and
outputs/e6_cost.json. Outputs: outputs/tab_main.tex (Table 2, P2 pressure), outputs/tab_cost.tex
(Table 3), outputs/tabS_P1.tex, outputs/tabS_So.tex (supplement)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bss import OUT  # noqa: E402

K = json.loads((OUT / "key_numbers.json").read_text())
cost = json.loads((OUT / "e6_cost.json").read_text())
ROWS = [("No measurements (prior)", "No measurements (prior)"),
        ("Prior + IDW (POD-E sites)", "Prior + IDW residual"),
        ("POD, LS", "POD, LS"), ("POD, $\\ell_1$", "POD, l1"),
        ("POD-E, LS", "POD-E, LS"), ("POD-E, $\\ell_1$", "POD-E, l1"),
        ("TBMD $(48,48,2)$, LS", "TBMD (48,48,2), LS"),
        ("TBMD $(48,48,2)$, $\\ell_1$", "TBMD (48,48,2), l1 [original]"),
        ("Random sites, POD-E, LS", "Random placement, POD-E LS (median of 20)"),
        ("Configured wells, POD-E, LS", "Configured well order, POD-E LS"),
        ("Configured wells, TBMD, $\\ell_1$", "Configured well order, TBMD l1")]
COLS = [("grid", 10), ("grid", 30), ("grid", 100), ("wells-joint", 10), ("wells-joint", 30), ("wells-pressure", 30)]


def fmt(v, sd, scale=1.0):
    v, sd = v * scale, sd * scale
    if v >= 10:
        return f"{v:.0f}\\,$\\pm$\\,{sd:.0f}"
    return f"{v:.2f}\\,$\\pm$\\,{sd:.2f}"


def table(P, metric, scale, caption, label):
    lines = ["\\begin{table}[tbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Grid-wide channels} & \\multicolumn{2}{c}{Wells ($p$, $S_o$)} & Wells ($p$) \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-7}",
             "Method & $N=10$ & $N=30$ & $N=100$ & $N=10$ & $N=30$ & $N=30$ \\\\", "\\midrule"]
    for name, lab in ROWS:
        cells = []
        for s, N in COLS:
            k = f"{P}|{lab}|{s}|{N}|{metric}|mean"
            if lab.startswith("Configured") and s == "grid":
                cells.append("--")
            elif k in K:
                cells.append(fmt(K[k], K[k.replace('|mean', '|sd')], scale))
            else:
                cells.append("--")
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


(OUT / "tab_main.tex").write_text(table(
    "P2", "rmse_p", 1.0,
    "Pressure RMSE (bar) in protocol P2, mean\\,$\\pm$\\,standard deviation over the ten held-out control scenarios. "
    "Placement is QR (grid) or DG (wells) on the basis of the same row unless stated; $r=16$ in every fold. "
    "The prior row uses no measurements and is repeated across columns. Random: per-fold median over 20 placements.",
    "tab:main"))
(OUT / "tabS_P1.tex").write_text(table(
    "P1", "rmse_p", 1.0,
    "As Table~2 of the main text, for protocol P1 (within-scenario temporal hold-out; prior = persistence).", "tab:S_P1"))
(OUT / "tabS_So.tex").write_text(table(
    "P2", "rmse_so", 1e3,
    "Oil-saturation RMSE ($\\times10^{-3}$) in protocol P2; layout as Table~2 of the main text. For pressure-only wells, "
    "$S_o$ is inferred through the joint basis.", "tab:S_So"))
c = cost
rows = [("POD basis (thin SVD)", "offline", f"{c['offline_pod_svd_s']:.2f} s"),
        ("TBMD basis (Tucker/HOOI, ranks (48,48,2,16))", "offline", f"{c['offline_tbmd_tucker_s']:.1f} s"),
        ("QR placement, $N=r$", "offline", f"{c['offline_qr_r_s']*1e3:.1f} ms"),
        ("QR + DG placement, $N=300$", "offline", f"{c['offline_qr_dg_300_s']:.2f} s"),
        ("DG ranking of 30 wells", "offline", f"{c['offline_block_dg_30wells_s']*1e3:.1f} ms"),
        ("LS estimate, 30 wells", "online, per snapshot", f"{c['online_ls_wells30_s_per_snapshot']*1e6:.1f} \\si{{\\micro\\second}}"),
        ("$\\ell_1$ (ADMM) estimate, 30 wells", "online, per snapshot", f"{c['online_l1_wells30_s_per_snapshot']*1e6:.0f} \\si{{\\micro\\second}}"),
        ("LS estimate, 30 grid channels", "online, per snapshot", f"{c['online_ls_grid30_s_per_snapshot']*1e6:.1f} \\si{{\\micro\\second}}"),
        ("$\\ell_1$ (ADMM) estimate, 30 grid channels", "online, per snapshot", f"{c['online_l1_grid30_s_per_snapshot']*1e6:.0f} \\si{{\\micro\\second}}")]
sp = c["hardware"].get("system_profiler")
if isinstance(sp, list):
    d = dict(item.split(": ", 1) for item in sp if ": " in item)
    cores = d.get("Total Number of Cores", "").replace(" Performance and ", " performance, ").replace(" Efficiency", " efficiency")
    hw = ", ".join(x for x in (d.get("Model Name"), d.get("Chip"), f"{cores} cores" if cores else "",
                               f"{d['Memory']} memory" if "Memory" in d else "") if x)
else:
    hw = c["hardware"]["machine"]
lines = ["\\begin{table}[tbp]", "\\centering", "\\small",
         f"\\caption{{Single-thread wall time (median of repeated runs) for P2 fold 1; online times are per snapshot, amortised over the 133 held-out snapshots of the fold solved jointly ($M=9{{,}}900$ channels, 1{{,}}197 training snapshots, $r={c['rank']}$). Hardware: {hw}; Python {c['hardware']['python']}, NumPy {c['hardware']['numpy']}.}}",
         "\\label{tab:cost}", "\\begin{tabular}{@{}llr@{}}", "\\toprule", "Step & Stage & Time \\\\", "\\midrule"]
lines += [f"{a} & {b} & {t} \\\\" for a, b, t in rows]
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT / "tab_cost.tex").write_text("\n".join(lines) + "\n")
print("tables written")
