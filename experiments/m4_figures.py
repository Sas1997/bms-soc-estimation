"""Headline figure for the README, from results/m4_metrics.csv, results/m4_traces.npz and the
validation tuning tables (results/m3_ekf_tuning_grid.csv, results/m3_nn_selection.csv).

Panels:
  A  nominal SoC RMSE by temperature (both test sets; 10 degC is the held-out temperature)
  B  SoC RMSE by robustness case (all test cycles)
  C  SoC error over time, UDDS at -20 degC, estimation started under load 20 points low
  D  validation cycles: nominal RMSE against RMSE after a late start, for every EKF
     noise setting on the tuning grid and for the selected networks

Run after m4_evaluate.py: python experiments/m4_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Colour follows the estimator everywhere. Coulomb counting is the reference
# baseline and is drawn in neutral grey.
COLORS = {
    "Coulomb counting": "#8a8984",
    "EKF, 3RC (calibrated)": "#2a78d6",
    "EKF, 2RC (HPPC only)": "#eb6834",
    "LSTM": "#1baf7a",
    "Transformer": "#eda100",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
NICE = {"n20degC": "-20", "n10degC": "-10", "0degC": "0", "10degC": "10*", "25degC": "25", "40degC": "40"}
CASES = ["nominal", "initial SoC -20 pts", "late start, SoC -20 pts", "sensor noise", "current bias +100 mA"]
CASE_NICE = ["nominal", "start at rest,\n20 pts low", "start under load,\n20 pts low", "sensor\nnoise", "current bias\n+100 mA"]


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def bars(ax, table, names, labels, value_fmt="{:.1f}", cap=None):
    """Grouped bars with a surface-coloured gap; values above cap are clipped and labelled."""
    n = len(names)
    w = 0.8 / n
    for k, name in enumerate(names):
        y = table.loc[name].to_numpy(dtype=float)
        x = np.arange(len(y)) + (k - (n - 1) / 2) * w
        shown = np.minimum(y, cap) if cap else y
        ax.bar(x, shown, w, color=COLORS[name], edgecolor=SURFACE, linewidth=1.5, label=name)
        if cap:
            for xi, yi in zip(x, y):
                if yi > cap:
                    ax.text(xi, cap * 1.01, value_fmt.format(yi), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    if cap:
        ax.set_ylim(0, cap * 1.1)


def trace(ax, tr, key, case, names, title):
    ref = tr[f"{key}|{case}|ref"]
    t = np.arange(ref.size) / 60
    for name in sorted(names, key=lambda n: n.startswith(("EKF", "Coulomb"))):   # networks first, filters on top
        net = not name.startswith(("EKF", "Coulomb"))
        ax.plot(t, 100 * (tr[f"{key}|{case}|{name}"] - ref), color=COLORS[name], lw=1.0 if net else 2.0, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "SoC error (points)")
    ax.set_xlabel("Time (min)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def tradeoff(ax):
    """Validation cycles: nominal RMSE against RMSE after a late start, one dot per EKF noise setting."""
    import json

    grid = pd.read_csv(RES / "m3_ekf_tuning_grid.csv")
    nn = pd.read_csv(RES / "m3_nn_selection.csv")
    chosen = json.loads((RES / "characterization" / "ekf_tuning.json").read_text())
    picked = json.loads((RES / "models" / "selected.json").read_text())
    late = "late start, SoC -20 pts"
    for variant, name in (("2rc", "EKF, 2RC (HPPC only)"), (next(v for v in chosen if v != "2rc"), "EKF, 3RC (calibrated)")):
        g = grid[grid.variant == variant]
        ax.scatter(g.nominal, g[late], s=16, color=COLORS[name], alpha=0.45, linewidths=0)
        c = chosen[variant]
        sel = g[(g.k_r == c["k_r"]) & (g.q_soc == c["q_soc"]) & (g.q_v == c["q_v"])].iloc[0]
        ax.scatter(sel.nominal, sel[late], s=90, color=COLORS[name], edgecolors=INK, linewidths=1.5, zorder=3)
        ax.annotate("selected", (sel.nominal, sel[late]), textcoords="offset points", xytext=(8, 6), fontsize=8.5, color=INK)
    for kind, name in (("lstm", "LSTM"), ("transformer", "Transformer")):
        r = nn[nn.checkpoint == picked[kind]].iloc[0]
        ax.scatter(r.nominal, r[late], s=110, marker="D", color=COLORS[name], edgecolors=INK, linewidths=1.5, zorder=3)
        off, ha = ((-10, -14), "right") if kind == "lstm" else ((10, 3), "left")
        ax.annotate(name, (r.nominal, r[late]), textcoords="offset points", xytext=off, ha=ha, fontsize=8.5, color=INK)
    style(ax, "SoC RMSE after a late start (points)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("SoC RMSE, nominal (points)", color=MUTED, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("D  Validation cycles: each dot is one EKF noise setting, diamonds are the networks",
                 loc="left", fontsize=10.5, color=INK)


def main():
    res = pd.read_csv(RES / "m4_metrics.csv")
    tr = np.load(RES / "m4_traces.npz")
    names = [n for n in COLORS if n in set(res.estimator)]
    closed = [n for n in names if n != "Coulomb counting"]

    nom = res[res.case == "nominal"]
    by_t = nom.pivot_table(index="estimator", columns="folder", values="rmse", aggfunc="mean")[ORDER]
    by_c = res.pivot_table(index="estimator", columns="case", values="rmse", aggfunc="mean")[CASES]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.2))

    cap_a = float(np.ceil(by_t.loc[closed].drop(columns="n20degC").to_numpy().max() * 1.6))
    bars(ax[0, 0], by_t, closed, [NICE[f] for f in ORDER], cap=cap_a)
    style(ax[0, 0], "SoC RMSE (points)")
    ax[0, 0].set_xlabel("Ambient temperature (°C).  * held out: no estimator saw 10 °C data", color=MUTED, fontsize=9)
    ax[0, 0].set_title("A  Accuracy by temperature, test cycles, nominal", loc="left", fontsize=10.5, color=INK)

    bars(ax[0, 1], by_c, names, CASE_NICE, cap=float(np.ceil(by_c.loc[closed].to_numpy().max() * 1.25)))
    style(ax[0, 1], "SoC RMSE (points)")
    ax[0, 1].set_title("B  Robustness, all 29 test cycles (noise: 10 mV, 100 mA, 1 °C)", loc="left", fontsize=10.5, color=INK)

    trace(ax[1, 0], tr, "n20degC|UDDS", "late start, SoC -20 pts", names,
          "C  UDDS at -20 °C, estimation started 20 min in, 20 points low")
    tradeoff(ax[1, 1])

    handles, labels = ax[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(names), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("State-of-charge estimation on LG HG2 test data: EKF vs neural networks", x=0.012, ha="left",
                 fontsize=13, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.0)
    fig.savefig(RES / "figures" / "headline.png", dpi=140)
    print("wrote", RES / "figures" / "headline.png")


if __name__ == "__main__":
    main()
