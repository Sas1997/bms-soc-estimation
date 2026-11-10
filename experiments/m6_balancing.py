"""Stretch goal: passive vs active cell balancing for a small series pack.

Cells: the selected identified cell model (results/characterization/selected.json)
at 25 degC, nominal capacity = C/20 discharge capacity at 25 degC. The dataset
has one cell, so cell-to-cell variation is ASSUMED, not measured:

    initial SoC        0.90 + N(0, 2 points)      per cell
    capacity           nominal x (1 + N(0, 1.5 %))
    all resistances    nominal x (1 + N(0, 5 %))

drawn for 200 packs of 4 cells in series (seed 7, values clipped to +-3 sigma).

Balancing circuits (src/socest/pack.py): passive shunt resistor, 20 ohm (about
0.2 A at 4 V); active adjacent cell-to-cell and active cell-to-pack, 0.2 A from
the donor cell, 85 % converter efficiency. Balancing decisions use SoC. Unless
stated, the controller knows each cell's true SoC.

Scenario 1, rest: the pack is idle; balance until the SoC spread (max - min) is
below 0.5 points. Report balancing time and energy lost (drop in stored chemical
energy, which includes the balancing circuit and the cells' own resistance).

Scenario 2, discharge: the pack starts from an end-of-charge state (the highest
cell at SoC 1.0, others below by their initial spread) and is discharged by the
25 degC UDDS current profile from the dataset, repeated, until the first cell
falls below 2.8 V (the cut-off of the dataset's capacity tests, data/DATA_CARD.md).
Balancing runs during the discharge (on above 1 point spread, off below 0.5).
Report energy delivered to the load relative to no balancing. Passive balancing
is usually applied at the end of charge, not during discharge, so two more cases
start the discharge with every cell full (top-balanced): one with no balancing
in discharge, one with active adjacent balancing in discharge. Any of the three
circuits can reach that state during charging if the charge lasts long enough;
the energy burned or converted while doing so comes from the charger and is not
counted in the delivered energy.

Sensitivity: converter efficiency, a fixed per-cell SoC estimation error in the
controller, and string length.

Writes results/m6_*.csv, results/tables/m6_balancing.md, results/figures/balancing.png.
Run: python experiments/m6_balancing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cell import capacity_fn, load_model  # noqa: E402
from socest.cycles import load_split  # noqa: E402
from socest.pack import Adjacent, CellToPack, Passive, PackModel, simulate_drive, simulate_rest  # noqa: E402

TEMP = 25.0
N_CELLS, N_PACKS, SEED = 4, 200, 7
SOC_MEAN, SOC_SD, CAP_SD, RES_SD = 0.90, 0.02, 0.015, 0.05
V_CUT = 2.8
ETA, I_BAL, R_BLEED = 0.85, 0.2, 20.0
RES, TAB, FIG = ROOT / "results", ROOT / "results" / "tables", ROOT / "results" / "figures"

TOP = "Top-balanced at end of charge"
TOP_ACT = "Top-balanced, plus active adjacent in discharge"
COLORS = {"No balancing": "#8a8984", "Passive (shunt resistor)": "#eb6834",
          "Active, adjacent cell-to-cell": "#1baf7a", "Active, cell-to-pack": "#2a78d6"}
CIRCUITS = ["Passive (shunt resistor)", "Active, adjacent cell-to-cell", "Active, cell-to-pack"]
# Drive-scenario cases: (label, colour, marker). A top-balanced start is a state any
# circuit can reach during charging, so it is drawn in neutral grey.
DRIVE_CASES = [("Passive (shunt resistor)", "Passive,\nin discharge", "#eb6834", "o"),
               ("Active, adjacent cell-to-cell", "Active adjacent,\nin discharge", "#1baf7a", "o"),
               ("Active, cell-to-pack", "Active cell-to-pack,\nin discharge", "#2a78d6", "o"),
               (TOP, "Top-balanced\nat end of charge", "#52514e", "o"),
               (TOP_ACT, "Top-balanced, plus\nactive adjacent", "#1baf7a", "D")]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def balancers(eta=ETA):
    return [Passive(r_bleed=R_BLEED), Adjacent(i_bal=I_BAL, eta=eta), CellToPack(i_bal=I_BAL, eta=eta)]


def draw_packs(n_packs, n_cells, seed):
    rng = np.random.default_rng(seed)

    def z(shape):
        return np.clip(rng.standard_normal(shape), -3, 3)

    soc0 = np.clip(SOC_MEAN + SOC_SD * z((n_packs, n_cells)), 0.0, 1.0)
    cap = 1.0 + CAP_SD * z((n_packs, n_cells))
    res = 1.0 + RES_SD * z((n_packs, n_cells))
    return soc0, cap, res


def drive_profile():
    r, u = next((r, u) for r, u in load_split("test_unseen_cycle") if r.folder == "25degC" and r.test == "UDDS")
    return u.current_a.to_numpy()


def summarise(df, value, by="method"):
    g = df.groupby(by, sort=False)[value]
    return pd.DataFrame({"median": g.median(), "p10": g.quantile(0.1), "p90": g.quantile(0.9)})


def main():
    t0 = time.time()
    cell = load_model(capacity_ah=capacity_fn()(TEMP))
    soc0, cap, res = draw_packs(N_PACKS, N_CELLS, SEED)
    i_prof = drive_profile()
    soc_drive0 = soc0 - soc0.max(axis=1, keepdims=True) + 1.0    # end of charge: highest cell full

    # ---------------- scenario 1: rest
    rest_rows, traces_rest = [], {}
    for bal in balancers():
        out = simulate_rest(PackModel(cell, TEMP, cap, res), bal, soc0, record=0)
        traces_rest[bal.name] = out["trace"]
        for p in range(N_PACKS):
            rest_rows.append({"pack": p, "method": bal.name, "t_balance_h": out["t_balance_h"][p],
                              "loss_wh": out["loss_wh"][p], "circuit_loss_wh": out["circuit_loss_wh"][p],
                              "loss_pct_of_stored": 100 * out["loss_wh"][p] / out["stored_wh_start"][p],
                              "initial_spread_pts": 100 * (soc0[p].max() - soc0[p].min()),
                              "final_spread_pts": 100 * out["final_spread"][p]})
        print(f"rest {bal.name}: {time.time() - t0:.0f} s", flush=True)
    rest = pd.DataFrame(rest_rows)
    rest.to_csv(RES / "m6_balancing_rest.csv", index=False)

    # ---------------- scenario 2: discharge
    drive_rows, traces_drive = [], {}
    base = None
    runs = [("No balancing", None, soc_drive0)] + [(b.name, b, soc_drive0) for b in balancers()] \
        + [(TOP, None, np.ones_like(soc_drive0)), (TOP_ACT, balancers()[1], np.ones_like(soc_drive0))]
    for name, bal, start in runs:
        out = simulate_drive(PackModel(cell, TEMP, cap, res), bal, start, i_prof, v_cut=V_CUT, record=0)
        traces_drive[name] = out["trace"]
        if name == "No balancing":
            base = out["energy_wh"]
        for p in range(N_PACKS):
            drive_rows.append({"pack": p, "method": name, "energy_wh": out["energy_wh"][p], "ah": out["ah"][p],
                               "gain_pct": 100 * (out["energy_wh"][p] / base[p] - 1),
                               "circuit_loss_wh": out["circuit_loss_wh"][p], "t_end_h": out["t_end_h"][p],
                               "end_spread_pts": 100 * (out["soc_end"][p].max() - out["soc_end"][p].min())})
        print(f"drive {name}: {time.time() - t0:.0f} s", flush=True)
    drive = pd.DataFrame(drive_rows)
    drive.to_csv(RES / "m6_balancing_drive.csv", index=False)

    # ---------------- sensitivity
    sens = []
    for eta in (0.70, 0.80, 0.90, 0.95):
        for bal in balancers(eta)[1:]:
            r = simulate_rest(PackModel(cell, TEMP, cap, res), bal, soc0)
            d = simulate_drive(PackModel(cell, TEMP, cap, res), bal, soc_drive0, i_prof, v_cut=V_CUT)
            sens.append({"study": "efficiency", "x": eta, "method": bal.name,
                         "rest_loss_wh": np.median(r["loss_wh"]), "rest_t_h": np.median(r["t_balance_h"]),
                         "drive_gain_pct": np.median(100 * (d["energy_wh"] / base - 1))})
    print(f"efficiency sweep: {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(SEED + 1)
    z = np.clip(rng.standard_normal((N_PACKS, N_CELLS)), -3, 3)
    for sd_pts in (0.0, 0.5, 1.0, 2.0):
        for bal in balancers():
            r = simulate_rest(PackModel(cell, TEMP, cap, res), bal, soc0, ctrl_offset=z * sd_pts / 100)
            sens.append({"study": "controller SoC error", "x": sd_pts, "method": bal.name,
                         "final_spread_pts": 100 * np.median(r["final_spread"]),
                         "final_spread_p90_pts": 100 * np.quantile(r["final_spread"], 0.9),
                         "rest_loss_wh": np.median(r["loss_wh"])})
    print(f"estimation-error sweep: {time.time() - t0:.0f} s", flush=True)
    for n in (4, 8, 12):
        s0, c, rr = draw_packs(100, n, SEED + n)
        for bal in balancers():
            r = simulate_rest(PackModel(cell, TEMP, c, rr), bal, s0, t_max_h=96.0)
            sens.append({"study": "string length", "x": n, "method": bal.name,
                         "rest_t_h": np.median(r["t_balance_h"]), "rest_loss_wh": np.median(r["loss_wh"]),
                         "rest_loss_wh_per_cell": np.median(r["loss_wh"]) / n,
                         "not_balanced": int(np.isnan(r["t_balance_h"]).sum())})
    print(f"string-length sweep: {time.time() - t0:.0f} s", flush=True)
    sens = pd.DataFrame(sens)
    sens.to_csv(RES / "m6_balancing_sensitivity.csv", index=False)

    write_table(rest, drive, sens, cell)
    plot(rest, drive, sens, traces_rest)
    print(f"done in {time.time() - t0:.0f} s")


def write_table(rest, drive, sens, cell):
    stored = rest.groupby("pack").first()
    t_rest = pd.concat({"balancing time (h)": summarise(rest, "t_balance_h"),
                        "energy lost (Wh)": summarise(rest, "loss_wh"),
                        "energy lost (% of stored)": summarise(rest, "loss_pct_of_stored")}, axis=1)
    t_rest[("of which in the circuit (%)", "median")] = (100 * rest.groupby("method", sort=False).circuit_loss_wh.median()
                                                      / rest.groupby("method", sort=False).loss_wh.median())
    t_drive = pd.concat({"energy delivered (Wh)": summarise(drive, "energy_wh"),
                         "change vs no balancing (%)": summarise(drive, "gain_pct"),
                         "circuit loss (Wh)": summarise(drive, "circuit_loss_wh"),
                         "SoC spread at cut-off (points)": summarise(drive, "end_spread_pts")}, axis=1)
    eff = sens[sens.study == "efficiency"].pivot_table(index="x", columns="method", values=["rest_loss_wh", "drive_gain_pct"])
    err = sens[sens.study == "controller SoC error"].pivot_table(index="x", columns="method", values="final_spread_pts")
    err90 = sens[sens.study == "controller SoC error"].pivot_table(index="x", columns="method", values="final_spread_p90_pts")
    strl = sens[sens.study == "string length"].pivot_table(index="x", columns="method", values=["rest_t_h", "rest_loss_wh_per_cell"])
    notb = int(sens[sens.study == "string length"].not_balanced.sum())

    def fmt(df):
        df = df.copy()
        df.columns = [" ".join(str(x) for x in c).strip() if isinstance(c, tuple) else c for c in df.columns]
        return df.round(2).to_markdown()

    md = ["# Cell balancing: passive vs active (simulation)", "",
          __doc__.split("Writes")[0].strip().split("\n\n", 1)[1], "",
          f"Nominal cell: {cell.capacity_ah:.3f} Ah. Median initial SoC spread in the pack: "
          f"{rest[rest.method == rest.method.iloc[0]].initial_spread_pts.median():.1f} points "
          f"(p10 {rest[rest.method == rest.method.iloc[0]].initial_spread_pts.quantile(0.1):.1f}, "
          f"p90 {rest[rest.method == rest.method.iloc[0]].initial_spread_pts.quantile(0.9):.1f}).", "",
          "## Scenario 1: balancing at rest (200 packs, median and 10th to 90th percentile)", "", fmt(t_rest), "",
          "## Scenario 2: balancing during a UDDS discharge to 2.8 V (200 packs)", "", fmt(t_drive), "",
          "## Sensitivity: converter efficiency (medians; passive does not depend on it)", "", fmt(eff), "",
          "## Sensitivity: SoC estimation error in the controller (rest)", "",
          "Each cell's SoC seen by the controller is off by a fixed amount drawn from N(0, sigma). "
          "Rows: sigma in SoC points. Values: true SoC spread when the controller stops (points), median and 90th percentile.", "",
          "Median:", "", fmt(err), "", "90th percentile:", "", fmt(err90), "",
          "## Sensitivity: string length (rest, 100 packs per length, medians)", "",
          f"Packs not balanced within 96 h: {notb}.", "", fmt(strl), ""]
    (TAB / "m6_balancing.md").write_text("\n".join(md))
    print(fmt(t_rest))
    print(fmt(t_drive))
    print(fmt(eff))
    print(fmt(err))
    print(fmt(err90))
    print(fmt(strl))


def style(ax, ylabel, xlabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def plot(rest, drive, sens, traces_rest):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.4))
    names = CIRCUITS

    a = ax[0, 0]
    for name in names:
        t, s = traces_rest[name]
        a.plot(t, 100 * (s.max(axis=1) - s.min(axis=1)), color=COLORS[name], lw=2, label=name)
    a.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    a.text(a.get_xlim()[1] if False else 0.02, 0.62, "target 0.5 points", color=MUTED, fontsize=8.5,
           transform=a.get_yaxis_transform())
    style(a, "SoC spread, max - min (points)", "Time at rest (h)")
    t_ex = rest[rest.pack == 0].t_balance_h.max()
    a.set_xlim(0, 1.6 * t_ex)
    a.set_ylim(bottom=0)
    a.set_title("A  One example pack balanced at rest", loc="left", fontsize=10.5, color=INK)

    a = ax[0, 1]
    for name in names:
        d = rest[rest.method == name]
        a.scatter(d.t_balance_h, d.loss_wh, s=14, color=COLORS[name], alpha=0.55, linewidths=0, label=name)
    style(a, "Energy lost while balancing (Wh)", "Balancing time (h)")
    a.set_xlim(left=0)
    a.set_ylim(bottom=0)
    a.set_title("B  Rest balancing, 200 packs: time vs energy lost", loc="left", fontsize=10.5, color=INK)

    a = ax[1, 0]
    rng = np.random.default_rng(0)
    for k, (name, tick, color, marker) in enumerate(DRIVE_CASES):
        d = drive[drive.method == name].gain_pct.to_numpy()
        a.scatter(k + rng.uniform(-0.18, 0.18, d.size), d, s=12 if marker == "o" else 10, marker=marker,
                  color=color, alpha=0.45, linewidths=0)
        q = np.quantile(d, [0.1, 0.5, 0.9])
        a.plot([k - 0.28, k + 0.28], [q[1]] * 2, color=INK, lw=2)
        a.text(k + 0.3, q[1], f"{q[1]:+.1f} %", va="center", fontsize=8.5, color=INK)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(range(len(DRIVE_CASES)))
    a.set_xticklabels([c[1] for c in DRIVE_CASES], fontsize=8.5)
    style(a, "Energy delivered vs no balancing (%)", "")
    a.xaxis.grid(False)
    a.set_title("C  Energy delivered in a UDDS discharge to 2.8 V, vs no balancing", loc="left", fontsize=10.5, color=INK)

    a = ax[1, 1]
    e = sens[sens.study == "controller SoC error"]
    gap = (e.groupby("x").final_spread_pts.max() - e.groupby("x").final_spread_pts.min()).max()
    d = e[e.method == "Active, adjacent cell-to-cell"].sort_values("x")
    a.plot(d.x, d.final_spread_pts, color=INK, lw=2, marker="o", ms=7)
    a.plot(d.x, d.final_spread_p90_pts, color=MUTED, lw=1.5, ls="--", marker="o", ms=5)
    a.annotate("median", (d.x.iloc[-1], d.final_spread_pts.iloc[-1]), textcoords="offset points", xytext=(-8, -16),
               ha="right", fontsize=8.5, color=INK)
    a.annotate("90th percentile", (d.x.iloc[-1], d.final_spread_p90_pts.iloc[-1]), textcoords="offset points",
               xytext=(-8, 4), ha="right", fontsize=8.5, color=MUTED)
    a.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    a.text(0.98, 0.62, "target 0.5 points", color=MUTED, fontsize=8.5, ha="right", transform=a.get_yaxis_transform())
    style(a, "True SoC spread when balancing stops (points)", "SoC estimation error per cell, 1 sigma (points)")
    a.set_ylim(bottom=0)
    a.set_title(f"D  Balancing on imperfect SoC estimates (all three circuits within {gap:.2f} points)",
                loc="left", fontsize=10.5, color=INK)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"Cell balancing in a simulated {N_CELLS}S pack of identified LG HG2 cells at 25 °C", x=0.012,
                 ha="left", fontsize=13, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.0)
    fig.savefig(FIG / "balancing.png", dpi=140)


if __name__ == "__main__":
    main()
