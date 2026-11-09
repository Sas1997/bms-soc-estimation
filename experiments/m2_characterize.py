"""M2: OCV curves, 1RC/2RC identification from HPPC, ECM validation on drive cycles.

Outputs (results/characterization/ unless noted):
  ocv.json                  model OCV(SoC, T): C/20 discharge branch + IR (selected, see m2_select_ecm.py)
  ocv_average.json          average of C/20 charge and discharge branches (reference only)
  hppc_fits.csv             one row per (temperature, SoC level, model)
  params_1rc.json, params_2rc.json
                            ParamTable over SoC x T built WITHOUT 10 degC, the
                            held-out temperature; at 10 degC the estimator must
                            interpolate between 0 and 25 degC like the ML model
  results/tables/ecm_fit_rmse.md
  results/figures/ocv_curves.png, ecm_params.png, hppc_fit_example.png

Run: python experiments/m2_characterize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.characterize import hppc_blocks, identify_hppc, ocv_from_c20  # noqa: E402
from socest.datasets.lg_hg2 import FOLDER_TEMPS  # noqa: E402
from socest.identify import rc_voltages  # noqa: E402
from socest.ocv import OCVCurve  # noqa: E402
from socest.params import ParamTable  # noqa: E402
from socest.split import HELD_OUT_FOLDER  # noqa: E402

PROC = ROOT / "data" / "processed" / "LG_HG2"
OUT = ROOT / "results" / "characterization"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FOLDERS = list(FOLDER_TEMPS)
LABEL = {f: f"{FOLDER_TEMPS[f]:g} °C" for f in FOLDERS}
TABLE_SOC = np.linspace(0.0, 1.0, 21)


def find(folder, test):
    hits = sorted((PROC / folder).glob(f"*_{test}.parquet"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{folder}/{test}: {hits}")
    return pd.read_parquet(hits[0])


def build_ocv(cap):
    rows, curves = {}, {}
    for f in FOLDERS:
        o = ocv_from_c20(pd.read_parquet(PROC / f / f"{cap.loc[f, 'source']}.parquet"))
        curves[f] = o
        rows[f] = {k: o[k] for k in ("cap_dis_ah", "cap_chg_ah", "charge_branch_soc_max", "top_half_gap_v")}
        rows[f]["mean_chg_minus_dis_mv"] = float(np.nanmean(o["v_chg"] - o["v_dis"]) * 1000)
    data = {"soc": curves[FOLDERS[0]]["soc"].tolist(), "temps": [FOLDER_TEMPS[f] for f in FOLDERS],
            "folders": FOLDERS, "ocv": [curves[f]["ocv"].tolist() for f in FOLDERS]}
    data["method"] = "average of C/20 discharge and charge branches (reference only)"
    (OUT / "ocv_average.json").write_text(json.dumps(data))
    return curves, pd.DataFrame(rows).T


MIN_SOC = 0.15


def ocv_discharge_ir(curves, table, folder):
    """C/20 discharge branch plus I_C20 * (R0 + sum R_i) at that SoC and temperature."""
    o = curves[folder]
    T = FOLDER_TEMPS[folder]
    r_tot = np.array([(lambda p: p.r0 + sum(p.r))(table(s, T)) for s in o["soc"]])
    return np.maximum.accumulate(o["v_dis"] + o["i_dis_median_a"] * r_tot)


def write_model_ocv(curves, table):
    data = {"soc": curves[FOLDERS[0]]["soc"].tolist(), "temps": [FOLDER_TEMPS[f] for f in FOLDERS],
            "folders": FOLDERS, "method": "C/20 discharge branch + I_C20 * (R0 + R1 + R2), 2RC table",
            "ocv": [ocv_discharge_ir(curves, table, f).tolist() for f in FOLDERS]}
    (OUT / "ocv.json").write_text(json.dumps(data))


def ocv_object(curves, folders):
    return OCVCurve(curves[folders[0]]["soc"], np.vstack([curves[f]["ocv"] for f in folders]),
                    temps=[FOLDER_TEMPS[f] for f in folders])


def fit_all(curves, cap, max_pairs=None, verbose=True):
    rows = []
    for f in FOLDERS:
        hp = find(f, "HPPC")
        ocv_f = ocv_object(curves, [f])
        T = FOLDER_TEMPS[f]
        for n_rc in (1, 2):
            fits = identify_hppc(hp, ocv_f, cap.loc[f, "c20_discharge_ah"], T, n_rc, max_pairs=max_pairs)
            for k, ft in enumerate(fits):
                p = ft.params
                row = {"folder": f, "temp": T, "block": k, "n_rc": n_rc, "soc_mid": ft.soc_mid,
                       "rmse_mv": ft.rmse_v * 1000, "success": ft.success, "r0": p.r0}
                for j in range(n_rc):
                    row[f"r{j + 1}"] = p.r[j]
                    row[f"tau{j + 1}"] = p.tau[j]
                rows.append(row)
            if verbose:
                print(f"  {f} {n_rc}RC: {len(fits)} blocks, median RMSE {np.median([x.rmse_v for x in fits]) * 1000:.2f} mV",
                      flush=True)
    return pd.DataFrame(rows)


def param_table(fits, n_rc, folders, min_soc=0.0):
    """ParamTable on TABLE_SOC. Blocks with soc_mid < min_soc are dropped and the
    table is held flat below the lowest kept block (np.interp edge hold)."""
    sub = fits[(fits.n_rc == n_rc) & fits.folder.isin(folders) & (fits.soc_mid >= min_soc)]
    temps = [FOLDER_TEMPS[f] for f in folders]

    def grid(col, f):
        g = sub[sub.folder == f].sort_values("soc_mid")
        return np.interp(TABLE_SOC, g.soc_mid, g[col])  # held flat beyond the measured SoC range

    r0 = np.array([grid("r0", f) for f in folders])
    r = np.array([[grid(f"r{j + 1}", f) for f in folders] for j in range(n_rc)])
    tau = np.array([[grid(f"tau{j + 1}", f) for f in folders] for j in range(n_rc)])
    return ParamTable(temps, TABLE_SOC, r0, r, tau)


def plot_ocv(curves, table):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cols = plt.cm.coolwarm(np.linspace(0, 1, len(FOLDERS)))
    for c, f in zip(cols, FOLDERS):
        ax[0].plot(curves[f]["soc"], ocv_discharge_ir(curves, table, f), color=c, label=LABEL[f],
                   ls="--" if f == HELD_OUT_FOLDER else "-")
    for a, f in ((ax[1], "25degC"), (ax[2], "n20degC")):
        o = curves[f]
        a.plot(o["soc"], o["v_dis"], color="tab:red", lw=1, label="C/20 discharge")
        a.plot(o["soc"], o["v_chg"], color="tab:blue", lw=1, label="C/20 charge (CC part)")
        a.plot(o["soc"], o["ocv"], ":", color="0.4", lw=1.5, label="average (reference)")
        a.plot(o["soc"], ocv_discharge_ir(curves, table, f), color="k", lw=1.5, label="model OCV (discharge + IR)")
        a.set(xlabel="SoC", ylabel="Voltage (V)", title=f"Construction at {LABEL[f]}")
    ax[0].set(xlabel="SoC", ylabel="OCV (V)", title="Model OCV per temperature (10 °C dashed: not used by the model)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ocv_curves.png", dpi=110)
    plt.close(fig)


def plot_params(fits):
    sub = fits[fits.n_rc == 2]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    cols = dict(zip(FOLDERS, plt.cm.coolwarm(np.linspace(0, 1, len(FOLDERS)))))
    for f in FOLDERS:
        g = sub[sub.folder == f].sort_values("soc_mid")
        style = dict(color=cols[f], marker="o", ms=3, label=LABEL[f], ls="--" if f == HELD_OUT_FOLDER else "-")
        ax[0].plot(g.soc_mid, g.r0 * 1000, **style)
        ax[1].plot(g.soc_mid, (g.r1 + g.r2) * 1000, **style)
        ax[2].plot(g.soc_mid, g.tau1, **style)
        ax[3].plot(g.soc_mid, g.tau2, **style)
    ax[0].set(ylabel="R0 (mΩ)", yscale="log")
    ax[1].set(ylabel="R1 + R2 (mΩ)", yscale="log")
    ax[2].set(ylabel="τ1 (s)")
    ax[3].set(ylabel="τ2 (s)")
    for a in ax:
        a.set_xlabel("SoC")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("2RC parameters from HPPC (dashed: 10 °C, held out from the EKF parameter table)")
    fig.tight_layout()
    fig.savefig(FIG / "ecm_params.png", dpi=110)
    plt.close(fig)


def plot_fit_example(curves, cap, fits, folder="25degC", block=4):
    hp = find(folder, "HPPC")
    a, b = hppc_blocks(hp)[block]
    seg = hp.iloc[a : b + 1]
    t, i, v = seg.time_s.to_numpy(), seg.current_a.to_numpy(), seg.voltage_v.to_numpy()
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    C = cap.loc[folder, "c20_discharge_ah"]
    soc0 = 1 - (hp.ah.iloc[a] - hp.ah.iloc[0]) / C
    soc = soc0 - np.r_[0, np.cumsum(i[:-1] * dt[:-1])] / (3600 * C)
    ocv = ocv_object(curves, [folder])(soc, FOLDER_TEMPS[folder])
    ocv = ocv + (v[0] - ocv[0])
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(t - t[0], v, "k", lw=1, label="measured")
    for n_rc, c in ((1, "tab:orange"), (2, "tab:blue")):
        r = fits[(fits.folder == folder) & (fits.n_rc == n_rc) & (fits.block == block)].iloc[0]
        vrc = sum(rc_voltages(i, dt, r[f"r{j + 1}"], r[f"tau{j + 1}"]) for j in range(n_rc))
        vm = ocv - vrc - r.r0 * i
        ax[0].plot(t - t[0], vm, color=c, lw=1, label=f"{n_rc}RC, RMSE {r.rmse_mv:.1f} mV")
        ax[1].plot(t - t[0], (vm - v) * 1000, color=c, lw=0.8)
    ax[0].set(ylabel="Voltage (V)", title=f"HPPC block at SoC ≈ {soc0:.2f}, {LABEL[folder]}")
    ax[1].set(ylabel="Error (mV)", xlabel="Time in block (s)")
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hppc_fit_example.png", dpi=110)
    plt.close(fig)


def main():
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    cap = pd.read_csv(PROC / "capacity.csv", index_col=0)

    print("OCV curves")
    curves, ocv_summary = build_ocv(cap)

    print("HPPC identification")
    fits = fit_all(curves, cap)
    fits.to_csv(OUT / "hppc_fits.csv", index=False)

    # Variant B (selected in experiments/m2_select_ecm.py on train + val drive cycles):
    # all pulse pairs, SoC levels >= 0.15, table held flat below.
    model_folders = [f for f in FOLDERS if f != HELD_OUT_FOLDER]
    tables = {n_rc: param_table(fits, n_rc, model_folders, min_soc=MIN_SOC) for n_rc in (1, 2)}
    for n_rc, tab in tables.items():
        tab.to_json(OUT / f"params_{n_rc}rc.json")
    # Model OCV: discharge branch + IR (also selected on train + val drive cycles).
    write_model_ocv(curves, tables[2])
    plot_ocv(curves, tables[2])
    plot_params(fits)
    plot_fit_example(curves, cap, fits)

    summ = (fits.groupby(["folder", "n_rc"], sort=False)
            .agg(blocks=("rmse_mv", "size"), median_rmse_mv=("rmse_mv", "median"), max_rmse_mv=("rmse_mv", "max"),
                 all_converged=("success", "all"))
            .reset_index())
    lines = ["# ECM fit quality on HPPC blocks", "",
             "One fit per SoC level (four 10 s discharge/charge pulse pairs and their rests). Voltage RMSE in mV.", "",
             summ.round(2).to_markdown(index=False), "", "# OCV construction (C/20)", "",
             ocv_summary.round(4).to_markdown(), ""]
    (TAB / "ecm_fit_rmse.md").write_text("\n".join(lines))
    print(summ.round(2).to_string(index=False))
    print(ocv_summary.round(4).to_string())


if __name__ == "__main__":
    main()
