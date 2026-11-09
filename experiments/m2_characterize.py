"""M2: build the cell model variants from the characterisation tests (C/20 and HPPC only).

No drive cycle is used here. Scoring the variants on training and validation
drive cycles is done in experiments/m2_select_ecm.py.

Model variants written to results/characterization/ (see variants.json):
  1rc, 2rc      linear 1RC / 2RC from the HPPC pulses; OCV = C/20 discharge branch + I R
  2rc_allsoc    as 2rc but keeping the SoC levels below 0.15 (shows why they are dropped)
  2rc_rest      linear 2RC; OCV = C/20 average shifted to the 1 h rest points of the HPPC test
  3rc           2rc_rest plus a slow RC branch fitted on the step + 1 h relaxation segments
  2rc_bv        2RC plus a Butler-Volmer term (one exchange current i0 per temperature)
  3rc_bv        2rc_bv plus the slow branch

Every parameter table is built WITHOUT the 10 degC tests: 10 degC is the held-out
temperature, so the model must interpolate between 0 and 25 degC there.

Other outputs:
  hppc_fits.csv                     one row per (temperature, SoC level, pulse-fit family)
  results/tables/ecm_fit_rmse.md    fit quality on the HPPC pulses
  results/tables/m2_cold_diagnostics.md  evidence for the slow branch and the Butler-Volmer term
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

from socest.characterize import hppc_blocks, hppc_levels, identify_hppc, identify_slow, ocv_from_c20  # noqa: E402
from socest.datasets.lg_hg2 import FOLDER_TEMPS  # noqa: E402
from socest.ecm import bv_overpotential  # noqa: E402
from socest.identify import rc_voltages  # noqa: E402
from socest.ocv import OCVCurve  # noqa: E402
from socest.params import ParamTable  # noqa: E402
from socest.split import HELD_OUT_FOLDER  # noqa: E402

PROC = ROOT / "data" / "processed" / "LG_HG2"
OUT = ROOT / "results" / "characterization"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FOLDERS = list(FOLDER_TEMPS)
MODEL_FOLDERS = [f for f in FOLDERS if f != HELD_OUT_FOLDER]
LABEL = {f: f"{FOLDER_TEMPS[f]:g} °C" for f in FOLDERS}
TABLE_SOC = np.linspace(0.0, 1.0, 21)
MIN_SOC = 0.15            # SoC levels below this are not used in the tables (held flat below)
I0_GRID = np.geomspace(0.1, 100.0, 13)   # candidate exchange currents, A


def find(folder, test):
    hits = sorted((PROC / folder).glob(f"*_{test}.parquet"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{folder}/{test}: {hits}")
    return pd.read_parquet(hits[0])


# ------------------------------------------------------------------ OCV
def build_ocv(cap):
    curves, rows = {}, {}
    for f in FOLDERS:
        o = ocv_from_c20(pd.read_parquet(PROC / f / f"{cap.loc[f, 'source']}.parquet"))
        curves[f] = o
        rows[f] = {k: o[k] for k in ("cap_dis_ah", "cap_chg_ah", "charge_branch_soc_max", "top_half_gap_v")}
        rows[f]["mean_chg_minus_dis_mv"] = float(np.nanmean(o["v_chg"] - o["v_dis"]) * 1000)
    return curves, pd.DataFrame(rows).T


def ocv_object(curves, folders, key="ocv"):
    return OCVCurve(curves[folders[0]]["soc"], np.vstack([curves[f][key] for f in folders]),
                    temps=[FOLDER_TEMPS[f] for f in folders])


def ocv_discharge_ir(curves, table, folder):
    """C/20 discharge branch plus I_C20 * (R0 + sum R_i) at that SoC and temperature."""
    o = curves[folder]
    T = FOLDER_TEMPS[folder]
    r_tot = np.array([(lambda p: p.r0 + sum(p.r))(table(s, T)) for s in o["soc"]])
    return np.maximum.accumulate(o["v_dis"] + o["i_dis_median_a"] * r_tot)


def write_ocv(name, method, curves, rows):
    data = {"soc": curves[FOLDERS[0]]["soc"].tolist(), "temps": [FOLDER_TEMPS[f] for f in FOLDERS],
            "folders": FOLDERS, "method": method, "ocv": [np.asarray(rows[f]).tolist() for f in FOLDERS]}
    (OUT / f"ocv_{name}.json").write_text(json.dumps(data))


# ------------------------------------------------------------------ pulse fits
def fit_pulses(curves, cap, f, n_rc, i0=float("inf")):
    return identify_hppc(find(f, "HPPC"), ocv_object(curves, [f]), cap.loc[f, "c20_discharge_ah"],
                         FOLDER_TEMPS[f], n_rc, i0=i0)


def pooled_rmse(fits):
    use = [x.rmse_v for x in fits if x.soc_mid >= MIN_SOC]
    return float(np.sqrt(np.mean(np.square(use))))


def choose_i0(curves, cap, f):
    """Exchange current that minimises the pooled 2RC pulse-fit RMSE at one temperature."""
    scores = [(pooled_rmse(fit_pulses(curves, cap, f, 2, i0)), i0) for i0 in I0_GRID]
    best = min(scores)
    return best[1], scores


def fits_frame(f, family, n_rc, fits, slow=None):
    rows = []
    for k, ft in enumerate(fits):
        p = ft.params
        row = {"folder": f, "temp": FOLDER_TEMPS[f], "level": k, "family": family, "n_rc": n_rc,
               "soc_mid": ft.soc_mid, "rmse_mv": ft.rmse_v * 1000, "success": ft.success, "r0": p.r0, "i0": p.i0}
        for j in range(n_rc):
            row[f"r{j + 1}"] = p.r[j]
            row[f"tau{j + 1}"] = p.tau[j]
        if slow is not None and slow[k] is not None:
            s = slow[k]
            row.update(r_slow=s.r, tau_slow=s.tau, offset_mv=s.offset_v * 1000, slow_rmse_mv=s.rmse_v * 1000)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tables
def param_table(fits, family, n_rc, folders=MODEL_FOLDERS, min_soc=MIN_SOC, slow=False):
    """ParamTable on TABLE_SOC from the per-level fits. Levels with soc_mid < min_soc
    are dropped and the table is held flat outside the kept SoC range."""
    sub = fits[(fits.family == family) & (fits.n_rc == n_rc) & fits.folder.isin(folders) & (fits.soc_mid >= min_soc)]
    temps = [FOLDER_TEMPS[f] for f in folders]

    def grid(col, f):
        g = sub[sub.folder == f].dropna(subset=[col]).sort_values("soc_mid")
        return np.interp(TABLE_SOC, g.soc_mid, g[col])

    r_cols = [f"r{j + 1}" for j in range(n_rc)] + (["r_slow"] if slow else [])
    tau_cols = [f"tau{j + 1}" for j in range(n_rc)] + (["tau_slow"] if slow else [])
    r0 = np.array([grid("r0", f) for f in folders])
    r = np.array([[grid(c, f) for f in folders] for c in r_cols])
    tau = np.array([[grid(c, f) for f in folders] for c in tau_cols])
    i0 = None
    if family == "bv":
        i0 = [float(sub[sub.folder == f].i0.iloc[0]) for f in folders]
    return ParamTable(temps, TABLE_SOC, r0, r, tau, i0)


def offsets(fits, family):
    """Median OCV offset (V) per folder from the slow-branch fits."""
    sub = fits[(fits.family == family) & (fits.n_rc == 2) & (fits.soc_mid >= MIN_SOC)]
    return {f: float(sub[sub.folder == f].offset_mv.median()) / 1000 for f in FOLDERS}


def rest_offsets(curves, cap):
    """Mean (rested voltage - C/20 average) at the end of the 1 h rests, per folder (V)."""
    out, rows = {}, []
    for f in FOLDERS:
        hp = find(f, "HPPC")
        C = cap.loc[f, "c20_discharge_ah"]
        t, v, ah = hp.time_s.to_numpy(), hp.voltage_v.to_numpy(), hp.ah.to_numpy()
        o = curves[f]
        d = []
        for lv in hppc_levels(hp):
            if lv["relax"] is None:
                continue
            b = lv["relax"][1]
            soc = 1 - (ah[b] - ah[0]) / C
            if soc < MIN_SOC:
                continue
            k30 = int(np.searchsorted(t, t[b] - 1800.0))
            d.append((v[b] - np.interp(soc, o["soc"], o["v_dis"]), v[b] - np.interp(soc, o["soc"], o["ocv"]),
                      v[b] - np.interp(soc, o["soc"], o["v_chg"]), v[b] - v[k30]))
        d = np.array(d) * 1000
        out[f] = float(d[:, 1].mean()) / 1000
        rows.append({"temperature": LABEL[f], "levels": len(d), "rest - C/20 discharge (mV)": d[:, 0].mean(),
                     "rest - C/20 average (mV)": d[:, 1].mean(), "rest - C/20 charge (mV)": np.nanmean(d[:, 2]),
                     "rise over last 30 min of rest (mV)": d[:, 3].mean()})
    return out, pd.DataFrame(rows)


def pulse_resistance_table():
    """10 s discharge-pulse resistance per pulse amplitude, SoC levels 0.2 to 0.9."""
    rows = []
    cap = pd.read_csv(PROC / "capacity.csv", index_col=0)
    for f in FOLDERS:
        hp = find(f, "HPPC")
        C = cap.loc[f, "c20_discharge_ah"]
        t, i, v, ah = hp.time_s.to_numpy(), hp.current_a.to_numpy(), hp.voltage_v.to_numpy(), hp.ah.to_numpy()
        acc = {}
        for a, b in hppc_blocks(hp):
            if not 0.2 < 1 - (ah[a] - ah[0]) / C < 0.9:
                continue
            on = i[a : b + 1] > 0.5
            for n, k in enumerate((np.flatnonzero(on[1:] & ~on[:-1]) + 1 + a)[:4]):
                e = k
                while e < b and i[e] > 0.5:
                    e += 1
                if t[e - 1] - t[k] >= 8:
                    acc.setdefault(n, []).append((np.median(i[k:e]), (v[k - 1] - v[e - 1]) / i[e - 1] * 1000))
        row = {"temperature": LABEL[f]}
        for n in sorted(acc):
            r = np.array(acc[n])
            row[f"pulse {n + 1}"] = f"{r[:, 1].mean():.0f} mΩ at {r[:, 0].mean():.2f} A"
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def plot_ocv(curves, ocv_rows):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cols = plt.cm.coolwarm(np.linspace(0, 1, len(FOLDERS)))
    for c, f in zip(cols, FOLDERS):
        ax[0].plot(curves[f]["soc"], ocv_rows[f], color=c, label=LABEL[f], ls="--" if f == HELD_OUT_FOLDER else "-")
    for a, f in ((ax[1], "25degC"), (ax[2], "n20degC")):
        o = curves[f]
        a.plot(o["soc"], o["v_dis"], color="tab:red", lw=1, label="C/20 discharge")
        a.plot(o["soc"], o["v_chg"], color="tab:blue", lw=1, label="C/20 charge (CC part)")
        a.plot(o["soc"], o["ocv"], ":", color="0.4", lw=1.5, label="C/20 average")
        a.plot(o["soc"], ocv_rows[f], color="k", lw=1.5, label="model OCV (average + rest offset)")
        a.set(xlabel="SoC", ylabel="Voltage (V)", title=f"Construction at {LABEL[f]}")
    ax[0].set(xlabel="SoC", ylabel="OCV (V)", title="Model OCV per temperature (10 °C dashed: not used by the model)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ocv_curves.png", dpi=110)
    plt.close(fig)


def plot_params(fits):
    sub = fits[(fits.family == "bv") & (fits.n_rc == 2) & (fits.soc_mid >= MIN_SOC)]
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.6))
    cols = dict(zip(FOLDERS, plt.cm.coolwarm(np.linspace(0, 1, len(FOLDERS)))))
    for f in FOLDERS:
        g = sub[sub.folder == f].sort_values("soc_mid")
        st = dict(color=cols[f], marker="o", ms=3, label=LABEL[f], ls="--" if f == HELD_OUT_FOLDER else "-")
        ax[0].plot(g.soc_mid, g.r0 * 1000, **st)
        ax[1].plot(g.soc_mid, (g.r1 + g.r2) * 1000, **st)
        ax[2].plot(g.soc_mid, g.r_slow * 1000, **st)
        ax[3].plot(g.soc_mid, g.tau_slow, **st)
    ax[0].set(ylabel="R0 (mΩ)", yscale="log")
    ax[1].set(ylabel="R1 + R2 (mΩ)", yscale="log")
    ax[2].set(ylabel="R slow (mΩ)", yscale="log")
    ax[3].set(ylabel="τ slow (s)", yscale="log")
    i0 = sub.groupby("folder", sort=False).i0.first()
    ax[4].semilogy([FOLDER_TEMPS[f] for f in i0.index], i0.values, "ko-")
    ax[4].set(xlabel="Temperature (°C)", ylabel="i0 (A)", title="Butler-Volmer exchange current")
    for a in ax[:4]:
        a.set_xlabel("SoC")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    fig.suptitle("3RC + Butler-Volmer parameters from HPPC (dashed: 10 °C, fitted for reference only, not in the model)")
    fig.tight_layout()
    fig.savefig(FIG / "ecm_params.png", dpi=110)
    plt.close(fig)


def plot_fit_example(curves, cap, fits, folder="n20degC", level=4):
    hp = find(folder, "HPPC")
    a, b = hppc_blocks(hp)[level]
    seg = hp.iloc[a : b + 1]
    t, i, v = seg.time_s.to_numpy(), seg.current_a.to_numpy(), seg.voltage_v.to_numpy()
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    C = cap.loc[folder, "c20_discharge_ah"]
    T = FOLDER_TEMPS[folder]
    soc0 = 1 - (hp.ah.iloc[a] - hp.ah.iloc[0]) / C
    soc = soc0 - np.r_[0, np.cumsum(i[:-1] * dt[:-1])] / (3600 * C)
    ocv = ocv_object(curves, [folder])(soc, T)
    ocv = ocv + (v[0] - ocv[0])
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(t - t[0], v, "k", lw=1, label="measured")
    for family, c, name in (("linear", "tab:orange", "2RC"), ("bv", "tab:blue", "2RC + Butler-Volmer")):
        r = fits[(fits.folder == folder) & (fits.family == family) & (fits.n_rc == 2) & (fits.level == level)].iloc[0]
        vm = ocv - sum(rc_voltages(i, dt, r[f"r{j}"], r[f"tau{j}"]) for j in (1, 2)) - r.r0 * i - bv_overpotential(i, T, r.i0)
        ax[0].plot(t - t[0], vm, color=c, lw=1, label=f"{name}, RMSE {r.rmse_mv:.1f} mV")
        ax[1].plot(t - t[0], (vm - v) * 1000, color=c, lw=0.8)
    ax[0].set(ylabel="Voltage (V)", title=f"HPPC pulses at SoC ≈ {soc0:.2f}, {LABEL[folder]}")
    ax[1].set(ylabel="Error (mV)", xlabel="Time in block (s)")
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hppc_fit_example.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    cap = pd.read_csv(PROC / "capacity.csv", index_col=0)
    curves, ocv_summary = build_ocv(cap)

    frames, i0_rows = [], []
    for f in FOLDERS:
        hp, T, C = find(f, "HPPC"), FOLDER_TEMPS[f], cap.loc[f, "c20_discharge_ah"]
        ocv_f = ocv_object(curves, [f])
        frames.append(fits_frame(f, "linear", 1, fit_pulses(curves, cap, f, 1)))
        lin = fit_pulses(curves, cap, f, 2)
        frames.append(fits_frame(f, "linear", 2, lin, identify_slow(hp, ocv_f, C, T, lin)))
        i0, scores = choose_i0(curves, cap, f)
        bv = fit_pulses(curves, cap, f, 2, i0)
        frames.append(fits_frame(f, "bv", 2, bv, identify_slow(hp, ocv_f, C, T, bv)))
        i0_rows.append({"temperature": LABEL[f], "i0 (A)": i0, "linear 2RC pooled RMSE (mV)": 1000 * pooled_rmse(lin),
                        "2RC + BV pooled RMSE (mV)": 1000 * pooled_rmse(bv)})
        print(f"  {f}: i0 = {i0:.3g} A, pulse RMSE {1000 * pooled_rmse(lin):.1f} -> {1000 * pooled_rmse(bv):.1f} mV", flush=True)
    fits = pd.concat(frames, ignore_index=True)
    fits.to_csv(OUT / "hppc_fits.csv", index=False)

    # --- parameter tables and OCV tables for each variant
    t_1rc = param_table(fits, "linear", 1)
    t_2rc = param_table(fits, "linear", 2)
    tables = {
        "1rc": t_1rc, "2rc": t_2rc,
        "2rc_allsoc": param_table(fits, "linear", 2, min_soc=0.0),
        "2rc_rest": t_2rc,
        "3rc": param_table(fits, "linear", 2, slow=True),
        "2rc_bv": param_table(fits, "bv", 2),
        "3rc_bv": param_table(fits, "bv", 2, slow=True),
    }
    rest_off, rest_table = rest_offsets(curves, cap)
    off_lin, off_bv = offsets(fits, "linear"), offsets(fits, "bv")
    avg = {f: curves[f]["ocv"] for f in FOLDERS}
    write_ocv("average", "average of C/20 discharge and charge branches", curves, avg)
    write_ocv("disir", "C/20 discharge branch + I_C20 * (R0 + R1 + R2), linear 2RC table", curves,
              {f: ocv_discharge_ir(curves, t_2rc, f) for f in FOLDERS})
    rest_rows = {f: avg[f] + rest_off[f] for f in FOLDERS}
    write_ocv("rest", "C/20 average + mean offset of the 1 h rest points (HPPC)", curves, rest_rows)
    write_ocv("slow_linear", "C/20 average + median offset from the slow-branch fits (linear family)", curves,
              {f: avg[f] + off_lin[f] for f in FOLDERS})
    write_ocv("slow_bv", "C/20 average + median offset from the slow-branch fits (BV family)", curves,
              {f: avg[f] + off_bv[f] for f in FOLDERS})
    ocv_of = {"1rc": "disir", "2rc": "disir", "2rc_allsoc": "disir", "2rc_rest": "rest", "3rc": "slow_linear",
              "2rc_bv": "rest", "3rc_bv": "slow_bv"}
    describe = {
        "1rc": "1RC, linear", "2rc": "2RC, linear (M2 baseline)", "2rc_allsoc": "2RC, linear, all SoC levels kept",
        "2rc_rest": "2RC, linear, rest-anchored OCV", "3rc": "2RC + slow branch, rest-anchored OCV",
        "2rc_bv": "2RC + Butler-Volmer, rest-anchored OCV", "3rc_bv": "2RC + slow branch + Butler-Volmer, rest-anchored OCV",
    }
    for name, tab in tables.items():
        tab.to_json(OUT / f"params_{name}.json")
    (OUT / "variants.json").write_text(json.dumps(
        {n: {"params": f"params_{n}.json", "ocv": f"ocv_{ocv_of[n]}.json", "description": describe[n]} for n in tables},
        indent=1))

    plot_ocv(curves, rest_rows)
    plot_params(fits)
    plot_fit_example(curves, cap, fits)

    # --- tables
    summ = (fits[fits.soc_mid >= MIN_SOC].groupby(["folder", "family", "n_rc"], sort=False)
            .agg(levels=("rmse_mv", "size"), median_rmse_mv=("rmse_mv", "median"), max_rmse_mv=("rmse_mv", "max"),
                 all_converged=("success", "all")).reset_index())
    (TAB / "ecm_fit_rmse.md").write_text("\n".join([
        "# ECM fit quality on HPPC pulse blocks", "",
        f"One fit per SoC level at or above {MIN_SOC} (four 10 s discharge/charge pulse pairs and their rests). "
        "Voltage RMSE in mV. `bv` adds the Butler-Volmer term.", "",
        summ.round(2).to_markdown(index=False), "", "# OCV construction (C/20)", "",
        ocv_summary.round(4).to_markdown(), ""]))
    off = pd.DataFrame({"rest points": rest_off, "slow fit, linear": off_lin, "slow fit, BV": off_bv}) * 1000
    off.index = [LABEL[f] for f in off.index]
    slow = fits[(fits.family == "bv") & (fits.soc_mid >= MIN_SOC)].groupby("folder", sort=False).agg(
        r_slow_mohm=("r_slow", lambda x: 1000 * x.median()), tau_slow_s=("tau_slow", "median"),
        slow_rmse_mv=("slow_rmse_mv", "median"))
    slow.index = [LABEL[f] for f in slow.index]
    (TAB / "m2_cold_diagnostics.md").write_text("\n".join([
        "# Why the baseline 2RC model is weak in the cold, and what was added", "",
        "All numbers come from the characterisation tests (C/20 and HPPC), not from drive cycles.", "",
        "## 1. Slow polarisation, not hysteresis", "",
        "Voltage at the end of each 1 h rest in the HPPC test compared with the C/20 curves at the same SoC. "
        "The rested cell sits close to the C/20 average at every temperature, and in the cold it is still "
        "rising after an hour. The large C/20 charge/discharge gap in the cold is therefore mostly slow "
        "relaxation, which a 60 s time constant cannot follow.", "",
        rest_table.round(1).to_markdown(index=False), "",
        "Added: a slow RC branch per SoC level and temperature, fitted on the discharge step and 1 h rest "
        "before each pulse block (fast parameters fixed), plus a constant OCV offset. Median per temperature:", "",
        slow.round(1).to_markdown(), "", "OCV offset relative to the C/20 average (mV):", "",
        off.round(1).to_markdown(), "",
        "## 2. Resistance falls with current in the cold", "",
        "Resistance at the end of each 10 s discharge pulse (SoC levels 0.2 to 0.9, mean):", "",
        pulse_resistance_table().to_markdown(index=False), "",
        "Added: a Butler-Volmer term with one exchange current i0 per temperature, chosen on a grid to "
        "minimise the pooled 2RC pulse-fit RMSE. A large i0 means the term is nearly linear and is absorbed "
        "by R0.", "", pd.DataFrame(i0_rows).round(2).to_markdown(index=False), ""]))
    print(summ.round(2).to_string(index=False))
    print(pd.DataFrame(i0_rows).round(2).to_string(index=False))
    print(off.round(1).to_string())
    print(slow.round(1).to_string())


if __name__ == "__main__":
    main()
