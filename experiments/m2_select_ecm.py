"""Choose the ECM identification variant using TRAINING and VALIDATION drive cycles only.

Variants (all fitted on HPPC, never on drive cycles):
  A  all four pulse pairs, all SoC levels
  B  all four pulse pairs, SoC levels >= 0.15 (table held flat below)
  C  first two pulse pairs only (1C and 2C discharge), SoC levels >= 0.15

Then, with the best variant, OCV definitions (see socest.characterize):
  average, discharge (raw C/20 discharge branch), discharge+IR

Score: median open-loop voltage RMSE over train + val cycles at the five
non-held-out temperatures, 1 Hz. Test cycles are not touched here.
Writes results/tables/m2_ecm_selection.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import m2_characterize as m2  # noqa: E402
from socest.cell import PROC, capacity_fn, load_ocv  # noqa: E402
from socest.ecm import TheveninModel  # noqa: E402
from socest.resample import to_uniform  # noqa: E402
from socest.split import HELD_OUT_FOLDER, load_index  # noqa: E402

VARIANTS = {"A": dict(max_pairs=None, min_soc=0.0), "B": dict(max_pairs=None, min_soc=0.15),
            "C": dict(max_pairs=2, min_soc=0.15)}


def main():
    cap = pd.read_csv(PROC / "capacity.csv", index_col=0)
    curves, _ = m2.build_ocv(cap)
    ocv = load_ocv()
    cap_of = capacity_fn()
    folders = [f for f in m2.FOLDERS if f != HELD_OUT_FOLDER]
    ix = load_index(PROC)
    sel = ix[ix.split.isin(["train", "val"])]
    data = [(r, to_uniform(pd.read_parquet(PROC / r.parquet), 1.0)) for _, r in sel.iterrows()]

    fits_cache, rows = {}, []
    for name, cfg in VARIANTS.items():
        key = cfg["max_pairs"]
        if key not in fits_cache:
            fits_cache[key] = m2.fit_all(curves, cap, max_pairs=key, verbose=False)
        fits = fits_cache[key]
        for n_rc in (1, 2):
            table = m2.param_table(fits, n_rc, folders, min_soc=cfg["min_soc"])
            for r, u in data:
                i, v, T = u.current_a.to_numpy(), u.voltage_v.to_numpy(), u.temp_degc.to_numpy()
                m = TheveninModel(ocv, table, cap_of(T[0]))
                vs, _ = m.simulate(i, 1.0, T, soc0=1.0)
                rows.append({"variant": name, "n_rc": n_rc, "folder": r.folder, "test": r.test,
                             "v_rmse_mv": 1000 * np.sqrt(np.mean((vs - v) ** 2))})
            print(f"variant {name} {n_rc}RC done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "m2_ecm_selection.csv", index=False)
    by_t = res.pivot_table(index=["variant", "n_rc"], columns="folder", values="v_rmse_mv", aggfunc="median")
    by_t = by_t[[f for f in m2.FOLDERS if f in by_t.columns]]
    by_t["overall median"] = res.groupby(["variant", "n_rc"]).v_rmse_mv.median()
    # OCV definitions, with variant B 2RC (best overall above)
    from socest.ocv import OCVCurve
    from socest.datasets.lg_hg2 import FOLDER_TEMPS
    tab = m2.param_table(fits_cache[None], 2, folders, min_soc=0.15)
    soc = curves[folders[0]]["soc"]
    defs = {"average": [curves[f]["ocv"] for f in folders],
            "discharge": [np.maximum.accumulate(curves[f]["v_dis"]) for f in folders],
            "discharge+IR": [m2.ocv_discharge_ir(curves, tab, f) for f in folders]}
    orows = []
    for name, rows_ in defs.items():
        o = OCVCurve(soc, np.vstack(rows_), temps=[FOLDER_TEMPS[f] for f in folders])
        for r, u in data:
            i, v, T = u.current_a.to_numpy(), u.voltage_v.to_numpy(), u.temp_degc.to_numpy()
            vs, _ = TheveninModel(o, tab, cap_of(T[0])).simulate(i, 1.0, T, soc0=1.0)
            e = 1000 * (vs - v)
            orows.append({"ocv": name, "folder": r.folder, "v_rmse_mv": np.sqrt(np.mean(e**2)), "bias_mv": e.mean()})
        print(f"OCV {name} done", flush=True)
    ores = pd.DataFrame(orows)
    o_t = ores.pivot_table(index="ocv", columns="folder", values="v_rmse_mv", aggfunc="median")
    o_t = o_t[[f for f in m2.FOLDERS if f in o_t.columns]]
    o_t["overall median"] = ores.groupby("ocv").v_rmse_mv.median()
    o_b = ores.pivot_table(index="ocv", columns="folder", values="bias_mv", aggfunc="median")
    o_b = o_b[[f for f in m2.FOLDERS if f in o_b.columns]]

    md = ["# ECM variant selection (train + val drive cycles only, open-loop voltage RMSE, mV, median)", "",
          __doc__.split("Then,")[0].split("Variants")[1].strip(), "", by_t.round(1).to_markdown(), "",
          "# OCV definition (variant B, 2RC)", "", "Median voltage RMSE (mV):", "", o_t.round(1).to_markdown(), "",
          "Median bias, model minus measured (mV):", "", o_b.round(1).to_markdown(), "",
          "Selected: variant B, 2RC, OCV = discharge+IR (lowest overall median RMSE).", ""]
    (ROOT / "results" / "tables" / "m2_ecm_selection.md").write_text("\n".join(md))
    print(by_t.round(1).to_string())
    print(o_t.round(1).to_string())


if __name__ == "__main__":
    main()
