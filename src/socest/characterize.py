"""Cell characterisation from the LG HG2 C/20 and HPPC tests.

OCV candidates from the C/20 test:
  average       mean of the C/20 discharge and charge branches at equal SoC, the
                usual way to cancel most of the low-rate drop (Plett, Battery
                Management Systems Vol. 1, 2015, ch. 2)
  discharge+IR  C/20 discharge branch plus I_C20 * (R0 + sum R_i) from the HPPC
                table; selected for the model (experiments/m2_select_ecm.py)
                because the drive cycles are discharge dominated and the cell
                relaxes toward the discharge side of its hysteresis loop. Each branch is
normalised by its own throughput (tester Ah counter). The constant-voltage tail
of the charge branch is dropped because its voltage is pinned at 4.2 V. Where
only the discharge branch exists (top of the SoC range), the discharge branch
is shifted by the mean half-gap measured over the adjacent overlap.

HPPC: the protocol (inspected in the 25 degC file) repeats, at each SoC level,
a 1 h rest followed by four pulse pairs (10 s discharge at 1C to 6C, 40 s rest,
10 s charge, 10 min rest), then a 0.3C discharge to the next level. One ECM is
fitted per SoC level on the whole four-pulse block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets.lg_hg2 import first_sustained_segment
from .identify import SegmentFit, fit_segment
from .ocv import OCVCurve

SOC_GRID = np.linspace(0.0, 1.0, 201)


def _branch(df, a, b, cap, discharge: bool):
    ah = df["ah"].to_numpy()[a : b + 1]
    v = df["voltage_v"].to_numpy()[a : b + 1]
    q = ah - ah[0]
    soc = 1.0 - q / cap if discharge else -q / cap
    return soc, v


def ocv_from_c20(df: pd.DataFrame, cv_limit_v: float = 4.199):
    """Return dict with soc grid, discharge, charge and averaged OCV (V).

    df is a processed C/20 frame (i > 0 discharge, ah positive = discharged).
    """
    t, i = df["time_s"].to_numpy(), df["current_a"].to_numpy()
    a, b, s = first_sustained_segment(t, i)
    if s != 1:
        raise ValueError("C/20 file must start with a discharge")
    ah = df["ah"].to_numpy()
    cap_d = ah[b] - ah[a]
    soc_d, v_d = _branch(df, a, b, cap_d, discharge=True)

    rest = first_sustained_segment(t[b + 1 :], i[b + 1 :])
    c, d = rest[0] + b + 1, rest[1] + b + 1
    cap_c = ah[c] - ah[d]
    soc_c, v_c = _branch(df, c, d, cap_c, discharge=False)
    keep = v_c < cv_limit_v  # drop the constant-voltage tail
    soc_c, v_c = soc_c[keep], v_c[keep]

    # Interpolate both branches on the grid (np.interp needs increasing x).
    o = np.argsort(soc_d)
    vd = np.interp(SOC_GRID, soc_d[o], v_d[o])
    o = np.argsort(soc_c)
    vc = np.interp(SOC_GRID, soc_c[o], v_c[o], left=np.nan, right=np.nan)
    both = ~np.isnan(vc)
    avg = np.where(both, 0.5 * (vd + vc), np.nan)

    # Top end: discharge branch plus the mean half-gap over the last 5 % of overlap.
    hi = np.flatnonzero(both)[-1]
    win = (SOC_GRID > SOC_GRID[hi] - 0.05) & both
    half_gap = np.mean(0.5 * (vc[win] - vd[win]))
    avg[hi + 1 :] = vd[hi + 1 :] + half_gap
    # Bottom end, if the charge branch starts above 0 (it should not, but be safe).
    lo = np.flatnonzero(both)[0]
    if lo > 0:
        win = (SOC_GRID < SOC_GRID[lo] + 0.05) & both
        avg[:lo] = vd[:lo] + np.mean(0.5 * (vc[win] - vd[win]))

    # Enforce monotonicity (tiny wiggles from logging resolution).
    avg = np.maximum.accumulate(avg)
    return {
        "soc": SOC_GRID, "ocv": avg, "v_dis": vd, "v_chg": vc,
        "cap_dis_ah": float(cap_d), "cap_chg_ah": float(cap_c),
        "i_dis_median_a": float(np.median(i[a : b + 1])),
        "charge_branch_soc_max": float(SOC_GRID[hi]), "top_half_gap_v": float(half_gap),
    }


def hppc_blocks(df: pd.DataFrame, long_rest_s: float = 1800.0):
    """Index ranges (a, b) of each pulse block: from 10 s before the first pulse
    after a long rest, to the start of the next long (>60 s) current segment."""
    t, i = df["time_s"].to_numpy(), df["current_a"].to_numpy()
    active = np.abs(i) > 0.05
    blocks = []
    k = 0
    n = len(t)
    while k < n:
        # find the start of a long rest
        if active[k]:
            k += 1
            continue
        j = k
        while j < n and not active[j]:
            j += 1
        if j >= n:
            break
        # The file itself opens at rest after a full charge, so a rest at k == 0
        # counts as a block start whatever its length.
        if t[j] - t[k] >= long_rest_s or k == 0:
            start = int(np.searchsorted(t, t[j] - 10.0))
            # block ends where the next sustained (>60 s) current segment begins
            seg = first_sustained_segment(t[j:], i[j:], min_duration_s=60.0, min_current_a=0.05)
            end = j + seg[0] if seg is not None else n - 1
            if end - start > 100:
                blocks.append((start, end))
            k = end if end > j else j + 1
        else:
            k = j
    return blocks


def truncate_to_pairs(df: pd.DataFrame, a: int, b: int, max_pairs: int) -> int:
    """End index of a block keeping only its first ``max_pairs`` pulse pairs.

    A pair starts at a rising edge of discharge current above 1 A; the block is
    cut 1 s before the (max_pairs + 1)-th such edge (inside the 10 min rest
    that precedes it, so the kept part ends at rest).
    """
    t = df["time_s"].to_numpy()[a : b + 1]
    on = df["current_a"].to_numpy()[a : b + 1] > 1.0
    rising = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    if on[0]:
        rising = np.r_[0, rising]
    if len(rising) <= max_pairs:
        return b
    cut = int(np.searchsorted(t, t[rising[max_pairs]] - 1.0))
    return a + cut


def identify_hppc(df: pd.DataFrame, ocv: OCVCurve, capacity_ah: float, temp: float, n_rc: int,
                  max_pairs: int | None = None):
    """Fit one n-RC model per HPPC block. SoC from the tester Ah counter, starting at 1.

    max_pairs limits each block to its first pulse pairs (1C and 2C discharge
    for max_pairs = 2 in this protocol).
    """
    fits: list[SegmentFit] = []
    ah = df["ah"].to_numpy()
    for a, b in hppc_blocks(df):
        if max_pairs is not None:
            b = truncate_to_pairs(df, a, b, max_pairs)
        soc0 = 1.0 - (ah[a] - ah[0]) / capacity_ah
        seg = df.iloc[a : b + 1]
        fits.append(fit_segment(seg["time_s"].to_numpy(), seg["current_a"].to_numpy(),
                                seg["voltage_v"].to_numpy(), soc0, capacity_ah, ocv, temp, n_rc=n_rc))
    return fits
