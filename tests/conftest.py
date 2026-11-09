import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socest.ecm import ECMParams, TheveninModel  # noqa: E402
from socest.ocv import OCVCurve  # noqa: E402

# Synthetic cell used only for unit tests. Values are of a plausible order for a
# 3 Ah 18650 cell but are NOT identified from any dataset.
SOC_BP = np.linspace(0.0, 1.0, 11)
OCV_BP = np.array([3.00, 3.35, 3.48, 3.56, 3.62, 3.68, 3.76, 3.86, 3.96, 4.07, 4.18])
PARAMS_2RC = ECMParams(r0=0.020, r=(0.010, 0.015), tau=(10.0, 200.0))
CAPACITY_AH = 3.0


@pytest.fixture
def ocv():
    return OCVCurve(SOC_BP, OCV_BP)


@pytest.fixture
def model(ocv):
    return TheveninModel(ocv, PARAMS_2RC, CAPACITY_AH)


@pytest.fixture
def pulse_profile():
    """1 Hz profile: rests, discharge pulses, a regen pulse. Current in A, i>0 discharge."""
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(12):
        blocks.append(np.zeros(60))
        blocks.append(np.full(120, rng.uniform(1.0, 6.0)))
        blocks.append(np.full(30, -rng.uniform(0.5, 3.0)))
    return np.concatenate(blocks)
