# ECM open-loop voltage prediction on drive cycles (1 Hz)

Median voltage RMSE over cycles (mV). Models:

- `1rc`: 1RC, linear
- `2rc`: 2RC, linear (M2 baseline)
- `3rc_dc_dis`: 2RC (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR

At 10 °C every model interpolates its parameters between 0 and 25 °C; no 10 °C data was used anywhere.

## Test cycles only (UDDS, HWFET, LA92, US06, and all cycles at 10 °C)

| temperature      |   1rc |   2rc |   3rc_dc_dis |
|:-----------------|------:|------:|-------------:|
| -20 °C           | 125.9 | 104   |         39.9 |
| -10 °C           |  81.6 |  60.5 |         25.8 |
| 0 °C             |  42.7 |  27.6 |         27.3 |
| 10 °C (held out) |  21.3 |  34.7 |         31.3 |
| 25 °C            |  13   |  16.9 |         12.1 |
| 40 °C            |   9.2 |  11.1 |         10   |

## All usable cycles (train, validation and test)

| temperature      |   1rc |   2rc |   3rc_dc_dis |
|:-----------------|------:|------:|-------------:|
| -20 °C           | 116.7 |  98.4 |         43.4 |
| -10 °C           |  78.7 |  63.1 |         30.2 |
| 0 °C             |  38.8 |  32.9 |         32.5 |
| 10 °C (held out) |  21.3 |  34.7 |         31.3 |
| 25 °C            |  13.1 |  16.9 |         14.8 |
| 40 °C            |  10.1 |  12.9 |         10.4 |

# Coulomb counting on the test sets (SoC error, percentage points, mean over cycles)

The reference SoC is itself built from the tester Ah counter with a known full-charge start, so coulomb counting with the correct initial SoC is near-exact by construction; it is shown to make that explicit. Its weaknesses are an unknown initial SoC and sensor bias, which it never corrects.

|                                       |   rmse |   mae |   max_abs |
|:--------------------------------------|-------:|------:|----------:|
| ('test_unseen_cycle', 'correct init') |   0.11 |  0.09 |      0.18 |
| ('test_unseen_cycle', 'init -20 pts') |  20.03 | 20.03 |     20.13 |
| ('test_unseen_cycle', '+50 mA bias')  |   2.33 |  2.02 |      4.03 |
| ('test_unseen_temp', 'correct init')  |   0.3  |  0.26 |      0.5  |
| ('test_unseen_temp', 'init -20 pts')  |  19.74 | 19.74 |     20    |
| ('test_unseen_temp', '+50 mA bias')   |   1.98 |  1.71 |      3.45 |
