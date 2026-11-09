# Slow-branch calibration on the training drive cycles

Why: the slow branch fitted on the HPPC relaxations is poorly identified (time
constants scatter from 100 s to 10 000 s between SoC levels) and overshoots on
drive cycles (results/tables/m2_ecm_selection.md, variants 3rc and 3rc_bv).
The neural networks learn from the training drive cycles, so the physics model
may use the same cycles. Validation, test and all 10 degC data stay untouched.

## `3rc_dc_avg`: 2RC (HPPC) + calibrated slow branch, OCV shape: C/20 average

Chosen: tau = 630 s, thermal gain g = 1.0, training voltage RMSE 32.1 mV.

|        |   OCV offset c (mV) |   R slow at SoC 1 (mΩ) |   R slow at SoC 0 (mΩ) |
|:-------|--------------------:|-----------------------:|-----------------------:|
| -20 °C |               -20.1 |                      0 |                  371.8 |
| -10 °C |               -26.9 |                      0 |                  126.8 |
| 0 °C   |               -10.2 |                      0 |                   14.4 |
| 25 °C  |                 7.3 |                      0 |                   43.5 |
| 40 °C  |                 5.9 |                      0 |                   37.6 |

Best training RMSE for each thermal gain:

|   g |   tau_s |   train_rmse_mv |
|----:|--------:|----------------:|
| 0   |  1582.2 |            36.4 |
| 0.5 |   998.6 |            33.7 |
| 1   |   630.3 |            32.1 |

## `3rc_dc_dis`: 2RC (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR

Chosen: tau = 316 s, thermal gain g = 1.0, training voltage RMSE 28.2 mV.

|        |   OCV offset c (mV) |   R slow at SoC 1 (mΩ) |   R slow at SoC 0 (mΩ) |
|:-------|--------------------:|-----------------------:|-----------------------:|
| -20 °C |                12.9 |                      0 |                  325.3 |
| -10 °C |                 2.7 |                      0 |                  124.8 |
| 0 °C   |                 0.9 |                      0 |                    1.7 |
| 25 °C  |                10.8 |                      0 |                    3.3 |
| 40 °C  |                 6.9 |                      0 |                    2.5 |

Best training RMSE for each thermal gain:

|   g |   tau_s |   train_rmse_mv |
|----:|--------:|----------------:|
| 0   |   998.6 |            34.2 |
| 0.5 |   630.3 |            30.7 |
| 1   |   316   |            28.2 |

## `3rc_bv_dc_avg`: 2RC + Butler-Volmer (HPPC) + calibrated slow branch, OCV shape: C/20 average

Chosen: tau = 1992 s, thermal gain g = 1.0, training voltage RMSE 33.2 mV.

|        |   OCV offset c (mV) |   R slow at SoC 1 (mΩ) |   R slow at SoC 0 (mΩ) |
|:-------|--------------------:|-----------------------:|-----------------------:|
| -20 °C |                -1.1 |                   51.1 |                  387.4 |
| -10 °C |               -11.6 |                    0   |                  128.3 |
| 0 °C   |                 8.1 |                    0   |                   30.3 |
| 25 °C  |                14.3 |                    0   |                   78.2 |
| 40 °C  |                 9.5 |                    0   |                   63.3 |

Best training RMSE for each thermal gain:

|   g |   tau_s |   train_rmse_mv |
|----:|--------:|----------------:|
| 0   |  5000   |            38.5 |
| 0.5 |  3155.7 |            35.3 |
| 1   |  1991.6 |            33.2 |

## `3rc_bv_dc_dis`: 2RC + Butler-Volmer (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR

Chosen: tau = 793 s, thermal gain g = 1.0, training voltage RMSE 30.3 mV.

|        |   OCV offset c (mV) |   R slow at SoC 1 (mΩ) |   R slow at SoC 0 (mΩ) |
|:-------|--------------------:|-----------------------:|-----------------------:|
| -20 °C |                31.5 |                      0 |                  336.6 |
| -10 °C |                16   |                      0 |                  101.3 |
| 0 °C   |                16.6 |                      0 |                    3.1 |
| 25 °C  |                12.1 |                      0 |                    5.2 |
| 40 °C  |                 8   |                      0 |                    4   |

Best training RMSE for each thermal gain:

|   g |   tau_s |   train_rmse_mv |
|----:|--------:|----------------:|
| 0   |  2507   |            36.6 |
| 0.5 |  1257   |            33   |
| 1   |   793.3 |            30.3 |
