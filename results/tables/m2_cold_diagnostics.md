# Why the baseline 2RC model is weak in the cold, and what was added

All numbers come from the characterisation tests (C/20 and HPPC), not from drive cycles.

## 1. Slow polarisation, not hysteresis

Voltage at the end of each 1 h rest in the HPPC test compared with the C/20 curves at the same SoC. The rested cell sits close to the C/20 average at every temperature, and in the cold it is still rising after an hour. The large C/20 charge/discharge gap in the cold is therefore mostly slow relaxation, which a 60 s time constant cannot follow.

| temperature   |   levels |   rest - C/20 discharge (mV) |   rest - C/20 average (mV) |   rest - C/20 charge (mV) |   rise over last 30 min of rest (mV) |
|:--------------|---------:|-----------------------------:|---------------------------:|--------------------------:|-------------------------------------:|
| -20 °C        |        8 |                         64.1 |                      -13   |                     -88.9 |                                  9.9 |
| -10 °C        |        8 |                         34.6 |                       -9.1 |                     -52.8 |                                  5.5 |
| 0 °C          |        8 |                         23.1 |                       -7.1 |                     -37.4 |                                  3.3 |
| 10 °C         |        8 |                         13.5 |                       -9.7 |                     -32.9 |                                  1.3 |
| 25 °C         |        8 |                         13.4 |                       -7.4 |                     -28.3 |                                  0.6 |
| 40 °C         |        8 |                          8.8 |                       -9.7 |                     -28.2 |                                  0.3 |

Added: a slow RC branch per SoC level and temperature, fitted on the discharge step and 1 h rest before each pulse block (fast parameters fixed), plus a constant OCV offset. Median per temperature:

|        |   r_slow_mohm |   tau_slow_s |   slow_rmse_mv |
|:-------|--------------:|-------------:|---------------:|
| -20 °C |         123.1 |       1062.3 |           10   |
| -10 °C |         145.3 |       2297.2 |            6.2 |
| 0 °C   |          21.1 |       1801.2 |            3.9 |
| 10 °C  |          32.6 |       5500   |            2.7 |
| 25 °C  |          10   |        980.9 |            1.4 |
| 40 °C  |           5.8 |        944.8 |            1.1 |

OCV offset relative to the C/20 average (mV):

|        |   rest points |   slow fit, linear |   slow fit, BV |
|:-------|--------------:|-------------------:|---------------:|
| -20 °C |         -13   |              -24   |          -14.1 |
| -10 °C |          -9.1 |              -12.8 |            2.5 |
| 0 °C   |          -7.1 |               -8.8 |           -5.5 |
| 10 °C  |          -9.7 |               -6.7 |           -5.2 |
| 25 °C  |          -7.4 |               -5.3 |           -5.3 |
| 40 °C  |          -9.7 |               -7.1 |           -7   |

## 2. Resistance falls with current in the cold

Resistance at the end of each 10 s discharge pulse (SoC levels 0.2 to 0.9, mean):

| temperature   | pulse 1          | pulse 2          | pulse 3          | pulse 4          |
|:--------------|:-----------------|:-----------------|:-----------------|:-----------------|
| -20 °C        | 191 mΩ at 0.75 A | 180 mΩ at 1.50 A | 159 mΩ at 3.00 A | 134 mΩ at 6.00 A |
| -10 °C        | 96 mΩ at 1.50 A  | 92 mΩ at 3.00 A  | 84 mΩ at 6.00 A  | 74 mΩ at 12.00 A |
| 0 °C          | 57 mΩ at 3.00 A  | 55 mΩ at 6.00 A  | 51 mΩ at 12.00 A | 48 mΩ at 18.00 A |
| 10 °C         | 40 mΩ at 2.90 A  | 39 mΩ at 5.80 A  | 38 mΩ at 11.60 A | 37 mΩ at 17.40 A |
| 25 °C         | 28 mΩ at 3.00 A  | 27 mΩ at 6.00 A  | 27 mΩ at 12.00 A | 26 mΩ at 18.00 A |
| 40 °C         | 24 mΩ at 3.00 A  | 24 mΩ at 6.00 A  | 23 mΩ at 12.00 A | 23 mΩ at 18.00 A |

Added: a Butler-Volmer term with one exchange current i0 per temperature, chosen on a grid to minimise the pooled 2RC pulse-fit RMSE. A large i0 means the term is nearly linear and is absorbed by R0.

| temperature   |   i0 (A) |   linear 2RC pooled RMSE (mV) |   2RC + BV pooled RMSE (mV) |
|:--------------|---------:|------------------------------:|----------------------------:|
| -20 °C        |     0.18 |                         25.7  |                       17.12 |
| -10 °C        |     0.56 |                         20.48 |                       14.86 |
| 0 °C          |     1.78 |                         13.08 |                       10.77 |
| 10 °C         |     3.16 |                          7.19 |                        6.83 |
| 25 °C         |    10    |                          3.43 |                        3.39 |
| 40 °C         |    10    |                          3.06 |                        3.03 |
