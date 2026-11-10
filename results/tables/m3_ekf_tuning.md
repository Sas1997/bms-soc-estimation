# EKF noise tuning on the validation cycles

The reference SoC is coulomb counting with a known start and the laboratory
current sensor, so with a correct initial SoC and a perfect sensor the best
"filter" is one that ignores voltage altogether. Tuning on that case alone would
switch the voltage feedback off. The objective therefore averages three cases
on each validation cycle:
    nominal            correct initial SoC, clean signals
    initial SoC -20    estimator starts at 0.80 instead of 1.00
    current bias       +50 mA added to the measured current
Score = mean over cases of the mean SoC RMSE over the validation cycles.

Grid (per 1 s step):
    k_r    scale on the temperature-dependent model voltage error sigma_v(T)
    q_soc  process noise on SoC (fraction)
    q_v    process noise on each RC voltage (V)
Fixed: P0 = diag(0.2^2, 0.01^2, ...).

## Model `2rc`

SoC RMSE in percentage points, best eight settings:

|   k_r |   q_soc |   q_v |   nominal |   initial SoC -20 pts |   current bias +50 mA |   score |
|------:|--------:|------:|----------:|----------------------:|----------------------:|--------:|
|   1   |       0 | 0.001 |     2.881 |                 3.483 |                 2.971 |   3.112 |
|   1   |       0 | 0.001 |     2.897 |                 3.49  |                 3.003 |   3.13  |
|   4   |       0 | 0.001 |     2.904 |                 3.971 |                 2.871 |   3.249 |
|   4   |       0 | 0.001 |     2.903 |                 3.97  |                 2.878 |   3.251 |
|   4   |       0 | 0.001 |     2.963 |                 3.966 |                 2.867 |   3.265 |
|   4   |       0 | 0     |     3.025 |                 4.05  |                 2.873 |   3.316 |
|   4   |       0 | 0     |     3.023 |                 4.049 |                 2.877 |   3.316 |
|   0.5 |       0 | 0.001 |     2.832 |                 4.144 |                 2.995 |   3.324 |

Worst setting on the grid scores 5.15.

## Model `3rc_dc_dis`

SoC RMSE in percentage points, best eight settings:

|   k_r |   q_soc |   q_v |   nominal |   initial SoC -20 pts |   current bias +50 mA |   score |
|------:|--------:|------:|----------:|----------------------:|----------------------:|--------:|
|     2 |       0 | 0.001 |     1.642 |                 2.752 |                 1.866 |   2.087 |
|     4 |       0 | 0     |     1.761 |                 2.761 |                 1.755 |   2.092 |
|     4 |       0 | 0     |     1.761 |                 2.761 |                 1.762 |   2.095 |
|     4 |       0 | 0     |     1.786 |                 2.733 |                 1.788 |   2.102 |
|     1 |       0 | 0.001 |     1.341 |                 3.241 |                 1.783 |   2.122 |
|     4 |       0 | 0.001 |     1.64  |                 3.03  |                 1.76  |   2.143 |
|     2 |       0 | 0.001 |     1.727 |                 2.813 |                 2.13  |   2.223 |
|     2 |       0 | 0.001 |     1.729 |                 2.814 |                 2.135 |   2.226 |

Worst setting on the grid scores 4.10.
