# EKF noise tuning on the validation cycles

The reference SoC is coulomb counting with a known start and the laboratory
current sensor, so with a correct initial SoC and a perfect sensor the best
"filter" is one that ignores voltage altogether. Tuning on that case alone would
switch the voltage feedback off. The objective therefore averages four cases
on each validation cycle:
    nominal            correct initial SoC, clean signals
    initial SoC -20    estimator starts at 0.80 instead of 1.00, cell at rest
    late start -20     estimation starts 20 min into the cycle, under load,
                       20 points below the true SoC
    current bias       +50 mA added to the measured current
Score = mean over cases of the mean SoC RMSE over the validation cycles.

The late-start case was added after the first full evaluation. Without it, a
wide grid lets the 2RC filter put all its process noise on the RC voltages: it
then corrects SoC once from the rest voltage and coulomb-counts afterwards,
which scores well on the other three cases and recovers poorly under load.
This is the one protocol decision that was informed by test-set results; it is
recorded in the README.

Grid (per 1 s step):
    k_r    scale on the temperature-dependent model voltage error sigma_v(T)
    q_soc  process noise on SoC (fraction)
    q_v    process noise on each RC voltage (V)
Fixed: P0 = diag(0.2^2, 0.01^2, ...).

## Model `2rc`

SoC RMSE in percentage points, best eight settings:

|   k_r |   q_soc |   q_v |   nominal |   initial SoC -20 pts |   late start, SoC -20 pts |   current bias +50 mA |   score |
|------:|--------:|------:|----------:|----------------------:|--------------------------:|----------------------:|--------:|
|   0.5 |  1e-06  |  0.01 |     0.988 |                 3.533 |                     5.109 |                 2.418 |   3.012 |
|   0.5 |  1e-05  |  0.01 |     0.99  |                 3.533 |                     5.109 |                 2.416 |   3.012 |
|   1   |  0.0001 |  0.01 |     1.463 |                 3.58  |                     4.683 |                 2.467 |   3.048 |
|   0.5 |  0.0001 |  0.01 |     1.121 |                 3.606 |                     5.124 |                 2.351 |   3.05  |
|   1   |  1e-05  |  0.01 |     1.395 |                 3.612 |                     4.684 |                 2.553 |   3.061 |
|   1   |  1e-06  |  0.01 |     1.395 |                 3.612 |                     4.684 |                 2.555 |   3.061 |
|   2   |  0.0001 |  0.01 |     1.803 |                 3.792 |                     4.506 |                 2.563 |   3.166 |
|   2   |  1e-05  |  0.01 |     1.767 |                 3.822 |                     4.494 |                 2.631 |   3.179 |

Worst setting on the grid scores 5.77. The best setting is at the edge of the grid for: k_r, q_soc, q_v.

## Model `3rc_dc_dis`

SoC RMSE in percentage points, best eight settings:

|   k_r |   q_soc |    q_v |   nominal |   initial SoC -20 pts |   late start, SoC -20 pts |   current bias +50 mA |   score |
|------:|--------:|-------:|----------:|----------------------:|--------------------------:|----------------------:|--------:|
|     4 |  1e-05  | 0.0001 |     1.761 |                 2.761 |                     1.995 |                 1.755 |   2.068 |
|     4 |  1e-06  | 0.0001 |     1.761 |                 2.761 |                     1.993 |                 1.762 |   2.069 |
|     2 |  0.0001 | 0.001  |     1.642 |                 2.752 |                     2.111 |                 1.866 |   2.093 |
|     4 |  0.0001 | 0.0001 |     1.786 |                 2.733 |                     2.133 |                 1.788 |   2.11  |
|     4 |  0.0001 | 0.001  |     1.64  |                 3.03  |                     2.038 |                 1.76  |   2.117 |
|     8 |  0.0001 | 0.0001 |     1.709 |                 3.006 |                     2.143 |                 1.665 |   2.131 |
|     1 |  0.0001 | 0.001  |     1.341 |                 3.241 |                     2.217 |                 1.783 |   2.145 |
|     8 |  1e-05  | 0.0001 |     1.711 |                 3.071 |                     2.094 |                 1.735 |   2.153 |

Worst setting on the grid scores 3.90. The best setting is at the edge of the grid for: q_v.
