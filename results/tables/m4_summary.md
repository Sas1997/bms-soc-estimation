# M4 results: SoC error on the test sets (percentage points)

Mean over test cycles. Unseen cycle: UDDS, HWFET, LA92, US06 at five temperatures (19 cycles). Unseen temperature: all usable cycles at 10 °C (10 cycles).

## Nominal conditions

| estimator             |   RMSE unseen cycle |   RMSE unseen temperature |   MAE unseen cycle |   MAE unseen temperature |   MAX unseen cycle |   MAX unseen temperature |
|:----------------------|--------------------:|--------------------------:|-------------------:|-------------------------:|-------------------:|-------------------------:|
| Coulomb counting      |                0.11 |                      0.3  |               0.09 |                     0.26 |               0.18 |                     0.5  |
| EKF, 2RC (HPPC only)  |                1.73 |                      1.34 |               1.53 |                     1.23 |               2.93 |                     2.04 |
| EKF, 3RC (calibrated) |                0.76 |                      1.15 |               0.68 |                     0.92 |               1.5  |                     2.56 |
| LSTM                  |                1.35 |                      1.42 |               1.06 |                     1.13 |               4.72 |                     5.42 |
| Transformer           |                1.62 |                      1.4  |               1.26 |                     1.15 |               5.93 |                     5.07 |

Coulomb counting is near-exact here by construction: the reference SoC is coulomb counting with a known full-charge start and the laboratory current sensor.

## Robustness: RMSE by case

### Unseen cycle

| estimator             |   nominal |   initial SoC -20 pts |   late start, SoC -20 pts |   sensor noise |   current bias +100 mA |
|:----------------------|----------:|----------------------:|--------------------------:|---------------:|-----------------------:|
| Coulomb counting      |      0.11 |                 20.03 |                     19.97 |           0.12 |                   4.62 |
| EKF, 2RC (HPPC only)  |      1.73 |                  1.77 |                      3.97 |           1.71 |                   2.77 |
| EKF, 3RC (calibrated) |      0.76 |                  0.76 |                      2.23 |           0.72 |                   1.63 |
| LSTM                  |      1.35 |                  1.35 |                      1.5  |           1.4  |                   1.89 |
| Transformer           |      1.62 |                  1.62 |                      1.83 |           1.64 |                   1.95 |

### Unseen temperature

| estimator             |   nominal |   initial SoC -20 pts |   late start, SoC -20 pts |   sensor noise |   current bias +100 mA |
|:----------------------|----------:|----------------------:|--------------------------:|---------------:|-----------------------:|
| Coulomb counting      |      0.3  |                 19.74 |                     19.73 |           0.39 |                   4.26 |
| EKF, 2RC (HPPC only)  |      1.34 |                  1.36 |                      2.37 |           1.36 |                   1.32 |
| EKF, 3RC (calibrated) |      1.15 |                  1.17 |                      2.18 |           1.13 |                   0.83 |
| LSTM                  |      1.42 |                  1.42 |                      1.52 |           1.45 |                   1.71 |
| Transformer           |      1.4  |                  1.4  |                      1.49 |           1.41 |                   1.7  |

Networks have no initial state, so their wrong-initial-SoC result at rest equals their nominal result. In the late-start case they begin with an empty input window.

Late start: time after which the error stays within 5 points (seconds from the start of estimation; `not_settled` counts cycles where the error is still outside 5 points at the end):

| estimator             |   median_s |   p90_s |   not_settled |
|:----------------------|-----------:|--------:|--------------:|
| Coulomb counting      |        nan |     nan |            29 |
| EKF, 2RC (HPPC only)  |         41 |     883 |             5 |
| EKF, 3RC (calibrated) |         63 |     881 |             0 |

## Nominal RMSE by temperature (both test sets)

| estimator             |   -20 °C |   -10 °C |   0 °C |   10 °C |   25 °C |   40 °C |
|:----------------------|---------:|---------:|-------:|--------:|--------:|--------:|
| Coulomb counting      |     0.05 |     0.05 |   0.12 |    0.3  |    0.21 |    0.15 |
| EKF, 2RC (HPPC only)  |     3.69 |     2.43 |   0.78 |    1.34 |    0.8  |    0.74 |
| EKF, 3RC (calibrated) |     1.62 |     0.59 |   0.56 |    1.15 |    0.5  |    0.45 |
| LSTM                  |     1.51 |     1.77 |   1.51 |    1.42 |    1.02 |    0.84 |
| Transformer           |     2.02 |     1.95 |   1.69 |    1.4  |    1.18 |    1.15 |

## Nominal RMSE by drive cycle (unseen-cycle set, mean over temperatures)

| estimator             |   HWFET |   LA92 |   UDDS |   US06 |
|:----------------------|--------:|-------:|-------:|-------:|
| Coulomb counting      |    0.07 |   0.1  |   0.11 |   0.16 |
| EKF, 2RC (HPPC only)  |    2.55 |   1.08 |   0.97 |   2.5  |
| EKF, 3RC (calibrated) |    0.53 |   0.51 |   0.51 |   1.45 |
| LSTM                  |    1.17 |   1.08 |   1.47 |   1.63 |
| Transformer           |    1.52 |   1.32 |   1.91 |   1.71 |

## Largest single-cycle maximum error, nominal

| estimator             |   max abs error |
|:----------------------|----------------:|
| Coulomb counting      |            0.58 |
| EKF, 2RC (HPPC only)  |            8.32 |
| EKF, 3RC (calibrated) |            5.46 |
| LSTM                  |            8.23 |
| Transformer           |            8.13 |
