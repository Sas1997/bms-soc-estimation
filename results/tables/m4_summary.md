# M4 results: SoC error on the test sets (percentage points)

Mean over test cycles. Unseen cycle: UDDS, HWFET, LA92, US06 at five temperatures (19 cycles). Unseen temperature: all usable cycles at 10 °C (10 cycles).

## Nominal conditions

| estimator             |   RMSE unseen cycle |   RMSE unseen temperature |   MAE unseen cycle |   MAE unseen temperature |   MAX unseen cycle |   MAX unseen temperature |
|:----------------------|--------------------:|--------------------------:|-------------------:|-------------------------:|-------------------:|-------------------------:|
| Coulomb counting      |                0.11 |                      0.3  |               0.09 |                     0.26 |               0.18 |                     0.5  |
| EKF, 2RC (HPPC only)  |                0.55 |                      0.51 |               0.52 |                     0.48 |               0.89 |                     0.76 |
| EKF, 3RC (calibrated) |                1.14 |                      1.78 |               1.02 |                     1.51 |               1.96 |                     2.84 |
| LSTM                  |                1.38 |                      1.45 |               1.08 |                     1.15 |               4.9  |                     5.65 |
| Transformer           |                1.62 |                      1.4  |               1.26 |                     1.15 |               5.93 |                     5.07 |

Coulomb counting is near-exact here by construction: the reference SoC is coulomb counting with a known full-charge start and the laboratory current sensor.

## Robustness: RMSE by case

### Unseen cycle

| estimator             |   nominal |   initial SoC -20 pts |   late start, SoC -20 pts |   sensor noise |   current bias +100 mA |
|:----------------------|----------:|----------------------:|--------------------------:|---------------:|-----------------------:|
| Coulomb counting      |      0.11 |                 20.03 |                     19.97 |           0.12 |                   4.62 |
| EKF, 2RC (HPPC only)  |      0.55 |                  0.97 |                      6.08 |           0.46 |                   3.46 |
| EKF, 3RC (calibrated) |      1.14 |                  1.56 |                      2.59 |           1.08 |                   1.93 |
| LSTM                  |      1.38 |                  1.38 |                      1.53 |           1.43 |                   1.84 |
| Transformer           |      1.62 |                  1.62 |                      1.83 |           1.64 |                   1.95 |

### Unseen temperature

| estimator             |   nominal |   initial SoC -20 pts |   late start, SoC -20 pts |   sensor noise |   current bias +100 mA |
|:----------------------|----------:|----------------------:|--------------------------:|---------------:|-----------------------:|
| Coulomb counting      |      0.3  |                 19.74 |                     19.73 |           0.39 |                   4.26 |
| EKF, 2RC (HPPC only)  |      0.51 |                  0.42 |                      2.35 |           0.41 |                   3.39 |
| EKF, 3RC (calibrated) |      1.78 |                  1.82 |                      2.89 |           1.76 |                   1.32 |
| LSTM                  |      1.45 |                  1.45 |                      1.54 |           1.49 |                   1.7  |
| Transformer           |      1.4  |                  1.4  |                      1.49 |           1.41 |                   1.7  |

Networks have no initial state, so their wrong-initial-SoC result at rest equals their nominal result. In the late-start case they begin with an empty input window.

Late start: time after which the error stays within 5 points (seconds from the start of estimation; `not_settled` counts cycles where the error is still outside 5 points at the end):

| estimator             |   median_s |   p90_s |   not_settled |
|:----------------------|-----------:|--------:|--------------:|
| Coulomb counting      |        nan |     nan |            29 |
| EKF, 2RC (HPPC only)  |        517 |    3073 |             6 |
| EKF, 3RC (calibrated) |        160 |    1540 |             0 |

## Nominal RMSE by temperature (both test sets)

| estimator             |   -20 °C |   -10 °C |   0 °C |   10 °C |   25 °C |   40 °C |
|:----------------------|---------:|---------:|-------:|--------:|--------:|--------:|
| Coulomb counting      |     0.05 |     0.05 |   0.12 |    0.3  |    0.21 |    0.15 |
| EKF, 2RC (HPPC only)  |     1.56 |     0.39 |   0.15 |    0.51 |    0.19 |    0.37 |
| EKF, 3RC (calibrated) |     1.9  |     1.09 |   1.09 |    1.78 |    0.9  |    0.67 |
| LSTM                  |     1.56 |     1.79 |   1.58 |    1.45 |    1.03 |    0.87 |
| Transformer           |     2.02 |     1.95 |   1.69 |    1.4  |    1.18 |    1.15 |

## Nominal RMSE by drive cycle (unseen-cycle set, mean over temperatures)

| estimator             |   HWFET |   LA92 |   UDDS |   US06 |
|:----------------------|--------:|-------:|-------:|-------:|
| Coulomb counting      |    0.07 |   0.1  |   0.11 |   0.16 |
| EKF, 2RC (HPPC only)  |    0.45 |   0.62 |   0.5  |   0.6  |
| EKF, 3RC (calibrated) |    0.87 |   0.52 |   0.65 |   2.49 |
| LSTM                  |    1.23 |   1.12 |   1.5  |   1.64 |
| Transformer           |    1.52 |   1.32 |   1.91 |   1.71 |

## Largest single-cycle maximum error, nominal

| estimator             |   max abs error |
|:----------------------|----------------:|
| Coulomb counting      |            0.58 |
| EKF, 2RC (HPPC only)  |            2.8  |
| EKF, 3RC (calibrated) |            5.92 |
| LSTM                  |            8.4  |
| Transformer           |            8.13 |
