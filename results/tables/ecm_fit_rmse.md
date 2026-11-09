# ECM fit quality on HPPC blocks

One fit per SoC level (four 10 s discharge/charge pulse pairs and their rests). Voltage RMSE in mV.

| folder   |   n_rc |   blocks |   median_rmse_mv |   max_rmse_mv | all_converged   |
|:---------|-------:|---------:|-----------------:|--------------:|:----------------|
| n20degC  |      1 |        9 |            18.87 |         52.53 | True            |
| n20degC  |      2 |        9 |            16.7  |         50.38 | True            |
| n10degC  |      1 |       10 |            15.75 |         44.01 | True            |
| n10degC  |      2 |       10 |            13.56 |         41.28 | True            |
| 0degC    |      1 |       10 |            13.25 |         27.88 | True            |
| 0degC    |      2 |       10 |             9.62 |         26.35 | True            |
| 10degC   |      1 |       11 |             8.88 |         15.72 | True            |
| 10degC   |      2 |       11 |             6.15 |         14.6  | True            |
| 25degC   |      1 |       12 |             5.33 |         11.22 | True            |
| 25degC   |      2 |       12 |             3.59 |          7.64 | True            |
| 40degC   |      1 |       11 |             4.3  |         10.73 | True            |
| 40degC   |      2 |       11 |             2.93 |          7.13 | True            |

# OCV construction (C/20)

|         |   cap_dis_ah |   cap_chg_ah |   charge_branch_soc_max |   top_half_gap_v |   mean_chg_minus_dis_mv |
|:--------|-------------:|-------------:|------------------------:|-----------------:|------------------------:|
| n20degC |       2.4923 |       2.4501 |                   0.915 |           0.0874 |                170.104  |
| n10degC |       2.5611 |       2.5993 |                   0.97  |           0.0501 |                104.493  |
| 0degC   |       2.5961 |       2.6844 |                   0.985 |           0.0321 |                 77.5634 |
| 10degC  |       2.6464 |       2.6935 |                   0.99  |           0.0213 |                 64.4454 |
| 25degC  |       2.7782 |       2.9685 |                   0.995 |           0.0153 |                 64.5651 |
| 40degC  |       2.6874 |       2.8644 |                   0.995 |           0.0142 |                 56.0981 |
