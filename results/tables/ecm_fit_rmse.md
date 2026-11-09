# ECM fit quality on HPPC pulse blocks

One fit per SoC level at or above 0.15 (four 10 s discharge/charge pulse pairs and their rests). Voltage RMSE in mV. `bv` adds the Butler-Volmer term.

| folder   | family   |   n_rc |   levels |   median_rmse_mv |   max_rmse_mv | all_converged   |
|:---------|:---------|-------:|---------:|-----------------:|--------------:|:----------------|
| n20degC  | linear   |      1 |        9 |            18.87 |         52.53 | True            |
| n20degC  | linear   |      2 |        9 |            16.7  |         50.38 | True            |
| n20degC  | bv       |      2 |        9 |             8.31 |         39.35 | True            |
| n10degC  | linear   |      1 |        9 |            14.35 |         44.01 | True            |
| n10degC  | linear   |      2 |        9 |            12.76 |         41.28 | True            |
| n10degC  | bv       |      2 |        9 |             9.31 |         32.09 | True            |
| 0degC    | linear   |      1 |        9 |            13.08 |         27.88 | True            |
| 0degC    | linear   |      2 |        9 |            10.11 |         26.35 | True            |
| 0degC    | bv       |      2 |        9 |             8.31 |         21.63 | True            |
| 10degC   | linear   |      1 |        9 |             8.81 |         15.72 | True            |
| 10degC   | linear   |      2 |        9 |             5.89 |         14.6  | True            |
| 10degC   | bv       |      2 |        9 |             5.58 |         12.46 | True            |
| 25degC   | linear   |      1 |        9 |             5.14 |          7.15 | True            |
| 25degC   | linear   |      2 |        9 |             3.17 |          4.95 | True            |
| 25degC   | bv       |      2 |        9 |             3.04 |          4.9  | True            |
| 40degC   | linear   |      1 |        9 |             4.01 |          5.85 | True            |
| 40degC   | linear   |      2 |        9 |             2.81 |          4.17 | True            |
| 40degC   | bv       |      2 |        9 |             2.72 |          4.15 | True            |

# OCV construction (C/20)

|         |   cap_dis_ah |   cap_chg_ah |   charge_branch_soc_max |   top_half_gap_v |   mean_chg_minus_dis_mv |
|:--------|-------------:|-------------:|------------------------:|-----------------:|------------------------:|
| n20degC |       2.4923 |       2.4501 |                   0.915 |           0.0874 |                170.104  |
| n10degC |       2.5611 |       2.5993 |                   0.97  |           0.0501 |                104.493  |
| 0degC   |       2.5961 |       2.6844 |                   0.985 |           0.0321 |                 77.5634 |
| 10degC  |       2.6464 |       2.6935 |                   0.99  |           0.0213 |                 64.4454 |
| 25degC  |       2.7782 |       2.9685 |                   0.995 |           0.0153 |                 64.5651 |
| 40degC  |       2.6874 |       2.8644 |                   0.995 |           0.0142 |                 56.0981 |
