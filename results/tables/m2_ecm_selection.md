# Cell model selection

Open-loop voltage prediction at 1 Hz, median voltage RMSE per temperature (mV). No test cycle and no 10 °C data are used.

## Validation cycles (Mixed7, Mixed8): used for selection

Score = mean of the temperature columns. 40 °C has no validation cycles.

| variant       |   n20degC |   n10degC |   0degC |   25degC |   score | model                                                                               |
|:--------------|----------:|----------:|--------:|---------:|--------:|:------------------------------------------------------------------------------------|
| 1rc           |     127.5 |      74.1 |    38.1 |     13.2 |    63.2 | 1RC, linear                                                                         |
| 2rc           |     109.8 |      53.5 |    33.6 |     16.5 |    53.3 | 2RC, linear (M2 baseline)                                                           |
| 2rc_allsoc    |     109.8 |      50.9 |   111.2 |    117.8 |    97.4 | 2RC, linear, all SoC levels kept                                                    |
| 2rc_rest      |     124.7 |      64.3 |    38.7 |     33.6 |    65.3 | 2RC, linear, rest-anchored OCV                                                      |
| 3rc           |     118.5 |      59   |    47.8 |     24.4 |    62.4 | 2RC + slow branch, rest-anchored OCV                                                |
| 2rc_bv        |     100.5 |      47.4 |    47.8 |     33.7 |    57.3 | 2RC + Butler-Volmer, rest-anchored OCV                                              |
| 3rc_bv        |      95   |      83.4 |    66.5 |     24.8 |    67.4 | 2RC + slow branch + Butler-Volmer, rest-anchored OCV                                |
| 3rc_dc_avg    |      52.1 |      32.2 |    34.7 |     24.4 |    35.9 | 2RC (HPPC) + calibrated slow branch, OCV shape: C/20 average                        |
| 3rc_dc_dis    |      48   |      31   |    30.6 |     15.3 |    31.2 | 2RC (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR                 |
| 3rc_bv_dc_avg |      51.2 |      36.5 |    42.6 |     21.6 |    38   | 2RC + Butler-Volmer (HPPC) + calibrated slow branch, OCV shape: C/20 average        |
| 3rc_bv_dc_dis |      47.3 |      36.9 |    39   |     15.5 |    34.7 | 2RC + Butler-Volmer (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR |

Median bias on the validation cycles, model minus measured (mV):

| variant       |   n20degC |   n10degC |   0degC |   25degC |
|:--------------|----------:|----------:|--------:|---------:|
| 1rc           |      86.5 |      57.1 |    29.6 |      2.7 |
| 2rc           |      54.4 |      36.4 |    -1.3 |     -6.4 |
| 2rc_allsoc    |      54.4 |      33.8 |   -46.6 |    -50.1 |
| 2rc_rest      |      82.4 |      50.5 |     8.6 |      8.4 |
| 3rc           |     -48.8 |     -23.1 |   -20.1 |     -2.3 |
| 2rc_bv        |      56.8 |      22.3 |    -6.6 |      7.6 |
| 3rc_bv        |     -56.3 |     -66   |   -43.5 |     -3.8 |
| 3rc_dc_avg    |       0.2 |      -2.7 |     0.6 |      2.5 |
| 3rc_dc_dis    |      -3.7 |       1.2 |     1.7 |      3.4 |
| 3rc_bv_dc_avg |       1.4 |      -3.8 |     0.1 |      2.7 |
| 3rc_bv_dc_dis |      -1.4 |      -1.1 |     1.7 |      3.6 |

## Training cycles (Mixed1 to Mixed6): reference only

Variants ending in `_dc` had their slow branch calibrated on these cycles.

| variant       |   n20degC |   n10degC |   0degC |   25degC |   40degC |   mean |
|:--------------|----------:|----------:|--------:|---------:|---------:|-------:|
| 1rc           |     115.9 |      80.8 |    40.3 |     12.9 |     10.3 |   52   |
| 2rc           |      97.4 |      64.9 |    34.1 |     18.1 |     12.9 |   45.5 |
| 2rc_allsoc    |      97.4 |      61.2 |   131.3 |    128.1 |     80.4 |   99.7 |
| 2rc_rest      |     114   |      74.5 |    40   |     32.7 |     27.4 |   57.7 |
| 3rc           |     114.7 |      66.8 |    46.9 |     23.8 |     24.5 |   55.4 |
| 2rc_bv        |      97.8 |      57.3 |    47   |     32.9 |     27.4 |   52.5 |
| 3rc_bv        |      93.9 |      91.6 |    66.6 |     24.3 |     24.5 |   60.2 |
| 3rc_dc_avg    |      45.9 |      33.3 |    36.1 |     25.9 |     19.7 |   32.2 |
| 3rc_dc_dis    |      43.4 |      30.2 |    32.5 |     14.8 |     11.5 |   26.5 |
| 3rc_bv_dc_avg |      44.5 |      35.1 |    43   |     21.9 |     18   |   32.5 |
| 3rc_bv_dc_dis |      41.6 |      33.8 |    39.6 |     15.2 |     11.5 |   28.3 |

Selected: `3rc_dc_dis` (2RC (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR).
