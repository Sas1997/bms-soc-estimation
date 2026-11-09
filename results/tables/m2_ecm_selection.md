# ECM variant selection (train + val drive cycles only, open-loop voltage RMSE, mV, median)

(all fitted on HPPC, never on drive cycles):
  A  all four pulse pairs, all SoC levels
  B  all four pulse pairs, SoC levels >= 0.15 (table held flat below)
  C  first two pulse pairs only (1C and 2C discharge), SoC levels >= 0.15

|          |   n20degC |   n10degC |   0degC |   25degC |   40degC |   overall median |
|:---------|----------:|----------:|--------:|---------:|---------:|-----------------:|
| ('A', 1) |     116.7 |      77.7 |    34.1 |     45.4 |     29.4 |             68.4 |
| ('A', 2) |      98.4 |      58.3 |   130.3 |    120.3 |     80.4 |            106   |
| ('B', 1) |     116.7 |      78.7 |    38.8 |     13.2 |     10.3 |             40.4 |
| ('B', 2) |      98.4 |      63.1 |    34.1 |     16.8 |     12.9 |             36.3 |
| ('C', 1) |     119.9 |      81.8 |    40.1 |     14.3 |     10.1 |             43.1 |
| ('C', 2) |     117.5 |      81.1 |    40.3 |     19.2 |     14.5 |             45.3 |

# OCV definition (variant B, 2RC)

Median voltage RMSE (mV):

| ocv          |   n20degC |   n10degC |   0degC |   25degC |   40degC |   overall median |
|:-------------|----------:|----------:|--------:|---------:|---------:|-----------------:|
| average      |     123.3 |      80.3 |    42.4 |     35.6 |     30.3 |             43.3 |
| discharge    |      89   |      52.7 |    38.4 |     22.2 |     16.5 |             41.3 |
| discharge+IR |      98.4 |      63.1 |    34.1 |     16.8 |     12.9 |             36.3 |

Median bias, model minus measured (mV):

| ocv          |   n20degC |   n10degC |   0degC |   25degC |   40degC |
|:-------------|----------:|----------:|--------:|---------:|---------:|
| average      |      92.6 |      61.8 |    16.9 |     15   |     13.7 |
| discharge    |      22.5 |      20.2 |   -16.9 |    -16.4 |    -11.2 |
| discharge+IR |      52.4 |      39.2 |    -1.3 |     -9   |     -5.7 |

Selected: variant B, 2RC, OCV = discharge+IR (lowest overall median RMSE).
