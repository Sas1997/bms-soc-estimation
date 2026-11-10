# Cell balancing: passive vs active (simulation)

Cells: the selected identified cell model (results/characterization/selected.json)
at 25 degC, nominal capacity = C/20 discharge capacity at 25 degC. The dataset
has one cell, so cell-to-cell variation is ASSUMED, not measured:

    initial SoC        0.90 + N(0, 2 points)      per cell
    capacity           nominal x (1 + N(0, 1.5 %))
    all resistances    nominal x (1 + N(0, 5 %))

drawn for 200 packs of 4 cells in series (seed 7, values clipped to +-3 sigma).

Balancing circuits (src/socest/pack.py): passive shunt resistor, 20 ohm (about
0.2 A at 4 V); active adjacent cell-to-cell and active cell-to-pack, 0.2 A from
the donor cell, 85 % converter efficiency. Balancing decisions use SoC. Unless
stated, the controller knows each cell's true SoC.

Scenario 1, rest: the pack is idle; balance until the SoC spread (max - min) is
below 0.5 points. Report balancing time and energy lost (drop in stored chemical
energy, which includes the balancing circuit and the cells' own resistance).

Scenario 2, discharge: the pack starts from an end-of-charge state (the highest
cell at SoC 1.0, others below by their initial spread) and is discharged by the
25 degC UDDS current profile from the dataset, repeated, until the first cell
falls below 2.8 V (the cut-off of the dataset's capacity tests, data/DATA_CARD.md).
Balancing runs during the discharge (on above 1 point spread, off below 0.5).
Report energy delivered to the load relative to no balancing. Passive balancing
is usually applied at the end of charge, not during discharge, so two more cases
start the discharge with every cell full (top-balanced): one with no balancing
in discharge, one with active adjacent balancing in discharge. Any of the three
circuits can reach that state during charging if the charge lasts long enough;
the energy burned or converted while doing so comes from the charger and is not
counted in the delivered energy.

Sensitivity: converter efficiency, a fixed per-cell SoC estimation error in the
controller, and string length.

Nominal cell: 2.778 Ah. Median initial SoC spread in the pack: 3.8 points (p10 1.8, p90 6.0).

## Scenario 1: balancing at rest (200 packs, median and 10th to 90th percentile)

| method                        |   balancing time (h) median |   balancing time (h) p10 |   balancing time (h) p90 |   energy lost (Wh) median |   energy lost (Wh) p10 |   energy lost (Wh) p90 |   energy lost (% of stored) median |   energy lost (% of stored) p10 |   energy lost (% of stored) p90 |   of which in the circuit (%) median |
|:------------------------------|----------------------------:|-------------------------:|-------------------------:|--------------------------:|-----------------------:|-----------------------:|-----------------------------------:|--------------------------------:|--------------------------------:|-------------------------------------:|
| Passive (shunt resistor)      |                        0.45 |                     0.18 |                     0.75 |                      0.79 |                   0.3  |                   1.39 |                               2.15 |                            0.83 |                            3.81 |                                99.79 |
| Active, adjacent cell-to-cell |                        0.25 |                     0.09 |                     0.47 |                      0.08 |                   0.03 |                   0.14 |                               0.22 |                            0.08 |                            0.38 |                                98.1  |
| Active, cell-to-pack          |                        0.46 |                     0.18 |                     0.75 |                      0.11 |                   0.04 |                   0.19 |                               0.3  |                            0.1  |                            0.53 |                                99.38 |

## Scenario 2: balancing during a UDDS discharge to 2.8 V (200 packs)

| method                                          |   energy delivered (Wh) median |   energy delivered (Wh) p10 |   energy delivered (Wh) p90 |   change vs no balancing (%) median |   change vs no balancing (%) p10 |   change vs no balancing (%) p90 |   circuit loss (Wh) median |   circuit loss (Wh) p10 |   circuit loss (Wh) p90 |   SoC spread at cut-off (points) median |   SoC spread at cut-off (points) p10 |   SoC spread at cut-off (points) p90 |
|:------------------------------------------------|-------------------------------:|----------------------------:|----------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|---------------------------:|------------------------:|------------------------:|----------------------------------------:|-------------------------------------:|-------------------------------------:|
| No balancing                                    |                          38.06 |                       36.93 |                       38.95 |                                0    |                             0    |                             0    |                       0    |                    0    |                    0    |                                    4.57 |                                 2.27 |                                 7.46 |
| Passive (shunt resistor)                        |                          37.68 |                       36.15 |                       38.64 |                               -1.34 |                            -3.06 |                            -0.56 |                       1.2  |                    0.68 |                    1.85 |                                    0.78 |                                 0.54 |                                 0.97 |
| Active, adjacent cell-to-cell                   |                          38.66 |                       37.83 |                       39.49 |                                1.55 |                             0.22 |                             3.1  |                       0.12 |                    0.06 |                    0.18 |                                    0.76 |                                 0.55 |                                 0.96 |
| Active, cell-to-pack                            |                          38.64 |                       37.79 |                       39.45 |                                1.49 |                             0.15 |                             2.89 |                       0.16 |                    0.09 |                    0.26 |                                    0.75 |                                 0.56 |                                 0.95 |
| Top-balanced at end of charge                   |                          39.44 |                       38.79 |                       39.97 |                                3.27 |                             0.86 |                             6.04 |                       0    |                    0    |                    0    |                                    2.8  |                                 1.59 |                                 4.8  |
| Top-balanced, plus active adjacent in discharge |                          39.82 |                       39.33 |                       40.2  |                                4.29 |                             2    |                             7.58 |                       0.04 |                    0.01 |                    0.08 |                                    0.73 |                                 0.54 |                                 0.94 |

## Sensitivity: converter efficiency (medians; passive does not depend on it)

|    x |   drive_gain_pct Active, adjacent cell-to-cell |   drive_gain_pct Active, cell-to-pack |   rest_loss_wh Active, adjacent cell-to-cell |   rest_loss_wh Active, cell-to-pack |
|-----:|-----------------------------------------------:|--------------------------------------:|---------------------------------------------:|------------------------------------:|
| 0.7  |                                           1.22 |                                  1.09 |                                         0.17 |                                0.22 |
| 0.8  |                                           1.49 |                                  1.34 |                                         0.11 |                                0.15 |
| 0.9  |                                           1.66 |                                  1.57 |                                         0.05 |                                0.07 |
| 0.95 |                                           1.74 |                                  1.7  |                                         0.03 |                                0.04 |

## Sensitivity: SoC estimation error in the controller (rest)

Each cell's SoC seen by the controller is off by a fixed amount drawn from N(0, sigma). Rows: sigma in SoC points. Values: true SoC spread when the controller stops (points), median and 90th percentile.

Median:

|   x |   Active, adjacent cell-to-cell |   Active, cell-to-pack |   Passive (shunt resistor) |
|----:|--------------------------------:|-----------------------:|---------------------------:|
| 0   |                            0.49 |                   0.49 |                       0.49 |
| 0.5 |                            1.02 |                   1.01 |                       1.03 |
| 1   |                            2    |                   1.97 |                       1.95 |
| 2   |                            3.87 |                   3.87 |                       3.87 |

90th percentile:

|   x |   Active, adjacent cell-to-cell |   Active, cell-to-pack |   Passive (shunt resistor) |
|----:|--------------------------------:|-----------------------:|---------------------------:|
| 0   |                            0.5  |                   0.5  |                       0.5  |
| 0.5 |                            1.68 |                   1.67 |                       1.71 |
| 1   |                            3.12 |                   3.12 |                       3.13 |
| 2   |                            6.27 |                   6.31 |                       6.18 |

## Sensitivity: string length (rest, 100 packs per length, medians)

Packs not balanced within 96 h: 0.

|   x |   rest_loss_wh_per_cell Active, adjacent cell-to-cell |   rest_loss_wh_per_cell Active, cell-to-pack |   rest_loss_wh_per_cell Passive (shunt resistor) |   rest_t_h Active, adjacent cell-to-cell |   rest_t_h Active, cell-to-pack |   rest_t_h Passive (shunt resistor) |
|----:|------------------------------------------------------:|---------------------------------------------:|-------------------------------------------------:|-----------------------------------------:|--------------------------------:|------------------------------------:|
|   4 |                                                  0.02 |                                         0.03 |                                             0.22 |                                     0.25 |                            0.46 |                                0.46 |
|   8 |                                                  0.04 |                                         0.04 |                                             0.37 |                                     0.45 |                            0.71 |                                0.78 |
|  12 |                                                  0.05 |                                         0.05 |                                             0.53 |                                     0.56 |                            0.81 |                                0.98 |
