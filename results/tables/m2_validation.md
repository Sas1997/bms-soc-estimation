# ECM open-loop voltage prediction on drive cycles (1 Hz)

Parameters from HPPC; OCV from C/20; 10 degC values interpolated, never fitted. Median over cycles at each temperature.

| folder   |   1RC median V RMSE (mV) |   2RC median V RMSE (mV) |   1RC, SoC > 10 % (mV) |   2RC, SoC > 10 % (mV) |
|:---------|-------------------------:|-------------------------:|-----------------------:|-----------------------:|
| n20degC  |                    116.7 |                     98.4 |                  116.7 |                   98.4 |
| n10degC  |                     78.7 |                     63.1 |                   78.7 |                   63.1 |
| 0degC    |                     38.8 |                     32.9 |                   38.6 |                   33.2 |
| 10degC   |                     21.3 |                     34.7 |                   21   |                   35.5 |
| 25degC   |                     13.1 |                     16.9 |                   11.1 |                   16.7 |
| 40degC   |                     10.1 |                     12.9 |                    7.9 |                   11.8 |

# Coulomb counting on the test sets (SoC error, percentage points, mean over cycles)

The reference SoC is itself built from the tester Ah counter with a known full-charge start, so coulomb counting with the correct initial SoC is near-exact by construction; it is shown to make that explicit. Its weaknesses are an unknown initial SoC and sensor bias, which it never corrects.

|                                       |   rmse |   mae |   max_abs |
|:--------------------------------------|-------:|------:|----------:|
| ('test_unseen_cycle', 'correct init') |   0.11 |  0.09 |      0.18 |
| ('test_unseen_cycle', 'init -20 pts') |  20.03 | 20.03 |     20.13 |
| ('test_unseen_cycle', '+50 mA bias')  |   2.33 |  2.02 |      4.03 |
| ('test_unseen_temp', 'correct init')  |   0.3  |  0.26 |      0.5  |
| ('test_unseen_temp', 'init -20 pts')  |  19.74 | 19.74 |     20    |
| ('test_unseen_temp', '+50 mA bias')   |   1.98 |  1.71 |      3.45 |
