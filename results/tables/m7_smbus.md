# M7: smart-battery (SMBus) input path on an emulated pack

Each test cycle is served by an emulated SBS 1.1 smart battery (`src/socest/smart_battery_emulator.py`) and read back over the full SMBus stack (`src/socest/smbus_link.py`): byte-level transfers, PEC on, 1 % of transfers fail like a NACK and 0.5 % of replies have a flipped bit. The emulated pack is one LG HG2 cell; its gauge is a simple OCV start plus coulomb counting, not a model of a commercial fuel gauge.

## Bus

| Cycle        |   Samples |   Reads OK |   Retried |   PEC mismatches caught |   Failed after retries |
|:-------------|----------:|-----------:|----------:|------------------------:|-----------------------:|
| 10degC_US06  |      3871 |      42583 |       648 |                     235 |                      0 |
| 25degC_UDDS  |     15965 |     175617 |      2622 |                     902 |                      0 |
| n20degC_UDDS |      8190 |      90092 |      1326 |                     479 |                      0 |

## SBS resolution vs the recording

Largest difference between the decoded SBS registers (1 mV, 1 mA, 0.1 K) and the recorded values:

| Cycle        |   Voltage, max abs (mV) |   Current, max abs (mA) |   Temperature, max abs (K) |
|:-------------|------------------------:|------------------------:|---------------------------:|
| 10degC_US06  |                     0.5 |                     0.5 |                      0.05  |
| 25degC_UDDS  |                     0.5 |                     0.5 |                      0.049 |
| n20degC_UDDS |                     0.5 |                     0.5 |                      0.049 |

## SoC accuracy on data read over SMBus (percentage points, correct initial SoC)

| Cycle        | Estimator                               |   RMSE, recorded data |   RMSE, read over SMBus |   Difference |
|:-------------|:----------------------------------------|----------------------:|------------------------:|-------------:|
| 10degC_US06  | EKF, 3RC (calibrated)                   |                 3.461 |                   3.449 |       -0.011 |
| 10degC_US06  | LSTM                                    |                 1.718 |                   1.718 |       -0     |
| 10degC_US06  | Transformer                             |                 1.405 |                   1.406 |        0.001 |
| 10degC_US06  | Coulomb counting                        |                 0.253 |                   0.252 |       -0.001 |
| 10degC_US06  | Emulated gauge, RelativeStateOfCharge() |               nan     |                   0.414 |      nan     |
| 25degC_UDDS  | EKF, 3RC (calibrated)                   |                 0.859 |                   0.858 |       -0.001 |
| 25degC_UDDS  | LSTM                                    |                 0.757 |                   0.758 |        0.001 |
| 25degC_UDDS  | Transformer                             |                 1.194 |                   1.196 |        0.001 |
| 25degC_UDDS  | Coulomb counting                        |                 0.178 |                   0.181 |        0.003 |
| 25degC_UDDS  | Emulated gauge, RelativeStateOfCharge() |               nan     |                   0.519 |      nan     |
| n20degC_UDDS | EKF, 3RC (calibrated)                   |                 0.372 |                   0.383 |        0.012 |
| n20degC_UDDS | LSTM                                    |                 2.34  |                   2.346 |        0.006 |
| n20degC_UDDS | Transformer                             |                 3.045 |                   3.05  |        0.005 |
| n20degC_UDDS | Coulomb counting                        |                 0.046 |                   0.049 |        0.003 |
| n20degC_UDDS | Emulated gauge, RelativeStateOfCharge() |               nan     |                   0.294 |      nan     |

Largest change in RMSE from reading the data over SMBus: 0.012 points.

## Register identification (vendor codes 0x30-0x3F)

Correct (right quantity, scale and sign, or right constant / no-answer verdict) on 48 of 48 code-cycle pairs; codes that never answered (0x34-0x3B) are left out of the table.

| Cycle        | Code   | Emulator put there          | Identified as                                                                                                       | Correct   |
|:-------------|:-------|:----------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------|
| 10degC_US06  | 0x30   | Temperature, 0.1 per count  | match Temperature, u16, 0.1 degC per count, offset 0.0074; R^2 0.994508, residual 0.0495                            | True      |
| 10degC_US06  | 0x31   | Current, 0.01 per count     | match Current, i16, 0.01 A per count (current positive = discharge); R^2 1.0, residual 0.00267                      | True      |
| 10degC_US06  | 0x32   | discharged_mah, 1 per count | match discharged_mah, u16, 1 mAh per count, offset 0.012; R^2 1.0, residual 0.27                                    | True      |
| 10degC_US06  | 0x33   | constant                    | constant 0x5a17 (23063)                                                                                             | True      |
| 10degC_US06  | 0x3c   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| 10degC_US06  | 0x3d   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| 10degC_US06  | 0x3e   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| 10degC_US06  | 0x3f   | Voltage, 0.001 per count    | match Voltage, u16, 0.001 V per count; R^2 1.0, residual 1.69e-16                                                   | True      |
| 25degC_UDDS  | 0x30   | Temperature, 0.1 per count  | match Temperature, u16, 0.1 degC per count, offset -0.0404; R^2 0.962628, residual 0.0294                           | True      |
| 25degC_UDDS  | 0x31   | Current, 0.01 per count     | match Current, i16, 0.01 A per count, offset -0.0001 (current positive = discharge); R^2 0.999997, residual 0.00285 | True      |
| 25degC_UDDS  | 0x32   | discharged_mah, 1 per count | match discharged_mah, u16, 1 mAh per count, offset -0.0079; R^2 1.0, residual 0.284                                 | True      |
| 25degC_UDDS  | 0x33   | constant                    | constant 0x5a17 (23063)                                                                                             | True      |
| 25degC_UDDS  | 0x3c   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| 25degC_UDDS  | 0x3d   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| 25degC_UDDS  | 0x3e   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| 25degC_UDDS  | 0x3f   | Voltage, 0.001 per count    | match Voltage, u16, 0.001 V per count; R^2 1.0, residual 1.96e-16                                                   | True      |
| n20degC_UDDS | 0x30   | Temperature, 0.1 per count  | match Temperature, i16, 0.1 degC per count, offset -0.0156; R^2 0.987243, residual 0.0475                           | True      |
| n20degC_UDDS | 0x31   | Current, 0.01 per count     | match Current, i16, 0.01 A per count, offset 0.0001 (current positive = discharge); R^2 0.999996, residual 0.00277  | True      |
| n20degC_UDDS | 0x32   | discharged_mah, 1 per count | match discharged_mah, u16, 1 mAh per count, offset 0.0154; R^2 1.0, residual 0.29                                   | True      |
| n20degC_UDDS | 0x33   | constant                    | constant 0x5a17 (23063)                                                                                             | True      |
| n20degC_UDDS | 0x3c   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| n20degC_UDDS | 0x3d   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| n20degC_UDDS | 0x3e   | constant                    | constant 0x0000 (0)                                                                                                 | True      |
| n20degC_UDDS | 0x3f   | Voltage, 0.001 per count    | match Voltage, u16, 0.001 V per count; R^2 1.0, residual 1.72e-16                                                   | True      |
