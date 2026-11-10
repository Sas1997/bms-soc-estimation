# Neural network selection on the validation cycles

SoC RMSE in percentage points, mean over the validation cycles. Score = mean of the three cases. Augmented checkpoints were trained with added sensor noise (5 mV, 50 mA, 0.5 °C) and a random current offset (50 mA standard deviation).

| checkpoint           | type        |   window_s |   parameters | augmented   |   epochs_run |   nominal |   initial SoC -20 pts |   current bias +50 mA |   score |
|:---------------------|:------------|-----------:|-------------:|:------------|-------------:|----------:|----------------------:|----------------------:|--------:|
| lstm                 | lstm        |        300 |         4769 | False       |           40 |     1.177 |                 1.177 |                 1.32  |   1.225 |
| lstm_aug             | lstm        |        300 |         4769 | True        |           40 |     1.212 |                 1.212 |                 1.331 |   1.252 |
| transformer          | transformer |        300 |        17697 | False       |           32 |     1.419 |                 1.419 |                 1.556 |   1.464 |
| transformer_w600_aug | transformer |        600 |        18177 | True        |           34 |     1.474 |                 1.474 |                 1.51  |   1.486 |
| transformer_w600     | transformer |        600 |        18177 | False       |           30 |     1.487 |                 1.487 |                 1.616 |   1.53  |
| transformer_causal   | transformer |        300 |        17697 | False       |           30 |     1.69  |                 1.69  |                 1.821 |   1.734 |

Selected: lstm: `lstm`, transformer: `transformer`.
