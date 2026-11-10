# Compute cost per estimator

One 1 Hz update, measured sample by sample; each estimator in its own Python process, PyTorch on one thread. Peak memory is the resident set size of that process, so it includes the interpreter and libraries (PyTorch for the networks, NumPy and pandas for all).

## cloud-vm

Intel(R) Xeon(R) Processor @ 2.30GHz, x86_64, Python 3.13.18, 1000 steps.

| estimator             |   ms per step (median) |   ms per step (99th pct) |   peak memory (MB) |   stored values |   storage (kB) |
|:----------------------|-----------------------:|-------------------------:|-------------------:|----------------:|---------------:|
| Coulomb counting      |                  0     |                    0.001 |            196.551 |               0 |          0     |
| EKF, 2RC (HPPC only)  |                  0.064 |                    0.161 |            200.742 |            1731 |         37.911 |
| EKF, 3RC (calibrated) |                  0.063 |                    0.12  |            200.637 |            1941 |         43.048 |
| LSTM                  |                  0.234 |                    0.419 |            690.812 |            4769 |         21.235 |
| Transformer           |                  0.287 |                    0.527 |            686.66  |           17697 |         77.653 |
