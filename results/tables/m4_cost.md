# Compute cost per estimator

One 1 Hz update, measured sample by sample; each estimator in its own Python process, PyTorch on one thread. Peak memory is the resident set size of that process, so it includes the interpreter and libraries (PyTorch for the networks, NumPy and pandas for all).

## cloud-vm

Intel(R) Xeon(R) Processor @ 2.30GHz, x86_64, Python 3.13.18, 1000 steps.

| estimator             |   ms per step (median) |   ms per step (99th pct) |   peak memory (MB) |   stored values |   storage (kB) |
|:----------------------|-----------------------:|-------------------------:|-------------------:|----------------:|---------------:|
| Coulomb counting      |                  0     |                    0     |            198.426 |               0 |          0     |
| EKF, 2RC (HPPC only)  |                  0.063 |                    0.098 |            198.27  |            1731 |         37.923 |
| EKF, 3RC (calibrated) |                  0.063 |                    0.101 |            198.195 |            1941 |         43.048 |
| LSTM                  |                  0.218 |                    0.321 |            685.82  |            4769 |         21.235 |
| Transformer           |                  0.305 |                    0.774 |            681.688 |           17697 |         77.653 |
