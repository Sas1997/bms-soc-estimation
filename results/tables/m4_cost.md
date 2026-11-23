# Compute cost per estimator

One 1 Hz update, measured sample by sample; each estimator in its own Python process, PyTorch and ONNX Runtime on one thread. Peak memory is the resident set size of that process, so it includes the interpreter and libraries (PyTorch or ONNX Runtime for the networks, NumPy and pandas for all).

## cloud-vm

Intel(R) Xeon(R) Processor @ 2.10GHz, x86_64, Python 3.13.18, 1000 steps.

| estimator                  |   ms per step (median) |   ms per step (99th pct) |   peak memory (MB) |   stored values |   storage (kB) |
|:---------------------------|-----------------------:|-------------------------:|-------------------:|----------------:|---------------:|
| Coulomb counting           |                  0     |                    0.001 |            199.578 |               0 |          0     |
| EKF, 2RC (HPPC only)       |                  0.11  |                    0.198 |            197.848 |            1731 |         37.923 |
| EKF, 3RC (calibrated)      |                  0.077 |                    0.131 |            201.754 |            1941 |         43.048 |
| LSTM (PyTorch)             |                  0.298 |                    0.571 |            679.586 |            4769 |         21.235 |
| Transformer (PyTorch)      |                  0.352 |                    0.761 |            686.668 |           17697 |         77.653 |
| LSTM (ONNX Runtime)        |                  0.136 |                    0.411 |            221.203 |            4769 |         23.139 |
| Transformer (ONNX Runtime) |                  0.177 |                    0.256 |            222.402 |           17697 |         99.136 |

## raspberry-pi-5

Raspberry Pi 5 Model B (8 GB), Arm Cortex-A76, aarch64, Python 3.13.5, 1000 steps.

| estimator                  |   ms per step (median) |   ms per step (99th pct) |   peak memory (MB) |   stored values |   storage (kB) |
|:---------------------------|-----------------------:|-------------------------:|-------------------:|----------------:|---------------:|
| Coulomb counting           |                  0.001 |                    0.001 |            109.812 |               0 |          0     |
| EKF, 2RC (HPPC only)       |                  0.153 |                    0.165 |            109.703 |            1731 |         37.923 |
| EKF, 3RC (calibrated)      |                  0.153 |                    0.166 |            109.656 |            1941 |         43.048 |
| LSTM (ONNX Runtime)        |                  0.335 |                    0.348 |            136.391 |            4769 |         23.139 |
| Transformer (ONNX Runtime) |                  0.445 |                    0.462 |            136.922 |           17697 |         99.136 |
