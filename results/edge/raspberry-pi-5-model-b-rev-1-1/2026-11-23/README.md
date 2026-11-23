# Edge runs on Raspberry Pi 5 Model B Rev 1.1

Collected 2026-11-23 23:23 UTC by `demo/collect_evidence.py` on the device itself.

| | |
|:--|:--|
| Board | Raspberry Pi 5 Model B Rev 1.1 |
| Operating system | Debian GNU/Linux 13 (trixie) |
| Kernel | 6.18.60+rpt-rpi-2712 |
| CPU cores / memory | 4 / 7.9 GB |
| Python | 3.13.5 |
| Code commit | `24d2f4bd1585` |
| Power | input 5.36 V; throttled flags 0x0 (0x0 = no under-voltage or throttling since boot) |
| Test suite on the device | 67 passed, 3 skipped in 12.82s |

## Compute cost on the device (one thread, 1,000 steps)

| Estimator | ms per step, median | 99th percentile | Peak memory (MB) |
|:--|--:|--:|--:|
| Coulomb counting | 0.001 | 0.001 | 109 |
| EKF, 2RC (HPPC only) | 0.151 | 0.163 | 109 |
| EKF, 3RC (calibrated) | 0.152 | 0.163 | 110 |
| LSTM (ONNX Runtime) | 0.339 | 0.354 | 136 |
| Transformer (ONNX Runtime) | 0.444 | 0.461 | 137 |

## Edge runs

Run IDs are the ones the live dashboard shows. RMSE in SoC percentage points against the reference; ms is the median compute time per 1 Hz step on the device.

| Run ID | Cycle | Mode | Samples | RMSE EKF / LSTM / Transformer / CC | ms EKF / LSTM / Transformer | CPU at end |
|:--|:--|:--|--:|:--|:--|--:|
| `20261123T012526Z-10degC_US06` | 10degC_US06 | not recorded (run made before this field existed) | 3871 | 3.57 / 1.72 / 1.41 / 19.78 | 0.20 / 0.35 / 0.45 | 37.0 °C |
| `20261123T015551Z-10degC_US06` | 10degC_US06 | not recorded (run made before this field existed) | 300 | 0.62 / 1.59 / 1.33 / 19.98 | 0.22 / 0.35 / 0.46 | 38.0 °C |
| `20261123T015910Z-25degC_UDDS` | 25degC_UDDS | not recorded (run made before this field existed) | 579 | 0.47 / 0.54 / 1.36 / 20.01 | 0.35 / 0.56 / 0.75 | 37.5 °C |
| `20261123T221005Z-25degC_UDDS` | 25degC_UDDS | streamed to wfa61d72.ala.us-east-1.emqxsl.com | 15965 | 0.87 / 0.76 / 1.19 / 20.15 | 0.34 / 0.57 / 0.74 | 35.3 °C |
| `20261123T225602Z-25degC_UDDS` | 25degC_UDDS | streamed to wfa61d72.ala.us-east-1.emqxsl.com | 15965 | 0.87 / 0.76 / 1.19 / 20.15 | 0.35 / 0.56 / 0.76 | 37.0 °C |
| `20261123T232410Z-10degC_US06` | 10degC_US06 | offline | 3871 | 3.57 / 1.72 / 1.41 / 19.78 | 0.21 / 0.35 / 0.46 | 41.4 °C |

## Verify

Every file here is listed with its SHA-256 checksum in `sha256sums.txt`:

```
sha256sum -c sha256sums.txt
```
