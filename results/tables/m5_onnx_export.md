# ONNX export of the selected networks

Exported with torch.onnx (opset 18) and checked with ONNX Runtime on every sample of the 25 °C UDDS test cycle (15965 samples at 1 Hz). Step-by-step check: first 600 samples.

| Network | Checkpoint | ONNX file (kB) | Max SoC difference, batch | Max SoC difference, step by step |
|:--|:--|--:|--:|--:|
| lstm | `lstm_aug` | 23 | 3.6e-07 | 3.6e-07 |
| transformer | `transformer` | 99 | 4.8e-07 | 4.2e-07 |
