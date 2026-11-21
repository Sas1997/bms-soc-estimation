# Replay data for the edge demo

A small excerpt of the LG 18650HG2 dataset: P. Kollmeyer, M. Naguib, M. Skells,
*LG 18650HG2 Li-ion Battery Data*, Mendeley Data, V2, 2020,
doi:10.17632/b5mj79w5w9.2, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

Changes from the original files: resampled to 1 Hz with charge preserved,
current sign changed to positive = discharge, and a reference SoC column added
(tester Ah counter normalised by the C/20 capacity at that temperature). See
data/DATA_CARD.md in this repository.

All three are test cycles: no estimator in this repository was fitted, tuned
or selected on them. 10 °C is the held-out temperature.

| File | Ambient | Cycle | Samples (1 Hz) |
|:--|:--|:--|--:|

| `25degC_UDDS.csv` | 25 °C | UDDS | 15965 |
| `n20degC_UDDS.csv` | -20 °C | UDDS | 8190 |
| `10degC_US06.csv` | 10 °C | US06 | 3871 |
