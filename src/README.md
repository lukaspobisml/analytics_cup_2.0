# Source modules

See the [project README](../README.md) for the implemented ground-pass algorithm,
quickstart, scientific scope, withdrawn results and unresolved limitations.

- `data/pose_loading.py`: streaming sample/full-match helpers and verified downloads.
- `data/data_loader.py`: sample roster helpers and retrospective frame-gap velocities.
- `features/pose_orientation.py`: finite/error/width gates, shoulder tables and reports.
- `features/lane_occlusion.py`: ground-pass heuristic and leave-one-out attribution.
- `features/PhasesOfPlayAggregator.py`: match-isolated phase aggregation.
- `visualization/pitch_viz.py`: generic XY lane map and optional ground-score time series.

The public import namespace remains `src.*`. Aerial scoring is explicitly
unsupported; body-pose z must not be interpreted as absolute elevation.
