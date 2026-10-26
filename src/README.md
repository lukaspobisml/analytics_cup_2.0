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

## Notebook error: `KeyError: 'Central_Break_Pose'`

Older plotters hard-coded temporal column names that differed from the notebook's
`Central Line Break` and other descriptive labels. The current plotter requires
`t_sec` and plots every other supplied score column by its actual name; no
`Central_Break_Pose` alias is needed.

Update to the fixed branch from your repository checkout:

```bash
git fetch origin
git switch lpobis-cen58041-dloi-pose-audit-and-fixes
git pull --ff-only
python -m pip install -e ".[notebook]"
```

Preserve any local changes before switching. In Jupyter, **restart the kernel**
and run all cells in `notebooks/defensive_lane_occlusion.ipynb`, not an older
root-level copy. A running kernel retains imported functions from the old
plotter even after files on disk have been updated. To check the loaded source:

```python
import inspect
from src.visualization.pitch_viz import plot_defensive_lanes
print(inspect.getsourcefile(plot_defensive_lanes))
print(inspect.getsource(plot_defensive_lanes))
```
