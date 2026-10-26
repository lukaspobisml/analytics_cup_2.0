# DLOI-Pose

DLOI-Pose is an exploratory ground-pass occlusion heuristic using player XY
positions, optional velocities and optional shoulder/head-direction proxies.
It is not a calibrated interception probability or a validated physical model.

**Aerial scoring is unsupported:** body-pose z is centroid-relative, not an
absolute height above the pitch. Historical aerial results are withdrawn.

The repository [README](https://github.com/lukaspobisml/analytics_cup_2.0#readme)
contains the current algorithm, installation instructions and limitations.
The ground-only demonstration is in `notebooks/defensive_lane_occlusion.ipynb`.
It uses the committed compressed sample without downloading a full match.

Build this documentation from the repository:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```
