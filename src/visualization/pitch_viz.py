"""Generic XY visualization of heuristic ground-pass scores."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge


def plot_defensive_lanes(
    carrier_xy: np.ndarray,
    ball_xy: np.ndarray,
    defenders_xy: np.ndarray,
    defenders_headings: List[Optional[np.ndarray]],
    targets: Dict[str, Tuple[np.ndarray, float]],
    teammates_xy: Optional[np.ndarray] = None,
    defenders_names: Optional[List[str]] = None,
    teammates_names: Optional[List[str]] = None,
    defenders_detected: Optional[List[bool]] = None,
    goalkeeper_xy: Optional[np.ndarray] = None,
    temporal_df: Optional[pd.DataFrame] = None,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    coverage_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> plt.Figure:
    """Plot every supplied target, without inferred players, teams or aerial physics.

    Heading wedges are illustrative directions, not physical reach/FOV estimates.
    ``temporal_df`` requires ``t_sec`` and one numeric score column per lane.
    Caller-supplied scores must be finite and within [0, 1].
    ``coverage_grid`` is (x coordinates, y coordinates, scores[y, x]) evaluated
    from this same ball origin. It shows lane risk, not pitch-control probability.
    """
    has_temporal = temporal_df is not None and not temporal_df.empty
    for names, points in ((defenders_names, defenders_xy), (teammates_names, teammates_xy)):
        if names is not None and (points is None or len(names) != len(points)):
            raise ValueError("Name lists must align with their player positions")
    if defenders_detected is not None and len(defenders_detected) != len(defenders_xy):
        raise ValueError("Detection flags must align with defender positions")
    if has_temporal:
        if "t_sec" not in temporal_df or len(temporal_df.columns) < 2:
            raise ValueError("temporal_df requires t_sec and at least one score column")
        values = temporal_df.to_numpy(dtype=float)
        scores = temporal_df.drop(columns="t_sec").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Temporal times/scores must be finite and scores in [0, 1]")
    for _, score in targets.values():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Target scores must be finite and in [0, 1]")
    if coverage_grid is not None:
        grid_x, grid_y, grid_scores = (np.asarray(value, dtype=float) for value in coverage_grid)
        if (
            grid_x.ndim != 1 or grid_y.ndim != 1
            or len(grid_x) < 2 or len(grid_y) < 2
            or grid_scores.shape != (len(grid_y), len(grid_x))
            or not all(np.isfinite(value).all() for value in (grid_x, grid_y, grid_scores))
            or not (np.diff(grid_x) > 0).all() or not (np.diff(grid_y) > 0).all()
            or ((grid_scores < 0) | (grid_scores > 1)).any()
        ):
            raise ValueError("coverage_grid requires increasing finite x/y and scores[y, x] in [0, 1]")

    fig, axes = plt.subplots(1, 2 if has_temporal else 1, figsize=(20 if has_temporal else 12, 9))
    ax = axes[0] if has_temporal else axes
    ax.set_facecolor("#f8fbf8")
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="black")
    ax.axvline(0, color="grey", linestyle="--")
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="grey", linewidth=0.7))
    for sign in (-1, 1):
        for depth, half_width in ((16.5, 20.16), (5.5, 9.16)):
            ax.plot([sign * 52.5, sign * (52.5 - depth), sign * (52.5 - depth), sign * 52.5],
                    [-half_width, -half_width, half_width, half_width], color="grey", linewidth=0.7)
    if coverage_grid is not None:
        surface = ax.pcolormesh(grid_x, grid_y, grid_scores, cmap="RdYlGn_r",
                               vmin=0, vmax=1, shading="nearest", alpha=0.22, zorder=0)
        fig.colorbar(surface, ax=ax, fraction=0.025, pad=0.02,
                     label="Ground-lane occlusion from ball (not pitch control)")
    for j, position in enumerate(defenders_xy):
        detected = defenders_detected is None or defenders_detected[j]
        ax.scatter(*position, color="#de2d26", alpha=1 if detected else 0.4,
                   label="Defenders (faded = not detected)" if j == 0 else None)
        if defenders_names is not None:
            ax.annotate(defenders_names[j], position, xytext=(0, 7), textcoords="offset points", fontsize=7)
        heading = defenders_headings[j] if j < len(defenders_headings) else None
        if heading is not None:
            angle = math.degrees(math.atan2(heading[1], heading[0]))
            ax.add_patch(Wedge(position, 3, angle - 30, angle + 30, color="red", alpha=0.1))
            ax.annotate("", position + np.asarray(heading) * 2.5, xytext=position,
                        arrowprops={"arrowstyle": "->", "color": "#990000", "lw": 1})
    if teammates_xy is not None:
        for j, position in enumerate(teammates_xy):
            ax.scatter(*position, color="#3182bd", label="Teammates" if j == 0 else None)
            if teammates_names is not None:
                ax.annotate(teammates_names[j], position, xytext=(0, -10),
                            textcoords="offset points", fontsize=7, ha="center", color="#084594")
    ax.scatter(*carrier_xy, marker="s", color="blue", label="Carrier")
    ax.scatter(*ball_xy, color="orange", edgecolors="black", label="Ball")
    if goalkeeper_xy is not None:
        ax.scatter(*goalkeeper_xy, marker="s", color="red", label="Defending goalkeeper")
    option_labels = []
    for index, (name, (position, score)) in enumerate(targets.items(), start=1):
        ax.plot([ball_xy[0], position[0]], [ball_xy[1], position[1]],
                color=plt.cm.RdYlGn_r(score), linewidth=1.5, alpha=0.85, zorder=1)
        ax.annotate(str(index), position, xytext=(7, 6), textcoords="offset points",
                    fontsize=8, weight="bold",
                    bbox={"boxstyle": "circle,pad=0.2", "fc": "white", "ec": plt.cm.RdYlGn_r(score)})
        option_labels.append(f"{index}. {name}: {score:.2f}")
    ax.set(xlim=(-55, 55), ylim=(-36, 36), xlabel="Pitch X [m]", ylabel="Pitch Y [m]",
           title="Ground-pass heuristic" + title_suffix)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    if has_temporal:
        for column in temporal_df.columns:
            if column != "t_sec":
                axes[1].plot(temporal_df["t_sec"], temporal_df[column], label=column)
        axes[1].set(xlabel="Elapsed time [s]", ylabel="Occlusion index", ylim=(0, 1),
                    title="Ground-pass score evolution")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.2)
    rows = ["     ".join(option_labels[i:i + 3]) for i in range(0, len(option_labels), 3)]
    bottom = min(0.4, 0.07 + 0.025 * len(rows))
    if rows:
        fig.text(0.5, 0.02, "\n".join(rows), ha="center", va="bottom", fontsize=8,
                 bbox={"boxstyle": "round", "fc": "#f4f6f4", "ec": "#cccccc"})
    fig.tight_layout(rect=(0, bottom if rows else 0, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
