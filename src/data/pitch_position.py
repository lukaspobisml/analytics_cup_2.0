"""
Clean Team & Tactical Position Visualizer (No DLOI Lanes or Cones)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


ROLE_ABBREVIATIONS: Dict[str, str] = {
    "Goalkeeper": "GK",
    "Central Defender": "CB",
    "Full Back": "FB",
    "Midfield": "CM",
    "Wide Attacker": "WA",
    "Center Forward": "CF",
}


def plot_team_positions_only(
    frame_record: Dict[str, Any],
    active_pids: Set[int],
    team_lookup: Dict[int, Dict[str, str]],
    show_camera_fov: bool = True,
    show_ghosts: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots strictly the players on the pitch grouped by team, displaying their
    tactical role abbreviation inside the marker and short name underneath.

    Args:
        frame_record: Single frame dictionary from SkillCorner JSONL.
        active_pids: Set of active on-pitch player IDs (22 players).
        team_lookup: Mapping of player_id to name, team_name, and position_group.
        show_camera_fov: Whether to draw the broadcast camera frustum polygon.
        show_ghosts: If True, also plots the 10 substituted-off frozen players in grey.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib.pyplot.Figure
    """
    fig, ax = plt.subplots(figsize=(14.0, 8.5), dpi=300)
    ax.set_facecolor("#f7faf7")

    # 1. Standard SkillCorner Pitch Geometry (105m x 68m centered at (0,0))
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34.0, -34.0, 34.0, 34.0, -34.0],
            color="#1f1f1f", lw=1.8, zorder=2)
    ax.plot([0, 0], [-34.0, 34.0], color="#666666", lw=1.2, ls="--", zorder=2)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(9.15 * np.cos(theta), 9.15 * np.sin(theta), color="#888888", lw=0.9, ls=":", zorder=2)

    # Both Penalty Boxes & 6-Yard Boxes (Left: x < -36m, Right: x > +36m)
    for sign in (-1, 1):
        ax.plot([sign * 36.0, sign * 36.0, sign * 52.5], [-20.16, 20.16, 20.16], color="#1f1f1f", lw=1.3, zorder=2)
        ax.plot([sign * 36.0, sign * 52.5], [-20.16, -20.16], color="#1f1f1f", lw=1.3, zorder=2)
        ax.plot([sign * 47.0, sign * 47.0, sign * 52.5], [-9.16, 9.16, 9.16], color="#1f1f1f", lw=1.0, zorder=2)
        ax.plot([sign * 47.0, sign * 52.5], [-9.16, -9.16], color="#1f1f1f", lw=1.0, zorder=2)
        ax.scatter([sign * 41.5], [0.0], color="#1f1f1f", s=18, zorder=3)
        ax.plot([sign * 52.5, sign * 53.5, sign * 53.5, sign * 52.5],
                [-3.66, -3.66, 3.66, 3.66], color="#000000", lw=1.8, zorder=2)

    # 2. Optional Broadcast Camera Field of View (FOV Polygon)
    if show_camera_fov and "image_corners_projection" in frame_record:
        icp = frame_record["image_corners_projection"]
        cam_poly = np.array([
            [icp["x_bottom_left"], icp["y_bottom_left"]],
            [icp["x_bottom_right"], icp["y_bottom_right"]],
            [icp["x_top_right"], icp["y_top_right"]],
            [icp["x_top_left"], icp["y_top_left"]],
        ])
        fov_patch = Polygon(
            cam_poly, closed=True, facecolor="#fff7bc", edgecolor="#d95f0e",
            lw=1.2, ls="-.", alpha=0.22, label="Broadcast Camera FOV", zorder=1
        )
        ax.add_patch(fov_patch)

    # 3. Separate Players by Team & Status
    carrier_id = frame_record.get("possession", {}).get("player_id")
    brisbane_plotted = False
    perth_plotted = False
    ghost_plotted = False

    for p in frame_record.get("player_data", []):
        pid = p["player_id"]
        px, py = p["x"], p["y"]
        is_det = bool(p.get("is_detected", False))

        if pid not in active_pids:
            if show_ghosts:
                ax.scatter(
                    [px], [py], color="#999999", s=120, marker="x", lw=1.5,
                    label="Substituted Ghost (Frozen)" if not ghost_plotted else None, zorder=4
                )
                ax.text(px, py - 1.4, f"Ghost #{pid}", fontsize=6.5, ha="center", color="#666666")
                ghost_plotted = True
            continue

        meta = team_lookup.get(pid, {})
        team_name = meta.get("team_name", "Unknown")
        short_name = meta.get("short_name", f"#{pid}")
        pos_group = meta.get("position_group", "Outfield")
        role_code = ROLE_ABBREVIATIONS.get(pos_group, "PL")

        is_brisbane = "Brisbane" in team_name
        if is_brisbane:
            fill_color = "#084594" if pid == carrier_id else "#2171b5"
            edge_color = "#ff7f0e" if pid == carrier_id else ("white" if is_det else "#08306b")
            txt_color = "#08306b"
            lbl = "Brisbane Roar FC (Home - In Possession)" if not brisbane_plotted else None
            brisbane_plotted = True
        else:
            fill_color = "#cb181d"
            edge_color = "white" if is_det else "#67000d"
            txt_color = "#67000d"
            lbl = "Perth Glory FC (Away - Out of Possession)" if not perth_plotted else None
            perth_plotted = True

        # Draw player node (Square for GK, Circle for Outfield; dashed border if off-camera)
        marker_shape = "s" if role_code == "GK" else "o"
        ax.scatter(
            [px], [py],
            color=fill_color,
            s=280 if pid == carrier_id else 230,
            marker=marker_shape,
            edgecolors=edge_color,
            lw=2.4 if pid == carrier_id else 1.8,
            ls="-" if is_det else "--",
            alpha=1.0 if is_det else 0.72,
            label=lbl,
            zorder=6,
        )

        # Role abbreviation inside the circle/square
        ax.text(
            px, py, role_code,
            fontsize=7.0, weight="bold", color="white",
            ha="center", va="center", zorder=7
        )

        # Player name + coordinates below marker
        ax.text(
            px, py - 1.65,
            f"{short_name}\n({px:.1f}, {py:.1f})",
            fontsize=7.2, weight="bold", color=txt_color,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=8,
        )

    # 4. Plot Ball
    ball_d = frame_record.get("ball_data", {})
    if ball_d.get("x") is not None:
        ax.scatter(
            [ball_d["x"]], [ball_d["y"]],
            color="#ff7f0e", s=110, edgecolors="black", lw=1.3,
            label="Ball", zorder=9
        )

    # Full 105m x 68m view so even Macklin Freke (Brisbane GK at x = -33.9m) is visible
    ax.set_xlim(-42.0, 54.5)
    ax.set_ylim(-36.0, 36.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Team Tactical Formations & Player Positions | Frame {frame_record['frame']} ({frame_record['timestamp']})",
        fontsize=12, weight="bold", pad=12
    )
    ax.set_xlabel("Pitch X Coordinate [meters] (Brisbane Roar Attacking $\\rightarrow$ | $\\leftarrow$ Perth Glory Attacking)", fontsize=10)
    ax.set_ylabel("Pitch Y Coordinate [meters]", fontsize=10)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.07),
        ncol=4, frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5
    )
    ax.grid(True, ls=":", color="#bbbbbb", alpha=0.55)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig