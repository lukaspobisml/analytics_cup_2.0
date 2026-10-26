"""
src/visualization/pitch_viz.py
==============================
Publication-grade tactical pitch & temporal window visualizer for Analytics Cup 2.0.
Renders uncluttered 3D Pose Cover Shadows, lettered badges [A]-[G], dashed 3D aerial
trajectories, and a synchronized 2-panel Spatial-Temporal layout.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge, Circle


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
) -> plt.Figure:
    """
    Renders the uncluttered spatial pitch map (Panel A) and, if temporal_df is provided,
    attaches the temporal decision window evolution chart (Panel B) in a single figure.
    """
    has_temporal = temporal_df is not None and not temporal_df.empty
    if has_temporal:
        fig = plt.figure(figsize=(18.5, 9.2), dpi=300)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.16)
        ax_pitch = fig.add_subplot(gs[0])
        ax_time = fig.add_subplot(gs[1])
    else:
        fig, ax_pitch = plt.subplots(figsize=(14.2, 8.8), dpi=300)
        ax_time = None

    fig.patch.set_facecolor("white")
    ax_pitch.set_facecolor("#f8fbf8")

    # 1. Pitch Geometry (105m x 68m)
    ax_pitch.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34.0, -34.0, 34.0, 34.0, -34.0], color="#222222", lw=1.6, zorder=2)
    ax_pitch.plot([0, 0], [-34.0, 34.0], color="#777777", lw=1.1, ls="--", zorder=2)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax_pitch.plot(9.15 * np.cos(theta), 9.15 * np.sin(theta), color="#999999", lw=0.9, ls=":", zorder=2)

    for sgn in [-1, 1]:
        ax_pitch.plot([sgn * 36.0, sgn * 36.0, sgn * 52.5], [-20.16, 20.16, 20.16], color="#222222", lw=1.2, zorder=2)
        ax_pitch.plot([sgn * 36.0, sgn * 52.5], [-20.16, -20.16], color="#222222", lw=1.2, zorder=2)
        ax_pitch.plot([sgn * 47.0, sgn * 47.0, sgn * 52.5], [-9.16, 9.16, 9.16], color="#444444", lw=0.9, zorder=2)
        ax_pitch.plot([sgn * 47.0, sgn * 52.5], [-9.16, -9.16], color="#444444", lw=0.9, zorder=2)
        ax_pitch.scatter([sgn * 41.5], [0.0], color="#222222", s=15, zorder=3)
        ax_pitch.plot([sgn * 52.5, sgn * 53.5, sgn * 53.5, sgn * 52.5], [-3.66, -3.66, 3.66, 3.66], color="#000000", lw=1.8, zorder=2)

    # 2. 3D Pose Reach & Gaze Cones
    for j, d_xy in enumerate(defenders_xy):
        heading = defenders_headings[j] if j < len(defenders_headings) else None
        if heading is not None:
            angle_deg = math.degrees(math.atan2(heading[1], heading[0]))
            fov_cone = Wedge(
                center=(d_xy[0], d_xy[1]), r=3.8,
                theta1=angle_deg - 50.0, theta2=angle_deg + 50.0,
                facecolor="#d62728", edgecolor="#990000", lw=0.5, alpha=0.15, zorder=3,
            )
            ax_pitch.add_patch(fov_cone)
        else:
            iso_circle = Circle(
                (d_xy[0], d_xy[1]), radius=2.0,
                facecolor="#d62728", edgecolor="none", alpha=0.07, zorder=3
            )
            ax_pitch.add_patch(iso_circle)

    # 3. Attacking Teammates (Brisbane Roar)
    if teammates_xy is not None and len(teammates_xy) > 0:
        ax_pitch.scatter(
            teammates_xy[:, 0], teammates_xy[:, 1],
            color="#3182bd", s=145, edgecolors="white", lw=1.5,
            label="Brisbane Roar (Attacking)", zorder=6
        )
        if teammates_names:
            for k, t_xy in enumerate(teammates_xy):
                y_off = -1.45
                if "O'Shea" in teammates_names[k]:
                    y_off = -1.55
                elif "Jelacic" in teammates_names[k]:
                    y_off = -1.65
                ax_pitch.text(
                    t_xy[0], t_xy[1] + y_off, teammates_names[k],
                    fontsize=7.2, ha="center", va="top", color="#08306b", weight="semibold", zorder=8
                )

    # 4. Ball Carrier & Ball
    ax_pitch.scatter([carrier_xy[0]], [carrier_xy[1]], color="#084594", s=210, edgecolors="#ff7f0e", lw=2.2, label="Ball Carrier (W. Shour)", zorder=8)
    ax_pitch.text(carrier_xy[0], carrier_xy[1] - 1.65, "W. Shour (Carrier)", fontsize=7.6, ha="center", va="top", color="#084594", weight="bold", zorder=9)
    ax_pitch.scatter([ball_xy[0]], [ball_xy[1]], color="#ff7f0e", s=95, edgecolors="black", lw=1.1, label="Ball", zorder=9)

    # 5. Defending Block (Perth Glory)
    plotted_det, plotted_off = False, False
    for j, d_xy in enumerate(defenders_xy):
        is_det = True if defenders_detected is None else defenders_detected[j]
        lbl = None
        if is_det and not plotted_det:
            lbl = "Perth Glory (3D Pose Detected)"
            plotted_det = True
        elif not is_det and not plotted_off:
            lbl = "Perth Glory (Off-Camera Extrapolated)"
            plotted_off = True

        ax_pitch.scatter(
            [d_xy[0]], [d_xy[1]],
            color="#de2d26" if is_det else "#fc9272",
            s=155, edgecolors="white" if is_det else "#cb181d",
            lw=1.5, ls="-" if is_det else "--", label=lbl, zorder=7
        )
        if defenders_names and j < len(defenders_names):
            ax_pitch.text(
                d_xy[0], d_xy[1] + 1.25, defenders_names[j],
                fontsize=7.1, ha="center", va="bottom", color="#67000d", weight="bold", zorder=8
            )
        heading = defenders_headings[j] if j < len(defenders_headings) else None
        if heading is not None:
            ax_pitch.arrow(
                d_xy[0], d_xy[1], heading[0] * 2.5, heading[1] * 2.5,
                head_width=0.65, head_length=0.65, fc="#99000d", ec="#99000d", alpha=0.9, lw=1.3, zorder=8
            )

    if goalkeeper_xy is not None:
        ax_pitch.scatter([goalkeeper_xy[0]], [goalkeeper_xy[1]], color="#fdbb84", s=165, marker="s", edgecolors="#d62728", lw=1.5, label="Goalkeeper (C. Cook)", zorder=6)
        ax_pitch.text(goalkeeper_xy[0] + 0.9, goalkeeper_xy[1] - 0.3, "C. Cook (GK)", fontsize=7.2, weight="bold", color="#7f2704")

    # 6. Passing Lanes & Lettered Badges [A]-[G]
    cmap = plt.cm.RdYlGn_r
    option_legend_items = []
    ordered_keys = [
        ("A", "Wide Wing Space (Brazete Run)"),
        ("B", "Right Flank to Feet (J. Brazete)"),
        ("C", "Midfield Pocket (J. O'Shea)"),
        ("D", "Left Half-Space (K. Jelacic)"),
        ("E", "Aerial Channel Lob (Jelacic Run)"),
        ("F", "Central Line Break (Pennington)"),
        ("G", "Backpass Safety (L. Herrington)"),
    ]

    if "Left Half-Space (K. Jelacic)" in targets and "Aerial Channel Lob (Jelacic Run)" in targets:
        jel_start = targets["Left Half-Space (K. Jelacic)"][0]
        jel_lob = targets["Aerial Channel Lob (Jelacic Run)"][0]
        ax_pitch.annotate(
            "", xy=(jel_lob[0] - 0.7, jel_lob[1] + 0.3), xytext=(jel_start[0] + 0.8, jel_start[1] - 0.3),
            arrowprops=dict(arrowstyle="->", color="#084594", lw=1.6, ls=":", mutation_scale=12), zorder=6
        )

    for badge, opt_name in ordered_keys:
        if opt_name not in targets:
            continue
        target_pos, score = targets[opt_name]
        lane_color = cmap(score)
        is_air = "Aerial" in opt_name or "Lob" in opt_name

        ax_pitch.plot(
            [carrier_xy[0], target_pos[0]], [carrier_xy[1], target_pos[1]],
            color=lane_color, lw=2.6 if not is_air else 2.3,
            ls="-" if not is_air else "--", alpha=0.92, zorder=5
        )
        ax_pitch.scatter(
            [target_pos[0]], [target_pos[1]],
            color=lane_color, s=245, edgecolors="#111111", lw=1.4, zorder=8
        )
        txt_contrast = "white" if (score > 0.65 or score < 0.30) else "#111111"
        ax_pitch.text(
            target_pos[0], target_pos[1], badge,
            fontsize=8.3, weight="bold", color=txt_contrast, ha="center", va="center", zorder=9
        )

        if badge == "C":
            tag_x, tag_y = target_pos[0] + 1.3, target_pos[1] + 1.2
        elif badge == "F":
            tag_x, tag_y = target_pos[0] + 1.4, target_pos[1] - 0.2
        elif badge == "G":
            tag_x, tag_y = target_pos[0] - 1.3, target_pos[1] + 1.2
        elif badge == "D":
            tag_x, tag_y = target_pos[0] - 1.3, target_pos[1] - 1.2
        elif badge == "E":
            tag_x, tag_y = target_pos[0] + 1.4, target_pos[1] - 1.1
        elif badge == "B":
            tag_x, tag_y = target_pos[0] + 1.2, target_pos[1] + 1.1
        else:
            tag_x, tag_y = target_pos[0], target_pos[1] + 1.3

        ax_pitch.text(
            tag_x, tag_y, f"{score:.2f}",
            fontsize=7.6, weight="bold", color="#111111", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=lane_color, lw=1.1, alpha=0.93),
            zorder=9
        )
        mode_tag = " [3D Lob]" if is_air else ""
        option_legend_items.append(f"[{badge}] {opt_name}{mode_tag}: {score:.2f}")

    ax_pitch.set_xlim(-12.5, 53.5)
    ax_pitch.set_ylim(-35.0, 35.0)
    ax_pitch.set_aspect("equal", adjustable="box")
    ax_pitch.set_title(
        "Panel A: Spatial DLOI-Pose & 3D Gaze Cones (Frame 124487, t = 0.00s)",
        fontsize=11.5, weight="bold", pad=10
    )
    ax_pitch.set_xlabel("Pitch X [m] (Brisbane Roar Attacking Direction ->)", fontsize=9.5)
    ax_pitch.set_ylabel("Pitch Y [m]", fontsize=9.5)
    ax_pitch.grid(True, ls=":", color="#cccccc", alpha=0.55)

    # 7. Optional Panel B: Temporal Decision Windows
    if has_temporal and ax_time is not None:
        ax_time.set_facecolor("#fcfdfc")
        t = temporal_df["t_sec"]
        ax_time.plot(t, temporal_df["Central_Break_Pose"], color="#d62728", lw=2.5, label="[F] Central Line Break (Blocked by Pennington)")
        ax_time.plot(t, temporal_df["Jelacic_Feet_Pose"], color="#9467bd", lw=2.3, ls="-.", label="[D] Left Half-Space to Feet (Closed by Gomulka)")
        ax_time.plot(t, temporal_df["Jelacic_Channel_Ground"], color="#8c564b", lw=1.8, ls=":", alpha=0.85, label="[E-Ground] Channel Pass on Turf (Blocked by Mrcela)")
        ax_time.plot(t, temporal_df["Jelacic_Channel_Aerial"], color="#1f77b4", lw=2.5, ls="--", label="[E-Aerial] 3D Channel Lob over Mrcela (Jelacic Run)")
        ax_time.plot(t, temporal_df["OShea_Pocket_Pose"], color="#ff7f0e", lw=2.1, label="[C] Midfield Pocket (J. O'Shea - Early Window)")
        ax_time.plot(t, temporal_df["Wide_Wing_Pose"], color="#2ca02c", lw=2.8, label="[A] Wide Wing Space (Brazete Run - Actual Pass)")

        ax_time.fill_between(
            t, temporal_df["Jelacic_Channel_Aerial"], temporal_df["Jelacic_Channel_Ground"],
            color="#1f77b4", alpha=0.12, label="3D Aerial Clearance Gain (Over Mrcela's 2.39m Jump)"
        )
        ax_time.axvline(2.40, color="#111111", ls=":", lw=1.8, zorder=4)
        ax_time.text(
            2.44, 0.88, "Actual Pass Release\n(t = 2.40s, Frame 124547)",
            fontsize=8.2, weight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffcc", ec="#888888", lw=0.9)
        )
        ax_time.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.06)
        ax_time.text(0.5, 0.04, "Early Central Window\n(0.0s - 1.0s)", fontsize=7.8, ha="center", color="#1b5e20", weight="semibold")

        ax_time.set_xlim(0.0, float(t.max()))
        ax_time.set_ylim(-0.02, 1.04)
        ax_time.set_title("Panel B: Temporal Evolution of Passing Windows & 3D Aerial Gain (25fps)", fontsize=11.5, weight="bold", pad=10)
        ax_time.set_xlabel("Build-Up Phase Time [seconds] (SkillCorner 25fps 3D Stream)", fontsize=9.5)
        ax_time.set_ylabel("DLOI-Pose Occlusion Score [0 = Open, 1 = Blocked]", fontsize=9.5)
        ax_time.grid(True, ls=":", color="#cccccc", alpha=0.65)
        ax_time.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.94)

    ax_pitch.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3 if has_temporal else 5,
        frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=7.8
    )

    if option_legend_items:
        row1 = "  •  ".join(option_legend_items[:4])
        row2 = "  •  ".join(option_legend_items[4:])
        card_text = f"Tactical Options Scorecard (Frame 124487, t = 0.00s):\n{row1}\n{row2}"
        fig.text(
            0.5, 0.01, card_text,
            ha="center", va="bottom", fontsize=7.8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.38", fc="#f2f6f2", ec="#bbbbbb", lw=1.0)
        )

    fig.suptitle(
        "Analytics Cup 2.0 | DLOI-Pose: Spatiotemporal Lane Occlusion with 3D Body Pose, Gaze & Aerial Physics" + title_suffix,
        fontsize=13, weight="bold", y=0.98
    )
    plt.subplots_adjust(top=0.90, bottom=0.22)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig