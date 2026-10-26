"""
src/features/pose_orientation.py
================================
Shoulder orientation and anatomical gating from SkillCorner 3D body-pose data.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

LEFT_SHOULDER = "lShoulder"
RIGHT_SHOULDER = "rShoulder"
MIN_SHOULDER_WIDTH_M = 1e-6
PLAUSIBLE_WIDTH_M = (0.15, 0.60)
DEFAULT_MAX_ERR_CM = 15.0


def shoulder_orientation_deg(lx: float, ly: float, rx: float, ry: float) -> Optional[float]:
    if not all(math.isfinite(v) for v in (lx, ly, rx, ry)):
        return None
    fx, fy = -(ry - ly), rx - lx
    if math.hypot(fx, fy) < MIN_SHOULDER_WIDTH_M:
        return None
    return math.degrees(math.atan2(fy, fx))


def fold_to_tracking_frame(pose_frame: int) -> int:
    return math.floor(pose_frame * (10 / 25) + 0.5)


def joint_xy(joint, max_err_cm=DEFAULT_MAX_ERR_CM):
    """Return finite XY for a joint with an acceptable nonnegative error bound."""
    if not isinstance(joint, dict):
        return None
    xyz = joint.get("xyz")
    err = joint.get("p90_mae_cm")
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
        return None
    values = (*xyz[:2], err)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return None
    return tuple(xyz[:2]) if 0 <= err <= max_err_cm else None


def shoulder_table(records, max_err_cm=DEFAULT_MAX_ERR_CM):
    """One row per player-frame, including rejected/missing shoulder observations."""
    if not math.isfinite(max_err_cm) or max_err_cm < 0:
        raise ValueError("max_err_cm must be finite and nonnegative")
    rows = []
    for record in records:
        for player in record.get("player_data") or []:
            joints = player.get("joints") or {}
            left = joint_xy(joints.get(LEFT_SHOULDER), max_err_cm)
            right = joint_xy(joints.get(RIGHT_SHOULDER), max_err_cm)
            width = math.hypot(right[0] - left[0], right[1] - left[1]) if left and right else None
            valid = width is not None and PLAUSIBLE_WIDTH_M[0] <= width <= PLAUSIBLE_WIDTH_M[1]
            rows.append({
                "frame": record["frame"], "period": record.get("period"),
                "player_id": player["player_id"], "width_m": width,
                "orientation_deg": shoulder_orientation_deg(*left, *right) if valid else None,
                "accepted": valid,
            })
    return pd.DataFrame(rows, columns=[
        "frame", "period", "player_id", "width_m", "orientation_deg", "accepted"
    ])


def gating_report(records, max_err_cm=DEFAULT_MAX_ERR_CM):
    """Count accepted/rejected shoulder observations per player (all periods)."""
    table = shoulder_table(records, max_err_cm)
    columns = ["player_id", "observations", "accepted", "rejected"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    report = table.groupby("player_id")["accepted"].agg(observations="size", accepted="sum")
    report["rejected"] = report["observations"] - report["accepted"]
    return report.reset_index()[columns]