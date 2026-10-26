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
    fx, fy = -(ry - ly), rx - lx
    if math.hypot(fx, fy) < MIN_SHOULDER_WIDTH_M:
        return None
    return math.degrees(math.atan2(fy, fx))


def fold_to_tracking_frame(pose_frame: int) -> int:
    return math.floor(pose_frame * (10 / 25) + 0.5)


def shoulder_table(records, max_err_cm=DEFAULT_MAX_ERR_CM):
    return pd.DataFrame()


def gating_report(records, max_err_cm=DEFAULT_MAX_ERR_CM):
    return pd.DataFrame()