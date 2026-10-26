"""
src/data/data_loader.py
=======================
Streaming loader, ghost-player filter, and kinematic velocity estimator
for SkillCorner 25fps 3D Body Pose and 10fps Tracking datasets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


MATCH_1925299_ROSTER: Dict[int, Dict[str, str]] = {
    11897: {"player_name": "Josh Risdon", "short_name": "J. Risdon", "team_name": "Perth Glory Football Club", "position_group": "Full Back"},
    50982: {"player_name": "Riley Warland", "short_name": "R. Warland", "team_name": "Perth Glory Football Club", "position_group": "Full Back"},
    50999: {"player_name": "Macklin Freke", "short_name": "M. Freke", "team_name": "Brisbane Roar FC", "position_group": "Goalkeeper"},
    51002: {"player_name": "Jay O'Shea", "short_name": "J. O'Shea", "team_name": "Brisbane Roar FC", "position_group": "Midfield"},
    51694: {"player_name": "Taras Gomulka", "short_name": "T. Gomulka", "team_name": "Perth Glory Football Club", "position_group": "Midfield"},
    51722: {"player_name": "Jack Hingert", "short_name": "J. Hingert", "team_name": "Brisbane Roar FC", "position_group": "Full Back"},
    69111: {"player_name": "Hosine Bility", "short_name": "H. Bility", "team_name": "Brisbane Roar FC", "position_group": "Central Defender"},
    287934: {"player_name": "Walid Shour", "short_name": "W. Shour", "team_name": "Brisbane Roar FC", "position_group": "Central Defender"},
    560983: {"player_name": "Antonee Burke-Gilroy", "short_name": "A. Burke-Gilroy", "team_name": "Brisbane Roar FC", "position_group": "Full Back"},
    560984: {"player_name": "David Williams", "short_name": "D. Williams", "team_name": "Perth Glory Football Club", "position_group": "Wide Attacker"},
    560985: {"player_name": "Cameron Cook", "short_name": "C. Cook", "team_name": "Perth Glory Football Club", "position_group": "Goalkeeper"},
    560986: {"player_name": "Keegan Jelacic", "short_name": "K. Jelacic", "team_name": "Brisbane Roar FC", "position_group": "Center Forward"},
    560988: {"player_name": "Marcus Ferkranus", "short_name": "M. Ferkranus", "team_name": "Brisbane Roar FC", "position_group": "Central Defender"},
    560989: {"player_name": "Tomislav Mrčela", "short_name": "T. Mrčela", "team_name": "Perth Glory Football Club", "position_group": "Central Defender"},
    795530: {"player_name": "Nicholas Pennington", "short_name": "N. Pennington", "team_name": "Perth Glory Football Club", "position_group": "Center Forward"},
    795533: {"player_name": "Jarrod Carluccio", "short_name": "J. Carluccio", "team_name": "Perth Glory Football Club", "position_group": "Full Back"},
    795537: {"player_name": "Thomas Waddingham", "short_name": "T. Waddingham", "team_name": "Brisbane Roar FC", "position_group": "Center Forward"},
    795541: {"player_name": "Kaelan Majekodunmi", "short_name": "K. Majekodunmi", "team_name": "Perth Glory Football Club", "position_group": "Central Defender"},
    809166: {"player_name": "Adriano Lebib", "short_name": "A. Lebib", "team_name": "Perth Glory Football Club", "position_group": "Central Defender"},
    965697: {"player_name": "Lucas Herrington", "short_name": "L. Herrington", "team_name": "Brisbane Roar FC", "position_group": "Central Defender"},
    965698: {"player_name": "Jacob Brazete", "short_name": "J. Brazete", "team_name": "Brisbane Roar FC", "position_group": "Midfield"},
    966115: {"player_name": "Khoa Ngo", "short_name": "K. Ngo", "team_name": "Perth Glory Football Club", "position_group": "Wide Attacker"},
}


def stream_jsonl(filepath: str | Path) -> Generator[Dict[str, Any], None, None]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_frame_entities(
    frame_dict: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ball, detected carrier and other detected players (not team/role filtered)."""
    ball_data = frame_dict.get("ball_data") or {}
    if any(ball_data.get(axis) is None for axis in ("x", "y")):
        raise ValueError("frame requires finite ball x/y coordinates")
    ball_xy = np.array([ball_data["x"], ball_data["y"]], dtype=np.float64)
    if not np.isfinite(ball_xy).all():
        raise ValueError("frame requires finite ball x/y coordinates")

    possession = frame_dict.get("possession") or {}
    carrier_id = possession.get("player_id")

    carrier_dict = None
    outfield_players = []
    for player in frame_dict.get("player_data", []):
        if not player.get("is_detected", False):
            continue
        if player.get("player_id") == carrier_id:
            carrier_dict = player
        else:
            outfield_players.append(player)

    return ball_xy, carrier_dict, outfield_players


def identify_active_players(frames: List[Dict[str, Any]], min_std_m: float = 0.05) -> Set[int]:
    """Sample-only movement filter; not a lineup/substitution detector."""
    if not math.isfinite(min_std_m) or min_std_m < 0:
        raise ValueError("min_std_m must be finite and nonnegative")
    trajectories: Dict[int, List[Tuple[float, float]]] = {}
    for frame in frames:
        for p in frame.get("player_data", []):
            pid = p["player_id"]
            trajectories.setdefault(pid, []).append((p["x"], p["y"]))

    active_pids: Set[int] = set()
    for pid, coords in trajectories.items():
        arr = np.array(coords, dtype=np.float64)
        if np.std(arr[:, 0]) > min_std_m or np.std(arr[:, 1]) > min_std_m:
            active_pids.add(pid)
    return active_pids


def build_team_lookup(project_root: Path) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {
        pid: dict(meta) for pid, meta in MATCH_1925299_ROSTER.items()
    }
    for fpath in project_root.rglob("aus1league_*aggregates*.csv"):
        try:
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                pid = int(row["player_id"])
                if pid not in lookup:
                    p_name = str(row["player_name"])
                    s_name = str(row.get("player_short_name", p_name))
                    lookup[pid] = {
                        "player_name": p_name,
                        "short_name": s_name,
                        "team_name": str(row["team_name"]),
                        "position_group": str(row["position_group"]),
                    }
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            raise ValueError(f"Cannot read team lookup from {fpath}: {exc}") from exc
    return lookup


def compute_frame_velocities(
    frames: List[Dict[str, Any]],
    frame_idx: int,
    window: int = 5,
    fps: float = 25.0,
) -> Dict[int, np.ndarray]:
    """Estimate velocity over up to ``window`` frame intervals within one period.

    Frame identifiers, not list offsets, determine elapsed time. Centered differences
    use future observations, so this helper is for retrospective analysis.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be an integer >= 1")
    if not isinstance(frame_idx, int) or not 0 <= frame_idx < len(frames):
        raise ValueError("frame_idx must index a supplied frame")
    period = frames[frame_idx].get("period")
    lower = upper = frame_idx
    while lower > max(0, frame_idx - window) and frames[lower - 1].get("period") == period:
        lower -= 1
    while upper < min(len(frames) - 1, frame_idx + window) and frames[upper + 1].get("period") == period:
        upper += 1
    idx_start = max(lower, frame_idx - window // 2)
    idx_end = min(upper, idx_start + window)
    idx_start = max(lower, idx_end - window)
    ids = [frame["frame"] for frame in frames[idx_start:idx_end + 1]]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in ids):
        raise ValueError("frame identifiers must be finite numbers")
    if any(end <= start for start, end in zip(ids, ids[1:])):
        raise ValueError("frame identifiers must increase within a period")
    dt = (ids[-1] - ids[0]) / fps
    if dt <= 1e-5:
        return {}

    pos_start = {p["player_id"]: np.array([p["x"], p["y"]], dtype=np.float64)
                 for p in frames[idx_start].get("player_data", [])}
    pos_end = {p["player_id"]: np.array([p["x"], p["y"]], dtype=np.float64)
               for p in frames[idx_end].get("player_data", [])}
    if any(not np.isfinite(pos).all() for pos in (*pos_start.values(), *pos_end.values())):
        raise ValueError("player coordinates must be finite")

    return {
        pid: (p_end - pos_start[pid]) / dt
        for pid, p_end in pos_end.items()
        if pid in pos_start
    }