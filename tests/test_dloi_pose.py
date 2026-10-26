"""
tests/test_dloi_pose.py
=======================
Automated verification suite for DLOI-Pose mathematical bounds,
kinematic acceleration gating, and aerial occlusion logic.
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Dynamicky pridá koreňový priečinok projektu do sys.path (funguje out-of-the-box)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.lane_occlusion import DefensiveLaneOcclusion


class TestDLOIPose(unittest.TestCase):
    def setUp(self):
        self.engine = DefensiveLaneOcclusion(v_ball_m_s=15.0)

    def test_empty_pitch_boundary(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([20.0, 0.0], dtype=np.float64)
        empty_defs = np.zeros((0, 2), dtype=np.float64)
        score = self.engine.compute_frame_occlusion(carrier, target, empty_defs)
        self.assertEqual(score, 0.0)

    def test_collinear_blocker_boundary(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([20.0, 0.0], dtype=np.float64)
        blocker_xy = np.array([[10.0, 0.0]], dtype=np.float64)
        score = self.engine.compute_frame_occlusion(carrier, target, blocker_xy, use_pose=False)
        self.assertTrue(0.90 <= score <= 1.0)

    def test_pose_alignment_sensitivity(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([15.0, 0.0], dtype=np.float64)
        defender_xy = np.array([[8.0, 2.5]], dtype=np.float64)

        facing_meta = [{
            "joints": {
                "lShoulder": {"xyz": [8.3, 2.5, 1.5], "p90_mae_cm": 4.0},
                "rShoulder": {"xyz": [7.7, 2.5, 1.5], "p90_mae_cm": 4.0},
                "neck": {"xyz": [8.0, 2.5, 1.6], "p90_mae_cm": 4.0},
                "nose": {"xyz": [8.0, 2.1, 1.6], "p90_mae_cm": 4.0},
            }
        }]
        turned_meta = [{
            "joints": {
                "lShoulder": {"xyz": [7.7, 2.5, 1.5], "p90_mae_cm": 4.0},
                "rShoulder": {"xyz": [8.3, 2.5, 1.5], "p90_mae_cm": 4.0},
                "neck": {"xyz": [8.0, 2.5, 1.6], "p90_mae_cm": 4.0},
                "nose": {"xyz": [8.0, 2.9, 1.6], "p90_mae_cm": 4.0},
            }
        }]

        score_facing = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=facing_meta, use_pose=True)
        score_turned = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=turned_meta, use_pose=True)
        self.assertGreater(score_facing, score_turned)

    def test_aerial_clearance_filtering(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([25.0, 0.0], dtype=np.float64)
        blocker_xy = np.array([[12.5, 0.0]], dtype=np.float64)

        score_ground = self.engine.compute_frame_occlusion(carrier, target, blocker_xy, is_aerial=False, use_pose=False)
        score_aerial = self.engine.compute_frame_occlusion(carrier, target, blocker_xy, is_aerial=True, use_pose=False)
        self.assertLess(score_aerial, score_ground)


if __name__ == "__main__":
    unittest.main()