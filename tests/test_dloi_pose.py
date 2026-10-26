"""
tests/test_dloi_pose.py
=======================
Automated verification suite for DLOI-Pose:
- Boundary conditions & numerical stability
- 3D pose alignment & visual gaze sensitivity
- Monotonicity across orientations (facing >= side >= turned)
- 3D aerial clearance gating vs standing/jumping vertical reach
- Zero-length passes, missing joints & high MAE noise resilience
"""

import sys
from pathlib import Path
import unittest
import numpy as np

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

    def test_zero_length_pass(self):
        carrier = np.array([10.0, 5.0], dtype=np.float64)
        target = np.array([10.0, 5.0], dtype=np.float64)
        defs = np.array([[10.0, 5.0]], dtype=np.float64)
        score = self.engine.compute_frame_occlusion(carrier, target, defs)
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

    def test_monotonic_orientation_hazard(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([20.0, 0.0], dtype=np.float64)
        defender_xy = np.array([[10.0, 2.0]], dtype=np.float64)

        meta_facing = [{
            "joints": {
                "lShoulder": {"xyz": [10.3, 2.0, 1.5], "p90_mae_cm": 3.0},
                "rShoulder": {"xyz": [9.7, 2.0, 1.5], "p90_mae_cm": 3.0},
                "neck": {"xyz": [10.0, 2.0, 1.6], "p90_mae_cm": 3.0},
                "nose": {"xyz": [10.0, 1.6, 1.6], "p90_mae_cm": 3.0},
            }
        }]
        meta_side = [{
            "joints": {
                "lShoulder": {"xyz": [10.0, 2.3, 1.5], "p90_mae_cm": 3.0},
                "rShoulder": {"xyz": [10.0, 1.7, 1.5], "p90_mae_cm": 3.0},
                "neck": {"xyz": [10.0, 2.0, 1.6], "p90_mae_cm": 3.0},
                "nose": {"xyz": [10.4, 2.0, 1.6], "p90_mae_cm": 3.0},
            }
        }]
        meta_turned = [{
            "joints": {
                "lShoulder": {"xyz": [9.7, 2.0, 1.5], "p90_mae_cm": 3.0},
                "rShoulder": {"xyz": [10.3, 2.0, 1.5], "p90_mae_cm": 3.0},
                "neck": {"xyz": [10.0, 2.0, 1.6], "p90_mae_cm": 3.0},
                "nose": {"xyz": [10.0, 2.4, 1.6], "p90_mae_cm": 3.0},
            }
        }]

        s_facing = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=meta_facing, use_pose=True)
        s_side = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=meta_side, use_pose=True)
        s_turned = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=meta_turned, use_pose=True)

        self.assertGreaterEqual(s_facing, s_side)
        self.assertGreaterEqual(s_side, s_turned)

    def test_missing_or_noisy_joints_fallback(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([20.0, 0.0], dtype=np.float64)
        defender_xy = np.array([[10.0, 1.5]], dtype=np.float64)

        noisy_meta = [{
            "joints": {
                "lShoulder": {"xyz": [10.3, 1.5, 1.5], "p90_mae_cm": 45.0},
                "rShoulder": {"xyz": [9.7, 1.5, 1.5], "p90_mae_cm": 50.0},
            }
        }]
        none_meta = [{"joints": None}]

        score_noisy = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=noisy_meta, use_pose=True)
        score_none = self.engine.compute_frame_occlusion(carrier, target, defender_xy, defenders_meta=none_meta, use_pose=True)
        score_base = self.engine.compute_frame_occlusion(carrier, target, defender_xy, use_pose=False)

        self.assertFalse(np.isnan(score_noisy))
        self.assertFalse(np.isnan(score_none))
        self.assertAlmostEqual(score_noisy, score_base, places=4)
        self.assertAlmostEqual(score_none, score_base, places=4)

    def test_aerial_clearance_filtering(self):
        carrier = np.array([0.0, 0.0], dtype=np.float64)
        target = np.array([25.0, 0.0], dtype=np.float64)
        blocker_xy = np.array([[12.5, 0.0]], dtype=np.float64)

        score_ground = self.engine.compute_frame_occlusion(carrier, target, blocker_xy, is_aerial=False, use_pose=False)
        score_aerial = self.engine.compute_frame_occlusion(carrier, target, blocker_xy, is_aerial=True, use_pose=False)
        self.assertLess(score_aerial, score_ground)

    def test_mathematical_bounds_and_stability(self):
        carrier = np.array([-52.0, -33.0], dtype=np.float64)
        target = np.array([52.0, 33.0], dtype=np.float64)
        defs = np.array([[-30.0, -10.0], [0.0, 0.0], [30.0, 15.0]], dtype=np.float64)

        score = self.engine.compute_frame_occlusion(carrier, target, defs, use_pose=False)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertFalse(np.isnan(score))


if __name__ == "__main__":
    unittest.main()