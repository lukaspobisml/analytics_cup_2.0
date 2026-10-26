"""
src/features/lane_occlusion.py
==============================
Dynamic Lane Occlusion Index with 3D Body Pose (DLOI-Pose).
Spatiotemporal reach envelopes, dual anatomical gating (torso + gaze),
3D parabolic aerial trajectories, and counterfactual cover shadow attribution.

Designed for native integration into PySport / databallpy.
Analytics Cup 2.0 | Football Track: Defensive Positioning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.features.pose_orientation import (
    shoulder_orientation_deg,
    PLAUSIBLE_WIDTH_M,
    DEFAULT_MAX_ERR_CM,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
)


class DefensiveLaneOcclusion:
    """
    Computes physically and biomechanically grounded passing lane occlusion (DLOI-Pose)
    supporting ground passes, 3D aerial trajectories, and counterfactual attribution.

    Attributes:
        r_body: Direct physical collision radius of a defender's torso in meters.
        r_lunge: Baseline physical lunge reach in meters without prior movement.
        max_aerial_z: Maximum vertical reach of an outfield defender (jump + head).
        w_torso: Weighting factor blending torso orientation (w_torso) with head gaze.
        n_samples: Number of numerical quadrature evaluation points along the passing ray.
        v_ball: Initial ball velocity off the foot in m/s (default 15.0 m/s).
    """

    def __init__(
        self,
        direct_body_radius: float = 0.65,
        base_lunge_reach: float = 1.45,
        max_aerial_reach_m: float = 2.40,
        torso_gaze_blend: float = 0.60,
        num_quadrature_samples: int = 60,
        v_ball_m_s: float = 15.0,
    ) -> None:
        self.r_body = direct_body_radius
        self.r_lunge = base_lunge_reach
        self.max_aerial_z = max_aerial_reach_m
        self.w_torso = torso_gaze_blend
        self.n_samples = num_quadrature_samples
        self.v_ball = v_ball_m_s

    def extract_torso_vector(self, player_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extracts validated 2D unit torso facing vector from 3D shoulder keypoints.
        Performs 5-factor anatomical quality filtering (MAE error, anatomical width).
        """
        joints = player_dict.get("joints")
        if not joints:
            return None
        l_sh, r_sh = joints.get(LEFT_SHOULDER), joints.get(RIGHT_SHOULDER)
        if not l_sh or not r_sh:
            return None

        lx, ly = l_sh["xyz"][0], l_sh["xyz"][1]
        rx, ry = r_sh["xyz"][0], r_sh["xyz"][1]
        deg = shoulder_orientation_deg(lx, ly, rx, ry)
        if deg is None:
            return None

        width = math.hypot(rx - lx, ry - ly)
        err = max(l_sh.get("p90_mae_cm") or math.inf, r_sh.get("p90_mae_cm") or math.inf)
        if not (PLAUSIBLE_WIDTH_M[0] <= width <= PLAUSIBLE_WIDTH_M[1] and err <= DEFAULT_MAX_ERR_CM):
            return None

        rad = math.radians(deg)
        return np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)

    def extract_gaze_vector(self, player_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extracts 2D unit head-gaze vector from 3D neck-to-nose keypoints.
        Provides the visual scanning direction independent of torso rotation.
        """
        joints = player_dict.get("joints")
        if not joints:
            return None
        neck, nose = joints.get("neck"), joints.get("nose")
        if not neck or not nose:
            return None

        err = max(neck.get("p90_mae_cm") or math.inf, nose.get("p90_mae_cm") or math.inf)
        if err > DEFAULT_MAX_ERR_CM:
            return None

        dx = nose["xyz"][0] - neck["xyz"][0]
        dy = nose["xyz"][1] - neck["xyz"][1]
        norm = math.hypot(dx, dy)
        if norm < 1e-4:
            return None

        return np.array([dx / norm, dy / norm], dtype=np.float64)

    def extract_pose_heading(self, player_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Blends 3D torso orientation (60%) and head visual gaze (40%) into a unified
        biomechanical orientation vector.
        """
        v_torso = self.extract_torso_vector(player_dict)
        v_gaze = self.extract_gaze_vector(player_dict)
        if v_torso is not None and v_gaze is not None:
            blended = self.w_torso * v_torso + (1.0 - self.w_torso) * v_gaze
            norm = np.linalg.norm(blended)
            return blended / norm if norm > 1e-5 else v_torso
        return v_torso if v_torso is not None else v_gaze

    def compute_frame_occlusion(
        self,
        ball_pos: np.ndarray,
        target_pos: np.ndarray,
        defenders_xy: np.ndarray,
        defenders_vel: Optional[np.ndarray] = None,
        defenders_meta: Optional[List[Dict[str, Any]]] = None,
        is_aerial: bool = False,
        use_pose: bool = True,
    ) -> float:
        """
        Calculates the DLOI-Pose passing lane occlusion score [0.0, 1.0].

        Args:
            ball_pos: 2D array [x, y] of origin / ball carrier position.
            target_pos: 2D array [x, y] of pass receiver or targeted spatial channel.
            defenders_xy: (N, 2) array of outfield defenders' positions.
            defenders_vel: Optional (N, 2) array of instantaneous velocity vectors (m/s).
            defenders_meta: Optional list of raw player dicts containing 3D keypoint dictionaries.
            is_aerial: If True, models a 3D parabolic aerial ball trajectory over defenders.
            use_pose: If True, modulates reaction time and acceleration via 3D keypoints.

        Returns:
            Float occlusion index in [0.0, 1.0], where 0.0 is open and 1.0 is occluded.
        """
        n_defs = len(defenders_xy)
        if n_defs == 0:
            return 0.0

        pass_vec = target_pos - ball_pos
        pass_len = float(np.linalg.norm(pass_vec))
        if pass_len < 1e-4:
            return 0.0

        # Discretize passing trajectory into quadrature evaluation points
        lambdas = np.linspace(0.05, 1.00, self.n_samples)
        ray_xy = ball_pos[None, :] + lambdas[:, None] * pass_vec[None, :]

        # 1. 3D Ball vertical trajectory profile
        if is_aerial:
            # Parabola peaking at lambda = 0.5 with realistic height clearance
            z_ball = 0.30 + 4.0 * 2.40 * lambdas * (1.0 - lambdas)
        else:
            z_ball = 0.15 + 0.10 * np.sin(lambdas * np.pi)

        # 2. Kinematic ball flight time with turf friction deceleration
        eff_speed = np.maximum(9.0, self.v_ball - 0.12 * pass_len * lambdas)
        t_flight = (lambdas * pass_len) / eff_speed
        t_control = 0.35 * np.clip((lambdas - 0.75) / 0.25, 0.0, 1.0)
        t_total = t_flight + t_control

        point_hazards = np.zeros(self.n_samples, dtype=np.float64)

        for j in range(n_defs):
            def_pos = defenders_xy[j]
            v_vec = defenders_vel[j] if defenders_vel is not None else np.zeros(2, dtype=np.float64)

            # Bounded momentum drift: defender in zone cannot drift blindly past reaction window
            t_drift = np.minimum(t_total, 0.35)[:, None]
            proj_def_pos = def_pos[None, :] + v_vec[None, :] * t_drift
            dist_to_ray = np.linalg.norm(ray_xy - proj_def_pos, axis=1)

            # Baseline isotropic parameters
            tau_react = np.full(self.n_samples, 0.28, dtype=np.float64)
            accel_eff = np.full(self.n_samples, 4.20, dtype=np.float64)
            lunge_reach = np.full(self.n_samples, self.r_lunge, dtype=np.float64)

            # Extract 3D head elevation if available
            def_head_z = 1.80
            if defenders_meta is not None and j < len(defenders_meta):
                jnts = defenders_meta[j].get("joints")
                if jnts and "nose" in jnts:
                    def_head_z = float(jnts["nose"]["xyz"][2])

            # 3. 3D Body Pose Biomechanical Modulation
            if use_pose and defenders_meta is not None and j < len(defenders_meta):
                heading = self.extract_pose_heading(defenders_meta[j])
                if heading is not None:
                    raw_delta = ray_xy - def_pos[None, :]
                    raw_dist = np.linalg.norm(raw_delta, axis=1) + 1e-5
                    cos_phi = np.sum(raw_delta * heading[None, :], axis=1) / raw_dist
                    alignment = (1.0 + cos_phi) / 2.0  # 1.0 = facing pass, 0.0 = back turned

                    # Facing passing line allows explosive reaction; back turned incurs turning delay
                    tau_react = 0.58 - 0.42 * alignment
                    accel_eff = 1.80 + 3.20 * alignment
                    lunge_reach = 0.75 + 1.00 * alignment

            # 4. Spatiotemporal Interception Reach Envelope
            dt_active = np.clip(t_total - tau_react, 0.0, 0.95)
            r_kinematic = lunge_reach + 0.5 * accel_eff * (dt_active ** 1.8)

            # 5. Point Hazard: Direct body collision + Kinematic interception
            raw_dist_body = np.linalg.norm(ray_xy - def_pos[None, :], axis=1)
            direct_body = np.exp(-0.5 * (raw_dist_body / self.r_body) ** 2)
            kinematic_hazard = np.exp(-0.5 * (dist_to_ray / r_kinematic) ** 2.6)
            defender_ground_hazard = np.maximum(direct_body, kinematic_hazard)

            # 6. 3D Aerial Clearance Filtering
            if is_aerial:
                max_vert_reach = def_head_z + 0.65  # standing head + explosive jump reach
                z_clearance = z_ball - max_vert_reach
                # Smooth exponential decay if ball clears defender's jumping reach
                aerial_gate = np.exp(-0.5 * (np.maximum(0.0, z_clearance) / 0.35) ** 2)
                defender_hazard = np.where(
                    z_ball > max_vert_reach,
                    defender_ground_hazard * aerial_gate,
                    defender_ground_hazard * 0.70,
                )
            else:
                defender_hazard = defender_ground_hazard

            point_hazards = np.maximum(point_hazards, defender_hazard)

        # 7. Integration: Bottleneck risk + Cumulative exposure
        bottleneck_risk = float(np.max(point_hazards))
        flight_penalty = 1.0 + 0.02 * pass_len
        # Univerzálna integrácia bez DeprecationWarning (kompatibilná s NumPy 1.x aj 2.x)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        cumulative_exposure = float(trapz_fn(point_hazards, lambdas)) * flight_penalty       
        occlusion = 1.0 - (1.0 - bottleneck_risk) * math.exp(-0.60 * cumulative_exposure)
        return float(np.clip(occlusion, 0.0, 1.0))

    def compute_counterfactual_attribution(
        self,
        ball_pos: np.ndarray,
        candidate_targets: Dict[str, Tuple[np.ndarray, bool]],
        defenders_xy: np.ndarray,
        defenders_vel: Optional[np.ndarray] = None,
        defenders_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Computes Leave-One-Out (LOO) Cover Shadow Attribution (\Delta DLOI) for each defender
        across all candidate passing options. Quantifies non-contact defensive contribution.

        Returns:
            List of dictionaries with marginal shadow values per defender.
        """
        n_defs = len(defenders_xy)
        if n_defs == 0:
            return []

        # Baseline: Full team DLOI scores
        full_scores = {}
        for opt_name, (tgt_xy, is_air) in candidate_targets.items():
            full_scores[opt_name] = self.compute_frame_occlusion(
                ball_pos, tgt_xy, defenders_xy, defenders_vel, defenders_meta,
                is_aerial=is_air, use_pose=True
            )

        attribution_results = []
        for j in range(n_defs):
            mask = [k for k in range(n_defs) if k != j]
            sub_xy = defenders_xy[mask]
            sub_vel = defenders_vel[mask] if defenders_vel is not None else None
            sub_meta = [defenders_meta[k] for k in mask] if defenders_meta is not None else None

            deltas = []
            for opt_name, (tgt_xy, is_air) in candidate_targets.items():
                without_score = self.compute_frame_occlusion(
                    ball_pos, tgt_xy, sub_xy, sub_vel, sub_meta,
                    is_aerial=is_air, use_pose=True
                )
                delta = max(0.0, full_scores[opt_name] - without_score)
                deltas.append(delta)

            pid = defenders_meta[j]["player_id"] if defenders_meta is not None else j
            attribution_results.append({
                "defender_idx": j,
                "player_id": pid,
                "peak_shadow_added": float(np.max(deltas)),
                "mean_shadow_added": float(np.mean(deltas)),
            })

        return attribution_results