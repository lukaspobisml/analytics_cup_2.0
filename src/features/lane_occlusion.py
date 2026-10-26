"""
src/features/lane_occlusion.py
==============================
Dynamic Lane Occlusion Index with 3D Body Pose (DLOI-Pose).
Spatiotemporal reach envelopes, dual anatomical gating (torso + gaze),
ground-pass heuristics and counterfactual cover shadow attribution.

Analytics Cup 2.0 | Football Track: Defensive Positioning.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.features.pose_orientation import (
    shoulder_orientation_deg,
    PLAUSIBLE_WIDTH_M,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    joint_xy,
)


class DefensiveLaneOcclusion:
    """
    Computes a heuristic ground-pass index, not a calibrated probability.
    Aerial scoring is unsupported without pitch-relative elevation inputs.

    Attributes:
        r_body: Direct physical collision radius of a defender's torso in meters.
        r_lunge: Baseline physical lunge reach in meters without prior movement.
        w_torso: Weight blending torso and head-direction proxies.
        n_samples: Number of numerical quadrature evaluation points along the passing ray.
        v_ball: Initial ball velocity off the foot in m/s (default 15.0 m/s).
    """

    def __init__(
        self,
        direct_body_radius: float = 0.65,
        base_lunge_reach: float = 1.45,
        max_aerial_reach_m: Optional[float] = None,
        torso_gaze_blend: float = 0.60,
        num_quadrature_samples: int = 60,
        v_ball_m_s: float = 15.0,
    ) -> None:
        if max_aerial_reach_m is not None:
            raise NotImplementedError("Aerial reach requires pitch-relative elevation inputs")
        for name, value in (
            ("direct_body_radius", direct_body_radius),
            ("base_lunge_reach", base_lunge_reach), ("v_ball_m_s", v_ball_m_s),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(torso_gaze_blend, bool) or not isinstance(torso_gaze_blend, Real) or not math.isfinite(torso_gaze_blend) or not 0 <= torso_gaze_blend <= 1:
            raise ValueError("torso_gaze_blend must be in [0, 1]")
        if isinstance(num_quadrature_samples, bool) or not isinstance(num_quadrature_samples, (int, np.integer)) or num_quadrature_samples < 2:
            raise ValueError("num_quadrature_samples must be an integer >= 2")
        self.r_body = direct_body_radius
        self.r_lunge = base_lunge_reach
        self.w_torso = torso_gaze_blend
        self.n_samples = num_quadrature_samples
        self.v_ball = v_ball_m_s

    def extract_torso_vector(self, player_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extracts validated 2D unit torso facing vector from 3D shoulder keypoints.
        Filters coordinate/error validity and anatomical shoulder width.
        """
        joints = player_dict.get("joints")
        if not joints:
            return None
        left = joint_xy(joints.get(LEFT_SHOULDER))
        right = joint_xy(joints.get(RIGHT_SHOULDER))
        if left is None or right is None:
            return None

        lx, ly = left
        rx, ry = right
        deg = shoulder_orientation_deg(lx, ly, rx, ry)
        if deg is None:
            return None

        width = math.hypot(rx - lx, ry - ly)
        if not PLAUSIBLE_WIDTH_M[0] <= width <= PLAUSIBLE_WIDTH_M[1]:
            return None

        rad = math.radians(deg)
        return np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)

    def extract_gaze_vector(self, player_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extracts 2D unit head-gaze vector from 3D neck-to-nose keypoints.
        This is a head-direction proxy, not measured eye gaze or scanning range.
        """
        joints = player_dict.get("joints")
        if not joints:
            return None
        neck, nose = joint_xy(joints.get("neck")), joint_xy(joints.get("nose"))
        if neck is None or nose is None:
            return None

        dx = nose[0] - neck[0]
        dy = nose[1] - neck[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-4:
            return None

        return np.array([dx / norm, dy / norm], dtype=np.float64)

    def extract_pose_heading(self, player_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Normalize the configured blend of torso/head-direction proxies.
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
            is_aerial: True raises NotImplementedError; absolute heights are unavailable.
            use_pose: If True, modulates reaction time and acceleration via 3D keypoints.

        Returns:
            Float occlusion index in [0.0, 1.0], where 0.0 is open and 1.0 is occluded.
        """
        if is_aerial:
            raise NotImplementedError(
                "Aerial scoring requires pitch-relative ball and defender heights; "
                "centroid-relative pose z cannot supply them."
            )
        ball_pos, target_pos, defenders_xy, defenders_vel = self._validate_inputs(
            ball_pos, target_pos, defenders_xy, defenders_vel, defenders_meta
        )
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

        # Arrival-time heuristic, not integrated turf-friction dynamics.
        eff_speed = np.maximum(min(9.0, self.v_ball), self.v_ball - 0.12 * pass_len * lambdas)
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
                    lunge_reach = self.r_lunge * (0.75 + alignment) / 1.45

            # 4. Spatiotemporal Interception Reach Envelope
            dt_active = np.clip(t_total - tau_react, 0.0, 0.95)
            r_kinematic = lunge_reach + 0.5 * accel_eff * (dt_active ** 1.8)

            # 5. Point Hazard: Direct body collision + Kinematic interception
            raw_dist_body = np.linalg.norm(ray_xy - def_pos[None, :], axis=1)
            direct_body = np.exp(-0.5 * (raw_dist_body / self.r_body) ** 2)
            kinematic_hazard = np.exp(-0.5 * (dist_to_ray / r_kinematic) ** 2.6)
            defender_ground_hazard = np.maximum(direct_body, kinematic_hazard)

            point_hazards = np.maximum(point_hazards, defender_ground_hazard)

        # 7. Integration: Bottleneck risk + Cumulative exposure
        bottleneck_risk = float(np.max(point_hazards))
        flight_penalty = 1.0 + 0.02 * pass_len
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
        Computes Leave-One-Out (LOO) Cover Shadow Attribution (delta DLOI) for each defender
        across all candidate passing options. Quantifies non-contact defensive contribution.

        Returns:
            List of dictionaries with marginal shadow values per defender.
        """
        ball_pos, _, defenders_xy, defenders_vel = self._validate_inputs(
            ball_pos, ball_pos, defenders_xy, defenders_vel, defenders_meta
        )
        if not candidate_targets:
            raise ValueError("candidate_targets must contain at least one lane")
        n_defs = len(defenders_xy)

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

            pid = defenders_meta[j].get("player_id", j) if defenders_meta is not None else j
            attribution_results.append({
                "defender_idx": j,
                "player_id": pid,
                "peak_shadow_added": float(np.max(deltas)),
                "mean_shadow_added": float(np.mean(deltas)),
            })

        return attribution_results

    @staticmethod
    def _validate_inputs(ball_pos, target_pos, defenders_xy, defenders_vel, defenders_meta):
        ball_pos = np.asarray(ball_pos, dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        defenders_xy = np.asarray(defenders_xy, dtype=np.float64)
        if ball_pos.shape != (2,) or target_pos.shape != (2,):
            raise ValueError("ball_pos and target_pos must have shape (2,)")
        if defenders_xy.ndim != 2 or defenders_xy.shape[1] != 2:
            raise ValueError("defenders_xy must have shape (N, 2)")
        arrays = [ball_pos, target_pos, defenders_xy]
        if defenders_vel is not None:
            defenders_vel = np.asarray(defenders_vel, dtype=np.float64)
            if defenders_vel.shape != defenders_xy.shape:
                raise ValueError("defenders_vel must match defenders_xy shape")
            arrays.append(defenders_vel)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("positions and velocities must be finite")
        if defenders_meta is not None:
            if len(defenders_meta) != len(defenders_xy):
                raise ValueError("defenders_meta must contain one entry per defender")
            if not all(isinstance(meta, dict) for meta in defenders_meta):
                raise ValueError("defenders_meta entries must be dictionaries")
            if any(meta.get("joints") is not None and not isinstance(meta["joints"], dict) for meta in defenders_meta):
                raise ValueError("joints must be a dictionary or None")
        return ball_pos, target_pos, defenders_xy, defenders_vel