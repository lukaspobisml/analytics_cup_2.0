"""Regression coverage for scientific scope, IO contracts and reproducibility."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_loader import compute_frame_velocities, extract_frame_entities
from src.data import pose_loading
from src.features import DefensiveLaneOcclusion, shoulder_table, gating_report
from src.features.PhasesOfPlayAggregator import PhasesOfPlayAggregator
from src.visualization.pitch_viz import plot_defensive_lanes

ROOT = Path(__file__).resolve().parents[1]


def pose(error=0, z=-100):
    return {"joints": {
        "lShoulder": {"xyz": [0.2, 0, z], "p90_mae_cm": error},
        "rShoulder": {"xyz": [-0.2, 0, z], "p90_mae_cm": error},
        "neck": {"xyz": [0, 0, z], "p90_mae_cm": error},
        "nose": {"xyz": [0, -0.2, z], "p90_mae_cm": error},
    }}


class TestModelContracts(unittest.TestCase):
    def setUp(self):
        self.model = DefensiveLaneOcclusion()
        self.xy = np.array([[10., 3.]])

    def test_invalid_parameters(self):
        cases = [
            {"direct_body_radius": 0}, {"direct_body_radius": None}, {"v_ball_m_s": "fast"},
            {"base_lunge_reach": -1}, {"v_ball_m_s": np.nan},
            {"torso_gaze_blend": 1.1}, {"torso_gaze_blend": np.inf},
            {"num_quadrature_samples": 0}, {"num_quadrature_samples": 1},
            {"num_quadrature_samples": 2.5}, {"num_quadrature_samples": True},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DefensiveLaneOcclusion(**kwargs)
        with self.assertRaises(NotImplementedError):
            DefensiveLaneOcclusion(max_aerial_reach_m=2.4)

    def test_invalid_inputs(self):
        cases = [
            {"ball_pos": [0, np.nan]}, {"target_pos": [1, 2, 3]},
            {"defenders_xy": [1, 2]}, {"defenders_vel": np.zeros((2, 2))},
            {"defenders_xy": [[np.inf, 0]]}, {"defenders_meta": []},
            {"defenders_meta": [None]}, {"defenders_meta": [{"joints": []}]},
        ]
        for overrides in cases:
            args = dict(ball_pos=[0, 0], target_pos=[20, 0], defenders_xy=self.xy)
            args.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.model.compute_frame_occlusion(**args)

    def test_zero_error_is_accepted_and_bad_errors_rejected(self):
        self.assertIsNotNone(self.model.extract_pose_heading(pose()))
        for error in (-1, np.nan, np.inf, None, 16):
            with self.subTest(error=error):
                self.assertIsNone(self.model.extract_pose_heading(pose(error)))

    def test_bad_optional_joints_fall_back(self):
        baseline = self.model.compute_frame_occlusion([0, 0], [20, 0], self.xy, use_pose=False)
        for nose in ({}, {"xyz": [np.nan, 0], "p90_mae_cm": 0}, {"xyz": [0], "p90_mae_cm": 0}):
            meta = [{"joints": {"nose": nose}}]
            self.assertEqual(baseline, self.model.compute_frame_occlusion([0, 0], [20, 0], self.xy, defenders_meta=meta))

    def test_ground_score_invariant_to_pose_z(self):
        scores = [self.model.compute_frame_occlusion([0, 0], [20, 0], self.xy, defenders_meta=[pose(z=z)])
                  for z in (-100, 0, 100, np.nan)]
        np.testing.assert_allclose(scores, scores[0])

    def test_configured_lunge_affects_pose_score(self):
        scores = [
            DefensiveLaneOcclusion(base_lunge_reach=reach).compute_frame_occlusion(
                [0, 0], [20, 0], self.xy, defenders_meta=[pose()])
            for reach in (0.5, 3.)
        ]
        self.assertGreater(scores[1], scores[0])

    def test_slow_ball_setting_not_overridden(self):
        scores = [DefensiveLaneOcclusion(v_ball_m_s=speed).compute_frame_occlusion(
            [0, 0], [8, 0], [[4, 2.5]], use_pose=False) for speed in (3, 8)]
        self.assertGreater(scores[0], scores[1])

    def test_attribution_contract_and_monotonicity(self):
        targets = {"lane": ([20, 0], False)}
        results = self.model.compute_counterfactual_attribution(
            [0, 0], targets, [[10, 0], [5, 8]], defenders_meta=[{}, {}])
        self.assertEqual([r["player_id"] for r in results], [0, 1])
        for row in results:
            self.assertTrue(0 <= row["mean_shadow_added"] <= row["peak_shadow_added"] <= 1)
        self.assertEqual(self.model.compute_counterfactual_attribution([0, 0], targets, np.empty((0, 2))), [])
        with self.assertRaises(ValueError):
            self.model.compute_counterfactual_attribution([0, 0], {}, self.xy)
        with self.assertRaises(ValueError):
            self.model.compute_counterfactual_attribution([0, 0], targets, self.xy, defenders_meta=[])
        with self.assertRaises(NotImplementedError):
            self.model.compute_counterfactual_attribution([0, 0], {"air": ([20, 0], True)}, np.empty((0, 2)))


class TestPoseTables(unittest.TestCase):
    def test_tables_and_empty_schemas(self):
        records = [{"frame": 5, "period": 1, "player_data": [
            dict(player_id=1, **pose()), {"player_id": 2, "joints": None}]}]
        table = shoulder_table(records)
        self.assertEqual(table["accepted"].tolist(), [True, False])
        self.assertAlmostEqual(table.loc[0, "orientation_deg"], -90)
        report = gating_report(iter(records)).set_index("player_id")
        self.assertEqual(report.loc[1, "accepted"], 1)
        self.assertEqual(report.loc[2, "rejected"], 1)
        self.assertEqual(list(shoulder_table([])), list(table))
        self.assertEqual(list(gating_report([])), ["player_id", "observations", "accepted", "rejected"])
        with self.assertRaises(ValueError):
            shoulder_table([], -1)

    def test_empty_joint_frame_can_be_filtered(self):
        table = pose_loading.joints_to_frame(iter([]))
        self.assertEqual(pose_loading.skeleton_segments(table, max_mae_cm=15), [])

    def test_committed_sample_counts(self):
        frames = list(pose_loading.iter_sample())
        self.assertEqual(len(frames), pose_loading.manifest()["sample"]["pose_frames"])
        self.assertEqual(len(pose_loading.joints_to_frame(iter(frames))), 143637)
        self.assertTrue(shoulder_table(iter(frames))["accepted"].any())


class TestLoaders(unittest.TestCase):
    @staticmethod
    def frame(number, x, period=1):
        return {"frame": number, "period": period, "player_data": [{"player_id": 1, "x": x, "y": 0}]}

    def test_sparse_velocity_uses_frame_time(self):
        frames = [self.frame(10, 0), self.frame(20, 4)]
        np.testing.assert_allclose(compute_frame_velocities(frames, 0, fps=10)[1], [4, 0])

    def test_velocity_does_not_cross_periods(self):
        frames = [self.frame(10, 0), self.frame(20, 4), self.frame(10, 100, 2)]
        np.testing.assert_allclose(compute_frame_velocities(frames, 1, fps=10)[1], [4, 0])
        self.assertEqual(compute_frame_velocities(frames, 2), {})

    def test_velocity_invalid_inputs(self):
        frames = [self.frame(10, 0), self.frame(9, 1)]
        for kwargs in ({"fps": 0}, {"window": 0}, {"frame_idx": 3}, {}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_frame_velocities(frames, **dict({"frame_idx": 0}, **kwargs))

    def test_missing_ball_is_not_fabricated(self):
        for record in ({}, {"ball_data": None}, {"ball_data": {"x": np.nan, "y": 0}}):
            with self.assertRaises(ValueError):
                extract_frame_entities(record)
        ball, carrier, others = extract_frame_entities({"ball_data": {"x": 1, "y": 2}, "possession": None})
        np.testing.assert_array_equal(ball, [1, 2])
        self.assertIsNone(carrier)
        self.assertEqual(others, [])

    def test_basic_loader_import_is_lazy(self):
        import importlib
        with patch("pandas.read_csv", side_effect=AssertionError("import performed IO")):
            importlib.reload(importlib.import_module("src.data.basic_loading"))

    def test_download_verifies_cache_and_new_content(self):
        payload = b"tiny test archive"
        entry = {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        meta = {"matches": {"1": {}}, "files": {"raw/1.jsonl.zip": entry}, "base_url": "https://example.invalid"}
        with tempfile.TemporaryDirectory() as directory, patch.object(pose_loading, "manifest", return_value=meta):
            path = Path(directory) / "1.jsonl.zip"
            path.write_bytes(b"x" * len(payload))
            with patch.object(pose_loading.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as request:
                self.assertEqual(pose_loading.download_match(1, directory), path)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(path.read_bytes(), payload)
            with patch.object(pose_loading.urllib.request, "urlopen") as request:
                pose_loading.download_match(1, directory)
                request.assert_not_called()
            path.write_bytes(b"old")
            with patch.object(pose_loading.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(ValueError, "verification"):
                    pose_loading.download_match(1, directory)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).glob("*.part")), [])


class TestPhases(unittest.TestCase):
    def setUp(self):
        self.source = pd.read_csv(ROOT / "data/matches/1925299/1925299_phases_of_play.csv")

    def test_matches_are_independent_and_input_not_mutated(self):
        second = self.source.copy()
        second["match_id"] = 2
        both = pd.concat([self.source, second], ignore_index=True)
        original = both.copy(deep=True)
        combined = PhasesOfPlayAggregator(both)
        pd.testing.assert_frame_equal(both, original)
        for method in ("get_in_possession_aggregates", "get_out_of_possession_aggregates"):
            actual = getattr(combined, method)()
            expected = getattr(PhasesOfPlayAggregator(self.source), method)()
            actual = actual[actual.match_id == 1925299].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual.sort_index(axis=1), expected.sort_index(axis=1))

    def test_period_boundary_not_linked(self):
        rows = self.source.iloc[:2].copy()
        rows.loc[rows.index[1], "period"] = 2
        aggregator = PhasesOfPlayAggregator(pd.concat([rows, self.source.iloc[[2]]], ignore_index=True))
        self.assertEqual(aggregator.phases_of_play_df.iloc[0]["team_in_possession_next_phase"], "no_next_phase")

    def test_ambiguous_team_subset_rejected(self):
        with self.assertRaises(ValueError):
            PhasesOfPlayAggregator(self.source.iloc[:1])

    def test_different_rosters_are_mapped_per_match(self):
        second = self.source.copy()
        second["match_id"] = 2
        second["team_in_possession_id"] += 10000
        aggregator = PhasesOfPlayAggregator(pd.concat([self.source, second], ignore_index=True))
        result = aggregator.phases_of_play_df
        opponents = result[result.match_id == 2]["team_out_of_possession_id"]
        self.assertTrue((opponents >= 10000).all())


class TestReproducibility(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_generic_plot_and_temporal_contract(self):
        fig = plot_defensive_lanes(np.zeros(2), np.zeros(2), np.empty((0, 2)), [],
                                  {"Any target name": (np.array([5, 2]), 0.2)},
                                  temporal_df=pd.DataFrame({"t_sec": [0, 1], "Any target name": [0.2, 0.3]}))
        self.assertEqual(len(fig.axes), 2)
        fig.canvas.draw()
        self.assertTrue(any(text.get_text() == "Any target name" for text in fig.axes[0].texts))
        with self.assertRaises(ValueError):
            plot_defensive_lanes(np.zeros(2), np.zeros(2), np.empty((0, 2)), [], {},
                                 temporal_df=pd.DataFrame({"wrong": [0]}))

    def test_notebook_source_executes_without_downloads(self):
        notebook = json.loads((ROOT / "notebooks/defensive_lane_occlusion.ipynb").read_text())
        namespace = {"display": lambda value: None}
        with patch.object(pose_loading.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")), patch.object(plt, "show"):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    exec(compile("".join(cell["source"]), "notebook", "exec"), namespace)
        self.assertEqual(len(namespace["fig"].axes), 2)
