import math
import json
import unittest

import numpy as np
import pandas as pd

from src.analysis.conflict_zone_analysis import (
    POTENTIAL_ZONE_CONFIG,
    build_spatial_figure,
    define_conflict_zone,
    get_conflict_zone_for_scenario,
    entry_order,
    occupancy_interval,
    occupancy_overlap,
    segment_intersection,
    synchronized_distance,
    zone_membership,
)
from src.analysis.trajectory_analysis import (
    deviation_frame,
    interaction_interval,
    interpolate_observed,
    normalize_trajectories,
    path_summary,
)


class TrajectoryAnalysisTests(unittest.TestCase):
    def test_interpolation_never_extrapolates(self) -> None:
        result = interpolate_observed(
            np.array([20.0, 50.0, 80.0]),
            np.array([2.0, 5.0, 8.0]),
            np.array([0.0, 20.0, 50.0, 80.0, 100.0]),
        )
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[-1]))
        np.testing.assert_allclose(result[1:4], [2, 5, 8])

    def test_normalization_preserves_real_duration(self) -> None:
        traces = pd.DataFrame({
            "participant_id": ["PedNYC1"] * 5,
            "recording_identifier": ["r1"] * 5,
            "scenario": [3] * 5,
            "scenario_time": [0, 1, 2, 3, 4],
            "aligned_time": [-1, 0, 1, 2, 3],
            "ped_speed": [0, .3, .4, .2, 0],
            "ped_x": [0, 0, 1, 2, 2],
            "ped_z": [0, 0, 0, 0, 0],
            "veh_x": [0, 1, 2, 3, 4],
            "veh_z": [1, 1, 1, 1, 1],
        })
        self.assertEqual(interaction_interval(traces), (1.0, 3.0))
        normalized, intervals = normalize_trajectories(traces, 11)
        self.assertEqual(intervals.interaction_duration.iloc[0], 2.0)
        self.assertEqual(normalized.progress_percent.min(), 0)
        self.assertEqual(normalized.progress_percent.max(), 100)

    def test_path_deviation_is_euclidean(self) -> None:
        normalized = pd.DataFrame({
            "participant_id": ["a", "b"],
            "recording_identifier": ["a", "b"],
            "scenario": [3, 3],
            "progress_percent": [0, 0],
            "ped_x": [0.0, 2.0],
            "ped_z": [0.0, 0.0],
            "veh_x": [0.0, 0.0],
            "veh_z": [0.0, 0.0],
        })
        summary = path_summary(normalized, "ped")
        deviation = deviation_frame(normalized, summary, "ped")
        np.testing.assert_allclose(deviation.deviation_m, [1.0, 1.0])


class ConflictAnalysisTests(unittest.TestCase):
    @staticmethod
    def _scenario_traces(scenario: int) -> pd.DataFrame:
        count = 160
        if scenario == 12:
            veh_x, veh_z = np.linspace(5, 0, count), np.zeros(count)
            ped_x, ped_z = np.zeros(count), np.linspace(-2, 2, count)
        elif scenario == 16:
            veh_x, veh_z = np.zeros(count), np.linspace(10, 0, count)
            ped_x, ped_z = np.linspace(-2, 2, count), np.full(count, 2.0)
        else:
            veh_x, veh_z = np.linspace(-2, 2, count), np.zeros(count)
            ped_x, ped_z = np.zeros(count), np.linspace(-2, 2, count)
        return pd.DataFrame({
            "recording_identifier": ["synthetic"] * count,
            "scenario_time": np.linspace(0, 10, count),
            "ped_x": ped_x, "ped_z": ped_z,
            "veh_x": veh_x, "veh_z": veh_z,
        })

    def test_scenarios_12_and_16_use_deterministic_turn_uncertainty_polygons(self) -> None:
        for scenario in (12, 16):
            traces = self._scenario_traces(scenario)
            first = get_conflict_zone_for_scenario(scenario, traces)
            second = get_conflict_zone_for_scenario(scenario, traces)
            self.assertEqual(first["conflict_zone_method"], "turn_uncertainty")
            self.assertEqual(first["shape"], "rectangle")
            self.assertGreaterEqual(len(first["polygon_xz_m"]), 3)
            np.testing.assert_allclose(first["polygon_xz_m"], second["polygon_xz_m"])
            polygon = np.asarray(first["polygon_xz_m"])
            width = float(polygon[:, 0].max() - polygon[:, 0].min())
            length = float(polygon[:, 1].max() - polygon[:, 1].min())
            area = .5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
            self.assertGreater(area, 0)
            self.assertAlmostEqual(width, POTENTIAL_ZONE_CONFIG[scenario]["uncertainty_zone_width_m"])
            self.assertAlmostEqual(length, POTENTIAL_ZONE_CONFIG[scenario]["uncertainty_zone_length_m"])

    def test_spatial_plot_uses_exact_exported_uncertainty_polygon(self) -> None:
        traces = self._scenario_traces(12)
        zone = get_conflict_zone_for_scenario(12, traces)
        exported_zone = json.loads(json.dumps(zone))
        event = pd.DataFrame([{
            "recording_identifier": "synthetic", "minimum_distance_time": 0.0,
            "pedestrian_entry_time": np.nan, "pedestrian_exit_time": np.nan,
            "vehicle_entry_time": np.nan, "vehicle_exit_time": np.nan,
        }])
        traces = traces.assign(participant_id="PedNYC1", scenario=12, aligned_time=traces.scenario_time,
                               ped_speed=1.0, veh_speed=1.0)
        figure = build_spatial_figure(traces, event, exported_zone)
        plotted = np.column_stack([figure.data[0].x[:-1], figure.data[0].y[:-1]])
        np.testing.assert_allclose(plotted, exported_zone["polygon_xz_m"])
        self.assertEqual(figure.data[0].name, "Uncertainty-based conflict zone")

    def test_expanded_polygon_controls_entry_and_exit_for_multiple_recordings(self) -> None:
        zone = {"shape": "rectangle", "polygon_xz_m": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}
        first = occupancy_interval(np.arange(5.0), np.array([-3, -1, 0, 1, 3]), np.zeros(5), zone)
        second = occupancy_interval(np.arange(5.0), np.array([3, 1, 0, -1, -3]), np.ones(5), zone)
        self.assertEqual(first, (1.0, 3.0))
        self.assertEqual(second, (1.0, 3.0))

    def test_existing_intersection_scenario_retains_geometric_method(self) -> None:
        zone = get_conflict_zone_for_scenario(3, self._scenario_traces(3))
        self.assertEqual(zone["conflict_zone_method"], "geometric_intersection")
        self.assertEqual(zone["shape"], "circle")

    def test_polygon_zone_membership(self) -> None:
        zone = {"shape": "corridor_intersection_polygon", "polygon_xz_m": [[0, 0], [2, 0], [2, 2], [0, 2]]}
        result = zone_membership(np.array([1.0, 3.0]), np.array([1.0, 1.0]), zone)
        self.assertEqual(result.tolist(), [True, False])

    def test_polygon_entry_and_never_entered_outcome(self) -> None:
        zone = {"shape": "corridor_intersection_polygon", "polygon_xz_m": [[0, 0], [2, 0], [2, 2], [0, 2]]}
        entered = occupancy_interval(np.arange(4.0), np.array([-1, 1, 1, 3]), np.ones(4), zone)
        missed = occupancy_interval(np.arange(3.0), np.array([3, 4, 5]), np.ones(3), zone)
        self.assertEqual(entered, (1.0, 2.0))
        self.assertTrue(all(math.isnan(value) for value in missed))

    def test_segment_intersection(self) -> None:
        point = segment_intersection(
            np.array([0.0, 0.0]), np.array([2.0, 0.0]),
            np.array([1.0, -1.0]), np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(point, [1, 0])

    def test_zone_entry_exit_and_order(self) -> None:
        zone = {"center_x_m": 0.0, "center_z_m": 0.0, "radius_m": 1.0}
        entry, exit_time = occupancy_interval(
            np.arange(5.0), np.array([2, 1, 0, 1, 2]), np.zeros(5), zone
        )
        self.assertEqual((entry, exit_time), (1.0, 3.0))
        self.assertEqual(entry_order(1.0, 2.0), ("pedestrian", 1.0))

    def test_occupancy_overlap(self) -> None:
        self.assertEqual(occupancy_overlap(1, 4, 3, 5), (True, 1.0))
        self.assertEqual(occupancy_overlap(1, 2, 3, 4), (False, 0.0))

    def test_distance_and_closing_rate(self) -> None:
        frame = pd.DataFrame({
            "scenario_time": [0.0, 1.0, 2.0],
            "ped_x": [0.0, 0.0, 0.0], "ped_z": [0.0, 0.0, 0.0],
            "veh_x": [3.0, 2.0, 1.0], "veh_z": [4.0, 0.0, 0.0],
        })
        result = synchronized_distance(frame)
        np.testing.assert_allclose(result.distance_m, [5, 2, 1])
        self.assertTrue((result.closing_rate_mps > 0).all())
        self.assertTrue(result.ttc_seconds.notna().all())

    def test_ttc_missing_during_separation(self) -> None:
        frame = pd.DataFrame({
            "scenario_time": [0.0, 1.0, 2.0],
            "ped_x": [0.0, 0.0, 0.0], "ped_z": [0.0, 0.0, 0.0],
            "veh_x": [1.0, 2.0, 3.0], "veh_z": [0.0, 0.0, 0.0],
        })
        result = synchronized_distance(frame)
        self.assertTrue(result.ttc_seconds.isna().all())
        self.assertTrue((result.approach_state == "separating").all())


if __name__ == "__main__":
    unittest.main()
