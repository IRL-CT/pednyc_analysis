import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis.entry_distance_manual_review import (
    analyze,
    first_circle_entry_time,
    interpolate_position,
    measure_recording,
    rank_within_scenarios,
)


ZONE = {
    "center_x_m": 0.0,
    "center_z_m": 0.0,
    "radius_m": 1.0,
    "geometrically_supported": True,
}


def recording(
    ped_x, ped_z, car_x, car_z, *, scenario=3, participant="PedNYC1", name="r1",
    time=None,
):
    count = len(ped_x)
    return pd.DataFrame({
        "participant_id": [participant] * count,
        "scenario": [scenario] * count,
        "recording_identifier": [name] * count,
        "scenario_time": np.arange(count, dtype=float) if time is None else time,
        "ped_x": ped_x,
        "ped_z": ped_z,
        "veh_x": car_x,
        "veh_z": car_z,
    })


def result_frame(scenario, values):
    return pd.DataFrame({
        "scenario_id": [scenario] * len(values),
        "recording_identifier": [f"r{index + 1}" for index in range(len(values))],
        "distance_at_first_conflict_zone_entry_m": values,
    })


class ConflictZoneEntryTests(unittest.TestCase):
    def test_entry_detection_interpolates_segment_circle_crossing(self):
        entry = first_circle_entry_time(
            np.array([0.0, 2.0]), np.array([-3.0, 1.0]), np.zeros(2), ZONE
        )
        self.assertAlmostEqual(entry, 1.0)

    def test_entry_detection_finds_crossing_between_two_outside_samples(self):
        entry = first_circle_entry_time(
            np.array([0.0, 2.0]), np.array([-2.0, 2.0]), np.zeros(2), ZONE
        )
        self.assertAlmostEqual(entry, 0.5)

    def test_interpolation_uses_requested_timestamp(self):
        position = interpolate_position(
            np.array([0.0, 2.0]), np.array([0.0, 4.0]), np.array([2.0, 6.0]), 0.5
        )
        np.testing.assert_allclose(position, [1.0, 3.0])

    def test_first_entrant_identification(self):
        one = recording(
            [-2.0, 0.0], [0.0, 0.0],
            [-3.0, 0.0], [0.0, 0.0], time=[0.0, 2.0],
        )
        result = measure_recording(one, ZONE)
        self.assertEqual(result["first_entrant"], "pedestrian")
        self.assertAlmostEqual(result["pedestrian_conflict_zone_entry_time"], 1.0)
        self.assertAlmostEqual(result["car_conflict_zone_entry_time"], 4.0 / 3.0)

    def test_distance_uses_both_positions_at_same_interpolated_time(self):
        one = recording(
            [0.0, 0.0], [3.0, 3.0],
            [-3.0, 1.0], [0.0, 0.0], time=[0.0, 2.0],
        )
        result = measure_recording(one, ZONE)
        self.assertEqual(result["entry_status"], "valid_single_actor_entry")
        self.assertAlmostEqual(result["first_conflict_zone_entry_time"], 1.0)
        self.assertAlmostEqual(result["distance_at_first_conflict_zone_entry_m"], np.sqrt(10.0))

    def test_single_actor_entry_is_measurable(self):
        one = recording([4, 4], [4, 4], [-2, 0], [0, 0])
        result = measure_recording(one, ZONE)
        self.assertEqual(result["entry_status"], "valid_single_actor_entry")
        self.assertEqual(result["first_entrant"], "car")
        self.assertTrue(np.isfinite(result["distance_at_first_conflict_zone_entry_m"]))

    def test_neither_actor_entry(self):
        one = recording([4, 4], [4, 4], [-4, -4], [-4, -4])
        result = measure_recording(one, ZONE)
        self.assertEqual(result["entry_status"], "no_actor_entry")
        self.assertTrue(np.isnan(result["distance_at_first_conflict_zone_entry_m"]))

    def test_missing_position_data_is_invalid(self):
        one = recording([np.nan, np.nan], [np.nan, np.nan], [-2, 0], [0, 0])
        result = measure_recording(one, ZONE)
        self.assertEqual(result["entry_status"], "invalid_data")
        self.assertIn("pedestrian position", result["data_quality_status"])


class ScenarioRankingTests(unittest.TestCase):
    def test_scenario_grouping_and_arithmetic_mean(self):
        source = pd.concat([
            result_frame(3, [1.0, 3.0]),
            result_frame(7, [10.0, 20.0]),
        ], ignore_index=True)
        source.loc[2:, "recording_identifier"] = ["s7r1", "s7r2"]
        ranked = rank_within_scenarios(source, top_n=1)
        np.testing.assert_allclose(ranked.scenario_mean_entry_distance_m, [2, 2, 15, 15])

    def test_signed_absolute_difference_and_descending_rank(self):
        ranked = rank_within_scenarios(result_frame(3, [1.0, 2.0, 6.0]), top_n=1)
        mean = 3.0
        np.testing.assert_allclose(ranked.scenario_mean_entry_distance_m, mean)
        np.testing.assert_allclose(ranked.signed_difference_from_mean_m, [-2, -1, 3])
        np.testing.assert_allclose(ranked.absolute_difference_from_mean_m, [2, 1, 3])
        self.assertEqual(ranked.within_scenario_deviation_rank.tolist(), [2, 3, 1])

    def test_top_three_flags(self):
        ranked = rank_within_scenarios(result_frame(3, [0, 1, 2, 3, 10]), top_n=3)
        self.assertEqual(int(ranked.manual_review_candidate.sum()), 3)
        self.assertTrue((ranked.loc[ranked.manual_review_candidate, "within_scenario_deviation_rank"] <= 3).all())

    def test_tie_at_ranking_boundary_is_deterministic_and_exactly_three(self):
        ranked = rank_within_scenarios(result_frame(3, [0, 0, 0, 0, 10]), top_n=3)
        flagged = ranked.loc[ranked.manual_review_candidate, "recording_identifier"].tolist()
        self.assertEqual(flagged, ["r1", "r2", "r5"])
        self.assertEqual(int(ranked.manual_review_candidate.sum()), 3)
        ranks = ranked.set_index("recording_identifier").within_scenario_deviation_rank
        self.assertEqual(int(ranks["r5"]), 1)
        self.assertEqual(int(ranks["r1"]), 2)
        self.assertEqual(int(ranks["r2"]), 3)


class UnsupportedGeometryTests(unittest.TestCase):
    def test_unsupported_scenario_is_retained_without_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trajectory = root / "trajectory"
            conflict = root / "conflict"
            scenario_dir = trajectory / "scenario_03"
            zone_dir = conflict / "scenario_03"
            scenario_dir.mkdir(parents=True)
            zone_dir.mkdir(parents=True)
            one = recording([-2, 0], [0, 0], [-3, 0], [0, 0])
            one.to_csv(scenario_dir / "raw_trajectory_samples.csv", index=False)
            pd.DataFrame({
                "recording_identifier": ["r1"],
                "participant_id": ["PedNYC1"],
                "scenario": [3],
                "source_path": ["source.csv"],
            }).to_csv(trajectory / "coordinate_diagnostics.csv", index=False)
            unsupported = dict(ZONE, geometrically_supported=False)
            (zone_dir / "conflict_zone_definition.json").write_text(
                json.dumps(unsupported), encoding="utf-8"
            )
            results, summary = analyze(
                trajectory, conflict, root / "output", [3], top_n=3
            )
            self.assertEqual(results.entry_status.iloc[0], "no_supported_conflict_zone")
            self.assertTrue(np.isnan(results.distance_at_first_conflict_zone_entry_m.iloc[0]))
            self.assertEqual(summary.conflict_zone_availability.iloc[0], "no_supported_conflict_zone")


if __name__ == "__main__":
    unittest.main()
