import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.analysis.behavior_clustering import apply_rigid_transform
from src.analysis.scenario_trajectory_clustering import (
    _labels, arc_length_resample, fixed_window, interaction_anchor,
    validate_pooled_features, who_went_first_group,
)


def test_scenario_relative_coordinate_normalization():
    parameters = {"translation_x_m": 10, "translation_z_m": 20, "rotation_degrees": -90,
                  "reflection_across_pedestrian_axis": False}
    x, z = apply_rigid_transform(pd.Series([10, 10]), pd.Series([20, 22]), parameters)
    np.testing.assert_allclose(x, [0, 2], atol=1e-10)
    np.testing.assert_allclose(z, [0, 0], atol=1e-10)


def test_arc_length_resampling_is_independent_of_original_timing_density():
    sparse = arc_length_resample(np.array([0, 1, 4]), np.zeros(3), 5)
    dense = arc_length_resample(np.array([0, .1, .2, 1, 4]), np.zeros(5), 5)
    np.testing.assert_allclose(sparse, dense)
    np.testing.assert_allclose(sparse[:, 0], [0, 1, 2, 3, 4])


def test_event_alignment_and_fixed_window_do_not_extrapolate():
    grid, values = fixed_window(np.array([9, 10, 11]), np.array([1, 2, 3]), 10, (-2, 2), 1)
    np.testing.assert_allclose(grid, [-2, -1, 0, 1, 2])
    assert math.isnan(values[0]) and math.isnan(values[-1])
    np.testing.assert_allclose(values[1:4], [1, 2, 3])


def test_entry_difference_sign_and_uncertainty_anchor():
    event = pd.Series({"pedestrian_entry_time": 4.0, "counterfactual_vehicle_arrival_time": 6.0})
    anchor, method, fallback = interaction_anchor(event, {"conflict_zone_method": "turn_uncertainty"})
    assert anchor == 4.0 and fallback and "turn_uncertainty" in method
    assert event.pedestrian_entry_time - event.counterfactual_vehicle_arrival_time == -2.0


def test_missing_conflict_zone_entry_has_no_closest_approach_fallback():
    anchor, method, _ = interaction_anchor(pd.Series(dtype=float), {"conflict_zone_method": "geometric_intersection"})
    assert math.isnan(anchor)
    assert method == "missing_zone_entry_no_fallback"


def test_missing_and_ambiguous_ground_truth_are_excluded_from_two_groups():
    assert who_went_first_group("Pedestrian") == "Pedestrian-first"
    assert who_went_first_group("Driver") == "Vehicle-first"
    assert who_went_first_group(None) is None
    assert who_went_first_group("Both") is None


def test_fixed_seed_clustering_is_deterministic():
    matrix = np.array([[0, 0], [.1, 0], [5, 5], [5.1, 5]])
    first = _labels(matrix, 2, False)
    second = _labels(matrix, 2, False)
    assert adjusted_rand_score(first, second) == 1


def test_raw_coordinate_pooling_is_prevented():
    validate_pooled_features(["minimum_distance_m", "ped_speed_change_mps"])
    with pytest.raises(ValueError, match="cannot be pooled"):
        validate_pooled_features(["ped_x", "minimum_distance_m"])
