from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.analysis.behavior_clustering import (
    AnalysisConfig,
    _cluster_labels,
    build_curve_matrix,
    build_matching_outputs,
    choose_interaction_event,
    extract_trial_features,
    normalize_trial_key,
    robust_scale_within_group,
    select_cluster_solution,
)


def test_trial_key_matching_is_exact_and_duplicate_motion_is_reported():
    assert normalize_trial_key(" PedNYC01 ", "Scenario 03") == ("PedNYC1", 3)
    assert normalize_trial_key("folder_PedNYC1_extra", "3") is None
    traces = pd.DataFrame({
        "recording_identifier": ["r1", "r2"],
        "participant_id": ["PedNYC1", "PedNYC1"], "scenario": [3, 3],
    })
    manifest = pd.DataFrame({
        "recording_identifier": ["r1", "r2"], "source_csv_path": ["a.csv", "b.csv"],
    })
    availability = pd.DataFrame({
        "pednyc_number": [1], "scenario_number": [3], "availability_status": ["available"],
        "processing_status": ["success"], "failure_exclusion_reason": [""],
    })
    ground_truth = pd.DataFrame({
        "trial_key": ["PedNYC1_scenario3"], "ground_truth_row_number": [3],
    })
    inventory, report = build_matching_outputs(traces, manifest, availability, ground_truth)
    assert report.match_status.eq("duplicate_motion_match").all()
    assert report.ground_truth_rows_for_trial_key.eq(1).all()
    assert inventory.motion_recording_count.iloc[0] == 2


def test_interaction_event_priority_does_not_use_unsupported_zone():
    one = pd.DataFrame({
        "scenario_time": [0.0, 1.0], "ped_x": [0, 1], "ped_z": [0, 0],
        "veh_x": [1, 0], "veh_z": [0, 0],
    })
    event = pd.Series({
        "pedestrian_entry_time": 1.0, "vehicle_entry_time": 3.0,
        "minimum_distance_time": 2.5,
    })
    supported = choose_interaction_event(one, event, {
        "geometrically_supported": True, "center_x_m": 0, "center_z_m": 0,
    })
    unsupported = choose_interaction_event(one, event, {
        "geometrically_supported": False, "center_x_m": 0, "center_z_m": 0,
    })
    assert supported["main_interaction_time"] == 2.0
    assert supported["main_interaction_method"] == "supported_conflict_zone_entry_midpoint"
    assert unsupported["main_interaction_time"] == 2.5
    assert "minimum_synchronized" in unsupported["main_interaction_method"]


def test_feature_speed_change_and_arrival_sign_convention():
    time = np.arange(11.0)
    traces = pd.DataFrame({
        "participant_id": ["PedNYC1"] * 11,
        "recording_identifier": ["PedNYC1_scenario3"] * 11,
        "scenario": [3] * 11, "scenario_time": time, "aligned_time": time - 1,
        "ped_x": time, "ped_z": np.zeros(11), "veh_x": np.zeros(11), "veh_z": time,
        "ped_speed": [0, .5, 1, 1, 1, 1, .5, .5, .5, .5, .5],
        "veh_speed": [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        "ped_acceleration": [0] * 11, "veh_acceleration": [0] * 11,
    })
    events = pd.DataFrame({
        "recording_identifier": ["PedNYC1_scenario3"],
        "pedestrian_entry_time": [4.0], "pedestrian_exit_time": [5.0],
        "vehicle_entry_time": [6.0], "vehicle_exit_time": [7.0],
        "minimum_distance_m": [1.0], "minimum_distance_time": [5.0],
        "ttc_at_minimum_distance_seconds": [np.nan], "vehicle_braked_before_entry": [False],
    })
    zones = {3: {"center_x_m": 0.0, "center_z_m": 0.0, "geometrically_supported": True}}
    features, event_output, _, _, transforms = extract_trial_features(
        traces, events, zones, AnalysisConfig()
    )
    assert features.arrival_time_difference_s.iloc[0] == -2.0
    assert features.measured_entity_entering_first.iloc[0] == "pedestrian"
    assert features.ped_speed_change_mps.iloc[0] < 0
    assert features.ped_longest_stationary_period_s.iloc[0] == 0
    assert transforms.maximum_step_distance_preservation_error_m.iloc[0] < 1e-10
    assert event_output.main_interaction_time.iloc[0] == 5.0


def test_scenario_adjustment_uses_each_scenario_median_and_iqr():
    frame = pd.DataFrame({
        "scenario": [3, 3, 3, 7, 7, 7],
        "feature": [0.0, 1.0, 2.0, 100.0, 102.0, 104.0],
    })
    scaled, audit = robust_scale_within_group(frame, ["feature"], "scenario")
    np.testing.assert_allclose(scaled.feature, [-1, 0, 1, -1, 0, 1])
    assert audit.groupby("scenario").imputed_count.sum().eq(0).all()


def test_curve_alignment_preserves_channels_and_missing_coverage():
    traces = pd.DataFrame({
        "recording_identifier": ["r1"] * 5, "participant_id": ["PedNYC1"] * 5,
        "scenario": [3] * 5, "scenario_time": [0, 1, 2, 3, 4],
        "ped_speed": [0, 1, 2, 1, 0], "veh_speed": [4, 3, 2, 1, 0],
    })
    events = pd.DataFrame({"recording_identifier": ["r1"], "main_interaction_time": [2.0]})
    config = AnalysisConfig(curve_window=(-3, 3), curve_step=1)
    long, ids, matrix, columns = build_curve_matrix(traces, events, config)
    assert ids.recording_identifier.tolist() == ["r1"]
    assert np.isnan(matrix[0, 0]) and np.isnan(matrix[0, 6])
    assert matrix[0, 3] == 2.0
    assert matrix[0, 10] == 2.0
    assert set(long.actor) == {"pedestrian", "vehicle"}
    assert len(columns) == 14


def test_synthetic_clustering_recovers_deliberately_separated_behaviors():
    rng = np.random.default_rng(42)
    first = rng.normal(loc=(-3, -3), scale=.15, size=(25, 2))
    second = rng.normal(loc=(3, 3), scale=.15, size=(25, 2))
    matrix = np.vstack([first, second])
    truth = np.repeat([0, 1], 25)
    labels = _cluster_labels("ward", 2, matrix, seed=42)
    assert adjusted_rand_score(truth, labels) > .99


def test_tiny_outlier_cluster_is_not_selected_as_defensible():
    metrics = pd.DataFrame([{
        "error": "", "method": "ward", "clusters": 2,
        "silhouette": .85, "davies_bouldin": .2, "calinski_harabasz": 200,
        "bootstrap_ari_mean": .95, "bootstrap_ari_p10": .90,
        "minimum_cluster_size": 4, "cluster_sizes": "151;4",
    }])
    solution = select_cluster_solution(metrics, AnalysisConfig())
    assert solution["status"] == "no_stable_separation"
    assert solution["required_minimum_cluster_size"] == 16
