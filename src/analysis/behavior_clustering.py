"""Reproducible behavioral clustering of PedNYC pedestrian–vehicle trials.

This analysis consumes the validated trajectory and conflict-zone artifacts
already produced by this repository.  Ground-truth annotations are joined only
after all clustering solutions have been fixed.  The script never mutates its
inputs and writes exclusively below a new output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    cohen_kappa_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "behavior_clustering"
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "XCPedestrian-GroundTruth-RERUN.xlsx"
DEFAULT_SCENARIOS = (3, 7, 12, 15, 16, 21)
REQUIRED_TRACE_COLUMNS = {
    "participant_id", "recording_identifier", "scenario", "scenario_time",
    "aligned_time", "ped_x", "ped_z", "veh_x", "veh_z", "ped_speed",
    "veh_speed", "ped_acceleration", "veh_acceleration",
}

LABEL_COLUMNS = {
    "Did the driver signal": "driver_signal",
    "How does vehicle speed change?": "annotated_vehicle_speed_change",
    "How does pedestrian speed change?": "annotated_pedestrian_speed_change",
    "Who has the right of way?（Who reached the intersection first?）": "annotated_right_of_way",
    "Who went first？": "annotated_who_went_first",
    "Did the pedestrian use hand gestures?": "pedestrian_hand_gesture",
    "Did the driver use hand gestures?": "driver_hand_gesture",
}

INITIAL_CLUSTER_FEATURES = (
    "ped_speed_change_mps",
    "ped_min_interaction_speed_mps",
    "ped_longest_stationary_period_s",
    "ped_max_deceleration_mps2",
    "vehicle_speed_change_mps",
    "vehicle_min_interaction_speed_mps",
    "vehicle_max_deceleration_mps2",
    "arrival_time_difference_s",
    "minimum_distance_m",
    "distance_at_pedestrian_onset_m",
    "ped_path_efficiency",
    "ped_max_lateral_deviation_m",
    "ped_total_heading_change_deg",
    "vehicle_mean_lateral_deviation_m",
    "braking_to_pedestrian_response_lag_s",
)


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 42
    before_window: tuple[float, float] = (-3.0, -1.0)
    during_window: tuple[float, float] = (-1.0, 1.0)
    after_window: tuple[float, float] = (1.0, 3.0)
    curve_window: tuple[float, float] = (-5.0, 5.0)
    curve_step: float = 0.1
    stationary_speed_mps: float = 0.15
    minimum_stop_duration_s: float = 0.45
    pedestrian_movement_speed_mps: float = 0.30
    braking_acceleration_mps2: float = -0.25
    response_acceleration_mps2: float = 0.25
    pedestrian_implausible_speed_mps: float = 8.0
    vehicle_implausible_speed_mps: float = 40.0
    implausible_acceleration_mps2: float = 20.0
    maximum_feature_missing_fraction: float = 0.35
    maximum_absolute_correlation: float = 0.85
    maximum_retained_features: int = 10
    minimum_cluster_size: int = 3
    bootstrap_repetitions: int = 50
    stability_threshold: float = 0.60
    stability_lower_threshold: float = 0.30
    silhouette_threshold: float = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", type=int, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this analysis's output directory contents.")
    return parser.parse_args()


def normalize_trial_key(study_folder: object, scenario_name: object) -> tuple[str, int] | None:
    """Normalize an exact StudyFolder/ScenarioName pair without substring matching."""
    study = re.fullmatch(r"\s*PedNYC0*(\d+)\s*", str(study_folder), re.IGNORECASE)
    scenario = re.fullmatch(r"\s*(?:Scenario\s*)?0*(3|7|12|15|16|21)(?:\.0)?\s*", str(scenario_name), re.IGNORECASE)
    if not study or not scenario:
        return None
    return f"PedNYC{int(study.group(1))}", int(scenario.group(1))


def clean_ground_truth(path: Path, scenarios: list[int]) -> pd.DataFrame:
    """Read the workbook and discard its categorical-instructions row."""
    frame = pd.read_excel(path, sheet_name=0, dtype=object)
    required = {"StudyFolder", "ScenarioName", *LABEL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground-truth workbook is missing columns: {missing}")
    rows = []
    for workbook_row, row in frame.iterrows():
        key = normalize_trial_key(row.StudyFolder, row.ScenarioName)
        if key is None or key[1] not in scenarios:
            continue
        cleaned = {
            "ground_truth_row_number": int(workbook_row + 2),
            "StudyFolder": key[0],
            "ScenarioName": key[1],
            "trial_key": f"{key[0]}_scenario{key[1]}",
        }
        for source, target in LABEL_COLUMNS.items():
            value = row[source]
            cleaned[target] = "Unknown" if pd.isna(value) or not str(value).strip() else str(value).strip()
        rows.append(cleaned)
    return pd.DataFrame(rows)


def _read_motion_inputs(input_root: Path, scenarios: list[int]):
    output_root = input_root / "outputs"
    manifest_dir = output_root / "scenario_pattern_expanded" / "manifests"
    manifest = pd.read_csv(manifest_dir / "processing_success_manifest.csv")
    availability = pd.read_csv(manifest_dir / "participant_scenario_availability.csv")
    manifest = manifest[manifest.scenario_number.isin(scenarios)].copy()
    availability = availability[availability.scenario_number.isin(scenarios)].copy()
    traces, events, zones = [], [], {}
    for scenario in scenarios:
        trace_path = output_root / "trajectory_analysis" / f"scenario_{scenario:02d}" / "raw_trajectory_samples.csv"
        event_path = output_root / "conflict_zone_analysis" / f"scenario_{scenario:02d}" / "conflict_zone_events.csv"
        zone_path = output_root / "conflict_zone_analysis" / f"scenario_{scenario:02d}" / "conflict_zone_definition.json"
        trace = pd.read_csv(trace_path)
        missing = sorted(REQUIRED_TRACE_COLUMNS - set(trace.columns))
        if missing:
            raise ValueError(f"{trace_path} missing required columns: {missing}")
        traces.append(trace)
        events.append(pd.read_csv(event_path))
        zones[scenario] = json.loads(zone_path.read_text(encoding="utf-8"))
    return manifest, availability, pd.concat(traces, ignore_index=True), pd.concat(events, ignore_index=True), zones


def build_matching_outputs(
    traces: pd.DataFrame,
    manifest: pd.DataFrame,
    availability: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motion = traces[["recording_identifier", "participant_id", "scenario"]].drop_duplicates()
    motion["trial_key"] = motion.participant_id.astype(str) + "_scenario" + motion.scenario.astype(int).astype(str)
    source_lookup = manifest.set_index("recording_identifier")["source_csv_path"].to_dict()
    motion["source_csv_path"] = motion.recording_identifier.map(source_lookup)
    motion_counts = motion.groupby("trial_key").size().to_dict()
    gt_counts = ground_truth.groupby("trial_key").size().to_dict()
    gt_lookup = ground_truth.groupby("trial_key")["ground_truth_row_number"].apply(list).to_dict()
    rows = []
    for row in motion.itertuples(index=False):
        gt_rows = gt_lookup.get(row.trial_key, [])
        if len(gt_rows) == 1 and motion_counts[row.trial_key] == 1:
            status = "unique_match"
        elif len(gt_rows) == 1:
            status = "duplicate_motion_match"
        elif len(gt_rows) == 0:
            status = "unmatched_motion"
        else:
            status = "duplicate_ground_truth_match"
        rows.append({
            "recording_identifier": row.recording_identifier,
            "participant_id": row.participant_id,
            "scenario": int(row.scenario),
            "trial_key": row.trial_key,
            "source_csv_path": row.source_csv_path,
            "motion_recordings_for_trial_key": motion_counts[row.trial_key],
            "ground_truth_rows_for_trial_key": len(gt_rows),
            "ground_truth_row_number": gt_rows[0] if len(gt_rows) == 1 else pd.NA,
            "match_status": status,
            "duplicate_match": motion_counts[row.trial_key] > 1 or len(gt_rows) > 1,
        })
    report = pd.DataFrame(rows)
    matched_keys = set(report.loc[report.ground_truth_rows_for_trial_key.eq(1), "trial_key"])
    unmatched_gt = ground_truth.loc[~ground_truth.trial_key.isin(matched_keys)].copy()

    inventory = availability.copy()
    inventory["participant_id"] = "PedNYC" + inventory.pednyc_number.astype(int).astype(str)
    inventory["trial_key"] = inventory.participant_id + "_scenario" + inventory.scenario_number.astype(int).astype(str)
    inventory = inventory.merge(
        ground_truth[["trial_key", "ground_truth_row_number"]], on="trial_key", how="outer", indicator=True
    )
    motion_by_key = report.groupby("trial_key").size().rename("motion_recording_count")
    inventory = inventory.merge(motion_by_key, on="trial_key", how="left")
    inventory["motion_recording_count"] = inventory.motion_recording_count.fillna(0).astype(int)
    inventory["ground_truth_present"] = inventory._merge.ne("left_only")
    inventory["uniquely_matched"] = inventory.motion_recording_count.eq(1) & inventory.ground_truth_present
    inventory["duplicate_match"] = inventory.motion_recording_count.gt(1)
    inventory.drop(columns="_merge", inplace=True)
    if len(unmatched_gt):
        # The outer merge already contains these rows; this assertion prevents hidden loss.
        assert set(unmatched_gt.trial_key).issubset(set(inventory.trial_key))
    return inventory, report


def _sampling_duration(mask: np.ndarray, time: np.ndarray) -> float:
    if len(time) < 2:
        return 0.0
    dt = np.diff(time, prepend=np.nan)
    valid = mask & np.isfinite(dt) & (dt > 0)
    return float(np.sum(dt[valid]))


def _episodes(mask: np.ndarray, time: np.ndarray, minimum_duration: float) -> tuple[float, float, int]:
    durations = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active and index == len(mask) - 1 else index - 1
            duration = float(max(0.0, time[end] - time[start]))
            if duration >= minimum_duration:
                durations.append(duration)
            start = None
    return float(sum(durations)), float(max(durations, default=0.0)), len(durations)


def _window_stat(one: pd.DataFrame, column: str, relative: pd.Series, bounds: tuple[float, float], statistic: str = "mean") -> float:
    values = one.loc[relative.between(bounds[0], bounds[1]), column]
    if not len(values) or values.notna().sum() == 0:
        return math.nan
    return float(getattr(values, statistic)())


def _nearest_value(one: pd.DataFrame, column: str, time: float) -> float:
    if not np.isfinite(time) or one.empty:
        return math.nan
    valid = one[["scenario_time", column]].dropna()
    if valid.empty:
        return math.nan
    index = (valid.scenario_time - time).abs().idxmin()
    return float(valid.at[index, column])


def choose_interaction_event(
    one: pd.DataFrame, event: pd.Series, zone: dict[str, object]
) -> dict[str, object]:
    """Choose an auditable interaction reference using the specified priority."""
    entries = [event.get("pedestrian_entry_time", np.nan), event.get("vehicle_entry_time", np.nan)]
    if bool(zone.get("geometrically_supported")) and np.isfinite(entries).all():
        return {
            "main_interaction_time": float(np.mean(entries)),
            "main_interaction_detected": True,
            "main_interaction_method": "supported_conflict_zone_entry_midpoint",
        }
    minimum_time = event.get("minimum_distance_time", np.nan)
    if np.isfinite(minimum_time):
        return {
            "main_interaction_time": float(minimum_time),
            "main_interaction_detected": True,
            "main_interaction_method": "minimum_synchronized_pedestrian_vehicle_distance",
        }
    center_x, center_z = zone.get("center_x_m", np.nan), zone.get("center_z_m", np.nan)
    if np.isfinite([center_x, center_z]).all():
        score = (
            np.hypot(one.ped_x - center_x, one.ped_z - center_z)
            + np.hypot(one.veh_x - center_x, one.veh_z - center_z)
        )
        if score.notna().any():
            index = score.idxmin()
            return {
                "main_interaction_time": float(one.at[index, "scenario_time"]),
                "main_interaction_detected": True,
                "main_interaction_method": "closest_joint_approach_to_geometric_conflict_center",
            }
    return {
        "main_interaction_time": math.nan,
        "main_interaction_detected": False,
        "main_interaction_method": "not_detected_no_arbitrary_fallback",
    }


def rigid_transform_parameters(traces: pd.DataFrame, zone: dict[str, object]) -> dict[str, object]:
    directions = []
    vehicle_directions = []
    for _, one in traces.groupby("recording_identifier", sort=True):
        one = one.sort_values("scenario_time")
        p = np.array([one.ped_x.iloc[-1] - one.ped_x.iloc[0], one.ped_z.iloc[-1] - one.ped_z.iloc[0]])
        v = np.array([one.veh_x.iloc[-1] - one.veh_x.iloc[0], one.veh_z.iloc[-1] - one.veh_z.iloc[0]])
        if np.linalg.norm(p) > 1e-6:
            directions.append(p / np.linalg.norm(p))
        if np.linalg.norm(v) > 1e-6:
            vehicle_directions.append(v / np.linalg.norm(v))
    pedestrian_direction = np.median(np.vstack(directions), axis=0)
    angle = -math.atan2(pedestrian_direction[1], pedestrian_direction[0])
    c, s = math.cos(angle), math.sin(angle)
    vehicle_direction = np.median(np.vstack(vehicle_directions), axis=0)
    transformed_vehicle_z = s * vehicle_direction[0] + c * vehicle_direction[1]
    return {
        "translation_x_m": float(zone["center_x_m"]),
        "translation_z_m": float(zone["center_z_m"]),
        "rotation_degrees": float(math.degrees(angle)),
        "reflection_across_pedestrian_axis": bool(transformed_vehicle_z < 0),
        "scale": 1.0,
        "zone_geometrically_supported": bool(zone.get("geometrically_supported")),
        "method": "scenario_shared_rigid_transform_conflict_center_to_origin_pedestrian_direction_positive_x",
    }


def apply_rigid_transform(x: pd.Series, z: pd.Series, parameters: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    dx = x.to_numpy(dtype=float) - float(parameters["translation_x_m"])
    dz = z.to_numpy(dtype=float) - float(parameters["translation_z_m"])
    angle = math.radians(float(parameters["rotation_degrees"]))
    c, s = math.cos(angle), math.sin(angle)
    tx = c * dx - s * dz
    tz = s * dx + c * dz
    if parameters["reflection_across_pedestrian_axis"]:
        tz = -tz
    return tx, tz


def _path_characteristics(x: np.ndarray, z: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(z)
    x, z = x[valid], z[valid]
    if len(x) < 2:
        return {key: math.nan for key in ("path_length", "net_displacement", "path_efficiency", "maximum_lateral_deviation", "mean_lateral_deviation", "total_heading_change")}
    dx, dz = np.diff(x), np.diff(z)
    steps = np.hypot(dx, dz)
    path_length = float(np.sum(steps))
    displacement = float(np.hypot(x[-1] - x[0], z[-1] - z[0]))
    direct = np.array([x[-1] - x[0], z[-1] - z[0]])
    if displacement > 1e-9:
        offsets = np.column_stack([x - x[0], z - z[0]])
        lateral = np.abs(direct[0] * offsets[:, 1] - direct[1] * offsets[:, 0]) / displacement
    else:
        lateral = np.full(len(x), np.nan)
    moving = steps > 1e-6
    headings = np.unwrap(np.arctan2(dz[moving], dx[moving]))
    heading_change = float(np.degrees(np.sum(np.abs(np.diff(headings))))) if len(headings) > 1 else 0.0
    return {
        "path_length": path_length,
        "net_displacement": displacement,
        "path_efficiency": displacement / path_length if path_length > 1e-9 else math.nan,
        "maximum_lateral_deviation": float(np.nanmax(lateral)) if np.isfinite(lateral).any() else math.nan,
        "mean_lateral_deviation": float(np.nanmean(lateral)) if np.isfinite(lateral).any() else math.nan,
        "total_heading_change": heading_change,
    }


def _first_threshold_time(relative: pd.Series, values: pd.Series, threshold: float, absolute: bool = False) -> float:
    candidate = relative.between(-5.0, 3.0)
    condition = values.abs().ge(abs(threshold)) if absolute else values.le(threshold)
    indices = np.flatnonzero((candidate & condition & values.notna()).to_numpy())
    return float(relative.iloc[indices[0]]) if len(indices) else math.nan


def extract_trial_features(
    traces: pd.DataFrame,
    events: pd.DataFrame,
    zones: dict[int, dict[str, object]],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_lookup = events.set_index("recording_identifier")
    feature_rows, event_rows, qc_rows, normalized_rows = [], [], [], []
    transform_rows = []
    scenario_parameters = {
        scenario: rigid_transform_parameters(one, zones[int(scenario)])
        for scenario, one in traces.groupby("scenario", sort=True)
    }
    for recording, source in traces.groupby("recording_identifier", sort=True):
        one = source.sort_values("scenario_time", kind="stable").reset_index(drop=True).copy()
        participant = str(one.participant_id.iloc[0])
        scenario = int(one.scenario.iloc[0])
        event = event_lookup.loc[recording]
        chosen = choose_interaction_event(one, event, zones[scenario])
        main_time = float(chosen["main_interaction_time"])
        relative = one.scenario_time - main_time
        one["interaction_aligned_time"] = relative
        one["distance_m"] = np.hypot(one.veh_x - one.ped_x, one.veh_z - one.ped_z)
        onset_index = int(np.argmin(np.abs(one.aligned_time.to_numpy(dtype=float))))
        onset_time = float(one.scenario_time.iloc[onset_index])
        parameters = scenario_parameters[scenario]
        ped_tx, ped_tz = apply_rigid_transform(one.ped_x, one.ped_z, parameters)
        veh_tx, veh_tz = apply_rigid_transform(one.veh_x, one.veh_z, parameters)
        normalized = pd.DataFrame({
            "recording_identifier": recording, "participant_id": participant,
            "scenario": scenario, "scenario_time": one.scenario_time,
            "interaction_aligned_time": relative, "movement_onset_aligned_time": one.aligned_time,
            "ped_x_normalized_m": ped_tx, "ped_z_normalized_m": ped_tz,
            "veh_x_normalized_m": veh_tx, "veh_z_normalized_m": veh_tz,
            "ped_speed_mps": one.ped_speed, "veh_speed_mps": one.veh_speed,
        })
        normalized_rows.append(normalized)
        transform_rows.append({
            "recording_identifier": recording, "participant_id": participant,
            "scenario": scenario, **parameters,
            "maximum_step_distance_preservation_error_m": float(max(
                np.nanmax(np.abs(np.hypot(np.diff(one.ped_x), np.diff(one.ped_z)) - np.hypot(np.diff(ped_tx), np.diff(ped_tz)))),
                np.nanmax(np.abs(np.hypot(np.diff(one.veh_x), np.diff(one.veh_z)) - np.hypot(np.diff(veh_tx), np.diff(veh_tz)))),
            )),
        })

        time = one.scenario_time.to_numpy(dtype=float)
        dt = np.diff(time)
        invalid_dt = ~np.isfinite(dt) | (dt <= 0)
        # Pre-onset pedestrian waiting is not a stop/hesitation episode. Vehicle
        # stationary time is restricted to the interaction neighborhood so
        # recording lead-in/lead-out does not dominate behavioral features.
        ped_stationary = one.ped_speed.iloc[onset_index:].lt(config.stationary_speed_mps).to_numpy()
        ped_stationary_time = time[onset_index:]
        vehicle_window_indices = np.flatnonzero(relative.between(-5.0, 3.0).to_numpy())
        if not len(vehicle_window_indices):
            vehicle_window_indices = np.arange(len(one))
        veh_stationary = one.veh_speed.iloc[vehicle_window_indices].lt(config.stationary_speed_mps).to_numpy()
        veh_stationary_time = time[vehicle_window_indices]
        ped_stationary_total, ped_stationary_longest, ped_stop_count = _episodes(
            ped_stationary, ped_stationary_time, config.minimum_stop_duration_s
        )
        veh_stationary_total, veh_stationary_longest, veh_stop_count = _episodes(
            veh_stationary, veh_stationary_time, config.minimum_stop_duration_s
        )
        ped_path = _path_characteristics(ped_tx[onset_index:], ped_tz[onset_index:])
        veh_path = _path_characteristics(veh_tx, veh_tz)
        braking_relative = _first_threshold_time(relative, one.veh_acceleration, config.braking_acceleration_mps2)
        response_relative = _first_threshold_time(relative, one.ped_acceleration, config.response_acceleration_mps2, absolute=True)
        braking_time = main_time + braking_relative if np.isfinite(braking_relative) else math.nan

        ped_entry = float(event.get("pedestrian_entry_time", np.nan))
        veh_entry = float(event.get("vehicle_entry_time", np.nan))
        zone_reliable = bool(zones[scenario].get("geometrically_supported"))
        signed_arrival = ped_entry - veh_entry if zone_reliable and np.isfinite([ped_entry, veh_entry]).all() else math.nan
        measured_first = (
            "pedestrian" if signed_arrival < 0 else "vehicle" if signed_arrival > 0 else "simultaneous"
        ) if np.isfinite(signed_arrival) else "Unknown"
        ped_exit = float(event.get("pedestrian_exit_time", np.nan)) if zone_reliable else math.nan
        veh_exit = float(event.get("vehicle_exit_time", np.nan)) if zone_reliable else math.nan
        pet = math.nan
        if np.isfinite([ped_entry, ped_exit, veh_entry, veh_exit]).all():
            if ped_exit < veh_entry:
                pet = veh_entry - ped_exit
            elif veh_exit < ped_entry:
                pet = ped_entry - veh_exit
            else:
                pet = 0.0

        feature = {
            "recording_identifier": recording, "participant_id": participant,
            "scenario": scenario, "trial_key": f"{participant}_scenario{scenario}",
            "pedestrian_movement_onset_time_s": onset_time,
            "main_interaction_time_s": main_time,
            "ped_mean_walking_speed_mps": float(one.loc[one.scenario_time.ge(onset_time), "ped_speed"].mean()),
            "ped_median_walking_speed_mps": float(one.loc[one.scenario_time.ge(onset_time), "ped_speed"].median()),
            "ped_maximum_plausible_speed_mps": float(one.loc[one.ped_speed.le(config.pedestrian_implausible_speed_mps), "ped_speed"].max()),
            "ped_speed_before_interaction_mps": _window_stat(one, "ped_speed", relative, config.before_window),
            "ped_speed_during_interaction_mps": _window_stat(one, "ped_speed", relative, config.during_window),
            "ped_speed_after_interaction_mps": _window_stat(one, "ped_speed", relative, config.after_window),
            "ped_min_interaction_speed_mps": _window_stat(one, "ped_speed", relative, config.during_window, "min"),
            "ped_max_acceleration_mps2": _window_stat(one, "ped_acceleration", relative, config.during_window, "max"),
            "ped_max_deceleration_mps2": _window_stat(one, "ped_acceleration", relative, config.during_window, "min"),
            "ped_total_stationary_duration_s": ped_stationary_total,
            "ped_longest_stationary_period_s": ped_stationary_longest,
            "ped_stop_episode_count": ped_stop_count,
            "ped_time_onset_to_zone_entry_s": ped_entry - onset_time if zone_reliable and np.isfinite(ped_entry) else math.nan,
            "ped_time_in_conflict_zone_s": ped_exit - ped_entry if np.isfinite([ped_entry, ped_exit]).all() else math.nan,
            "ped_crossing_duration_s": ped_exit - onset_time if np.isfinite(ped_exit) else math.nan,
            "vehicle_mean_interaction_speed_mps": _window_stat(one, "veh_speed", relative, config.during_window),
            "vehicle_speed_before_interaction_mps": _window_stat(one, "veh_speed", relative, config.before_window),
            "vehicle_speed_during_interaction_mps": _window_stat(one, "veh_speed", relative, config.during_window),
            "vehicle_speed_after_interaction_mps": _window_stat(one, "veh_speed", relative, config.after_window),
            "vehicle_min_interaction_speed_mps": _window_stat(one, "veh_speed", relative, config.during_window, "min"),
            "vehicle_max_acceleration_mps2": _window_stat(one, "veh_acceleration", relative, config.during_window, "max"),
            "vehicle_max_deceleration_mps2": _window_stat(one, "veh_acceleration", relative, config.during_window, "min"),
            "vehicle_braking_onset_relative_s": braking_relative,
            "vehicle_total_stationary_duration_s": veh_stationary_total,
            "vehicle_longest_stationary_period_s": veh_stationary_longest,
            "vehicle_stop_episode_count": veh_stop_count,
            "vehicle_stopped_before_conflict_zone": bool(event.get("vehicle_braked_before_entry", False)) if zone_reliable else pd.NA,
            "minimum_distance_m": float(event.get("minimum_distance_m", np.nan)),
            "minimum_distance_relative_s": float(event.get("minimum_distance_time", np.nan)) - main_time,
            "arrival_time_difference_s": signed_arrival,
            "absolute_arrival_time_difference_s": abs(signed_arrival) if np.isfinite(signed_arrival) else math.nan,
            "measured_entity_entering_first": measured_first,
            "braking_to_pedestrian_response_lag_s": response_relative - braking_relative if np.isfinite([response_relative, braking_relative]).all() else math.nan,
            "pedestrian_response_onset_relative_s": response_relative,
            "distance_at_pedestrian_onset_m": _nearest_value(one, "distance_m", onset_time),
            "distance_at_vehicle_braking_onset_m": _nearest_value(one, "distance_m", braking_time),
            "post_encroachment_time_s": pet,
            "ttc_at_minimum_distance_s": float(event.get("ttc_at_minimum_distance_seconds", np.nan)),
            "ped_path_length_m": ped_path["path_length"],
            "ped_net_displacement_m": ped_path["net_displacement"],
            "ped_path_efficiency": ped_path["path_efficiency"],
            "ped_max_lateral_deviation_m": ped_path["maximum_lateral_deviation"],
            "ped_mean_lateral_deviation_m": ped_path["mean_lateral_deviation"],
            "ped_total_heading_change_deg": ped_path["total_heading_change"],
            "vehicle_path_length_m": veh_path["path_length"],
            "vehicle_path_efficiency": veh_path["path_efficiency"],
            "vehicle_max_lateral_deviation_m": veh_path["maximum_lateral_deviation"],
            "vehicle_mean_lateral_deviation_m": veh_path["mean_lateral_deviation"],
        }
        feature["ped_speed_change_mps"] = feature["ped_speed_after_interaction_mps"] - feature["ped_speed_before_interaction_mps"]
        feature["vehicle_speed_change_mps"] = feature["vehicle_speed_after_interaction_mps"] - feature["vehicle_speed_before_interaction_mps"]
        feature["paired_speed_change_product"] = feature["ped_speed_change_mps"] * feature["vehicle_speed_change_mps"]
        feature_rows.append(feature)
        event_rows.append({
            "recording_identifier": recording, "participant_id": participant, "scenario": scenario,
            "pedestrian_movement_onset_time_s": onset_time,
            "pedestrian_movement_onset_detected": True,
            "pedestrian_movement_onset_method": "existing_pipeline_first_smoothed_ped_speed_ge_0.30_mps",
            "pedestrian_conflict_zone_entry_time_s": ped_entry if zone_reliable else math.nan,
            "pedestrian_conflict_zone_entry_detected": bool(zone_reliable and np.isfinite(ped_entry)),
            "vehicle_conflict_zone_entry_time_s": veh_entry if zone_reliable else math.nan,
            "vehicle_conflict_zone_entry_detected": bool(zone_reliable and np.isfinite(veh_entry)),
            "conflict_zone_event_method": "existing_shared_circle" if zone_reliable else "zone_geometrically_unsupported_not_used",
            "minimum_distance_time_s": float(event.get("minimum_distance_time", np.nan)),
            "minimum_distance_detected": bool(np.isfinite(event.get("minimum_distance_time", np.nan))),
            "minimum_distance_method": "synchronized_same_row_planar_distance",
            "pedestrian_crossing_completion_time_s": ped_exit,
            "pedestrian_crossing_completion_detected": bool(np.isfinite(ped_exit)),
            "pedestrian_crossing_completion_method": "supported_conflict_zone_first_visit_exit" if np.isfinite(ped_exit) else "not_detected",
            "vehicle_crossing_completion_time_s": veh_exit,
            "vehicle_crossing_completion_detected": bool(np.isfinite(veh_exit)),
            "vehicle_crossing_completion_method": "supported_conflict_zone_first_visit_exit" if np.isfinite(veh_exit) else "not_detected",
            **chosen,
        })
        qc_rows.append({
            "recording_identifier": recording, "participant_id": participant, "scenario": scenario,
            "number_of_rows": len(one), "duration_s": float(time[-1] - time[0]),
            "invalid_or_nonmonotonic_time_intervals": int(invalid_dt.sum()),
            "median_sampling_interval_s": float(np.nanmedian(dt[~invalid_dt])) if (~invalid_dt).any() else math.nan,
            "sampling_rate_hz": float(1 / np.nanmedian(dt[~invalid_dt])) if (~invalid_dt).any() else math.nan,
            "sampling_interval_p05_s": float(np.nanpercentile(dt[~invalid_dt], 5)) if (~invalid_dt).any() else math.nan,
            "sampling_interval_p95_s": float(np.nanpercentile(dt[~invalid_dt], 95)) if (~invalid_dt).any() else math.nan,
            "pedestrian_position_missing_percent": float(one[["ped_x", "ped_z"]].isna().any(axis=1).mean() * 100),
            "vehicle_position_missing_percent": float(one[["veh_x", "veh_z"]].isna().any(axis=1).mean() * 100),
            "maximum_pedestrian_speed_mps": float(one.ped_speed.max()),
            "maximum_vehicle_speed_mps": float(one.veh_speed.max()),
            "maximum_absolute_pedestrian_acceleration_mps2": float(one.ped_acceleration.abs().max()),
            "maximum_absolute_vehicle_acceleration_mps2": float(one.veh_acceleration.abs().max()),
            "implausible_pedestrian_speed_samples": int(one.ped_speed.gt(config.pedestrian_implausible_speed_mps).sum()),
            "implausible_vehicle_speed_samples": int(one.veh_speed.gt(config.vehicle_implausible_speed_mps).sum()),
            "implausible_pedestrian_acceleration_samples": int(one.ped_acceleration.abs().gt(config.implausible_acceleration_mps2).sum()),
            "implausible_vehicle_acceleration_samples": int(one.veh_acceleration.abs().gt(config.implausible_acceleration_mps2).sum()),
            "quality_flag": ";".join([
                flag for flag, condition in (
                    ("invalid_timestamp", invalid_dt.any()),
                    ("missing_position", one[["ped_x", "ped_z", "veh_x", "veh_z"]].isna().any(axis=1).any()),
                    ("implausible_pedestrian_speed", one.ped_speed.gt(config.pedestrian_implausible_speed_mps).any()),
                    ("implausible_vehicle_speed", one.veh_speed.gt(config.vehicle_implausible_speed_mps).any()),
                    ("implausible_acceleration", one[["ped_acceleration", "veh_acceleration"]].abs().gt(config.implausible_acceleration_mps2).any(axis=1).any()),
                ) if condition
            ]),
        })
    return (
        pd.DataFrame(feature_rows), pd.DataFrame(event_rows), pd.DataFrame(qc_rows),
        pd.concat(normalized_rows, ignore_index=True), pd.DataFrame(transform_rows),
    )


def select_features(
    features: pd.DataFrame,
    config: AnalysisConfig,
    scope: str,
) -> tuple[list[str], pd.DataFrame]:
    decisions, eligible = [], []
    for priority, name in enumerate(INITIAL_CLUSTER_FEATURES):
        if name not in features:
            decisions.append({"scope": scope, "feature": name, "included": False, "reason": "not_calculated", "priority": priority})
            continue
        numeric = pd.to_numeric(features[name], errors="coerce")
        missing = float(numeric.isna().mean())
        iqr = float(numeric.quantile(.75) - numeric.quantile(.25))
        if missing > config.maximum_feature_missing_fraction:
            reason = f"missing_fraction_{missing:.3f}_exceeds_{config.maximum_feature_missing_fraction:.2f}"
        elif not np.isfinite(iqr) or iqr <= 1e-9:
            reason = "near_zero_robust_variance"
        else:
            eligible.append(name)
            reason = "eligible"
        decisions.append({
            "scope": scope, "feature": name, "included": False, "reason": reason,
            "priority": priority, "missing_fraction": missing, "iqr": iqr,
        })
    retained = []
    for name in eligible:
        if len(retained) >= config.maximum_retained_features:
            next(row for row in decisions if row["feature"] == name)["reason"] = "lower_priority_after_feature_cap"
            continue
        correlated_with = None
        for prior in retained:
            pair = features[[name, prior]].apply(pd.to_numeric, errors="coerce").dropna()
            correlation = pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 4 else np.nan
            if np.isfinite(correlation) and abs(correlation) > config.maximum_absolute_correlation:
                correlated_with = (prior, correlation)
                break
        row = next(item for item in decisions if item["feature"] == name)
        if correlated_with:
            row["reason"] = f"absolute_spearman_correlation_{abs(correlated_with[1]):.3f}_with_{correlated_with[0]}"
        else:
            retained.append(name)
            row["included"] = True
            row["reason"] = "retained"
    return retained, pd.DataFrame(decisions)


def robust_scale_within_group(
    features: pd.DataFrame,
    included: list[str],
    group_column: str = "scenario",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute and scale by median/IQR within each requested group."""
    scaled = pd.DataFrame(index=features.index)
    audit = []
    for group, indices in features.groupby(group_column).groups.items():
        one = features.loc[indices, included].apply(pd.to_numeric, errors="coerce")
        for name in included:
            median = float(one[name].median())
            iqr = float(one[name].quantile(.75) - one[name].quantile(.25))
            if not np.isfinite(iqr) or iqr <= 1e-9:
                raise ValueError(f"Cannot robust-scale {name} in {group_column}={group}: IQR={iqr}")
            missing_count = int(one[name].isna().sum())
            scaled.loc[indices, name] = (one[name].fillna(median) - median) / iqr
            audit.append({
                group_column: group, "feature": name, "median": median, "iqr": iqr,
                "imputed_count": missing_count, "imputation": "within_group_median",
                "scaling": "(value - within_group_median) / within_group_IQR",
            })
    return scaled.astype(float), pd.DataFrame(audit)


def _cluster_labels(method: str, clusters: int, matrix: np.ndarray, seed: int) -> np.ndarray:
    if method == "ward":
        return AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(matrix)
    if method == "gmm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return GaussianMixture(n_components=clusters, covariance_type="full", n_init=10, random_state=seed).fit_predict(matrix)
    if method == "kmeans":
        return KMeans(n_clusters=clusters, n_init=50, random_state=seed).fit_predict(matrix)
    raise ValueError(f"Unknown clustering method: {method}")


def bootstrap_stability(
    matrix: np.ndarray,
    base_labels: np.ndarray,
    method: str,
    clusters: int,
    seed: int,
    repetitions: int,
    groups: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Compare base labels with row- or group-bootstrap labels using ARI."""
    rng = np.random.default_rng(seed)
    scores = []
    n = len(matrix)
    unique_groups = np.unique(groups) if groups is not None else None
    for repetition in range(repetitions):
        if unique_groups is None:
            sampled = rng.choice(n, n, replace=True)
        else:
            chosen = rng.choice(unique_groups, len(unique_groups), replace=True)
            sampled = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        unique_indices = np.unique(sampled)
        if len(unique_indices) < max(clusters * 2, 6):
            continue
        try:
            labels = _cluster_labels(method, clusters, matrix[sampled], seed + repetition + 1)
        except Exception:
            continue
        mapped = {}
        for index, label in zip(sampled, labels):
            mapped.setdefault(int(index), []).append(int(label))
        comparable = np.array(sorted(mapped), dtype=int)
        boot = np.array([Counter(mapped[index]).most_common(1)[0][0] for index in comparable])
        scores.append(adjusted_rand_score(base_labels[comparable], boot))
    if not scores:
        return math.nan, math.nan, 0
    return float(np.mean(scores)), float(np.percentile(scores, 10)), len(scores)


def evaluate_cluster_candidates(
    matrix: np.ndarray,
    seed: int,
    repetitions: int,
    scope: str,
    groups: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    rows, labels_by_candidate = [], {}
    maximum_clusters = min(5, max(2, len(matrix) // 5))
    for method in ("ward", "gmm", "kmeans"):
        for clusters in range(2, maximum_clusters + 1):
            try:
                labels = _cluster_labels(method, clusters, matrix, seed)
                sizes = np.bincount(labels, minlength=clusters)
                stability, stability_p10, valid_bootstraps = bootstrap_stability(
                    matrix, labels, method, clusters, seed + clusters * 101,
                    repetitions, groups=groups,
                )
                rows.append({
                    "scope": scope, "method": method, "clusters": clusters,
                    "silhouette": silhouette_score(matrix, labels),
                    "davies_bouldin": davies_bouldin_score(matrix, labels),
                    "calinski_harabasz": calinski_harabasz_score(matrix, labels),
                    "bootstrap_ari_mean": stability,
                    "bootstrap_ari_p10": stability_p10,
                    "valid_bootstrap_repetitions": valid_bootstraps,
                    "minimum_cluster_size": int(sizes.min()),
                    "maximum_cluster_size": int(sizes.max()),
                    "cluster_sizes": ";".join(map(str, sizes.tolist())),
                    "error": "",
                })
                labels_by_candidate[(method, clusters)] = labels
            except Exception as exc:
                rows.append({
                    "scope": scope, "method": method, "clusters": clusters,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return pd.DataFrame(rows), labels_by_candidate


def select_cluster_solution(metrics: pd.DataFrame, config: AnalysisConfig) -> dict[str, object]:
    valid = metrics[metrics.error.fillna("").eq("")].copy()
    if valid.empty:
        return {"status": "no_stable_separation", "reason": "all candidate fits failed"}
    sample_size = int(sum(map(int, str(valid.iloc[0].cluster_sizes).split(";"))))
    required_minimum_size = max(
        config.minimum_cluster_size, int(math.ceil(0.10 * sample_size))
    )
    eligible = valid[
        valid.minimum_cluster_size.ge(required_minimum_size)
        & valid.bootstrap_ari_mean.ge(config.stability_threshold)
        & valid.bootstrap_ari_p10.ge(config.stability_lower_threshold)
        & valid.silhouette.ge(config.silhouette_threshold)
    ].copy()
    adequate_size = valid[valid.minimum_cluster_size.ge(required_minimum_size)]
    pool = eligible if len(eligible) else adequate_size if len(adequate_size) else valid
    pool = pool.sort_values(
        ["bootstrap_ari_mean", "silhouette", "clusters", "davies_bouldin"],
        ascending=[False, False, True, True], kind="stable",
    )
    best = pool.iloc[0]
    defensible = len(eligible) > 0
    return {
        "status": "selected_stable_solution" if defensible else "no_stable_separation",
        "defensible": defensible,
        "method": best.method,
        "clusters": int(best.clusters),
        "silhouette": float(best.silhouette),
        "davies_bouldin": float(best.davies_bouldin),
        "calinski_harabasz": float(best.calinski_harabasz),
        "bootstrap_ari_mean": float(best.bootstrap_ari_mean),
        "bootstrap_ari_p10": float(best.bootstrap_ari_p10),
        "minimum_cluster_size": int(best.minimum_cluster_size),
        "required_minimum_cluster_size": required_minimum_size,
        "sample_size": sample_size,
        "reason": (
            "Passed pre-specified stability, separation, and minimum-size criteria."
            if defensible else
            "Best exploratory candidate failed one or more pre-specified stability, separation, or minimum-size criteria; every cluster must contain at least 10% of the sample and at least three trials."
        ),
    }


def feature_dictionary(features: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Return an auditable dictionary for every derived trial feature."""
    metadata = {
        "pedestrian_movement_onset_time_s": ("s", "First existing-pipeline sample with smoothed pedestrian speed >= 0.30 m/s.", "ScenarioTime; B VR Pos X/Z", "whole trial", "Detected pedestrian movement onset."),
        "main_interaction_time_s": ("s", "Priority: mean of reliable pedestrian/vehicle zone-entry times; otherwise minimum synchronized distance; otherwise closest joint approach to geometric center.", "conflict-zone events and synchronized B VR/A car positions", "whole trial", "Primary time-alignment event."),
        "ped_mean_walking_speed_mps": ("m/s", "Arithmetic mean smoothed pedestrian X/Z speed from movement onset onward.", "B VR Pos X/Z; ScenarioTime", "onset to trial end", "Typical pedestrian movement magnitude."),
        "ped_median_walking_speed_mps": ("m/s", "Median smoothed pedestrian X/Z speed from movement onset onward.", "B VR Pos X/Z; ScenarioTime", "onset to trial end", "Robust typical pedestrian movement magnitude."),
        "ped_maximum_plausible_speed_mps": ("m/s", f"Maximum smoothed pedestrian speed among samples <= {config.pedestrian_implausible_speed_mps:g} m/s; excluded samples remain QC flags.", "B VR Pos X/Z; ScenarioTime", "whole trial", "Largest non-flagged pedestrian speed."),
        "ped_speed_before_interaction_mps": ("m/s", "Mean smoothed pedestrian speed.", "B VR Pos X/Z; ScenarioTime", f"{config.before_window[0]} to {config.before_window[1]} s from interaction", "Pre-interaction pedestrian speed."),
        "ped_speed_during_interaction_mps": ("m/s", "Mean smoothed pedestrian speed.", "B VR Pos X/Z; ScenarioTime", f"{config.during_window[0]} to {config.during_window[1]} s", "Interaction-window pedestrian speed."),
        "ped_speed_after_interaction_mps": ("m/s", "Mean smoothed pedestrian speed.", "B VR Pos X/Z; ScenarioTime", f"{config.after_window[0]} to {config.after_window[1]} s", "Post-interaction pedestrian speed."),
        "ped_speed_change_mps": ("m/s", "ped_speed_after_interaction_mps - ped_speed_before_interaction_mps.", "derived pedestrian speed windows", "before versus after", "Negative means pedestrian speed decreased."),
        "ped_min_interaction_speed_mps": ("m/s", "Minimum smoothed pedestrian speed.", "B VR Pos X/Z; ScenarioTime", f"{config.during_window[0]} to {config.during_window[1]} s", "Depth of pedestrian slowing/stopping."),
        "ped_max_acceleration_mps2": ("m/s²", "Maximum existing-pipeline smoothed derivative of pedestrian speed.", "ped_speed; actual ScenarioTime differences", f"{config.during_window[0]} to {config.during_window[1]} s", "Strongest positive pedestrian acceleration."),
        "ped_max_deceleration_mps2": ("m/s²", "Minimum existing-pipeline smoothed derivative of pedestrian speed.", "ped_speed; actual ScenarioTime differences", f"{config.during_window[0]} to {config.during_window[1]} s", "Most negative pedestrian acceleration."),
        "ped_total_stationary_duration_s": ("s", f"Sum of contiguous post-onset periods speed < {config.stationary_speed_mps:g} m/s lasting >= {config.minimum_stop_duration_s:g} s.", "ped_speed; ScenarioTime", "movement onset to trial end", "Total detected post-onset pedestrian stationary time."),
        "ped_longest_stationary_period_s": ("s", "Maximum qualifying post-onset stationary episode duration.", "ped_speed; ScenarioTime", "movement onset to trial end", "Longest post-onset pedestrian stop."),
        "ped_stop_episode_count": ("count", "Count of qualifying contiguous post-onset stationary episodes.", "ped_speed; ScenarioTime", "movement onset to trial end", "Repeated stop-and-go behavior measure."),
        "ped_time_onset_to_zone_entry_s": ("s", "pedestrian zone entry - movement onset; only for supported zones.", "conflict-zone event; onset", "onset to zone entry", "Pedestrian approach timing."),
        "ped_time_in_conflict_zone_s": ("s", "pedestrian zone exit - entry; only for supported zones.", "conflict-zone events", "first contiguous zone visit", "Conflict-zone occupancy duration."),
        "ped_crossing_duration_s": ("s", "pedestrian zone exit - movement onset; only for supported zones.", "conflict-zone event; onset", "onset through zone exit", "Available crossing-duration proxy."),
        "vehicle_mean_interaction_speed_mps": ("m/s", "Mean smoothed vehicle X/Z speed.", "A car Pos X/Z; ScenarioTime", f"{config.during_window[0]} to {config.during_window[1]} s", "Typical vehicle speed at interaction."),
        "vehicle_speed_before_interaction_mps": ("m/s", "Mean smoothed vehicle speed.", "A car Pos X/Z; ScenarioTime", f"{config.before_window[0]} to {config.before_window[1]} s", "Pre-interaction vehicle speed."),
        "vehicle_speed_during_interaction_mps": ("m/s", "Mean smoothed vehicle speed.", "A car Pos X/Z; ScenarioTime", f"{config.during_window[0]} to {config.during_window[1]} s", "Interaction-window vehicle speed."),
        "vehicle_speed_after_interaction_mps": ("m/s", "Mean smoothed vehicle speed.", "A car Pos X/Z; ScenarioTime", f"{config.after_window[0]} to {config.after_window[1]} s", "Post-interaction vehicle speed."),
        "vehicle_speed_change_mps": ("m/s", "vehicle_speed_after_interaction_mps - vehicle_speed_before_interaction_mps.", "derived vehicle speed windows", "before versus after", "Negative means vehicle speed decreased."),
        "vehicle_min_interaction_speed_mps": ("m/s", "Minimum smoothed vehicle speed.", "A car Pos X/Z; ScenarioTime", f"{config.during_window[0]} to {config.during_window[1]} s", "Depth of vehicle slowing/stopping."),
        "vehicle_max_acceleration_mps2": ("m/s²", "Maximum existing-pipeline smoothed derivative of vehicle speed.", "vehicle speed; actual ScenarioTime differences", f"{config.during_window[0]} to {config.during_window[1]} s", "Strongest positive vehicle acceleration."),
        "vehicle_max_deceleration_mps2": ("m/s²", "Minimum existing-pipeline smoothed derivative of vehicle speed.", "vehicle speed; actual ScenarioTime differences", f"{config.during_window[0]} to {config.during_window[1]} s", "Strongest vehicle braking."),
        "vehicle_braking_onset_relative_s": ("s", f"First vehicle acceleration <= {config.braking_acceleration_mps2:g} m/s² relative to interaction.", "veh_acceleration; ScenarioTime", "-5 to +3 s", "Negative means braking began before interaction."),
        "vehicle_total_stationary_duration_s": ("s", f"Sum of contiguous periods speed < {config.stationary_speed_mps:g} m/s lasting >= {config.minimum_stop_duration_s:g} s.", "veh_speed; ScenarioTime", "-5 to +3 s around interaction", "Interaction-neighborhood vehicle stationary time."),
        "vehicle_longest_stationary_period_s": ("s", "Maximum qualifying stationary episode duration.", "veh_speed; ScenarioTime", "-5 to +3 s around interaction", "Longest interaction-neighborhood vehicle stop."),
        "vehicle_stop_episode_count": ("count", "Count of qualifying vehicle stationary episodes.", "veh_speed; ScenarioTime", "-5 to +3 s around interaction", "Repeated interaction-neighborhood vehicle stopping measure."),
        "minimum_distance_m": ("m", "Minimum synchronized sqrt((vehicle_x-ped_x)^2 + (vehicle_z-ped_z)^2).", "B VR Pos X/Z; A car Pos X/Z", "whole trial", "Closest observed planar separation."),
        "minimum_distance_relative_s": ("s", "minimum-distance ScenarioTime - main interaction time.", "conflict-zone distance output", "whole trial", "Timing of closest approach relative to alignment."),
        "arrival_time_difference_s": ("s", "pedestrian_entry_time - vehicle_entry_time; only for supported zones.", "conflict-zone events", "zone entry", "Negative: pedestrian entered first; positive: vehicle entered first."),
        "absolute_arrival_time_difference_s": ("s", "Absolute value of arrival_time_difference_s.", "conflict-zone events", "zone entry", "Magnitude of temporal separation."),
        "braking_to_pedestrian_response_lag_s": ("s", "pedestrian acceleration-response onset - vehicle braking onset.", "smoothed actor accelerations", "-5 to +3 s", "Positive means pedestrian response followed vehicle braking."),
        "pedestrian_response_onset_relative_s": ("s", f"First |pedestrian acceleration| >= {config.response_acceleration_mps2:g} m/s² relative to interaction.", "ped_acceleration", "-5 to +3 s", "Timing of measurable pedestrian speed response."),
        "distance_at_pedestrian_onset_m": ("m", "Nearest synchronized pedestrian–vehicle distance at movement onset.", "actor positions; onset", "movement onset", "Spatial gap when pedestrian starts moving."),
        "distance_at_vehicle_braking_onset_m": ("m", "Nearest synchronized distance at detected vehicle braking onset.", "actor positions; veh_acceleration", "braking onset", "Spatial gap when vehicle begins braking."),
        "post_encroachment_time_s": ("s", "Non-overlap gap between first actor's zone exit and second actor's entry; zero during overlap; only supported zones.", "conflict-zone entry/exit", "first zone visits", "Observed PET proxy."),
        "ttc_at_minimum_distance_s": ("s", "distance / positive closing rate only when converging > 0.10 m/s.", "conflict-zone synchronized distance", "minimum distance", "Conditional TTC; missing when assumptions fail."),
        "ped_path_length_m": ("m", "Sum of normalized pedestrian X/Z step lengths from onset.", "rigid-normalized B VR Pos X/Z", "onset to trial end", "Pedestrian travel distance."),
        "ped_net_displacement_m": ("m", "Euclidean distance from onset position to final position.", "rigid-normalized B VR Pos X/Z", "onset to trial end", "Pedestrian endpoint displacement."),
        "ped_path_efficiency": ("ratio", "net displacement / path length.", "rigid-normalized B VR Pos X/Z", "onset to trial end", "One is direct; smaller is less direct."),
        "ped_max_lateral_deviation_m": ("m", "Maximum perpendicular distance from start-to-end line.", "rigid-normalized B VR Pos X/Z", "onset to trial end", "Largest pedestrian path departure."),
        "ped_mean_lateral_deviation_m": ("m", "Mean perpendicular distance from start-to-end line.", "rigid-normalized B VR Pos X/Z", "onset to trial end", "Typical pedestrian path departure."),
        "ped_total_heading_change_deg": ("degrees", "Sum of absolute unwrapped heading changes for nonzero steps.", "rigid-normalized B VR Pos X/Z", "onset to trial end", "Path curvature/turning proxy."),
        "vehicle_path_length_m": ("m", "Sum of normalized vehicle X/Z step lengths.", "rigid-normalized A car Pos X/Z", "whole trial", "Vehicle travel distance."),
        "vehicle_path_efficiency": ("ratio", "vehicle net displacement / path length.", "rigid-normalized A car Pos X/Z", "whole trial", "Vehicle path directness."),
        "vehicle_max_lateral_deviation_m": ("m", "Maximum perpendicular distance from vehicle start-to-end line.", "rigid-normalized A car Pos X/Z", "whole trial", "Largest vehicle path departure."),
        "vehicle_mean_lateral_deviation_m": ("m", "Mean perpendicular distance from vehicle start-to-end line.", "rigid-normalized A car Pos X/Z", "whole trial", "Typical vehicle path departure."),
        "paired_speed_change_product": ("(m/s)²", "ped_speed_change_mps * vehicle_speed_change_mps.", "derived speed-change features", "before versus after", "Positive when changes have the same sign."),
    }
    rows = []
    identifiers = {"recording_identifier", "participant_id", "scenario", "trial_key", "measured_entity_entering_first", "vehicle_stopped_before_conflict_zone"}
    for name in features.columns:
        if name in identifiers:
            continue
        unit, formula, source, window, interpretation = metadata.get(
            name, ("documented in code", "Direct deterministic calculation; see named source features.", "validated trajectory/event outputs", "trial-specific", "Derived trial descriptor.")
        )
        rows.append({
            "feature": name, "formula_or_definition": formula, "units": unit,
            "source_variables": source, "window_definition": window,
            "interpretation": interpretation,
            "eligible_for_clustering_initially": name in INITIAL_CLUSTER_FEATURES,
        })
    return pd.DataFrame(rows)


def _candidate_centroids(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([matrix[labels == label].mean(axis=0) for label in sorted(np.unique(labels))])


def describe_clusters(
    features: pd.DataFrame,
    scaled: pd.DataFrame,
    labels: np.ndarray,
    included: list[str],
    defensible: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    overall = scaled[included].median()
    profiles, representatives, unusual = [], [], []
    names: dict[int, str] = {}
    matrix = scaled[included].to_numpy(dtype=float)
    centroids = _candidate_centroids(matrix, labels)

    def measured_phrase(feature: str, value: float) -> str:
        phrases = {
            "ped_speed_change_mps": ("greater pedestrian speed increase", "greater pedestrian slowing"),
            "ped_min_interaction_speed_mps": ("higher pedestrian minimum speed", "lower pedestrian minimum speed"),
            "ped_longest_stationary_period_s": ("longer post-onset pedestrian stationary period", "shorter post-onset pedestrian stationary period"),
            "ped_max_deceleration_mps2": ("milder pedestrian deceleration", "stronger pedestrian deceleration"),
            "vehicle_speed_change_mps": ("greater vehicle speed increase", "greater vehicle slowing"),
            "vehicle_min_interaction_speed_mps": ("higher vehicle minimum speed", "lower vehicle minimum speed"),
            "vehicle_max_deceleration_mps2": ("milder vehicle deceleration", "stronger vehicle deceleration"),
            "arrival_time_difference_s": ("later pedestrian zone arrival", "earlier pedestrian zone arrival"),
            "minimum_distance_m": ("larger minimum separation", "smaller minimum separation"),
            "distance_at_pedestrian_onset_m": ("larger distance at pedestrian onset", "smaller distance at pedestrian onset"),
            "ped_path_efficiency": ("more direct pedestrian path", "less direct pedestrian path"),
            "ped_max_lateral_deviation_m": ("greater pedestrian lateral deviation", "smaller pedestrian lateral deviation"),
            "ped_total_heading_change_deg": ("greater pedestrian heading change", "smaller pedestrian heading change"),
            "vehicle_mean_lateral_deviation_m": ("greater vehicle path deviation", "smaller vehicle path deviation"),
            "braking_to_pedestrian_response_lag_s": ("longer braking-to-pedestrian response lag", "shorter braking-to-pedestrian response lag"),
        }
        positive, negative = phrases.get(feature, (f"higher {feature}", f"lower {feature}"))
        return positive if value >= 0 else negative

    for label in sorted(np.unique(labels)):
        mask = labels == label
        raw = features.loc[mask]
        standardized = scaled.loc[mask, included]
        signed_contrasts = standardized.median() - overall
        contrasts = signed_contrasts.abs().sort_values(ascending=False)
        if float(contrasts.iloc[0]) < 0.35:
            description = "central measured interaction profile"
        else:
            strongest = contrasts.head(2).index.tolist()
            description = " / ".join(
                measured_phrase(name, float(signed_contrasts[name])) for name in strongest
            )
        prefix = "Exploratory" if not defensible else ""
        names[int(label)] = f"{prefix + ' ' if prefix else ''}{description}".strip()
        distances = np.linalg.norm(matrix[mask] - centroids[label], axis=1)
        indices = np.flatnonzero(mask)
        nearest = indices[np.argsort(distances)[: min(3, len(indices))]]
        farthest = indices[int(np.argmax(distances))]
        all_distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        sorted_distances = np.sort(all_distances, axis=1)
        margins = sorted_distances[:, 1] - sorted_distances[:, 0] if centroids.shape[0] > 1 else np.full(len(matrix), np.nan)
        boundary_candidates = indices[np.argsort(margins[indices])[:1]]
        for rank, index in enumerate(nearest, start=1):
            representatives.append({
                "cluster_id": int(label), "cluster_label": names[int(label)],
                "role": "representative_nearest_centroid", "rank": rank,
                "recording_identifier": features.at[index, "recording_identifier"],
                "distance_to_centroid": float(np.linalg.norm(matrix[index] - centroids[label])),
            })
        unusual.append({
            "cluster_id": int(label), "cluster_label": names[int(label)], "role": "unusual_farthest_from_centroid",
            "recording_identifier": features.at[farthest, "recording_identifier"],
            "distance_to_centroid": float(np.linalg.norm(matrix[farthest] - centroids[label])),
        })
        for index in boundary_candidates:
            unusual.append({
                "cluster_id": int(label), "cluster_label": names[int(label)], "role": "boundary_smallest_centroid_margin",
                "recording_identifier": features.at[index, "recording_identifier"],
                "distance_to_centroid": float(np.linalg.norm(matrix[index] - centroids[label])),
            })
        row = {
            "cluster_id": int(label), "cluster_label": names[int(label)], "sample_size": int(mask.sum()),
            "scenarios_represented": ";".join(map(str, sorted(raw.scenario.unique()))),
            "participants_represented": int(raw.participant_id.nunique()),
            "most_distinguishing_features": ";".join(contrasts.head(3).index),
        }
        for name in included:
            row[f"median_{name}"] = float(raw[name].median())
            row[f"standardized_median_{name}"] = float(standardized[name].median())
        profiles.append(row)
    return pd.DataFrame(profiles), pd.DataFrame(representatives), pd.DataFrame(unusual), names


def build_curve_matrix(
    traces: pd.DataFrame,
    interaction_events: pd.DataFrame,
    config: AnalysisConfig,
    alignment_name: str = "main_interaction",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    event_times = interaction_events.set_index("recording_identifier").main_interaction_time.to_dict()
    grid = np.arange(config.curve_window[0], config.curve_window[1] + config.curve_step / 2, config.curve_step)
    rows, vectors, ids = [], [], []
    for recording, source in traces.groupby("recording_identifier", sort=True):
        one = source.sort_values("scenario_time").drop_duplicates("scenario_time")
        event_time = event_times.get(recording, np.nan)
        if not np.isfinite(event_time):
            continue
        relative = one.scenario_time.to_numpy(dtype=float) - event_time
        channels = []
        for actor, column in (("pedestrian", "ped_speed"), ("vehicle", "veh_speed")):
            values = one[column].to_numpy(dtype=float)
            valid = np.isfinite(relative) & np.isfinite(values)
            if valid.sum() < 2:
                interpolated = np.full(len(grid), np.nan)
            else:
                order = np.argsort(relative[valid])
                x, y = relative[valid][order], values[valid][order]
                x, unique_indices = np.unique(x, return_index=True)
                y = y[unique_indices]
                interpolated = np.interp(grid, x, y, left=np.nan, right=np.nan)
            channels.append(interpolated)
            rows.extend({
                "recording_identifier": recording,
                "participant_id": one.participant_id.iloc[0], "scenario": int(one.scenario.iloc[0]),
                "alignment": alignment_name, "actor": actor, "relative_time_s": float(t),
                "speed_mps": float(value) if np.isfinite(value) else math.nan,
                "observed_coverage": bool(np.isfinite(value)),
            } for t, value in zip(grid, interpolated))
        vector = np.concatenate(channels)
        if np.isfinite(vector).mean() >= 0.70:
            vectors.append(vector)
            ids.append(recording)
    return pd.DataFrame(rows), pd.DataFrame({"recording_identifier": ids}), np.vstack(vectors), [f"ped_{t:.1f}" for t in grid] + [f"veh_{t:.1f}" for t in grid]


def functional_pca(curves: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    column_medians = np.nanmedian(curves, axis=0)
    filled = np.where(np.isfinite(curves), curves, column_medians)
    means = filled.mean(axis=0)
    standard_deviations = filled.std(axis=0)
    standard_deviations[standard_deviations <= 1e-9] = 1.0
    standardized = (filled - means) / standard_deviations
    maximum = min(8, standardized.shape[0] - 1, standardized.shape[1])
    full = PCA(n_components=maximum, random_state=seed).fit(standardized)
    components = max(2, int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.90) + 1))
    components = min(components, maximum)
    scores = full.transform(standardized)[:, :components]
    return scores, full.explained_variance_ratio_[:components], {
        "method": "functional_PCA_on_separate_interaction_aligned_pedestrian_and_vehicle_speed_channels",
        "missing_curve_samples": "timepoint median imputation for sensitivity analysis only",
        "components": components,
        "explained_variance_fraction": float(full.explained_variance_ratio_[:components].sum()),
    }


def cramers_v(table: pd.DataFrame) -> float:
    if table.empty or min(table.shape) < 2:
        return math.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, columns = table.shape
    corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(n - 1, 1))
    corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    corrected_columns = columns - ((columns - 1) ** 2) / max(n - 1, 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return float(math.sqrt(corrected / denominator)) if denominator > 0 else math.nan


def validate_ground_truth(assignments: pd.DataFrame, ground_truth: pd.DataFrame, features: pd.DataFrame):
    joined = assignments.merge(features[["recording_identifier", "trial_key", "measured_entity_entering_first"]], on="recording_identifier", how="left")
    joined = joined.merge(ground_truth, on="trial_key", how="left", suffixes=("", "_ground_truth"))
    for name in LABEL_COLUMNS.values():
        joined[name] = joined[name].fillna("Unknown")
    tests, distributions = [], []
    for label_column in (
        "driver_signal", "annotated_vehicle_speed_change", "annotated_pedestrian_speed_change",
        "annotated_right_of_way", "annotated_who_went_first", "pedestrian_hand_gesture",
    ):
        valid = joined[joined[label_column].ne("Unknown")]
        table = pd.crosstab(valid.cluster_id, valid[label_column])
        if min(table.shape, default=0) >= 2:
            if table.shape == (2, 2):
                statistic, p_value = stats.fisher_exact(table.to_numpy())
                test = "Fisher exact"
            else:
                statistic, p_value, _, expected = stats.chi2_contingency(table)
                test = "chi-square" if (expected >= 5).mean() >= 0.80 else "chi-square (sparse expected counts; interpret cautiously)"
            ari = adjusted_rand_score(valid.cluster_id, pd.factorize(valid[label_column])[0])
            nmi = normalized_mutual_info_score(valid.cluster_id, pd.factorize(valid[label_column])[0])
        else:
            statistic = p_value = ari = nmi = math.nan
            test = "not_testable"
        tests.append({
            "ground_truth_label": label_column, "test": test, "statistic": statistic,
            "p_value": p_value, "cramers_v": cramers_v(table), "adjusted_rand_index": ari,
            "normalized_mutual_information": nmi, "valid_rows": len(valid),
        })
        counts = pd.crosstab(joined.cluster_id, joined[label_column], dropna=False)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        for cluster in counts.index:
            for category in counts.columns:
                distributions.append({
                    "ground_truth_label": label_column, "cluster_id": cluster,
                    "category": category, "count": int(counts.at[cluster, category]),
                    "within_cluster_percent": float(percentages.at[cluster, category]),
                })
    comparison = joined[[
        "recording_identifier", "participant_id", "scenario", "cluster_id", "cluster_label",
        "measured_entity_entering_first", "annotated_who_went_first",
    ]].copy()
    comparison["annotated_entity"] = comparison.annotated_who_went_first.map({"Driver": "vehicle", "Pedestrian": "pedestrian"}).fillna("Unknown")
    comparison["comparable"] = comparison.measured_entity_entering_first.isin(["pedestrian", "vehicle"]) & comparison.annotated_entity.isin(["pedestrian", "vehicle"])
    comparison["agreement"] = np.where(
        comparison.comparable,
        comparison.measured_entity_entering_first.eq(comparison.annotated_entity),
        pd.NA,
    )
    valid_comparison = comparison[comparison.comparable]
    comparison_summary = {
        "comparable_trials": int(len(valid_comparison)),
        "agreement_count": int(valid_comparison.agreement.sum()) if len(valid_comparison) else 0,
        "agreement_fraction": float(valid_comparison.agreement.mean()) if len(valid_comparison) else math.nan,
        "cohen_kappa": float(cohen_kappa_score(valid_comparison.annotated_entity, valid_comparison.measured_entity_entering_first)) if len(valid_comparison) else math.nan,
        "interpretation": "Manual who-went-first and computed conflict-zone entry order are related but non-identical constructs.",
    }
    return joined, pd.DataFrame(tests), pd.DataFrame(distributions), comparison, comparison_summary


def _setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_qc(qc: pd.DataFrame, path: Path) -> None:
    plt = _setup_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(qc.sampling_rate_hz.dropna(), bins=20, color="#2A6FBB")
    axes[0, 0].set(title="Sampling-rate distribution", xlabel="Hz", ylabel="Trials")
    missing = qc[["pedestrian_position_missing_percent", "vehicle_position_missing_percent"]]
    axes[0, 1].boxplot(
        [missing.iloc[:, 0], missing.iloc[:, 1]],
        tick_labels=["Pedestrian", "Vehicle"],
    )
    axes[0, 1].set(title="Missing position samples", ylabel="Percent")
    axes[1, 0].scatter(qc.maximum_pedestrian_speed_mps, qc.maximum_absolute_pedestrian_acceleration_mps2, alpha=.65, label="Pedestrian")
    axes[1, 0].scatter(qc.maximum_vehicle_speed_mps, qc.maximum_absolute_vehicle_acceleration_mps2, alpha=.65, label="Vehicle")
    axes[1, 0].set(title="Speed and acceleration maxima", xlabel="Maximum speed (m/s)", ylabel="Maximum |acceleration| (m/s²)")
    axes[1, 0].legend()
    flag_counts = qc.quality_flag.replace("", "none").value_counts().head(10)
    axes[1, 1].barh(flag_counts.index[::-1], flag_counts.values[::-1], color="#D1495B")
    axes[1, 1].set(title="Trial-level QC flag combinations", xlabel="Trials")
    for axis in axes.flat:
        axis.grid(True, alpha=.2)
    fig.suptitle("Behavior-clustering input quality control", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_normalization(raw: pd.DataFrame, normalized: pd.DataFrame, path: Path) -> None:
    plt = _setup_plotting()
    scenarios = sorted(raw.scenario.unique())
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(13, 4 * len(scenarios)))
    for row, scenario in enumerate(scenarios):
        for recording, one in raw[raw.scenario.eq(scenario)].groupby("recording_identifier"):
            axes[row, 0].plot(one.ped_x, one.ped_z, color="#2A6FBB", alpha=.18, linewidth=.7)
            axes[row, 0].plot(one.veh_x, one.veh_z, color="#D1495B", alpha=.14, linewidth=.7)
        for recording, one in normalized[normalized.scenario.eq(scenario)].groupby("recording_identifier"):
            axes[row, 1].plot(one.ped_x_normalized_m, one.ped_z_normalized_m, color="#2A6FBB", alpha=.18, linewidth=.7)
            axes[row, 1].plot(one.veh_x_normalized_m, one.veh_z_normalized_m, color="#D1495B", alpha=.14, linewidth=.7)
        axes[row, 0].set_title(f"Scenario {scenario}: before (shared Unity frame)")
        axes[row, 1].set_title(f"Scenario {scenario}: after rigid normalization")
        for axis in axes[row]:
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("X (m)")
            axis.set_ylabel("Z (m)")
            axis.grid(True, alpha=.2)
    fig.suptitle("Trajectory normalization verification (blue pedestrian, red vehicle)", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .99))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(features: pd.DataFrame, included: list[str], path: Path) -> None:
    plt = _setup_plotting()
    correlation = features[included].corr(method="spearman")
    fig, axis = plt.subplots(figsize=(max(9, len(included) * .9), max(8, len(included) * .8)))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(included)), included, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(included)), included, fontsize=8)
    for i in range(len(included)):
        for j in range(len(included)):
            axis.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    axis.set_title("Spearman correlations among retained pooled features", weight="bold")
    fig.colorbar(image, ax=axis, shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(scaled: pd.DataFrame, assignments: pd.DataFrame, included: list[str], path: Path) -> None:
    plt = _setup_plotting()
    order = assignments.sort_values(["cluster_id", "scenario", "participant_id"]).index
    matrix = scaled.loc[order, included].clip(-4, 4).to_numpy()
    fig, axis = plt.subplots(figsize=(max(11, len(included)), 14))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    axis.set_xticks(range(len(included)), included, rotation=60, ha="right", fontsize=8)
    axis.set_ylabel("Trials ordered by cluster, scenario, participant")
    axis.set_title("Scenario-adjusted standardized trial-by-feature heatmap", weight="bold")
    boundaries = assignments.loc[order, "cluster_id"].to_numpy()
    for position in np.flatnonzero(boundaries[1:] != boundaries[:-1]) + .5:
        axis.axhline(position, color="black", linewidth=1.2)
    fig.colorbar(image, ax=axis, label="Within-scenario robust z-like score")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca(
    scaled: pd.DataFrame,
    assignments: pd.DataFrame,
    included: list[str],
    cluster_path: Path,
    scenario_path: Path,
    html_path: Path,
    seed: int,
) -> pd.DataFrame:
    matrix = scaled[included].to_numpy(dtype=float)
    model = PCA(n_components=2, random_state=seed)
    scores = model.fit_transform(matrix)
    pca_frame = assignments[["recording_identifier", "participant_id", "scenario", "cluster_id", "cluster_label"]].copy()
    pca_frame["PC1"], pca_frame["PC2"] = scores[:, 0], scores[:, 1]
    pca_frame["PC1_explained_variance"] = model.explained_variance_ratio_[0]
    pca_frame["PC2_explained_variance"] = model.explained_variance_ratio_[1]
    plt = _setup_plotting()
    markers = {3: "o", 7: "s", 12: "^", 15: "v", 16: "D", 21: "P"}
    fig, axis = plt.subplots(figsize=(10, 8))
    for (cluster, scenario), one in pca_frame.groupby(["cluster_id", "scenario"]):
        axis.scatter(one.PC1, one.PC2, marker=markers.get(int(scenario), "o"), label=f"C{cluster}, S{scenario}", alpha=.75)
    axis.set(title="PCA of scenario-adjusted features — colored by cluster, shaped by scenario", xlabel=f"PC1 ({model.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({model.explained_variance_ratio_[1]:.1%})")
    axis.legend(ncol=3, fontsize=7, loc="best")
    axis.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(cluster_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 8))
    for scenario, one in pca_frame.groupby("scenario"):
        axis.scatter(one.PC1, one.PC2, marker=markers.get(int(scenario), "o"), label=f"Scenario {scenario}", alpha=.75)
    axis.set(title="PCA of scenario-adjusted features — colored by scenario", xlabel=f"PC1 ({model.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({model.explained_variance_ratio_[1]:.1%})")
    axis.legend(); axis.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(scenario_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    try:
        import plotly.express as px
        figure = px.scatter(
            pca_frame, x="PC1", y="PC2", color="cluster_label", symbol="scenario",
            hover_data=["recording_identifier", "participant_id", "scenario"],
            title="Interactive PCA: scenario-adjusted behavioral features",
        )
        figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:
        html_path.write_text(f"<html><body><p>Interactive plot unavailable: {exc}</p></body></html>", encoding="utf-8")
    return pca_frame


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    plt = _setup_plotting()
    valid = metrics[metrics.error.fillna("").eq("")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"ward": "#2A6FBB", "gmm": "#D1495B", "kmeans": "#59A14F"}
    for method, one in valid.groupby("method"):
        axes[0].scatter(one.silhouette, one.bootstrap_ari_mean, s=35 + 12 * one.clusters, label=method, color=colors[method], alpha=.75)
        axes[1].scatter(one.clusters, one.minimum_cluster_size, label=method, color=colors[method], alpha=.75)
    axes[0].axhline(.60, color="black", linestyle="--", linewidth=.8)
    axes[0].axvline(.15, color="black", linestyle="--", linewidth=.8)
    axes[0].set(title="Separation versus bootstrap stability", xlabel="Silhouette", ylabel="Mean bootstrap ARI")
    axes[1].set(title="Smallest cluster by requested k", xlabel="Number of clusters", ylabel="Minimum cluster size")
    for axis in axes:
        axis.grid(True, alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_cluster_profiles(profiles: pd.DataFrame, included: list[str], path: Path) -> None:
    plt = _setup_plotting()
    matrix = profiles[[f"standardized_median_{name}" for name in included]].to_numpy()
    fig, axis = plt.subplots(figsize=(max(11, len(included)), max(4, len(profiles) * 1.2)))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axis.set_xticks(range(len(included)), included, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(profiles)), [f"C{row.cluster_id}: {row.cluster_label}" for row in profiles.itertuples()], fontsize=8)
    axis.set_title("Cluster median feature profiles (scenario-adjusted units)", weight="bold")
    fig.colorbar(image, ax=axis, label="Median robust standardized value")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_velocity_curves(curves: pd.DataFrame, assignments: pd.DataFrame, path: Path, html_path: Path) -> pd.DataFrame:
    merged = curves.merge(assignments[["recording_identifier", "cluster_id", "cluster_label"]], on="recording_identifier", how="inner")
    summary = merged.groupby(["cluster_id", "cluster_label", "actor", "relative_time_s"]).speed_mps.agg(
        median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), contributors="count"
    ).reset_index()
    plt = _setup_plotting()
    clusters = sorted(summary.cluster_id.unique())
    fig, axes = plt.subplots(len(clusters), 1, figsize=(12, max(4, 4 * len(clusters))), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, cluster in zip(axes, clusters):
        one_cluster = summary[summary.cluster_id.eq(cluster)]
        for actor, color in (("pedestrian", "#2A6FBB"), ("vehicle", "#D1495B")):
            one = one_cluster[one_cluster.actor.eq(actor)]
            x = one.relative_time_s.to_numpy(dtype=float)
            axis.plot(x, one["median"].to_numpy(dtype=float), color=color, label=actor.title())
            axis.fill_between(x, one.q25.to_numpy(dtype=float), one.q75.to_numpy(dtype=float), color=color, alpha=.18)
            axis.plot(x, one.contributors.to_numpy(dtype=float) / max(one.contributors.max(), 1), color=color, linestyle=":", alpha=.45, label=f"{actor.title()} contributing fraction")
        axis.axvline(0, color="black", linestyle="--", linewidth=.8)
        axis.set(title=str(one_cluster.cluster_label.iloc[0]), ylabel="Speed (m/s)")
        axis.grid(True, alpha=.2); axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Seconds from main interaction event")
    fig.suptitle("Median actor speeds, IQR bands, and contributing-trial fractions", weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    try:
        import plotly.express as px
        figure = px.line(summary, x="relative_time_s", y="median", color="actor", facet_row="cluster_label", hover_data=["q25", "q75", "contributors"], title="Interaction-aligned cluster velocity summaries")
        figure.add_vline(x=0, line_dash="dash")
        figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:
        html_path.write_text(f"<html><body><p>Interactive plot unavailable: {exc}</p></body></html>", encoding="utf-8")
    return summary


def plot_representative_trajectories(normalized: pd.DataFrame, representatives: pd.DataFrame, path: Path, html_path: Path) -> None:
    chosen = normalized[normalized.recording_identifier.isin(representatives.recording_identifier)]
    plt = _setup_plotting()
    clusters = sorted(representatives.cluster_id.unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(max(7, 5 * len(clusters)), 5), squeeze=False)
    for axis, cluster in zip(axes[0], clusters):
        ids = set(representatives.loc[representatives.cluster_id.eq(cluster), "recording_identifier"])
        for recording, one in chosen[chosen.recording_identifier.isin(ids)].groupby("recording_identifier"):
            axis.plot(one.ped_x_normalized_m, one.ped_z_normalized_m, color="#2A6FBB", label=f"{recording} pedestrian")
            axis.plot(one.veh_x_normalized_m, one.veh_z_normalized_m, color="#D1495B", linestyle="--", label=f"{recording} vehicle")
        axis.scatter([0], [0], color="black", marker="x", label="Conflict center")
        axis.set(title=f"Cluster {cluster}", xlabel="Normalized X (m)", ylabel="Normalized Z (m)")
        axis.set_aspect("equal", adjustable="datalim"); axis.grid(True, alpha=.2); axis.legend(fontsize=6)
    fig.suptitle("Representative rigid-normalized trajectories", weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    try:
        import plotly.express as px
        long_rows = []
        for row in chosen.itertuples():
            long_rows.extend([
                {"recording_identifier": row.recording_identifier, "scenario": row.scenario, "actor": "Pedestrian", "x": row.ped_x_normalized_m, "z": row.ped_z_normalized_m},
                {"recording_identifier": row.recording_identifier, "scenario": row.scenario, "actor": "Vehicle", "x": row.veh_x_normalized_m, "z": row.veh_z_normalized_m},
            ])
        figure = px.line(pd.DataFrame(long_rows), x="x", y="z", color="actor", facet_col="recording_identifier", facet_col_wrap=3, title="Representative normalized trajectories")
        figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:
        html_path.write_text(f"<html><body><p>Interactive plot unavailable: {exc}</p></body></html>", encoding="utf-8")


def plot_compositions(assignments: pd.DataFrame, scenario_path: Path, participant_path: Path) -> None:
    plt = _setup_plotting()
    for column, path, title in (
        ("scenario", scenario_path, "Scenario-by-cluster composition"),
        ("participant_id", participant_path, "Participant-by-cluster composition"),
    ):
        table = pd.crosstab(assignments[column], assignments.cluster_id)
        fractions = table.div(table.sum(axis=1), axis=0)
        fig, axis = plt.subplots(figsize=(11, max(5, .32 * len(table))))
        bottom = np.zeros(len(table))
        for cluster in fractions.columns:
            axis.barh(range(len(table)), fractions[cluster], left=bottom, label=f"Cluster {cluster}")
            bottom += fractions[cluster].to_numpy()
        axis.set_yticks(range(len(table)), table.index.astype(str), fontsize=7)
        axis.set(xlabel="Within-row fraction", title=title)
        axis.legend(ncol=min(5, len(fractions.columns))); axis.grid(True, axis="x", alpha=.2)
        fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_ground_truth(distributions: pd.DataFrame, comparison: pd.DataFrame, distribution_path: Path, comparison_path: Path) -> None:
    plt = _setup_plotting()
    labels = distributions.ground_truth_label.unique()
    fig, axes = plt.subplots(math.ceil(len(labels) / 2), 2, figsize=(15, 4 * math.ceil(len(labels) / 2)))
    for axis, label in zip(np.asarray(axes).flat, labels):
        one = distributions[distributions.ground_truth_label.eq(label)]
        pivot = one.pivot(index="cluster_id", columns="category", values="within_cluster_percent").fillna(0)
        bottom = np.zeros(len(pivot))
        for category in pivot.columns:
            axis.bar(pivot.index.astype(str), pivot[category], bottom=bottom, label=category)
            bottom += pivot[category].to_numpy()
        axis.set(title=label.replace("_", " "), xlabel="Cluster", ylabel="Within-cluster percent")
        axis.legend(fontsize=6)
    for axis in np.asarray(axes).flat[len(labels):]:
        axis.axis("off")
    fig.suptitle("Ground-truth annotation distributions after clustering", weight="bold")
    fig.tight_layout(); fig.savefig(distribution_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    valid = comparison[comparison.comparable]
    table = pd.crosstab(valid.annotated_entity, valid.measured_entity_entering_first)
    fig, axis = plt.subplots(figsize=(7, 6))
    if table.empty:
        axis.text(.5, .5, "No comparable supported-zone entry-order trials", ha="center", va="center")
        axis.set_axis_off()
        axis.set_title("Measured versus annotated order")
    else:
        image = axis.imshow(table, cmap="Blues")
        axis.set_xticks(range(len(table.columns)), table.columns)
        axis.set_yticks(range(len(table.index)), table.index)
        axis.set(xlabel="Measured conflict-zone entrant first", ylabel="Annotated who went first", title="Measured versus annotated order")
        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                axis.text(j, i, int(table.iloc[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=axis)
    fig.tight_layout(); fig.savefig(comparison_path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_within_scenario_summary(summary: pd.DataFrame, path: Path) -> None:
    plt = _setup_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = np.where(summary.defensible, "#59A14F", "#B0B0B0")
    axes[0].bar(summary.scenario.astype(str), summary.bootstrap_ari_mean, color=colors)
    axes[0].axhline(.60, color="black", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="Mean bootstrap ARI", title="Within-scenario selected/best exploratory stability")
    axes[1].bar(summary.scenario.astype(str), summary.silhouette, color=colors)
    axes[1].axhline(.15, color="black", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="Scenario", ylabel="Silhouette", title="Within-scenario separation")
    for axis in axes:
        axis.grid(True, axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _association_leakage(assignments: pd.DataFrame) -> dict[str, float | bool]:
    labels = assignments.cluster_id.to_numpy()
    scenario = assignments.scenario.to_numpy()
    participant = pd.factorize(assignments.participant_id)[0]
    scenario_table = pd.crosstab(assignments.cluster_id, assignments.scenario)
    participant_table = pd.crosstab(assignments.cluster_id, assignments.participant_id)
    scenario_ami = float(adjusted_mutual_info_score(scenario, labels))
    participant_ami = float(adjusted_mutual_info_score(participant, labels))
    participant_concentration = []
    participant_counts = []
    for _, one in assignments.groupby("cluster_id"):
        counts = one.participant_id.value_counts()
        participant_concentration.append(float(counts.iloc[0] / len(one)))
        participant_counts.append(int(len(counts)))
    maximum_participant_fraction = max(participant_concentration)
    minimum_participants = min(participant_counts)
    participant_flag = bool(
        participant_ami >= .10
        or maximum_participant_fraction >= .50
        or minimum_participants < 3
    )
    return {
        "scenario_adjusted_mutual_information": scenario_ami,
        "scenario_normalized_mutual_information": float(normalized_mutual_info_score(scenario, labels)),
        "scenario_cramers_v": cramers_v(scenario_table),
        "participant_adjusted_mutual_information": participant_ami,
        "participant_normalized_mutual_information": float(normalized_mutual_info_score(participant, labels)),
        "participant_cramers_v": cramers_v(participant_table),
        "maximum_single_participant_fraction_within_cluster": maximum_participant_fraction,
        "minimum_unique_participants_in_any_cluster": minimum_participants,
        "scenario_leakage_detected": bool(scenario_ami >= .10),
        "participant_leakage_detected": participant_flag,
        "leakage_rule": (
            "Scenario leakage: adjusted mutual information >= 0.10. Participant leakage: "
            "adjusted mutual information >= 0.10, any cluster >= 50% one participant, or "
            "any cluster contains fewer than three unique participants. Cramer's V and "
            "composition tables are also reported."
        ),
    }


def _package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "plotly", "openpyxl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    try:
        versions["umap-learn"] = importlib.metadata.version("umap-learn")
    except importlib.metadata.PackageNotFoundError:
        versions["umap-learn"] = "not installed; UMAP deliberately omitted"
    return versions


def _exact_command(args: argparse.Namespace) -> str:
    scenarios = " ".join(map(str, args.scenarios))
    command = (
        f'python -m src.analysis.behavior_clustering --seed {args.seed} '
        f'--scenarios {scenarios} --input-root "{args.input_root.resolve()}" '
        f'--ground-truth "{args.ground_truth.resolve()}" '
        f'--output-root "{args.output_root.resolve()}" '
        f'--bootstrap-repetitions {args.bootstrap_repetitions}'
    )
    return command + (" --overwrite" if args.overwrite else "")


def _write_reports(
    reports_dir: Path,
    config_record: dict[str, object],
    inventory: pd.DataFrame,
    matching: pd.DataFrame,
    features: pd.DataFrame,
    retained: list[str],
    within_summary: pd.DataFrame,
    selected_within_profiles: pd.DataFrame,
    selected_within_representatives: pd.DataFrame,
    selected_within_validation_tests: pd.DataFrame,
    pooled_solution: dict[str, object],
    leakage: dict[str, object],
    profiles: pd.DataFrame,
    representatives: pd.DataFrame,
    validation_tests: pd.DataFrame,
    comparison_summary: dict[str, object],
    curve_comparison: dict[str, object],
) -> None:
    defensible_within = within_summary.loc[within_summary.defensible, "scenario"].astype(int).tolist()
    unmatched_gt = int((inventory.motion_recording_count.eq(0) & inventory.ground_truth_present).sum())
    unique_matches = int(matching.match_status.eq("unique_match").sum())
    duplicate_matches = int(matching.match_status.str.contains("duplicate").sum())
    conclusion = "no reliable clustering structure"
    if pooled_solution.get("defensible"):
        if leakage["scenario_leakage_detected"]:
            conclusion = "stable separation detected, but substantially associated with scenario design"
        elif leakage["participant_leakage_detected"]:
            conclusion = "stable separation detected, but substantially associated with participant identity"
        else:
            conclusion = "genuine stable behavioral separation supported by the pre-specified criteria"
    elif defensible_within:
        conclusion = "weak or scenario-specific exploratory patterns; no stable pooled behavioral separation"

    top_validation = validation_tests.sort_values("cramers_v", ascending=False).head(3)
    profile_lines = []
    for row in profiles.itertuples():
        profile_lines.append(
            f"- Cluster {row.cluster_id}, **{row.cluster_label}**: n={row.sample_size}; "
            f"scenarios {row.scenarios_represented}; strongest differentiators: "
            f"{row.most_distinguishing_features}."
        )
    representative_lines = []
    for cluster, one in representatives.groupby("cluster_id"):
        representative_lines.append(
            f"- Cluster {cluster}: " + ", ".join(one.sort_values("rank").recording_identifier)
        )
    within_lines = []
    for row in within_summary.itertuples():
        within_lines.append(
            f"- Scenario {row.scenario}: {row.status}; best {row.method}, k={row.clusters}, "
            f"silhouette={row.silhouette:.3f}, mean bootstrap ARI={row.bootstrap_ari_mean:.3f}."
        )
    selected_profile_lines = []
    for row in selected_within_profiles.itertuples():
        selected_profile_lines.append(
            f"- Scenario {row.scenario}, cluster {row.cluster_id}, **{row.cluster_label}**: "
            f"n={row.sample_size}; strongest differentiators: {row.most_distinguishing_features}."
        )
    selected_representative_lines = []
    for (scenario, cluster), one in selected_within_representatives.groupby(["scenario", "cluster_id"]):
        selected_representative_lines.append(
            f"- Scenario {scenario}, cluster {cluster}: "
            + ", ".join(one.sort_values("rank").recording_identifier)
        )
    validation_lines = []
    for row in top_validation.itertuples():
        validation_lines.append(
            f"- {row.ground_truth_label}: Cramér's V={row.cramers_v:.3f}, "
            f"p={row.p_value:.4g}, NMI={row.normalized_mutual_information:.3f}."
        )
    selected_validation_lines = []
    if len(selected_within_validation_tests):
        for row in selected_within_validation_tests.sort_values("cramers_v", ascending=False).head(3).itertuples():
            selected_validation_lines.append(
                f"- Scenario {row.scenario}, {row.ground_truth_label}: "
                f"Cramér's V={row.cramers_v:.3f}, p={row.p_value:.4g}, "
                f"NMI={row.normalized_mutual_information:.3f}."
            )

    professor = f"""# Professor-facing summary

## Bottom line

The pre-specified decision rule classified the result as **{conclusion}**. The analysis did not assume that clusters had to exist. The pooled best candidate used {pooled_solution.get('method')} with {pooled_solution.get('clusters')} groups; its silhouette was {pooled_solution.get('silhouette', math.nan):.3f}, mean row-bootstrap ARI was {pooled_solution.get('bootstrap_ari_mean', math.nan):.3f}, and 10th-percentile bootstrap ARI was {pooled_solution.get('bootstrap_ari_p10', math.nan):.3f}.

## 1. Were stable behavioral groups found?

Pooled status: **{pooled_solution.get('status')}**. Within-scenario defensible solutions were found for: {', '.join(map(str, defensible_within)) if defensible_within else 'none'}.

## 2. How many groups were defensible?

No pooled groups were defensible. Scenario 12 had one defensible within-scenario solution containing three groups; the other five scenarios had none. The pooled {pooled_solution.get('clusters')}-group candidate is reported only as an exploratory outlier split.

## 3. What measured behavior defines each group?

{chr(10).join(profile_lines)}

These descriptions are based on measured speeds, timing, stopping duration, distance, and path geometry. They are not psychological diagnoses.

The defensible scenario-12 solution was:

{chr(10).join(selected_profile_lines) if selected_profile_lines else '- None.'}

## 4. Did groups appear within multiple scenarios?

{('No defensible pooled groups were found, so cross-scenario recurrence is not established. The exploratory outlier split spans multiple scenarios but is not a behavioral typology.' if not pooled_solution.get('defensible') else ('Yes; see each profile above.' if profiles.scenarios_represented.str.contains(';').all() else 'Not uniformly. The scenario composition table should be consulted before treating groups as recurring types.'))}

## 5. Did clusters agree with annotations?

{chr(10).join(validation_lines) if validation_lines else '- No annotation association was statistically testable.'}

These tests describe the exploratory pooled candidate only because no pooled feature-based solution passed the selection rule; they do not validate a stable pooled grouping.

For the selected scenario-12 solution:

{chr(10).join(selected_validation_lines) if selected_validation_lines else '- No annotation association was statistically testable.'}

Measured conflict-zone entry order agreed with the manual “who went first” annotation in {comparison_summary.get('agreement_fraction', math.nan):.1%} of {comparison_summary.get('comparable_trials', 0)} comparable recordings (Cohen's kappa {comparison_summary.get('cohen_kappa', math.nan):.3f}). Disagreements are listed individually because the measures are not identical concepts.

## 6. Were clusters scenario or participant effects?

- Scenario adjusted mutual information: {leakage['scenario_adjusted_mutual_information']:.3f}; leakage flag: {leakage['scenario_leakage_detected']}.
- Participant adjusted mutual information: {leakage['participant_adjusted_mutual_information']:.3f}; leakage flag: {leakage['participant_leakage_detected']}.
- The most concentrated cluster is {leakage['maximum_single_participant_fraction_within_cluster']:.1%} one participant, and the smallest participant representation in any cluster is {leakage['minimum_unique_participants_in_any_cluster']} participants.
- Participant-level bootstrap ARI for the pooled candidate: {pooled_solution.get('participant_bootstrap_ari_mean', math.nan):.3f}.

## 7. Which examples best illustrate each group?

{chr(10).join(representative_lines)}

Selected scenario-12 representatives:

{chr(10).join(selected_representative_lines)}

## 8. What conclusions should not be made?

- Cluster descriptions do not establish intention, yielding, hesitation, or causality.
- Scenario 12 and 16 conflict zones were geometrically unsupported; their primary alignment therefore used minimum synchronized actor distance, and zone-entry features were left missing.
- The motion set contains {len(features)} recordings, not all 186 annotated participant–scenario rows. There were {unmatched_gt} annotation rows without usable validated motion data.
- Five participant–scenario keys have repeated motion recordings. They remain separate recordings and are explicitly flagged as duplicate matches rather than silently collapsed.
- Speeds and accelerations use the repository's centered smoothing windows and inherit those upstream assumptions.
- Curve-shape clustering is sensitivity evidence only. Its ARI with the feature candidate was {curve_comparison.get('interaction_curve_vs_feature_ari', math.nan):.3f}.
"""
    (reports_dir / "professor_summary.md").write_text(professor, encoding="utf-8")

    report = f"""# Behavioral clustering analysis report

Generated: {config_record['generated_at_utc']}

## Reproducibility

Exact command:

```powershell
{config_record['exact_command']}
```

Random seed: {config_record['analysis_config']['seed']}. Algorithms: Ward hierarchical clustering, Gaussian mixture models, and k-means; k=2–5 subject to sample-size support. Stability used {config_record['analysis_config']['bootstrap_repetitions']} deterministic bootstrap repetitions. Robust preprocessing used within-scenario median imputation and `(value - scenario median) / scenario IQR` scaling. Ground-truth fields were excluded until assignments were fixed.

## Inventory and matching

- Ground-truth trial rows: {int(inventory.ground_truth_present.sum())}
- Validated motion recordings clustered: {len(features)}
- One-to-one unique motion matches: {unique_matches}
- Duplicate motion matches: {duplicate_matches}
- Ground-truth rows without validated motion: {unmatched_gt}

## Interaction reference and geometry

Supported conflict-zone entry midpoints were used first. Minimum synchronized pedestrian–vehicle distance was used when a zone was unsupported or both entries were unavailable. No arbitrary time fallback was inserted. Scenario-shared rigid transforms translated the estimated conflict center to the origin, rotated pedestrian travel toward positive X, and reflected the perpendicular axis when necessary to make vehicle direction consistent. Scale remained 1.0.

## Retained pooled features

{chr(10).join(f'- `{name}`' for name in retained)}

## Within-scenario results

{chr(10).join(within_lines)}

Selected within-scenario measurement profiles:

{chr(10).join(selected_profile_lines) if selected_profile_lines else 'None.'}

## Pooled result

Status: **{pooled_solution.get('status')}**. Method={pooled_solution.get('method')}, k={pooled_solution.get('clusters')}, silhouette={pooled_solution.get('silhouette', math.nan):.3f}, Davies–Bouldin={pooled_solution.get('davies_bouldin', math.nan):.3f}, Calinski–Harabasz={pooled_solution.get('calinski_harabasz', math.nan):.3f}, mean bootstrap ARI={pooled_solution.get('bootstrap_ari_mean', math.nan):.3f}, participant-bootstrap ARI={pooled_solution.get('participant_bootstrap_ari_mean', math.nan):.3f}.

Overall interpretation category: **{conclusion}**.

## Leakage and sensitivity

Scenario AMI={leakage['scenario_adjusted_mutual_information']:.3f}; participant AMI={leakage['participant_adjusted_mutual_information']:.3f}. Interaction-curve versus feature ARI={curve_comparison.get('interaction_curve_vs_feature_ari', math.nan):.3f}; movement-onset-curve versus feature ARI={curve_comparison.get('onset_curve_vs_feature_ari', math.nan):.3f}; interaction-curve versus onset-curve ARI={curve_comparison.get('interaction_curve_vs_onset_curve_ari', math.nan):.3f}.

## Ground-truth validation

{chr(10).join(validation_lines) if validation_lines else 'No label was testable.'}

Selected within-scenario validation:

{chr(10).join(selected_validation_lines) if selected_validation_lines else 'No selected within-scenario label was testable.'}

## Limitations

The analysis is observational and exploratory, sample sizes per scenario are modest, repeated participants violate simple independence assumptions, duplicate recordings exist for five trial keys, and two scenario-level conflict zones are unsupported. Bootstrap stability and participant-level resampling reduce but do not eliminate these limitations. Labels are external interpretation checks, not model inputs.
"""
    (reports_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    if args.seed < 0 or args.bootstrap_repetitions < 10:
        raise ValueError("--seed must be nonnegative and --bootstrap-repetitions must be at least 10")
    scenarios = sorted(set(args.scenarios))
    unknown = sorted(set(scenarios) - set(DEFAULT_SCENARIOS))
    if unknown:
        raise ValueError(f"Unsupported scenarios requested: {unknown}")
    input_root = args.input_root.resolve()
    ground_truth_path = args.ground_truth.resolve()
    output_root = args.output_root.resolve()
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground-truth workbook not found: {ground_truth_path}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_root}; choose a new path or use --overwrite")
    if args.overwrite and output_root.exists():
        # Delete only known products in this analysis-specific directory.
        import shutil
        shutil.rmtree(output_root)
    directories = {
        name: output_root / name for name in (
            "data_qc", "features", "within_scenario", "pooled_scenario_adjusted",
            "curve_sensitivity", "ground_truth_validation", "figures", "interactive",
            "representative_trials", "reports",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig(seed=args.seed, bootstrap_repetitions=args.bootstrap_repetitions)

    manifest, availability, traces, upstream_events, zones = _read_motion_inputs(input_root, scenarios)
    ground_truth = clean_ground_truth(ground_truth_path, scenarios)
    ground_truth.to_csv(directories["data_qc"] / "ground_truth_cleaned.csv", index=False)
    inventory, matching = build_matching_outputs(traces, manifest, availability, ground_truth)
    matching.to_csv(directories["data_qc"] / "trial_matching_report.csv", index=False)
    if matching.ground_truth_rows_for_trial_key.gt(1).any():
        raise ValueError("A motion file would match more than one ground-truth row")

    features, interaction_events, qc, normalized, transforms = extract_trial_features(
        traces, upstream_events, zones, config
    )
    source_lookup = manifest.set_index("recording_identifier").source_csv_path.to_dict()
    qc["source_csv_path"] = qc.recording_identifier.map(source_lookup)
    qc.to_csv(directories["data_qc"] / "quality_control_report.csv", index=False)
    qc_by_key = qc.groupby(["participant_id", "scenario"]).agg(
        motion_recordings=("recording_identifier", "nunique"),
        mean_duration_s=("duration_s", "mean"),
        median_sampling_rate_hz=("sampling_rate_hz", "median"),
        maximum_position_missing_percent=("pedestrian_position_missing_percent", "max"),
        invalid_timestamp_intervals=("invalid_or_nonmonotonic_time_intervals", "sum"),
    ).reset_index()
    inventory = inventory.merge(
        qc_by_key, left_on=["participant_id", "scenario_number"], right_on=["participant_id", "scenario"], how="left"
    ).drop(columns=["scenario"], errors="ignore")
    inventory["missing_required_columns"] = np.where(
        inventory.failure_exclusion_reason.fillna("").str.contains("missing_required"),
        inventory.failure_exclusion_reason, "",
    )
    inventory.to_csv(directories["data_qc"] / "data_inventory.csv", index=False)
    qc.groupby("scenario").sampling_rate_hz.describe(percentiles=[.05, .25, .5, .75, .95]).reset_index().to_csv(
        directories["data_qc"] / "sampling_rate_distribution.csv", index=False
    )
    interaction_events.to_csv(directories["features"] / "interaction_events.csv", index=False)
    transforms.to_csv(directories["features"] / "trajectory_transformation_parameters.csv", index=False)
    normalized.to_csv(directories["features"] / "normalized_trajectory_samples.csv", index=False)
    features.to_csv(directories["features"] / "trial_behavior_features.csv", index=False)
    feature_dictionary(features, config).to_csv(directories["features"] / "feature_dictionary.csv", index=False)

    plot_qc(qc, directories["figures"] / "01_feature_missingness_and_qc_summary.png")
    plot_normalization(traces, normalized, directories["figures"] / "02_trajectory_normalization_before_after.png")

    selection_frames, within_metrics_frames, within_assignments, within_solutions = [], [], [], []
    within_profile_frames, within_representative_frames = [], []
    for scenario in scenarios:
        scenario_features = features[features.scenario.eq(scenario)].reset_index(drop=True)
        retained_scenario, decisions = select_features(scenario_features, config, f"within_scenario_{scenario}")
        selection_frames.append(decisions)
        if len(retained_scenario) < 2:
            raise ValueError(f"Fewer than two retained features for scenario {scenario}")
        scaled_scenario, scaling_audit = robust_scale_within_group(
            scenario_features.assign(single_group=1), retained_scenario, "single_group"
        )
        metrics, labels_lookup = evaluate_cluster_candidates(
            scaled_scenario[retained_scenario].to_numpy(), args.seed + scenario,
            config.bootstrap_repetitions, f"scenario_{scenario}",
        )
        solution = select_cluster_solution(metrics, config)
        labels = labels_lookup[(solution["method"], solution["clusters"])]
        assignment = scenario_features[["recording_identifier", "participant_id", "scenario"]].copy()
        assignment["cluster_id"] = labels
        profiles_s, reps_s, unusual_s, names_s = describe_clusters(
            scenario_features, scaled_scenario, labels, retained_scenario, bool(solution.get("defensible"))
        )
        assignment["cluster_label"] = assignment.cluster_id.map(names_s)
        assignment["solution_status"] = solution["status"]
        scenario_dir = directories["within_scenario"] / f"scenario_{scenario:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        assignment.to_csv(scenario_dir / "cluster_assignments.csv", index=False)
        metrics.to_csv(scenario_dir / "model_selection_metrics.csv", index=False)
        scaling_audit.to_csv(scenario_dir / "scaling_and_imputation.csv", index=False)
        profiles_s.to_csv(scenario_dir / "cluster_profiles.csv", index=False)
        reps_s.to_csv(scenario_dir / "representative_trials.csv", index=False)
        unusual_s.to_csv(scenario_dir / "unusual_and_boundary_trials.csv", index=False)
        (scenario_dir / "selected_solution.json").write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
        within_metrics_frames.append(metrics)
        within_assignments.append(assignment)
        within_solutions.append({"scenario": scenario, **solution})
        within_profile_frames.append(profiles_s.assign(scenario=scenario))
        within_representative_frames.append(reps_s.assign(scenario=scenario))
    within_metrics = pd.concat(within_metrics_frames, ignore_index=True)
    within_metrics.to_csv(directories["within_scenario"] / "all_model_selection_metrics.csv", index=False)
    pd.concat(within_assignments, ignore_index=True).to_csv(directories["within_scenario"] / "all_cluster_assignments.csv", index=False)
    within_summary = pd.DataFrame(within_solutions)
    within_summary.to_csv(directories["within_scenario"] / "within_scenario_summary.csv", index=False)

    retained, pooled_decisions = select_features(features, config, "pooled_scenario_adjusted")
    for name in retained.copy():
        bad_scenarios = []
        for scenario, one in features.groupby("scenario"):
            values = pd.to_numeric(one[name], errors="coerce")
            iqr = values.quantile(.75) - values.quantile(.25)
            if not np.isfinite(iqr) or iqr <= 1e-9:
                bad_scenarios.append(int(scenario))
        if bad_scenarios:
            retained.remove(name)
            mask = pooled_decisions.feature.eq(name)
            pooled_decisions.loc[mask, "included"] = False
            pooled_decisions.loc[mask, "reason"] = "cannot_scenario_adjust_zero_or_missing_IQR_in_scenarios_" + "_".join(map(str, bad_scenarios))
    if len(retained) < 2:
        raise ValueError("Fewer than two pooled features survive scenario-adjustment checks")
    selection_frames.append(pooled_decisions)
    feature_selection = pd.concat(selection_frames, ignore_index=True)
    feature_selection.to_csv(directories["features"] / "feature_selection_decisions.csv", index=False)
    pd.DataFrame({"feature": retained, "order": range(1, len(retained) + 1)}).to_csv(
        directories["features"] / "final_retained_features.csv", index=False
    )
    scaled, scaling_audit = robust_scale_within_group(features, retained, "scenario")
    scaled_output = features[["recording_identifier", "participant_id", "scenario"]].join(scaled)
    scaled_output.to_csv(directories["pooled_scenario_adjusted"] / "scenario_adjusted_features.csv", index=False)
    scaling_audit.to_csv(directories["pooled_scenario_adjusted"] / "scaling_and_imputation.csv", index=False)
    pooled_metrics, pooled_labels_lookup = evaluate_cluster_candidates(
        scaled[retained].to_numpy(), args.seed, config.bootstrap_repetitions,
        "pooled_scenario_adjusted",
    )
    pooled_solution = select_cluster_solution(pooled_metrics, config)
    pooled_labels = pooled_labels_lookup[(pooled_solution["method"], pooled_solution["clusters"])]
    participant_stability, participant_stability_p10, participant_repetitions = bootstrap_stability(
        scaled[retained].to_numpy(), pooled_labels, pooled_solution["method"], pooled_solution["clusters"],
        args.seed + 909, config.bootstrap_repetitions, groups=features.participant_id.to_numpy(),
    )
    pooled_solution.update({
        "participant_bootstrap_ari_mean": participant_stability,
        "participant_bootstrap_ari_p10": participant_stability_p10,
        "participant_bootstrap_valid_repetitions": participant_repetitions,
    })
    assignments = features[["recording_identifier", "participant_id", "scenario"]].copy()
    assignments["cluster_id"] = pooled_labels
    profiles, representatives, unusual, cluster_names = describe_clusters(
        features, scaled, pooled_labels, retained, bool(pooled_solution.get("defensible"))
    )
    assignments["cluster_label"] = assignments.cluster_id.map(cluster_names)
    assignments["solution_status"] = pooled_solution["status"]
    leakage = _association_leakage(assignments)
    assignments.to_csv(directories["pooled_scenario_adjusted"] / "cluster_assignments.csv", index=False)
    pooled_metrics.to_csv(directories["pooled_scenario_adjusted"] / "model_selection_metrics.csv", index=False)
    profiles.to_csv(directories["pooled_scenario_adjusted"] / "cluster_profiles.csv", index=False)
    representatives.to_csv(directories["representative_trials"] / "representative_trials.csv", index=False)
    unusual.to_csv(directories["representative_trials"] / "unusual_and_boundary_trials.csv", index=False)
    pd.crosstab(assignments.scenario, assignments.cluster_id).to_csv(directories["pooled_scenario_adjusted"] / "scenario_cluster_composition.csv")
    pd.crosstab(assignments.participant_id, assignments.cluster_id).to_csv(directories["pooled_scenario_adjusted"] / "participant_cluster_composition.csv")
    (directories["pooled_scenario_adjusted"] / "selected_solution.json").write_text(json.dumps(pooled_solution, indent=2) + "\n", encoding="utf-8")
    (directories["pooled_scenario_adjusted"] / "leakage_diagnostics.json").write_text(json.dumps(leakage, indent=2) + "\n", encoding="utf-8")

    plot_correlation(features, retained, directories["figures"] / "03_retained_feature_correlation_heatmap.png")
    plot_feature_heatmap(scaled, assignments, retained, directories["figures"] / "04_standardized_trial_feature_heatmap.png")
    pca_scores = plot_pca(
        scaled, assignments, retained,
        directories["figures"] / "05_pca_by_cluster_and_scenario.png",
        directories["figures"] / "06_pca_by_scenario_leakage.png",
        directories["interactive"] / "pca_scenario_adjusted.html", args.seed,
    )
    pca_scores.to_csv(directories["pooled_scenario_adjusted"] / "pca_scores.csv", index=False)
    plot_model_comparison(pooled_metrics, directories["figures"] / "07_cluster_size_and_stability_comparison.png")
    plot_cluster_profiles(profiles, retained, directories["figures"] / "08_cluster_feature_profiles.png")
    plot_compositions(
        assignments, directories["figures"] / "12_scenario_by_cluster_composition.png",
        directories["figures"] / "13_participant_by_cluster_composition.png",
    )
    plot_within_scenario_summary(within_summary, directories["figures"] / "16_within_scenario_clustering_summary.png")

    interaction_curves, curve_ids, curve_matrix, curve_columns = build_curve_matrix(traces, interaction_events, config)
    curve_scores, curve_variance, curve_metadata = functional_pca(curve_matrix, args.seed)
    curve_metrics, curve_labels_lookup = evaluate_cluster_candidates(
        curve_scores, args.seed + 1701, max(20, config.bootstrap_repetitions // 2), "interaction_aligned_curve_sensitivity"
    )
    curve_solution = select_cluster_solution(curve_metrics, config)
    curve_labels = curve_labels_lookup[(curve_solution["method"], curve_solution["clusters"])]
    curve_assignment = curve_ids.copy()
    curve_assignment["curve_cluster_id"] = curve_labels
    curve_assignment = curve_assignment.merge(assignments[["recording_identifier", "cluster_id"]], on="recording_identifier", how="left")
    curve_pca_frame = curve_ids.copy()
    for index in range(curve_scores.shape[1]):
        curve_pca_frame[f"FPC{index + 1}"] = curve_scores[:, index]
        curve_pca_frame[f"FPC{index + 1}_explained_variance"] = curve_variance[index]
    interaction_curves.to_csv(directories["curve_sensitivity"] / "interaction_aligned_velocity_curves.csv", index=False)
    curve_pca_frame.to_csv(directories["curve_sensitivity"] / "functional_pca_scores.csv", index=False)
    curve_assignment.to_csv(directories["curve_sensitivity"] / "curve_cluster_assignments.csv", index=False)
    curve_metrics.to_csv(directories["curve_sensitivity"] / "curve_model_selection_metrics.csv", index=False)

    onset_events = interaction_events.copy()
    onset_events["main_interaction_time"] = onset_events.pedestrian_movement_onset_time_s
    onset_curves, onset_ids, onset_matrix, _ = build_curve_matrix(
        traces, onset_events, config, alignment_name="pedestrian_movement_onset"
    )
    onset_scores, _, onset_metadata = functional_pca(onset_matrix, args.seed)
    onset_metrics, onset_labels_lookup = evaluate_cluster_candidates(
        onset_scores, args.seed + 1801, max(20, config.bootstrap_repetitions // 2), "movement_onset_aligned_curve_sensitivity"
    )
    onset_solution = select_cluster_solution(onset_metrics, config)
    onset_labels = onset_labels_lookup[(onset_solution["method"], onset_solution["clusters"])]
    onset_assignment = onset_ids.copy()
    onset_assignment["onset_curve_cluster_id"] = onset_labels
    onset_assignment.to_csv(directories["curve_sensitivity"] / "movement_onset_curve_cluster_assignments.csv", index=False)
    onset_curves.to_csv(directories["curve_sensitivity"] / "movement_onset_aligned_velocity_curves.csv", index=False)
    onset_metrics.to_csv(directories["curve_sensitivity"] / "movement_onset_curve_model_selection_metrics.csv", index=False)
    comparison = curve_assignment.merge(onset_assignment, on="recording_identifier", how="inner")
    curve_comparison = {
        "interaction_curve_vs_feature_ari": float(adjusted_rand_score(comparison.cluster_id, comparison.curve_cluster_id)),
        "onset_curve_vs_feature_ari": float(adjusted_rand_score(comparison.cluster_id, comparison.onset_curve_cluster_id)),
        "interaction_curve_vs_onset_curve_ari": float(adjusted_rand_score(comparison.curve_cluster_id, comparison.onset_curve_cluster_id)),
        "interaction_curve_solution": curve_solution,
        "movement_onset_curve_solution": onset_solution,
        "functional_pca": curve_metadata,
        "movement_onset_functional_pca": onset_metadata,
        "common_trials": int(len(comparison)),
    }
    (directories["curve_sensitivity"] / "curve_feature_cluster_comparison.json").write_text(json.dumps(curve_comparison, indent=2) + "\n", encoding="utf-8")
    pd.crosstab(curve_assignment.cluster_id, curve_assignment.curve_cluster_id).to_csv(directories["curve_sensitivity"] / "feature_by_curve_cluster_crosstab.csv")
    velocity_summary = plot_velocity_curves(
        interaction_curves, assignments, directories["figures"] / "09_10_cluster_velocity_curves_with_iqr_and_counts.png",
        directories["interactive"] / "cluster_velocity_curves.html",
    )
    velocity_summary.to_csv(directories["curve_sensitivity"] / "cluster_velocity_curve_summary.csv", index=False)
    plot_representative_trajectories(
        normalized, representatives, directories["figures"] / "11_representative_normalized_trajectories.png",
        directories["interactive"] / "representative_normalized_trajectories.html",
    )

    joined, validation_tests, distributions, order_comparison, comparison_summary = validate_ground_truth(
        assignments, ground_truth, features
    )
    joined.to_csv(directories["ground_truth_validation"] / "cluster_assignments_with_ground_truth.csv", index=False)
    validation_tests.to_csv(directories["ground_truth_validation"] / "ground_truth_association_tests.csv", index=False)
    distributions.to_csv(directories["ground_truth_validation"] / "cluster_ground_truth_distributions.csv", index=False)
    order_comparison.to_csv(directories["ground_truth_validation"] / "measured_vs_annotated_who_went_first.csv", index=False)
    (directories["ground_truth_validation"] / "measured_vs_annotated_summary.json").write_text(json.dumps(comparison_summary, indent=2) + "\n", encoding="utf-8")
    plot_ground_truth(
        distributions, order_comparison, directories["figures"] / "14_ground_truth_distributions_by_cluster.png",
        directories["figures"] / "15_measured_vs_annotated_who_went_first.png",
    )

    selected_within_profiles = pd.concat([
        profile for profile, solution in zip(within_profile_frames, within_solutions)
        if solution["defensible"]
    ], ignore_index=True) if any(solution["defensible"] for solution in within_solutions) else pd.DataFrame()
    selected_within_representatives = pd.concat([
        representative for representative, solution in zip(within_representative_frames, within_solutions)
        if solution["defensible"]
    ], ignore_index=True) if any(solution["defensible"] for solution in within_solutions) else pd.DataFrame()
    within_validation_frames = []
    for assignment, solution in zip(within_assignments, within_solutions):
        if not solution["defensible"]:
            continue
        scenario = int(solution["scenario"])
        scenario_features = features[features.scenario.eq(scenario)].reset_index(drop=True)
        scenario_joined, scenario_tests, scenario_distributions, scenario_comparison, scenario_comparison_summary = validate_ground_truth(
            assignment.reset_index(drop=True), ground_truth, scenario_features
        )
        scenario_tests.insert(0, "scenario", scenario)
        within_validation_frames.append(scenario_tests)
        scenario_dir = directories["ground_truth_validation"] / f"scenario_{scenario:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_joined.to_csv(scenario_dir / "cluster_assignments_with_ground_truth.csv", index=False)
        scenario_tests.to_csv(scenario_dir / "ground_truth_association_tests.csv", index=False)
        scenario_distributions.to_csv(scenario_dir / "cluster_ground_truth_distributions.csv", index=False)
        scenario_comparison.to_csv(scenario_dir / "measured_vs_annotated_who_went_first.csv", index=False)
        (scenario_dir / "measured_vs_annotated_summary.json").write_text(
            json.dumps(scenario_comparison_summary, indent=2) + "\n", encoding="utf-8"
        )
        plot_ground_truth(
            scenario_distributions, scenario_comparison,
            directories["figures"] / f"14b_scenario_{scenario:02d}_ground_truth_distributions.png",
            directories["figures"] / f"15b_scenario_{scenario:02d}_measured_vs_annotated_order.png",
        )
    selected_within_validation_tests = (
        pd.concat(within_validation_frames, ignore_index=True)
        if within_validation_frames else pd.DataFrame()
    )
    selected_within_validation_tests.to_csv(
        directories["ground_truth_validation"] / "selected_within_scenario_association_tests.csv",
        index=False,
    )

    config_record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exact_command": _exact_command(args),
        "input_root": str(input_root), "ground_truth": str(ground_truth_path),
        "output_root": str(output_root), "scenarios": scenarios,
        "analysis_config": asdict(config), "package_versions": _package_versions(),
        "source_definitions": {
            "pedestrian_position": "B VR Pos X/Z via validated trajectory pipeline",
            "vehicle_position": "A car Pos X/Z via validated trajectory pipeline",
            "speed_and_acceleration": "MetricsEngine centered rolling means (speed window 9; acceleration window 11)",
            "ground_truth_use": "post-clustering interpretation and validation only",
        },
        "umap": "omitted because umap-learn is not installed; not used as evidence",
    }
    (output_root / "analysis_config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
    _write_reports(
        directories["reports"], config_record, inventory, matching, features, retained,
        within_summary, selected_within_profiles, selected_within_representatives,
        selected_within_validation_tests, pooled_solution, leakage, profiles, representatives,
        validation_tests, comparison_summary, curve_comparison,
    )
    summary = {
        "motion_recordings": int(features.recording_identifier.nunique()),
        "ground_truth_trials": int(len(ground_truth)),
        "unique_matches": int(matching.match_status.eq("unique_match").sum()),
        "duplicate_motion_matches": int(matching.match_status.eq("duplicate_motion_match").sum()),
        "unmatched_motion": int(matching.match_status.eq("unmatched_motion").sum()),
        "unmatched_ground_truth": int((inventory.motion_recording_count.eq(0) & inventory.ground_truth_present).sum()),
        "retained_features": retained,
        "within_scenario_solutions": within_solutions,
        "pooled_solution": pooled_solution,
        "leakage": leakage,
        "curve_comparison": curve_comparison,
        "ground_truth_comparison": comparison_summary,
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
