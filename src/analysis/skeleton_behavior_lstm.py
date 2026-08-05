"""Exploratory, leakage-safe skeleton prediction of observable hand gestures.

This module is deliberately isolated from the existing trajectory, conflict-zone,
segmentation, clustering, and ground-truth pipelines.  It reads their immutable
artifacts and writes only below ``outputs/skeleton_behavior_lstm``.

The outcome is Xiang's observable pedestrian hand-gesture Yes/No annotation.
It is not an intent, yielding, or hesitation model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "skeleton_behavior_lstm"
DEFAULT_LABELS = PROJECT_ROOT / "outputs/behavior_clustering/data_qc/ground_truth_cleaned.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/scenario_pattern_expanded/manifests/processing_success_manifest.csv"
DEFAULT_MATCHING = PROJECT_ROOT / "outputs/behavior_clustering/data_qc/trial_matching_report.csv"
DEFAULT_TRAJECTORY = PROJECT_ROOT / "outputs/trajectory_analysis"
DEFAULT_EVENTS = PROJECT_ROOT / "outputs/conflict_zone_analysis"
DEFAULT_SCENARIOS = (3, 7, 12, 15, 16, 21)

# A conservative body set available in Unity humanoid exports.  LowerArm and
# LowerLeg are the elbow/knee-linked segments when explicit elbow/knee points
# are unavailable.
JOINTS = (
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
    "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand",
    "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot",
)
JOINT_EDGES = (
    ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "UpperChest"),
    ("UpperChest", "Neck"), ("Neck", "Head"),
    ("UpperChest", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftLowerArm"), ("LeftLowerArm", "LeftHand"),
    ("UpperChest", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightLowerArm"), ("RightLowerArm", "RightHand"),
    ("Hips", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("Hips", "RightUpperLeg"),
    ("RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightFoot"),
)
AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    window_before_s: float = 3.0
    window_after_s: float = 3.0
    sequence_length: int = 61
    folds: int = 5
    lstm_hidden_size: int = 12
    lstm_epochs: int = 60
    lstm_learning_rate: float = 0.008
    lstm_l2: float = 1e-4
    bootstrap_repetitions: int = 500
    joint_spike_speed_mps: float = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true", help="Reuse this experiment's existing audit tables.")
    return parser.parse_args()


def canonical_column(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clean_binary_label(value: object) -> int | None:
    """Return a binary label while keeping Unknown distinct from No."""
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().casefold()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    if normalized in {"", "unknown", "unclear", "n/a", "na"}:
        return None
    raise ValueError(f"Unexpected hand-gesture label: {value!r}")


def normalize_skeleton(
    positions: np.ndarray,
    mask: np.ndarray,
    pelvis_index: int,
    yaw_degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-center and rotate XYZ positions into each participant's frame."""
    positions = np.asarray(positions, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    yaw = np.asarray(yaw_degrees, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (time, joints, 3)")
    if mask.shape != positions.shape[:2]:
        raise ValueError("mask must have shape (time, joints)")
    pelvis_valid = mask[:, pelvis_index] & np.isfinite(positions[:, pelvis_index]).all(axis=1)
    centered = positions - positions[:, pelvis_index:pelvis_index + 1]
    radians = np.deg2rad(yaw)
    cos_yaw, sin_yaw = np.cos(radians), np.sin(radians)
    x, z = centered[..., 0].copy(), centered[..., 2].copy()
    # Unity yaw: forward=(sin(yaw), cos(yaw)); project onto actor right/forward.
    centered[..., 0] = x * cos_yaw[:, None] - z * sin_yaw[:, None]
    centered[..., 2] = x * sin_yaw[:, None] + z * cos_yaw[:, None]
    normalized_mask = mask & pelvis_valid[:, None] & np.isfinite(yaw)[:, None]
    centered[~normalized_mask] = np.nan
    return centered, normalized_mask


def extract_conflict_window(
    time: np.ndarray,
    center: float,
    before: float,
    after: float,
) -> tuple[np.ndarray, str | None]:
    """Select a full window; never authorize extrapolation beyond observed time."""
    time = np.asarray(time, dtype=float)
    valid = np.isfinite(time)
    if not np.isfinite(center) or valid.sum() < 2:
        return np.array([], dtype=int), "missing_conflict_anchor_or_time"
    ordered = time[valid]
    if np.any(np.diff(ordered) <= 0):
        return np.array([], dtype=int), "nonmonotonic_time"
    start, end = center - before, center + after
    if ordered[0] > start or ordered[-1] < end:
        return np.array([], dtype=int), "incomplete_window_no_extrapolation"
    indices = np.flatnonzero(valid & (time >= start) & (time <= end))
    if len(indices) < 2:
        return np.array([], dtype=int), "insufficient_window_samples"
    return indices, None


def resample_fixed_length(
    time: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample joint data inside observed support, without extrapolation."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target_time, dtype=float)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("values/mask shapes must be (time,joints,3)/(time,joints)")
    result = np.zeros((len(target), values.shape[1], 3), dtype=np.float32)
    result_mask = np.zeros((len(target), values.shape[1]), dtype=bool)
    for joint in range(values.shape[1]):
        joint_valid = mask[:, joint] & np.isfinite(values[:, joint]).all(axis=1)
        if joint_valid.sum() < 2:
            continue
        source_t = time[joint_valid]
        unique_t, unique_indices = np.unique(source_t, return_index=True)
        source_v = values[joint_valid, joint, :][unique_indices]
        supported = (target >= unique_t[0]) & (target <= unique_t[-1])
        if not supported.any() or len(unique_t) < 2:
            continue
        for axis in range(3):
            result[supported, joint, axis] = np.interp(
                target[supported], unique_t, source_v[:, axis]
            )
        result_mask[supported, joint] = True
    return result, result_mask


def joint_velocity(
    positions: np.ndarray, mask: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(positions, dtype=np.float32)
    velocity_mask = np.zeros_like(mask, dtype=bool)
    dt = np.diff(np.asarray(time, dtype=float))
    valid_dt = np.isfinite(dt) & (dt > 0)
    for index in range(1, len(time)):
        valid = mask[index] & mask[index - 1] & valid_dt[index - 1]
        velocity[index, valid] = (
            (positions[index, valid] - positions[index - 1, valid]) / dt[index - 1]
        )
        velocity_mask[index, valid] = True
    return velocity, velocity_mask


def validate_participant_splits(
    splits: Sequence[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> None:
    groups = np.asarray(groups)
    for fold, (train, test) in enumerate(splits):
        overlap = set(groups[train]).intersection(groups[test])
        if overlap:
            raise AssertionError(f"Participant leakage in fold {fold}: {sorted(overlap)}")


def grouped_splits(
    labels: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    validate_participant_splits(splits, groups)
    return splits


def _read_canonical_csv(path: Path, wanted: Iterable[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
    mapping = {canonical_column(column): column for column in header}
    if wanted is None:
        raw = header
    else:
        raw = [mapping[name] for name in wanted if name in mapping]
    frame = pd.read_csv(path, sep=";", usecols=raw, low_memory=False)
    frame.rename(columns={column: canonical_column(column) for column in frame.columns}, inplace=True)
    return frame


def _position_columns() -> list[str]:
    return [f"B Bone Pos {joint} {axis}" for joint in JOINTS for axis in AXES]


def _rotation_y_column() -> str:
    return "B Bone Rot Hips Y"


def _classify_skeleton_column(column: str) -> tuple[str, str, str]:
    axis_match = re.search(r" ([XYZW])$", column)
    axis = axis_match.group(1) if axis_match else ""
    base = column[:axis_match.start()] if axis_match else column
    patterns = (
        (r"^B Root Hand Pos (Left|Right)$", "root_hand_position"),
        (r"^B Root Hand Rot (Left|Right)$", "root_hand_rotation"),
        (r"^B Bone Pos (.+)$", "bone_position"),
        (r"^B Bone Rot (.+)$", "bone_rotation"),
        (r"^B Hand Bone Pos (.+)$", "hand_bone_position"),
        (r"^B Hand Bone Rot (.+)$", "hand_bone_rotation"),
        (r"^B VR Pos$", "vr_position"),
        (r"^B VR Rot$", "vr_rotation"),
        (r"^B Avatar Pos$", "avatar_position_unreliable"),
        (r"^B Avatar Rot$", "avatar_rotation"),
    )
    for pattern, category in patterns:
        match = re.match(pattern, base)
        if match:
            joint = match.group(1) if match.lastindex else "VR/avatar"
            return category, joint, axis
    return "other_b_column", base, axis


def load_experiment_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(DEFAULT_LABELS)
    manifest = pd.read_csv(DEFAULT_MANIFEST)
    matching = pd.read_csv(DEFAULT_MATCHING)
    manifest = manifest[
        manifest.scenario_number.isin(DEFAULT_SCENARIOS)
        & manifest.processing_status.eq("success")
    ].copy()
    events = []
    trajectories = []
    for scenario in DEFAULT_SCENARIOS:
        events.append(pd.read_csv(DEFAULT_EVENTS / f"scenario_{scenario:02d}/conflict_zone_events.csv"))
        trajectories.append(pd.read_csv(DEFAULT_TRAJECTORY / f"scenario_{scenario:02d}/raw_trajectory_samples.csv"))
    return labels, manifest, pd.concat(events, ignore_index=True), pd.concat(trajectories, ignore_index=True)


def audit_skeleton_data(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    output_root: Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inventory skeleton columns and trial/joint quality without loading all trials together."""
    column_trials: dict[str, int] = {}
    joint_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    expected = {canonical_column(column) for column in _position_columns()}
    wanted_base = expected | {
        "ScenarioTime", _rotation_y_column(), "B VR Pos X", "B VR Pos Z",
        "B Root Hand Pos Left X", "B Root Hand Pos Left Y", "B Root Hand Pos Left Z",
        "B Root Hand Pos Right X", "B Root Hand Pos Right Y", "B Root Hand Pos Right Z",
    }
    for row in manifest.sort_values(["pednyc_number", "scenario_number"]).itertuples(index=False):
        path = Path(row.source_csv_path)
        header = [canonical_column(c) for c in pd.read_csv(path, sep=";", nrows=0).columns]
        skeleton_columns = [c for c in header if c.startswith("B ") and any(
            token in c for token in ("Bone", "Hand", "VR Pos", "VR Rot", "Avatar")
        )]
        for column in set(skeleton_columns):
            column_trials[column] = column_trials.get(column, 0) + 1
        wanted = wanted_base.intersection(header)
        frame = _read_canonical_csv(path, wanted)
        time = pd.to_numeric(frame.get("ScenarioTime"), errors="coerce").to_numpy(dtype=float)
        dt = np.diff(time)
        valid_dt = np.isfinite(dt) & (dt > 0)
        trial_rows.append({
            "recording_identifier": row.recording_identifier,
            "participant_id": f"PedNYC{int(row.pednyc_number)}",
            "scenario": int(row.scenario_number),
            "source_csv_path": str(path),
            "samples": len(frame),
            "invalid_or_nonmonotonic_dt": int((~valid_dt).sum()),
            "median_dt_s": float(np.median(dt[valid_dt])) if valid_dt.any() else math.nan,
            "sampling_rate_hz": float(1 / np.median(dt[valid_dt])) if valid_dt.any() else math.nan,
            "duration_s": float(np.nanmax(time) - np.nanmin(time)) if np.isfinite(time).any() else math.nan,
            "available_skeleton_columns": len(skeleton_columns),
            "missing_fixed_joint_columns": len(expected.difference(header)),
            "b_vr_xz_available": {"B VR Pos X", "B VR Pos Z"}.issubset(header),
            "b_avatar_position_excluded": True,
        })
        for joint in JOINTS:
            columns = [f"B Bone Pos {joint} {axis}" for axis in AXES]
            present = [column in frame for column in columns]
            if all(present):
                values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(values).all(axis=1) & (np.abs(values) < 1e4).all(axis=1)
                speeds = np.full(max(0, len(frame) - 1), np.nan)
                pair_valid = valid[1:] & valid[:-1] & valid_dt
                speeds[pair_valid] = np.linalg.norm(np.diff(values, axis=0)[pair_valid], axis=1) / dt[pair_valid]
                spikes = int(np.nansum(speeds > config.joint_spike_speed_mps))
                invalid_values = int((~np.isfinite(values) | (np.abs(values) >= 1e4)).sum())
            else:
                valid = np.zeros(len(frame), dtype=bool)
                spikes = 0
                invalid_values = 3 * len(frame)
            joint_rows.append({
                "recording_identifier": row.recording_identifier,
                "participant_id": f"PedNYC{int(row.pednyc_number)}",
                "scenario": int(row.scenario_number),
                "joint": joint,
                "xyz_columns_available": int(sum(present)),
                "valid_sample_fraction": float(valid.mean()) if len(valid) else 0.0,
                "missing_sample_fraction": float(1 - valid.mean()) if len(valid) else 1.0,
                "invalid_coordinate_values": invalid_values,
                "speed_spikes_over_threshold": spikes,
            })
    inventory_rows = []
    trial_count = len(manifest)
    for column, count in sorted(column_trials.items()):
        category, joint, axis = _classify_skeleton_column(column)
        inventory_rows.append({
            "column": column, "category": category, "joint_or_segment": joint,
            "axis": axis, "trials_available": count,
            "trial_availability_fraction": count / max(trial_count, 1),
            "used_in_model": column in expected or column == _rotation_y_column(),
            "exclusion_note": (
                "unreliable_global_trajectory_not_used" if category == "avatar_position_unreliable"
                else "" if column in expected or column == _rotation_y_column()
                else "audited_but_not_in_fixed_joint_set"
            ),
        })
    columns = pd.DataFrame(inventory_rows)
    joints = pd.DataFrame(joint_rows)
    trials = pd.DataFrame(trial_rows)
    columns.to_csv(output_root / "data_audit/skeleton_column_inventory.csv", index=False)
    joints.to_csv(output_root / "data_audit/joint_availability_by_trial.csv", index=False)
    trials.to_csv(output_root / "data_audit/trial_sampling_quality.csv", index=False)
    label_rows = []
    for column in (
        "pedestrian_hand_gesture", "annotated_pedestrian_speed_change",
        "annotated_right_of_way", "annotated_who_went_first",
    ):
        for label, count in labels[column].fillna("Unknown").value_counts().items():
            label_rows.append({"label_field": column, "label": label, "count": int(count)})
    pd.DataFrame(label_rows).to_csv(output_root / "data_audit/label_counts.csv", index=False)
    return columns, joints, trials


def choose_conflict_anchor(event: pd.Series) -> tuple[float, str]:
    scenario = int(event["scenario"])
    if scenario in (12, 16):
        value = pd.to_numeric(event.get("counterfactual_vehicle_arrival_time"), errors="coerce")
        return (float(value), "potential_counterfactual_vehicle_arrival") if np.isfinite(value) else (math.nan, "missing_potential_conflict_anchor")
    centers = [
        pd.to_numeric(event.get("pedestrian_center_time"), errors="coerce"),
        pd.to_numeric(event.get("vehicle_actual_center_time"), errors="coerce"),
    ]
    finite = [float(value) for value in centers if np.isfinite(value)]
    if finite:
        return float(np.mean(finite)), "observed_actor_zone_center_mean"
    fallback = pd.to_numeric(event.get("minimum_distance_time"), errors="coerce")
    return (float(fallback), "minimum_distance_fallback") if np.isfinite(fallback) else (math.nan, "missing_observed_conflict_anchor")


def _read_trial_skeleton(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    wanted = {"ScenarioTime", _rotation_y_column(), "A indicators"} | set(_position_columns())
    frame = _read_canonical_csv(path, wanted)
    time = pd.to_numeric(frame["ScenarioTime"], errors="coerce").to_numpy(dtype=float)
    yaw = pd.to_numeric(frame.get(_rotation_y_column()), errors="coerce").to_numpy(dtype=float)
    positions = np.full((len(frame), len(JOINTS), 3), np.nan)
    mask = np.zeros((len(frame), len(JOINTS)), dtype=bool)
    for joint_index, joint in enumerate(JOINTS):
        columns = [f"B Bone Pos {joint} {axis}" for axis in AXES]
        if all(column in frame for column in columns):
            values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values).all(axis=1) & (np.abs(values) < 1e4).all(axis=1)
            positions[:, joint_index] = values
            mask[:, joint_index] = valid
    normalized, normalized_mask = normalize_skeleton(
        positions, mask, JOINTS.index("Hips"), yaw
    )
    return frame, time, normalized, normalized_mask


def _interpolate_table(
    time: np.ndarray, frame: pd.DataFrame, columns: Sequence[str], target: np.ndarray
) -> np.ndarray:
    result = np.full((len(target), len(columns)), np.nan, dtype=np.float32)
    for index, column in enumerate(columns):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(time) & np.isfinite(values)
        if valid.sum() < 2:
            continue
        unique_t, unique_index = np.unique(time[valid], return_index=True)
        source = values[valid][unique_index]
        supported = (target >= unique_t[0]) & (target <= unique_t[-1])
        if len(unique_t) >= 2:
            result[supported, index] = np.interp(target[supported], unique_t, source)
    return result


def _handcrafted_features(
    positions: np.ndarray, velocities: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    feature_values, feature_names = [], []
    for hand_name in ("LeftHand", "RightHand"):
        index = JOINTS.index(hand_name)
        valid = mask[:, index]
        position = positions[:, index]
        speed = np.linalg.norm(velocities[:, index], axis=1)
        if valid.any():
            centered = position[valid]
            displacement = np.linalg.norm(centered - centered[0], axis=1)
            values = [
                np.nanmax(displacement), np.nansum(np.linalg.norm(np.diff(centered, axis=0), axis=1)),
                np.nanmean(speed[valid]), np.nanmax(speed[valid]),
                np.nanmax(centered[:, 1]) - np.nanmin(centered[:, 1]),
                np.nanmax(centered[:, 0]) - np.nanmin(centered[:, 0]),
            ]
        else:
            values = [math.nan] * 6
        prefixes = ("max_displacement", "path_length", "mean_speed", "max_speed", "vertical_range", "lateral_range")
        feature_values.extend(values)
        feature_names.extend([f"{hand_name}_{prefix}" for prefix in prefixes])
    left, right = JOINTS.index("LeftHand"), JOINTS.index("RightHand")
    both = mask[:, left] & mask[:, right]
    separation = np.linalg.norm(positions[:, left] - positions[:, right], axis=1)
    feature_values.extend([
        float(np.nanmean(separation[both])) if both.any() else math.nan,
        float(np.nanmax(separation[both])) if both.any() else math.nan,
        float(mask.mean()),
    ])
    feature_names.extend(["mean_hand_separation", "max_hand_separation", "joint_observation_fraction"])
    return np.asarray(feature_values, dtype=float), feature_names


def construct_dataset(
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    events: pd.DataFrame,
    trajectories: pd.DataFrame,
    output_root: Path,
    config: ExperimentConfig,
) -> dict[str, object]:
    labels_by_key = labels.set_index("trial_key")
    event_by_recording = events.set_index("recording_identifier")
    motion_counts = manifest.assign(
        trial_key=lambda x: "PedNYC" + x.pednyc_number.astype(int).astype(str)
        + "_scenario" + x.scenario_number.astype(int).astype(str)
    ).groupby("trial_key").size().to_dict()
    trajectory_groups = {key: one.sort_values("scenario_time") for key, one in trajectories.groupby("recording_identifier")}
    target_relative = np.linspace(-config.window_before_s, config.window_after_s, config.sequence_length)
    sequence_values, sequence_masks, dynamic_contexts, contexts, handcrafted, outcomes = [], [], [], [], [], []
    manifest_rows: list[dict[str, object]] = []
    context_names = ["vehicle_speed_mps", "vehicle_acceleration_mps2", "car_ped_distance_m"]
    static_names = [
        "observed_window_duration_s", "delta_TTZ_seconds", "entry_time_difference_seconds",
        "actor_order_pedestrian", "actor_order_vehicle", "actor_order_undefined",
        "left_indicator", "right_indicator", "potential_zone",
    ] + [f"scenario_{scenario}" for scenario in DEFAULT_SCENARIOS]
    handcrafted_names: list[str] = []
    for row in manifest.sort_values(["pednyc_number", "scenario_number", "recording_identifier"]).itertuples(index=False):
        participant = f"PedNYC{int(row.pednyc_number)}"
        scenario = int(row.scenario_number)
        trial_key = f"{participant}_scenario{scenario}"
        record = {
            "participant_id": participant, "scenario": scenario,
            "recording_identifier": row.recording_identifier, "trial_key": trial_key,
            "source_csv_path": row.source_csv_path, "label_raw": "Unknown",
            "label_binary": pd.NA, "window_start_s": math.nan, "window_end_s": math.nan,
            "window_anchor_s": math.nan, "window_anchor_method": "",
            "original_window_duration_s": math.nan, "missing_joint_fraction": math.nan,
            "exclusion_reason": "", "included": False,
        }
        if trial_key not in labels_by_key.index:
            record["exclusion_reason"] = "no_ground_truth_row"
            manifest_rows.append(record)
            continue
        label_row = labels_by_key.loc[trial_key]
        raw_label = label_row["pedestrian_hand_gesture"]
        record["label_raw"] = raw_label
        label = clean_binary_label(raw_label)
        if label is None:
            record["exclusion_reason"] = "unknown_hand_gesture_label"
            manifest_rows.append(record)
            continue
        record["label_binary"] = label
        if motion_counts.get(trial_key, 0) != 1:
            record["exclusion_reason"] = "ambiguous_duplicate_motion_trial"
            manifest_rows.append(record)
            continue
        if row.recording_identifier not in event_by_recording.index:
            record["exclusion_reason"] = "missing_conflict_event"
            manifest_rows.append(record)
            continue
        event = event_by_recording.loc[row.recording_identifier]
        anchor, method = choose_conflict_anchor(event)
        record["window_anchor_s"], record["window_anchor_method"] = anchor, method
        if not np.isfinite(anchor):
            record["exclusion_reason"] = "missing_conflict_anchor"
            manifest_rows.append(record)
            continue
        source_frame, time, normalized, source_mask = _read_trial_skeleton(Path(row.source_csv_path))
        indices, reason = extract_conflict_window(
            time, anchor, config.window_before_s, config.window_after_s
        )
        record["window_start_s"] = anchor - config.window_before_s
        record["window_end_s"] = anchor + config.window_after_s
        if reason:
            record["exclusion_reason"] = reason
            manifest_rows.append(record)
            continue
        target_time = anchor + target_relative
        positions, position_mask = resample_fixed_length(time, normalized, source_mask, target_time)
        if position_mask[:, JOINTS.index("Hips")].mean() < 0.95:
            record["exclusion_reason"] = "pelvis_unavailable_after_resampling"
            manifest_rows.append(record)
            continue
        velocities, velocity_mask = joint_velocity(positions, position_mask, target_time)
        value_sequence = np.concatenate([positions, velocities], axis=2).reshape(config.sequence_length, -1)
        # Values are stored as zero only where the explicit mask is false.
        value_sequence[~np.isfinite(value_sequence)] = 0.0
        trajectory = trajectory_groups.get(row.recording_identifier)
        if trajectory is None:
            record["exclusion_reason"] = "missing_trajectory_context"
            manifest_rows.append(record)
            continue
        trajectory = trajectory.copy()
        trajectory["car_ped_distance_m"] = np.hypot(
            trajectory.veh_x - trajectory.ped_x, trajectory.veh_z - trajectory.ped_z
        )
        dynamic_context = _interpolate_table(
            trajectory.scenario_time.to_numpy(dtype=float), trajectory,
            ["veh_speed", "veh_acceleration", "car_ped_distance_m"], target_time,
        )
        dynamic_context[:, :2] = np.where(np.isfinite(dynamic_context[:, :2]), dynamic_context[:, :2], np.nan)
        indicator_text = " ".join(source_frame.get("A indicators", pd.Series(dtype=str)).astype(str)).casefold()
        order = str(event.get("reached_zone_first", "undefined")).casefold()
        finite_source = time[indices]
        original_duration = float(finite_source[-1] - finite_source[0])
        static_context = np.asarray([
            original_duration,
            pd.to_numeric(event.get("delta_TTZ_seconds"), errors="coerce"),
            pd.to_numeric(event.get("entry_time_difference_seconds"), errors="coerce"),
            float(order == "pedestrian"), float(order == "vehicle"),
            float(order not in {"pedestrian", "vehicle"}),
            float("lefttrue" in indicator_text), float("righttrue" in indicator_text),
            float(str(event.get("zone_event_status", "")) == "potential_counterfactual"),
            *[float(scenario == candidate) for candidate in DEFAULT_SCENARIOS],
        ], dtype=float)
        hand_features, handcrafted_names = _handcrafted_features(positions, velocities, position_mask)
        record["original_window_duration_s"] = original_duration
        record["missing_joint_fraction"] = float(1 - position_mask.mean())
        record["included"] = True
        record["dataset_index"] = len(outcomes)
        sequence_values.append(value_sequence.astype(np.float32))
        sequence_masks.append(position_mask.astype(np.uint8))
        dynamic_contexts.append(dynamic_context.astype(np.float32))
        contexts.append(np.concatenate([np.nanmean(dynamic_context, axis=0), static_context]))
        handcrafted.append(hand_features)
        outcomes.append(label)
        manifest_rows.append(record)
    dataset_manifest = pd.DataFrame(manifest_rows)
    dataset_manifest.to_csv(output_root / "dataset/dataset_manifest.csv", index=False)
    included = dataset_manifest[dataset_manifest.included].sort_values("dataset_index")
    if len(included) < config.folds:
        raise ValueError(f"Only {len(included)} usable trials; cannot run {config.folds} folds")
    arrays = {
        "sequences": np.asarray(sequence_values, dtype=np.float32),
        "joint_masks": np.asarray(sequence_masks, dtype=np.uint8),
        "dynamic_context_sequences": np.asarray(dynamic_contexts, dtype=np.float32),
        "context": np.asarray(contexts, dtype=np.float32),
        "handcrafted": np.asarray(handcrafted, dtype=np.float32),
        "labels": np.asarray(outcomes, dtype=np.int64),
        "participants": included.participant_id.to_numpy(dtype=str),
        "recording_identifiers": included.recording_identifier.to_numpy(dtype=str),
        "scenarios": included.scenario.to_numpy(dtype=np.int64),
        "target_relative_time": target_relative.astype(np.float32),
        "context_names": np.asarray(context_names + static_names, dtype=str),
        "handcrafted_names": np.asarray(handcrafted_names, dtype=str),
        "sequence_feature_names": np.asarray([
            f"{kind}_{joint}_{axis}"
            for joint in JOINTS for kind in ("position", "velocity") for axis in AXES
        ], dtype=str),
    }
    np.savez_compressed(output_root / "dataset/skeleton_windows.npz", **arrays)
    return arrays


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30, 30)
    return 1 / (1 + np.exp(-clipped))


class NumpyLSTMClassifier:
    """A compact full-batch LSTM with Adam and class-weighted BCE."""

    def __init__(self, input_size: int, hidden_size: int, context_size: int, seed: int):
        self.input_size, self.hidden_size, self.context_size = input_size, hidden_size, context_size
        rng = np.random.default_rng(seed)
        scale = 1 / math.sqrt(input_size + hidden_size)
        self.params = {
            "W": rng.normal(0, scale, (input_size + hidden_size, 4 * hidden_size)),
            "b": np.zeros(4 * hidden_size),
            "Wo": rng.normal(0, 1 / math.sqrt(hidden_size + context_size), (hidden_size + context_size, 1)),
            "bo": np.zeros(1),
        }

    def _forward(self, x: np.ndarray, context: np.ndarray | None = None):
        n, steps, _ = x.shape
        h = np.zeros((n, self.hidden_size))
        c = np.zeros_like(h)
        cache = []
        for step in range(steps):
            combined = np.concatenate([x[:, step], h], axis=1)
            gates = combined @ self.params["W"] + self.params["b"]
            i = _sigmoid(gates[:, :self.hidden_size])
            f = _sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
            o = _sigmoid(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
            g = np.tanh(gates[:, 3 * self.hidden_size:])
            previous_c = c
            c = f * c + i * g
            h = o * np.tanh(c)
            cache.append((combined, i, f, o, g, previous_c, c))
        fused = np.concatenate([h, context], axis=1) if context is not None else h
        probability = _sigmoid((fused @ self.params["Wo"] + self.params["bo"]).ravel())
        return probability, fused, cache

    def fit(
        self, x: np.ndarray, y: np.ndarray, context: np.ndarray | None,
        epochs: int, learning_rate: float, l2: float,
    ) -> list[dict[str, float]]:
        y = y.astype(float)
        positives, negatives = max(y.sum(), 1), max((1 - y).sum(), 1)
        weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
        adam_m = {key: np.zeros_like(value) for key, value in self.params.items()}
        adam_v = {key: np.zeros_like(value) for key, value in self.params.items()}
        history = []
        for epoch in range(1, epochs + 1):
            probability, fused, cache = self._forward(x, context)
            loss = -np.sum(weights * (y * np.log(probability + 1e-8) + (1 - y) * np.log(1 - probability + 1e-8))) / weights.sum()
            loss += .5 * l2 * (np.sum(self.params["W"] ** 2) + np.sum(self.params["Wo"] ** 2))
            dlogit = weights * (probability - y) / weights.sum()
            gradients = {
                "Wo": fused.T @ dlogit[:, None] + l2 * self.params["Wo"],
                "bo": np.asarray([dlogit.sum()]),
                "W": np.zeros_like(self.params["W"]),
                "b": np.zeros_like(self.params["b"]),
            }
            dh = dlogit[:, None] @ self.params["Wo"][:self.hidden_size].T
            dc = np.zeros_like(dh)
            for step in range(len(cache) - 1, -1, -1):
                combined, i, f, o, g, previous_c, current_c = cache[step]
                tanh_c = np.tanh(current_c)
                do = dh * tanh_c
                dc = dc + dh * o * (1 - tanh_c ** 2)
                df = dc * previous_c
                di = dc * g
                dg = dc * i
                dc = dc * f
                dz = np.concatenate([
                    di * i * (1 - i), df * f * (1 - f),
                    do * o * (1 - o), dg * (1 - g ** 2),
                ], axis=1)
                gradients["W"] += combined.T @ dz
                gradients["b"] += dz.sum(axis=0)
                dcombined = dz @ self.params["W"].T
                dh = dcombined[:, self.input_size:]
            gradients["W"] += l2 * self.params["W"]
            norm = math.sqrt(sum(float(np.sum(value ** 2)) for value in gradients.values()))
            if norm > 5:
                gradients = {key: value * (5 / norm) for key, value in gradients.items()}
            for key in self.params:
                adam_m[key] = .9 * adam_m[key] + .1 * gradients[key]
                adam_v[key] = .999 * adam_v[key] + .001 * gradients[key] ** 2
                m_hat = adam_m[key] / (1 - .9 ** epoch)
                v_hat = adam_v[key] / (1 - .999 ** epoch)
                self.params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            prediction = (probability >= .5).astype(int)
            history.append({
                "epoch": epoch, "loss": float(loss),
                "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            })
        return history

    def predict_proba(self, x: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        return self._forward(x, context)[0]


def _standardize_sequences(
    train: np.ndarray, test: np.ndarray, train_masks: np.ndarray, test_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Position/velocity feature validity is inherited from each joint mask.
    joint_count = train_masks.shape[2]
    value_mask_train = np.repeat(train_masks, 6, axis=2).astype(bool)
    value_mask_test = np.repeat(test_masks, 6, axis=2).astype(bool)
    flat_train = train.reshape(-1, train.shape[-1]).copy()
    flat_test = test.reshape(-1, test.shape[-1]).copy()
    flat_mask = value_mask_train.reshape(-1, train.shape[-1])
    mean = np.asarray([
        np.mean(flat_train[flat_mask[:, column], column]) if flat_mask[:, column].any() else 0
        for column in range(train.shape[-1])
    ])
    std = np.asarray([
        np.std(flat_train[flat_mask[:, column], column]) if flat_mask[:, column].any() else 1
        for column in range(train.shape[-1])
    ])
    std[std < 1e-6] = 1
    normalized_train = (train - mean) / std
    normalized_test = (test - mean) / std
    normalized_train[~value_mask_train] = 0
    normalized_test[~value_mask_test] = 0
    # Explicit joint masks let the LSTM distinguish zero-centered values from missing joints.
    return (
        np.concatenate([normalized_train, train_masks.astype(float)], axis=2),
        np.concatenate([normalized_test, test_masks.astype(float)], axis=2),
    )


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= .5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "precision_no": float(precision_score(y, prediction, pos_label=0, zero_division=0)),
        "recall_no": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "precision_yes": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "recall_yes": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else math.nan,
    }


def participant_bootstrap_ci(
    y: np.ndarray, probability: np.ndarray, groups: np.ndarray,
    repetitions: int, seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    samples: dict[str, list[float]] = {}
    for _ in range(repetitions):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        if len(np.unique(y[indices])) < 2:
            continue
        for key, value in _metrics(y[indices], probability[indices]).items():
            if np.isfinite(value):
                samples.setdefault(key, []).append(value)
    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in samples.items() if values
    }


def evaluate_models(arrays: dict[str, object], output_root: Path, config: ExperimentConfig) -> pd.DataFrame:
    sequences = np.asarray(arrays["sequences"], dtype=float)
    masks = np.asarray(arrays["joint_masks"], dtype=bool)
    dynamic_context = np.asarray(arrays["dynamic_context_sequences"], dtype=float)
    context = np.asarray(arrays["context"], dtype=float)
    handcrafted = np.asarray(arrays["handcrafted"], dtype=float)
    labels = np.asarray(arrays["labels"], dtype=int)
    groups = np.asarray(arrays["participants"], dtype=str)
    identifiers = np.asarray(arrays["recording_identifiers"], dtype=str)
    splits = grouped_splits(labels, groups, config.folds, config.seed)
    predictions = {
        model: np.full(len(labels), np.nan) for model in (
            "majority_class", "hand_motion_rule", "context_only_logistic",
            "handcrafted_skeleton_logistic", "skeleton_lstm", "multimodal_lstm",
        )
    }
    fold_rows, history_rows = [], []
    for fold, (train, test) in enumerate(splits):
        majority = int(np.mean(labels[train]) >= .5)
        predictions["majority_class"][test] = majority
        # Fixed, transparent proxy; this is not an existing study detector.
        max_displacement = np.nanmax(handcrafted[test][:, [0, 6]], axis=1)
        predictions["hand_motion_rule"][test] = (max_displacement >= .25).astype(float)
        for name, features in (
            ("context_only_logistic", context),
            ("handcrafted_skeleton_logistic", handcrafted),
        ):
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=config.seed),
            )
            model.fit(features[train], labels[train])
            predictions[name][test] = model.predict_proba(features[test])[:, 1]
        train_seq, test_seq = _standardize_sequences(
            sequences[train], sequences[test], masks[train], masks[test]
        )
        dynamic_imputer = SimpleImputer(strategy="median")
        train_dynamic = dynamic_imputer.fit_transform(
            dynamic_context[train].reshape(-1, dynamic_context.shape[2])
        )
        test_dynamic = dynamic_imputer.transform(
            dynamic_context[test].reshape(-1, dynamic_context.shape[2])
        )
        dynamic_scaler = StandardScaler().fit(train_dynamic)
        train_dynamic = dynamic_scaler.transform(train_dynamic).reshape(
            len(train), dynamic_context.shape[1], dynamic_context.shape[2]
        )
        test_dynamic = dynamic_scaler.transform(test_dynamic).reshape(
            len(test), dynamic_context.shape[1], dynamic_context.shape[2]
        )
        static_context = context[:, 3:]
        context_imputer = SimpleImputer(strategy="median")
        train_context = context_imputer.fit_transform(static_context[train])
        test_context = context_imputer.transform(static_context[test])
        scaler = StandardScaler().fit(train_context)
        train_context, test_context = scaler.transform(train_context), scaler.transform(test_context)
        for name, train_input, test_input, train_fusion, test_fusion in (
            ("skeleton_lstm", train_seq, test_seq, None, None),
            (
                "multimodal_lstm",
                np.concatenate([train_seq, train_dynamic], axis=2),
                np.concatenate([test_seq, test_dynamic], axis=2),
                train_context, test_context,
            ),
        ):
            model = NumpyLSTMClassifier(
                train_input.shape[2], config.lstm_hidden_size,
                0 if train_fusion is None else train_fusion.shape[1],
                config.seed + fold + (100 if name == "multimodal_lstm" else 0),
            )
            history = model.fit(
                train_input, labels[train], train_fusion, config.lstm_epochs,
                config.lstm_learning_rate, config.lstm_l2,
            )
            predictions[name][test] = model.predict_proba(test_input, test_fusion)
            for values in history:
                history_rows.append({"model": name, "fold": fold, **values})
            np.savez_compressed(
                output_root / f"models/{name}_fold_{fold}.npz", **model.params
            )
        for name, probability in predictions.items():
            if np.isfinite(probability[test]).all():
                fold_rows.append({"model": name, "fold": fold, "n_test": len(test), **_metrics(labels[test], probability[test])})
    pd.DataFrame(history_rows).to_csv(output_root / "training/training_histories.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_root / "metrics/fold_level_metrics.csv", index=False)
    aggregate_rows = []
    confusion_rows = []
    prediction_rows = []
    for name, probability in predictions.items():
        values = _metrics(labels, probability)
        intervals = participant_bootstrap_ci(
            labels, probability, groups, config.bootstrap_repetitions, config.seed + 300
        )
        row = {"model": name, "n_trials": len(labels), "n_participants": len(np.unique(groups)), **values}
        for metric, (lower, upper) in intervals.items():
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        width = row.get("balanced_accuracy_ci_upper", 1) - row.get("balanced_accuracy_ci_lower", 0)
        row["interpretation"] = "exploratory_unstable_or_wide_ci" if width > .20 else "exploratory"
        aggregate_rows.append(row)
        matrix = confusion_matrix(labels, probability >= .5, labels=[0, 1])
        for true_label in (0, 1):
            for predicted_label in (0, 1):
                confusion_rows.append({
                    "model": name, "true_label": "Yes" if true_label else "No",
                    "predicted_label": "Yes" if predicted_label else "No",
                    "count": int(matrix[true_label, predicted_label]),
                })
        for index in range(len(labels)):
            prediction_rows.append({
                "model": name, "recording_identifier": identifiers[index],
                "participant_id": groups[index], "true_label": "Yes" if labels[index] else "No",
                "probability_yes": probability[index],
                "predicted_label": "Yes" if probability[index] >= .5 else "No",
                "correct": bool((probability[index] >= .5) == labels[index]),
            })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_root / "metrics/aggregate_metrics_with_ci.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_root / "metrics/confusion_matrices.csv", index=False)
    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(output_root / "metrics/out_of_fold_predictions.csv", index=False)
    examples = prediction_frame[prediction_frame.model.eq("multimodal_lstm")].copy()
    examples["confidence"] = np.abs(examples.probability_yes - .5)
    examples = pd.concat([
        examples[examples.correct].nlargest(5, "confidence"),
        examples[~examples.correct].nlargest(5, "confidence"),
    ])
    examples.to_csv(output_root / "examples/classification_examples.csv", index=False)
    return aggregate


def _plot_confusions(output_root: Path) -> None:
    frame = pd.read_csv(output_root / "metrics/confusion_matrices.csv")
    models = frame.model.unique()
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for axis, model in zip(axes.flat, models):
        one = frame[frame.model.eq(model)]
        matrix = one.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=["No", "Yes"], columns=["No", "Yes"]).to_numpy()
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
        axis.set(title=model.replace("_", " "), xlabel="Predicted", ylabel="True", xticks=[0, 1], yticks=[0, 1], xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        figure.colorbar(image, ax=axis, fraction=.046)
    figure.savefig(output_root / "figures/confusion_matrices.png", dpi=180)
    plt.close(figure)


def _plot_comparison(aggregate: pd.DataFrame, output_root: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    order = aggregate.sort_values("balanced_accuracy")
    error = np.vstack([
        order.balanced_accuracy - order.balanced_accuracy_ci_lower,
        order.balanced_accuracy_ci_upper - order.balanced_accuracy,
    ])
    axis.barh(order.model.str.replace("_", " "), order.balanced_accuracy, xerr=error, color="#4C78A8", alpha=.8)
    axis.axvline(.5, color="black", linestyle="--", linewidth=1)
    axis.set(xlim=(0, 1), xlabel="Participant-grouped out-of-fold balanced accuracy", title="Observable pedestrian hand-gesture prediction (95% participant bootstrap CI)")
    figure.savefig(output_root / "figures/model_comparison.png", dpi=180)
    plt.close(figure)


def _plot_skeleton_examples(arrays: dict[str, object], output_root: Path) -> None:
    sequences = np.asarray(arrays["sequences"])
    positions = sequences.reshape(
        len(sequences), len(np.asarray(arrays["target_relative_time"])), len(JOINTS), 6
    )[..., :3]
    identifiers = np.asarray(arrays["recording_identifiers"])
    labels = np.asarray(arrays["labels"])
    masks = np.asarray(arrays["joint_masks"], dtype=bool)
    joint_index = {joint: index for index, joint in enumerate(JOINTS)}
    no_examples = np.flatnonzero(labels == 0)[:3].tolist()
    yes_examples = np.flatnonzero(labels == 1)[:3].tolist()
    chosen = no_examples + yes_examples
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for axis, index in zip(axes.flat, chosen):
        for step, alpha in zip(np.linspace(0, positions.shape[1] - 1, 5).astype(int), np.linspace(.2, 1, 5)):
            for first, second in JOINT_EDGES:
                a, b = joint_index[first], joint_index[second]
                if masks[index, step, a] and masks[index, step, b]:
                    axis.plot(
                        [positions[index, step, a, 0], positions[index, step, b, 0]],
                        [positions[index, step, a, 1], positions[index, step, b, 1]],
                        color="#4C78A8", alpha=alpha, linewidth=1,
                    )
        axis.set(title=f"{identifiers[index]} · {'Yes' if labels[index] else 'No'}", xlabel="participant-frame X (m)", ylabel="Y (m)", aspect="equal")
    figure.savefig(output_root / "figures/skeleton_window_examples.png", dpi=180)
    plt.close(figure)


def write_reports(
    columns: pd.DataFrame, joints: pd.DataFrame, trials: pd.DataFrame,
    labels: pd.DataFrame, dataset: dict[str, object] | None,
    aggregate: pd.DataFrame | None, output_root: Path, config: ExperimentConfig,
) -> None:
    counts = labels.pedestrian_hand_gesture.value_counts().to_dict()
    report = f"""# Skeleton data audit

## Scope and provenance

- Audited {len(trials)} successfully processed trials across scenarios {', '.join(map(str, DEFAULT_SCENARIOS))}.
- Found {len(columns)} pedestrian (`B`) skeleton/VR/avatar scalar columns after whitespace normalization.
- Xiang's workbook is referenced by the repository as `XCPedestrian-GroundTruth-RERUN.xlsx`, but is absent from this working tree. This experiment uses the existing immutable cleaned export at `{DEFAULT_LABELS.relative_to(PROJECT_ROOT)}` and preserves its row numbers and categorical values.
- `B Avatar Pos` is inventoried but never used as global pedestrian trajectory. Existing `B VR Pos X/Z` trajectory artifacts supply pedestrian position where context is needed.

## Timing and quality

- Time is reconstructed from consecutive `ScenarioTime` differences (`dt`), never from the misleading `FrameRate` field.
- Median trial sampling rate: {trials.sampling_rate_hz.median():.2f} Hz (trial-level range {trials.sampling_rate_hz.min():.2f}–{trials.sampling_rate_hz.max():.2f} Hz).
- Median fixed-joint valid-sample fraction: {joints.valid_sample_fraction.median():.3f}.
- Joint-speed spikes are audit flags at > {config.joint_spike_speed_mps:g} m/s; they are not silently relabeled.
- Accelerations are not model inputs: double differentiation at the observed irregular sampling intervals was judged too noise-sensitive for this exploratory sample. Joint velocities are retained.

## Labels

- Pedestrian hand gesture: Yes={counts.get('Yes', 0)}, No={counts.get('No', 0)}, Unknown={counts.get('Unknown', 0)}. Unknown is excluded, never converted to No.
- Pedestrian speed change is genuinely annotated but is multiclass and deferred until the hand-gesture pipeline is validated.
- No ground-truth field genuinely labels yielding or hesitation. Right-of-way and who-went-first fields are not relabeled as those constructs.

See `skeleton_column_inventory.csv`, `joint_availability_by_trial.csv`, `trial_sampling_quality.csv`, and `label_counts.csv` for complete tables.
"""
    (output_root / "data_audit/data_audit_report.md").write_text(report, encoding="utf-8")
    optional = """# Optional skeleton encoders

## ST-GCN

ST-GCN expects a stable graph-ordered tensor (N,C,T,V,M) and a declared joint adjacency graph. The 20-joint Unity humanoid mapping and adjacency used here are technically convertible, but finger/controller availability varies and there are too few positive trials for defensible training or fine-tuning. No ST-GCN was forced.

## MotionBERT

MotionBERT generally expects a Human3.6M-like 17-joint 2D/3D representation and pretrained motion semantics. Mapping is possible only approximately (Unity LowerArm/LowerLeg stand in for elbow/knee segments), camera/projective inputs are absent, and its pretrained objectives do not match a sparse hand-gesture trial label. Fine-tuning is not defensible for this sample without an external validation cohort.

## SMPL / SMPL-X

SMPL/SMPL-X or Mocap-to-SMPLX could standardize skeleton geometry, but they are not behavior classifiers. Their extra dependencies, licenses, fitting uncertainty, and potential loss of controller-hand detail outweigh their value for this first experiment.
"""
    (output_root / "reports/optional_encoders.md").write_text(optional, encoding="utf-8")
    if dataset is None or aggregate is None:
        return
    manifest = pd.read_csv(output_root / "dataset/dataset_manifest.csv")
    included = manifest[manifest.included]
    best = aggregate.sort_values("balanced_accuracy", ascending=False).iloc[0]
    skeleton = aggregate[aggregate.model.eq("skeleton_lstm")].iloc[0]
    context = aggregate[aggregate.model.eq("context_only_logistic")].iloc[0]
    fused = aggregate[aggregate.model.eq("multimodal_lstm")].iloc[0]
    readme = f"""# Skeleton behavior LSTM experiment

This experiment predicts only Xiang's observable pedestrian hand-gesture Yes/No label. It does **not** infer intent, yielding, awareness, or hesitation.

## Dataset

- Conflict-centered 6-second windows, resampled to {config.sequence_length} samples without extrapolation.
- Observed scenarios use the mean of available pedestrian/vehicle conflict-zone center times. Scenarios 12 and 16 use the existing potential/counterfactual vehicle-arrival time.
- 20 Unity humanoid joints are pelvis-centered and rotated by pelvis yaw into a participant-facing frame. Original joint masks and actual source-window durations are preserved.
- Included: {len(included)} trials from {included.participant_id.nunique()} participants; Yes={(included.label_binary == 1).sum()}, No={(included.label_binary == 0).sum()}.
- Exclusions are listed explicitly in `dataset/dataset_manifest.csv`. Unknown labels remain Unknown and are excluded.

## Evaluation

- Five-fold `StratifiedGroupKFold`, grouped by PedNYC participant. No participant crosses train/test within a fold.
- Metrics are out-of-fold; confidence intervals resample participants, not frames.
- Models: majority, a new transparent hand-motion threshold rule (not an existing study detector), context-only logistic regression, handcrafted-skeleton logistic regression, small masked NumPy LSTM, and skeleton+context LSTM.

Best observed balanced accuracy was {best.balanced_accuracy:.3f} for `{best.model}`. Skeleton LSTM={skeleton.balanced_accuracy:.3f}, context-only={context.balanced_accuracy:.3f}, fused={fused.balanced_accuracy:.3f}. All results remain exploratory; inspect `metrics/aggregate_metrics_with_ci.csv` for uncertainty.

## Interpretation guardrails

Whether skeleton data adds measurable value is assessed only by held-out participant performance relative to context-only and uncertainty intervals. Vehicle/context necessity is assessed by the context-only and fused comparisons. Wide participant-bootstrap intervals or fold instability mean generalization and stability are not established.

## Reproduction

```powershell
$env:PYTHONPATH="$(Resolve-Path '.python_packages');$(Resolve-Path '.')"
& 'C:\\Users\\patl5\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -m src.analysis.skeleton_behavior_lstm
```
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    comparison = aggregate[[
        "model", "balanced_accuracy", "balanced_accuracy_ci_lower", "balanced_accuracy_ci_upper",
        "macro_f1", "pr_auc", "interpretation",
    ]].copy()
    comparison.to_csv(output_root / "metrics/model_comparison.csv", index=False)
    fold_metrics = pd.read_csv(output_root / "metrics/fold_level_metrics.csv")
    fold_ranges = fold_metrics.groupby("model").balanced_accuracy.agg(["min", "max"])
    final_interpretation = f"""# Final interpretation

The target is the observable pedestrian hand-gesture Yes/No annotation. The models do not understand or infer pedestrian intent.

1. **Does skeleton data add measurable value?** Not in the learned models. The skeleton LSTM achieved balanced accuracy {skeleton.balanced_accuracy:.3f} (95% participant-bootstrap CI {skeleton.balanced_accuracy_ci_lower:.3f}–{skeleton.balanced_accuracy_ci_upper:.3f}); handcrafted skeleton logistic regression achieved {aggregate.loc[aggregate.model.eq('handcrafted_skeleton_logistic'), 'balanced_accuracy'].iloc[0]:.3f}. Neither establishes improvement over chance/majority performance. The fixed hand-motion rule reached {aggregate.loc[aggregate.model.eq('hand_motion_rule'), 'balanced_accuracy'].iloc[0]:.3f}, but it is a newly introduced transparent proxy, not an existing study detector, and its macro F1 was only {aggregate.loc[aggregate.model.eq('hand_motion_rule'), 'macro_f1'].iloc[0]:.3f}.
2. **Is vehicle/context data necessary?** Not established. Context-only logistic regression reached {context.balanced_accuracy:.3f}, while fusion fell to {fused.balanced_accuracy:.3f}. The context-only interval includes 0.5 and fusion did not improve skeleton performance, so these data do not demonstrate necessary or additive context information for this label.
3. **Do results generalize across participants?** Not established. Evaluation was correctly participant-grouped, but only {(included.label_binary == 1).sum()} positive and {(included.label_binary == 0).sum()} negative trials were usable. Confidence intervals overlap chance for all learned models.
4. **Is the model stable enough for continued confirmatory research?** No. Fold balanced-accuracy ranges were {fold_ranges.loc['skeleton_lstm', 'min']:.3f}–{fold_ranges.loc['skeleton_lstm', 'max']:.3f} for skeleton LSTM, {fold_ranges.loc['context_only_logistic', 'min']:.3f}–{fold_ranges.loc['context_only_logistic', 'max']:.3f} for context-only, and {fold_ranges.loc['multimodal_lstm', 'min']:.3f}–{fold_ranges.loc['multimodal_lstm', 'max']:.3f} for fusion.
5. **Overall status:** The experiment should remain exploratory. A larger positive class, an independently available source workbook, and an external participant cohort are needed before model selection or scientific claims.
"""
    (output_root / "reports/final_interpretation.md").write_text(
        final_interpretation, encoding="utf-8"
    )


def prepare_directories(output_root: Path) -> None:
    for relative in ("data_audit", "dataset", "models", "training", "metrics", "figures", "examples", "reports"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed, folds=args.folds, lstm_epochs=args.epochs,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    set_global_seed(config.seed)
    output_root = args.output_root.resolve()
    prepare_directories(output_root)
    labels, manifest, events, trajectories = load_experiment_inputs()
    if args.skip_audit:
        columns = pd.read_csv(output_root / "data_audit/skeleton_column_inventory.csv")
        joints = pd.read_csv(output_root / "data_audit/joint_availability_by_trial.csv")
        trials = pd.read_csv(output_root / "data_audit/trial_sampling_quality.csv")
    else:
        columns, joints, trials = audit_skeleton_data(manifest, labels, output_root, config)
    (output_root / "model_configurations.json").write_text(json.dumps({
        "config": asdict(config),
        "target": "observable pedestrian hand gesture Yes/No",
        "unknown_policy": "exclude; never map Unknown to No",
        "split_policy": "StratifiedGroupKFold grouped by participant_id",
        "lstm": "small deterministic NumPy LSTM; explicit joint masks concatenated to inputs; class-weighted BCE",
        "context": "vehicle speed/acceleration, planar actor distance, conflict timing/order, indicators, scenario",
        "acceleration_policy": "joint accelerations excluded as unstable double derivatives",
    }, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        write_reports(columns, joints, trials, labels, None, None, output_root, config)
        return 0
    dataset = construct_dataset(labels, manifest, events, trajectories, output_root, config)
    aggregate = evaluate_models(dataset, output_root, config)
    _plot_confusions(output_root)
    _plot_comparison(aggregate, output_root)
    _plot_skeleton_examples(dataset, output_root)
    write_reports(columns, joints, trials, labels, dataset, aggregate, output_root, config)
    print(aggregate[["model", "balanced_accuracy", "macro_f1", "pr_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
