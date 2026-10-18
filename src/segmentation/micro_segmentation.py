from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_DIR = ROOT / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

DEFAULT_FEATURE_CSV = ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv"
DEFAULT_MACRO_CSV = (
    ROOT
    / "outputs"
    / "graphs"
    / "macro_segmentation"
    / "macro_segments_PedNYC1_scenario3_v2.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "graphs" / "micro_segmentation"

TIME_COL = "ScenarioTime"
DT_COL = "dt"
GAME_TIME_COL = "GameTime"
FRAME_COL = "Frame Number"

PED_SPEED_COLS = ["ped_speed_xz_smooth", "ped_speed_xz"]
PED_ACCEL_COLS = ["ped_accel_xz_smooth", "ped_accel_xz"]
HEAD_TURN_COLS = ["head_turn_rate"]
HEAD_BODY_DIFF_COLS = ["head_body_yaw_diff"]
PED_X_COLS = ["ped_x"]
PED_Z_COLS = ["ped_z"]
DISTANCE_COLS = ["car_ped_distance_xz_smooth", "car_ped_distance_xz"]
CLOSING_COLS = ["distance_closing_smooth", "distance_closing"]
DISTANCE_CHANGE_COLS = ["distance_change"]
CAR_SPEED_COLS = ["car_speed_xz_smooth", "car_speed_xz"]
CAR_ACCEL_COLS = ["car_accel_xz_smooth", "car_accel_xz"]
AVATAR_GAP_COLS = ["avatar_vr_gap_xz"]

# Change-point detection settings. These are intentionally easy to tune.
# Boundary detection runs on residuals after subtracting a local robust trend,
# so small micro-behaviors are not drowned out by the larger macro movement.
TREND_WINDOW_SECONDS = 1.00
LOCAL_NORMALIZATION_SECONDS = 1.50
CHANGE_SCORE_SMOOTH_SECONDS = 0.12
CHANGE_LAG_SECONDS = [0.0, 0.17, 0.50]
CHANGE_LAG_WEIGHTS = [1.0, 1.0, 1.5]
CHANGE_SIGNAL_WEIGHTS = {
    "speed": 1.0,
    "accel": 2.0,
    "head_turn": 1.0,
}
CHANGE_SCORE_PERCENTILE = 50
PEAK_PROMINENCE_FRACTION = 0.35
MIN_BOUNDARY_GAP_SECONDS = 0.15
MIN_MICRO_SEGMENT_SECONDS = 0.25
MAX_MICRO_SEGMENT_SECONDS = 2.50

# Explicit raw-signal detectors. These add boundaries for short hesitations and
# pauses that a statistical score can miss.
HESITATION_DROP_MIN_MPS = 0.18
HESITATION_DROP_FRACTION = 0.22
HESITATION_RECOVERY_FRACTION = 0.35
HESITATION_MIN_SECONDS = 0.15
HESITATION_MAX_SECONDS = 1.50
PAUSE_SPEED_MAX_MPS = 0.10
PAUSE_MIN_SECONDS = 0.12

# Descriptive tags. These are independent measured dimensions, not final intent
# labels. Keep them separated so review can filter/recombine dimensions later.
MOTION_TAGS = [
    "near_stationary",
    "pausing",
    "hesitating",
    "speed_increasing",
    "speed_decreasing",
    "speed_steady",
    "fluctuating",
    "mixed_motion",
]
HEAD_TAGS = ["head_checking", "head_active", "head_still"]
CAR_TAGS = ["yielding", "proceeding", "conflicted", "neutral"]
TAG_DIMENSIONS = {
    "motion": MOTION_TAGS,
    "head": HEAD_TAGS,
    "car_context": CAR_TAGS,
}
TAG_OBSERVABILITY = {
    "near_stationary": "observable",
    "pausing": "observable",
    "speed_increasing": "observable",
    "speed_decreasing": "observable",
    "speed_steady": "observable",
    "fluctuating": "observable",
    "head_active": "observable",
    "head_still": "observable",
    "neutral": "observable",
    "hesitating": "inferred",
    "mixed_motion": "inferred",
    "head_checking": "inferred",
    "yielding": "inferred",
    "proceeding": "inferred",
    "conflicted": "inferred",
}
OBSERVABILITY_ORDER = ["observable", "inferred"]
OBSERVABILITY_LABELS = {
    "observable": "Observable Kinematics",
    "inferred": "Inferred Latent Behaviors",
}
TAG_DEFINITIONS = {
    "near_stationary": "Mean VR-derived pedestrian speed is below 0.15 m/s or the segment contains near-zero speed.",
    "pausing": "VR-derived speed stays below 0.10 m/s for at least 0.12 seconds.",
    "hesitating": "Speed shows a short measured dip: drop >0.18 m/s and >22%, with partial recovery within 1.5 seconds.",
    "speed_increasing": "Pedestrian speed slope is positive enough to indicate rising speed.",
    "speed_decreasing": "Pedestrian speed slope is negative enough to indicate falling speed.",
    "speed_steady": "Speed coefficient of variation is low while mean speed is above walking threshold.",
    "fluctuating": "Speed coefficient of variation is elevated without a clear increasing/decreasing slope.",
    "mixed_motion": "Motion does not cleanly match the other measured motion tags.",
    "head_checking": "Head turn spike is brief and exceeds 120 deg/s.",
    "head_active": "Head turn rate or head/body yaw difference is elevated.",
    "head_still": "Head turn activity stays below the head_checking/head_active thresholds.",
    "yielding": "Near-car context with measured slowing, pausing, or hesitation.",
    "proceeding": "Near-car context with measured steady or increasing movement.",
    "conflicted": "Near-car context with fluctuating motion or a high-jerk hesitation.",
    "neutral": "Car-pedestrian distance never drops below the near-car threshold.",
}
TAG_COLORS = {
    "near_stationary": "#4E79A7",
    "pausing": "#A0CBE8",
    "hesitating": "#F28E2B",
    "speed_increasing": "#59A14F",
    "speed_decreasing": "#E15759",
    "fluctuating": "#B07AA1",
    "mixed_motion": "#9C755F",
    "head_active": "#EDC948",
    "head_checking": "#FFBE7D",
    "head_still": "#BAB0AC",
    "speed_steady": "#76B7B2",
    "yielding": "#D37295",
    "proceeding": "#8CD17D",
    "conflicted": "#B6992D",
    "neutral": "#D0D0D0",
}

# Display-only vocabulary for the slide-ready Gantt. Analytical tags and CSVs
# continue to use the original labels above.
PRESENTATION_ROWS = {
    "Observed Motion": [
        ("motion", "near_stationary", "Low movement", "#3B82F6"),
        ("motion", "pausing", "Pause", "#60A5FA"),
        ("motion", "speed_increasing", "Speeding up", "#16A34A"),
        ("motion", "speed_decreasing", "Slowing down", "#2563EB"),
        ("motion", "speed_steady", "Steady speed", "#14B8A6"),
        ("motion", "fluctuating", "Variable speed", "#0F766E"),
        ("head", "head_active", "Head movement", "#22C55E"),
        ("head", "head_still", "Low head movement", "#6EE7B7"),
    ],
    "Possible Behaviors": [
        ("motion", "hesitating", "Possible hesitation", "#F97316"),
        ("car_context", "yielding", "Possible yielding", "#9333EA"),
    ],
}
PRESENTATION_FOOTNOTE = "Possible behaviors are rule-based interpretations of measured motion and proximity."
NEAR_STATIONARY_SPEED = 0.15
WALKING_MEAN_SPEED = 0.25
SPEED_SLOPE_MIN = 0.12
STEADY_SPEED_CV_MAX = 0.22
FLUCTUATING_SPEED_CV_MIN = 0.22
HEAD_CHECK_TURN_RATE = 120.0
HEAD_CHECK_MAX_SECONDS = 0.50
HEAD_ACTIVE_MEAN_TURN_RATE = 25.0
HEAD_ACTIVE_TURN_RATE = 50.0
HEAD_ACTIVE_YAW_DIFF = 20.0
NEAR_CAR_DISTANCE_METERS = 12.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create descriptive micro-segments inside existing macro segments. "
            "Segments are found by change points, then described with measured features."
        )
    )
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--macro-csv", type=Path, default=DEFAULT_MACRO_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--participant", type=int, default=1)
    parser.add_argument("--scenario", default="3")
    parser.add_argument(
        "--scenario-stem",
        help="Scenario-specific output stem, for example PedNYC1_scenario3.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def pick_col(df, candidates, required=False, purpose="column"):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required {purpose}. Tried: {candidates}")
    return None


def numeric_series(df, col, default=np.nan):
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def is_seconds_like(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return False
    duration = float(np.nanmax(values) - np.nanmin(values))
    return 0 < duration <= 60 and abs(float(np.nanmin(values))) <= 5


def reconstruct_elapsed_time(df):
    if TIME_COL in df.columns:
        scenario_time = numeric_series(df, TIME_COL).to_numpy(dtype=float)
        if is_seconds_like(scenario_time):
            return scenario_time - scenario_time[0], "ScenarioTime_seconds"

    if DT_COL in df.columns:
        dt = numeric_series(df, DT_COL).to_numpy(dtype=float)
        good_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
        if len(good_dt):
            median_dt = float(np.nanmedian(good_dt))
            dt_clean = dt.copy()
            bad = ~np.isfinite(dt_clean) | (dt_clean <= 0) | (dt_clean > 1.0)
            dt_clean[bad] = median_dt
            elapsed = np.zeros(len(df), dtype=float)
            elapsed[1:] = np.cumsum(dt_clean[1:])
            return elapsed, "cumulative_dt_seconds"

    if GAME_TIME_COL in df.columns:
        game_time = numeric_series(df, GAME_TIME_COL).to_numpy(dtype=float)
        if np.any(np.isfinite(game_time)):
            return game_time - game_time[np.isfinite(game_time)][0], "GameTime_minus_start"

    if FRAME_COL in df.columns and DT_COL in df.columns:
        frames = numeric_series(df, FRAME_COL).to_numpy(dtype=float)
        dt = numeric_series(df, DT_COL).to_numpy(dtype=float)
        median_dt = float(np.nanmedian(dt[np.isfinite(dt) & (dt > 0)]))
        return (frames - frames[np.isfinite(frames)][0]) * median_dt, "Frame_Number_times_dt"

    raise ValueError("Could not reconstruct elapsed seconds from ScenarioTime, dt, GameTime, or frame number.")


def seconds_to_frames(time_sec, seconds):
    if len(time_sec) < 2:
        return 1
    dt = np.nanmedian(np.diff(time_sec))
    if not np.isfinite(dt) or dt <= 0:
        return 1
    return max(1, int(round(seconds / dt)))


def rolling_mean(values, window):
    window = max(1, int(window))
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def safe_stat(values, func):
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return np.nan
    return float(func(arr))


def linear_slope(time_sec, values):
    t = np.asarray(time_sec, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = t[mask] - t[mask][0]
    if np.nanmax(x) <= np.nanmin(x):
        return np.nan
    return float(np.polyfit(x, y[mask], 1)[0])


def fill_numeric(values):
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr).interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def robust_trend(values, time_sec):
    arr = fill_numeric(values)
    trend_window = seconds_to_frames(time_sec, TREND_WINDOW_SECONDS)
    if trend_window % 2 == 0:
        trend_window += 1
    median = (
        pd.Series(arr)
        .rolling(window=max(3, trend_window), center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    mean_window = max(1, seconds_to_frames(time_sec, CHANGE_SCORE_SMOOTH_SECONDS))
    return rolling_mean(median, mean_window)


def residual_signal(values, time_sec):
    arr = fill_numeric(values)
    trend = robust_trend(arr, time_sec)
    return arr - trend, trend


def multi_lag_delta(values, time_sec):
    arr = fill_numeric(values)
    combined = np.zeros(len(arr), dtype=float)
    for lag_seconds, lag_weight in zip(CHANGE_LAG_SECONDS, CHANGE_LAG_WEIGHTS):
        lag = 1 if lag_seconds <= 0 else seconds_to_frames(time_sec, lag_seconds)
        lag = max(1, min(lag, max(1, len(arr) - 1)))
        delta = np.zeros(len(arr), dtype=float)
        delta[lag:] = np.abs(arr[lag:] - arr[:-lag])
        delta[:lag] = delta[lag] if len(arr) > lag else 0.0
        scale = float(np.nanstd(delta))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        combined += lag_weight * (delta / scale)
    return combined


def rolling_local_zscore(values, time_sec):
    arr = fill_numeric(values)
    window = max(3, seconds_to_frames(time_sec, LOCAL_NORMALIZATION_SECONDS))
    local_mean = pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    local_std = pd.Series(arr).rolling(window=window, center=True, min_periods=2).std(ddof=0).to_numpy(dtype=float).copy()
    fallback = float(np.nanstd(arr))
    if not np.isfinite(fallback) or fallback < 1e-9:
        fallback = 1.0
    local_std[~np.isfinite(local_std) | (local_std < 1e-9)] = fallback
    z = (arr - local_mean) / local_std
    return np.maximum(z, 0.0)


def local_peak_indices(values, threshold, min_distance_frames=1, prominence_min=0.0):
    values = np.asarray(values, dtype=float)
    peaks = []
    if len(values) < 3:
        return peaks
    for i in range(1, len(values) - 1):
        if values[i] >= threshold and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            local_floor = max(values[i - 1], values[i + 1])
            if values[i] - local_floor >= prominence_min:
                peaks.append(i)
    if not peaks or min_distance_frames <= 1:
        return peaks
    kept = []
    for peak in sorted(peaks, key=lambda idx: values[idx], reverse=True):
        if all(abs(peak - other) >= min_distance_frames for other in kept):
            kept.append(peak)
    return sorted(kept)
    return peaks


def merge_close_boundaries(boundaries, time_sec, scores):
    if not boundaries:
        return []
    boundaries = sorted(set(int(b) for b in boundaries))
    merged = [boundaries[0]]
    for boundary in boundaries[1:]:
        prev = merged[-1]
        if time_sec[boundary] - time_sec[prev] < MIN_BOUNDARY_GAP_SECONDS:
            if scores[boundary] > scores[prev]:
                merged[-1] = boundary
        else:
            merged.append(boundary)
    return merged


def split_long_intervals(boundaries, time_sec):
    changed = True
    boundaries = sorted(set(int(b) for b in boundaries))
    while changed:
        changed = False
        expanded = [boundaries[0]]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            duration = time_sec[right] - time_sec[left]
            if duration > MAX_MICRO_SEGMENT_SECONDS:
                midpoint_time = time_sec[left] + duration / 2.0
                midpoint = int(np.argmin(np.abs(time_sec - midpoint_time)))
                if left < midpoint < right:
                    expanded.append(midpoint)
                    changed = True
            expanded.append(right)
        boundaries = sorted(set(expanded))
    return boundaries


def macro_row_to_indices(macro_row, df):
    if {"start_idx", "end_idx"}.issubset(macro_row.index):
        return max(0, int(macro_row["start_idx"])), min(len(df) - 1, int(macro_row["end_idx"]))

    start_col = "time_start_sec" if "time_start_sec" in macro_row.index else "ScenarioTime_start"
    end_col = "time_end_sec" if "time_end_sec" in macro_row.index else "ScenarioTime_end"
    start_t = float(macro_row[start_col])
    end_t = float(macro_row[end_col])
    elapsed = df["elapsed_time_sec"].to_numpy(dtype=float)
    idx = np.where((elapsed >= start_t) & (elapsed <= end_t))[0]
    if len(idx) == 0:
        raise ValueError(f"No feature rows found for macro segment {macro_row.get('segment_id', 'unknown')}")
    return int(idx[0]), int(idx[-1])


def compute_change_score(seg):
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)
    speed_col = pick_col(seg, PED_SPEED_COLS, required=True, purpose="pedestrian speed")
    accel_col = pick_col(seg, PED_ACCEL_COLS)
    head_col = pick_col(seg, HEAD_TURN_COLS)

    speed_residual, speed_trend = residual_signal(numeric_series(seg, speed_col), time_sec)
    accel_residual, accel_trend = residual_signal(numeric_series(seg, accel_col, default=0.0), time_sec)
    head_residual, head_trend = residual_signal(numeric_series(seg, head_col, default=0.0), time_sec)

    speed_score = multi_lag_delta(speed_residual, time_sec)
    accel_score = multi_lag_delta(accel_residual, time_sec)
    head_score = multi_lag_delta(head_residual, time_sec)

    raw_score = (
        CHANGE_SIGNAL_WEIGHTS["speed"] * speed_score
        + CHANGE_SIGNAL_WEIGHTS["accel"] * accel_score
        + CHANGE_SIGNAL_WEIGHTS["head_turn"] * head_score
    )
    window = seconds_to_frames(time_sec, CHANGE_SCORE_SMOOTH_SECONDS)
    smooth_score = rolling_mean(raw_score, window)
    score = rolling_local_zscore(smooth_score, time_sec)
    diagnostics = {
        "speed_trend": speed_trend,
        "speed_residual": speed_residual,
        "accel_trend": accel_trend,
        "accel_residual": accel_residual,
        "head_turn_trend": head_trend,
        "head_turn_residual": head_residual,
        "change_score_raw": raw_score,
    }
    return score, diagnostics


def add_boundary_source(sources, idx, source):
    idx = int(idx)
    sources.setdefault(idx, set()).add(source)


def detect_pause_boundaries(speed, time_sec):
    speed = fill_numeric(speed)
    is_pause = speed < PAUSE_SPEED_MAX_MPS
    boundaries = []
    start = None
    for i, active in enumerate(is_pause):
        if active and start is None:
            start = i
        if start is not None and (not active or i == len(is_pause) - 1):
            end = i if active and i == len(is_pause) - 1 else i - 1
            duration = time_sec[end] - time_sec[start] if end > start else 0.0
            if duration >= PAUSE_MIN_SECONDS:
                boundaries.extend([start, end])
            start = None
    return boundaries


def detect_hesitation_boundaries(speed, time_sec):
    speed = fill_numeric(speed)
    boundaries = []
    if len(speed) < 5:
        return boundaries
    search_frames = max(2, seconds_to_frames(time_sec, HESITATION_MAX_SECONDS))
    for i in range(1, len(speed) - 1):
        if not (speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]):
            continue
        left_start = max(0, i - search_frames)
        right_end = min(len(speed), i + search_frames + 1)
        left_slice = speed[left_start:i + 1]
        right_slice = speed[i:right_end]
        if len(left_slice) < 2 or len(right_slice) < 2:
            continue
        left_peak_offset = int(np.nanargmax(left_slice))
        left_peak_idx = left_start + left_peak_offset
        right_peak_idx = i + int(np.nanargmax(right_slice))
        pre_max = speed[left_peak_idx]
        post_max = speed[right_peak_idx]
        drop = pre_max - speed[i]
        if pre_max <= 0 or drop < HESITATION_DROP_MIN_MPS or drop / pre_max < HESITATION_DROP_FRACTION:
            continue
        recovery = post_max - speed[i]
        if recovery < HESITATION_RECOVERY_FRACTION * drop:
            continue
        half_level = speed[i] + drop / 2.0
        low_indices = np.where(speed[left_peak_idx:right_peak_idx + 1] <= half_level)[0]
        if len(low_indices):
            half_start = left_peak_idx + int(low_indices[0])
            half_end = left_peak_idx + int(low_indices[-1])
        else:
            half_start = left_peak_idx
            half_end = right_peak_idx
        duration = time_sec[half_end] - time_sec[half_start]
        if HESITATION_MIN_SECONDS <= duration <= HESITATION_MAX_SECONDS:
            boundaries.extend([left_peak_idx, i, right_peak_idx])
    return boundaries


def find_boundaries(seg):
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)
    if len(seg) < 3:
        return [0, len(seg) - 1], np.zeros(len(seg), dtype=float), {0: {"macro_start"}, len(seg) - 1: {"macro_end"}}, {}

    score, diagnostics = compute_change_score(seg)
    threshold = float(np.nanpercentile(score, CHANGE_SCORE_PERCENTILE))
    prominence_min = PEAK_PROMINENCE_FRACTION * float(np.nanmax(score)) if np.any(np.isfinite(score)) else 0.0
    min_peak_gap = seconds_to_frames(time_sec, MIN_BOUNDARY_GAP_SECONDS)
    candidates = local_peak_indices(score, threshold, min_peak_gap, prominence_min)
    sources = {}
    for idx in candidates:
        add_boundary_source(sources, idx, "residual_multiscale_peak")

    speed_col = pick_col(seg, PED_SPEED_COLS, required=True, purpose="pedestrian speed")
    speed = numeric_series(seg, speed_col).to_numpy(dtype=float)
    for idx in detect_pause_boundaries(speed, time_sec):
        candidates.append(idx)
        add_boundary_source(sources, idx, "pause_speed_run")
    for idx in detect_hesitation_boundaries(speed, time_sec):
        candidates.append(idx)
        add_boundary_source(sources, idx, "hesitation_speed_dip")

    candidates = merge_close_boundaries(candidates, time_sec, score)
    boundaries = [0] + [c for c in candidates if 0 < c < len(seg) - 1] + [len(seg) - 1]
    boundaries = split_long_intervals(sorted(set(boundaries)), time_sec)
    add_boundary_source(sources, 0, "macro_start")
    add_boundary_source(sources, len(seg) - 1, "macro_end")
    for idx in boundaries:
        sources.setdefault(int(idx), {"long_segment_split"})
    return boundaries, score, sources, diagnostics


def displacement_stats(seg):
    x_col = pick_col(seg, PED_X_COLS)
    z_col = pick_col(seg, PED_Z_COLS)
    if x_col is None or z_col is None:
        return np.nan, np.nan, np.nan
    x = numeric_series(seg, x_col).to_numpy(dtype=float)
    z = numeric_series(seg, z_col).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(z)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    x = x[mask]
    z = z[mask]
    displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (z[-1] - z[0]) ** 2))
    path_length = float(np.nansum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    progress_ratio = displacement / path_length if path_length > 1e-9 else np.nan
    return displacement, path_length, progress_ratio


def zero_speed_duration(speed, time_sec):
    speed = np.asarray(speed, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)
    if len(speed) < 2:
        return 0.0
    dt = np.diff(time_sec, prepend=time_sec[0])
    if len(dt) > 1:
        dt[0] = np.nanmedian(dt[1:])
    dt[~np.isfinite(dt) | (dt < 0)] = 0.0
    return float(np.nansum(dt[speed < PAUSE_SPEED_MAX_MPS]))


def speed_drop_recovery(speed):
    speed = fill_numeric(speed)
    if len(speed) < 3:
        return np.nan, np.nan
    min_idx = int(np.nanargmin(speed))
    pre_max = float(np.nanmax(speed[:min_idx + 1])) if min_idx > 0 else float(speed[min_idx])
    post_max = float(np.nanmax(speed[min_idx:])) if min_idx < len(speed) - 1 else float(speed[min_idx])
    drop = pre_max - float(speed[min_idx])
    recovery = post_max - float(speed[min_idx])
    return drop, recovery


def classify_motion_tag(features):
    # Exactly one motion tag is assigned from VR-derived pedestrian motion.
    if features["zero_velocity_duration_sec"] >= PAUSE_MIN_SECONDS and features["speed_min"] < PAUSE_SPEED_MAX_MPS:
        return "pausing"
    if (
        features["speed_drop"] >= HESITATION_DROP_MIN_MPS
        and features["duration_sec"] <= HESITATION_MAX_SECONDS
        and features["speed_mean"] > WALKING_MEAN_SPEED
        and features["speed_recovery"] >= HESITATION_RECOVERY_FRACTION * features["speed_drop"]
    ):
        return "hesitating"
    if features["speed_mean"] < NEAR_STATIONARY_SPEED or features["speed_min"] < 0.05:
        return "near_stationary"
    if features["speed_slope"] > SPEED_SLOPE_MIN and features["speed_mean"] > WALKING_MEAN_SPEED:
        return "speed_increasing"
    if features["speed_slope"] < -SPEED_SLOPE_MIN and features["speed_mean"] > WALKING_MEAN_SPEED:
        return "speed_decreasing"
    if features["speed_cv"] < STEADY_SPEED_CV_MAX and features["speed_mean"] > WALKING_MEAN_SPEED:
        return "speed_steady"
    if features["speed_mean"] > WALKING_MEAN_SPEED:
        return "fluctuating"
    return "mixed_motion"


def classify_head_tag(features):
    # Exactly one head tag is assigned from observed head-turn signals.
    if features["head_turn_rate_max_abs"] >= HEAD_CHECK_TURN_RATE and features["duration_sec"] <= HEAD_CHECK_MAX_SECONDS:
        return "head_checking"
    if (
        features["head_turn_rate_mean_abs"] >= HEAD_ACTIVE_MEAN_TURN_RATE
        or features["head_turn_rate_max_abs"] >= HEAD_ACTIVE_TURN_RATE
        or features["head_body_yaw_diff_max_abs"] >= HEAD_ACTIVE_YAW_DIFF
    ):
        return "head_active"
    return "head_still"


def classify_car_tag(features, motion_tag):
    # Car context is deliberately separate from motion/head tags.
    near_car = np.isfinite(features["distance_min"]) and features["distance_min"] < NEAR_CAR_DISTANCE_METERS
    if not near_car:
        return "neutral"
    if motion_tag == "fluctuating" or (motion_tag == "hesitating" and features["accel_jerk_abs_mean"] > 0.8):
        return "conflicted"
    if motion_tag in {"speed_decreasing", "pausing", "hesitating"}:
        return "yielding"
    if motion_tag in {"speed_increasing", "speed_steady"}:
        return "proceeding"
    return "neutral"


def tag_evidence(features, motion_tag, head_tag, car_tag):
    motion_evidence = {
        "near_stationary": f"mean speed {features['speed_mean']:.2f} m/s; min {features['speed_min']:.2f} m/s",
        "pausing": f"speed < {PAUSE_SPEED_MAX_MPS:.2f} m/s for {features['zero_velocity_duration_sec']:.2f} s",
        "hesitating": f"speed drop {features['speed_drop']:.2f} m/s with recovery {features['speed_recovery']:.2f} m/s",
        "speed_increasing": f"speed slope {features['speed_slope']:.2f} m/s/s",
        "speed_decreasing": f"speed slope {features['speed_slope']:.2f} m/s/s",
        "speed_steady": f"speed CV {features['speed_cv']:.2f}",
        "fluctuating": f"speed CV {features['speed_cv']:.2f}",
        "mixed_motion": "motion did not match the stronger motion rules",
    }[motion_tag]
    head_evidence = {
        "head_checking": f"brief head turn max {features['head_turn_rate_max_abs']:.1f} deg/s",
        "head_active": f"head turn mean {features['head_turn_rate_mean_abs']:.1f}, max {features['head_turn_rate_max_abs']:.1f} deg/s",
        "head_still": f"head turn mean {features['head_turn_rate_mean_abs']:.1f}, max {features['head_turn_rate_max_abs']:.1f} deg/s",
    }[head_tag]
    car_evidence = {
        "yielding": f"near-car slowing context; min distance {features['distance_min']:.2f} m",
        "proceeding": f"near-car proceeding context; min distance {features['distance_min']:.2f} m",
        "conflicted": f"near-car fluctuating/high-jerk context; min distance {features['distance_min']:.2f} m",
        "neutral": f"min distance {features['distance_min']:.2f} m" if np.isfinite(features["distance_min"]) else "distance unavailable",
    }[car_tag]
    return motion_evidence, head_evidence, car_evidence


def color_for_tag(tag):
    return TAG_COLORS.get(tag, TAG_COLORS["mixed_motion"])


def describe_segment(features):
    parts = [
        f"speed {features['speed_start']:.2f}->{features['speed_end']:.2f} m/s",
        f"mean {features['speed_mean']:.2f}, cv {features['speed_cv']:.2f}",
        f"accel min/max {features['accel_min']:.2f}/{features['accel_max']:.2f}",
    ]
    if np.isfinite(features["distance_start"]) and np.isfinite(features["distance_end"]):
        parts.append(f"distance {features['distance_start']:.2f}->{features['distance_end']:.2f} m")
    if np.isfinite(features["head_turn_rate_max_abs"]):
        parts.append(f"head max {features['head_turn_rate_max_abs']:.1f} deg/s")
    if np.isfinite(features["progress_ratio"]):
        parts.append(f"progress ratio {features['progress_ratio']:.2f}")
    return "; ".join(parts)


def extract_features(global_df, macro_row, micro_id, local_start, local_end, local_seg, change_score):
    seg = local_seg.iloc[local_start:local_end + 1].copy()
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)
    if len(seg) < 2:
        return None
    duration = float(time_sec[-1] - time_sec[0])
    if duration < MIN_MICRO_SEGMENT_SECONDS:
        return None

    speed = numeric_series(seg, pick_col(seg, PED_SPEED_COLS, required=True)).to_numpy(dtype=float)
    accel = numeric_series(seg, pick_col(seg, PED_ACCEL_COLS)).to_numpy(dtype=float)
    head_turn = numeric_series(seg, pick_col(seg, HEAD_TURN_COLS), default=0.0).to_numpy(dtype=float)
    head_body = numeric_series(seg, pick_col(seg, HEAD_BODY_DIFF_COLS), default=0.0).to_numpy(dtype=float)
    distance = numeric_series(seg, pick_col(seg, DISTANCE_COLS)).to_numpy(dtype=float)
    closing = numeric_series(seg, pick_col(seg, CLOSING_COLS), default=np.nan).to_numpy(dtype=float)
    distance_change = numeric_series(seg, pick_col(seg, DISTANCE_CHANGE_COLS), default=np.nan).to_numpy(dtype=float)
    car_speed = numeric_series(seg, pick_col(seg, CAR_SPEED_COLS), default=np.nan).to_numpy(dtype=float)
    car_accel = numeric_series(seg, pick_col(seg, CAR_ACCEL_COLS), default=np.nan).to_numpy(dtype=float)
    avatar_gap = numeric_series(seg, pick_col(seg, AVATAR_GAP_COLS), default=np.nan).to_numpy(dtype=float)
    displacement, path_length, progress_ratio = displacement_stats(seg)

    features = {
        "micro_segment_id": micro_id,
        "macro_segment_id": int(macro_row["segment_id"]),
        "macro_label": str(macro_row.get("macro_label", "")),
        "start_idx": int(seg.index[0]),
        "end_idx": int(seg.index[-1]),
        "start_time_sec": float(time_sec[0]),
        "end_time_sec": float(time_sec[-1]),
        "duration_sec": duration,
        "start_frame": seg[FRAME_COL].iloc[0] if FRAME_COL in seg.columns else np.nan,
        "end_frame": seg[FRAME_COL].iloc[-1] if FRAME_COL in seg.columns else np.nan,
        "ScenarioTime_start_raw": seg[TIME_COL].iloc[0] if TIME_COL in seg.columns else np.nan,
        "ScenarioTime_end_raw": seg[TIME_COL].iloc[-1] if TIME_COL in seg.columns else np.nan,
        "change_score_mean": safe_stat(change_score[local_start:local_end + 1], np.nanmean),
        "change_score_max": safe_stat(change_score[local_start:local_end + 1], np.nanmax),
        "speed_start": safe_stat(speed[:1], np.nanmean),
        "speed_end": safe_stat(speed[-1:], np.nanmean),
        "speed_mean": safe_stat(speed, np.nanmean),
        "speed_std": safe_stat(speed, np.nanstd),
        "speed_min": safe_stat(speed, np.nanmin),
        "speed_max": safe_stat(speed, np.nanmax),
        "speed_slope": linear_slope(time_sec, speed),
        "accel_start": safe_stat(accel[:1], np.nanmean),
        "accel_end": safe_stat(accel[-1:], np.nanmean),
        "accel_mean": safe_stat(accel, np.nanmean),
        "accel_std": safe_stat(accel, np.nanstd),
        "accel_min": safe_stat(accel, np.nanmin),
        "accel_max": safe_stat(accel, np.nanmax),
        "accel_jerk_abs_mean": safe_stat(np.abs(np.diff(fill_numeric(accel))), np.nanmean),
        "head_turn_rate_mean_abs": safe_stat(np.abs(head_turn), np.nanmean),
        "head_turn_rate_max_abs": safe_stat(np.abs(head_turn), np.nanmax),
        "head_body_yaw_diff_mean_abs": safe_stat(np.abs(head_body), np.nanmean),
        "head_body_yaw_diff_max_abs": safe_stat(np.abs(head_body), np.nanmax),
        "distance_start": safe_stat(distance[:1], np.nanmean),
        "distance_end": safe_stat(distance[-1:], np.nanmean),
        "distance_mean": safe_stat(distance, np.nanmean),
        "distance_min": safe_stat(distance, np.nanmin),
        "distance_slope": linear_slope(time_sec, distance),
        "distance_closing_mean": safe_stat(closing, np.nanmean),
        "distance_change_mean": safe_stat(distance_change, np.nanmean),
        "car_speed_mean": safe_stat(car_speed, np.nanmean),
        "car_accel_mean": safe_stat(car_accel, np.nanmean),
        "avatar_vr_gap_max": safe_stat(avatar_gap, np.nanmax),
        "displacement_xz": displacement,
        "path_length_xz": path_length,
        "progress_ratio": progress_ratio,
    }
    features["speed_delta"] = features["speed_end"] - features["speed_start"]
    features["speed_cv"] = features["speed_std"] / features["speed_mean"] if features["speed_mean"] > 1e-9 else np.nan
    features["speed_drop"], features["speed_recovery"] = speed_drop_recovery(speed)
    features["zero_velocity_duration_sec"] = zero_speed_duration(speed, time_sec)
    features["distance_delta"] = features["distance_end"] - features["distance_start"]
    motion_tag = classify_motion_tag(features)
    head_tag = classify_head_tag(features)
    car_tag = classify_car_tag(features, motion_tag)
    motion_evidence, head_evidence, car_evidence = tag_evidence(features, motion_tag, head_tag, car_tag)
    features["motion_tag"] = motion_tag
    features["head_tag"] = head_tag
    features["car_tag"] = car_tag
    features["motion_tag_type"] = TAG_OBSERVABILITY[motion_tag]
    features["head_tag_type"] = TAG_OBSERVABILITY[head_tag]
    features["car_tag_type"] = TAG_OBSERVABILITY[car_tag]
    features["has_inferred_behavior"] = any(
        TAG_OBSERVABILITY[tag] == "inferred" for tag in [motion_tag, head_tag, car_tag]
    )
    features["motion_evidence"] = motion_evidence
    features["head_evidence"] = head_evidence
    features["car_context_evidence"] = car_evidence
    features["tagging_note"] = "Tags are separated by dimension; combine motion_tag/head_tag/car_tag later during analysis if needed."
    features["primary_evidence"] = " | ".join([motion_evidence, head_evidence, car_evidence])
    features["supporting_evidence"] = describe_segment(features)
    features["columns_used"] = "ped_speed_xz_smooth|ped_accel_xz_smooth|head_turn_rate|head_body_yaw_diff|car_ped_distance_xz_smooth|distance_closing_smooth|car_speed_xz_smooth|car_accel_xz_smooth|avatar_vr_gap_xz"
    features["notes"] = "Motion/head/car tags are independent descriptive measurements; avatar_vr_gap_xz is audit-only and not used for movement boundaries."
    features["plain_description"] = describe_segment(features)
    features["signals_used_for_boundaries"] = "residual ped_speed_xz_smooth|residual ped_accel_xz_smooth|residual head_turn_rate|raw speed pause/hesitation detectors"
    return features


def build_micro_segments(features_df, macro_df):
    rows = []
    frame_df = features_df.copy()
    frame_df["micro_segment_id"] = -1
    frame_df["motion_tag"] = "unassigned"
    frame_df["head_tag"] = "unassigned"
    frame_df["car_tag"] = "unassigned"
    frame_df["motion_tag_type"] = "unassigned"
    frame_df["head_tag_type"] = "unassigned"
    frame_df["car_tag_type"] = "unassigned"
    frame_df["has_inferred_behavior"] = False
    frame_df["macro_segment_id"] = -1
    frame_df["change_score"] = np.nan
    frame_df["change_score_raw"] = np.nan
    frame_df["ped_speed_trend"] = np.nan
    frame_df["ped_speed_residual"] = np.nan
    frame_df["ped_accel_trend"] = np.nan
    frame_df["ped_accel_residual"] = np.nan
    frame_df["head_turn_trend"] = np.nan
    frame_df["head_turn_residual"] = np.nan
    for tag in MOTION_TAGS:
        frame_df[f"is_motion_{tag}"] = False
    for tag in HEAD_TAGS:
        frame_df[f"is_{tag}"] = False
    for tag in CAR_TAGS:
        frame_df[f"is_car_{tag}"] = False

    next_micro_id = 0
    boundary_records = []

    for _, macro_row in macro_df.iterrows():
        start_idx, end_idx = macro_row_to_indices(macro_row, frame_df)
        macro_seg = frame_df.iloc[start_idx:end_idx + 1].copy()
        boundaries, score, boundary_sources, diagnostics = find_boundaries(macro_seg)
        frame_df.loc[macro_seg.index, "change_score"] = score
        for diag_col, values in diagnostics.items():
            target_col = diag_col
            if diag_col == "speed_trend":
                target_col = "ped_speed_trend"
            elif diag_col == "speed_residual":
                target_col = "ped_speed_residual"
            elif diag_col == "accel_trend":
                target_col = "ped_accel_trend"
            elif diag_col == "accel_residual":
                target_col = "ped_accel_residual"
            frame_df.loc[macro_seg.index, target_col] = values
        frame_df.loc[start_idx:end_idx, "macro_segment_id"] = int(macro_row["segment_id"])

        for boundary in boundaries:
            boundary_records.append({
                "macro_segment_id": int(macro_row["segment_id"]),
                "boundary_idx": int(macro_seg.index[boundary]),
                "boundary_time_sec": float(macro_seg["elapsed_time_sec"].iloc[boundary]),
                "change_score": float(score[boundary]) if len(score) else np.nan,
                "boundary_source": "|".join(sorted(boundary_sources.get(int(boundary), {"unknown"}))),
            })

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            features = extract_features(frame_df, macro_row, next_micro_id, left, right, macro_seg, score)
            if features is None:
                continue
            rows.append(features)
            frame_df.loc[features["start_idx"]:features["end_idx"], "micro_segment_id"] = next_micro_id
            frame_df.loc[features["start_idx"]:features["end_idx"], "motion_tag"] = features["motion_tag"]
            frame_df.loc[features["start_idx"]:features["end_idx"], "head_tag"] = features["head_tag"]
            frame_df.loc[features["start_idx"]:features["end_idx"], "car_tag"] = features["car_tag"]
            frame_df.loc[features["start_idx"]:features["end_idx"], "motion_tag_type"] = features["motion_tag_type"]
            frame_df.loc[features["start_idx"]:features["end_idx"], "head_tag_type"] = features["head_tag_type"]
            frame_df.loc[features["start_idx"]:features["end_idx"], "car_tag_type"] = features["car_tag_type"]
            frame_df.loc[features["start_idx"]:features["end_idx"], "has_inferred_behavior"] = features["has_inferred_behavior"]
            frame_df.loc[features["start_idx"]:features["end_idx"], f"is_motion_{features['motion_tag']}"] = True
            frame_df.loc[features["start_idx"]:features["end_idx"], f"is_{features['head_tag']}"] = True
            frame_df.loc[features["start_idx"]:features["end_idx"], f"is_car_{features['car_tag']}"] = True
            next_micro_id += 1

    return pd.DataFrame(rows), frame_df, pd.DataFrame(boundary_records)


def summarize_segments(segments_df, macro_df):
    if segments_df.empty:
        return pd.DataFrame()
    long_rows = []
    for dimension, column in [("motion", "motion_tag"), ("head", "head_tag"), ("car_context", "car_tag")]:
        dim_df = segments_df.copy()
        dim_df["tag_dimension"] = dimension
        dim_df["tag"] = dim_df[column]
        dim_df["tag_type"] = dim_df["tag"].map(TAG_OBSERVABILITY)
        long_rows.append(dim_df)
    long_df = pd.concat(long_rows, ignore_index=True)
    summary = (
        long_df
        .groupby(["macro_segment_id", "macro_label", "tag_dimension", "tag_type", "tag"], dropna=False)
        .agg(
            count=("micro_segment_id", "count"),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            first_start_time_sec=("start_time_sec", "min"),
            last_end_time_sec=("end_time_sec", "max"),
            speed_mean=("speed_mean", "mean"),
            distance_min=("distance_min", "min"),
        )
        .reset_index()
    )
    duration_map = macro_df.set_index("segment_id")["duration_sec"].to_dict()
    summary["macro_duration_sec"] = summary["macro_segment_id"].map(duration_map)
    summary["percent_of_macro_segment"] = np.where(
        summary["macro_duration_sec"] > 0,
        100.0 * summary["total_duration_sec"] / summary["macro_duration_sec"],
        np.nan,
    )
    return summary.sort_values(["macro_segment_id", "tag_type", "tag_dimension", "first_start_time_sec", "tag"])


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        return plt, Patch
    except ImportError:
        print("Skipping plots: matplotlib is not installed.")
        return None, None


def shade_macro_segments(ax, macro_df, label_top=False):
    for i, row in macro_df.iterrows():
        color = "#F7F7F7" if i % 2 == 0 else "#ECECEC"
        ax.axvspan(row["time_start_sec"], row["time_end_sec"], color=color, alpha=0.45, zorder=0)
        ax.axvline(row["time_start_sec"], color="#7F1D1D", linewidth=1.1, alpha=0.85, linestyle="--", zorder=1)
        if label_top:
            ax.text(
                (row["time_start_sec"] + row["time_end_sec"]) / 2,
                0.98,
                f"M{int(row['segment_id'])}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#7F1D1D",
            )
    if len(macro_df):
        ax.axvline(macro_df["time_end_sec"].iloc[-1], color="#7F1D1D", linewidth=1.1, alpha=0.85, linestyle="--", zorder=1)


def draw_boundaries(ax, boundary_df):
    for t in boundary_df["boundary_time_sec"].dropna().unique():
        ax.axvline(float(t), color="#2563EB", linewidth=0.7, alpha=0.45, zorder=2)


def scenario_time_limits(macro_df, frame_df=None):
    max_time = np.nan
    if macro_df is not None and len(macro_df):
        max_time = float(np.nanmax(macro_df["time_end_sec"]))
    if (not np.isfinite(max_time)) and frame_df is not None and "elapsed_time_sec" in frame_df.columns:
        max_time = float(np.nanmax(frame_df["elapsed_time_sec"]))
    return 0.0, max_time if np.isfinite(max_time) else 1.0


def separated_gantt_rows():
    blocks = []
    y_labels = []
    y_positions = []
    y_lookup = {}
    y = 0
    for tag_type in OBSERVABILITY_ORDER:
        block_start = y
        for dimension, tags in TAG_DIMENSIONS.items():
            for tag in tags:
                if TAG_OBSERVABILITY[tag] != tag_type:
                    continue
                y_lookup[(dimension, tag)] = y
                y_labels.append(f"{dimension}: {tag}")
                y_positions.append(y)
                y += 1
        blocks.append((tag_type, block_start, y - 1))
        y += 0.9
    return y_lookup, y_positions, y_labels, blocks


def draw_separated_gantt(ax, segments_df):
    y_lookup, y_positions, y_labels, blocks = separated_gantt_rows()
    for tag_type, start_y, end_y in blocks:
        color = "#F8FAFC" if tag_type == "observable" else "#F3F0EA"
        ax.axhspan(start_y - 0.5, end_y + 0.5, color=color, alpha=0.65, zorder=-2)
        ax.text(
            0.01,
            (start_y + end_y) / 2.0,
            OBSERVABILITY_LABELS[tag_type],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )
    for _, row in segments_df.iterrows():
        entries = [
            ("motion", row["motion_tag"]),
            ("head", row["head_tag"]),
            ("car_context", row["car_tag"]),
        ]
        for dimension, tag in entries:
            y = y_lookup.get((dimension, tag))
            if y is None:
                continue
            ax.barh(
                y,
                row["duration_sec"],
                left=row["start_time_sec"],
                height=0.52,
                color=color_for_tag(tag),
                edgecolor="white",
                linewidth=0.8,
            )
    if len(blocks) > 1:
        ax.axhline(blocks[0][2] + 0.7, color="#555555", linewidth=1.4, alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    return y_lookup


def presentation_gantt_rows(segments_df):
    """Return compact display rows, omitting unused analytical tags."""
    columns = {"motion": "motion_tag", "head": "head_tag", "car_context": "car_tag"}
    lookup, positions, labels, blocks, colors = {}, [], [], [], {}
    y = 0.0
    for section, candidates in PRESENTATION_ROWS.items():
        rows = [row for row in candidates if row[1] in set(segments_df[columns[row[0]]].astype(str).str.strip())]
        if not rows:
            continue
        start = y
        for dimension, tag, display_label, color in rows:
            lookup[(dimension, tag)] = y
            positions.append(y)
            labels.append(display_label)
            colors[tag] = color
            y += 1.0
        blocks.append((section, start, y - 1.0))
        y += 0.65
    return lookup, positions, labels, blocks, colors


def draw_presentation_gantt(ax, segments_df):
    lookup, positions, labels, blocks, colors = presentation_gantt_rows(segments_df)
    for index, (section, start, end) in enumerate(blocks):
        background = "#EFF6FF" if section == "Observed Motion" else "#FFF7ED"
        ax.axhspan(start - 0.48, end + 0.48, color=background, alpha=0.72, zorder=-2)
        if index:
            ax.axhline(start - 0.33, color="#64748B", linewidth=1.2, alpha=0.8)
        ax.text(
            0.008, (start + end) / 2.0, section, transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=12, fontweight="bold", color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
        )
    columns = {"motion": "motion_tag", "head": "head_tag", "car_context": "car_tag"}
    for _, row in segments_df.iterrows():
        for dimension, column in columns.items():
            tag = str(row[column]).strip()
            y = lookup.get((dimension, tag))
            if y is not None:
                ax.barh(y, row["duration_sec"], left=row["start_time_sec"], height=0.58,
                        color=colors[tag], edgecolor="white", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=12)
    return positions


def tag_legend_handles(Patch):
    handles = []
    for tags in TAG_DIMENSIONS.values():
        for tag in tags:
            handles.append(Patch(facecolor=TAG_COLORS[tag], label=tag))
    return handles


def plot_standard_three_panel(frame_df, macro_df, boundary_df, output_path):
    plt, _ = load_matplotlib()
    if plt is None:
        return False
    time = frame_df["elapsed_time_sec"].to_numpy(dtype=float)
    panels = [
        ("Pedestrian speed", pick_col(frame_df, PED_SPEED_COLS), "m/s"),
        ("Pedestrian acceleration", pick_col(frame_df, PED_ACCEL_COLS), "m/s^2"),
        ("Head turn rate", pick_col(frame_df, HEAD_TURN_COLS), "deg/s"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for ax, (title, col, ylabel) in zip(axes, panels):
        shade_macro_segments(ax, macro_df)
        draw_boundaries(ax, boundary_df)
        if col is not None:
            ax.plot(time, numeric_series(frame_df, col), color="#222222", linewidth=1.3, label=col)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(*scenario_time_limits(macro_df, frame_df))
    axes[-1].set_xlabel("Scenario elapsed time, seconds")
    fig.suptitle("Micro-Segmentation Change Boundaries: Core Signals", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_stacked_comparison(frame_df, macro_df, segments_df, boundary_df, output_path):
    plt, Patch = load_matplotlib()
    if plt is None:
        return False
    time = frame_df["elapsed_time_sec"].to_numpy(dtype=float)
    panels = [
        ("Speed", pick_col(frame_df, PED_SPEED_COLS), "m/s"),
        ("Acceleration", pick_col(frame_df, PED_ACCEL_COLS), "m/s^2"),
        ("Head turn", pick_col(frame_df, HEAD_TURN_COLS), "deg/s"),
        ("Distance to car", pick_col(frame_df, DISTANCE_COLS), "m"),
        ("Distance closing/change", pick_col(frame_df, CLOSING_COLS) or pick_col(frame_df, DISTANCE_CHANGE_COLS), "value"),
        ("Car speed/accel", pick_col(frame_df, CAR_SPEED_COLS) or pick_col(frame_df, CAR_ACCEL_COLS), "value"),
    ]
    fig, axes = plt.subplots(7, 1, figsize=(22, 14), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1.6]})
    for ax, (title, col, ylabel) in zip(axes[:6], panels):
        shade_macro_segments(ax, macro_df)
        draw_boundaries(ax, boundary_df)
        if col is not None:
            ax.plot(time, numeric_series(frame_df, col), color="#222222", linewidth=1.1, label=col)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "signal unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(*scenario_time_limits(macro_df, frame_df))

    ax = axes[-1]
    shade_macro_segments(ax, macro_df, label_top=True)
    draw_separated_gantt(ax, segments_df)
    ax.invert_yaxis()
    ax.set_title("Observable kinematics vs. inferred latent tags", loc="left", fontsize=10)
    ax.set_xlabel("Scenario elapsed time, seconds")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(*scenario_time_limits(macro_df, frame_df))
    handles = tag_legend_handles(Patch)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Stacked Signal Comparison With Macro and Micro Segments", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def build_gantt_figure(plt, Patch, segments_df, macro_df):
    _, y_positions, _, _, _ = presentation_gantt_rows(segments_df)
    fig, ax = plt.subplots(figsize=(20, max(7, 0.62 * len(y_positions) + 2.4)))
    shade_macro_segments(ax, macro_df, label_top=True)
    draw_presentation_gantt(ax, segments_df)
    ax.invert_yaxis()
    ax.set_xlabel("Scenario elapsed time, seconds", fontsize=13)
    ax.set_ylabel("")
    ax.set_title("Observed Motion and Possible Behaviors", fontsize=18, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(*scenario_time_limits(macro_df))
    fig.text(0.5, 0.018, PRESENTATION_FOOTNOTE, ha="center", va="bottom", fontsize=10.5, color="#475569")
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    return fig, ax


def plot_gantt(segments_df, macro_df, output_path):
    plt, Patch = load_matplotlib()
    if plt is None:
        return False
    fig, _ = build_gantt_figure(plt, Patch, segments_df, macro_df)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    args = parse_args()
    if not args.feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {args.feature_csv}")
    if not args.macro_csv.exists():
        raise FileNotFoundError(f"Macro segment CSV not found: {args.macro_csv}")

    output_dir = args.output_dir / "v6"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = clean_columns(pd.read_csv(args.feature_csv))
    macro_df = clean_columns(pd.read_csv(args.macro_csv))
    pick_col(features, PED_SPEED_COLS, required=True, purpose="pedestrian speed")
    features = features.reset_index(drop=True)
    features["elapsed_time_sec"], time_source = reconstruct_elapsed_time(features)

    if not is_seconds_like(features["elapsed_time_sec"].to_numpy(dtype=float)):
        duration = features["elapsed_time_sec"].iloc[-1] - features["elapsed_time_sec"].iloc[0]
        raise ValueError(f"Elapsed time is not in expected seconds range after reconstruction: {duration}")

    segments_df, frame_df, boundary_df = build_micro_segments(features, macro_df)
    summary_df = summarize_segments(segments_df, macro_df)
    definitions_df = pd.DataFrame(
        [
            {
                "tag_dimension": dimension,
                "tag": tag,
                "tag_type": TAG_OBSERVABILITY[tag],
                "definition": TAG_DEFINITIONS[tag],
                "color": TAG_COLORS[tag],
                "labeling_note": (
                    "Observable tags are direct sensor-threshold measurements; inferred tags are structured hypotheses. "
                    "Motion, head, and car-context tags are intentionally not pre-fused."
                ),
            }
            for dimension, tags in TAG_DIMENSIONS.items()
            for tag in tags
        ]
    )

    identity = args.scenario_stem or f"PedNYC{args.participant}_scenario{args.scenario}"
    segments_csv = output_dir / f"micro_segments_descriptive_{identity}_v6.csv"
    frame_csv = output_dir / f"features_with_descriptive_micro_segments_{identity}_v6.csv"
    summary_csv = output_dir / f"micro_segment_summary_by_macro_{identity}_v6.csv"
    boundary_csv = output_dir / f"micro_change_boundaries_{identity}_v6.csv"
    definitions_csv = output_dir / "micro_event_tag_definitions_v6.csv"
    standard_png = output_dir / f"micro_standard_3panel_{identity}_v6.png"
    stacked_png = output_dir / f"micro_stacked_observable_inferred_{identity}_v6.png"
    gantt_png = output_dir / f"micro_gantt_observable_inferred_{identity}_v6.png"

    segments_df.to_csv(segments_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    boundary_df.to_csv(boundary_csv, index=False)
    definitions_df.to_csv(definitions_csv, index=False)

    wrote_standard = wrote_stacked = wrote_gantt = False
    if not args.no_plot:
        wrote_standard = plot_standard_three_panel(frame_df, macro_df, boundary_df, standard_png)
        wrote_stacked = plot_stacked_comparison(frame_df, macro_df, segments_df, boundary_df, stacked_png)
        wrote_gantt = plot_gantt(segments_df, macro_df, gantt_png)

    print("\nDone.")
    print(f"Time source used: {time_source}")
    print(f"Scenario elapsed duration: {frame_df['elapsed_time_sec'].iloc[-1]:.3f} seconds")
    print(f"Saved descriptive micro-segments CSV: {segments_csv}")
    print(f"Saved frame-level CSV: {frame_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved boundary CSV: {boundary_csv}")
    print(f"Saved event tag definitions CSV: {definitions_csv}")
    if wrote_standard:
        print(f"Saved standard 3-panel plot: {standard_png}")
    if wrote_stacked:
        print(f"Saved stacked comparison plot: {stacked_png}")
    if wrote_gantt:
        print(f"Saved Gantt plot: {gantt_png}")
    print(f"\nMicro-segments with any inferred tag: {int(segments_df['has_inferred_behavior'].sum()) if not segments_df.empty else 0}")
    print("\nMicro-segments by motion tag:")
    print(segments_df["motion_tag"].value_counts().to_string() if not segments_df.empty else "none")
    print("\nMicro-segments by head tag:")
    print(segments_df["head_tag"].value_counts().to_string() if not segments_df.empty else "none")
    print("\nMicro-segments by car-context tag:")
    print(segments_df["car_tag"].value_counts().to_string() if not segments_df.empty else "none")


if __name__ == "__main__":
    main()
