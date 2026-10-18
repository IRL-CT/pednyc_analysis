import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the established PedNYC macro-v2 algorithm.")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "graphs" / "macro_segmentation")
    parser.add_argument("--segments-csv", type=Path)
    parser.add_argument("--plot-path", type=Path)
    parser.add_argument(
        "--conflict-events-csv",
        type=Path,
        help="Conflict-zone event table; when supplied, actor occupancy is overlaid on the plot.",
    )
    parser.add_argument(
        "--recording-identifier",
        help="Row to use from the conflict event table (for example, PedNYC1_scenario3).",
    )
    return parser.parse_args()


ARGS = parse_args()
INPUT_CSV = ARGS.input_csv

OUTPUT_DIR = ARGS.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTS_CSV = ARGS.segments_csv or (OUTPUT_DIR / "macro_segments_PedNYC1_scenario3_v2.csv")
PLOT_PATH = ARGS.plot_path or (OUTPUT_DIR / "macro_segments_PedNYC1_scenario3_v2.png")
SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)


def infer_recording_identifier(input_csv):
    """Infer the canonical recording ID from a scenario feature filename."""
    match = re.search(r"(PedNYC\d+_scenario\d+)", Path(input_csv).stem, flags=re.IGNORECASE)
    return match.group(1) if match else None


def infer_conflict_events_path(input_csv):
    """Locate the scenario-level event table associated with a feature file."""
    identifier = infer_recording_identifier(input_csv)
    match = re.search(r"scenario(\d+)", identifier or "", flags=re.IGNORECASE)
    if not match:
        return None
    return ROOT / "outputs" / "conflict_zone_analysis" / f"scenario_{int(match.group(1)):02d}" / "conflict_zone_events.csv"


def load_conflict_zone_occupancy(events_csv, recording_identifier):
    """Return valid pedestrian/vehicle conflict-zone occupancy intervals."""
    if events_csv is None or recording_identifier is None or not Path(events_csv).is_file():
        return []

    events = pd.read_csv(events_csv)
    if "recording_identifier" not in events.columns:
        return []
    selected = events.loc[
        events["recording_identifier"].astype(str).str.casefold()
        == str(recording_identifier).casefold()
    ]
    if selected.empty:
        return []

    row = selected.iloc[0]
    intervals = []
    for actor, start_col, end_col in (
        ("Pedestrian", "pedestrian_entry_time", "pedestrian_exit_time"),
        ("Vehicle", "vehicle_actual_entry_time", "vehicle_actual_exit_time"),
    ):
        if start_col not in row.index or end_col not in row.index:
            continue
        start = pd.to_numeric(row[start_col], errors="coerce")
        end = pd.to_numeric(row[end_col], errors="coerce")
        if np.isfinite(start) and np.isfinite(end) and end >= start:
            intervals.append((actor, float(start), float(end)))
    return intervals


# ============================================================
# Columns
# ============================================================

TIME_COL = "ScenarioTime"
DT_COL = "dt"
SPEED_COL = "ped_speed_xz_smooth"

OPTIONAL_DISTANCE_COL = "car_ped_distance_xz_smooth"
OPTIONAL_CLOSING_COL = "distance_closing_smooth"


# ============================================================
# Tunable settings
# ============================================================

STOP_SPEED = 0.15
WALK_SPEED = 0.30

# This is only used to find the general region.
# The final boundary is refined to a local kink.
ACCEL_SLOPE = 0.12
HARD_DECEL_SLOPE = -0.25

SMOOTH_WINDOW_SEC = 0.45
MIN_SUSTAIN_SEC = 0.45

# Larger backtrack means the boundary can move farther back to the true kink.
START_BACKTRACK_SEC = 2.50
DECEL_BACKTRACK_SEC = 2.00

# Used to find first major movement, not tiny early walking.
MAIN_SPEED_FRACTION = 0.40
MAIN_SPEED_MIN = 0.60
FUTURE_MAIN_WINDOW_SEC = 4.00

# Used to find acceleration end.
ACCEL_END_SEARCH_SEC = 4.00

# Used to avoid calling early middle wiggles the final deceleration.
FINAL_DECEL_AFTER_FRACTION = 0.50

NEAR_CAR_DISTANCE = 10.0


# ============================================================
# Helper functions
# ============================================================

def seconds_to_frames(time_sec, seconds):
    dt = np.nanmedian(np.diff(time_sec))
    if not np.isfinite(dt) or dt <= 0:
        return 3
    return max(1, int(round(seconds / dt)))


def odd_window_from_seconds(time_sec, seconds):
    window = seconds_to_frames(time_sec, seconds)
    window = max(window, 3)
    if window % 2 == 0:
        window += 1
    return window


def smooth_series(values, time_sec, window_sec):
    window = odd_window_from_seconds(time_sec, window_sec)

    s = pd.Series(values)
    smoothed = (
        s.rolling(window=window, center=True, min_periods=1)
         .median()
         .rolling(window=window, center=True, min_periods=1)
         .mean()
    )

    return smoothed.to_numpy()


def build_elapsed_time_seconds(df):
    """
    Use dt to build actual elapsed scenario time.
    This should give about 31 seconds for this scenario.
    Raw ScenarioTime is preserved in the output CSV.
    """
    if DT_COL in df.columns:
        dt = pd.to_numeric(df[DT_COL], errors="coerce").to_numpy(dtype=float)

        good_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
        if len(good_dt) > 0:
            median_dt = float(np.nanmedian(good_dt))
        else:
            median_dt = 1 / 30

        dt_clean = dt.copy()
        bad = ~np.isfinite(dt_clean) | (dt_clean <= 0) | (dt_clean > 1.0)
        dt_clean[bad] = median_dt

        time_sec = np.zeros(len(df), dtype=float)
        time_sec[1:] = np.cumsum(dt_clean[1:])

        return time_sec, "cumulative_dt_seconds"

    raw_time = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    elapsed = raw_time - raw_time[0]

    return elapsed, "raw_ScenarioTime_minus_start"


def find_true_runs(mask):
    runs = []
    in_run = False
    start = None

    for i, value in enumerate(mask):
        if value and not in_run:
            start = i
            in_run = True
        elif not value and in_run:
            runs.append((start, i - 1))
            in_run = False

    if in_run:
        runs.append((start, len(mask) - 1))

    return runs


def keep_long_runs(runs, min_frames):
    return [(s, e) for s, e in runs if (e - s + 1) >= min_frames]


def argmin_between(y, start_idx, end_idx):
    start_idx = max(0, int(start_idx))
    end_idx = min(len(y) - 1, int(end_idx))

    if end_idx <= start_idx:
        return start_idx

    local_idx = int(np.nanargmin(y[start_idx:end_idx + 1]))
    return start_idx + local_idx


def argmax_between(y, start_idx, end_idx):
    start_idx = max(0, int(start_idx))
    end_idx = min(len(y) - 1, int(end_idx))

    if end_idx <= start_idx:
        return start_idx

    local_idx = int(np.nanargmax(y[start_idx:end_idx + 1]))
    return start_idx + local_idx


def first_major_accel_run(accel_runs, speed_trend, time_sec):
    """
    Ignore early tiny movement.
    Pick the first acceleration run that leads to real main movement soon after.
    """
    max_speed = float(np.nanmax(speed_trend))
    main_speed = max(MAIN_SPEED_MIN, MAIN_SPEED_FRACTION * max_speed)
    future_frames = seconds_to_frames(time_sec, FUTURE_MAIN_WINDOW_SEC)

    for s, e in accel_runs:
        future_end = min(len(speed_trend) - 1, e + future_frames)
        future_max = float(np.nanmax(speed_trend[s:future_end + 1]))

        if future_max >= main_speed:
            return s, e

    return accel_runs[0] if accel_runs else None


def refine_start_to_kink(speed_trend, accel_start_raw, time_sec):
    """
    Move the acceleration boundary backward to the local low point
    before the sustained rise. This places it on the visual kink.
    """
    lookback = seconds_to_frames(time_sec, START_BACKTRACK_SEC)
    search_start = max(0, accel_start_raw - lookback)
    search_end = accel_start_raw

    return argmin_between(speed_trend, search_start, search_end)


def refine_decel_to_kink(speed_trend, decel_start_raw, time_sec):
    """
    Move the deceleration boundary backward to the local high point
    before the sustained final drop.
    """
    lookback = seconds_to_frames(time_sec, DECEL_BACKTRACK_SEC)
    search_start = max(0, decel_start_raw - lookback)
    search_end = decel_start_raw

    return argmax_between(speed_trend, search_start, search_end)


def refine_stop_to_valley(speed_trend, stop_raw, time_sec):
    """
    Move the final stopped boundary to the valley at the bottom
    of the final slowdown.
    """
    window = seconds_to_frames(time_sec, 0.90)
    search_start = max(0, stop_raw - window)
    search_end = min(len(speed_trend) - 1, stop_raw + window)

    return argmin_between(speed_trend, search_start, search_end)


def clean_boundaries(boundaries, n):
    cleaned = []

    for b in boundaries:
        b = int(np.clip(b, 0, n - 1))

        if not cleaned or b > cleaned[-1]:
            cleaned.append(b)

    return cleaned


# ============================================================
# Boundary detection
# ============================================================

def detect_macro_boundaries(time_sec, speed_trend, slope):
    n = len(time_sec)
    total_duration = float(time_sec[-1] - time_sec[0])

    min_frames = seconds_to_frames(time_sec, MIN_SUSTAIN_SEC)

    low_motion = speed_trend < STOP_SPEED
    accelerating = (slope > ACCEL_SLOPE) & (speed_trend > 0.03)
    hard_decelerating = (slope < HARD_DECEL_SLOPE) & (speed_trend > STOP_SPEED)

    accel_runs = keep_long_runs(find_true_runs(accelerating), min_frames)
    low_runs = keep_long_runs(find_true_runs(low_motion), min_frames)
    hard_decel_runs = keep_long_runs(find_true_runs(hard_decelerating), min_frames)

    # ------------------------------------------------------------
    # 1. Start of acceleration: use slope to find region,
    # then local minimum to place boundary on kink.
    # ------------------------------------------------------------
    major_accel = first_major_accel_run(accel_runs, speed_trend, time_sec)

    if major_accel is not None:
        accel_start_raw, accel_end_raw = major_accel
        movement_start = refine_start_to_kink(
            speed_trend,
            accel_start_raw,
            time_sec
        )
    else:
        # Fallback
        walk_idxs = np.where(speed_trend >= WALK_SPEED)[0]
        movement_start = int(walk_idxs[0]) if len(walk_idxs) else 0
        accel_end_raw = movement_start

    # ------------------------------------------------------------
    # 2. End of acceleration: first local high after the rise.
    # This should land near the top of the first big ramp.
    # ------------------------------------------------------------
    accel_search_start = movement_start + seconds_to_frames(time_sec, 0.50)
    accel_search_end = movement_start + seconds_to_frames(time_sec, ACCEL_END_SEARCH_SEC)
    accel_search_end = min(n - 1, accel_search_end)

    if accel_search_end > accel_search_start:
        accel_end = argmax_between(
            speed_trend,
            accel_search_start,
            accel_search_end
        )
    else:
        accel_end = min(n - 1, movement_start + min_frames)

    # ------------------------------------------------------------
    # 3. Find the post-movement low-motion region.
    # This is the first real low-speed run after the main action.
    # ------------------------------------------------------------
    peak_idx = int(np.nanargmax(speed_trend))
    midpoint_time = total_duration * 0.50

    final_low_raw = None

    for s, e in low_runs:
        if s > peak_idx and time_sec[s] > midpoint_time:
            final_low_raw = s
            break

    if final_low_raw is None:
        after_peak = np.where(
            (np.arange(n) > peak_idx) &
            (speed_trend < STOP_SPEED)
        )[0]

        final_low_raw = int(after_peak[0]) if len(after_peak) else n - 1

    stop_start = refine_stop_to_valley(
        speed_trend,
        final_low_raw,
        time_sec
    )

    # ------------------------------------------------------------
    # 4. Start of final deceleration.
    # Use hard negative slope only after the middle of the trial.
    # Then refine backward to local max/kink.
    # ------------------------------------------------------------
    decel_candidates = []

    for s, e in hard_decel_runs:
        if s <= accel_end:
            continue

        if s >= stop_start:
            continue

        if time_sec[s] < total_duration * FINAL_DECEL_AFTER_FRACTION:
            continue

        drop_to_stop = float(speed_trend[s] - np.nanmin(speed_trend[s:stop_start + 1]))

        if drop_to_stop >= 0.35:
            decel_candidates.append((s, e, drop_to_stop))

    if decel_candidates:
        # Choose the first strong final-deceleration run.
        decel_start_raw = sorted(decel_candidates, key=lambda x: x[0])[0][0]

        decel_start = refine_decel_to_kink(
            speed_trend,
            decel_start_raw,
            time_sec
        )
    else:
        # Fallback: use highest point before stop.
        decel_start = argmax_between(
            speed_trend,
            accel_end,
            stop_start
        )

    boundaries = clean_boundaries(
        [
            0,
            movement_start,
            accel_end,
            decel_start,
            stop_start,
            n - 1,
        ],
        n
    )

    return boundaries


# ============================================================
# Segment stats and labels
# ============================================================

def assign_labels(boundaries, df):
    base_labels = [
        "pre_movement_low_motion",
        "acceleration",
        "main_movement",
        "deceleration",
        "post_movement_low_motion",
    ]

    labels = []

    for i in range(len(boundaries) - 1):
        if i < len(base_labels):
            labels.append(base_labels[i])
        else:
            labels.append("extra_macro_segment")

    if OPTIONAL_DISTANCE_COL in df.columns:
        for i, label in enumerate(labels):
            if label == "main_movement":
                s = boundaries[i]
                e = boundaries[i + 1]

                dist = pd.to_numeric(
                    df[OPTIONAL_DISTANCE_COL].iloc[s:e + 1],
                    errors="coerce"
                )

                min_dist = float(np.nanmin(dist))

                if np.isfinite(min_dist) and min_dist <= NEAR_CAR_DISTANCE:
                    labels[i] = "crossing_near_car"

    return labels


def segment_stats(df, start_idx, end_idx, time_sec, speed_trend):
    seg = df.iloc[start_idx:end_idx + 1]

    seg_t = time_sec[start_idx:end_idx + 1]
    seg_speed = speed_trend[start_idx:end_idx + 1]

    duration = float(seg_t[-1] - seg_t[0])
    start_speed = float(seg_speed[0])
    end_speed = float(seg_speed[-1])
    mean_speed = float(np.nanmean(seg_speed))
    max_speed = float(np.nanmax(seg_speed))
    min_speed = float(np.nanmin(seg_speed))

    if len(seg_t) >= 2:
        x = seg_t - seg_t[0]
        mean_slope = float(np.polyfit(x, seg_speed, 1)[0])
    else:
        mean_slope = 0.0

    out = {
        "ScenarioTime_start_raw": float(seg[TIME_COL].iloc[0]),
        "ScenarioTime_end_raw": float(seg[TIME_COL].iloc[-1]),
        "time_start_sec": float(seg_t[0]),
        "time_end_sec": float(seg_t[-1]),
        "duration_sec": duration,
        "start_speed": start_speed,
        "end_speed": end_speed,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "min_speed": min_speed,
        "mean_slope": mean_slope,
    }

    if OPTIONAL_DISTANCE_COL in df.columns:
        dist = pd.to_numeric(seg[OPTIONAL_DISTANCE_COL], errors="coerce")
        out["min_car_ped_distance_xz_smooth"] = float(np.nanmin(dist))
        out["mean_car_ped_distance_xz_smooth"] = float(np.nanmean(dist))

    if OPTIONAL_CLOSING_COL in df.columns:
        closing = pd.to_numeric(seg[OPTIONAL_CLOSING_COL], errors="coerce")
        out["mean_distance_closing_smooth"] = float(np.nanmean(closing))

    return out


# ============================================================
# Plot
# ============================================================

def plot_segments(time_sec, speed_raw, speed_trend, boundaries, labels, conflict_occupancy=None):
    color_map = {
        "pre_movement_low_motion": "#f4cccc",
        "acceleration": "#d9ead3",
        "main_movement": "#cfe2f3",
        "crossing_near_car": "#cfe2f3",
        "deceleration": "#fce5cd",
        "post_movement_low_motion": "#f4cccc",
        "extra_macro_segment": "#eeeeee",
    }

    fig, ax = plt.subplots(figsize=(18, 8))

    used_labels = set()

    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        label = labels[i]

        ax.axvspan(
            time_sec[s],
            time_sec[e],
            color=color_map.get(label, "#eeeeee"),
            alpha=0.35,
            label=label if label not in used_labels else None
        )

        used_labels.add(label)

        mid = (time_sec[s] + time_sec[e]) / 2
        y_top = np.nanmax(speed_raw) * 0.92

        ax.text(
            mid,
            y_top,
            label,
            ha="center",
            va="top",
            fontsize=10
        )

    conflict_styles = {
        "Pedestrian": {"color": "#7b3294", "hatch": "///"},
        "Vehicle": {"color": "#008837", "hatch": "\\\\\\"},
    }
    for actor, start, end in conflict_occupancy or []:
        style = conflict_styles[actor]
        ax.axvspan(
            start,
            end,
            facecolor=style["color"],
            edgecolor=style["color"],
            alpha=0.16,
            hatch=style["hatch"],
            linewidth=1.5,
            label=f"{actor} in conflict zone",
            zorder=2,
        )

    ax.plot(
        time_sec,
        speed_raw,
        linewidth=2,
        label="ped_speed_xz_smooth"
    )

    ax.plot(
        time_sec,
        speed_trend,
        linestyle="--",
        linewidth=2,
        label="macro trend smoothing"
    )

    for b in boundaries:
        ax.axvline(
            time_sec[b],
            color="darkred",
            linewidth=2,
            alpha=0.85
        )

    ax.set_title("Pedestrian Macro Segmentation V2: Kink-Aligned")
    ax.set_xlabel("Scenario elapsed time, seconds")
    ax.set_ylabel("Pedestrian speed XZ smooth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    # Fix hidden spaces in column names.
    df.columns = df.columns.astype(str).str.strip()

    required = [TIME_COL, SPEED_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("\nMissing required columns:")
        for c in missing:
            print(f"  - {repr(c)}")

        print("\nAvailable columns:")
        for c in df.columns:
            print(f"  - {repr(c)}")

        raise ValueError("Required columns missing.")

    df = df.sort_values(TIME_COL).reset_index(drop=True)

    speed_raw = pd.to_numeric(df[SPEED_COL], errors="coerce").to_numpy(dtype=float)
    time_sec, time_source = build_elapsed_time_seconds(df)

    valid = np.isfinite(speed_raw) & np.isfinite(time_sec)
    df = df.loc[valid].reset_index(drop=True)
    speed_raw = speed_raw[valid]
    time_sec = time_sec[valid]

    speed_trend = smooth_series(
        speed_raw,
        time_sec,
        window_sec=SMOOTH_WINDOW_SEC
    )

    slope = np.gradient(speed_trend, time_sec)

    boundaries = detect_macro_boundaries(
        time_sec=time_sec,
        speed_trend=speed_trend,
        slope=slope
    )

    labels = assign_labels(boundaries, df)
    conflict_occupancy = load_conflict_zone_occupancy(
        ARGS.conflict_events_csv or infer_conflict_events_path(INPUT_CSV),
        ARGS.recording_identifier or infer_recording_identifier(INPUT_CSV),
    )

    rows = []

    for seg_id in range(len(boundaries) - 1):
        s = boundaries[seg_id]
        e = boundaries[seg_id + 1]

        stats = segment_stats(
            df=df,
            start_idx=s,
            end_idx=e,
            time_sec=time_sec,
            speed_trend=speed_trend
        )

        rows.append({
            "segment_id": seg_id,
            "macro_label": labels[seg_id],
            "start_idx": s,
            "end_idx": e,
            **stats,
        })

    segments_df = pd.DataFrame(rows)
    segments_df.to_csv(SEGMENTS_CSV, index=False)

    plot_segments(
        time_sec=time_sec,
        speed_raw=speed_raw,
        speed_trend=speed_trend,
        boundaries=boundaries,
        labels=labels,
        conflict_occupancy=conflict_occupancy,
    )

    print("\nDone.")
    print(f"Time source used: {time_source}")
    print(f"Scenario duration plotted: {time_sec[-1]:.3f} seconds")

    print(f"\nSaved macro segment CSV to:\n  {SEGMENTS_CSV}")
    print(f"Saved macro segmentation plot to:\n  {PLOT_PATH}")
    if conflict_occupancy:
        print("Conflict-zone occupancy overlaid:")
        for actor, start, end in conflict_occupancy:
            print(f"  {actor}: {start:.3f}-{end:.3f} seconds")

    print("\nMacro segments:")
    print(
        segments_df[
            [
                "segment_id",
                "macro_label",
                "time_start_sec",
                "time_end_sec",
                "duration_sec",
                "mean_speed",
                "mean_slope",
            ]
        ]
    )


if __name__ == "__main__":
    main()
