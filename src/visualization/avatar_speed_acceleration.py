"""Plot raw avatar channels and unsmoothed finite-difference kinematics.

The workflow intentionally preserves source order and extreme values. A speed
is only calculated when the current row and its immediate predecessor have a
finite, strictly increasing ``ScenarioTime`` and the required positions are
numeric. Acceleration is the finite difference of consecutive calculated
speeds using the current row's actual time difference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.avatar_position_rotation import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MEASUREMENTS,
    PARTICIPANTS,
    discover_decoded_csvs,
    inspect_header,
    map_normalized_columns,
    normalize_header,
    read_relevant_data,
)


TIME_COLUMN = "ScenarioTime"
POSITION_COLUMNS = ("B Avatar Pos X", "B Avatar Pos Y", "B Avatar Pos Z")
ROTATION_COLUMNS = ("B Avatar Rot X", "B Avatar Rot Y", "B Avatar Rot Z")
SPEED_COLUMNS = ("avatar_speed_xz", "avatar_speed_3d")
ACCELERATION_COLUMNS = ("avatar_acceleration_xz", "avatar_acceleration_3d")
DEFAULT_SPEED_THRESHOLD = 5.0
DEFAULT_ACCELERATION_THRESHOLD = 20.0


@dataclass
class ProcessResult:
    participant: str
    scenario: str
    source_path: Path
    figure_path: Path | None
    sample_path: Path | None
    summary: dict[str, object] | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--participants", nargs="+", choices=PARTICIPANTS, default=list(PARTICIPANTS)
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--position-units",
        choices=("raw", "meters"),
        default="raw",
        help=(
            "Use 'meters' only when calibration/documentation confirms the avatar "
            "positions are metric. Meter-based diagnostic flags are disabled for raw units."
        ),
    )
    parser.add_argument("--speed-threshold", type=float, default=DEFAULT_SPEED_THRESHOLD)
    parser.add_argument(
        "--acceleration-threshold", type=float, default=DEFAULT_ACCELERATION_THRESHOLD
    )
    return parser.parse_args()


def calculate_avatar_kinematics(
    time: pd.Series,
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
    rotations: dict[str, pd.Series] | None = None,
    *,
    thresholds_enabled: bool = False,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    acceleration_threshold: float = DEFAULT_ACCELERATION_THRESHOLD,
) -> pd.DataFrame:
    """Return unsmoothed sample-level derivatives while preserving row order."""
    series = [time, x, y, z]
    if len({len(item) for item in series}) != 1:
        raise ValueError("time and position series must have equal lengths")

    result = pd.DataFrame(
        {
            # CSV header is physical line 1, so the first ordinary source row is 2.
            "original_row_number": np.arange(2, len(time) + 2, dtype=int),
            TIME_COLUMN: pd.to_numeric(time, errors="coerce").to_numpy(dtype=float),
            POSITION_COLUMNS[0]: pd.to_numeric(x, errors="coerce").to_numpy(dtype=float),
            POSITION_COLUMNS[1]: pd.to_numeric(y, errors="coerce").to_numpy(dtype=float),
            POSITION_COLUMNS[2]: pd.to_numeric(z, errors="coerce").to_numpy(dtype=float),
        }
    )
    if rotations:
        for name in ROTATION_COLUMNS:
            result[name] = pd.to_numeric(rotations[name], errors="coerce").to_numpy(dtype=float)

    result["dt"] = result[TIME_COLUMN].diff()
    result["dx"] = result[POSITION_COLUMNS[0]].diff()
    result["dy"] = result[POSITION_COLUMNS[1]].diff()
    result["dz"] = result[POSITION_COLUMNS[2]].diff()

    status = np.full(len(result), "valid", dtype=object)
    if len(result):
        status[0] = "no_preceding_sample"
    missing_time = result["dt"].isna().to_numpy(copy=True)
    duplicate_time = result["dt"].eq(0).to_numpy(copy=True)
    negative_time = result["dt"].lt(0).to_numpy(copy=True)
    if len(result):
        missing_time[0] = False
    status[missing_time] = "missing_time"
    status[duplicate_time] = "duplicate_or_zero_time"
    status[negative_time] = "negative_time"
    result["time_interval_status"] = status
    result["valid_time_interval"] = result["time_interval_status"].eq("valid")

    valid_dt = result["valid_time_interval"]
    valid_xz = valid_dt & result[["dx", "dz"]].notna().all(axis=1)
    valid_3d = valid_dt & result[["dx", "dy", "dz"]].notna().all(axis=1)
    result["avatar_speed_xz"] = np.nan
    result["avatar_speed_3d"] = np.nan
    result.loc[valid_xz, "avatar_speed_xz"] = (
        np.hypot(result.loc[valid_xz, "dx"], result.loc[valid_xz, "dz"])
        / result.loc[valid_xz, "dt"]
    )
    result.loc[valid_3d, "avatar_speed_3d"] = (
        np.sqrt(
            result.loc[valid_3d, "dx"] ** 2
            + result.loc[valid_3d, "dy"] ** 2
            + result.loc[valid_3d, "dz"] ** 2
        )
        / result.loc[valid_3d, "dt"]
    )

    # Acceleration is deliberately derived from speed, not position.
    result["avatar_acceleration_xz"] = result["avatar_speed_xz"].diff() / result["dt"]
    result["avatar_acceleration_3d"] = result["avatar_speed_3d"].diff() / result["dt"]
    result.loc[~valid_dt, list(ACCELERATION_COLUMNS)] = np.nan

    if thresholds_enabled:
        result["diagnostic_speed_flag"] = result[list(SPEED_COLUMNS)].gt(
            speed_threshold
        ).any(axis=1)
        result["diagnostic_acceleration_flag"] = result[
            list(ACCELERATION_COLUMNS)
        ].abs().gt(acceleration_threshold).any(axis=1)
    else:
        result["diagnostic_speed_flag"] = pd.Series(
            pd.array([pd.NA] * len(result), dtype="boolean")
        )
        result["diagnostic_acceleration_flag"] = pd.Series(
            pd.array([pd.NA] * len(result), dtype="boolean")
        )
    return result


def _finite_max_and_index(frame: pd.DataFrame, columns: tuple[str, ...], absolute: bool):
    values = frame.loc[:, columns].abs() if absolute else frame.loc[:, columns]
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).any():
        return np.nan, None, None
    flat_index = int(np.nanargmax(array))
    row_index, column_index = np.unravel_index(flat_index, array.shape)
    return float(array[row_index, column_index]), int(row_index), columns[column_index]


def _percentile(series: pd.Series, percentile: float = 99.0) -> float:
    finite = series[np.isfinite(series.to_numpy(dtype=float))]
    return float(np.percentile(finite, percentile)) if len(finite) else np.nan


def summarize_samples(
    samples: pd.DataFrame,
    participant: str,
    scenario: str,
    source_path: Path,
    thresholds_enabled: bool,
) -> dict[str, object]:
    speed_max, speed_row, speed_metric = _finite_max_and_index(
        samples, SPEED_COLUMNS, absolute=False
    )
    acceleration_max, acceleration_row, acceleration_metric = _finite_max_and_index(
        samples, ACCELERATION_COLUMNS, absolute=True
    )
    interval_status = samples["time_interval_status"]
    invalid_mask = interval_status.isin(
        ("missing_time", "duplicate_or_zero_time", "negative_time")
    )

    def location(row: int | None, column: str | None, prefix: str) -> dict[str, object]:
        if row is None:
            return {
                f"{prefix}_metric": None,
                f"{prefix}_time": np.nan,
                f"{prefix}_source_row": pd.NA,
                f"{prefix}_dt": np.nan,
            }
        return {
            f"{prefix}_metric": column,
            f"{prefix}_time": samples.at[row, TIME_COLUMN],
            f"{prefix}_source_row": int(samples.at[row, "original_row_number"]),
            f"{prefix}_dt": samples.at[row, "dt"],
        }

    summary: dict[str, object] = {
        "participant": participant,
        "scenario": scenario,
        "source_file": str(source_path),
        "number_of_rows": len(samples),
        "valid_time_intervals": int(interval_status.eq("valid").sum()),
        "invalid_or_duplicate_time_intervals": int(invalid_mask.sum()),
        "missing_time_intervals": int(interval_status.eq("missing_time").sum()),
        "duplicate_or_zero_time_intervals": int(
            interval_status.eq("duplicate_or_zero_time").sum()
        ),
        "negative_time_intervals": int(interval_status.eq("negative_time").sum()),
        "maximum_avatar_speed_xz": samples["avatar_speed_xz"].max(skipna=True),
        "99th_percentile_avatar_speed_xz": _percentile(samples["avatar_speed_xz"]),
        "maximum_avatar_speed_3d": samples["avatar_speed_3d"].max(skipna=True),
        "99th_percentile_avatar_speed_3d": _percentile(samples["avatar_speed_3d"]),
        "maximum_absolute_avatar_acceleration_xz": samples[
            "avatar_acceleration_xz"
        ].abs().max(skipna=True),
        "99th_percentile_absolute_avatar_acceleration_xz": _percentile(
            samples["avatar_acceleration_xz"].abs()
        ),
        "maximum_absolute_avatar_acceleration_3d": samples[
            "avatar_acceleration_3d"
        ].abs().max(skipna=True),
        "99th_percentile_absolute_avatar_acceleration_3d": _percentile(
            samples["avatar_acceleration_3d"].abs()
        ),
        "number_of_flagged_speed_samples": (
            int(samples["diagnostic_speed_flag"].sum()) if thresholds_enabled else pd.NA
        ),
        "number_of_flagged_acceleration_samples": (
            int(samples["diagnostic_acceleration_flag"].sum())
            if thresholds_enabled
            else pd.NA
        ),
        "thresholds_applied": thresholds_enabled,
        "maximum_speed_across_series": speed_max,
        "maximum_absolute_acceleration_across_series": acceleration_max,
    }
    summary.update(location(speed_row, speed_metric, "maximum_speed"))
    summary.update(
        location(acceleration_row, acceleration_metric, "maximum_absolute_acceleration")
    )
    return summary


def _add_diagnostic_inset(axis, time: pd.Series, values: pd.Series, color: str, flags):
    finite = np.isfinite(time.to_numpy(dtype=float)) & np.isfinite(values.to_numpy(dtype=float))
    if not finite.any():
        return
    inset = axis.inset_axes([0.64, 0.17, 0.345, 0.70])
    inset.plot(time, values, color=color, linewidth=0.65)
    if flags is not None:
        marked = finite & flags
        inset.scatter(time[marked], values[marked], color="#C62828", s=8, zorder=3)
    finite_values = values.to_numpy(dtype=float)[finite]
    low, high = np.percentile(finite_values, [0.5, 99.5])
    if np.isclose(low, high):
        padding = max(abs(float(low)) * 0.01, 1e-9)
        low, high = low - padding, high + padding
    inset.set_ylim(float(low), float(high))
    inset.set_title("Diagnostic zoom — values not removed", fontsize=6, pad=2)
    inset.grid(True, alpha=0.2, linewidth=0.4)
    inset.tick_params(labelsize=5, length=2)


def create_figure(
    samples: pd.DataFrame,
    participant: str,
    scenario: str,
    output_path: Path,
    position_units: str,
    speed_threshold: float,
    acceleration_threshold: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = position_units == "meters"
    position_unit = "m" if metric else "raw position units"
    speed_unit = "m/s" if metric else "raw units/s"
    acceleration_unit = "m/s²" if metric else "raw units/s²"
    panels = [
        *(item.column for item in MEASUREMENTS),
        *SPEED_COLUMNS,
        *ACCELERATION_COLUMNS,
    ]
    titles = [
        "B Avatar Pos X", "B Avatar Pos Y", "B Avatar Pos Z",
        "B Avatar Rot X", "B Avatar Rot Y", "B Avatar Rot Z",
        "Avatar Horizontal Speed XZ", "Avatar 3D Speed XYZ",
        "Avatar Horizontal Acceleration XZ", "Avatar 3D Acceleration XYZ",
    ]
    units = [position_unit] * 3 + ["raw rotation units"] * 3 + [speed_unit] * 2 + [acceleration_unit] * 2
    colors = ["#315C8C"] * 6 + ["#087E8B", "#006D77", "#8F2D56", "#6A1B4D"]

    fig, axes = plt.subplots(10, 1, figsize=(17, 37), sharex=True)
    try:
        for index, (axis, column, title, unit, color) in enumerate(
            zip(axes, panels, titles, units, colors)
        ):
            axis.plot(samples[TIME_COLUMN], samples[column], color=color, linewidth=0.8)
            flags = None
            if metric and index in (6, 7):
                flags = samples[column].gt(speed_threshold).fillna(False).to_numpy(dtype=bool)
            elif metric and index in (8, 9):
                flags = samples[column].abs().gt(acceleration_threshold).fillna(False).to_numpy(dtype=bool)
            if flags is not None and flags.any():
                axis.scatter(
                    samples.loc[flags, TIME_COLUMN], samples.loc[flags, column],
                    color="#C62828", s=10, zorder=3, label="Diagnostic threshold exceeded",
                )
                axis.legend(loc="upper left", fontsize=7)
            axis.set_title(title, fontsize=12, weight="bold")
            axis.set_ylabel(unit, fontsize=9)
            axis.grid(True, alpha=0.25, linewidth=0.55)
            axis.tick_params(axis="x", labelbottom=True, labelsize=8)
            if index >= 6:
                _add_diagnostic_inset(
                    axis, samples[TIME_COLUMN], samples[column], color, flags
                )
        axes[-1].set_xlabel("ScenarioTime (s)", fontsize=10)
        fig.suptitle(
            f"{participant} — Scenario {scenario} — Raw Avatar Position, Rotation, Speed, and Acceleration",
            fontsize=18, weight="bold", y=0.999,
        )
        fig.tight_layout(rect=(0.02, 0.01, 0.99, 0.99), h_pad=1.4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        fig.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
        temporary.replace(output_path)
    finally:
        plt.close(fig)


def process_file(
    participant: str,
    scenario: str,
    source_path: Path,
    figure_path: Path,
    sample_path: Path,
    args: argparse.Namespace,
) -> ProcessResult:
    if not args.overwrite and (figure_path.exists() or sample_path.exists()):
        return ProcessResult(
            participant, scenario, source_path, None, None, None,
            "output exists; use --overwrite to replace this workflow's files",
        )
    try:
        header = inspect_header(source_path)
        normalized, duplicates = map_normalized_columns(header.columns)
        required = (TIME_COLUMN, *POSITION_COLUMNS, *ROTATION_COLUMNS)
        missing = [
            name for name in required
            if normalize_header(name) not in normalized or normalize_header(name) in duplicates
        ]
        if missing:
            raise ValueError(f"missing or ambiguous required columns: {missing}")
        source_columns = [normalized[normalize_header(name)] for name in required]
        frame = read_relevant_data(source_path, header, source_columns)
        numeric = {
            name: pd.to_numeric(frame[source], errors="coerce")
            for name, source in zip(required, source_columns)
        }
        metric = args.position_units == "meters"
        samples = calculate_avatar_kinematics(
            numeric[TIME_COLUMN],
            numeric[POSITION_COLUMNS[0]],
            numeric[POSITION_COLUMNS[1]],
            numeric[POSITION_COLUMNS[2]],
            {name: numeric[name] for name in ROTATION_COLUMNS},
            thresholds_enabled=metric,
            speed_threshold=args.speed_threshold,
            acceleration_threshold=args.acceleration_threshold,
        )
        summary = summarize_samples(samples, participant, scenario, source_path, metric)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(sample_path, index=False)
        create_figure(
            samples, participant, scenario, figure_path, args.position_units,
            args.speed_threshold, args.acceleration_threshold, args.dpi,
        )
        return ProcessResult(
            participant, scenario, source_path, figure_path, sample_path, summary
        )
    except Exception as exc:
        return ProcessResult(
            participant, scenario, source_path, None, None, None,
            f"{type(exc).__name__}: {exc}",
        )


def run(args: argparse.Namespace) -> list[ProcessResult]:
    if args.dpi <= 0 or args.speed_threshold <= 0 or args.acceleration_threshold <= 0:
        raise ValueError("DPI and diagnostic thresholds must be positive")
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    discovered = discover_decoded_csvs(data_root, args.participants)
    totals: dict[tuple[str, str], int] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for participant, scenario, _ in discovered:
        totals[(participant, scenario)] = totals.get((participant, scenario), 0) + 1

    results: list[ProcessResult] = []
    for participant, scenario, source_path in discovered:
        key = (participant, scenario)
        occurrences[key] = occurrences.get(key, 0) + 1
        suffix = "" if totals[key] == 1 else f"_{occurrences[key]:02d}"
        stem = f"{participant}_scenario{scenario}_avatar_position_rotation_speed_acceleration{suffix}"
        figure_path = output_root / participant / f"{stem}.png"
        sample_path = output_root / "sample-level" / participant / f"{stem}_samples.csv"
        result = process_file(
            participant, scenario, source_path, figure_path, sample_path, args
        )
        results.append(result)
        marker = "CREATED" if result.error is None else "SKIPPED"
        print(f"[{marker}] {participant} scenario {scenario}: {result.figure_path or result.error}")

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [result.summary for result in results if result.summary is not None]
    pd.DataFrame(summaries).to_csv(
        output_root / "avatar_speed_acceleration_summary.csv", index=False
    )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "files_discovered": len(discovered),
        "files_processed": sum(result.error is None for result in results),
        "figures_created": sum(result.figure_path is not None for result in results),
        "position_units": args.position_units,
        "thresholds_applied": args.position_units == "meters",
        "speed_threshold": args.speed_threshold if args.position_units == "meters" else None,
        "acceleration_threshold": (
            args.acceleration_threshold if args.position_units == "meters" else None
        ),
        "unit_rationale": (
            "Raw units used: repository code says Unity world coordinates are treated as meters "
            "but also states that no independent calibration artifact was found."
            if args.position_units == "raw"
            else "Caller explicitly selected meters."
        ),
        "failures": [
            {"source_file": str(result.source_path), "error": result.error}
            for result in results if result.error is not None
        ],
    }
    (output_root / "avatar_speed_acceleration_run_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main() -> int:
    args = parse_args()
    results = run(args)
    processed = sum(result.error is None for result in results)
    print(
        f"\nDiscovered {len(results)} files; processed {processed}; "
        f"created {sum(result.figure_path is not None for result in results)} figures."
    )
    return 0 if processed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
