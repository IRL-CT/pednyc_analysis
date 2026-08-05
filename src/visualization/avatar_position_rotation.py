"""Plot raw avatar position and rotation traces for decoded PedNYC CSVs.

The default discovery scope is the repository's processed-data tree. It is
searched recursively, so discovery does not depend on an exact directory or
file naming template. Every discovered decoded file is either rendered or
included in the machine-readable skip report.

No values are transformed, aggregated, resampled, clipped, smoothed, or filled.
Numeric conversion failures remain NaN, which Matplotlib renders as line gaps.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "avatar graphs"
PARTICIPANTS = ("PedNYC1", "PedNYC2", "PedNYC3")
DELIMITERS = (";", ",", "\t", "|")
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252")


@dataclass(frozen=True)
class Measurement:
    column: str
    title: str
    unit: str = "raw units"


MEASUREMENTS = (
    Measurement("B Avatar Pos X", "Avatar Position X"),
    Measurement("B Avatar Pos Y", "Avatar Position Y"),
    Measurement("B Avatar Pos Z", "Avatar Position Z"),
    Measurement("B Avatar Rot X", "Avatar Rotation X"),
    Measurement("B Avatar Rot Y", "Avatar Rotation Y"),
    Measurement("B Avatar Rot Z", "Avatar Rotation Z"),
)

# These are only considered when ScenarioTime is absent or has no numeric data.
# Earlier entries are more scenario-specific and therefore preferred.
ELAPSED_TIME_CANDIDATES = (
    "game time",
    "gametime",
    "game timestamp",
    "gametimestamp",
    "session time",
    "sessiontime",
    "elapsed time",
    "elapsedtime",
    "realtime since startup",
    "realtimesincestartup",
    "timestamp",
    "time stamp",
    "datetime",
    "date time",
    "utc time",
    "time",
)


@dataclass(frozen=True)
class HeaderSpec:
    delimiter: str
    encoding: str
    columns: tuple[str, ...]


@dataclass
class FileReport:
    participant: str
    scenario: str
    source_path: str
    status: str
    delimiter: str | None = None
    encoding: str | None = None
    source_row_count: int | None = None
    output_path: str | None = None
    panel_count: int | None = None
    time_column: str | None = None
    time_method: str | None = None
    x_axis_label: str | None = None
    missing_columns: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)
    numeric_sample_counts: dict[str, int] = field(default_factory=dict)
    valid_sample_counts: dict[str, int] = field(default_factory=dict)
    plotted_sample_counts: dict[str, int] = field(default_factory=dict)
    reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing canonical pednyc1, pednyc2, pednyc3 folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Top-level output folder (default: repository/avatar graphs).",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        choices=PARTICIPANTS,
        default=list(PARTICIPANTS),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace avatar PNGs previously generated at the exact output paths.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def normalize_header(value: object) -> str:
    """Return a conservative comparison key for inconsistent CSV headers."""
    text = unicodedata.normalize("NFKC", str(value)).replace("\ufeff", "")
    text = text.strip().strip(";,").strip().strip("\"'")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def discover_decoded_csvs(data_root: Path, participants: list[str]) -> list[tuple[str, str, Path]]:
    """Find decoded CSVs recursively without requiring an exact path template."""
    discovered: list[tuple[str, str, Path]] = []
    requested = {participant.casefold(): participant for participant in participants}
    for path in Path(data_root).rglob("*.csv"):
        relative_parts = path.relative_to(data_root).parts
        decoded_marker = "decoded" in path.stem.casefold() or any(
            part.casefold() == "decoded" for part in relative_parts[:-1]
        )
        if not decoded_marker:
            continue
        participant = participant_from_path(path, requested)
        if participant is None:
            continue
        scenario = scenario_from_path(path)
        discovered.append((participant, scenario, path.resolve()))
    return sorted(discovered, key=lambda item: (int(item[0][6:]), _natural_key(item[1]), str(item[2])))


def participant_from_path(path: Path, requested: dict[str, str]) -> str | None:
    """Resolve a selected participant from any path component or the filename."""
    for part in reversed(path.parts):
        token = re.sub(r"[^a-z0-9]", "", part.casefold())
        match = re.search(r"pednyc0*(\d+)", token)
        if match:
            key = f"pednyc{int(match.group(1))}"
            if key in requested:
                return requested[key]
    return None


def scenario_from_path(path: Path) -> str:
    for parent in path.parents:
        match = re.fullmatch(r"scenario(.+)", parent.name, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.search(r"scenario[-_]?([A-Za-z0-9]+)", path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _natural_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def inspect_header(path: Path) -> HeaderSpec:
    """Choose encoding/delimiter by required-header matches, then width."""
    required = {normalize_header(item.column) for item in MEASUREMENTS}
    scenario_key = normalize_header("ScenarioTime")
    best: tuple[tuple[int, int, int], HeaderSpec] | None = None
    errors: list[str] = []
    for encoding in ENCODINGS:
        for delimiter in DELIMITERS:
            try:
                header = pd.read_csv(path, sep=delimiter, encoding=encoding, nrows=0)
            except (OSError, UnicodeError, pd.errors.ParserError) as exc:
                errors.append(f"{encoding}/{delimiter!r}: {exc}")
                continue
            columns = tuple(str(column) for column in header.columns)
            normalized = {normalize_header(column) for column in columns}
            score = (
                len(required.intersection(normalized)),
                int(scenario_key in normalized),
                len(columns),
            )
            candidate = HeaderSpec(delimiter, encoding, columns)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None and best[0][0] == len(required):
            break
    if best is None or len(best[1].columns) <= 1:
        details = "; ".join(errors[-3:]) or "no parse attempt succeeded"
        raise ValueError(f"could not parse a multi-column header ({details})")
    return best[1]


def map_normalized_columns(columns: tuple[str, ...]) -> tuple[dict[str, str], dict[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(normalize_header(column), []).append(column)
    unique = {key: values[0] for key, values in grouped.items() if len(values) == 1}
    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    return unique, duplicates


def relevant_time_columns(normalized: dict[str, str]) -> list[str]:
    keys = [normalize_header("ScenarioTime"), *ELAPSED_TIME_CANDIDATES]
    return [normalized[key] for key in keys if key in normalized]


def read_relevant_data(path: Path, header: HeaderSpec, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=header.delimiter,
        encoding=header.encoding,
        usecols=columns,
        low_memory=False,
    )


def choose_time_axis(
    frame: pd.DataFrame, normalized: dict[str, str]
) -> tuple[pd.Series, str | None, str, str]:
    """Use ScenarioTime, then elapsed timestamp, then sample number."""
    scenario_key = normalize_header("ScenarioTime")
    if scenario_key in normalized and normalized[scenario_key] in frame:
        column = normalized[scenario_key]
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            return numeric.astype(float), column, "ScenarioTime (seconds; source values unchanged)", "Scenario Time (s)"

    for key in ELAPSED_TIME_CANDIDATES:
        column = normalized.get(key)
        if column is None or column not in frame:
            continue
        elapsed, detail = elapsed_seconds(frame[column], key)
        if elapsed.notna().any():
            return elapsed, column, detail, "Scenario Time (s)"

    sample_number = pd.Series(np.arange(1, len(frame) + 1, dtype=float), index=frame.index)
    return sample_number, None, "sample number (last resort; no usable time column)", "Sample Number (time unavailable)"


def elapsed_seconds(series: pd.Series, normalized_name: str) -> tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        scale, scale_note = _numeric_time_scale(numeric, normalized_name)
        first = float(numeric.dropna().iloc[0])
        return (numeric.astype(float) - first) * scale, f"elapsed seconds from {normalized_name!r}; {scale_note}"

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        first = parsed.dropna().iloc[0]
        return (parsed - first).dt.total_seconds(), f"elapsed seconds from datetime column {normalized_name!r}"

    return pd.Series(np.nan, index=series.index, dtype=float), f"unusable time column {normalized_name!r}"


def _numeric_time_scale(series: pd.Series, normalized_name: str) -> tuple[float, str]:
    tokens = set(normalized_name.split())
    if tokens.intersection({"millisecond", "milliseconds", "msec", "ms"}):
        return 1e-3, "column name indicates milliseconds"
    if tokens.intersection({"microsecond", "microseconds", "usec", "us"}):
        return 1e-6, "column name indicates microseconds"
    if tokens.intersection({"nanosecond", "nanoseconds", "nsec", "ns"}):
        return 1e-9, "column name indicates nanoseconds"

    magnitude = float(series.dropna().abs().median())
    if magnitude >= 1e17:
        return 1e-9, "epoch-like magnitude interpreted as nanoseconds"
    if magnitude >= 1e14:
        return 1e-6, "epoch-like magnitude interpreted as microseconds"
    if magnitude >= 1e11:
        return 1e-3, "epoch-like magnitude interpreted as milliseconds"
    return 1.0, "numeric source assumed to be seconds"


def plot_file(
    participant: str,
    scenario: str,
    source_path: Path,
    output_path: Path,
    dpi: int,
    overwrite: bool,
) -> FileReport:
    report = FileReport(participant, scenario, str(source_path), "skipped")
    try:
        header = inspect_header(source_path)
        report.delimiter = "TAB" if header.delimiter == "\t" else header.delimiter
        report.encoding = header.encoding
        normalized, duplicates = map_normalized_columns(header.columns)
        required_keys = {normalize_header(item.column): item.column for item in MEASUREMENTS}

        ambiguous = [required_keys[key] for key in required_keys if key in duplicates]
        missing = [
            requested for key, requested in required_keys.items()
            if key not in normalized and key not in duplicates
        ]
        report.missing_columns = sorted(missing + [f"{name} (ambiguous duplicate)" for name in ambiguous])
        if report.missing_columns:
            report.reason = "missing or ambiguous required measurement columns"
            return report

        measurement_columns = [normalized[normalize_header(item.column)] for item in MEASUREMENTS]
        selected = list(dict.fromkeys(measurement_columns + relevant_time_columns(normalized)))
        frame = read_relevant_data(source_path, header, selected)
        report.source_row_count = len(frame)

        numeric_measurements: dict[str, pd.Series] = {}
        for item, source_column in zip(MEASUREMENTS, measurement_columns):
            values = pd.to_numeric(frame[source_column], errors="coerce").astype(float)
            numeric_measurements[item.column] = values
            report.numeric_sample_counts[item.column] = int(values.notna().sum())
        report.empty_columns = [name for name, count in report.numeric_sample_counts.items() if count == 0]
        if report.empty_columns:
            report.reason = "required measurement columns contain no numeric samples"
            return report

        time_values, time_column, time_method, x_axis_label = choose_time_axis(frame, normalized)
        report.time_column = time_column
        report.time_method = time_method
        report.x_axis_label = x_axis_label
        report.valid_sample_counts = {
            name: int((time_values.notna() & values.notna()).sum())
            for name, values in numeric_measurements.items()
        }

        if output_path.exists() and not overwrite:
            report.reason = "output already exists (use --overwrite to replace this exact avatar graph)"
            return report

        plotted_counts, panel_count = create_figure(
            time_values,
            numeric_measurements,
            participant,
            scenario,
            x_axis_label,
            output_path,
            dpi,
        )
        if plotted_counts != report.valid_sample_counts:
            raise RuntimeError(
                "plotted sample counts do not match the valid source sample counts"
            )
        report.plotted_sample_counts = plotted_counts
        report.panel_count = panel_count
        report.output_path = str(output_path)
        report.status = "produced"
        return report
    except Exception as exc:  # Keep the batch going and make every failure auditable.
        report.reason = f"{type(exc).__name__}: {exc}"
        return report


def create_figure(
    time_values: pd.Series,
    numeric_measurements: dict[str, pd.Series],
    participant: str,
    scenario: str,
    x_axis_label: str,
    output_path: Path,
    dpi: int,
) -> tuple[dict[str, int], int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(16, 24), sharex=True)
    plotted_counts: dict[str, int] = {}
    try:
        for axis, item in zip(axes, MEASUREMENTS):
            values = numeric_measurements[item.column]
            line = axis.plot(time_values, values, color="#315C8C", linewidth=0.9)[0]
            expected = int((time_values.notna() & values.notna()).sum())
            plotted = int(
                np.count_nonzero(
                    np.isfinite(np.asarray(line.get_xdata(), dtype=float))
                    & np.isfinite(np.asarray(line.get_ydata(), dtype=float))
                )
            )
            if plotted != expected:
                raise RuntimeError(
                    f"plot sample verification failed for {item.column}: expected {expected}, got {plotted}"
                )
            plotted_counts[item.column] = plotted
            axis.set_title(item.title, fontsize=14, weight="bold")
            axis.set_xlabel(x_axis_label, fontsize=11)
            axis.set_ylabel(f"{item.column} ({item.unit})", fontsize=11)
            axis.grid(True, alpha=0.25, linewidth=0.6)
            axis.tick_params(axis="x", labelbottom=True)

        if len(fig.axes) != 6:
            raise RuntimeError(f"figure panel verification failed: expected 6, got {len(fig.axes)}")
        fig.suptitle(
            f"{participant} \N{EN DASH} Scenario {scenario} \N{EN DASH} Avatar Position and Rotation",
            fontsize=20,
            weight="bold",
            y=0.999,
        )
        fig.tight_layout(rect=(0.02, 0.01, 0.99, 0.985), h_pad=2.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        fig.savefig(temporary_path, dpi=dpi, format="png", bbox_inches="tight")
        temporary_path.replace(output_path)
        return plotted_counts, len(fig.axes)
    finally:
        plt.close(fig)


def write_reports(
    data_root: Path,
    output_root: Path,
    reports: list[FileReport],
    discovered_count: int,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "avatar_graphs_file_report.csv"
    json_path = output_root / "avatar_graphs_run_summary.json"
    rows: list[dict[str, object]] = []
    for report in reports:
        row = asdict(report)
        row["missing_columns"] = "; ".join(report.missing_columns)
        row["empty_columns"] = "; ".join(report.empty_columns)
        row["numeric_sample_counts"] = json.dumps(report.numeric_sample_counts, sort_keys=True)
        row["valid_sample_counts"] = json.dumps(report.valid_sample_counts, sort_keys=True)
        row["plotted_sample_counts"] = json.dumps(report.plotted_sample_counts, sort_keys=True)
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else list(asdict(FileReport("", "", "", "")).keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure_counts = {
        participant: sum(
            report.status == "produced" and report.participant == participant for report in reports
        )
        for participant in PARTICIPANTS
    }
    skipped = [asdict(report) for report in reports if report.status != "produced"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "csv_files_found": discovered_count,
        "figures_produced": figure_counts,
        "skipped_file_count": len(skipped),
        "skipped_files": skipped,
        "verification": {
            "all_discovered_files_reported": len(reports) == discovered_count,
            "all_successful_figures_have_six_panels": all(
                report.panel_count == 6 for report in reports if report.status == "produced"
            ),
            "all_successful_sample_counts_match": all(
                report.plotted_sample_counts == report.valid_sample_counts
                for report in reports if report.status == "produced"
            ),
            "note": (
                "Valid sample counts are rows where both the chosen time and source measurement are numeric; "
                "each plotted count must equal its corresponding valid sample count. Source measurement counts "
                "are also retained so missing time values remain auditable."
            ),
        },
        "unit_assumptions": {
            "all_six_measurements": "raw units; no independent calibration metadata was found",
            "project_context": (
                "Existing project analysis treats Unity world position units as meters and some yaw fields as "
                "degrees, but that is not sufficient to assert calibrated units for all six requested raw fields."
            ),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return csv_path, json_path


def run(args: argparse.Namespace) -> tuple[list[FileReport], Path, Path]:
    if args.dpi < 200:
        raise ValueError("--dpi must be at least 200")
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    discovered = discover_decoded_csvs(data_root, args.participants)
    reports: list[FileReport] = []
    occurrence_counts: dict[tuple[str, str], int] = {}
    totals: dict[tuple[str, str], int] = {}
    for participant, scenario, _ in discovered:
        key = (participant, scenario)
        totals[key] = totals.get(key, 0) + 1
    for participant, scenario, source_path in discovered:
        key = (participant, scenario)
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
        suffix = "" if totals[key] == 1 else f"_{occurrence_counts[key]:02d}"
        filename = f"{participant}_scenario{scenario}_avatar_position_rotation{suffix}.png"
        output_path = output_root / participant / filename
        report = plot_file(
            participant,
            scenario,
            source_path,
            output_path,
            args.dpi,
            args.overwrite,
        )
        reports.append(report)
        marker = "CREATED" if report.status == "produced" else "SKIPPED"
        detail = report.output_path or report.reason
        print(f"[{marker}] {participant} scenario {scenario}: {detail}")
    csv_report, json_report = write_reports(data_root, output_root, reports, len(discovered))
    return reports, csv_report, json_report


def main() -> int:
    args = parse_args()
    reports, csv_report, json_report = run(args)
    produced = sum(report.status == "produced" for report in reports)
    skipped = len(reports) - produced
    print(f"\nFound {len(reports)} decoded CSV files; produced {produced} figures; skipped {skipped}.")
    print(f"File report: {csv_report}")
    print(f"Run summary: {json_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
