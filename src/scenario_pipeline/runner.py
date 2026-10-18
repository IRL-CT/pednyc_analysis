"""Execution, validation, rendering fallback, and reporting for one scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from .scenario import ScenarioPaths


EXPECTED_SCENARIO3_BOUNDARIES = np.array(
    [0.0, 7.828857, 10.790771, 20.499023, 28.346191, 31.002930],
    dtype=float,
)
SCENARIO3_BOUNDARY_TOLERANCE_SEC = 0.06


class PipelineSanityError(ValueError):
    """Raised when an input or output violates a pipeline invariant."""


@dataclass
class PipelineRunResult:
    """Structured state accumulated during a scenario run."""

    paths: ScenarioPaths
    commands: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    time_sources: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    statistics: dict[str, object] = field(default_factory=dict)
    regression: dict[str, object] = field(default_factory=dict)


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read and validate the logger's semicolon-delimited raw CSV."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Raw input does not exist: {path}")
    frame = pd.read_csv(path, sep=";")
    if frame.shape[1] <= 1:
        raise PipelineSanityError(
            f"Raw delimiter validation failed for {path}: parsed {frame.shape[1]} column"
        )
    if frame.empty:
        raise PipelineSanityError(f"Raw input is empty: {path}")
    return frame


def reconstruct_elapsed_time(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Mirror the canonical micro-v6 time-source cascade without importing its CLI."""
    columns = {str(column).strip(): column for column in frame.columns}
    if "ScenarioTime" in columns:
        scenario_time = pd.to_numeric(frame[columns["ScenarioTime"]], errors="coerce").to_numpy(dtype=float)
        if is_seconds_like(scenario_time):
            return scenario_time - scenario_time[0], "ScenarioTime_seconds"
    if "dt" in columns:
        dt = pd.to_numeric(frame[columns["dt"]], errors="coerce").to_numpy(dtype=float)
        good = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
        if len(good):
            median_dt = float(np.nanmedian(good))
            clean = dt.copy()
            clean[~np.isfinite(clean) | (clean <= 0) | (clean > 1.0)] = median_dt
            elapsed = np.zeros(len(frame), dtype=float)
            elapsed[1:] = np.cumsum(clean[1:])
            return elapsed, "cumulative_dt_seconds"
    if "GameTime" in columns:
        game_time = pd.to_numeric(frame[columns["GameTime"]], errors="coerce").to_numpy(dtype=float)
        finite = game_time[np.isfinite(game_time)]
        if len(finite):
            return game_time - finite[0], "GameTime_minus_start"
    if "Frame Number" in columns and "dt" in columns:
        frames = pd.to_numeric(frame[columns["Frame Number"]], errors="coerce").to_numpy(dtype=float)
        dt = pd.to_numeric(frame[columns["dt"]], errors="coerce").to_numpy(dtype=float)
        good = dt[np.isfinite(dt) & (dt > 0)]
        if len(good) and np.any(np.isfinite(frames)):
            first = frames[np.isfinite(frames)][0]
            return (frames - first) * float(np.nanmedian(good)), "Frame_Number_times_dt"
    raise PipelineSanityError("Could not reconstruct elapsed seconds from the canonical time-source cascade")


def is_seconds_like(values: Iterable[float]) -> bool:
    """Apply the canonical micro-v6 seconds-like range check."""
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) < 2:
        return False
    duration = float(np.nanmax(finite) - np.nanmin(finite))
    return 0 < duration <= 60 and abs(float(np.nanmin(finite))) <= 5


def validate_elapsed_time(values: np.ndarray) -> None:
    """Require finite, monotonic, seconds-like elapsed time."""
    elapsed = np.asarray(values, dtype=float)
    if len(elapsed) < 2 or not np.all(np.isfinite(elapsed)):
        raise PipelineSanityError("Elapsed time must contain at least two finite values")
    if np.any(np.diff(elapsed) < 0):
        raise PipelineSanityError("Elapsed time is not monotonic")
    if not is_seconds_like(elapsed):
        raise PipelineSanityError(
            f"Elapsed duration is not seconds-like: {elapsed[-1] - elapsed[0]:.6f}"
        )


def metrics_time_source(decoded: pd.DataFrame) -> str:
    """Describe whether Stage 1 retains a usable dt alias or derives one."""
    lookup = {"".join(ch.lower() for ch in str(c) if ch.isalnum()): c for c in decoded.columns}
    for alias in ("dt", "deltatime"):
        if alias in lookup:
            values = pd.to_numeric(decoded[lookup[alias]], errors="coerce")
            if not values.isna().all():
                return f"existing_dt:{str(lookup[alias]).strip()}"
    return "ScenarioTime.diff"


def assert_fresh_metrics_input(paths: ScenarioPaths, features: pd.DataFrame) -> None:
    """Reject accidental use of the known stale scenario-3 feature artifact."""
    stale = paths.project_root / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv"
    if paths.metrics_csv.resolve() == stale.resolve():
        raise PipelineSanityError("Pipeline metrics path resolves to the known stale committed feature CSV")
    required = {"dt", "ped_speed_xz_smooth", "ped_accel_xz_smooth", "movement_state_v3"}
    missing = sorted(required - {str(c).strip() for c in features.columns})
    if missing:
        raise PipelineSanityError(
            "Generated metrics file is missing canonical Stage 1 columns; it may be stale: "
            + ", ".join(missing)
        )


def validate_vr_position_aliases(features: pd.DataFrame) -> None:
    """Verify that canonical pedestrian aliases resolve to B VR, not Avatar, positions."""
    frame = features.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    required = {"ped_x", "ped_z", "B VR Pos X", "B VR Pos Z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PipelineSanityError(f"Required VR position columns are missing: {missing}")
    for alias, source in (("ped_x", "B VR Pos X"), ("ped_z", "B VR Pos Z")):
        left = pd.to_numeric(frame[alias], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True):
            raise PipelineSanityError(f"{alias} is not an exact alias of {source}; refusing non-VR movement input")


def validate_nonempty_csvs(paths: Iterable[Path]) -> None:
    """Require every listed CSV to exist and contain at least one data row."""
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise PipelineSanityError(f"Required CSV is missing or empty: {path}")
        if pd.read_csv(path, nrows=1).empty:
            raise PipelineSanityError(f"Required CSV has no data rows: {path}")


def validate_micro_outputs(
    paths: ScenarioPaths,
    features: pd.DataFrame,
    macros: pd.DataFrame,
    micro: pd.DataFrame,
    definitions: pd.DataFrame,
) -> None:
    """Validate tag coverage, macro linkage, indices, and scenario-specific names."""
    for path in (*paths.core_csvs(), *paths.expected_plots()):
        if path.name != "micro_event_tag_definitions_v6.csv" and paths.scenario_stem not in path.name:
            raise PipelineSanityError(f"Output filename does not contain {paths.scenario_stem}: {path.name}")
    tags = set(definitions["tag"].dropna().astype(str))
    for column in ("motion_tag", "head_tag", "car_tag"):
        unknown = sorted(set(micro[column].dropna().astype(str)) - tags)
        if unknown:
            raise PipelineSanityError(f"Unknown {column} values not present in tag definitions: {unknown}")
    macro_ids = set(pd.to_numeric(macros["segment_id"], errors="raise").astype(int))
    micro_macro_ids = set(pd.to_numeric(micro["macro_segment_id"], errors="raise").astype(int))
    if not micro_macro_ids.issubset(macro_ids):
        raise PipelineSanityError(f"Micro segments reference missing macro IDs: {sorted(micro_macro_ids - macro_ids)}")
    if (pd.to_numeric(micro["start_idx"]) < 0).any() or (pd.to_numeric(micro["end_idx"]) >= len(features)).any():
        raise PipelineSanityError("Micro segment frame indices fall outside the regenerated metrics frame")
    macro_by_id = macros.set_index(pd.to_numeric(macros["segment_id"]).astype(int))
    for _, row in micro.iterrows():
        macro = macro_by_id.loc[int(row["macro_segment_id"])]
        if int(row["start_idx"]) < int(macro["start_idx"]) or int(row["end_idx"]) > int(macro["end_idx"]):
            raise PipelineSanityError(
                f"Micro segment {row['micro_segment_id']} lies outside macro {row['macro_segment_id']}"
            )


def macro_midpoints(macros: pd.DataFrame, preview_time: float | None = None) -> list[float]:
    """Return an explicit preview time or one midpoint per macro segment."""
    if preview_time is not None:
        return [float(preview_time)]
    return [
        (float(row["time_start_sec"]) + float(row["time_end_sec"])) / 2.0
        for _, row in macros.iterrows()
    ]


def find_ffmpeg(explicit: Path | None = None) -> Path | None:
    """Resolve ffmpeg in the same priority order as the canonical renderer."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for variable in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]))
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        return bundled.resolve() if bundled.is_file() else None
    except (AttributeError, ImportError, OSError, RuntimeError):
        return None


def compare_scenario3_regression(paths: ScenarioPaths) -> dict[str, object]:
    """Compare generated scenario 3 elapsed boundaries and ordered v6 tags."""
    if paths.participant != 1 or paths.scenario_number != 3:
        return {"applicable": False}
    ground_macro = paths.project_root / "outputs" / "graphs" / "macro_segmentation" / "macro_segments_PedNYC1_scenario3_v2.csv"
    ground_micro = paths.project_root / "outputs" / "graphs" / "micro_segmentation" / "v6" / "micro_segments_descriptive_PedNYC1_scenario3_v6.csv"
    ground_boundaries = paths.project_root / "outputs" / "graphs" / "micro_segmentation" / "v6" / "micro_change_boundaries_PedNYC1_scenario3_v6.csv"
    generated_macro = pd.read_csv(paths.macro_csv, index_col=False, skipinitialspace=True)
    generated_micro = pd.read_csv(paths.micro_segments_csv, index_col=False, skipinitialspace=True)
    # The committed v6 artifact contains padded headers and must explicitly
    # disable pandas' implicit-index inference before column normalization.
    expected_micro = pd.read_csv(ground_micro, index_col=False, skipinitialspace=True)
    generated_macro.columns = generated_macro.columns.astype(str).str.strip()
    generated_micro.columns = generated_micro.columns.astype(str).str.strip()
    expected_micro.columns = expected_micro.columns.astype(str).str.strip()
    expected_boundary_rows = len(pd.read_csv(ground_boundaries, index_col=False, skipinitialspace=True))
    generated_boundary_rows = len(pd.read_csv(paths.micro_boundaries_csv))
    boundaries = np.concatenate(
        [generated_macro["time_start_sec"].to_numpy(dtype=float), [float(generated_macro["time_end_sec"].iloc[-1])]]
    )
    expected_macro = pd.read_csv(ground_macro, index_col=False, skipinitialspace=True)
    expected_macro.columns = expected_macro.columns.astype(str).str.strip()
    labels_match = generated_macro["macro_label"].astype(str).str.strip().tolist() == expected_macro["macro_label"].astype(str).str.strip().tolist()
    boundary_match = len(boundaries) == len(EXPECTED_SCENARIO3_BOUNDARIES) and bool(
        np.allclose(boundaries, EXPECTED_SCENARIO3_BOUNDARIES, atol=SCENARIO3_BOUNDARY_TOLERANCE_SEC, rtol=0)
    )
    tag_columns = ["motion_tag", "head_tag", "car_tag"]
    tag_sequence_match = generated_micro[tag_columns].astype(str).apply(lambda c: c.str.strip()).values.tolist() == expected_micro[tag_columns].astype(str).apply(lambda c: c.str.strip()).values.tolist()
    return {
        "applicable": True,
        "tolerance_sec": SCENARIO3_BOUNDARY_TOLERANCE_SEC,
        "generated_boundaries": boundaries.tolist(),
        "expected_boundaries": EXPECTED_SCENARIO3_BOUNDARIES.tolist(),
        "boundary_match": boundary_match,
        "macro_labels_match": labels_match,
        "generated_micro_segments": len(generated_micro),
        "expected_micro_segments": len(expected_micro),
        "tag_sequence_match": tag_sequence_match,
        "generated_boundary_records": generated_boundary_rows,
        "expected_boundary_records": expected_boundary_rows,
        "boundary_record_count_match": generated_boundary_rows == expected_boundary_rows,
    }


class ScenarioRunner:
    """Run canonical scripts with scenario-specific paths and validate their outputs."""

    def __init__(
        self,
        paths: ScenarioPaths,
        *,
        python_executable: Path | None = None,
        skip_render: bool = False,
        preview_only: bool = False,
        preview_time: float | None = None,
        ffmpeg: Path | None = None,
        force: bool = False,
    ) -> None:
        self.paths = paths
        self.python = Path(python_executable or sys.executable)
        self.skip_render = skip_render
        self.preview_only = preview_only
        self.preview_time = preview_time
        self.ffmpeg = ffmpeg
        self.force = force
        self.result = PipelineRunResult(paths)

    def run(self) -> PipelineRunResult:
        """Execute Stages 0-4, validate, compare regression, and write the report."""
        if not self.force:
            existing = [path for path in self.paths.core_csvs() if path.exists()]
            if existing:
                raise FileExistsError(
                    f"Scenario outputs already exist (for example {existing[0]}). "
                    "Pass --force to overwrite them with a freshly regenerated pipeline run."
                )
        raw = read_raw_csv(self.paths.raw_csv)
        self.result.statistics["raw_rows"] = len(raw)
        self.result.statistics["raw_columns"] = raw.shape[1]
        try:
            self._run_core_stages()
            self._validate_and_collect(raw)
            self.render_gantt()
            self.result.regression = compare_scenario3_regression(self.paths)
        except Exception as exc:
            self.result.failures.append(f"{type(exc).__name__}: {exc}")
            self._write_report()
            raise
        self._write_report()
        return self.result

    def _run_core_stages(self) -> None:
        self.decode_scenario()
        decoded = pd.read_csv(self.paths.decoded_csv, sep=";")
        self.result.time_sources["metrics_v1"] = metrics_time_source(decoded)
        self.create_metrics()
        self.run_macro_segmentation()
        self.run_micro_segmentation()

    def decode_scenario(self) -> None:
        """Run canonical Stage 0 for this scenario."""
        p = self.paths
        self._run_stage("decode", [
            "src/preprocess/decode_first.py", "--input-csv", str(p.raw_csv), "--output-csv", str(p.decoded_csv),
        ])

    def create_metrics(self) -> None:
        """Run canonical Stage 1, including actual supported diagnostics."""
        p = self.paths
        self._run_stage("metrics_v1", [
            "src/features/create_metrics_and_graph.py",
            "--input-csv", str(p.decoded_csv),
            "--output-csv", str(p.metrics_csv),
            "--segment-csv", str(p.legacy_segments_csv),
            "--graph-dir", str(p.metrics_graph_dir),
            "--fourframe-dir", str(p.fourframe_dir),
        ])

    def run_macro_segmentation(self) -> None:
        """Run canonical macrosegmentation v2 without changing its algorithm."""
        p = self.paths
        self._run_stage("macro_v2", [
            "src/segmentation/macro_segmentation.py",
            "--input-csv", str(p.metrics_csv),
            "--output-dir", str(p.macro_dir),
            "--segments-csv", str(p.macro_csv),
            "--plot-path", str(p.macro_png),
            "--conflict-events-csv", str(
                p.project_root / "outputs" / "conflict_zone_analysis"
                / f"scenario_{p.scenario_number:02d}" / "conflict_zone_events.csv"
            ),
            "--recording-identifier", p.scenario_stem,
        ])

    def run_micro_segmentation(self) -> None:
        """Run canonical microsegmentation v6 with a scenario-specific stem."""
        p = self.paths
        self._run_stage("micro_v6", [
            "src/segmentation/micro_segmentation.py",
            "--feature-csv", str(p.metrics_csv),
            "--macro-csv", str(p.macro_csv),
            "--output-dir", str(p.micro_dir),
            "--scenario-stem", p.scenario_stem,
        ])

    def _run_stage(self, name: str, arguments: list[str]) -> None:
        command = [str(self.python), *arguments]
        self.result.commands.append(subprocess.list2cmdline(command))
        try:
            subprocess.run(command, cwd=self.paths.project_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{name} failed with exit code {exc.returncode}") from exc

    def _validate_and_collect(self, raw: pd.DataFrame) -> None:
        p = self.paths
        decoded = pd.read_csv(p.decoded_csv, sep=";")
        if len(decoded) != len(raw):
            raise PipelineSanityError(f"Decode row mismatch: raw={len(raw)}, decoded={len(decoded)}")
        features = pd.read_csv(p.metrics_csv)
        assert_fresh_metrics_input(p, features)
        validate_vr_position_aliases(features)
        elapsed, micro_source = reconstruct_elapsed_time(features)
        validate_elapsed_time(elapsed)
        self.result.time_sources["decode"] = "preserve_raw_rows"
        self.result.time_sources["macro_v2"] = "cumulative_dt_seconds" if "dt" in features.columns else "raw_ScenarioTime_minus_start"
        self.result.time_sources["micro_v6"] = micro_source
        dt = pd.to_numeric(features["dt"], errors="coerce").to_numpy(dtype=float)
        valid_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
        if not len(valid_dt):
            raise PipelineSanityError("No positive reasonable dt values (0 < dt < 1 second)")
        median_dt = float(np.nanmedian(valid_dt))
        if not 0 < median_dt < 1.0:
            raise PipelineSanityError(f"Median dt is unreasonable: {median_dt}")
        macros = pd.read_csv(p.macro_csv)
        micro = pd.read_csv(p.micro_segments_csv)
        definitions = pd.read_csv(p.tag_definitions_csv)
        validate_nonempty_csvs(p.core_csvs())
        validate_micro_outputs(p, features, macros, micro, definitions)
        self.result.statistics.update({
            "decoded_rows": len(decoded),
            "decoded_columns": decoded.shape[1],
            "metrics_rows": len(features),
            "elapsed_duration_sec": float(elapsed[-1] - elapsed[0]),
            "median_valid_dt_sec": median_dt,
            "sampling_frequency_hz": 1.0 / median_dt,
            "macro_segments": len(macros),
            "micro_segments": len(micro),
            "boundary_records": len(pd.read_csv(p.micro_boundaries_csv)),
            "motion_counts": micro["motion_tag"].value_counts().sort_index().to_dict(),
            "head_counts": micro["head_tag"].value_counts().sort_index().to_dict(),
            "car_counts": micro["car_tag"].value_counts().sort_index().to_dict(),
            "micro_with_inferred": int(micro["has_inferred_behavior"].astype(bool).sum()),
            "micro_without_inferred": int((~micro["has_inferred_behavior"].astype(bool)).sum()),
        })
        self.result.statistics["macro_table"] = macros
        for name, path in {
            "decoded_csv": p.decoded_csv,
            "metrics_csv": p.metrics_csv,
            "legacy_segments_csv": p.legacy_segments_csv,
            "macro_csv": p.macro_csv,
            "macro_png": p.macro_png,
            "micro_segments_csv": p.micro_segments_csv,
            "micro_frame_csv": p.micro_frame_csv,
            "micro_summary_csv": p.micro_summary_csv,
            "micro_boundaries_csv": p.micro_boundaries_csv,
            "tag_definitions_csv": p.tag_definitions_csv,
            "micro_standard_png": p.micro_standard_png,
            "micro_stacked_png": p.micro_stacked_png,
            "micro_gantt_png": p.micro_gantt_png,
        }.items():
            self.result.artifacts[name] = path
        for index, plot in enumerate(sorted(p.metrics_graph_dir.glob("[0-9][0-9]_*.png")), start=1):
            self.result.artifacts[f"metrics_plot_{index:02d}"] = plot
        for name in ("00_original_decoded_column_inventory.txt", "01_final_metrics_column_inventory.txt"):
            self.result.artifacts[f"metrics_{Path(name).stem}"] = p.metrics_graph_dir / name
            self.result.artifacts[f"fourframe_{Path(name).stem}"] = p.fourframe_dir / name
        self.result.artifacts["fourframe_metrics_csv"] = p.fourframe_dir / p.metrics_csv.name
        self.result.artifacts["fourframe_legacy_segments_csv"] = p.fourframe_dir / p.legacy_segments_csv.name
        for index, plot in enumerate(sorted(p.fourframe_dir.glob("[0-9][0-9]_*.png")), start=1):
            self.result.artifacts[f"fourframe_metrics_plot_{index:02d}"] = plot

    def render_gantt(self) -> None:
        """Render an MP4 when possible, otherwise write requested/midpoint previews."""
        p = self.paths
        if self.skip_render:
            self.result.skipped.append("Gantt MP4 and renderer previews (--skip-render)")
            return
        macros = pd.read_csv(p.macro_csv)
        encoder = find_ffmpeg(self.ffmpeg)
        if self.preview_only or encoder is None:
            if encoder is None and not self.preview_only:
                self.result.warnings.append("ffmpeg was unavailable; generated macro-midpoint Gantt previews instead of MP4")
            p.preview_dir.mkdir(parents=True, exist_ok=True)
            for index, time_value in enumerate(macro_midpoints(macros, self.preview_time)):
                output = p.preview_dir / f"{p.scenario_stem}_macro_preview_{index:02d}_{time_value:.3f}s.png"
                self._run_stage("gantt_preview", [
                    "src/segmentation/render_micro_gantt_mp4.py",
                    "--segments-csv", str(p.micro_segments_csv),
                    "--macro-csv", str(p.macro_csv),
                    "--output", str(p.gantt_mp4),
                    "--preview-time", str(time_value),
                    "--preview-output", str(output),
                ])
                self.result.artifacts[f"gantt_preview_{index:02d}"] = output
            return
        arguments = [
            "src/segmentation/render_micro_gantt_mp4.py",
            "--segments-csv", str(p.micro_segments_csv),
            "--macro-csv", str(p.macro_csv),
            "--output", str(p.gantt_mp4),
            "--ffmpeg", str(encoder),
        ]
        self._run_stage("gantt_mp4", arguments)
        self.result.artifacts["gantt_mp4"] = p.gantt_mp4

    def _write_report(self) -> None:
        p = self.paths
        p.report_md.parent.mkdir(parents=True, exist_ok=True)
        stats = self.result.statistics
        lines = [
            f"# {p.scenario_stem} canonical pipeline report",
            "",
            "## Input and timing",
            "",
            f"- Input: `{p.raw_csv.name}`",
            f"- Scenario ID: {p.scenario_number}",
            f"- Scenario stem: `{p.scenario_stem}`",
            f"- Raw rows: {stats.get('raw_rows', 'unavailable')}",
            f"- Decoded rows: {stats.get('decoded_rows', 'unavailable')}",
            f"- Decoded columns: {stats.get('decoded_columns', 'unavailable')}",
            f"- Elapsed duration: {_fmt(stats.get('elapsed_duration_sec'))} seconds",
            f"- Median valid dt: {_fmt(stats.get('median_valid_dt_sec'))} seconds",
            f"- Approximate sampling frequency: {_fmt(stats.get('sampling_frequency_hz'))} Hz",
            "",
            "### Time sources",
            "",
        ]
        for stage, source in self.result.time_sources.items():
            lines.append(f"- {stage}: `{source}`")
        lines.extend(["", "## Macro segments", ""])
        macro_table = stats.get("macro_table")
        if isinstance(macro_table, pd.DataFrame) and not macro_table.empty:
            columns = [c for c in ("segment_id", "macro_label", "start_idx", "end_idx", "time_start_sec", "time_end_sec", "duration_sec") if c in macro_table]
            lines.extend(_markdown_table(macro_table[columns]))
        else:
            lines.append("Macro output unavailable.")
        lines.extend([
            "",
            f"Macro segment count: {stats.get('macro_segments', 'unavailable')}",
            f"Micro segment count: {stats.get('micro_segments', 'unavailable')}",
            f"Micro boundary record count: {stats.get('boundary_records', 'unavailable')}",
            "",
            "## Micro tag counts",
            "",
        ])
        for label, key in (("Motion", "motion_counts"), ("Head", "head_counts"), ("Car context", "car_counts")):
            lines.append(f"### {label}")
            lines.append("")
            counts = stats.get(key, {})
            if isinstance(counts, dict) and counts:
                lines.extend(f"- `{tag}`: {count}" for tag, count in counts.items())
            else:
                lines.append("- unavailable")
            lines.append("")
        lines.extend([
            "### Observability",
            "",
            f"- Micro segments with at least one inferred tag: {stats.get('micro_with_inferred', 'unavailable')}",
            f"- Micro segments with only observable tags: {stats.get('micro_without_inferred', 'unavailable')}",
            "",
            "## Scenario 3 regression",
            "",
        ])
        if self.result.regression.get("applicable"):
            regression = self.result.regression
            lines.extend([
                f"- Boundary tolerance: {regression['tolerance_sec']} seconds",
                f"- Elapsed boundaries match: {regression['boundary_match']}",
                f"- Macro labels match: {regression['macro_labels_match']}",
                f"- Generated/expected micro segments: {regression['generated_micro_segments']}/{regression['expected_micro_segments']}",
                f"- Ordered motion/head/car tag sequences match: {regression['tag_sequence_match']}",
                f"- Generated/expected boundary records: {regression['generated_boundary_records']}/{regression['expected_boundary_records']}",
            ])
        else:
            lines.append("Not applicable to this scenario.")
        lines.extend(["", "## Artifact checklist", ""])
        for name, path in sorted(self.result.artifacts.items()):
            status = "generated" if path.is_file() and path.stat().st_size > 0 else "missing"
            lines.append(f"- [{ 'x' if status == 'generated' else ' ' }] `{name}` — {status}: `{path}`")
        lines.extend(["", "## Warnings and anomalies", ""])
        messages = self.result.warnings or ["None."]
        lines.extend(f"- {message}" for message in messages)
        lines.extend(["", "## Failed or skipped outputs", ""])
        failures = self.result.failures or ["None."]
        lines.extend(f"- {message}" for message in failures)
        lines.extend(f"- Skipped: {message}" for message in self.result.skipped)
        lines.extend(["", "## Reproduction commands", ""])
        lines.extend(f"```text\n{command}\n```" for command in self.result.commands)
        p.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.result.artifacts["report_md"] = p.report_md


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    return f"{number:.6f}" if math.isfinite(number) else "unavailable"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines
