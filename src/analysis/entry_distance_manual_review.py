"""Rank within-scenario entry-event distances for later manual review.

This module is deliberately descriptive.  A manual-review flag identifies a
recording as one of the largest absolute deviations from its scenario mean; it
does not assign or imply an interaction label.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.scenario_metadata import SCENARIO_DESCRIPTIONS, scenario_description
from src.analysis.trajectory_analysis import PROJECT_ROOT


TOP_N_REVIEW = 3
DEFAULT_SCENARIOS = tuple(SCENARIO_DESCRIPTIONS)
DEFAULT_TRAJECTORY_INPUT = PROJECT_ROOT / "outputs/trajectory_analysis"
DEFAULT_CONFLICT_ZONE_INPUT = PROJECT_ROOT / "outputs/conflict_zone_analysis"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/entry_distance_manual_review"

OUTPUT_COLUMNS = [
    "participant_id",
    "scenario_id",
    "recording_path",
    "recording_identifier",
    "car_conflict_zone_entry_time",
    "pedestrian_conflict_zone_entry_time",
    "first_conflict_zone_entry_time",
    "first_entrant",
    "distance_at_first_conflict_zone_entry_m",
    "scenario_mean_entry_distance_m",
    "signed_difference_from_mean_m",
    "absolute_difference_from_mean_m",
    "within_scenario_deviation_rank",
    "manual_review_candidate",
    "entry_status",
    "data_quality_status",
    "manual_interaction_label",
    "manual_review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", type=int, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--trajectory-input", type=Path, default=DEFAULT_TRAJECTORY_INPUT)
    parser.add_argument("--conflict-zone-input", type=Path, default=DEFAULT_CONFLICT_ZONE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n-review", type=int, default=TOP_N_REVIEW)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing manual-review output directory.",
    )
    return parser.parse_args()


def _ordered_actor_samples(
    time: np.ndarray, x: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"time": time, "x": x, "z": z}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    frame = frame.sort_values("time").drop_duplicates("time", keep="first")
    return (
        frame.time.to_numpy(dtype=float),
        frame.x.to_numpy(dtype=float),
        frame.z.to_numpy(dtype=float),
    )


def first_circle_entry_time(
    time: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    zone: dict[str, object],
) -> float:
    """Return the first linearly interpolated entry into a circular zone."""
    time, x, z = _ordered_actor_samples(time, x, z)
    if len(time) == 0:
        return math.nan
    center = np.array([float(zone["center_x_m"]), float(zone["center_z_m"])])
    radius = float(zone["radius_m"])
    if radius <= 0 or not np.isfinite([*center, radius]).all():
        raise ValueError("Conflict-zone circle must have finite coordinates and positive radius.")
    points = np.column_stack([x, z])
    if np.linalg.norm(points[0] - center) <= radius:
        return float(time[0])
    radius_squared = radius * radius
    for index in range(len(time) - 1):
        start = points[index]
        delta = points[index + 1] - start
        duration = time[index + 1] - time[index]
        if duration <= 0:
            continue
        relative = start - center
        a = float(np.dot(delta, delta))
        if a == 0:
            continue
        b = 2.0 * float(np.dot(relative, delta))
        c = float(np.dot(relative, relative) - radius_squared)
        discriminant = b * b - 4.0 * a * c
        if discriminant < -1e-12:
            continue
        root_term = math.sqrt(max(0.0, discriminant))
        roots = sorted(((-b - root_term) / (2.0 * a), (-b + root_term) / (2.0 * a)))
        valid_roots = [root for root in roots if -1e-12 <= root <= 1.0 + 1e-12]
        if valid_roots:
            fraction = min(1.0, max(0.0, valid_roots[0]))
            return float(time[index] + fraction * duration)
    return math.nan


def interpolate_position(
    time: np.ndarray, x: np.ndarray, z: np.ndarray, event_time: float
) -> tuple[float, float]:
    """Interpolate an X/Z position without extrapolating beyond observed time."""
    time, x, z = _ordered_actor_samples(time, x, z)
    if (
        not np.isfinite(event_time)
        or len(time) == 0
        or event_time < time[0] - 1e-9
        or event_time > time[-1] + 1e-9
    ):
        return math.nan, math.nan
    return float(np.interp(event_time, time, x)), float(np.interp(event_time, time, z))


def _empty_result(
    participant: str,
    scenario: int,
    recording: str,
    path: str,
    entry_status: str,
    data_quality_status: str,
) -> dict[str, object]:
    return {
        "participant_id": participant,
        "scenario_id": scenario,
        "recording_path": path,
        "recording_identifier": recording,
        "car_conflict_zone_entry_time": math.nan,
        "pedestrian_conflict_zone_entry_time": math.nan,
        "first_conflict_zone_entry_time": math.nan,
        "first_entrant": "",
        "distance_at_first_conflict_zone_entry_m": math.nan,
        "scenario_mean_entry_distance_m": math.nan,
        "signed_difference_from_mean_m": math.nan,
        "absolute_difference_from_mean_m": math.nan,
        "within_scenario_deviation_rank": pd.NA,
        "manual_review_candidate": False,
        "entry_status": entry_status,
        "data_quality_status": data_quality_status,
        "manual_interaction_label": "",
        "manual_review_notes": "",
    }


def measure_recording(
    one: pd.DataFrame, zone: dict[str, object], recording_path: str = ""
) -> dict[str, object]:
    """Measure synchronized actor distance at the earlier valid zone entry."""
    required = {
        "participant_id", "scenario", "recording_identifier", "scenario_time",
        "ped_x", "ped_z", "veh_x", "veh_z",
    }
    participant = str(one.participant_id.iloc[0]) if "participant_id" in one and len(one) else ""
    scenario = int(one.scenario.iloc[0]) if "scenario" in one and len(one) else -1
    recording = (
        str(one.recording_identifier.iloc[0])
        if "recording_identifier" in one and len(one) else ""
    )
    if len(one) < 2 or not required.issubset(one.columns):
        return _empty_result(
            participant, scenario, recording, recording_path,
            "invalid_data", "invalid_data: missing required columns or usable rows",
        )
    try:
        ordered = one.sort_values("scenario_time").drop_duplicates("scenario_time")
        time = pd.to_numeric(ordered.scenario_time, errors="coerce").to_numpy(dtype=float)
        ped_x = pd.to_numeric(ordered.ped_x, errors="coerce").to_numpy(dtype=float)
        ped_z = pd.to_numeric(ordered.ped_z, errors="coerce").to_numpy(dtype=float)
        car_x = pd.to_numeric(ordered.veh_x, errors="coerce").to_numpy(dtype=float)
        car_z = pd.to_numeric(ordered.veh_z, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(time).sum() < 2:
            raise ValueError("fewer than two finite timestamps")
        if len(_ordered_actor_samples(time, ped_x, ped_z)[0]) < 2:
            raise ValueError("fewer than two finite pedestrian position samples")
        if len(_ordered_actor_samples(time, car_x, car_z)[0]) < 2:
            raise ValueError("fewer than two finite car position samples")
        car_entry = first_circle_entry_time(time, car_x, car_z, zone)
        pedestrian_entry = first_circle_entry_time(time, ped_x, ped_z, zone)
        entries = [value for value in (car_entry, pedestrian_entry) if np.isfinite(value)]
        result = _empty_result(
            participant, scenario, recording, recording_path,
            "no_actor_entry", "valid_data",
        )
        result["car_conflict_zone_entry_time"] = car_entry
        result["pedestrian_conflict_zone_entry_time"] = pedestrian_entry
        if not entries:
            return result
        event_time = min(entries)
        if np.isfinite(car_entry) and np.isfinite(pedestrian_entry):
            if abs(car_entry - pedestrian_entry) <= 1e-9:
                first_entrant = "simultaneous"
            else:
                first_entrant = "car" if car_entry < pedestrian_entry else "pedestrian"
            entry_status = "valid_both_actor_entry"
        elif np.isfinite(car_entry):
            first_entrant = "car"
            entry_status = "valid_single_actor_entry"
        else:
            first_entrant = "pedestrian"
            entry_status = "valid_single_actor_entry"
        car_position = interpolate_position(time, car_x, car_z, event_time)
        pedestrian_position = interpolate_position(time, ped_x, ped_z, event_time)
        if not np.isfinite([*car_position, *pedestrian_position]).all():
            raise ValueError("both actor positions are not available at the entry timestamp")
        result.update({
            "first_conflict_zone_entry_time": event_time,
            "first_entrant": first_entrant,
            "distance_at_first_conflict_zone_entry_m": float(np.hypot(
                car_position[0] - pedestrian_position[0],
                car_position[1] - pedestrian_position[1],
            )),
            "entry_status": entry_status,
        })
        return result
    except Exception as error:
        return _empty_result(
            participant, scenario, recording, recording_path,
            "invalid_data", f"invalid_data: {type(error).__name__}: {error}",
        )


def rank_within_scenarios(
    results: pd.DataFrame, top_n: int = TOP_N_REVIEW
) -> pd.DataFrame:
    """Add arithmetic scenario means, deviations, ordinal ranks, and flags.

    Ties are resolved deterministically by recording identifier so exactly
    ``top_n`` recordings are selected when at least that many are measurable.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least one")
    ranked = results.copy()
    for column in (
        "scenario_mean_entry_distance_m", "signed_difference_from_mean_m",
        "absolute_difference_from_mean_m",
    ):
        ranked[column] = np.nan
    ranked["within_scenario_deviation_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked["manual_review_candidate"] = False
    valid = ranked.distance_at_first_conflict_zone_entry_m.notna()
    for scenario, indices in ranked.loc[valid].groupby("scenario_id", sort=True).groups.items():
        values = ranked.loc[indices, "distance_at_first_conflict_zone_entry_m"].astype(float)
        mean = float(values.mean())
        ranked.loc[indices, "scenario_mean_entry_distance_m"] = mean
        signed = values - mean
        ranked.loc[indices, "signed_difference_from_mean_m"] = signed
        ranked.loc[indices, "absolute_difference_from_mean_m"] = signed.abs()
        ordered = ranked.loc[indices].sort_values(
            ["absolute_difference_from_mean_m", "recording_identifier"],
            ascending=[False, True], kind="mergesort",
        ).index
        ranked.loc[ordered, "within_scenario_deviation_rank"] = np.arange(1, len(ordered) + 1)
        ranked.loc[ordered[:top_n], "manual_review_candidate"] = True
    return ranked


def _actor_event_positions(
    one: pd.DataFrame, event_time: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    time = one.scenario_time.to_numpy(dtype=float)
    return (
        interpolate_position(time, one.veh_x.to_numpy(dtype=float), one.veh_z.to_numpy(dtype=float), event_time),
        interpolate_position(time, one.ped_x.to_numpy(dtype=float), one.ped_z.to_numpy(dtype=float), event_time),
    )


def build_review_panel(
    one: pd.DataFrame, result: pd.Series, zone: dict[str, object]
) -> go.Figure:
    figure = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Complete trajectories and shared conflict zone", "Synchronized distance over time"),
        horizontal_spacing=.10,
    )
    angles = np.linspace(0, 2 * np.pi, 160)
    figure.add_trace(go.Scatter(
        x=float(zone["center_x_m"]) + float(zone["radius_m"]) * np.cos(angles),
        y=float(zone["center_z_m"]) + float(zone["radius_m"]) * np.sin(angles),
        mode="lines", fill="toself", name="Shared conflict zone",
        line={"color": "#7A5195", "width": 2}, fillcolor="rgba(122,81,149,.14)",
    ), row=1, col=1)
    colors = {"car": "#D1495B", "pedestrian": "#2A6FBB"}
    figure.add_trace(go.Scatter(
        x=one.veh_x, y=one.veh_z, mode="lines", name="Car trajectory",
        line={"color": colors["car"], "width": 2.5},
    ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=one.ped_x, y=one.ped_z, mode="lines", name="Pedestrian trajectory",
        line={"color": colors["pedestrian"], "width": 2.5},
    ), row=1, col=1)
    event_time = float(result.first_conflict_zone_entry_time)
    car_position, pedestrian_position = _actor_event_positions(one, event_time)
    figure.add_trace(go.Scatter(
        x=[car_position[0], pedestrian_position[0]],
        y=[car_position[1], pedestrian_position[1]], mode="lines",
        name="Entry-event separation", line={"color": "#222", "width": 2, "dash": "dot"},
    ), row=1, col=1)
    for actor, position, symbol in (
        ("car", car_position, "square"), ("pedestrian", pedestrian_position, "circle")
    ):
        figure.add_trace(go.Scatter(
            x=[position[0]], y=[position[1]], mode="markers",
            name=f"{actor.title()} at first entry",
            marker={"color": colors[actor], "size": 12, "symbol": symbol, "line": {"color": "white", "width": 1}},
        ), row=1, col=1)
    distance_frame = one.sort_values("scenario_time").drop_duplicates("scenario_time").copy()
    distance_frame["distance_m"] = np.hypot(
        distance_frame.veh_x - distance_frame.ped_x,
        distance_frame.veh_z - distance_frame.ped_z,
    )
    figure.add_trace(go.Scatter(
        x=distance_frame.scenario_time, y=distance_frame.distance_m,
        mode="lines", name="Synchronized car–pedestrian distance",
        line={"color": "#3B7EA1", "width": 2.5},
        hovertemplate="Scenario time: %{x:.3f} s<br>Distance: %{y:.3f} m<extra></extra>",
    ), row=1, col=2)
    figure.add_vline(
        x=event_time, line={"color": "#111", "width": 2, "dash": "dash"},
        annotation_text="First conflict-zone entry", annotation_position="top right",
        row=1, col=2,
    )
    details = (
        f"Participant {escape(str(result.participant_id))} · Scenario {int(result.scenario_id)} · "
        f"First entrant: {escape(str(result.first_entrant))}<br>"
        f"Entry distance {result.distance_at_first_conflict_zone_entry_m:.3f} m · "
        f"Scenario mean {result.scenario_mean_entry_distance_m:.3f} m · "
        f"Signed difference {result.signed_difference_from_mean_m:+.3f} m · "
        f"Absolute difference {result.absolute_difference_from_mean_m:.3f} m · "
        f"Rank {int(result.within_scenario_deviation_rank)}<br>"
        f"Source: {escape(str(result.recording_path))}"
    )
    figure.update_layout(
        title={
            "text": f"<b>Manual-review panel</b><br><span style='font-size:13px'>{details}</span>",
            "x": .5, "xanchor": "center", "y": .98, "yanchor": "top",
        },
        template="plotly_white", height=760,
        legend={"orientation": "h", "x": 0, "y": 1.04, "yanchor": "bottom"},
        margin={"l": 75, "r": 45, "t": 185, "b": 75},
    )
    figure.update_xaxes(title_text="World X position (m)", row=1, col=1)
    figure.update_yaxes(title_text="World Z position (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_xaxes(title_text="Scenario time (s)", row=1, col=2)
    figure.update_yaxes(title_text="Horizontal distance (m)", rangemode="tozero", row=1, col=2)
    return figure


def build_validation_figure(traces: pd.DataFrame, zone: dict[str, object], scenario: int) -> go.Figure:
    figure = go.Figure()
    angles = np.linspace(0, 2 * np.pi, 160)
    figure.add_trace(go.Scatter(
        x=float(zone["center_x_m"]) + float(zone["radius_m"]) * np.cos(angles),
        y=float(zone["center_z_m"]) + float(zone["radius_m"]) * np.sin(angles),
        mode="lines", fill="toself", name="Shared conflict zone",
        line={"color": "#7A5195", "width": 2}, fillcolor="rgba(122,81,149,.14)",
    ))
    for _, one in traces.groupby("recording_identifier", sort=True):
        participant = str(one.participant_id.iloc[0])
        figure.add_trace(go.Scatter(
            x=one.veh_x, y=one.veh_z, mode="lines", name="Car",
            legendgroup="car", showlegend=not any(trace.name == "Car" for trace in figure.data),
            line={"color": "#D1495B", "width": 1}, opacity=.28,
            text=[participant] * len(one), hovertemplate="%{text} · car<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=one.ped_x, y=one.ped_z, mode="lines", name="Pedestrian",
            legendgroup="pedestrian", showlegend=not any(trace.name == "Pedestrian" for trace in figure.data),
            line={"color": "#2A6FBB", "width": 1}, opacity=.28,
            text=[participant] * len(one), hovertemplate="%{text} · pedestrian<extra></extra>",
        ))
        figure.add_annotation(
            x=float(one.ped_x.iloc[-1]), y=float(one.ped_z.iloc[-1]), text=participant,
            showarrow=False, font={"size": 9, "color": "#2A6FBB"},
        )
    figure.update_layout(
        title=f"Scenario {scenario}: conflict-zone trajectory validation",
        template="plotly_white", height=800,
        xaxis={"title": "World X position (m)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "World Z position (m)"},
        margin={"l": 80, "r": 45, "t": 100, "b": 70},
    )
    return figure


def build_distribution_figure(valid: pd.DataFrame, scenario: int) -> go.Figure:
    ordered = valid.sort_values("within_scenario_deviation_rank")
    mean = float(ordered.scenario_mean_entry_distance_m.iloc[0])
    colors = np.where(ordered.manual_review_candidate, "#E45756", "#4C78A8")
    figure = go.Figure(go.Scatter(
        x=ordered.distance_at_first_conflict_zone_entry_m,
        y=ordered.participant_id,
        mode="markers+text",
        text=ordered.participant_id,
        textposition="middle right",
        marker={"color": colors, "size": 10},
        customdata=np.column_stack([
            ordered.within_scenario_deviation_rank,
            ordered.signed_difference_from_mean_m,
            ordered.absolute_difference_from_mean_m,
        ]),
        hovertemplate=(
            "%{y}<br>Distance: %{x:.3f} m<br>Rank: %{customdata[0]}"
            "<br>Signed difference: %{customdata[1]:+.3f} m"
            "<br>Absolute difference: %{customdata[2]:.3f} m<extra></extra>"
        ),
    ))
    figure.add_vline(
        x=mean, line={"color": "#111", "width": 2, "dash": "dash"},
        annotation_text=f"Arithmetic mean: {mean:.3f} m",
        annotation_position="top right",
    )
    figure.update_layout(
        title=f"Scenario {scenario}: entry-event distance ranking",
        template="plotly_white", height=max(650, 25 * len(ordered) + 220),
        xaxis={"title": "Distance at first conflict-zone entry (m)", "rangemode": "tozero"},
        yaxis={"title": "Participant (rank order)", "categoryorder": "array", "categoryarray": ordered.participant_id.tolist()},
        showlegend=False, margin={"l": 105, "r": 130, "t": 100, "b": 75},
    )
    return figure


def _write_table_html(frame: pd.DataFrame, path: Path, title: str) -> None:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    table = display.to_html(index=False, escape=True, classes="rankings")
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title><style>body{{font:14px system-ui;margin:2rem;color:#222}}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.4rem;text-align:left}"
        "th{background:#f3f4f6;position:sticky;top:0}tr:nth-child(even){background:#fafafa}</style></head><body>"
        f"<h1>{escape(title)}</h1>{table}</body></html>", encoding="utf-8",
    )


def _safe_panel_name(recording: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", recording) + ".html"


def _review_index(candidates: pd.DataFrame, panel_paths: dict[str, str]) -> str:
    sections = []
    for scenario, rows in candidates.groupby("scenario_id", sort=True):
        entries = []
        for row in rows.sort_values("within_scenario_deviation_rank").itertuples():
            href = quote(panel_paths[str(row.recording_identifier)].replace("\\", "/"), safe="/._-")
            entries.append(
                "<tr>"
                f"<td>{int(row.within_scenario_deviation_rank)}</td>"
                f"<td><a href='{href}'>{escape(str(row.participant_id))}</a></td>"
                f"<td>{escape(str(row.first_entrant))}</td>"
                f"<td>{row.distance_at_first_conflict_zone_entry_m:.3f}</td>"
                f"<td>{row.scenario_mean_entry_distance_m:.3f}</td>"
                f"<td>{row.signed_difference_from_mean_m:+.3f}</td>"
                f"<td>{row.absolute_difference_from_mean_m:.3f}</td>"
                f"<td>{escape(str(row.recording_path))}</td></tr>"
            )
        sections.append(
            f"<section><h2>Scenario {int(scenario)} — {escape(scenario_description(int(scenario)))}</h2>"
            "<table><thead><tr><th>Rank</th><th>Participant / panel</th><th>First entrant</th>"
            "<th>Entry distance (m)</th><th>Scenario mean (m)</th><th>Signed difference (m)</th>"
            "<th>Absolute difference (m)</th><th>Source recording</th></tr></thead><tbody>"
            + "".join(entries) + "</tbody></table></section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Entry-distance manual review</title>"
        "<style>body{font:15px system-ui;margin:2rem;color:#222}section{margin:2.5rem 0}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.5rem;text-align:left}"
        "th{background:#f3f4f6}a{color:#1261a0}</style></head><body>"
        "<h1>Entry-distance manual-review candidates</h1>"
        "<p>Flags indicate only a large absolute difference from the same-scenario arithmetic mean. "
        "No interaction classification is assigned or implied.</p>" + "".join(sections) + "</body></html>"
    )


def _zone_for_scenario(conflict_dir: Path, scenario: int) -> dict[str, object] | None:
    path = conflict_dir / f"scenario_{scenario:02d}" / "conflict_zone_definition.json"
    if not path.exists():
        return None
    zone = json.loads(path.read_text(encoding="utf-8"))
    required = ("center_x_m", "center_z_m", "radius_m", "geometrically_supported")
    if not all(key in zone for key in required):
        return None
    return zone


def analyze(
    trajectory_dir: Path,
    conflict_dir: Path,
    output_dir: Path,
    scenarios: list[int],
    top_n: int = TOP_N_REVIEW,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete ranking and manual-review artifact workflow."""
    if top_n < 1:
        raise ValueError("top_n must be at least one")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
    diagnostics_path = trajectory_dir / "coordinate_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    diagnostics = diagnostics[diagnostics.scenario.isin(scenarios)].copy()
    source_lookup = diagnostics.set_index("recording_identifier").source_path.to_dict()
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="entry_distance_manual_review_", dir=output_parent))
    all_results: list[dict[str, object]] = []
    trace_lookup: dict[int, pd.DataFrame] = {}
    zones: dict[int, dict[str, object] | None] = {}
    try:
        for scenario in sorted(scenarios):
            zone = _zone_for_scenario(conflict_dir, scenario)
            zones[scenario] = zone
            scenario_diagnostics = diagnostics[diagnostics.scenario.eq(scenario)]
            trace_path = trajectory_dir / f"scenario_{scenario:02d}" / "raw_trajectory_samples.csv"
            try:
                traces = pd.read_csv(trace_path)
                trace_lookup[scenario] = traces
            except Exception as error:
                traces = pd.DataFrame()
                for row in scenario_diagnostics.itertuples():
                    all_results.append(_empty_result(
                        str(row.participant_id), scenario, str(row.recording_identifier),
                        str(row.source_path), "invalid_data",
                        f"invalid_data: {type(error).__name__}: {error}",
                    ))
                continue
            grouped = {key: one for key, one in traces.groupby("recording_identifier", sort=True)}
            recording_ids = sorted(set(grouped) | set(scenario_diagnostics.recording_identifier.astype(str)))
            for recording in recording_ids:
                path = str(source_lookup.get(recording, ""))
                if recording not in grouped:
                    diagnostic = scenario_diagnostics[
                        scenario_diagnostics.recording_identifier.astype(str).eq(recording)
                    ]
                    participant = str(diagnostic.participant_id.iloc[0]) if len(diagnostic) else ""
                    all_results.append(_empty_result(
                        participant, scenario, recording, path,
                        "invalid_data", "invalid_data: recording missing from trajectory samples",
                    ))
                elif zone is None or not bool(zone.get("geometrically_supported", False)):
                    one = grouped[recording]
                    all_results.append(_empty_result(
                        str(one.participant_id.iloc[0]), scenario, recording, path,
                        "no_supported_conflict_zone", "not_evaluable: no geometrically supported conflict zone",
                    ))
                else:
                    all_results.append(measure_recording(grouped[recording], zone, path))
        results = rank_within_scenarios(pd.DataFrame(all_results), top_n=top_n)
        results = results[OUTPUT_COLUMNS].sort_values(
            ["scenario_id", "within_scenario_deviation_rank", "recording_identifier"],
            na_position="last", kind="mergesort",
        ).reset_index(drop=True)
        candidates = results[results.manual_review_candidate].copy().sort_values(
            ["scenario_id", "within_scenario_deviation_rank"], kind="mergesort"
        )
        results.to_csv(staging / "recording_level_entry_distance_ranking.csv", index=False)
        candidates.to_csv(staging / "manual_review_candidates.csv", index=False)
        summary_rows = []
        panel_paths: dict[str, str] = {}
        for scenario in sorted(scenarios):
            rows = results[results.scenario_id.eq(scenario)]
            zone = zones[scenario]
            valid = rows[rows.distance_at_first_conflict_zone_entry_m.notna()]
            summary_rows.append({
                "scenario_id": scenario,
                "total_recordings_found": int(len(rows)),
                "valid_distance_measurements": int(len(valid)),
                "mean_entry_event_distance_m": (
                    float(valid.scenario_mean_entry_distance_m.iloc[0]) if len(valid) else math.nan
                ),
                "manual_review_candidates": int(rows.manual_review_candidate.sum()),
                "neither_actor_entered": int(rows.entry_status.eq("no_actor_entry").sum()),
                "invalid_data": int(rows.entry_status.eq("invalid_data").sum()),
                "conflict_zone_availability": (
                    "geometrically_supported" if zone is not None and bool(zone.get("geometrically_supported", False))
                    else "no_supported_conflict_zone"
                ),
            })
            if not len(valid) or zone is None or scenario not in trace_lookup:
                continue
            scenario_dir = staging / f"scenario_{scenario:02d}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            traces = trace_lookup[scenario]
            build_validation_figure(traces, zone, scenario).write_html(
                scenario_dir / "conflict_zone_trajectory_validation.html",
                include_plotlyjs=True, full_html=True,
                config={"displaylogo": False, "responsive": True},
            )
            build_distribution_figure(valid, scenario).write_html(
                scenario_dir / "entry_distance_distribution.html",
                include_plotlyjs=True, full_html=True,
                config={"displaylogo": False, "responsive": True},
            )
            ranked_table = rows.sort_values("within_scenario_deviation_rank", na_position="last")
            ranked_table.to_csv(scenario_dir / "ranked_recordings.csv", index=False)
            _write_table_html(
                ranked_table, scenario_dir / "ranked_recordings.html",
                f"Scenario {scenario} complete entry-distance ranking",
            )
            for _, candidate in candidates[candidates.scenario_id.eq(scenario)].iterrows():
                recording = str(candidate.recording_identifier)
                one = traces[traces.recording_identifier.astype(str).eq(recording)]
                panel_name = _safe_panel_name(recording)
                build_review_panel(one, candidate, zone).write_html(
                    scenario_dir / panel_name, include_plotlyjs=True, full_html=True,
                    config={"displaylogo": False, "responsive": True},
                )
                panel_paths[recording] = f"scenario_{scenario:02d}/{panel_name}"
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(staging / "scenario_summary.csv", index=False)
        (staging / "manual_review_index.html").write_text(
            _review_index(candidates, panel_paths), encoding="utf-8"
        )
        (staging / "analysis_metadata.json").write_text(json.dumps({
            "purpose": "Within-scenario descriptive ranking for manual review only",
            "top_n_review": top_n,
            "distance_formula": "sqrt((car_x-pedestrian_x)^2 + (car_z-pedestrian_z)^2)",
            "time_synchronization": "Both actor positions linearly interpolated at the same first valid conflict-zone entry time",
            "scenario_statistic": "ordinary arithmetic mean of valid entry-event distances",
            "tie_handling": "Absolute-deviation ties are ordered by recording_identifier; exactly top_n are flagged when available",
            "automatic_interaction_classification": False,
        }, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            if overwrite:
                shutil.rmtree(output_dir)
            elif not any(output_dir.iterdir()):
                output_dir.rmdir()
        staging.replace(output_dir)
        return results, summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.scenarios) - set(SCENARIO_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Missing centralized descriptions for scenarios: {unknown}")
    analyze(
        args.trajectory_input.resolve(), args.conflict_zone_input.resolve(),
        args.output_dir.resolve(), args.scenarios, args.top_n_review, args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
