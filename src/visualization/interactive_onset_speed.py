"""Export standalone interactive onset-aligned pedestrian-speed plots.

This consumes the recording-level trace checkpoint written by
``src/scenario_pattern_expanded.py``. The checkpoint already contains the
existing movement-onset detection, 9-sample centered rolling-mean speed, and
onset alignment, so this exporter deliberately does not repeat those steps.

The group statistic matches the static six-panel figure: trajectories are
linearly interpolated within (never beyond) each recording's available time
range on a 0.1-second grid; the center is the median and the band is the 25th
to 75th percentile (IQR).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS = (3, 7, 12, 15, 16, 21)
DEFAULT_INPUT = PROJECT_ROOT / "outputs/scenario_pattern_expanded/traces/aligned_speed_raw_traces.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/interactive_speed_plots"
DEFAULT_VEHICLE_OUTPUT = PROJECT_ROOT / "outputs/interactive_vehicle_speed_plots"
DEFAULT_TRAJECTORY_INPUT = PROJECT_ROOT / "outputs/trajectory_analysis"
DEFAULT_EVENT_INPUT = PROJECT_ROOT / "outputs/conflict_zone_analysis"
GRID_SECONDS = 0.1
SCENARIO_LABELS = {
    3: {
        "vehicle_approach": "Orthogonal",
        "pedestrian_position": "Curb near vehicle lane, same side of road",
        "main_title": "Orthogonal approach — pedestrian near vehicle lane, same side",
        "subtitle": "Scenario 3 · Onset-aligned pedestrian speed",
    },
    7: {
        "vehicle_approach": "Orthogonal",
        "pedestrian_position": "Curb near vehicle lane, opposite side of road",
        "main_title": "Orthogonal approach — pedestrian near vehicle lane, opposite side",
        "subtitle": "Scenario 7 · Onset-aligned pedestrian speed",
    },
    12: {
        "vehicle_approach": "Orthogonal",
        "pedestrian_position": "Curb far from vehicle lane, opposite side of road",
        "main_title": "Orthogonal approach — pedestrian far from vehicle lane, opposite side",
        "subtitle": "Scenario 12 · Onset-aligned pedestrian speed",
    },
    15: {
        "vehicle_approach": "Parallel/contra-directional",
        "pedestrian_position": "Curb near vehicle lane, opposite side of road",
        "main_title": (
            "Parallel/contra-directional approach — "
            "pedestrian near vehicle lane, opposite side"
        ),
        "subtitle": "Scenario 15 · Onset-aligned pedestrian speed",
    },
    16: {
        "vehicle_approach": "Parallel/contra-directional",
        "pedestrian_position": "Curb far from vehicle lane, opposite side of road",
        "main_title": (
            "Parallel/contra-directional approach — "
            "pedestrian far from vehicle lane, opposite side"
        ),
        "subtitle": "Scenario 16 · Onset-aligned pedestrian speed",
    },
    21: {
        "vehicle_approach": "Parallel/contra-directional",
        "pedestrian_position": "Curb near vehicle lane, same side of road",
        "main_title": (
            "Parallel/contra-directional approach — "
            "pedestrian near vehicle lane, same side"
        ),
        "subtitle": "Scenario 21 · Onset-aligned pedestrian speed",
    },
}


@dataclass(frozen=True)
class ScenarioSummary:
    scenario: int
    maximum_n: int
    minimum_summary_n: int
    coverage_fraction: float
    participant_count: int
    minimum_end_time: float
    maximum_end_time: float
    low_coverage_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", type=int, default=list(DEFAULT_SCENARIOS))
    parser.add_argument(
        "--coverage-threshold", type=float, default=0.5,
        help="Fraction of maximum contributors required for the group summary.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_traces(path: Path, scenarios: list[int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "pednyc_number", "scenario_number", "recording_identifier",
        "aligned_time", "pedestrian_speed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame["scenario_number"].isin(scenarios)].copy()
    frame["aligned_time"] = pd.to_numeric(frame["aligned_time"], errors="coerce")
    frame["pedestrian_speed"] = pd.to_numeric(frame["pedestrian_speed"], errors="coerce")
    frame = frame.dropna(subset=["aligned_time", "pedestrian_speed"])
    absent = sorted(set(scenarios).difference(frame["scenario_number"].unique()))
    if absent:
        raise ValueError(f"No valid traces found for scenarios: {absent}")
    return frame


def _grid_for(traces: pd.DataFrame) -> np.ndarray:
    lower = math.floor(float(traces["aligned_time"].min()) / GRID_SECONDS) * GRID_SECONDS
    upper = math.ceil(float(traces["aligned_time"].max()) / GRID_SECONDS) * GRID_SECONDS
    return np.arange(lower, upper + GRID_SECONDS / 2, GRID_SECONDS)


def calculate_summary(
    traces: pd.DataFrame, coverage_fraction: float, speed_column: str = "pedestrian_speed"
) -> tuple[pd.DataFrame, ScenarioSummary]:
    """Calculate the static figure's pointwise median and IQR on a full grid."""
    if not 0 < coverage_fraction <= 1:
        raise ValueError("--coverage-threshold must be in (0, 1].")
    scenario = int(traces["scenario_number"].iloc[0])
    grid = _grid_for(traces)
    curves: list[np.ndarray] = []
    end_times: list[float] = []
    for _, one in traces.groupby("recording_identifier", sort=True):
        one = one.sort_values("aligned_time").drop_duplicates("aligned_time")
        if len(one) < 2:
            continue
        x = one["aligned_time"].to_numpy()
        y = one[speed_column].to_numpy()
        curves.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
        end_times.append(float(x[-1]))
    if not curves:
        raise ValueError(f"Scenario {scenario} has no trace with at least two samples.")
    stack = np.vstack(curves)
    contributors = np.isfinite(stack).sum(axis=0)
    maximum_n = int(contributors.max())
    minimum_summary_n = int(math.ceil(coverage_fraction * maximum_n))
    covered = contributors >= minimum_summary_n
    median = np.full(len(grid), np.nan)
    q1 = np.full(len(grid), np.nan)
    q3 = np.full(len(grid), np.nan)
    any_data = contributors > 0
    median[any_data] = np.nanmedian(stack[:, any_data], axis=0)
    q1[any_data], q3[any_data] = np.nanpercentile(stack[:, any_data], [25, 75], axis=0)
    median[~covered] = np.nan
    q1[~covered] = np.nan
    q3[~covered] = np.nan
    summary = pd.DataFrame({
        "aligned_time": grid,
        "median_speed": median,
        "q1_speed": q1,
        "q3_speed": q3,
        "contributing_recordings": contributors,
        "summary_visible": covered,
    })
    metadata = ScenarioSummary(
        scenario=scenario,
        maximum_n=maximum_n,
        minimum_summary_n=minimum_summary_n,
        coverage_fraction=coverage_fraction,
        participant_count=len(curves),
        minimum_end_time=min(end_times),
        maximum_end_time=max(end_times),
        low_coverage_points=int((any_data & ~covered).sum()),
    )
    return summary, metadata


def _low_coverage_ranges(summary: pd.DataFrame) -> list[tuple[float, float]]:
    low = (summary["contributing_recordings"].gt(0) & ~summary["summary_visible"]).to_numpy()
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(np.append(low, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append((
                float(summary["aligned_time"].iloc[start] - GRID_SECONDS / 2),
                float(summary["aligned_time"].iloc[index - 1] + GRID_SECONDS / 2),
            ))
            start = None
    return ranges


def build_figure(
    traces: pd.DataFrame, summary: pd.DataFrame, metadata: ScenarioSummary,
    events: pd.DataFrame | None = None, speed_column: str = "pedestrian_speed",
    actor: str = "pedestrian",
) -> go.Figure:
    try:
        label = SCENARIO_LABELS[metadata.scenario]
    except KeyError as error:
        raise ValueError(
            f"No descriptive condition label configured for scenario {metadata.scenario}."
        ) from error
    figure = go.Figure()
    for identity, one in traces.groupby("recording_identifier", sort=True):
        one = one.sort_values("aligned_time")
        pednyc = int(one["pednyc_number"].iloc[0])
        original = one["original_time"] if "original_time" in one else one["aligned_time"]
        event_matches = events[events.recording_identifier.eq(identity)] if events is not None and not events.empty else pd.DataFrame()
        zone_status = "potential/counterfactual" if len(event_matches) and event_matches.iloc[0].get("zone_event_status") == "potential_counterfactual" else "observed/actual"
        customdata = np.column_stack([
            np.repeat(f"PedNYC{pednyc}", len(one)),
            np.repeat(identity, len(one)),
            original,
            np.repeat(zone_status, len(one)),
        ])
        figure.add_trace(go.Scatter(
            x=one["aligned_time"], y=one[speed_column], mode="lines",
            name=identity, legendgroup=identity, showlegend=True,
            line={"color": "#4C78A8", "width": 1}, opacity=0.22,
            customdata=customdata,
            meta={
                "role": "participant", "participant_id": f"PedNYC{pednyc}",
                "recording_identifier": identity, "scenario": metadata.scenario,
            },
            hovertemplate=(
                "Participant: %{customdata[0]}<br>"
                f"Scenario: {metadata.scenario}<br>"
                "Aligned time: %{x:.2f} s<br>Original scenario time: %{customdata[2]:.2f} s<br>"
                f"Smoothed {actor} ground-plane speed: %{{y:.3f}} m/s<br>Conflict-zone event: trace sample<br>Zone status: %{{customdata[3]}}<extra></extra>"
            ),
        ))
        if events is not None and not events.empty:
            matches = event_matches
            if len(matches):
                event = matches.iloc[0]
                onset = float(original.iloc[0] - one.aligned_time.iloc[0])
                potential = event.get("zone_event_status") == "potential_counterfactual"
                if actor == "pedestrian":
                    specs = [("Pedestrian zone entry", "pedestrian_entry_time", "triangle-right"), ("Pedestrian zone center / closest point", "pedestrian_center_time", "diamond"), ("Pedestrian zone exit", "pedestrian_exit_time", "x")]
                    specs.append(("Vehicle arrival at zone", "counterfactual_vehicle_arrival_time" if potential else "vehicle_entry_time", "square"))
                    entry, exit_time = event.get("pedestrian_entry_time"), event.get("pedestrian_exit_time")
                    if np.isfinite(entry) and np.isfinite(exit_time):
                        mask = original.between(entry, exit_time)
                        if mask.any():
                            figure.add_trace(go.Scatter(
                                x=one.loc[mask, "aligned_time"],
                                y=one.loc[mask, speed_column],
                                mode="lines",
                                fill="tozeroy",
                                name=(
                                    "Potential conflict-zone occupancy"
                                    if potential else "Conflict-zone occupancy"
                                ),
                                legendgroup=identity,
                                showlegend=False,
                                line={"color": "rgba(0,0,0,0)", "width": 0},
                                fillcolor=(
                                    "rgba(148,103,189,.16)"
                                    if potential else "rgba(90,90,90,.16)"
                                ),
                                hoverinfo="skip",
                                meta={
                                    "role": "zone_occupancy",
                                    "participant_id": f"PedNYC{pednyc}",
                                    "display": "shaded_conflict_zone",
                                },
                            ))
                elif potential:
                    specs = [("Counterfactual vehicle arrival", "counterfactual_vehicle_arrival_time", "diamond"), ("Pedestrian arrival at zone", "pedestrian_entry_time", "triangle-up")]
                else:
                    specs = [("Actual vehicle zone entry", "vehicle_actual_entry_time", "triangle-right"), ("Actual vehicle zone center / closest point", "vehicle_actual_center_time", "diamond"), ("Actual vehicle zone exit", "vehicle_actual_exit_time", "x"), ("Pedestrian arrival at zone", "pedestrian_entry_time", "triangle-up")]
                for event_name, column, symbol in specs:
                    event_time = event.get(column, np.nan)
                    if not np.isfinite(event_time):
                        continue
                    index = int(np.argmin(np.abs(original.to_numpy() - event_time)))
                    status = "counterfactual" if (potential and actor == "vehicle") else ("potential zone" if potential else "actual observed")
                    figure.add_trace(go.Scatter(x=[one.aligned_time.iloc[index]], y=[one[speed_column].iloc[index]], mode="markers", name=event_name, legendgroup=identity, showlegend=False, marker={"size":8,"symbol":symbol,"color":"#9467BD" if potential else "#555555"}, customdata=[[f"PedNYC{pednyc}", metadata.scenario, event_time, event_name, status]], meta={"role":"zone_event_marker","participant_id":f"PedNYC{pednyc}","event_status":status}, hovertemplate="Participant: %{customdata[0]}<br>Scenario: %{customdata[1]}<br>Aligned time: %{x:.2f} s<br>Original scenario time: %{customdata[2]:.2f} s<br>Speed: %{y:.3f} m/s<br>Event: %{customdata[3]}<br>Status: %{customdata[4]}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=summary["aligned_time"], y=summary["q1_speed"], mode="lines",
        line={"width": 0}, hoverinfo="skip", showlegend=False,
        meta={"role": "variability_band"},
    ))
    figure.add_trace(go.Scatter(
        x=summary["aligned_time"], y=summary["q3_speed"], mode="lines",
        line={"width": 0}, fill="tonexty", fillcolor="rgba(245,133,24,.24)",
        name="Interquartile range", hoverinfo="skip", meta={"role": "variability_band"},
    ))
    figure.add_trace(go.Scatter(
        x=summary["aligned_time"], y=summary["median_speed"], mode="lines",
        name="Median speed", line={"color": "#D1495B", "width": 3},
        meta={"role": "group_summary"},
        hovertemplate="Median speed: %{y:.3f} m/s<br>Time: %{x:.2f} s<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=summary["aligned_time"], y=summary["contributing_recordings"], mode="lines",
        name="Contributing recordings", yaxis="y2",
        line={"color": "#666666", "width": 1.5, "dash": "dash"},
        meta={"role": "contributor_count"},
        hovertemplate="Contributing recordings: %{y:.0f}<br>Time: %{x:.2f} s<extra></extra>",
    ))
    shapes = [{
        "type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1,
        "xref": "x", "yref": "paper",
        "line": {"color": "#222222", "width": 1.2, "dash": "dot"},
    }]
    annotations = []
    threshold_pct = metadata.coverage_fraction * 100
    for start, end in _low_coverage_ranges(summary):
        shapes.append({
            "type": "rect", "x0": start, "x1": end, "y0": 0, "y1": 1,
            "xref": "x", "yref": "paper", "layer": "below",
            "fillcolor": "rgba(128,128,128,.12)", "line": {"width": 0},
        })
        annotations.append({
            "x": (start + end) / 2, "y": 0.98, "xref": "x", "yref": "paper",
            "text": (
                "Individual recordings continue; group summary suppressed because "
                f"fewer than {threshold_pct:g}% of recordings remain"
            ),
            "showarrow": False, "font": {"size": 10, "color": "#555555"},
            "bgcolor": "rgba(255,255,255,.78)",
        })
    wrapped_main_title = label["main_title"].replace(" — ", " —<br>")
    subtitle = label["subtitle"] if actor == "pedestrian" else f"Scenario {metadata.scenario} · Vehicle speed aligned to pedestrian movement onset"
    figure.update_layout(
        title={
            "text": (
                f"<b>{wrapped_main_title}</b>"
                f"<br><span style='font-size:14px'>{subtitle}</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 21},
        },
        template="plotly_white", hovermode="closest",
        height=760,
        xaxis={"title": "Seconds from pedestrian movement onset", "showgrid": True},
        yaxis={"title": f"Smoothed {actor} ground-plane speed (m/s)", "rangemode": "tozero", "range": [0, 16] if actor == "vehicle" else None},
        yaxis2={
            "title": "Contributing recordings", "overlaying": "y", "side": "right",
            "range": [0, metadata.maximum_n * 1.15], "rangemode": "tozero",
        },
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.025,
            "yanchor": "bottom",
            "font": {"size": 12},
            "entrywidth": 190,
            "entrywidthmode": "pixels",
            "traceorder": "normal",
        },
        margin={"l": 80, "r": 85, "t": 170, "b": 75},
        shapes=shapes, annotations=annotations,
    )
    return figure


CLICK_SCRIPT = r"""
(function () {
  const plot = document.getElementById('{plot_id}');
  let selectedParticipant = null;
  function participantIndices() {
    const indices = [];
    plot.data.forEach((trace, index) => {
      if (trace.meta && trace.meta.role === 'participant') indices.push(index);
    });
    return indices;
  }
  function resetParticipants() {
    const indices = participantIndices();
    if (indices.length) Plotly.restyle(plot, {
      'line.color': indices.map(() => '#4C78A8'),
      'line.width': indices.map(() => 1),
      'opacity': indices.map(() => 0.22)
    }, indices);
    selectedParticipant = null;
  }
  plot.on('plotly_click', function (event) {
    if (!event.points.length) return;
    const clicked = plot.data[event.points[0].curveNumber];
    if (!clicked.meta || clicked.meta.role !== 'participant') return;
    const participant = clicked.meta.participant_id;
    if (selectedParticipant === participant) {
      resetParticipants();
      return;
    }
    const indices = participantIndices();
    const selected = indices.filter(
      index => plot.data[index].meta.participant_id === participant
    );
    Plotly.restyle(plot, {
      'line.color': indices.map(index =>
        plot.data[index].meta.participant_id === participant ? '#173F5F' : '#A8B9C8'),
      'line.width': indices.map(index =>
        plot.data[index].meta.participant_id === participant ? 4 : 0.7),
      'opacity': indices.map(index =>
        plot.data[index].meta.participant_id === participant ? 1 : 0.06)
    }, indices);
    if (selected.length === 1) Plotly.moveTraces(plot, selected[0], indices.length - 1);
    selectedParticipant = participant;
  });
  plot.on('plotly_doubleclick', function () {
    resetParticipants();
    return false;
  });
  plot.setAttribute('data-click-highlight-ready', 'true');
})();
"""


def export_scenario(
    traces: pd.DataFrame, output_path: Path, coverage_fraction: float,
    events: pd.DataFrame | None = None, speed_column: str = "pedestrian_speed",
    actor: str = "pedestrian",
) -> ScenarioSummary:
    summary, metadata = calculate_summary(traces, coverage_fraction, speed_column)
    figure = build_figure(traces, summary, metadata, events, speed_column, actor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path, include_plotlyjs=True, full_html=True, post_script=CLICK_SCRIPT,
        config={"displaylogo": False, "responsive": True},
    )
    return metadata


def load_trajectory_speed(input_root: Path, scenario: int) -> pd.DataFrame:
    frame = pd.read_csv(input_root / f"scenario_{scenario:02d}" / "raw_trajectory_samples.csv")
    return frame.rename(columns={"scenario":"scenario_number", "scenario_time":"original_time", "ped_speed":"pedestrian_speed", "veh_speed":"vehicle_speed"})


def main() -> int:
    args = parse_args()
    results = []
    vehicle_results = []
    for scenario in args.scenarios:
        one = load_trajectory_speed(DEFAULT_TRAJECTORY_INPUT, scenario)
        events = pd.read_csv(DEFAULT_EVENT_INPUT / f"scenario_{scenario:02d}" / "conflict_zone_events.csv")
        path = args.output_dir.resolve() / f"scenario_{scenario:02d}_onset_aligned.html"
        metadata = export_scenario(one, path, args.coverage_threshold, events)
        results.append({**metadata.__dict__, "output": str(path)})
        vehicle_path = DEFAULT_VEHICLE_OUTPUT / f"scenario_{scenario:02d}_onset_aligned_vehicle_speed.html"
        vehicle_metadata = export_scenario(one, vehicle_path, args.coverage_threshold, events, "vehicle_speed", "vehicle")
        vehicle_results.append({**vehicle_metadata.__dict__, "output": str(vehicle_path)})
        print(
            f"Scenario {scenario}: maximum n={metadata.maximum_n}, "
            f"minimum summary n={metadata.minimum_summary_n} -> {path}"
        )
    manifest = args.output_dir.resolve() / "validation_summary.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (DEFAULT_VEHICLE_OUTPUT / "validation_summary.json").write_text(json.dumps(vehicle_results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
