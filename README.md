# PedNYC analysis

## Scenario-agnostic canonical v2/v6 pipeline

Install `requirements.txt`, then give the pipeline a compatible logger CSV. It
runs Unity-array decoding, metrics v1, macrosegmentation v2, microsegmentation
v6, all canonical diagnostic plots, Gantt rendering, validation, and a
scenario-specific Markdown report without changing the tuned algorithms.

```powershell
python src\scenario_pipeline\run_scenario.py data\raw\CSV_Scenario-Ped-3_Session-temp_2024-02-22-13-48-59.csv
python src\scenario_pipeline\run_scenario.py --scenario-id 3
python src\scenario_pipeline\run_scenario.py --all --continue-on-error
```

Use `--output-root PATH` to relocate generated artifacts, `--skip-render` to
skip renderer outputs, `--preview-only` to request one preview per macro
segment, or `--preview-time SECONDS` for one preview. Pass an encoder using
`--ffmpeg PATH`. If no encoder is available, the normal run continues and
writes macro-midpoint previews plus a warning in the report.

Decoded and metrics tables are written below
`data/processed/pednyc1/scenarioN`. Canonical graphs and the report are written
below `outputs/graphs/exact_v6/pednyc1/scenarioN`; the complete micro output set
is under its `micro/v6` directory. Stage 1 also writes scenario-specific copies
below `4frame_view/pednyc1/scenarioN/metrics_v1`.

Every micro segment retains three independent dimensions: `motion_tag`,
`head_tag`, and `car_tag`. They are deliberately not fused. Observable tags are
direct threshold measurements; inferred tags are structured hypotheses such as
hesitation, yielding, proceeding, conflicted motion, or brief head checking.

Known limitations:

- Inputs must use the same logger column layout and filename convention
  `CSV_Scenario-Ped-N_Session-*.csv`.
- Scenario 3 is the only compatible raw CSV currently committed.
- `ScenarioTime` is used when it is seconds-like; the canonical fallback logic
  uses cleaned `dt`, then `GameTime`, then frame number timing.
- The committed scenario-3 ground-truth raw-time columns came from a different
  local feature file. Regression therefore compares elapsed boundaries, macro
  labels, and ordered motion/head/car tags rather than raw-time bytes.
- Macro boundary candidates can collapse; reports state the actual segment
  count instead of assuming every scenario has five segments.
- The Stage 1 implementation produces up to ten numbered diagnostic PNGs; the
  obsolete kink-workflow PNG is not part of the canonical Stage 1 output.

## Animated micro-segmentation Gantt MP4

Generate the exact v6 micro-segmentation data, then render the compositing-ready
video:

```powershell
python src\segmentation\micro_segmentation.py
python src\segmentation\render_micro_gantt_mp4.py
```

The renderer writes:

`outputs/graphs/micro_segmentation/v6/micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4`

The default is 1280x720 H.264 at 30 fps. The red playhead advances in scenario
time, and the bottom card shows the active macro segment, micro-segment, motion,
head, and car-context tags. Install `requirements.txt`, put ffmpeg on `PATH`, or
pass an explicit encoder with `--ffmpeg C:\path\to\ffmpeg.exe`.

Render a single layout preview without invoking ffmpeg:

```powershell
python src\segmentation\render_micro_gantt_mp4.py --preview-time 13.577
```

## Interactive onset-aligned pedestrian-speed plots

After the expanded scenario-pattern analysis has written
`outputs/scenario_pattern_expanded/traces/aligned_speed_raw_traces.csv`, export
the six standalone Plotly HTML files with:

```powershell
python src\visualization\interactive_onset_speed.py
```

The default outputs are written to `outputs/interactive_speed_plots`. Use
`--scenarios 3 7`, `--coverage-threshold 0.5`, `--input PATH`, or
`--output-dir PATH` to override the defaults. Individual recordings retain
their full available onset-aligned durations. The pointwise group summary
matches the static six-panel plot: median and 25th–75th percentile band after
within-recording interpolation to a 0.1-second grid. It is suppressed wherever
fewer than the requested fraction of the scenario's maximum contributors
remain.

## Pedestrian and vehicle trajectory analysis

Run the raw coordinate audit first:

```powershell
python -m src.analysis.trajectory_analysis --phase raw
```

After reviewing `outputs/trajectory_analysis/coordinate_frame_summary.csv` and
the six raw joint-trajectory HTML files, generate normalized-progress paths,
mean paths, covariance regions, deviations, and recording-level metrics:

```powershell
python -m src.analysis.trajectory_analysis --phase full
```

Options include `--scenarios`, `--input`, `--output-dir`, and
`--progress-points`. Normalized progress is a resampling axis, not normalized
physical speed or distance. The interval begins at detected pedestrian
movement onset and ends at the final subsequent sample at or above the
existing 0.15 m/s movement threshold.

## Pedestrian–vehicle conflict-zone analysis

Run this only after the trajectory outputs have been generated and reviewed:

```powershell
python -m src.analysis.conflict_zone_analysis
```

Options include `--scenarios`, `--input`, `--output-dir`,
`--conflict-zone-buffer`, and `--coverage-threshold`. The default estimated
zone buffer is 2.0 m. Scenario-level JSON metadata reports whether the observed
path geometry supports the estimated shared zone; unsupported zones are not
silently interpreted as crossing conflicts.

Scenarios 3, 7, 15, and 21 retain that observed-path circular definition.
Scenarios 12 and 16 instead use one fixed scenario-level potential zone: the
intersection of a 0.75 m half-width representative pedestrian corridor and a
1.35 m half-width counterfactual vehicle corridor. Scenario 12 continues the
east-to-west approach lane straight. In world coordinates, Scenario 16
approaches south (decreasing Z) and its alternative turns east (increasing X),
which is a vehicle-relative left turn. These zones represent unresolved
plausible maneuvers, not realized vehicle occupancy.

All coordinates and corridor widths are in meters. The representative path is
the pointwise median of recording paths resampled at 160 equal path-progress
increments. The corridor intersection is evaluated on a deterministic 0.05 m
world-coordinate grid and stored as its convex-hull polygon. Scenario 16 uses
an 8.0 m circular turn radius. These are uncertainty-based interaction zones,
not observed collision points: they assume the configured alternative maneuver
was plausible until commitment, do not model driver signaling or sight lines,
and do not imply that the recorded vehicle occupied the polygon. Recordings in
which the pedestrian never enters retain missing entry/exit values and an
explicit `never_entered_zone` status; actual vehicle entry/exit is likewise
not applicable for these counterfactual zones.

Pedestrian entry, center/closest-point, and exit are mapped from each recording's
observed positions. Entry and exit speeds and a documented one-second-before
versus one-second-after entry speed/acceleration window are included in the
event CSV. For potential zones, actual vehicle occupancy columns remain empty;
counterfactual arrival uses distance along the configured alternative from the
maneuver-commitment position divided by median observed approach speed above
0.25 m/s. `delta_TTZ_seconds` is pedestrian entry time minus counterfactual
vehicle arrival time, so a negative value means the pedestrian arrives first.

Running `python -m src.visualization.interactive_onset_speed` after this analysis
regenerates pedestrian graphs in `outputs/interactive_speed_plots` and matching
vehicle graphs in `outputs/interactive_vehicle_speed_plots`. Both use time zero
at each recording's detected pedestrian movement onset, retain participant
legend toggles, and transform all event timestamps with that same onset offset.

## Entry-distance manual review

After the trajectory and conflict-zone outputs exist, generate descriptive
within-scenario rankings and a visual manual-review package with:

```powershell
python -m src.analysis.entry_distance_manual_review
```

For each geometrically supported scenario, both actors' positions are linearly
interpolated at the same first conflict-zone entry time. The synchronized
horizontal distance is compared only with the ordinary arithmetic mean for
that scenario. By default, the three largest absolute differences are flagged
for manual inspection. Ties are ordered deterministically by recording ID so
the review package contains exactly three recordings when at least three valid
measurements exist. The flag is not an interaction classification and no
automatic interaction or no-interaction label is produced.

Artifacts are written to `outputs/entry_distance_manual_review`. Use
`--top-n-review` to change the review count or `--overwrite` to replace that
output directory on a later run.

## Avatar position and rotation plots

Render the six raw avatar position/rotation channels for every decoded CSV
found recursively for PedNYC1, PedNYC2, and PedNYC3 with:

```powershell
python -m src.visualization.avatar_position_rotation
```

Figures are written beneath the top-level `avatar graphs` directory. The
workflow normalizes inconsistent header whitespace and punctuation, detects
the CSV delimiter, preserves missing values as plot gaps, and uses
`ScenarioTime` without resampling. It writes CSV and JSON audit reports beside
the participant folders. Existing figure paths are not replaced unless
`--overwrite` is supplied; `--data-root`, `--output-root`, `--participants`,
and `--dpi` provide reusable overrides.

To add unsmoothed horizontal/3D speed and acceleration diagnostics while
leaving those six-panel figures untouched, run:

```powershell
python -m src.visualization.avatar_speed_acceleration
```

This creates separate ten-panel figures, per-sample inspection CSVs, and
`avatar graphs/avatar_speed_acceleration_summary.csv`. Invalid, duplicate,
missing, and negative `ScenarioTime` intervals remain in the sample tables but
do not receive derivatives. The default labels positions as raw units and does
not apply meter-based flags because the repository has no independent avatar
position calibration. If metric calibration is established, pass
`--position-units meters`; the configurable defaults are then 5 m/s and
20 m/s².

## Behavioral clustering analysis

After the validated trajectory and conflict-zone workflows have completed,
run the deterministic trial-level behavioral grouping analysis with:

```powershell
python -m src.analysis.behavior_clustering --seed 42 --ground-truth XCPedestrian-GroundTruth-RERUN.xlsx
```

The analysis writes only beneath `outputs/behavior_clustering`. It uses the
validated `B VR Pos X/Z` pedestrian coordinates and `A car Pos X/Z` vehicle
coordinates, aligns recordings to a supported conflict-zone event or minimum
synchronized actor distance, applies scenario-shared rigid trajectory
normalization, and evaluates Ward, Gaussian-mixture, and k-means solutions.
Ground-truth annotations are joined only after cluster assignments are fixed.
The report explicitly returns “no stable separation” when the pre-specified
bootstrap stability, silhouette, and minimum-cluster-size criteria are not met.
