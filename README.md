# PedNYC branch reference: actions, commands, and results

This is a self-contained handoff for a person or ChatGPT choosing what to run in this pedestrian–driver VR project. It covers the current working tree on **`analysis/scenario-trajectory-clustering`**, inspected **2026-10-03**, including local changes and artifacts rather than just committed files. Commands were checked against source; analyses, training, rendering, and tests were not rerun. Only this README was edited.

## Using this README with ChatGPT

Paste this README and describe your desired action, participant/scenario/recording, project location, and whether you want to view results or generate new ones. Ask for **prerequisites → exact PowerShell commands → output paths → checks**. For example: “Give me commands to export Scenario 7 pedestrian and vehicle speed HTML from existing trajectory/event tables.”

ChatGPT should use the actual entry points below, identify missing inputs and writes/overwrites, preserve recording identity, and distinguish saved results from capabilities requiring a new run. Replace placeholder paths. Do not invent flags or assume an absent workbook exists. The professor-facing scientific graph guide is preserved at the end.

## Navigation

- [Research and methods](#research-overview-and-interpretation)
- [Git snapshot](#git-snapshot-and-sharing-limits)
- [Setup](#setup)
- [Choose an action](#choose-an-action)
- [A. Canonical raw-recording pipeline](#a-canonical-raw-recording-pipeline)
- [B. Participant-folder batch processing](#b-participant-folder-batch-processing)
- [C. Individual stages and Gantt rendering](#c-individual-stages-and-gantt-rendering)
- [D. Scenario-pattern analysis](#d-scenario-pattern-analysis)
- [E. Trajectories, zones, and interactive speed](#e-trajectories-zones-and-interactive-speed)
- [F. Entry-distance review](#f-entry-distance-review)
- [G. Clustering and annotated order](#g-clustering-and-annotated-order)
- [H. Skeleton gesture models](#h-skeleton-gesture-models)
- [I. Avatar diagnostics](#i-avatar-diagnostics)
- [J. Presentation and video tools](#j-presentation-and-video-tools)
- [K. Reusable API and annotation templates](#k-reusable-api-and-annotation-templates)
- [Tests and troubleshooting](#tests-and-troubleshooting)
- [Branch map and historical tools](#branch-map-and-historical-tools)
- [Scientific graph guide](#scientific-graph-guide)

## Research overview and interpretation

The accessible analyses investigate whether movement patterns differ across VR conditions, how actor paths/proximity and estimated zone arrivals relate in time, whether reproducible interaction groups emerge, and whether observable pedestrian hand gestures can be predicted from skeleton motion and vehicle context. These are questions implemented by the analyses; no accessible document establishes a formal CHI poster plan or an agreed final hypothesis.

| Scenario | Configured condition |
|---|---|
| 3 | Orthogonal; pedestrian near vehicle lane, same side |
| 7 | Orthogonal; near lane, opposite side |
| 12 | Orthogonal; far from lane, opposite side |
| 15 | Parallel/contra-directional; near lane, opposite side |
| 16 | Parallel/contra-directional; far from lane, opposite side |
| 21 | Parallel/contra-directional; near lane, same side |

These descriptions come from `src/analysis/scenario_metadata.py`; they are condition labels, not verified reconstructions of every recording. Start with the [scientific graph guide](#scientific-graph-guide) for results and illustrations.

### Data, identities, time, and coordinates

Raw logger CSVs are semicolon-separated and contain base64/Unity vector and joint arrays; decoded CSVs expand those channels into columns. Derived tables generally use commas. The reusable loader normalizes aliases; pedestrian world position is **B VR Pos X/Z**, vehicle world position **A car Pos X/Z**. Y is vertical. Horizontal distance/speed use X/Z. Analyses label world coordinates as meters, but independent physical calibration is not supplied. Avatar positions are separate, uncalibrated channels. Yaw is degrees and yaw rate degrees/s.

A participant key such as `PedNYC1` identifies a study folder; a scenario is not a unique recording. Expanded analysis retains timestamped duplicates and adds source-stem suffixes to `recording_identifier`. It does not concatenate separate recordings. Thirty participant folders contribute to the expanded study (1?13 and 15?31; no 14). Successful **recording / unique-participant** counts for scenarios 3, 7, 12, 15, 16, 21 are **26/26, 23/23, 26/25, 28/26, 25/24, 27/25**. Eight available recordings failed processing; 23 participant/scenario cells are marked unavailable. Read statuses rather than treating every manifest row as usable.

Workbook matching uses normalized exact `StudyFolder` + `ScenarioName` keys, accepting case/leading-zero variations and supported numeric scenario strings. General clustering marks duplicate or unmatched motion/annotation keys rather than confirming identity. Scenario clustering deterministically retains the first sorted motion recording per key and records duplicates as exclusions; skeleton modeling excludes ambiguous duplicate motion trials. An exact folder/scenario join cannot prove which timestamped encounter an annotation describes. Missing/Unknown labels are not negative observations. The external workbook and cleaned matching exports are absent locally, so original annotation identity cannot be rechecked here.

The class-based workflow sorts finite timestamps, removes duplicate times, shifts the first usable time to zero, and uses positive time differences. Trajectory `scenario_time` is this elapsed time, not necessarily the original logger clock. Onset-aligned graphs subtract the first smoothed pedestrian speed ?0.30 m/s, falling back to elapsed zero if never reached. Progress plots resample path length; progress is dimensionless and does not preserve speed. Skeleton/clustering windows use a conflict-event anchor, not movement onset. Canonical standalone stages have separate time reconstruction and thresholds; do not combine their tables by row or assume identical clocks.

The two raw Scenario 3 files illustrate the identity issue: `13-48-59` has 648 full-width data rows ending at 31.002930 s; `16-58-19` has 566 ending at 26.343018 s. Whitespace/blank lines are not data rows. The local first raw file has uncommitted formatting changes. Recipes A/C select the shorter **16:58** recording; they do not reproduce the corrected **31.003 s** presentation. That presentation's decoded source has 648 rows and a matching saved/current SHA-256 `2FB6329BA8E2223D1367325CD32C1C9502EB98930224A3C877E63CE05F84E251`. This verifies the decoded source identity, not an original video match or every downstream artifact's freshness.

### Methods that affect interpretation

| Workflow | Construction, evaluation, and limits |
|---|---|
| Reusable API / pilot / expanded | Position samples exceeding 8 m/s step speed are masked and interpolated in both directions **for both actors**. Centered smoothing uses 9 samples for speed/distance and 11 for acceleration; sample windows are not fixed durations. Missing computed speed/acceleration can become zero, and absent head cues can become zero: absence is not confirmed stillness. Macro/micro labels are threshold heuristics, not validated intent classes. |
| Expanded feature comparison | Thirteen retained features: median/peak pedestrian speed, positive acceleration, deceleration magnitude, low-motion proportion, stop-start count, duration; head-event count and median absolute head rate; minimum actor distance, vehicle speed/deceleration around pedestrian onset, and `median_closing_speed`. Features >25% missing or zero variance are excluded; remaining missing values use feature-wise medians across recordings, then global mean/population-SD z scores. Distances are Euclidean in this feature space; scenario-level cosine comparison is secondary. Redundant duration/count features are deliberately excluded. |
| Important unit caveat | Reusable `closing_distance` is smoothed **negative distance change per sample**, without division by elapsed time. Thus expanded `median_closing_speed` is not m/s and can depend on sampling rate. Conflict-analysis closing speed separately uses a time derivative in m/s. Do not equate these fields. |
| Expanded inference | Compares recording-pair distance distributions; 2,000 within-participant scenario-label permutations, seed 42. The saved median contrast is 0.430 (4.390 minus 3.960), with pair-bootstrap interval 0.354?0.525 and standardized effect 0.215. Pairs share recordings; their counts are not independent sample sizes, and this bootstrap interval is not a participant-level effect interval. |
| General behavior clustering (G) | Up to 10 retained features from actor speed/deceleration/stationary behavior, signed arrival difference, distance, path shape, and response lag. Exclude >35%-missing/constant/highly correlated features (absolute correlation >0.85); median-impute and median/IQR scale within scenario. Compare Ward, full-covariance GMM, and k-means, k=2?5 subject to sample size. Accept only minimum size ?max(3, ceiling(10% of n)), bootstrap mean ARI ?0.60, 10th-percentile ARI ?0.30, silhouette ?0.15. Fallback assignments remain exploratory. No saved general-clustering output exists locally. |
| Scenario trajectory clustering (G) | Window ?3 to +5 s around first finite actor zone arrival; potential-zone arrival is counterfactual. No closest-approach anchor fallback. Translate/rotate paths by a scenario-shared zone frame; resample 50 arc-length points. Average-linkage clusters use pairwise path RMS distance; response clusters use Ward with within-scenario median/IQR scaling. k=2?4; accept minimum size ?3, mean ARI ?0.70, and leave-one-feature-out minimum ARI ?0.70 for feature models. Stability uses **80% subsampling without replacement**, despite saved `bootstrap_ari` names/graph labels. Silhouette ranks candidates but has no acceptance cutoff here. Pooled analysis excludes absolute coordinates. |
| Skeleton gesture experiment (H) | Observable hand-gesture Yes/No only. Twenty joints are pelvis-centered and rotated by hips yaw; positions, velocities, and missingness masks are used, not joint acceleration. Six-second conflict windows (?3 s) resample to 61 points. Compare majority, fixed hand-displacement rule (?0.25 in normalized position units), context logistic, handcrafted-skeleton logistic, NumPy skeleton LSTM, and context-fused LSTM. LSTM defaults: hidden size 12, 60 epochs, learning rate 0.008, L2 0.0001. Five stratified participant-grouped folds, seed 42; imputation/scaling fit on training folds; classify at 0.5. Saved 95% intervals use 500 participant bootstraps. |
| Skeleton exclusions | Saved dataset: 155 candidate rows, 138 included from 30 participants (25 Yes, 113 No); 11 ambiguous-duplicate and 6 incomplete-window exclusions. Other source checks can exclude Unknown labels, missing anchors/context, or inadequate pelvis coverage. This is exploratory held-out-participant evaluation, not a deployed detector or validated screening rule. |
| Zones and timing (E) | First contiguous sampled occupancy, not interpolated boundary crossing. Four observed-path scenarios use estimated 2 m circles. Current 12/16 potential rectangles are 4.5?5.0 / 4.5?4.5 m; their actual vehicle occupancy is deliberately undefined. Counterfactual arrival uses modeled remaining route and observed approach speed, not a recorded turn. `entry_time_difference_seconds` is an **absolute** gap with actor-order label separate; modeled `delta_TTZ_seconds` is pedestrian entry minus counterfactual arrival. Distance/closing TTC is a proximity heuristic, not collision prediction. Entry-review F uses interpolated circle crossing and can differ from E. |

Full expanded execution and `--postprocess-only` are **not equivalent**: postprocessing additionally excludes successful summaries whose smoothed peak pedestrian speed exceeds the configured 8 m/s threshold and writes final sample sizes/representative traces. The saved 155-recording tables are compatible with that postprocessing path; the JSON configuration does not identify which invocation produced them. Saved verification flags check limited structure, not scientific validity: full-run `no_silent_zero_imputation` is a nonnegative-count check and cannot prove its label's claim.

Class API behavior labels prioritize head checks, possible yielding/hesitation, low motion, acceleration/deceleration, and crossing commitment; confidence values are heuristic constants. Nominal micro windows are 1 s with 0.5 s steps, adaptively bounded 0.25?2.5 s; event merging uses a 0.15 s gap and 0.25 s minimum duration. These differ from canonical descriptive motion/head/car tags. YAML has 41 pipeline fields and controls API/pilot/expanded; trajectory/conflict use dataclass defaults, general clustering uses its own `AnalysisConfig`, and canonical stages use separate constants. Changing YAML alone will not update all workflows.

## Git snapshot and sharing limits

Inspected branch: **`analysis/scenario-trajectory-clustering`**, HEAD **`dc30e4cc635790461dfd865db0dfa50d0c6f00f5`** (`interactions and outputs`). Closest available comparison is parent **`ace7c52ad104a610151c810146efaa902cdc01be`**, also local `refactored` / `origin/refactored`; this branch is one commit ahead. Remote refs were not fetched. The broader `main` ancestor is `a6ec5b73cf6ea6fad5e973bddd5ad51422a4f498`.

Before this audit's documentation edits, Git reported **28 modified tracked paths, 33 deleted paths, and 158 untracked files**, with no staged changes. All 33 reported deletions are under inaccessible `outputs/entry_distance_manual_review`; the directory exists but access failed, so physical deletion is unconfirmed. Modified items include README, one raw CSV, five source modules, one test, and 20 outputs. The untracked files comprise 91 scenario-clustering artifacts, 61 presentation artifacts, and six Python files (scenario clustering, three presentation scripts, and their two new tests).

A **clone of HEAD** gets the committed 1,122-file snapshot, not these local modifications, untracked clustering/presentation code/results, or the **361 ignored decoded CSVs** present locally. `.gitignore` also excludes processed/class outputs, participant graph trees, package caches, and 4-frame participant folders; some files already tracked inside ignored trees are still included by Git. Copy the actual required data/results explicitly for a local-folder handoff. Inventory/manifests use absolute paths and need relocation; a README alone cannot supply those inputs.

Relative to `refactored`, the committed change adds general clustering, entry review, skeleton models, avatar diagnostics, related tests, and saved outputs, and modifies conflict zones/interactive speed (248 changed paths: 204 added, 44 modified). Canonical raw processing, reusable API, root wrappers, pilot/expanded comparison, and trajectory analysis are **inherited functionality**. Scenario-specific clustering and corrected presentations are local additions, not committed branch functionality. The local potential-rectangle revision and presentation-related segmentation changes differ from HEAD.

Most outputs lack code revision and upstream input hashes: their **freshness against the present working tree is unverified**, even when their values/configuration agree with inspected source. Current zone/speed validation describes only scenarios 12/16. The presentation's decoded source hash matches, but this does not certify unrelated outputs. No original study protocol, identity adjudication, calibrated coordinate transform, formal poster decision, or agreed next step can be reconstructed from the accessible files. See the audit for inspected evidence and inaccessible material.

## Setup

Run commands from the **project root in PowerShell**. `$py` below means your configured Python executable. On the inspected computer, Python 3.12 exists at the path below, but `python`, `py`, `ffmpeg`, and `ffprobe` were not found on PATH; `.venv` lacks `Scripts/python.exe`. Do not assume that environment is ready.

For a new environment, use an installed Python executable. The bootstrap path is verified here; replace it on another computer:

```powershell
$bootstrapPython = 'C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $bootstrapPython -m venv .venv
$py = (Resolve-Path '.venv\Scripts\python.exe').Path
& $py -m pip install -r requirements.txt
& $py -m pip install openpyxl
& $py -c "import numpy, pandas, matplotlib, scipy, sklearn, plotly, yaml, imageio_ffmpeg; print('Analysis imports OK')"
```

`openpyxl` is needed for workbook reads and **is missing from requirements.txt**. Requirements are unpinned; setup does not guarantee identical historical reproduction. Local `.python_packages`/`.analysis_packages` caches are not portable environments. Some scripts insert `.python_packages` into `sys.path`; check it if packages differ from your chosen environment. Scenario-clustering code uses `groupby.apply(..., include_groups=False)`, requiring recent pandas (2.2+).

MP4 tools need ffmpeg. Canonical rendering tries `--ffmpeg`, PATH, then the executable bundled by `imageio-ffmpeg`. The 4-frame compositor separately needs **ffmpeg and ffprobe on PATH**. Viewing existing PNG/HTML/MP4 files needs no Python setup.

### Current inputs and blockers

| Input | Availability and use |
|---|---|
| `data/raw/` | Two Scenario 3 raw CSVs, timestamps `13-48-59` and `16-58-19`. Use an explicit recording path. |
| `csv_pednyc1/csv/` | Raw participant-folder inputs for B, including other scenarios and Practice. |
| `data/processed/decoded_inventory.csv` | 361 rows; all 361 destination files existed at inspection. Requires `pednyc`, `scenario`, `destination_path`. The canonical runner does not generate this copied-decoded inventory. |
| Expanded `manifests/processing_success_manifest.csv` | 163 available-recording rows; all source paths existed. Saved report has 155 successes and 8 failures; do not count every row as successful. |
| Trajectory/conflict outputs | Saved tables and graphs for the six main scenarios. Subset runs rewrite validation summaries for that selection. |
| `XCPedestrian-GroundTruth-RERUN.xlsx` | Absent from project. Annotation runs require the real workbook/schema. A saved/default reference points outside the project to Downloads. |
| `outputs/behavior_clustering/` | Absent. Its cleaned labels and trial-matching CSV are prerequisites for H. |
| `outputs/entry_distance_manual_review/` | Exists but could not be enumerated here; saved contents are unverified. Use a fresh output path for a new run. |

Inventories/manifests contain **absolute source paths**. After moving/sharing the folder, update path columns in working copies to the actual decoded files and pass those copies where supported. An output override does not automatically redirect downstream inputs. Preserve timestamped duplicates and recording identifiers. Some generated files are ignored by Git, so a clone may lack local artifacts.

## Choose an action

| Action | Recipe | Starting point → result |
|---|---|---|
| Read findings/open graphs | Scientific graph guide | Existing files; no execution |
| Process a raw recording end-to-end | A | Logger CSV → canonical tables, plots, optional MP4/report |
| List/process participant-folder scenarios | B | `csv_pednycN/csv` → exact v6 segmentation |
| Decode only, rerun a stage, render existing segments | C | Explicit input/output CSVs |
| Compare scenarios across participants | D | Decoded inventory → features, statistics, figures, manifests |
| Inspect trajectories, proximity, entry/exit, actor speed | E | Expanded manifest → trajectories → zones → HTML |
| Flag unusual entry distances for manual review | F | Trajectories and supported zones |
| Cluster paths/responses or compare manual who-went-first | G | Motion/event tables, manifests, external workbook |
| Audit skeleton channels/train gesture models | H | G's default cleaned labels/matching plus decoded/motion inputs |
| Plot avatar position, rotation, derivatives | I | Decoded PedNYC1–3 files |
| Build corrected Scenario 3 presentation/video | J | Authoritative decoded CSV and zone JSON |
| Generate merged events/human annotation template | K | Class-based API; root v6 wrappers do not produce merged events |

## A. Canonical raw-recording pipeline

Preferred entry point: [run_scenario.py](src/scenario_pipeline/run_scenario.py). Choose an explicit recording and a fresh artifact root:

```powershell
& $py src/scenario_pipeline/run_scenario.py 'data/raw/CSV_Scenario-Ped-3_Session-temp_2024-02-22-16-58-19.csv' --participant 1 --output-root outputs/reproduction/scenario3_165819 --preview-only
```

Remove `--preview-only` for MP4; use `--skip-render` to skip renderer outputs, or `--preview-time 15` for one elapsed-time preview. `--ffmpeg 'C:/path/to/ffmpeg.exe'` overrides the encoder. Existing core CSVs are refused without `--force`; prefer a fresh output root.

Under the artifact root, writes:

- `data/processed/pednyc1/scenario3/decoded_clean_PedNYC1_scenario3.csv`, `features_PedNYC1_scenario3_metrics_v1.csv`, legacy v1 segments.
- `outputs/graphs/exact_v6/pednyc1/scenario3/{metrics_v1,macro_v2,micro/v6}/` and `PedNYC1_scenario3_report.md`.
- `4frame_view/pednyc1/scenario3/metrics_v1/` diagnostic copies.

Read the report for time sources, completeness, boundaries/tags, failures, and Scenario 3 regression. Do not assume five macro segments or a passing historical regression. Macro overlays can read existing conflict events from the **original project root**, even with relocated output; these are not newly estimated for the selected recording. Check identity/time compatibility.

Selections: positional raw CSV, `--raw-path`, `--scenario-id N` (unique `data/raw` match), or `--all`; `--continue-on-error` continues multi-input runs. **Current `--scenario-id 3` is ambiguous, and `--all` sends both Scenario 3 files to the same output paths.** Run explicit files to separate roots. Names must match `CSV_Scenario-Ped-N_Session-*.csv`; this runner does not accept Practice. `--participant` labels outputs, not participant verification.

## B. Participant-folder batch processing

These discover `csv_pednycN/csv/CSV_Scenario-Ped-N_Session-*.csv` or Practice and reject multiple matches for one participant/scenario.

```powershell
& $py run_batch.py -p 1 --list
& $py run_single.py -p 1 -s 7
& $py run_batch.py -p 1 -s 3 7 12 15 16 21
& $py run_batch.py -p 1 --pattern '1*'
& $py run_batch.py -p 1 --count 2
& $py run_batch.py -p 1 --all
& $py run_v6_batch.py -p 1 -s 3 7
& $py compare_v6_labels.py -p 1 -s 3 7
```

Processing commands are **alternatives**, not a sequence to run in full. Wrappers run decode → metrics → macro → micro in the canonical participant/scenario data and `exact_v6` folders, but do not add A's consolidated report/validation or MP4. They have no output-relocation/force guard and can overwrite outputs. `--no-plots` suppresses only **micro PNGs**; metrics/macro plots still run. Practice discovery is supported here.

Comparison requires the canonical micro segment CSVs and writes `outputs/graphs/exact_v6/pednyc1/v6_label_comparison_3_7.csv`: counts/durations/proportions for motion/head/car tags, not ground truth.

## C. Individual stages and Gantt rendering

Explicit paths avoid standalone scripts' older root-level Scenario 3 defaults. This operates in A's fresh artifact root:

```powershell
$runRoot = 'outputs/reproduction/scenario3_165819'
$stem = 'PedNYC1_scenario3'
$decoded = "$runRoot/data/processed/pednyc1/scenario3/decoded_clean_$stem.csv"
$features = "$runRoot/data/processed/pednyc1/scenario3/features_${stem}_metrics_v1.csv"
$graphRoot = "$runRoot/outputs/graphs/exact_v6/pednyc1/scenario3"
$macro = "$graphRoot/macro_v2/macro_segments_${stem}_v2.csv"
$micro = "$graphRoot/micro/v6/micro_segments_descriptive_${stem}_v6.csv"

& $py src/preprocess/decode_first.py --input-csv 'data/raw/CSV_Scenario-Ped-3_Session-temp_2024-02-22-16-58-19.csv' --output-csv $decoded
& $py src/features/create_metrics_and_graph.py --input-csv $decoded --output-csv $features --segment-csv "$runRoot/data/processed/pednyc1/scenario3/segments_${stem}_v1.csv" --graph-dir "$graphRoot/metrics_v1" --fourframe-dir "$runRoot/4frame_view/pednyc1/scenario3/metrics_v1"
& $py src/segmentation/macro_segmentation.py --input-csv $features --output-dir "$graphRoot/macro_v2" --segments-csv $macro --plot-path "$graphRoot/macro_v2/macro_segments_${stem}_v2.png"
& $py src/segmentation/micro_segmentation.py --feature-csv $features --macro-csv $macro --output-dir "$graphRoot/micro" --scenario-stem $stem
& $py src/segmentation/render_micro_gantt_mp4.py --segments-csv $micro --macro-csv $macro --output "$graphRoot/micro/v6/micro_gantt_${stem}_v6.mp4"
```

Run only needed stages after their prerequisites. Decoder expands Unity/base64 arrays from semicolon-separated raw CSVs without dropping rows; decoded output uses semicolons, features/segments commas. Metrics makes up to ten PNGs; micro makes descriptive segments, frame assignments, boundaries, macro summaries, tag definitions and three plot layouts under `v6`.

Preview without ffmpeg:

```powershell
& $py src/segmentation/render_micro_gantt_mp4.py --segments-csv $micro --macro-csv $macro --preview-time 15 --preview-output "$graphRoot/micro/v6/preview_15s.png"
```

Renderer options: `--fps` (30), `--width` (1280), `--height` (720), `--dpi` (100), `--crf` (18), `--preset` (medium), `--ffmpeg`, `--output`, `--preview-time`, `--preview-output`. Standalone stages overwrite targets without A's guard. For another scenario change **all paths and the stem**.

Macro supports `--conflict-events-csv` and `--recording-identifier` for occupancy overlays, and tries to infer existing project events from input identity. Overlays are optional. Micro accepts `--participant`/`--scenario` or explicit `--scenario-stem`; `--no-plot` skips PNGs. These standalone tuned stages have no `--config` option.

## D. Scenario-pattern analysis

Preferred study-level comparison: [scenario_pattern_expanded.py](src/scenario_pattern_expanded.py). Reads existing decoded files from the inventory, not raw folders. It analyzes the fixed six scenarios independently using reusable feature/segmentation classes and YAML configuration.

```powershell
& $py -m src.scenario_pattern_expanded --inventory data/processed/decoded_inventory.csv --config config_default.yaml --output-dir outputs/scenario_pattern_expanded_new
```

Default output is `outputs/scenario_pattern_expanded`; a nonempty directory requires `--overwrite`. Other flags: `--permutations 2000`, `--bootstraps 2000`, `--pilot-output outputs/scenario_pattern_pilot`, `--postprocess-only`. Postprocessing needs existing recording summaries, availability and aligned traces in the selected folder and rewrites summaries/statistics/figures; it is not a dry run.

Outputs: `manifests/` (availability/processing), `intermediate/` (per-recording metrics/segments/events), `tables/` (raw/standardized features, decisions, sample sizes, pair comparisons, sequences, similarity/clustering), `traces/`, `figures/`, `config/`, `logs/`, and `report/analysis_report.md` plus verification JSON. Read exclusions, feature inclusion and participant-aware permutation results before claims. Creating `_new` does not redirect downstream defaults from the original `outputs/scenario_pattern_expanded`.

Older **pilot**:

```powershell
& $py -m src.scenario_pattern_pilot --pednyc-sample-size 5 --scenario-min 3 --scenario-max 21 --seed 42 --inventory data/processed/decoded_inventory.csv --config config_default.yaml --output-dir outputs/scenario_pattern_pilot_new
```

Samples eligible participants, aggregates scenario features, and writes report/figures/verification; also supports `--overwrite`. Historical exploratory context; prefer expanded results for study-level interpretation.

## E. Trajectories, zones, and interactive speed

Dependency order: expanded manifest + decoded sources → raw coordinate audit → full trajectories → conflict zones → speed HTML. Existing artifacts can satisfy prerequisites; opening a graph does not require rerunning the pipeline.

```powershell
& $py -m src.analysis.trajectory_analysis --phase raw --scenarios 3 7 12 15 16 21
# Inspect outputs/trajectory_analysis/coordinate_frame_summary.csv and raw trajectory HTML before continuing.
& $py -m src.analysis.trajectory_analysis --phase full --scenarios 3 7 12 15 16 21
& $py -m src.analysis.conflict_zone_analysis --scenarios 3 7 12 15 16 21
& $py -m src.visualization.interactive_onset_speed --scenarios 3 7 12 15 16 21
```

These overwrite same-named default artifacts. Replace the scenario list for a subset; validation summaries then describe only that invocation.

| Tool | Inputs/options | Outputs/checks |
|---|---|---|
| `trajectory_analysis` | `--input` is a **processing manifest CSV**, default expanded manifest; `--output-dir` default `outputs/trajectory_analysis`; `--progress-points` 101; `--phase raw` default | Raw X/Z tables/joint HTML, recording/coordinate diagnostics, validation. Full phase checks frames then writes normalized-progress paths, means/covariance, deviations, metrics, intervals and standardized joint HTML. Full phase includes raw processing; it is not just rendering. |
| `conflict_zone_analysis` | `--input` is a **trajectory output directory**; `--output-dir` default `outputs/conflict_zone_analysis`; `--conflict-zone-buffer` 2.0; `--coverage-threshold` 0.5 | Per-scenario zone JSON, events, synchronized distances, spatial/distance/timing HTML; combined summary/validation. Check supported geometry, missing/never-entered events and modeled status. |
| `interactive_onset_speed` | Reads fixed project `outputs/trajectory_analysis` and `outputs/conflict_zone_analysis`; options `--scenarios`, `--coverage-threshold` 0.5, `--output-dir` | Pedestrian `scenario_NN_onset_aligned.html` in `outputs/interactive_speed_plots`; vehicle `scenario_NN_onset_aligned_vehicle_speed.html` in `outputs/interactive_vehicle_speed_plots`; validation JSONs. **`--input` is parsed but ignored by current main; `--output-dir` moves only pedestrian output. No vehicle-output/input override exists.** |

Alternate expanded manifest: pass it to trajectory `--input`. Alternate trajectory outputs: pass their directory to conflict `--input`. Clustering `--input-root` expects a complete project-style `outputs/...` hierarchy, not an arbitrary scenario directory. Interactive-speed CLI cannot currently consume relocated trajectory/event folders without a code change; do not invent flags.

Pedestrian position is `B VR Pos X/Z`, vehicle `A car Pos X/Z`; avatar is not the global pedestrian path. Normalized progress is path resampling, not physical speed. Entry/exit is estimated zone geometry, not a surveyed crosswalk boundary. Scenarios 12/16 model alternative maneuvers: modeled arrival is not actual vehicle occupancy.

## F. Entry-distance review

```powershell
& $py -m src.analysis.entry_distance_manual_review --scenarios 3 7 15 21 --top-n-review 3 --output-dir outputs/entry_distance_manual_review_new
```

Reads default trajectory/conflict folders; overrides `--trajectory-input`, `--conflict-zone-input`. Needs recording diagnostics/raw samples and supported zone JSON. Default scenario list is all six, but the implementation assumes `radius_m` and **supports circular zones only**. Current Scenario 12/16 rectangles lack that field: measurements become `invalid_data`, and circle-based rendering is incompatible. Use the four scenarios in the example; supporting rectangles requires a code change. It linearly interpolates each actor's first circle crossing, measures synchronized separation at the earlier valid entry, compares it with the scenario arithmetic mean, and flags the largest absolute deviations (three by default).

Writes `manual_review_index.html`, ranking/candidate/summary CSVs, metadata, and scenario distribution/ranking/trajectory/recording HTML. Flags are **inspection priorities**, not interaction classifications. `--overwrite` replaces the selected output directory through staging; use a fresh folder to preserve previous reviews.

## G. Clustering and annotated order

Two programs require expanded manifests, trajectory raw samples, zone JSON/events and a real annotation workbook. First sheet must contain `StudyFolder`, `ScenarioName`, and categorical fields defined in `LABEL_COLUMNS` in [behavior_clustering.py](src/analysis/behavior_clustering.py). Reads need openpyxl. Inspect duplicate/unmatched study/scenario keys.

Required workbook columns (exact spelling/punctuation matters):

| Workbook header | Meaning |
|---|---|
| `StudyFolder` | Participant/study key |
| `ScenarioName` | Scenario key |
| `Did the driver signal` | `driver_signal` |
| `How does vehicle speed change?` | `annotated_vehicle_speed_change` |
| `How does pedestrian speed change?` | `annotated_pedestrian_speed_change` |
| `Who has the right of way?（Who reached the intersection first?）` | `annotated_right_of_way` |
| `Who went first？` | `annotated_who_went_first` |
| `Did the pedestrian use hand gestures?` | `pedestrian_hand_gesture` |
| `Did the driver use hand gestures?` | `driver_hand_gesture` |

The hand-gesture target uses Yes/No; missing or unrecognized values must not be silently made negative.

```powershell
# Replace the placeholder; stop if Test-Path is false.
$groundTruth = 'C:/path/to/XCPedestrian-GroundTruth-RERUN.xlsx'
if (-not (Test-Path -LiteralPath $groundTruth -PathType Leaf)) { throw 'Supply the real annotation workbook before running.' }

# General behavior analysis; default output is absent and supplies H's prerequisites.
& $py -m src.analysis.behavior_clustering --input-root . --ground-truth $groundTruth --output-root outputs/behavior_clustering --seed 42 --bootstrap-repetitions 50

# Separate scenario-specific path/response analysis; preserve saved results.
& $py -m src.analysis.scenario_trajectory_clustering --input-root . --ground-truth $groundTruth --output-root outputs/scenario_trajectory_clustering_new --seed 42 --bootstrap-repetitions 50
```

**General behavior clustering:** Ward/GMM/k-means comparisons, within-scenario and adjusted pooled summaries, robustness/sensitivity, post-clustering annotation associations, measured-vs-annotated order. Supports `--scenarios 3 7 ...`. Outputs include `data_qc/ground_truth_cleaned.csv`, `trial_matching_report.csv`, QC/inventory; `features/`; clustering/validation/ground-truth directories; figures/HTML; config; `reports/analysis_report.md` and `professor_summary.md`. The report may return no stable separation; displayed candidates are not necessarily accepted.

**Scenario trajectory clustering:** fixed six scenarios (**no `--scenarios` flag**), −3 to +5 s window, 50 arc-length path points, path-shape/response groups, pooled comparison, manual Pedestrian-first/Vehicle-first overlays and speed plots. Retains one deterministic recording per study/scenario key and logs duplicates. Writes `summary_results.csv`, assignments/validation/features/prototypes, `within_scenario/`, `cross_scenario/`, `who_went_first_groups/`, `interactive_html/`, `static_figures/`, methodology and run manifest. Default workbook path is the original user's external Downloads folder; explicitly supply it when sharing.

Both support `--overwrite`, which **recursively deletes the selected output folder before regeneration**. Never point it at the project root/shared parent; prefer a fresh folder. Manual order, zone order, right-of-way and yielding are distinct constructs.

## H. Skeleton gesture models

Needs G's general analysis exports at **default** `outputs/behavior_clustering/data_qc/ground_truth_cleaned.csv` and `trial_matching_report.csv`, expanded manifest/decoded sources, and all six scenario trajectory/events at defaults. These cleaned labels/matching files are currently absent; even audit-only is blocked until restored/generated.

```powershell
& $py -m src.analysis.skeleton_behavior_lstm --audit-only --output-root outputs/skeleton_behavior_audit_new
& $py -m src.analysis.skeleton_behavior_lstm --output-root outputs/skeleton_behavior_lstm_new --seed 42 --folds 5 --epochs 60 --bootstrap-repetitions 500
```

Audit-only is optional: full run audits by default. `--skip-audit` reuses audit tables in the **selected output root** but still loads inputs. No CLI overrides exist for labels, matching, manifest, scenarios, trajectories or events; G's nondefault output will not be found automatically.

Predicts only observable hand-gesture Yes/No; Unknown is excluded. Outputs: `data_audit/`, `dataset/` (windows NPZ/manifest), `models/` (fold NPZs), `training/`, `metrics/` (out-of-fold predictions, folds, confusion matrices, intervals), `figures/`, `examples/`, `reports/`, configuration and generated README. The current LSTM uses NumPy; TensorFlow/PyTorch are not needed. No standalone new-recording inference CLI is supplied.

**No overwrite guard:** same-named artifacts are replaced. Audit-only also writes reports/configuration; use fresh output roots. Check participant-grouped splits, class counts, confidence intervals and fold stability before claims.

## I. Avatar diagnostics

```powershell
& $py -m src.visualization.avatar_position_rotation --participants PedNYC1 PedNYC2 PedNYC3
& $py -m src.visualization.avatar_speed_acceleration --participants PedNYC1 PedNYC2 PedNYC3
```

Searches decoded files under `data/processed/pednyc1`, `pednyc2`, `pednyc3`, including duplicates/Practice. Participant choices are those three names only. Overrides: `--data-root`, `--output-root`, `--dpi`, `--overwrite`; default output **`avatar graphs`** (quote paths containing spaces).

Position/rotation: six raw channels against ScenarioTime, PNGs and file audit CSV/JSON. Kinematics: separate ten-panel PNGs, `sample-level/` tables, summary CSV/JSON. Invalid/duplicate/negative time intervals do not receive derivatives. Default units are **raw**. Use `--position-units meters` only after independent calibration; meter-mode diagnostic thresholds default to `--speed-threshold 5`, `--acceleration-threshold 20`. Flags are diagnostics, not behaviors. Existing figures are skipped unless `--overwrite`.

## J. Presentation and video tools

[create_pednyc1_scenario3_presentation.py](scripts/create_pednyc1_scenario3_presentation.py) is a fixed Scenario 3 exporter, not a parameterized CLI:

```powershell
& $py scripts/create_pednyc1_scenario3_presentation.py
& $py scripts/verify_pednyc1_scenario3_corrected.py
```

Creation reads `data/processed/pednyc1/scenario3/decoded_clean_PedNYC1_scenario3.csv` and `outputs/conflict_zone_analysis/scenario_03/conflict_zone_definition.json`. Regenerates metrics/macro/micro, computes synchronized occupancy, and writes four figures, encounter/timeline MP4, definitions/methods, hashes and metadata to `outputs/presentation/pednyc1_scenario3_audit_corrected`. **That folder exists, so creation currently refuses to run.** No output/overwrite flag exists; a variant needs code/path adaptation. Do not delete saved results automatically.

Corrected verification checks source/derived coordinates/time and video and **writes/overwrites three QA PNGs**, despite describing itself as read-only. `verify_pednyc1_scenario3_presentation.py` targets older `pednyc1_scenario3_workflow` and also writes frames. Neither establishes an original screen-capture match.

Other local summary/video tools have **no CLI path overrides**:

| Command | Fixed input → output; conditions |
|---|---|
| `& $py src/segmentation/summarize_macro_segments.py` | Root-level `data/processed/features_PedNYC1_scenario3_metrics_v1.csv` + `outputs/graphs/macro_segmentation/macro_segments_PedNYC1_scenario3_v2.csv` → multisignal summary and `features_with_macro_segments.csv`; separate legacy inference thresholds. |
| `& $py src/visualization/animate_macro_segmentation.py` | Previous frame-level macro CSV → `4frame_view/macro_segmentation/ped_macro_segmentation_pointer.mp4` and static PNG; needs ffmpeg. |
| `& $py src/visualization/animate_metrics_with_pointer.py` | Root-level Scenario 3 features → individual pointer animations in `outputs/animations/scenario3_metrics_v1` and 4-frame copies; ffmpeg with GIF fallback. |
| `& $py 4frame_view/composite4_frame.py` | Fixed base 4-frame MP4 + legacy micro Gantt MP4 → `4frame_view/PedNYC1_scenario3_4frame_micro_gantt_v6.mp4`; ffmpeg/ffprobe on PATH. Verify source recording/synchronization. |
| `& $py src/segmentation/kink_segmentation.py` | Root-level features → kink point/segment CSVs and PNG in `outputs/graphs/scenario3_metrics_v1/kink_segmentation`, plus 4-frame PNG. Historical exploratory segmentation. |

These hard-code `C:/Users/patl5/OneDrive/Desktop/BURE/pednyc_analysis` or video paths and may overwrite targets; on another machine they need adaptation. Root-level sources can differ from nested canonical sources; verify identity/time, not just filenames. Prefer A/C for portable canonical rendering.

## K. Reusable API and annotation templates

[src/pipeline.py](src/pipeline.py) implements the **separate class workflow**: DatasetConfig, PipelineConfig, MetricsEngine, class-based macro/micro, BehaviorClassifier and EventMerger. No `python -m src.pipeline` CLI exists. Example for an available recording:

```powershell
& $py -c "from pathlib import Path; from src.config import DatasetConfig, PipelineConfig; from src.pipeline import ScenarioPipeline; pipeline = ScenarioPipeline(DatasetConfig(Path.cwd()), PipelineConfig.from_yaml(Path('config_default.yaml'))); result = pipeline.process(1, '7', save_plots=True); print(result['output_paths'])"
& $py validate.py -p 1 -s 7
```

API writes `outputs/processed/pednyc1/scenario7/{frame_metrics,macro_segments,micro_windows,merged_events}.csv` and `outputs/graphs/pednyc1/scenario7` plots, replacing same-named files. `validate.py` reads **this** `merged_events.csv`, writes `validation_template.csv` with prediction plus empty human label/correct/notes. It creates a template, not a validation score. Root `run_single.py` does not supply this input.

`config_default.yaml` controls this API and D's pilot/expanded analysis, **not** all standalone canonical scripts, whose constants are separate. Do not assume identical time reconstruction/thresholds between workflows.

## Tests and troubleshooting

Tests use unittest and pytest; full-suite execution can generate temporary test data/rendering. These are full or focused alternatives, and were not run for this documentation edit:

```powershell
& $py -m pytest tests -q
& $py -m pytest tests/test_scenario_pipeline.py tests/test_micro_presentation_gantt.py -q
& $py -m pytest tests/test_trajectory_conflict_analysis.py tests/test_interactive_onset_speed.py tests/test_entry_distance_manual_review.py -q
& $py -m pytest tests/test_scenario_trajectory_clustering.py tests/test_behavior_clustering.py tests/test_skeleton_behavior_lstm.py -q
& $py -m pytest tests/test_avatar_position_rotation.py tests/test_avatar_speed_acceleration.py -q
```

CLI-equipped entry points support `--help`; fixed-path diagnostics/animations may execute at import and have no help mode. Do not import every module to discover commands.

| Symptom | Check/action |
|---|---|
| Python not recognized / missing dependency | Set `$py`, install into that interpreter, verify imports; caches may not be active. |
| Multiple Scenario 3 matches | Choose timestamped path and separate output root. Do not guess recording identity. |
| Existing output | Fresh directory or tool-specific authorized overwrite; `--force` and `--overwrite` are not interchangeable. |
| Missing merged events | Use K's API, not v6 wrappers. |
| Missing workbook/cleaned labels | Supply real workbook to G, inspect matches, generate default data_qc before H; Unknown is not No. |
| No decoded files after moving project | Repair inventory `destination_path`/manifest `source_csv_path` in working copies; verify files. |
| New output ignored downstream | Supply supported input overrides; interactive speed and skeleton keep fixed inputs. |
| Bad timing/giant derivatives | Check ScenarioTime, positive actual deltas, delimiter/coordinates; do not substitute FrameRate or assume avatar metric calibration. |
| Missing entry / rejected cluster / chance-level model | Inspect exclusions, statuses, acceptance and uncertainty; these are not necessarily software failures. |
| No MP4 | Check encoder or use preview; compositor needs ffprobe too. |

## Branch map and historical tools

| Path | Responsibility |
|---|---|
| `src/scenario_pipeline/` | Current raw CLI, identity/path map (`scenario.py`), orchestration/validation/reports (`runner.py`). |
| `src/preprocess/loader.py`, `cleaner.py` | Reusable loading, validation/required columns and cleaning; imported, not CLIs. |
| `src/features/{column_map,metrics_engine,behavior,merger}.py` | Aliases, reusable metrics, threshold labels, event merging; APIs. |
| `src/segmentation/macro.py`, `micro.py` | Reusable macro segments/micro windows, separate from standalone canonical scripts. |
| `src/visualization/plots.py` | Plot functions called by API/analysis; no CLI. |
| `src/analysis/scenario_metadata.py` | Shared scenario descriptions. |
| `scripts/` | Fixed presentation generation/verification (J). |
| `data/raw`, `csv_pednyc1/csv` | Distinct raw discovery routes (A vs B). |
| `data/processed` | Decoded/features/inventories/participant folders; large files may be ignored by Git. |
| `outputs/graphs/exact_v6`, `outputs/scenario_pattern_*`, analysis/clustering/model folders | Separate workflows' outputs; follow recipe paths. |
| `avatar graphs`, `4frame_view` | Diagnostics and duplicated/composited presentation artifacts. |
| `docs/CODEX_STATE.md` | Historical architecture/regression; availability claims can predate current files. |
| `.python_packages`, `.analysis_packages`, `.venv`, `__pycache__`, `.pytest_cache` | Local packages/environments/caches, not scientific inputs or portable setup. |
| `backup-my-empty-branch-20260717152145` | Backup tree, not active code to blend into current workflows. |
| `setup_project.py` | `& $py setup_project.py` creates scaffolding, touches README/requirements/.gitignore only if absent; does not install or ingest data. |

Additional diagnostics and archived scripts:

- `& $py src/preprocess/check_columns.py`: scans `$env:PEDNYC_DATA_DIR` (default external `/export/CAR_WORK/Goedicke_PedestrianXC/Studies`), writes `data/interim/csv_dimensions_report.csv`. Participant extraction expects a `Studies` path component; arbitrary roots can yield error rows. It is not the decoded-inventory generator.
- `src/preprocess/decode.py`, `data_clean.py`, `clean_validation.py`: fixed root-level Scenario 3 inspections despite their names (avatar-speed/delimiter/spike checks or vehicle-velocity inspection). Invoke `& $py <script-path>` only after verifying assumed input/delimiter. Actual Unity decoding is `decode_first.py`.
- `oldscripts/basic_plot.py`, `basic_interaction.py`: early Scenario 3 plots. `oldscripts/{decode_first,decode,data_clean,clean_validation,check_columns}.py`: historical variants. `oldscripts/convolution/{features,animate_graph_pointer}.py`: early features/animation. `oldscripts/feature_outputs/{1d_convo,pedestrian_behavior,plot_new}.py`: autoencoder/behavior/plot exploration. `oldscripts/4frame_view/{feature_slection,infer_ped,overlay_ped_behavior}.py`: early head/arms, v3 rules and overlay. Many reference external historical `psl64_analysis`; the autoencoder needs TensorFlow outside current requirements. These are not the recommended current routes.
- Old v3 rules have a real CLI: `& $py oldscripts/4frame_view/infer_ped.py --input 'C:/path/to/compatible_head_arms_features.csv' --output-dir outputs/legacy_v3_new`. Input must match old head/arms feature schema. Other archived scripts need path/dependency adaptation; merely running their filename is not a verified portable recipe.
- `oldscripts/batch_replay.py` controls keyboard/mouse in the external StrangeLand game to load `.replay` files. Needs game/replay paths, running game, config changes, `pyautogui`/`pygetwindow`. It is not offline CSV analysis; no current one-command replay-to-inventory export exists.

**Scientific boundaries:** preserve measured/inferred tags, do not infer yielding from speed alone, turn modeled arrival into observed occupancy, relabel who-went-first as intent, or use avatar as verified global position. A valid command does not establish a scientific conclusion.

---

## Scientific graph guide

This project studies pedestrian–driver interactions in VR across scenarios **3, 7, 12, 15, 16, and 21**. The available graphs let us compare movement patterns, inspect synchronized paths and proximity, and examine the timing of motion, head/body cues, and driver signaling. They describe recorded behavior; they do not independently establish intent, yielding, or causality.

## Start here

1. [Pedestrian speed across all six scenarios](outputs/scenario_pattern_expanded/figures/aligned_speed_small_multiples.png) — the clearest overview of typical motion and variation between recordings.
2. [Verified pedestrian–driver trajectories](outputs/presentation/pednyc1_scenario3_audit_corrected/verified_pedestrian_driver_trajectories.png) — understand the geometry of one synchronized Scenario 3 encounter.
3. [Driver–pedestrian interaction timeline](outputs/presentation/pednyc1_scenario3_audit_corrected/driver_pedestrian_interaction_timeline.png) — see how measured motion, signaling, and zone occupancy relate in that encounter.
4. [Scenario 3 pedestrian–vehicle distance](outputs/conflict_zone_analysis/scenario_03/car_ped_distance.html) — interactively inspect proximity across recordings, with zone-entry markers.
5. [Hand-gesture model comparison](outputs/skeleton_behavior_lstm/figures/model_comparison.png) — assess the limits of the attempted automated gesture analysis.

![Onset-aligned pedestrian speed across six scenarios](outputs/scenario_pattern_expanded/figures/aligned_speed_small_multiples.png)

## Graph guide

PNG links open static images; HTML links open interactive graphs as described below. Scenario 3 examples illustrate a recording or scenario, not a population-wide conclusion.

| Graph | Scenario / scope | File | What it shows | Main takeaway or limitation |
|---|---|---|---|---|
| Onset-aligned pedestrian speed | All six; 155 successful recordings | [PNG](outputs/scenario_pattern_expanded/figures/aligned_speed_small_multiples.png) | Individual speed traces, median, interquartile range, and contributor count | Scenario profiles differ descriptively, but there is substantial variation; late summaries have fewer contributors. |
| Verified pedestrian–driver trajectories | PedNYC1, scenario 3; corrected presentation | [PNG](outputs/presentation/pednyc1_scenario3_audit_corrected/verified_pedestrian_driver_trajectories.png) | Both actors' world X/Z paths, direction/time annotations, and the existing conflict circle | Paths cross spatially; the synchronized timeline shows separate occupancy times. A path intersection alone is not a collision. |
| Driver–pedestrian interaction timeline | Same corrected recording | [PNG](outputs/presentation/pednyc1_scenario3_audit_corrected/driver_pedestrian_interaction_timeline.png) | Motion/head states, right signal, proximity, occupancy, and possible interpretations | Pedestrian occupancy precedes driver occupancy. Orange “possible yielding” intervals are unvalidated rules. |
| Pedestrian motion and interaction features | Same corrected recording | [PNG](outputs/presentation/pednyc1_scenario3_audit_corrected/pedestrian_motion_and_interaction_features.png) | Speed, acceleration, body yaw, head-turn rate, head/body yaw difference, distance, occupancy, and hand/shoulder height differences | Supports inspection of coincident cues; head movement is not confirmed attention or communication. |
| Driver motion and signaling features | Same corrected recording | [PNG](outputs/presentation/pednyc1_scenario3_audit_corrected/driver_motion_and_signaling_features.png) | Driver motion, yaw, indicators, and interaction context | Signaling is directly decoded; braking alone does not establish yielding. |
| Pedestrian–vehicle distance | Scenario 3, multiple recordings | [HTML](outputs/conflict_zone_analysis/scenario_03/car_ped_distance.html) | Synchronized horizontal separation over time, summary band, and entry/minimum-distance markers | Quantifies proximity; distance is between logged actor positions, not clearance between body/vehicle edges. |
| Conflict-zone entry, exit, and occupancy | Scenario 3, multiple recordings | [HTML](outputs/conflict_zone_analysis/scenario_03/conflict_zone_timing.html) | Pedestrian and vehicle occupancy intervals | Shows measured order for the estimated zone, not an independently annotated yielding decision or surveyed crosswalk boundary. |
| Onset-aligned pedestrian / vehicle speed | Scenario 3, multiple recordings | [Pedestrian HTML](outputs/interactive_speed_plots/scenario_03_onset_aligned.html), [vehicle HTML](outputs/interactive_vehicle_speed_plots/scenario_03_onset_aligned_vehicle_speed.html) | Actor speed with the same pedestrian-onset time origin and zone-event markers | Compare both actors' timing while preserving individual recordings; slowing has multiple possible explanations. |
| Potential conflict-zone geometry | Scenarios 12 and 16 | [12 HTML](outputs/conflict_zone_analysis/scenario_12/conflict_zone_spatial.html), [16 HTML](outputs/conflict_zone_analysis/scenario_16/conflict_zone_spatial.html) | Observed paths plus a configured alternative vehicle corridor and fixed rectangular zone | These are potential interaction zones. Counterfactual arrival is modeled, not actual vehicle occupancy. |
| Scenario 3 clustering diagnostics | Scenario 3; saved exploratory analysis | [PNG](outputs/scenario_trajectory_clustering/static_figures/scenario_03_diagnostics.png) | Cluster separation, stability, size, feature profiles, and annotation composition | Path-shape clustering passes the saved criteria; conflict-response clustering fails feature-removal robustness. Cluster labels are not behavioral diagnoses. |
| Observable hand-gesture prediction | Six scenarios; 138 included trials, 30 participants | [PNG](outputs/skeleton_behavior_lstm/figures/model_comparison.png) | Held-out participant performance and uncertainty for rules, logistic models, and LSTMs | Learned models do not establish above-chance generalization; the hand rule is an exploratory proxy. |

## How to read the graphs

- **Overview and interactive speed:** horizontal axis is seconds relative to each recording's pedestrian movement onset (first speed ≥0.30 m/s; scenario start is the fallback). Vertical axis is horizontal speed in m/s. Speed uses X/Z displacement and a centered nine-sample mean. Pale blue traces are recordings, red is the pointwise median, orange is the 25th–75th percentile band, and dashed gray is contributor count on the right axis. The band describes variability, not a confidence interval. Interactive summaries use a 0.1 s interpolation grid and are hidden below 50% of the maximum contributor count; gray shading marks low coverage. Static panels show −5 to +30 s; HTML retains full available durations. Entry/exit markers identify the estimated zone: hover for event type and timestamp.
- **Corrected Scenario 3 presentation:** time is elapsed scenario seconds, 0–31.003 s. Spatial axes are world X/Z in meters; pedestrian is blue and driver red. Speed is m/s, acceleration m/s², yaw/difference degrees, and head-turn rate degrees/s. Acceleration differentiates smoothed speed and uses an 11-sample mean; planar distance uses a nine-sample mean. Blue/red shading marks actor occupancy. The circle has radius 2 m; “nearby” means distance <12 m, not a validated safety boundary. Timeline background bands identify macro movement phases; green rows show head activity, purple/magenta rows driver slowing/signaling, yellow proximity/occupancy, and orange hypotheses. Candidate hand raises require raw hand height > shoulder height with no minimum duration. [Exact definitions](outputs/presentation/pednyc1_scenario3_audit_corrected/definitions_table.csv) and [methods](outputs/presentation/pednyc1_scenario3_audit_corrected/methods_summary.md) explain the rules.
- **Distance and occupancy HTML:** distance is synchronized Euclidean X/Z separation in meters. Timing bars use elapsed scenario time in seconds (the reusable loader shifts the first usable timestamp to zero); blue is pedestrian and red vehicle. Zone entry/exit refers to the first contiguous visit. Scenarios 3, 7, 15, and 21 use estimated 2 m circles. Scenarios 12 and 16 instead use fixed potential rectangles, respectively 4.5 × 5.0 m and 4.5 × 4.5 m, with a dashed purple counterfactual vehicle path. Missing entry means no detected entry, not time zero.
- **Clustering diagnostics:** k is the number of candidate groups; silhouette measures separation and adjusted Rand index (ARI) measures agreement between cluster assignments. The dashed stability reference is 0.70. Feature-profile colors show standardized values (red higher, blue lower), not physical units. The acceptance panel explicitly distinguishes passing path groups from failing response groups.
- **Model comparison:** horizontal axis is balanced accuracy (mean sensitivity across the two classes); dashed 0.5 is the majority/chance reference. Blue bars are out-of-fold estimates; black intervals are 95% participant-bootstrap confidence intervals. Participants are separated between training and testing within each fold.

## What the analyses support—and what remains uncertain

**Scenario-related movement structure:** the [expanded report](outputs/scenario_pattern_expanded/report/analysis_report.md) documents 155 successful recordings from 30 participants and eight processing failures. Scenario counts are 26, 23, 26, 28, 25, and 27 in the order 3, 7, 12, 15, 16, 21. Same-scenario/different-participant recordings have smaller median standardized feature distance (3.960) than different-scenario/different-participant recordings (4.390); participant-aware permutation p=0.0005. These are feature distances, not meters. Same-participant/different-scenario distance is smaller still (3.529), so scenario does not demonstrably dominate participant identity.

**A source-matched encounter:** in the corrected Scenario 3 example, pedestrian occupancy is 15.366–18.316 s and driver occupancy 23.033–23.846 s. This supports pedestrian-first occupancy of that estimated zone. It does not establish why either actor changed speed. No matching original screen capture was established; the accompanying animation reconstructs the logged VR paths.

**Potential CHI poster directions:** no explicit CHI poster plan or named feasibility study was found in accessible documentation. The existing analyses test the following capabilities; describing them as poster directions is a synthesis, not evidence of an agreed plan:

- **Scenario-specific movement patterns:** supported by the expanded feature-distance comparison; any claim about psychological state or causality remains untested.
- **Automatic interaction groups:** the [saved clustering results](outputs/scenario_trajectory_clustering/summary_results.csv) accept path-shape solutions only for scenarios 3 (k=2, sizes 17/9) and 12 (k=3, sizes 3/18/4). All selected conflict-response solutions fail the stated acceptance criteria. The [pooled solution](outputs/scenario_trajectory_clustering/cross_scenario/result.json) also fails minimum size (147 versus 2 trials). This is evidence that these tested configurations did not produce acceptable groups, not proof that behavioral grouping is impossible.
- **Skeleton-based gesture detection and added vehicle context:** the [saved interpretation](outputs/skeleton_behavior_lstm/reports/final_interpretation.md) reports skeleton LSTM balanced accuracy 0.485 (95% CI 0.421–0.569), context-only 0.576, and fusion 0.463. Learned-model intervals include chance and folds vary substantially. Generalization and added context value are **inconclusive**. The tested models did not establish useful performance; these results do not show that skeleton information is inherently unusable. The new hand-motion rule reaches 0.610 (95% CI 0.539?0.689), but macro F1 is only 0.450 and positive precision is 0.231 despite recall 0.840; it produces many false positives. Only 25 positive and 113 negative trials were usable.

Annotation-based findings have a provenance limit: the clustering manifest references an external `Downloads/XCPedestrian-GroundTruth-RERUN.xlsx`, absent from this project; the skeleton audit references a cleaned export under `outputs/behavior_clustering`, which is also absent here. Saved labels and predictions exist, but their source matches cannot be independently confirmed from this folder. Manual “who went first” is not automatically equivalent to zone entry order or yielding. Foot-based gait interpretation remains unvalidated; it is not an established result or an executable gait-analysis workflow.

## Which versions to use

Use the expanded scenario-pattern results rather than the 29-recording pilot, and `outputs/presentation/pednyc1_scenario3_audit_corrected` for the single-recording presentation rather than `pednyc1_scenario3_workflow`. `oldscripts`, duplicated `4frame_view` figures, and older v1/kink graphs are historical or diagnostic material.

The current Scenario 12/16 spatial and speed HTML files incorporate the fixed potential rectangles. Their validation summaries cover only those two scenarios, not a fresh full six-scenario run. The saved clustering manifest records `turn_uncertainty` but does not establish regeneration against the current rectangles; treat downstream clustering/gesture outputs as separate saved analyses, not validation of the latest zone revision. Avatar position/rotation plots also exist, but avatar position is uncalibrated and is not the verified global pedestrian trajectory. The manual-review output directory could not be accessed and is not included in this guide.

## Opening interactive graphs locally

Download or copy the project folder with its `outputs` subfolders intact. Open a linked `.html` file in a modern browser (double-click it or use **Open with**). Markdown viewers may show HTML source or a download link; open the downloaded file in the browser.

The selected HTML files contain their data and Plotly JavaScript, with no external script links. **No Python packages, internet connection, or local server command are required to view them.** Hover for values and recording IDs, zoom with the toolbar, and use legend entries to toggle traces where available. Share the HTML files themselves alongside the README; a screenshot loses the interactive detail.
