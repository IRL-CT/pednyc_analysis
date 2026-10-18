# PedNYC1 Scenario 3 presentation workflow

## Verified input

- Exact raw input: `data\raw\CSV_Scenario-Ped-3_Session-temp_2024-02-22-16-58-19.csv`
- Exact processed input: `outputs\presentation\pednyc1_scenario3_workflow\decoded_clean_PedNYC1_scenario3_canonical_copy.csv` (566 rows)
- Common time axis: ScenarioTime minus first sample, 0.000000–26.343018 s.
- Macro output: `outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.csv` (5 unchanged segments).
- Micro output: `outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_segments_descriptive_PedNYC1_scenario3_v6.csv` (24 unchanged segments).
- Conflict-zone definition: `outputs\conflict_zone_analysis\scenario_03\conflict_zone_definition.json`; circle centered at (11.846832, 58.392967) m, radius 2.0 m, robust median of recording-level observed path intersections. Occupancy is recomputed from this unchanged definition and the canonical synchronized rows because the existing PedNYC1 event row does not match this recording.

## Input columns

- **Time:** ScenarioTime
- **Pedestrian position:** B VR Pos X/Y/Z
- **Pedestrian orientation:** B VR Rot X/Y/Z
- **Head orientation:** B Bone Rot Head X/Y/Z
- **Hands:** B Bone Pos LeftHand X/Y/Z; B Bone Pos RightHand X/Y/Z
- **Shoulders:** B Bone Pos LeftShoulder X/Y/Z; B Bone Pos RightShoulder X/Y/Z
- **Driver position:** A car Pos X/Y/Z
- **Driver orientation:** A car Rot X/Y/Z
- **Driver speed:** derived planar driver speed (A velocity is also recorded)
- **Driver acceleration:** derived planar acceleration; A accel is recorded control input
- **Driver signaling:** A indicators (all missing)

## Workflow and methods

Pedestrian motion uses the validated `B VR Pos X/Z` coordinates, never avatar position. Pedestrian and driver planar speeds are consecutive X/Z displacement divided by positive ScenarioTime Δt and then a centered 9-sample rolling mean. Acceleration is the derivative of that smoothed speed and then a centered 11-sample rolling mean. Distance is synchronized planar Euclidean distance and uses the same 9-sample mean. Head-turn rate unwraps head yaw before differentiation; head/body difference is the wrapped head-yaw minus VR-body-yaw angle.

Candidate hand raises are the raw conditions `B Bone Pos LeftHand Y > B Bone Pos LeftShoulder Y` and `B Bone Pos RightHand Y > B Bone Pos RightShoulder Y`. No smoothing or minimum duration was added, so brief tracking noise may appear. These are candidates, not confirmed gestures.

The 12 m proximity threshold, pedestrian motion/head thresholds, speed-dip rule, and possible-yielding rule are reused from microsegmentation v6. `Possible yielding` means a microsegment is within 12 m and its pedestrian motion tag is slowing, pausing, or speed dip with recovery; it does not establish intent. Driver measured-state rows reuse configured acceleration thresholds: ≥ +0.35 m/s² speeding up, ≤ −0.25 m/s² slowing down, otherwise steady while speed is ≥0.15 m/s. No extra duration filter is applied.

Pedestrian conflict-zone occupancy: 19.696533–22.868896 s. Driver occupancy: 16.413330–17.056152 s. These are the first-to-last sampled circle-membership times on the canonical trajectories.

## Limitations

- `A indicators` is missing for all 566 canonical samples. No numeric-to-direction mapping exists to verify, so no No/Left/Right labels were assigned and signal rows are omitted from the Gantt.
- The existing `outputs/conflict_zone_analysis/scenario_03/conflict_zone_events.csv` PedNYC1 row reports pedestrian 15.366–18.316 s and driver 23.033–23.846 s, but direct circle membership for the canonical trajectories is pedestrian 19.697–22.869 s and driver 16.413–17.056 s. The presentation uses the latter, recomputed with the unchanged existing circle equation.
- No original screen-capture video matching the canonical 16:58:19 recording was located. The MP4 therefore animates the correct recorded VR pedestrian/driver trajectories beside the synchronized Gantt; it does not substitute the older 13:48:59 capture.
- The canonical report records a failed regression against older Scenario 3 ground truth (boundaries and labels differed). This presentation faithfully uses the current canonical macro/micro files and does not claim equivalence to the older run.

## Synchronization checks

- All figures and the MP4 use 0.000000–26.343018 s.
- Macro boundaries were read, not recomputed or changed.
- Micro intervals were read, not renamed in analytical CSVs; presentation-only labels are applied at render time.
- Conflict entries/exits were recomputed with the unchanged validated circle definition and checked against sample-level canonical trajectories; the mismatched existing event row is retained untouched and reported above.
