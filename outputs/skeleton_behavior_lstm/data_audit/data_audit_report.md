# Skeleton data audit

## Scope and provenance

- Audited 155 successfully processed trials across scenarios 3, 7, 12, 15, 16, 21.
- Found 648 pedestrian (`B`) skeleton/VR/avatar scalar columns after whitespace normalization.
- Xiang's workbook is referenced by the repository as `XCPedestrian-GroundTruth-RERUN.xlsx`, but is absent from this working tree. This experiment uses the existing immutable cleaned export at `outputs\behavior_clustering\data_qc\ground_truth_cleaned.csv` and preserves its row numbers and categorical values.
- `B Avatar Pos` is inventoried but never used as global pedestrian trajectory. Existing `B VR Pos X/Z` trajectory artifacts supply pedestrian position where context is needed.

## Timing and quality

- Time is reconstructed from consecutive `ScenarioTime` differences (`dt`), never from the misleading `FrameRate` field.
- Median trial sampling rate: 23.95 Hz (trial-level range 17.77–24.09 Hz).
- Median fixed-joint valid-sample fraction: 1.000.
- Joint-speed spikes are audit flags at > 10 m/s; they are not silently relabeled.
- Accelerations are not model inputs: double differentiation at the observed irregular sampling intervals was judged too noise-sensitive for this exploratory sample. Joint velocities are retained.

## Labels

- Pedestrian hand gesture: Yes=30, No=133, Unknown=23. Unknown is excluded, never converted to No.
- Pedestrian speed change is genuinely annotated but is multiclass and deferred until the hand-gesture pipeline is validated.
- No ground-truth field genuinely labels yielding or hesitation. Right-of-way and who-went-first fields are not relabeled as those constructs.

See `skeleton_column_inventory.csv`, `joint_availability_by_trial.csv`, `trial_sampling_quality.csv`, and `label_counts.csv` for complete tables.
