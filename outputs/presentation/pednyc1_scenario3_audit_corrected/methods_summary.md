# PedNYC1 Scenario 3 corrected presentation workflow

## Source and synchronization

- Single authoritative input: `C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\decoded_clean_PedNYC1_scenario3.csv`
- SHA-256: `2FB6329BA8E2223D1367325CD32C1C9502EB98930224A3C877E63CE05F84E251`
- Source shape: 648 rows x 1,006 columns.
- Common time axis: `ScenarioTime - ScenarioTime[0]`, 0.000000 to 31.002930 s.
- No source rows were removed. One shared dataframe supplies both actors and every signal.
- Derived metrics: `outputs\presentation\pednyc1_scenario3_audit_corrected\derived_metrics_from_authoritative.csv`. This is a derivative, not an independent source.
- Macro output: `outputs\presentation\pednyc1_scenario3_audit_corrected\macro\macro_segments_PedNYC1_scenario3_authoritative.csv` (5 regenerated segments).
- Micro output: `outputs\presentation\pednyc1_scenario3_audit_corrected\micro\v6\micro_segments_descriptive_PedNYC1_scenario3_v6.csv` (29 regenerated segments).

## Authoritative columns

- Pedestrian position: `B VR Pos X`, `B VR Pos Z`; body yaw: `B VR Rot Y`.
- Head orientation: `B  Bone Rot Head X/Y/Z`.
- Hands: `B  Bone Pos LeftHand X/Y/Z`, `B  Bone Pos RightHand X/Y/Z`.
- Shoulders: `B  Bone Pos LeftShoulder X/Y/Z`, `B  Bone Pos RightShoulder X/Y/Z`.
- Driver position: `A car Pos X`, `A car Pos Z`; yaw: `A car Rot Y`; signaling: `A indicators`.

## Calculations

Planar speed is consecutive X/Z displacement divided by positive ScenarioTime delta, followed by a centered 9-sample rolling mean. Acceleration is the derivative of smoothed speed, followed by a centered 11-sample mean. Distance is synchronized planar Euclidean distance with the same 9-sample mean. Head-turn rate unwraps head yaw before differentiation; head/body difference is wrapped to [-180, 180] degrees.

Candidate hand raises are raw `hand Y > corresponding shoulder Y` conditions with no smoothing or minimum duration. Signaling is decoded directly: `LeftFalse_RightFalse` = No signal, `LeftTrue_RightFalse` = Left signal, `LeftFalse_RightTrue` = Right signal.

The existing conflict-zone circle is centered at (11.846832, 58.392967) m with radius 2.0 m. Newly computed occupancy is pedestrian 15.366455-18.316406 s and driver 23.032715-23.845947 s.

The 12 m proximity threshold, pedestrian micro rules, and possible-yielding rule are reused without modification. Possible yielding remains an unvalidated interpretation. Driver rows use existing acceleration thresholds: >= +0.35 m/s2 speeding up, <= -0.25 m/s2 slowing down, otherwise steady while speed >= 0.15 m/s.

No matching original screen capture was established from this CSV alone. The MP4 animates the authoritative VR trajectories and regenerated Gantt rather than substituting another recording.
