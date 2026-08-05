# Skeleton behavior LSTM experiment

This experiment predicts only Xiang's observable pedestrian hand-gesture Yes/No label. It does **not** infer intent, yielding, awareness, or hesitation.

## Dataset

- Conflict-centered 6-second windows, resampled to 61 samples without extrapolation.
- Observed scenarios use the mean of available pedestrian/vehicle conflict-zone center times. Scenarios 12 and 16 use the existing potential/counterfactual vehicle-arrival time.
- 20 Unity humanoid joints are pelvis-centered and rotated by pelvis yaw into a participant-facing frame. Original joint masks and actual source-window durations are preserved.
- Included: 138 trials from 30 participants; Yes=25, No=113.
- Exclusions are listed explicitly in `dataset/dataset_manifest.csv`. Unknown labels remain Unknown and are excluded.

## Evaluation

- Five-fold `StratifiedGroupKFold`, grouped by PedNYC participant. No participant crosses train/test within a fold.
- Metrics are out-of-fold; confidence intervals resample participants, not frames.
- Models: majority, a new transparent hand-motion threshold rule (not an existing study detector), context-only logistic regression, handcrafted-skeleton logistic regression, small masked NumPy LSTM, and skeleton+context LSTM.

Best observed balanced accuracy was 0.610 for `hand_motion_rule`. Skeleton LSTM=0.485, context-only=0.576, fused=0.463. All results remain exploratory; inspect `metrics/aggregate_metrics_with_ci.csv` for uncertainty.

## Interpretation guardrails

Whether skeleton data adds measurable value is assessed only by held-out participant performance relative to context-only and uncertainty intervals. Vehicle/context necessity is assessed by the context-only and fused comparisons. Wide participant-bootstrap intervals or fold instability mean generalization and stability are not established.

## Reproduction

```powershell
$env:PYTHONPATH="$(Resolve-Path '.python_packages');$(Resolve-Path '.')"
& 'C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m src.analysis.skeleton_behavior_lstm
```
