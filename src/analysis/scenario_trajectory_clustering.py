"""Reproducible within- and cross-scenario PedNYC trajectory clustering."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.analysis.behavior_clustering import (
    AnalysisConfig, _read_motion_inputs, apply_rigid_transform, clean_ground_truth,
    cramers_v, extract_trial_features, rigid_transform_parameters,
    robust_scale_within_group, validate_ground_truth,
)
from src.analysis.scenario_metadata import scenario_description
from src.analysis.trajectory_analysis import PROJECT_ROOT


SCENARIOS = (3, 7, 12, 15, 16, 21)
OUTPUT = PROJECT_ROOT / "outputs/scenario_trajectory_clustering"
GROUND_TRUTH = Path(r"C:\Users\patl5\Downloads\XCPedestrian-GroundTruth-RERUN.xlsx")
PATH_POINTS = 50
SEED = 42
RESPONSE_FEATURES = [
    "ped_time_onset_to_zone_entry_s", "arrival_time_difference_s",
    "ped_speed_at_vehicle_arrival_mps", "vehicle_speed_at_pedestrian_arrival_mps",
    "ped_speed_change_mps", "ped_min_interaction_speed_mps",
    "ped_total_stationary_duration_s", "vehicle_speed_change_mps",
    "minimum_distance_m", "post_encroachment_time_s",
]
POOLED_FEATURES = [
    "ped_speed_change_mps", "ped_min_interaction_speed_mps",
    "ped_total_stationary_duration_s", "vehicle_speed_change_mps",
    "minimum_distance_m", "ped_path_efficiency", "ped_max_lateral_deviation_m",
]


def who_went_first_group(value: object) -> str | None:
    return {"Pedestrian": "Pedestrian-first", "Driver": "Vehicle-first"}.get(value)


def validate_pooled_features(columns: list[str]) -> None:
    forbidden = [name for name in columns if name in {"ped_x", "ped_z", "veh_x", "veh_z"} or "normalized_m" in name]
    if forbidden:
        raise ValueError(f"Raw/scenario-relative coordinates cannot be pooled: {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def arc_length_resample(x: np.ndarray, z: np.ndarray, points: int = PATH_POINTS) -> np.ndarray:
    """Resample a planar path at equal cumulative-distance increments."""
    path = np.column_stack([np.asarray(x, float), np.asarray(z, float)])
    path = path[np.isfinite(path).all(axis=1)]
    if len(path) < 2:
        raise ValueError("A path needs at least two finite points.")
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    if len(path) < 2:
        raise ValueError("A path needs nonzero displacement.")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    target = np.linspace(0, distance[-1], points)
    return np.column_stack([
        np.interp(target, distance, path[:, 0]),
        np.interp(target, distance, path[:, 1]),
    ])


def fixed_window(time: np.ndarray, values: np.ndarray, anchor: float,
                 window: tuple[float, float] = (-3.0, 5.0), step: float = .1) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(window[0], window[1] + step / 2, step)
    valid = np.isfinite(time) & np.isfinite(values)
    relative = np.asarray(time, float)[valid] - anchor
    values = np.asarray(values, float)[valid]
    if len(relative) < 2:
        return grid, np.full(len(grid), np.nan)
    order = np.argsort(relative)
    return grid, np.interp(grid, relative[order], values[order], left=np.nan, right=np.nan)


def interaction_anchor(event: pd.Series, zone: dict[str, object]) -> tuple[float, str, bool]:
    """First actual or explicitly counterfactual zone arrival; never closest approach."""
    ped = float(event.get("pedestrian_entry_time", np.nan))
    if zone.get("conflict_zone_method") == "turn_uncertainty" or zone.get("zone_type") == "potential_counterfactual":
        veh = float(event.get("counterfactual_vehicle_arrival_time", np.nan))
        method = "first_actor_turn_uncertainty_zone_arrival"
        fallback = True
    else:
        veh = float(event.get("vehicle_entry_time", np.nan))
        method = "first_actor_geometric_conflict_zone_entry"
        fallback = False
    finite = [value for value in (ped, veh) if np.isfinite(value)]
    return (min(finite), method, fallback) if finite else (math.nan, "missing_zone_entry_no_fallback", fallback)


def _pairwise_rms(paths: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.sum((paths[:, None] - paths[None, :]) ** 2, axis=3), axis=2))


def _labels(matrix: np.ndarray, k: int, precomputed: bool) -> np.ndarray:
    kwargs = {"n_clusters": k, "linkage": "average" if precomputed else "ward"}
    if precomputed:
        try:
            return AgglomerativeClustering(metric="precomputed", **kwargs).fit_predict(matrix)
        except TypeError:
            return AgglomerativeClustering(affinity="precomputed", **kwargs).fit_predict(matrix)
    return AgglomerativeClustering(**kwargs).fit_predict(matrix)


def candidate_metrics(matrix: np.ndarray, repetitions: int, seed: int,
                      precomputed: bool, feature_matrix: np.ndarray | None = None) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows, assignments = [], {}
    n = len(matrix)
    for k in range(2, min(4, n - 1) + 1):
        base = _labels(matrix, k, precomputed)
        assignments[k] = base
        stability = []
        for _ in range(repetitions):
            subset = np.sort(rng.choice(n, max(k + 2, math.ceil(.8 * n)), replace=False))
            submatrix = matrix[np.ix_(subset, subset)] if precomputed else matrix[subset]
            stability.append(adjusted_rand_score(base[subset], _labels(submatrix, k, precomputed)))
        sensitivity = []
        if feature_matrix is not None and feature_matrix.shape[1] > 2:
            for column in range(feature_matrix.shape[1]):
                reduced = np.delete(feature_matrix, column, axis=1)
                sensitivity.append(adjusted_rand_score(base, _labels(reduced, k, False)))
        sizes = np.bincount(base, minlength=k)
        rows.append({
            "k": k, "silhouette": silhouette_score(matrix, base, metric="precomputed") if precomputed else silhouette_score(matrix, base),
            "bootstrap_ari_mean": float(np.mean(stability)), "bootstrap_ari_p10": float(np.quantile(stability, .1)),
            "minimum_cluster_size": int(sizes.min()), "cluster_sizes": ";".join(map(str, sizes)),
            "feature_removal_ari_min": float(min(sensitivity)) if sensitivity else math.nan,
            "accepted": bool(sizes.min() >= 3 and np.mean(stability) >= .70 and (not sensitivity or min(sensitivity) >= .70)),
        })
    return pd.DataFrame(rows), assignments


def choose_candidate(metrics: pd.DataFrame) -> pd.Series:
    accepted = metrics[metrics.accepted]
    pool = accepted if len(accepted) else metrics
    return pool.sort_values(["accepted", "bootstrap_ari_mean", "silhouette"], ascending=False).iloc[0]


def medoid_indices(distance: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    return {int(label): int(indices[np.argmin(distance[np.ix_(indices, indices)].sum(axis=1))])
            for label in np.unique(labels) for indices in [np.flatnonzero(labels == label)]}


def _bootstrap_difference(frame: pd.DataFrame, column: str, seed: int, repetitions: int = 300) -> tuple[float, float, float]:
    valid = frame.dropna(subset=[column, "who_went_first_group"])
    groups = {name: one for name, one in valid.groupby("who_went_first_group")}
    if set(groups) != {"Pedestrian-first", "Vehicle-first"}:
        return math.nan, math.nan, math.nan
    observed = groups["Pedestrian-first"][column].median() - groups["Vehicle-first"][column].median()
    rng, draws = np.random.default_rng(seed), []
    for _ in range(repetitions):
        values = []
        for name in ("Pedestrian-first", "Vehicle-first"):
            one = groups[name]
            participants = one.participant_id.unique()
            sampled = rng.choice(participants, len(participants), replace=True)
            values.append(np.median(np.concatenate([one.loc[one.participant_id.eq(p), column].to_numpy(float) for p in sampled])))
        draws.append(values[0] - values[1])
    return float(observed), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def _write_interactive(scenario: int, normalized: pd.DataFrame, curves: pd.DataFrame,
                       assignments: pd.DataFrame, zone: dict[str, object], root: Path) -> None:
    merged = normalized.merge(assignments, on=["recording_identifier", "participant_id", "scenario"], how="inner")
    title = f"Scenario {scenario}: {scenario_description(scenario)}"
    figure = px.line(merged, x="ped_x_normalized_m", y="ped_z_normalized_m", color="cluster_label",
                     line_group="recording_identifier", hover_name="recording_identifier",
                     hover_data=["participant_id", "analysis_type", "event_anchor", "fallback_event"], title=title)
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    analysis = str(assignments.analysis_type.iloc[0])
    figure.write_html(root / f"scenario_{scenario:02d}_{analysis}_trajectories.html", include_plotlyjs=True)
    for actor in ("pedestrian", "vehicle"):
        one = curves[curves.actor.eq(actor)].merge(
            assignments[["recording_identifier", "participant_id", "scenario", "cluster_label"]],
            on=["recording_identifier", "participant_id", "scenario"],
        )
        fig = px.line(one, x="relative_time_s", y="speed_mps", color="cluster_label", line_group="recording_identifier",
                      hover_name="recording_identifier", hover_data=["participant_id", "event_anchor", "fallback_event"],
                      title=f"{title} — {actor} speed; t=0 first zone arrival")
        fig.add_vrect(x0=0, x1=1, fillcolor="#9467BD", opacity=.10, line_width=0,
                      annotation_text="conflict/uncertainty event")
        fig.write_html(root / f"scenario_{scenario:02d}_{analysis}_{actor}_speed.html", include_plotlyjs=True)


def _write_static_diagnostics(scenario: int, metrics: pd.DataFrame, assignments: pd.DataFrame,
                              features: pd.DataFrame, ground_truth: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = assignments[assignments.analysis_type.eq("conflict_response_clusters")]
    joined = selected.merge(features, on=["recording_identifier", "participant_id", "scenario"]).merge(
        ground_truth, on="trial_key", how="left")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for analysis, one in metrics.groupby("analysis_type"):
        axes[0, 0].plot(one.k, one.silhouette, marker="o", label=analysis.replace("_clusters", ""))
        axes[0, 1].plot(one.k, one.bootstrap_ari_mean, marker="o", label=analysis.replace("_clusters", ""))
    axes[0, 0].set(title="Silhouette by candidate k", xlabel="k", ylabel="Silhouette")
    axes[0, 1].axhline(.70, color="black", linestyle="--", linewidth=.8); axes[0, 1].set(title="Bootstrap stability", xlabel="k", ylabel="Mean ARI")
    sizes = selected.cluster_label.value_counts(); axes[0, 2].bar(sizes.index, sizes.values); axes[0, 2].set(title="Conflict-response cluster sizes", ylabel="Trials"); axes[0, 2].tick_params(axis="x", rotation=30)
    profile_columns = [c for c in RESPONSE_FEATURES if c in joined and joined[c].notna().any()][:6]
    profile = joined.groupby("cluster_label")[profile_columns].median()
    standardized = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    image = axes[1, 0].imshow(standardized, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axes[1, 0].set(title="Median feature profiles", yticks=range(len(profile)), yticklabels=profile.index,
                   xticks=range(len(profile_columns)), xticklabels=[x.replace("_", "\n") for x in profile_columns]); axes[1, 0].tick_params(axis="x", labelsize=6)
    fig.colorbar(image, ax=axes[1, 0], fraction=.05)
    label = "annotated_who_went_first"
    composition = pd.crosstab(joined.cluster_label, joined.get(label), normalize="index") if label in joined else pd.DataFrame()
    if len(composition): composition.plot(kind="bar", stacked=True, ax=axes[1, 1], legend=True)
    axes[1, 1].set(title="Ground-truth who-went-first composition", ylabel="Within-cluster fraction")
    accepted = metrics.groupby("analysis_type").accepted.any(); axes[1, 2].bar(accepted.index, accepted.astype(int)); axes[1, 2].set(title="Any accepted candidate", ylim=(0, 1.2), ylabel="1=yes")
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.suptitle(f"Scenario {scenario}: {scenario_description(scenario)}", weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap_repetitions < 10:
        raise ValueError("At least 10 bootstrap repetitions are required.")
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    if args.overwrite and output.exists():
        import shutil
        shutil.rmtree(output)
    directories = {name: output / name for name in (
        "within_scenario", "cross_scenario", "who_went_first_groups", "interactive_html", "static_figures"
    )}
    for directory in directories.values(): directory.mkdir(parents=True, exist_ok=True)

    manifest, availability, traces, events, zones = _read_motion_inputs(args.input_root.resolve(), list(SCENARIOS))
    config = replace(AnalysisConfig(), seed=args.seed, bootstrap_repetitions=args.bootstrap_repetitions, curve_window=(-3.0, 5.0))
    features, extracted_events, qc, normalized, transforms = extract_trial_features(traces, events, zones, config)
    ground_truth = clean_ground_truth(args.ground_truth.resolve(), list(SCENARIOS))

    # One deterministic recording per StudyFolder + ScenarioName; duplicates are auditable exclusions.
    features = features.sort_values("recording_identifier")
    duplicate = features.duplicated("trial_key", keep="first")
    exclusions = features.loc[duplicate, ["recording_identifier", "participant_id", "scenario", "trial_key"]].copy()
    exclusions["exclusion_reason"] = "duplicate_motion_recording_for_studyfolder_scenario"
    keep_ids = set(features.loc[~duplicate, "recording_identifier"])
    features = features[~duplicate].reset_index(drop=True)
    traces = traces[traces.recording_identifier.isin(keep_ids)]
    normalized = normalized[normalized.recording_identifier.isin(keep_ids)]
    extracted_events = extracted_events[extracted_events.recording_identifier.isin(keep_ids)]
    source_lookup = manifest.set_index("recording_identifier").source_csv_path.to_dict()
    event_lookup = events.set_index("recording_identifier")

    anchors = []
    for row in features.itertuples():
        anchor, method, fallback = interaction_anchor(event_lookup.loc[row.recording_identifier], zones[row.scenario])
        anchors.append({"recording_identifier": row.recording_identifier, "event_anchor_time_s": anchor,
                        "event_anchor": method, "fallback_event": fallback})
    anchors = pd.DataFrame(anchors)
    features = features.merge(anchors, on="recording_identifier")
    features["source_file"] = features.recording_identifier.map(source_lookup)
    features["conflict_zone_definition_used"] = features.scenario.map(lambda s: zones[int(s)].get("conflict_zone_method", zones[int(s)].get("method")))
    features["data_coverage_fraction"] = features.recording_identifier.map(
        qc.set_index("recording_identifier").pedestrian_position_missing_percent.rsub(100).div(100).to_dict()
    )
    features["exclusion_reason"] = ""

    # Actor-arrival speeds are evaluated on observed speed at the corresponding actual/counterfactual timestamp.
    event_by_id = event_lookup
    for name in ("ped_speed_at_vehicle_arrival_mps", "vehicle_speed_at_pedestrian_arrival_mps"):
        features[name] = np.nan
    path_rows, curve_rows, assignment_rows, validation_rows, prototype_rows = [], [], [], [], []
    scenario_path_objects: dict[int, tuple[list[str], np.ndarray]] = {}
    for scenario in SCENARIOS:
        zone, transform = zones[scenario], rigid_transform_parameters(traces[traces.scenario.eq(scenario)], zones[scenario])
        ids, paths = [], []
        for recording, one in traces[traces.scenario.eq(scenario)].groupby("recording_identifier", sort=True):
            anchor = float(anchors.loc[anchors.recording_identifier.eq(recording), "event_anchor_time_s"].iloc[0])
            relevant = one[one.scenario_time.between(anchor - 3, anchor + 5)] if np.isfinite(anchor) else one.iloc[0:0]
            if len(relevant) < 3:
                exclusions.loc[len(exclusions)] = [recording, one.participant_id.iloc[0], scenario, f"{one.participant_id.iloc[0]}_scenario{scenario}", "missing_conflict_zone_entry_or_window"]
                continue
            x, z = apply_rigid_transform(relevant.ped_x, relevant.ped_z, transform)
            try: path = arc_length_resample(x, z)
            except ValueError: continue
            ids.append(recording); paths.append(path)
            for index, point in enumerate(path):
                path_rows.append({"recording_identifier": recording, "participant_id": one.participant_id.iloc[0], "scenario": scenario,
                                  "path_progress": index / (PATH_POINTS - 1), "x_normalized_m": point[0], "z_normalized_m": point[1]})
            event = event_by_id.loc[recording]
            ped_entry = float(event.get("pedestrian_entry_time", np.nan))
            veh_arrival = float(event.get("counterfactual_vehicle_arrival_time", np.nan)) if zone.get("conflict_zone_method") == "turn_uncertainty" else float(event.get("vehicle_entry_time", np.nan))
            index = features.index[features.recording_identifier.eq(recording)][0]
            features.loc[index, "ped_speed_at_vehicle_arrival_mps"] = float(one.iloc[np.argmin(abs(one.scenario_time - veh_arrival))].ped_speed) if np.isfinite(veh_arrival) else np.nan
            features.loc[index, "vehicle_speed_at_pedestrian_arrival_mps"] = float(one.iloc[np.argmin(abs(one.scenario_time - ped_entry))].veh_speed) if np.isfinite(ped_entry) else np.nan
            # Recalculate response summaries around this analysis's required
            # first-arrival anchor (the legacy extractor used an entry midpoint).
            relative = one.scenario_time - anchor
            def median_between(column: str, low: float, high: float) -> float:
                values = one.loc[relative.between(low, high), column]
                return float(values.median()) if len(values) else math.nan
            ped_before, ped_after = median_between("ped_speed", -3, -1), median_between("ped_speed", 1, 3)
            veh_before, veh_after = median_between("veh_speed", -3, -1), median_between("veh_speed", 1, 3)
            window = relative.between(-3, 5).to_numpy()
            dt = np.r_[np.diff(one.scenario_time.to_numpy(float)), 0.0]
            features.loc[index, "arrival_time_difference_s"] = ped_entry - veh_arrival if np.isfinite([ped_entry, veh_arrival]).all() else np.nan
            features.loc[index, "ped_time_onset_to_zone_entry_s"] = ped_entry - features.loc[index, "pedestrian_movement_onset_time_s"] if np.isfinite(ped_entry) else np.nan
            features.loc[index, "ped_speed_change_mps"] = ped_after - ped_before
            features.loc[index, "vehicle_speed_change_mps"] = veh_after - veh_before
            features.loc[index, "ped_min_interaction_speed_mps"] = median_between("ped_speed", -1, 1) if not relative.between(-1, 1).any() else float(one.loc[relative.between(-1, 1), "ped_speed"].min())
            features.loc[index, "ped_total_stationary_duration_s"] = float(dt[window & one.ped_speed.lt(config.stationary_speed_mps).to_numpy()].sum())
            for actor, column in (("pedestrian", "ped_speed"), ("vehicle", "veh_speed")):
                grid, values = fixed_window(one.scenario_time.to_numpy(), one[column].to_numpy(), anchor)
                curve_rows.extend({"recording_identifier": recording, "participant_id": one.participant_id.iloc[0], "scenario": scenario,
                                   "actor": actor, "relative_time_s": t, "speed_mps": value,
                                   "event_anchor": anchors.loc[anchors.recording_identifier.eq(recording), "event_anchor"].iloc[0],
                                   "fallback_event": bool(anchors.loc[anchors.recording_identifier.eq(recording), "fallback_event"].iloc[0])}
                                  for t, value in zip(grid, values))
        path_array = np.stack(paths); distance = _pairwise_rms(path_array); scenario_path_objects[scenario] = (ids, path_array)
        metrics, labels_lookup = candidate_metrics(distance, args.bootstrap_repetitions, args.seed + scenario, True)
        selected = choose_candidate(metrics); labels = labels_lookup[int(selected.k)]; medoids = medoid_indices(distance, labels)
        metrics.assign(scenario=scenario, analysis_type="path_shape_clusters").to_csv(directories["within_scenario"] / f"scenario_{scenario:02d}_path_candidates.csv", index=False)
        for i, recording in enumerate(ids):
            assignment_rows.append({"recording_identifier": recording, "participant_id": features.loc[features.recording_identifier.eq(recording), "participant_id"].iloc[0],
                                    "scenario": scenario, "analysis_type": "path_shape_clusters", "cluster_id": int(labels[i]),
                                    "cluster_label": f"Path {int(labels[i])+1}", "accepted_solution": bool(selected.accepted),
                                    "event_anchor": anchors.loc[anchors.recording_identifier.eq(recording), "event_anchor"].iloc[0],
                                    "fallback_event": bool(anchors.loc[anchors.recording_identifier.eq(recording), "fallback_event"].iloc[0])})
        validation_rows.extend(metrics.assign(scenario=scenario, analysis_type="path_shape_clusters").to_dict("records"))
        for label, medoid in medoids.items(): prototype_rows.append({"scenario": scenario, "analysis_type": "path_shape_clusters", "cluster_id": label, "cluster_size": int((labels == label).sum()), "medoid_participant": ids[medoid]})

        response = features[features.scenario.eq(scenario) & features.event_anchor_time_s.notna()].copy()
        usable = [name for name in RESPONSE_FEATURES if response[name].notna().mean() >= .65 and
                  response[name].nunique(dropna=True) > 1 and
                  float(response[name].quantile(.75) - response[name].quantile(.25)) > 1e-9]
        scaled, _ = robust_scale_within_group(response.assign(group=1), usable, "group")
        matrix = scaled[usable].to_numpy()
        rmetrics, rlookup = candidate_metrics(matrix, args.bootstrap_repetitions, args.seed + 100 + scenario, False, matrix)
        rselected = choose_candidate(rmetrics); rlabels = rlookup[int(rselected.k)]; rdistance = np.linalg.norm(matrix[:, None] - matrix[None, :], axis=2); rmedoids = medoid_indices(rdistance, rlabels)
        rmetrics.assign(scenario=scenario, analysis_type="conflict_response_clusters", retained_features=";".join(usable)).to_csv(directories["within_scenario"] / f"scenario_{scenario:02d}_response_candidates.csv", index=False)
        for i, row in enumerate(response.itertuples()):
            assignment_rows.append({"recording_identifier": row.recording_identifier, "participant_id": row.participant_id, "scenario": scenario,
                                    "analysis_type": "conflict_response_clusters", "cluster_id": int(rlabels[i]), "cluster_label": f"Response {int(rlabels[i])+1}",
                                    "accepted_solution": bool(rselected.accepted), "event_anchor": row.event_anchor, "fallback_event": row.fallback_event})
        validation_rows.extend(rmetrics.assign(scenario=scenario, analysis_type="conflict_response_clusters", retained_features=";".join(usable)).to_dict("records"))
        for label, medoid in rmedoids.items(): prototype_rows.append({"scenario": scenario, "analysis_type": "conflict_response_clusters", "cluster_id": label, "cluster_size": int((rlabels == label).sum()), "medoid_participant": response.iloc[medoid].recording_identifier})

    assignments = pd.DataFrame(assignment_rows); validations = pd.DataFrame(validation_rows); prototypes = pd.DataFrame(prototype_rows)
    paths = pd.DataFrame(path_rows); curves = pd.DataFrame(curve_rows)
    for scenario in SCENARIOS:
        for analysis_type in ("path_shape_clusters", "conflict_response_clusters"):
            a = assignments[(assignments.scenario.eq(scenario)) & assignments.analysis_type.eq(analysis_type)]
            _write_interactive(scenario, normalized, curves, a, zones[scenario], directories["interactive_html"])

    # Cross-scenario behavior only: no raw coordinates.
    retained = [name for name in POOLED_FEATURES if features[name].notna().mean() >= .65 and all(
        float(one[name].quantile(.75) - one[name].quantile(.25)) > 1e-9
        for _, one in features.groupby("scenario")
    )]
    validate_pooled_features(retained)
    pooled_scaled, _ = robust_scale_within_group(features, retained, "scenario")
    pooled_matrix = pooled_scaled[retained].to_numpy(); pmetrics, plookup = candidate_metrics(pooled_matrix, args.bootstrap_repetitions, args.seed + 500, False, pooled_matrix)
    pselected = choose_candidate(pmetrics); plabels = plookup[int(pselected.k)]
    pooled = features[["recording_identifier", "participant_id", "scenario"]].copy(); pooled["cluster_id"] = plabels
    table = pd.crosstab(pooled.scenario, pooled.cluster_id); table.to_csv(directories["cross_scenario"] / "scenario_cluster_contingency.csv")
    pmetrics.to_csv(directories["cross_scenario"] / "candidate_validation.csv", index=False)
    (directories["cross_scenario"] / "result.json").write_text(json.dumps({"selected": pselected.to_dict(), "cramers_v_scenario": cramers_v(table), "raw_coordinates_excluded": True, "retained_features": retained}, indent=2), encoding="utf-8")

    # Ground truth is post-clustering only.
    gt_columns = [column for column in ground_truth.columns if column.startswith("annotated_") or column in ("driver_signal", "pedestrian_hand_gesture")]
    joined = features.merge(ground_truth[["trial_key", *gt_columns]], on="trial_key", how="left")
    joined["who_went_first_group"] = joined.get("annotated_who_went_first", pd.Series(index=joined.index, dtype=object)).map(who_went_first_group)
    ambiguous = joined[joined.who_went_first_group.isna()][["recording_identifier", "participant_id", "scenario", "trial_key"]].copy(); ambiguous["exclusion_reason"] = "missing_or_ambiguous_who_went_first"
    exclusions = pd.concat([exclusions, ambiguous], ignore_index=True)
    grouped = joined.dropna(subset=["who_went_first_group"]).copy()
    grouped.to_csv(directories["who_went_first_groups"] / "trials_with_groups.csv", index=False)
    association_tests, association_distributions = [], []
    for analysis_type, one_assignment in assignments.groupby("analysis_type"):
        _, tests, distributions, _, _ = validate_ground_truth(
            one_assignment.reset_index(drop=True), ground_truth, features
        )
        tests.insert(0, "analysis_type", analysis_type)
        distributions.insert(0, "analysis_type", analysis_type)
        association_tests.append(tests); association_distributions.append(distributions)
    pd.concat(association_tests, ignore_index=True).to_csv(
        output / "ground_truth_association_tests.csv", index=False
    )
    pd.concat(association_distributions, ignore_index=True).to_csv(
        output / "ground_truth_cluster_composition.csv", index=False
    )
    # Enrich prototype rows with scenario-standardized profiles and annotation
    # proportions; medoids remain real recordings, never synthetic averages.
    enriched_prototypes = []
    for prototype in prototypes.to_dict("records"):
        members = assignments[
            assignments.scenario.eq(prototype["scenario"])
            & assignments.analysis_type.eq(prototype["analysis_type"])
            & assignments.cluster_id.eq(prototype["cluster_id"])
        ]
        member_features = features[features.recording_identifier.isin(members.recording_identifier)]
        scenario_features = features[features.scenario.eq(prototype["scenario"])]
        names = [name for name in POOLED_FEATURES if name in features]
        profile = {}
        for name in names:
            median = scenario_features[name].median(); iqr = scenario_features[name].quantile(.75) - scenario_features[name].quantile(.25)
            profile[name] = float((member_features[name].median() - median) / iqr) if np.isfinite(iqr) and iqr > 1e-9 else math.nan
        annotations = grouped[grouped.recording_identifier.isin(members.recording_identifier)].who_went_first_group.value_counts(normalize=True)
        medoid_recording = prototype["medoid_participant"]
        prototype["medoid_recording"] = medoid_recording
        matches = features[features.recording_identifier.eq(medoid_recording)]
        prototype["medoid_participant"] = matches.participant_id.iloc[0] if len(matches) else medoid_recording
        prototype["standardized_feature_profile"] = json.dumps(profile, sort_keys=True)
        prototype["main_trajectory_characteristics"] = "; ".join(
            f"{name}={member_features[name].median():.3g}" for name in
            ("ped_path_efficiency", "ped_max_lateral_deviation_m", "ped_speed_change_mps")
        )
        prototype["ground_truth_label_proportions"] = json.dumps(annotations.to_dict(), sort_keys=True)
        enriched_prototypes.append(prototype)
    prototypes = pd.DataFrame(enriched_prototypes)
    group_summary = grouped.groupby(["scenario", "who_went_first_group"]).agg(
        trials=("recording_identifier", "size"), entry_time_difference_median=("arrival_time_difference_s", "median"),
        stopped_duration_median=("ped_total_stationary_duration_s", "median"), path_deviation_median=("ped_max_lateral_deviation_m", "median"),
        minimum_distance_median=("minimum_distance_m", "median"), ped_speed_at_vehicle_arrival_median=("ped_speed_at_vehicle_arrival_mps", "median"),
        vehicle_speed_at_pedestrian_arrival_median=("vehicle_speed_at_pedestrian_arrival_mps", "median"),
    ).reset_index()
    comparisons = []
    for scenario, one in grouped.groupby("scenario"):
        for feature in ("arrival_time_difference_s", "ped_total_stationary_duration_s", "ped_max_lateral_deviation_m", "minimum_distance_m", "ped_speed_at_vehicle_arrival_mps", "vehicle_speed_at_pedestrian_arrival_mps"):
            effect, low, high = _bootstrap_difference(one, feature, args.seed + scenario)
            groups = [x[feature].dropna() for _, x in one.groupby("who_went_first_group")]
            p = mannwhitneyu(*groups).pvalue if len(groups) == 2 and all(len(x) for x in groups) else math.nan
            comparisons.append({"scenario": scenario, "feature": feature, "median_difference_pedestrian_minus_vehicle": effect, "ci_2_5": low, "ci_97_5": high, "mann_whitney_p": p})
    group_summary.to_csv(directories["who_went_first_groups"] / "group_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(directories["who_went_first_groups"] / "continuous_comparisons.csv", index=False)
    assignments.merge(grouped[["recording_identifier", "who_went_first_group"]], on="recording_identifier", how="inner").groupby(["scenario", "analysis_type"]).apply(
        lambda x: adjusted_rand_score(pd.factorize(x.who_went_first_group)[0], x.cluster_id), include_groups=False
    ).rename("ari_with_who_went_first").reset_index().to_csv(directories["who_went_first_groups"] / "cluster_agreement.csv", index=False)

    for scenario in SCENARIOS:
        group_one = grouped[grouped.scenario.eq(scenario)][["recording_identifier", "who_went_first_group"]]
        group_paths = paths[paths.scenario.eq(scenario)].merge(group_one, on="recording_identifier")
        fig = px.line(group_paths, x="x_normalized_m", y="z_normalized_m", color="who_went_first_group",
                      line_group="recording_identifier", hover_name="recording_identifier",
                      title=f"Scenario {scenario} — pedestrian-first versus vehicle-first paths")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.write_html(directories["who_went_first_groups"] / f"scenario_{scenario:02d}_trajectory_overlay.html", include_plotlyjs=True)
        for actor in ("pedestrian", "vehicle"):
            group_curves = curves[(curves.scenario.eq(scenario)) & curves.actor.eq(actor)].merge(group_one, on="recording_identifier")
            fig = px.line(group_curves, x="relative_time_s", y="speed_mps", color="who_went_first_group",
                          line_group="recording_identifier", hover_name="recording_identifier",
                          title=f"Scenario {scenario} — {actor} speed by manual who-went-first group")
            fig.add_vline(x=0, line_dash="dash")
            fig.write_html(directories["who_went_first_groups"] / f"scenario_{scenario:02d}_{actor}_speed.html", include_plotlyjs=True)
        _write_static_diagnostics(
            scenario, validations[validations.scenario.eq(scenario)], assignments[assignments.scenario.eq(scenario)],
            features[features.scenario.eq(scenario)], ground_truth,
            directories["static_figures"] / f"scenario_{scenario:02d}_diagnostics.png",
        )

    features.to_csv(output / "trial_features.csv", index=False); exclusions.to_csv(output / "exclusions.csv", index=False)
    assignments.to_csv(output / "cluster_assignments.csv", index=False); validations.to_csv(output / "cluster_validation.csv", index=False); prototypes.to_csv(output / "cluster_prototypes.csv", index=False)
    summary = validations.loc[validations.groupby(["scenario", "analysis_type"]).bootstrap_ari_mean.idxmax()].copy(); summary.to_csv(output / "summary_results.csv", index=False)
    paths.to_csv(directories["within_scenario"] / "arc_length_resampled_paths.csv", index=False); curves.to_csv(directories["within_scenario"] / "event_aligned_speed_curves.csv", index=False)
    run_manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "bootstrap_repetitions": args.bootstrap_repetitions,
                    "scenarios": list(SCENARIOS), "input_root": str(args.input_root.resolve()), "ground_truth": str(args.ground_truth.resolve()),
                    "coordinate_source": "B VR Pos X/Z via validated trajectory pipeline", "path_points": PATH_POINTS, "event_window_seconds": [-3, 5],
                    "conflict_zone_methods": {str(s): zones[s].get("conflict_zone_method", zones[s].get("method")) for s in SCENARIOS},
                    "valid_trials_by_scenario": features.groupby("scenario").size().astype(int).to_dict(), "excluded_rows": int(len(exclusions)),
                    "pooled_cramers_v_scenario": cramers_v(table), "package_note": "No additional dependencies installed."}
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    methodology = f"""# Scenario trajectory clustering methodology

Inputs are the validated `outputs/trajectory_analysis/scenario_*/raw_trajectory_samples.csv` files, conflict-zone JSON/event CSV files, processing manifests, and `{args.ground_truth.resolve()}`. Pedestrian position is B VR Pos X/Z; avatar position is never used.

Paths are restricted to -3 to +5 seconds around the first actor's zone arrival, translated to the shared scenario zone center, rotated toward positive pedestrian travel, and resampled to {PATH_POINTS} equal cumulative-distance increments. Path clustering uses average-linkage agglomeration on pairwise RMS spatial distance. Response clustering uses Ward linkage on compact robustly standardized timing, speed, stopping, distance, and PET features. Candidate k is 2–4; acceptance requires minimum cluster size 3, mean subsample ARI at least 0.70, and leave-one-feature-out ARI at least 0.70 where applicable. Seed={args.seed}; bootstrap repetitions={args.bootstrap_repetitions}.

For Scenarios 12 and 16, t=0 uses the first pedestrian entry or explicitly modeled counterfactual vehicle arrival in the existing `turn_uncertainty` zone. This is labeled as a fallback/counterfactual anchor and is not an observed vehicle-zone entry. A trial with neither arrival is excluded rather than aligned to closest approach. Speed uses the upstream centered nine-sample smoothing. Missing numeric features are median-imputed only within scenario by the robust scaler; labels remain missing. Ground truth is joined after clustering and is never used as an unsupervised feature. Cross-scenario clustering uses within-scenario standardized behavioral summaries and explicitly excludes raw coordinates. Results are exploratory; repeated participants, small groups, modeled uncertainty arrivals, missing annotations, and sensitivity to window/feature choices limit generalization.
"""
    (output / "methodology.md").write_text(methodology, encoding="utf-8")
    return run_manifest


def main() -> int:
    args = parse_args(); print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
