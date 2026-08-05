from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.avatar_position_rotation import (
    MEASUREMENTS,
    choose_time_axis,
    discover_decoded_csvs,
    elapsed_seconds,
    inspect_header,
    map_normalized_columns,
    normalize_header,
    read_relevant_data,
)


def test_normalize_header_handles_spacing_quotes_and_semicolons():
    assert normalize_header('  "B   Avatar Pos X ;"  ') == "b avatar pos x"
    assert normalize_header("\ufeff ScenarioTime ") == "scenariotime"


def test_semicolon_csv_matches_required_headers_and_preserves_missing_values(tmp_path: Path):
    path = tmp_path / "decoded.csv"
    columns = {f"  {item.column} ;": [1.0, np.nan, 3.0] for item in MEASUREMENTS}
    columns[" ScenarioTime "] = [10.0, 10.1, 10.2]
    pd.DataFrame(columns).to_csv(path, sep=";", index=False)

    header = inspect_header(path)
    normalized, duplicates = map_normalized_columns(header.columns)
    assert not duplicates
    selected = [normalized[normalize_header(item.column)] for item in MEASUREMENTS]
    selected.append(normalized[normalize_header("ScenarioTime")])
    frame = read_relevant_data(path, header, selected)

    assert header.delimiter == ";"
    assert len(frame) == 3
    assert pd.isna(frame[selected[0]].iloc[1])


def test_scenario_time_is_not_zeroed_or_resampled():
    frame = pd.DataFrame({" ScenarioTime ": [4.25, np.nan, 4.75]})
    normalized, _ = map_normalized_columns(tuple(frame.columns))
    values, column, method, label = choose_time_axis(frame, normalized)

    assert column == " ScenarioTime "
    assert values.iloc[0] == 4.25
    assert pd.isna(values.iloc[1])
    assert values.iloc[2] == 4.75
    assert "source values unchanged" in method
    assert label == "Scenario Time (s)"


def test_timestamp_fallback_is_converted_to_elapsed_seconds():
    elapsed, detail = elapsed_seconds(
        pd.Series([1_000.0, 1_250.0, np.nan]), "timestamp ms"
    )
    assert elapsed.iloc[0] == 0.0
    assert elapsed.iloc[1] == 0.25
    assert pd.isna(elapsed.iloc[2])
    assert "milliseconds" in detail


def test_discovery_uses_only_decoded_csvs_in_requested_participant_roots(tmp_path: Path):
    decoded = tmp_path / "pednyc1" / "scenario3" / "odd_name_decoded.csv"
    derivative = decoded.parent / "segments_PedNYC1_scenario3_v1.csv"
    decoded.parent.mkdir(parents=True)
    decoded.touch()
    derivative.touch()

    found = discover_decoded_csvs(tmp_path, ["PedNYC1"])

    assert found == [("PedNYC1", "3", decoded.resolve())]
