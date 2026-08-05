import numpy as np
import pandas as pd

from src.visualization.avatar_speed_acceleration import calculate_avatar_kinematics


def test_known_3d_speed_and_acceleration_with_actual_time_differences():
    result = calculate_avatar_kinematics(
        pd.Series([0.0, 2.0, 3.0]),
        pd.Series([0.0, 2.0, 3.0]),
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series([0.0, 0.0, 0.0]),
    )

    assert np.isnan(result.loc[0, "avatar_speed_3d"])
    np.testing.assert_allclose(result.loc[1:, "avatar_speed_3d"], [1.0, 1.0])
    assert np.isnan(result.loc[1, "avatar_acceleration_3d"])
    assert result.loc[2, "avatar_acceleration_3d"] == 0.0


def test_xz_and_3d_speed_are_distinct_and_acceleration_comes_from_speed():
    result = calculate_avatar_kinematics(
        pd.Series([0.0, 1.0, 3.0]),
        pd.Series([0.0, 3.0, 3.0]),
        pd.Series([0.0, 4.0, 4.0]),
        pd.Series([0.0, 0.0, 8.0]),
    )

    assert result.loc[1, "avatar_speed_xz"] == 3.0
    assert result.loc[1, "avatar_speed_3d"] == 5.0
    assert result.loc[2, "avatar_speed_xz"] == 4.0
    assert result.loc[2, "avatar_speed_3d"] == 4.0
    assert result.loc[2, "avatar_acceleration_xz"] == 0.5
    assert result.loc[2, "avatar_acceleration_3d"] == -0.5


def test_invalid_time_intervals_remain_visible_and_do_not_get_derivatives():
    result = calculate_avatar_kinematics(
        pd.Series([0.0, 1.0, 1.0, 0.5, np.nan, 2.0]),
        pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([0.0] * 6),
        pd.Series([0.0] * 6),
    )

    assert result["time_interval_status"].tolist() == [
        "no_preceding_sample", "valid", "duplicate_or_zero_time",
        "negative_time", "missing_time", "missing_time",
    ]
    assert np.isnan(result.loc[2:, "avatar_speed_xz"]).all()
    assert result["original_row_number"].tolist() == [2, 3, 4, 5, 6, 7]


def test_threshold_flags_are_diagnostic_and_can_be_disabled_for_raw_units():
    raw = calculate_avatar_kinematics(
        pd.Series([0.0, 1.0]), pd.Series([0.0, 10.0]),
        pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0]),
    )
    metric = calculate_avatar_kinematics(
        pd.Series([0.0, 1.0]), pd.Series([0.0, 10.0]),
        pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0]),
        thresholds_enabled=True,
    )

    assert raw["diagnostic_speed_flag"].isna().all()
    assert bool(metric.loc[1, "diagnostic_speed_flag"])
