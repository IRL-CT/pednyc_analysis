import numpy as np

from src.analysis.skeleton_behavior_lstm import (
    NumpyLSTMClassifier,
    clean_binary_label,
    extract_conflict_window,
    grouped_splits,
    normalize_skeleton,
    resample_fixed_length,
)


def test_unknown_label_is_excluded_never_no():
    assert clean_binary_label("Yes") == 1
    assert clean_binary_label("No") == 0
    assert clean_binary_label("Unknown") is None
    assert clean_binary_label(np.nan) is None


def test_participant_grouped_splits_have_no_leakage():
    groups = np.repeat([f"PedNYC{i}" for i in range(10)], 2)
    labels = np.tile([0, 1], 10)
    for train, test in grouped_splits(labels, groups, 5, 42):
        assert set(groups[train]).isdisjoint(groups[test])


def test_skeleton_normalization_centers_pelvis_and_rotates_yaw():
    positions = np.array([[[10, 2, 20], [11, 3, 20]]], dtype=float)
    mask = np.ones((1, 2), dtype=bool)
    normalized, normalized_mask = normalize_skeleton(positions, mask, 0, np.array([90.0]))
    np.testing.assert_allclose(normalized[0, 0], [0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(normalized[0, 1], [0, 1, 1], atol=1e-7)
    assert normalized_mask.all()


def test_conflict_window_requires_full_observed_support():
    time = np.arange(0, 11, dtype=float)
    indices, reason = extract_conflict_window(time, 5, 3, 3)
    np.testing.assert_array_equal(indices, np.arange(2, 9))
    assert reason is None
    indices, reason = extract_conflict_window(time, 1, 3, 3)
    assert len(indices) == 0
    assert reason == "incomplete_window_no_extrapolation"


def test_resampling_is_fixed_length_and_preserves_missing_joint_mask():
    time = np.array([0.0, 1.0, 2.0])
    values = np.zeros((3, 2, 3), dtype=float)
    values[:, 0, 0] = [0, 1, 2]
    values[:, 1, 0] = [0, 10, 20]
    mask = np.array([[1, 0], [1, 0], [1, 0]], dtype=bool)
    target = np.linspace(0, 2, 5)
    result, result_mask = resample_fixed_length(time, values, mask, target)
    assert result.shape == (5, 2, 3)
    np.testing.assert_allclose(result[:, 0, 0], target)
    assert result_mask[:, 0].all()
    assert not result_mask[:, 1].any()


def test_lstm_is_deterministic_with_fixed_seed():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 5, 4))
    y = np.array([0, 1] * 4)
    first = NumpyLSTMClassifier(4, 3, 0, seed=42)
    second = NumpyLSTMClassifier(4, 3, 0, seed=42)
    first.fit(x, y, None, epochs=3, learning_rate=.01, l2=0)
    second.fit(x, y, None, epochs=3, learning_rate=.01, l2=0)
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x), atol=1e-12)
