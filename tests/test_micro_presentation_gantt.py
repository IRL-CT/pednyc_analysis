import pandas as pd

from src.segmentation.micro_segmentation import presentation_gantt_rows
from src.segmentation.render_micro_gantt_mp4 import PRESENTATION_LABELS, row_layout


def sample_segments():
    return pd.DataFrame({
        "motion_tag": ["pausing", "hesitating", "mixed_motion"],
        "head_tag": ["head_active", "head_still", "head_checking"],
        "car_tag": ["neutral", "yielding", "proceeding"],
    })


def test_static_presentation_filters_and_renames_without_mutating_tags():
    segments = sample_segments()
    original = segments.copy(deep=True)
    _, _, labels, _, _ = presentation_gantt_rows(segments)
    assert labels == [
        "Pause", "Head movement", "Low head movement",
        "Possible hesitation", "Possible yielding",
    ]
    pd.testing.assert_frame_equal(segments, original)


def test_video_layout_omits_empty_and_removed_rows():
    layout = row_layout(sample_segments())
    assert layout["labels"] == [
        "Pause", "Head movement", "Low head movement",
        "Possible hesitation", "Possible yielding",
    ]
    for removed in ("head_checking", "mixed_motion", "neutral", "conflicted", "proceeding"):
        assert removed not in PRESENTATION_LABELS
