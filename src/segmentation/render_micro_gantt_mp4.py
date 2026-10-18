"""Render the PedNYC v6 micro-segmentation Gantt as an H.264 MP4.

The renderer uses the exact v6 segment CSV and macro-segment CSV. It is designed
for the top-right quadrant of the four-frame PedNYC verification video.
"""

from pathlib import Path
import argparse
import math
import os
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_DIR = ROOT / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SEGMENTS_CSV = (
    ROOT
    / "outputs"
    / "graphs"
    / "micro_segmentation"
    / "v6"
    / "micro_segments_descriptive_PedNYC1_scenario3_v6.csv"
)
DEFAULT_MACRO_CSV = (
    ROOT
    / "outputs"
    / "graphs"
    / "macro_segmentation"
    / "macro_segments_PedNYC1_scenario3_v2.csv"
)
DEFAULT_OUTPUT_MP4 = (
    ROOT
    / "outputs"
    / "graphs"
    / "micro_segmentation"
    / "v6"
    / "micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4"
)

OBSERVABLE_ROWS = [
    ("motion", "near_stationary"),
    ("motion", "pausing"),
    ("motion", "speed_increasing"),
    ("motion", "speed_decreasing"),
    ("motion", "speed_steady"),
    ("motion", "fluctuating"),
    ("head", "head_active"),
    ("head", "head_still"),
    ("car_context", "neutral"),
]
INFERRED_ROWS = [
    ("motion", "hesitating"),
    ("motion", "mixed_motion"),
    ("head", "head_checking"),
    ("car_context", "yielding"),
    ("car_context", "proceeding"),
    ("car_context", "conflicted"),
]
PRESENTATION_ROWS = {
    "Observed Motion": [
        ("motion", "near_stationary", "Low movement", "#3B82F6"),
        ("motion", "pausing", "Pause", "#60A5FA"),
        ("motion", "speed_increasing", "Speeding up", "#16A34A"),
        ("motion", "speed_decreasing", "Slowing down", "#2563EB"),
        ("motion", "speed_steady", "Steady speed", "#14B8A6"),
        ("motion", "fluctuating", "Variable speed", "#0F766E"),
        ("head", "head_active", "Head movement", "#22C55E"),
        ("head", "head_still", "Low head movement", "#6EE7B7"),
    ],
    "Possible Behaviors": [
        ("motion", "hesitating", "Possible hesitation", "#F97316"),
        ("car_context", "yielding", "Possible yielding", "#9333EA"),
    ],
}
PRESENTATION_FOOTNOTE = "Possible behaviors are rule-based interpretations of measured motion and proximity."
TAG_COLUMNS = {
    "motion": "motion_tag",
    "head": "head_tag",
    "car_context": "car_tag",
}
TAG_COLORS = {
    "near_stationary": "#4A6FA5",
    "pausing": "#8ECAE6",
    "speed_increasing": "#7CB342",
    "speed_decreasing": "#C62828",
    "speed_steady": "#4DB6AC",
    "fluctuating": "#9C6ADE",
    "head_active": "#FBC02D",
    "head_still": "#9E9E9E",
    "neutral": "#E0E0E0",
    "hesitating": "#E65100",
    "mixed_motion": "#795548",
    "head_checking": "#FF9800",
    "yielding": "#AD1457",
    "proceeding": "#81C784",
    "conflicted": "#C0A130",
}
PRESENTATION_LABELS = {tag: label for rows in PRESENTATION_ROWS.values() for _, tag, label, _ in rows}
PRESENTATION_COLORS = {tag: color for rows in PRESENTATION_ROWS.values() for _, tag, _, color in rows}

BACKGROUND = "#0F0F1A"
PLOT_BACKGROUND = "#121421"
OBSERVABLE_BACKGROUND = "#151827"
INFERRED_BACKGROUND = "#1A1522"
INFO_BACKGROUND = "#191B2A"
TEXT = "#F1F5F9"
MUTED_TEXT = "#AEB7C6"
GRID = "#3A3D4D"
PLAYHEAD = "#FF3B4D"
MACRO_LINE = "#C97979"


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the PedNYC v6 behavior Gantt as a compositing-ready H.264 MP4."
    )
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument("--macro-csv", type=Path, default=DEFAULT_MACRO_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MP4)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)
    parser.add_argument("--dpi", type=positive_int, default=100)
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality; lower is higher quality.")
    parser.add_argument("--preset", default="medium", help="libx264 encoding preset.")
    parser.add_argument("--ffmpeg", type=Path, help="Optional explicit path to ffmpeg executable.")
    parser.add_argument(
        "--preview-time",
        type=float,
        help="Render one PNG at this scenario time instead of encoding the MP4.",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        help="PNG path for --preview-time; defaults beside the MP4 output.",
    )
    return parser.parse_args()


def require_columns(df, columns, source_name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def load_inputs(segments_path, macro_path):
    if not segments_path.exists():
        raise FileNotFoundError(f"Micro-segment CSV not found: {segments_path}")
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro-segment CSV not found: {macro_path}")

    # index_col=False is required for the padded legacy Scenario 3 fixture and
    # is harmless for canonical scenario-specific outputs.
    segments = pd.read_csv(segments_path, index_col=False)
    macros = pd.read_csv(macro_path, index_col=False)
    segments.columns = segments.columns.astype(str).str.strip()
    macros.columns = macros.columns.astype(str).str.strip()
    require_columns(
        segments,
        [
            "micro_segment_id",
            "macro_segment_id",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "motion_tag",
            "head_tag",
            "car_tag",
        ],
        "micro-segment CSV",
    )
    require_columns(
        macros,
        ["segment_id", "time_start_sec", "time_end_sec"],
        "macro-segment CSV",
    )
    segments = segments.sort_values(["start_time_sec", "micro_segment_id"]).reset_index(drop=True)
    macros = macros.sort_values("time_start_sec").reset_index(drop=True)
    duration = float(max(segments["end_time_sec"].max(), macros["time_end_sec"].max()))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid scenario duration: {duration}")

    supported = {tag for _, tag in OBSERVABLE_ROWS + INFERRED_ROWS}
    used = (
        set(segments["motion_tag"].astype(str).str.strip())
        | set(segments["head_tag"].astype(str).str.strip())
        | set(segments["car_tag"].astype(str).str.strip())
    )
    unsupported = sorted(used - supported)
    if unsupported:
        raise ValueError(f"No Gantt row configured for tags: {unsupported}")
    return segments, macros, duration


def row_layout(segments):
    row_positions = {}
    positions = []
    labels = []
    y = 0.0
    blocks = []
    for section, candidates in PRESENTATION_ROWS.items():
        rows = [row for row in candidates if row[1] in set(segments[TAG_COLUMNS[row[0]]].astype(str).str.strip())]
        if not rows:
            continue
        start = y
        for dimension, tag, label, _ in rows:
            row_positions[(dimension, tag)] = y
            positions.append(y)
            labels.append(label)
            y += 1.0
        blocks.append((section, start, y - 1.0))
        y += 0.65
    return {
        "lookup": row_positions,
        "positions": positions,
        "labels": labels,
        "blocks": blocks,
        "start": positions[0],
        "end": positions[-1],
    }


def latest_active_row(df, current_time, start_col, end_col):
    active = df[(df[start_col] <= current_time + 1e-9) & (df[end_col] >= current_time - 1e-9)]
    return None if active.empty else active.iloc[-1]


def create_scene(segments, macros, duration, width, height, dpi):
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BACKGROUND)
    ax = fig.add_axes([0.205, 0.26, 0.775, 0.665], facecolor=PLOT_BACKGROUND)
    layout = row_layout(segments)

    for section, start, end in layout["blocks"]:
        color = OBSERVABLE_BACKGROUND if section == "Observed Motion" else INFERRED_BACKGROUND
        ax.axhspan(start - 0.5, end + 0.5, color=color, zorder=-4)
    for i, row in macros.iterrows():
        if i % 2:
            ax.axvspan(
                float(row["time_start_sec"]),
                float(row["time_end_sec"]),
                color="#FFFFFF",
                alpha=0.018,
                zorder=-3,
            )
        start = float(row["time_start_sec"])
        ax.axvline(start, color=MACRO_LINE, linewidth=1.0, linestyle="--", alpha=0.85, zorder=1)
        center = (start + float(row["time_end_sec"])) / 2.0
        ax.text(
            center,
            0.985,
            f"M{int(row['segment_id'])}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#E5A3A3",
            fontsize=8,
            fontweight="bold",
        )
    ax.axvline(duration, color=MACRO_LINE, linewidth=1.0, linestyle="--", alpha=0.85, zorder=1)

    for _, row in segments.iterrows():
        for dimension, column in TAG_COLUMNS.items():
            tag = str(row[column]).strip()
            y = layout["lookup"].get((dimension, tag))
            if y is None:
                continue
            ax.barh(
                y,
                float(row["duration_sec"]),
                left=float(row["start_time_sec"]),
                height=0.56,
                color=PRESENTATION_COLORS[tag],
                edgecolor=BACKGROUND,
                linewidth=0.5,
                alpha=0.96,
                zorder=3,
            )

    for index, (section, start, end) in enumerate(layout["blocks"]):
        if index:
            ax.axhline(start - 0.33, color="#77798A", linewidth=1.2, alpha=0.9, zorder=4)
        ax.text(
            0.008, (start + end) / 2.0, section, transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=10, fontweight="bold", color=TEXT,
            bbox={"facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.78, "pad": 2.2}, zorder=5,
        )

    ax.set_xlim(0.0, duration)
    ax.set_ylim(layout["start"] - 0.55, layout["end"] + 0.55)
    ax.invert_yaxis()
    ax.set_yticks(layout["positions"])
    ax.set_yticklabels(layout["labels"], color=TEXT, fontsize=10)
    ax.set_xlabel("Scenario elapsed time, seconds", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors=MUTED_TEXT, labelsize=8)
    ax.tick_params(axis="y", colors=TEXT, length=0)
    ax.xaxis.grid(True, color=GRID, alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#6A6D7C")
        spine.set_linewidth(0.8)
    ax.set_title(
        "Observed Motion and Possible Behaviors",
        color=TEXT,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.text(0.5, 0.205, PRESENTATION_FOOTNOTE, ha="center", color=MUTED_TEXT, fontsize=8.5)

    info_ax = fig.add_axes([0.02, 0.025, 0.96, 0.15], facecolor=INFO_BACKGROUND, zorder=8)
    info_ax.set_xlim(0.0, 1.0)
    info_ax.set_ylim(0.0, 1.0)
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    for spine in info_ax.spines.values():
        spine.set_color("#3D4155")
        spine.set_linewidth(1.0)
    info_ax.text(
        0.02,
        0.76,
        "ACTIVE / OVERLAPPING FEATURES",
        transform=info_ax.transAxes,
        color="#8FA3BF",
        fontsize=7.5,
        fontweight="bold",
        zorder=9,
    )
    info_text = info_ax.text(
        0.02,
        0.43,
        "",
        transform=info_ax.transAxes,
        color=TEXT,
        fontsize=10.2,
        fontweight="bold",
        zorder=9,
    )
    time_text = info_ax.text(
        0.98,
        0.14,
        "",
        transform=info_ax.transAxes,
        ha="right",
        color=MUTED_TEXT,
        fontsize=9.2,
        family="monospace",
        zorder=9,
    )

    playhead = ax.axvline(0.0, color=PLAYHEAD, linewidth=2.1, alpha=1.0, zorder=10)
    tooltip = ax.annotate(
        "0.000 s",
        xy=(0.35, 0.985),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        color="#FFFFFF",
        fontsize=8.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": PLAYHEAD, "edgecolor": "#FFB0B8"},
        zorder=11,
        annotation_clip=False,
    )
    return fig, ax, playhead, tooltip, info_text, time_text


def update_scene(current_time, segments, macros, duration, playhead, tooltip, info_text, time_text):
    current_time = max(0.0, min(float(current_time), duration))
    playhead.set_xdata([current_time, current_time])
    tooltip_x = min(max(current_time, 0.42), max(0.42, duration - 0.42))
    tooltip.xy = (tooltip_x, 0.985)
    tooltip.set_position((tooltip_x, 0.985))
    tooltip.set_text(f"{current_time:.3f} s")

    segment = latest_active_row(segments, current_time, "start_time_sec", "end_time_sec")
    macro = latest_active_row(macros, current_time, "time_start_sec", "time_end_sec")
    macro_id = int(macro["segment_id"]) if macro is not None else None
    if segment is None:
        prefix = f"M{macro_id}" if macro_id is not None else "Outside macro"
        info = f"{prefix} \u00b7 no assigned micro-segment at this timestamp"
    else:
        if macro_id is None:
            macro_id = int(segment["macro_segment_id"])
        labels = []
        for column in TAG_COLUMNS.values():
            tag = str(segment[column]).strip()
            if tag in PRESENTATION_LABELS:
                labels.append(PRESENTATION_LABELS[tag])
        detail = " \u00b7 ".join(labels) if labels else "No displayed presentation label"
        info = f"M{macro_id} \u00b7 segment {int(segment['micro_segment_id'])} \u2014 {detail}"
    info_text.set_text(info)
    time_text.set_text(f"{current_time:.3f} / {duration:.3f} s")
    return playhead, tooltip, info_text, time_text


def resolve_ffmpeg(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for variable in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]))
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    try:
        import imageio_ffmpeg

        packaged_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if packaged_ffmpeg.is_file():
            return packaged_ffmpeg.resolve()
    except (AttributeError, ImportError, OSError, RuntimeError):
        pass
    raise FileNotFoundError(
        "ffmpeg was not found. Install requirements.txt, put ffmpeg on PATH, "
        "or pass --ffmpeg C:\\path\\to\\ffmpeg.exe."
    )


def render_preview(args, segments, macros, duration):
    preview_time = max(0.0, min(float(args.preview_time), duration))
    preview_path = args.preview_output or args.output.with_name(args.output.stem + f"_{preview_time:.3f}s.png")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _, playhead, tooltip, info_text, time_text = create_scene(
        segments, macros, duration, args.width, args.height, args.dpi
    )
    update_scene(preview_time, segments, macros, duration, playhead, tooltip, info_text, time_text)
    fig.savefig(preview_path, dpi=args.dpi, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved {args.width}x{args.height} preview: {preview_path}")


def render_mp4(args, segments, macros, duration):
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, _, playhead, tooltip, info_text, time_text = create_scene(
        segments, macros, duration, args.width, args.height, args.dpi
    )
    frame_count = math.ceil(duration * args.fps)

    def update_frame(frame_index):
        current_time = duration if frame_index == frame_count - 1 else frame_index / args.fps
        return update_scene(
            current_time,
            segments,
            macros,
            duration,
            playhead,
            tooltip,
            info_text,
            time_text,
        )

    movie = animation.FuncAnimation(
        fig,
        update_frame,
        frames=frame_count,
        interval=1000.0 / args.fps,
        blit=True,
        repeat=False,
    )
    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        metadata={"title": "PedNYC micro-segmentation Gantt", "artist": "PedNYC"},
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-movflags",
            "+faststart",
        ],
    )

    def progress(frame_index, total_frames):
        if frame_index == 0 or (frame_index + 1) % (args.fps * 5) == 0 or frame_index + 1 == total_frames:
            print(f"Encoding frame {frame_index + 1}/{total_frames}", flush=True)

    movie.save(args.output, writer=writer, dpi=args.dpi, progress_callback=progress)
    plt.close(fig)
    print(f"Saved MP4: {args.output}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frame rate: {args.fps} fps")
    print(f"Frames: {frame_count}")
    print(f"Scenario endpoint: {duration:.6f} s")
    print(f"Container duration: {frame_count / args.fps:.6f} s")
    print(f"ffmpeg: {ffmpeg}")


def main():
    args = parse_args()
    segments, macros, duration = load_inputs(args.segments_csv, args.macro_csv)
    if args.preview_time is not None:
        render_preview(args, segments, macros, duration)
    else:
        render_mp4(args, segments, macros, duration)


if __name__ == "__main__":
    main()
