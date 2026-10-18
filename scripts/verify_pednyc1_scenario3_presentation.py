"""Read-only verification helper for the presentation package."""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "outputs/presentation/pednyc1_scenario3_workflow"
d = pd.read_csv(out / "decoded_clean_PedNYC1_scenario3_canonical_copy.csv", sep=";")
d.columns = d.columns.str.strip()
zone = json.loads((root / "outputs/conflict_zone_analysis/scenario_03/conflict_zone_definition.json").read_text())
events = pd.read_csv(root / "outputs/conflict_zone_analysis/scenario_03/conflict_zone_events.csv")
event = events.loc[events.recording_identifier.eq("PedNYC1_scenario3")].iloc[0]
t = pd.to_numeric(d.ScenarioTime)
checks = {}
for actor, x, z, entry, exit_ in (
    ("pedestrian", "B VR Pos X", "B VR Pos Z", "pedestrian_entry_time", "pedestrian_exit_time"),
    ("driver", "A car Pos X", "A car Pos Z", "vehicle_entry_time", "vehicle_exit_time"),
):
    inside = np.hypot(pd.to_numeric(d[x]) - zone["center_x_m"], pd.to_numeric(d[z]) - zone["center_z_m"]) <= zone["radius_m"]
    indices = np.flatnonzero(inside)
    checks[actor] = {
        "trajectory_first_sample_s": float(t.iloc[indices[0]]),
        "trajectory_last_sample_s": float(t.iloc[indices[-1]]),
        "event_entry_s": float(event[entry]),
        "event_exit_s": float(event[exit_]),
    }
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
video = out / "synchronized_vr_encounter_and_timeline.mp4"
for seconds in (0, 13, 25):
    subprocess.run([ffmpeg, "-y", "-ss", str(seconds), "-i", str(video), "-frames:v", "1", str(out / f"qa_video_frame_{seconds:02d}s.png")], check=True, capture_output=True)
print(json.dumps(checks, indent=2))
