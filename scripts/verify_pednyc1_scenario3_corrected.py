"""Read-only checks for the authoritative Scenario 3 presentation package."""
import hashlib, json, subprocess
from pathlib import Path
import imageio_ffmpeg
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
source=root/"data/processed/pednyc1/scenario3/decoded_clean_PedNYC1_scenario3.csv"
out=root/"outputs/presentation/pednyc1_scenario3_audit_corrected"
src=pd.read_csv(source,sep=";"); src.columns=src.columns.str.strip()
derived=pd.read_csv(out/"derived_metrics_from_authoritative.csv"); derived.columns=derived.columns.str.strip()
h=hashlib.sha256(source.read_bytes()).hexdigest().upper()
checks={"source_sha256":h,"source_rows":len(src),"derived_rows":len(derived),"same_scenario_time":bool(np.allclose(src.ScenarioTime,derived.ScenarioTime)),"same_ped_xz":bool(np.allclose(src[["B VR Pos X","B VR Pos Z"]],derived[["B VR Pos X","B VR Pos Z"]])),"same_driver_xz":bool(np.allclose(src[["A car Pos X","A car Pos Z"]],derived[["A car Pos X","A car Pos Z"]]))}
for name,path in (("macro",out/"macro/macro_segments_PedNYC1_scenario3_authoritative.csv"),("micro",out/"micro/v6/micro_segments_descriptive_PedNYC1_scenario3_v6.csv")):
    frame=pd.read_csv(path); checks[f"{name}_source_hash_unique"]=frame.source_sha256.unique().tolist(); checks[f"{name}_time_max"]=float(frame.filter(regex="end_time_sec|time_end_sec").max().max())
ff=imageio_ffmpeg.get_ffmpeg_exe(); video=out/"synchronized_vr_encounter_and_timeline.mp4"
probe=subprocess.run([ff,"-i",str(video)],capture_output=True,text=True).stderr
checks["video_reports_31_seconds"]="Duration: 00:00:31." in probe
for sec in (0,15,30): subprocess.run([ff,"-y","-ss",str(sec),"-i",str(video),"-frames:v","1",str(out/f"qa_video_frame_{sec:02d}s.png")],check=True,capture_output=True)
print(json.dumps(checks,indent=2))
