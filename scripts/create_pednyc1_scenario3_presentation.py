"""Create presentation-only PedNYC1 Scenario 3 workflow artifacts.

This exporter reads canonical outputs and writes only below a new presentation
folder.  It does not modify analytical CSV labels, boundaries, or thresholds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "presentation" / "pednyc1_scenario3_audit_corrected"
SOURCE = ROOT / "data" / "processed" / "pednyc1" / "scenario3" / "decoded_clean_PedNYC1_scenario3.csv"
DECODED = OUT / "derived_metrics_from_authoritative.csv"
MACRO = OUT / "macro" / "macro_segments_PedNYC1_scenario3_authoritative.csv"
MICRO = OUT / "micro" / "v6" / "micro_segments_descriptive_PedNYC1_scenario3_v6.csv"
ZONE = ROOT / "outputs" / "conflict_zone_analysis" / "scenario_03" / "conflict_zone_definition.json"
EVENTS = ROOT / "outputs" / "conflict_zone_analysis" / "scenario_03" / "conflict_zone_events.csv"

SMOOTH_SPEED = 9
SMOOTH_ACCEL = 11
NEAR_DISTANCE = 12.0
SPEED_LOW = 0.15
WALKING = 0.25
SPEEDING_ACCEL = 0.35
SLOWING_ACCEL = -0.25
HEAD_MEAN = 25.0
HEAD_RATE = 50.0
HEAD_YAW_DIFF = 20.0

COLORS = {"ped": "#2563EB", "ped2": "#16A34A", "driver": "#DC2626", "driver2": "#7C3AED", "interaction": "#EAB308", "inferred": "#F97316"}


def read_decoded() -> pd.DataFrame:
    df = pd.read_csv(DECODED, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce")


def find_col(df: pd.DataFrame, name: str) -> str:
    normalized = {" ".join(str(c).split()): c for c in df.columns}
    key = " ".join(name.split())
    if key not in normalized:
        raise KeyError(name)
    return normalized[key]


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["time"] = num(df, find_col(df, "ScenarioTime"))
    out["time"] -= out["time"].iloc[0]
    dt = out.time.diff().where(lambda x: x > 0)
    dt = dt.fillna(dt.median())
    for actor, prefix in (("ped", "B VR"), ("driver", "A car")):
        out[f"{actor}_x"] = num(df, find_col(df, f"{prefix} Pos X"))
        out[f"{actor}_z"] = num(df, find_col(df, f"{prefix} Pos Z"))
        out[f"{actor}_yaw"] = num(df, find_col(df, f"{prefix} Rot Y"))
        raw_speed = np.hypot(out[f"{actor}_x"].diff(), out[f"{actor}_z"].diff()).div(dt).replace([np.inf, -np.inf], np.nan).fillna(0)
        out[f"{actor}_speed"] = raw_speed.rolling(SMOOTH_SPEED, center=True, min_periods=1).mean()
        raw_accel = out[f"{actor}_speed"].diff().div(dt).replace([np.inf, -np.inf], np.nan).fillna(0)
        out[f"{actor}_accel"] = raw_accel.rolling(SMOOTH_ACCEL, center=True, min_periods=1).mean()
    out["distance"] = np.hypot(out.driver_x - out.ped_x, out.driver_z - out.ped_z).rolling(SMOOTH_SPEED, center=True, min_periods=1).mean()
    head = num(df, find_col(df, "B Bone Rot Head Y"))
    out["head_yaw"] = head
    unwrapped = np.unwrap(np.deg2rad(head.interpolate(limit_direction="both")))
    out["head_rate"] = pd.Series(np.rad2deg(pd.Series(unwrapped).diff().div(dt)), index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["head_body_diff"] = ((head - out.ped_yaw + 180) % 360) - 180
    for side in ("Left", "Right"):
        hand = num(df, find_col(df, f"B Bone Pos {side}Hand Y"))
        shoulder = num(df, find_col(df, f"B Bone Pos {side}Shoulder Y"))
        out[f"{side.lower()}_hand_delta"] = hand - shoulder
        out[f"{side.lower()}_hand_up"] = hand > shoulder
    indicator_col = find_col(df, "A indicators")
    out["driver_signal_raw"] = df[indicator_col].astype("string").str.strip()
    out["driver_signal"] = out.driver_signal_raw.map({"LeftFalse_RightFalse":"No signal","LeftTrue_RightFalse":"Left signal","LeftFalse_RightTrue":"Right signal"}).fillna("Unknown")
    return out


def contiguous_intervals(time: pd.Series, mask: pd.Series) -> list[tuple[float, float]]:
    m = mask.fillna(False).to_numpy(bool)
    t = time.to_numpy(float)
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False])
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def zone_event_from_trajectory(m: pd.DataFrame, zone: dict) -> pd.Series:
    """Apply the existing circle definition to the canonical synchronized rows."""
    values = {}
    for actor, prefix in (("pedestrian", "ped"), ("vehicle", "driver")):
        inside = np.hypot(m[f"{prefix}_x"] - zone["center_x_m"], m[f"{prefix}_z"] - zone["center_z_m"]) <= zone["radius_m"]
        indices = np.flatnonzero(inside.to_numpy(bool))
        if not len(indices):
            values[f"{actor}_entry_time"] = np.nan
            values[f"{actor}_exit_time"] = np.nan
        else:
            values[f"{actor}_entry_time"] = float(m.time.iloc[indices[0]])
            values[f"{actor}_exit_time"] = float(m.time.iloc[indices[-1]])
    return pd.Series(values)


def style_axes(axes, duration):
    for ax in np.ravel(axes):
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=.22, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)


def add_zone_marks(ax, event):
    ax.axvspan(event.pedestrian_entry_time, event.pedestrian_exit_time, color=COLORS["ped"], alpha=.10)
    ax.axvline(event.pedestrian_entry_time, color=COLORS["ped"], lw=1.2, ls="--")
    ax.axvline(event.pedestrian_exit_time, color=COLORS["ped"], lw=1.2, ls=":")
    ax.axvspan(event.vehicle_entry_time, event.vehicle_exit_time, color=COLORS["driver"], alpha=.10)
    ax.axvline(event.vehicle_entry_time, color=COLORS["driver"], lw=1.2, ls="--")
    ax.axvline(event.vehicle_exit_time, color=COLORS["driver"], lw=1.2, ls=":")


def pedestrian_figure(m, event):
    fig, axes = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
    fig.suptitle("Pedestrian Motion and Interaction Features", fontsize=24, weight="bold", y=.995)
    axes[0].plot(m.time, m.ped_speed, color=COLORS["ped"], lw=2); axes[0].set_ylabel("Speed\n(m/s)")
    axes[1].plot(m.time, m.ped_accel, color=COLORS["ped2"], lw=1.8); axes[1].axhline(0,color="#6B7280",lw=.8); axes[1].set_ylabel("Acceleration\n(m/s²)")
    axes[2].plot(m.time, m.ped_yaw, color="#0F766E", lw=1.8); axes[2].set_ylabel("Body yaw\n(degrees)")
    axes[3].plot(m.time, m.head_rate, color="#059669", lw=1.4, label="Head-turn rate")
    axes[3].plot(m.time, m.head_body_diff, color="#2563EB", lw=1.2, alpha=.8, label="Head–body yaw difference")
    axes[3].set_ylabel("Head activity\n(deg/s; deg)"); axes[3].legend(loc="upper right", ncol=2)
    axes[4].plot(m.time, m.distance, color="#B45309", lw=2, label="Planar distance")
    axes[4].axhline(NEAR_DISTANCE, color=COLORS["interaction"], ls="--", lw=1.8, label="Near-driver threshold: 12 m")
    axes[4].set_ylabel("Distance\n(m)"); axes[4].legend(loc="upper right", ncol=2)
    axes[5].broken_barh([(event.pedestrian_entry_time,event.pedestrian_exit_time-event.pedestrian_entry_time)],(1.1,.7),facecolor=COLORS["ped"])
    axes[5].broken_barh([(event.vehicle_entry_time,event.vehicle_exit_time-event.vehicle_entry_time)],(.1,.7),facecolor=COLORS["driver"])
    axes[5].set_yticks([.45,1.45],["Driver", "Pedestrian"]); axes[5].set_ylabel("Conflict zone")
    axes[6].plot(m.time, m.left_hand_delta, color="#2563EB", label="Left hand − left shoulder")
    axes[6].plot(m.time, m.right_hand_delta, color="#16A34A", label="Right hand − right shoulder")
    axes[6].axhline(0,color="#111827",lw=1,ls="--",label="Hand above shoulder when > 0")
    for side,color in (("left", "#2563EB"),("right", "#16A34A")):
        axes[6].fill_between(m.time,0,m[f"{side}_hand_delta"],where=m[f"{side}_hand_up"],color=color,alpha=.18)
    axes[6].set_ylabel("Vertical difference\n(m)"); axes[6].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Scenario elapsed time (s)")
    for ax in axes[:5]: add_zone_marks(ax,event)
    style_axes(axes,float(m.time.max()))
    fig.text(.01,.006,f"Source SHA-256 {source_hash()} | Speed: X/Z VR-position displacement ÷ ScenarioTime Δt, centered 9-sample mean. Acceleration: derivative of smoothed speed, centered 11-sample mean. Hand candidates: raw Y(hand) > Y(shoulder).",fontsize=8)
    fig.tight_layout(rect=[0,.025,1,.975])
    fig.savefig(OUT/"pedestrian_motion_and_interaction_features.png",dpi=300,facecolor="white")
    plt.close(fig)


def trajectory_figure(m, zone):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(m.ped_x, m.ped_z, color=COLORS["ped"], lw=2.2, label="Pedestrian")
    ax.plot(m.driver_x, m.driver_z, color=COLORS["driver"], lw=2.2, label="Driver")
    for x, z, color, marker, label in (
        (m.ped_x.iloc[0], m.ped_z.iloc[0], COLORS["ped"], "o", "Pedestrian start"),
        (m.ped_x.iloc[-1], m.ped_z.iloc[-1], COLORS["ped"], "X", "Pedestrian end"),
        (m.driver_x.iloc[0], m.driver_z.iloc[0], COLORS["driver"], "s", "Driver start"),
        (m.driver_x.iloc[-1], m.driver_z.iloc[-1], COLORS["driver"], "X", "Driver end"),
    ):
        ax.scatter(x, z, s=85, color=color, marker=marker, zorder=5, label=label)
    stride = 80
    for actor, color in (("ped", COLORS["ped"]), ("driver", COLORS["driver"])):
        x=m[f"{actor}_x"].to_numpy(); z=m[f"{actor}_z"].to_numpy()
        for i in range(stride, len(m)-1, stride):
            ax.annotate("", xy=(x[i+1],z[i+1]), xytext=(x[i-1],z[i-1]), arrowprops={"arrowstyle":"->","color":color,"lw":1.6})
            ax.text(x[i],z[i],f"{m.time.iloc[i]:.0f}s",fontsize=8,color=color)
    ax.add_patch(plt.Circle((zone["center_x_m"],zone["center_z_m"]),zone["radius_m"],color=COLORS["interaction"],alpha=.16,label="Existing conflict zone"))
    ax.set_aspect("equal",adjustable="box"); ax.set_xlabel("World X position (m)"); ax.set_ylabel("World Z position (m)")
    ax.set_title("Verified Pedestrian–Driver Trajectories",fontsize=22,weight="bold")
    ax.grid(True,alpha=.25); ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,-.08),frameon=False)
    fig.text(.01,.01,f"Source: {SOURCE} | SHA-256 {source_hash()}",fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(OUT/"verified_pedestrian_driver_trajectories.png",dpi=300,facecolor="white"); plt.close(fig)


def source_hash():
    h=hashlib.sha256()
    with SOURCE.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest().upper()


def driver_figure(m,event):
    fig, axes = plt.subplots(5,1,figsize=(16,14),sharex=True)
    fig.suptitle("Driver Motion and Signaling Features",fontsize=24,weight="bold",y=.995)
    axes[0].plot(m.time,m.driver_speed,color=COLORS["driver"],lw=2); axes[0].set_ylabel("Speed\n(m/s)")
    axes[1].plot(m.time,m.driver_accel,color=COLORS["driver2"],lw=1.8); axes[1].axhline(0,color="#6B7280",lw=.8); axes[1].set_ylabel("Acceleration\n(m/s²)")
    axes[2].plot(m.time,m.driver_yaw,color="#7C2D12",lw=1.8); axes[2].set_ylabel("Yaw\n(degrees)")
    signal_codes=m.driver_signal.map({"No signal":0,"Left signal":1,"Right signal":2,"Unknown":np.nan})
    axes[3].step(m.time,signal_codes,where="post",color="#7C3AED",lw=2)
    axes[3].set_yticks([0,1,2],["No signal","Left signal","Right signal"]); axes[3].set_ylabel("Signal\n(state)")
    axes[4].broken_barh([(event.vehicle_entry_time,event.vehicle_exit_time-event.vehicle_entry_time)],(.2,.6),facecolor=COLORS["driver"])
    axes[4].set_yticks([.5],["Driver inside zone"]); axes[4].set_ylabel("Conflict zone")
    axes[-1].set_xlabel("Scenario elapsed time (s)")
    for ax in axes[:4]: add_zone_marks(ax,event)
    style_axes(axes,float(m.time.max()))
    fig.text(.01,.006,f"Source SHA-256 {source_hash()} | Speed: driver X/Z displacement ÷ ScenarioTime Δt, centered 9-sample mean. Acceleration: derivative of smoothed speed, centered 11-sample mean. Signal strings: A indicators.",fontsize=8)
    fig.tight_layout(rect=[0,.025,1,.97])
    fig.savefig(OUT/"driver_motion_and_signaling_features.png",dpi=300,facecolor="white")
    plt.close(fig)


def gantt_data(m,event,micro):
    rows=[]
    def add(label,actor,status,intervals,color):
        if intervals: rows.append((actor,label,status,intervals,color))
    mapping={"near_stationary":"Low movement","pausing":"Pause","speed_increasing":"Speeding up","speed_decreasing":"Slowing down","speed_steady":"Steady speed","fluctuating":"Variable speed","head_active":"Head movement","head_still":"Low head movement"}
    for tag,label in mapping.items():
        col="head_tag" if tag.startswith("head_") else "motion_tag"
        intervals=[(float(r.start_time_sec),float(r.end_time_sec)) for r in micro.itertuples() if getattr(r,col)==tag]
        add(label,"Pedestrian","Measured",intervals,COLORS["ped2"] if "Head" in label or "head" in label.lower() else COLORS["ped"])
    add("Left hand above shoulder","Pedestrian","Measured",contiguous_intervals(m.time,m.left_hand_up),"#1D4ED8")
    add("Right hand above shoulder","Pedestrian","Measured",contiguous_intervals(m.time,m.right_hand_up),"#059669")
    add("Inside conflict zone","Pedestrian","Interaction",[(event.pedestrian_entry_time,event.pedestrian_exit_time)],COLORS["interaction"])
    add("Speeding up","Driver","Measured",contiguous_intervals(m.time,m.driver_accel>=SPEEDING_ACCEL),"#DC2626")
    add("Slowing down","Driver","Measured",contiguous_intervals(m.time,m.driver_accel<=SLOWING_ACCEL),"#7C3AED")
    add("Steady speed","Driver","Measured",contiguous_intervals(m.time,(m.driver_accel>SLOWING_ACCEL)&(m.driver_accel<SPEEDING_ACCEL)&(m.driver_speed>=SPEED_LOW)),"#BE123C")
    add("Left signal","Driver","Measured",contiguous_intervals(m.time,m.driver_signal.eq("Left signal")),"#7C3AED")
    add("Right signal","Driver","Measured",contiguous_intervals(m.time,m.driver_signal.eq("Right signal")),"#A21CAF")
    add("Inside conflict zone","Driver","Interaction",[(event.vehicle_entry_time,event.vehicle_exit_time)],COLORS["interaction"])
    add("Driver nearby (<12 m)","Interaction","Interaction",contiguous_intervals(m.time,m.distance<NEAR_DISTANCE),COLORS["interaction"])
    for tag,label in (("hesitating","Speed dip with recovery"),("yielding","Possible yielding")):
        col="motion_tag" if tag=="hesitating" else "car_tag"
        intervals=[(float(r.start_time_sec),float(r.end_time_sec)) for r in micro.itertuples() if getattr(r,col)==tag]
        add(label,"Interaction","Inferred",intervals,COLORS["inferred"])
    return rows


def draw_gantt(ax,rows,macros,duration,playhead=None,compact_macros=False):
    phase_names={"pre_movement_low_motion":"pre-movement low motion","acceleration":"acceleration","crossing_near_car":"crossing near driver","deceleration":"deceleration","post_movement_low_motion":"post-movement low motion"}
    for r in macros.itertuples():
        ax.axvspan(r.time_start_sec,r.time_end_sec,color="#E5E7EB" if r.segment_id%2==0 else "#F9FAFB",zorder=-3)
        caption = f"M{r.segment_id}" if compact_macros else f"M{r.segment_id}: {phase_names.get(r.macro_label,r.macro_label)}"
        ax.text((r.time_start_sec+r.time_end_sec)/2,1.012,caption,transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=8.5)
    labels=[]
    for y,(actor,label,status,intervals,color) in enumerate(rows):
        labels.append(f"{actor} — {label}")
        for start,end in intervals: ax.barh(y,max(end-start,.02),left=start,height=.62,color=color,edgecolor="white",lw=.5)
    ax.set_yticks(range(len(rows)),labels,fontsize=10); ax.invert_yaxis(); ax.set_xlim(0,duration); ax.set_xlabel("Scenario elapsed time (s)"); ax.grid(True,axis="x",alpha=.2)
    if playhead is not None: return ax.axvline(playhead,color="#111827",lw=2.2,zorder=10)


def gantt_figure(rows,macros,duration):
    fig,ax=plt.subplots(figsize=(18,max(10,.52*len(rows)+3)))
    draw_gantt(ax,rows,macros,duration)
    fig.suptitle("Driver–Pedestrian Interaction Timeline",fontsize=24,weight="bold")
    legend=[Patch(color=COLORS["ped"],label="Pedestrian measured"),Patch(color=COLORS["driver"],label="Driver measured"),Patch(color=COLORS["interaction"],label="Interaction / proximity"),Patch(color=COLORS["inferred"],label="Possible interpretation")]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.965),ncol=4,frameon=False)
    fig.text(.01,.008,f"Possible behaviors are rule-based interpretations of measured motion and proximity and have not yet been validated as intent. | Source SHA-256 {source_hash()}",fontsize=9,style="italic")
    fig.tight_layout(rect=[0,.035,1,.93]); fig.savefig(OUT/"driver_pedestrian_interaction_timeline.png",dpi=300,facecolor="white"); plt.close(fig)


def render_video(m,rows,macros,zone,duration):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    plt.rcParams["animation.ffmpeg_path"]=ffmpeg
    fig=plt.figure(figsize=(16,9),dpi=100)
    gs=fig.add_gridspec(2,1,height_ratios=[1,1.25],hspace=.27)
    ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1])
    ax0.plot(m.ped_x,m.ped_z,color="#93C5FD",lw=2,label="Pedestrian path"); ax0.plot(m.driver_x,m.driver_z,color="#FCA5A5",lw=2,label="Driver path")
    circle=plt.Circle((zone["center_x_m"],zone["center_z_m"]),zone["radius_m"],color=COLORS["interaction"],alpha=.18,label="Conflict zone"); ax0.add_patch(circle)
    peddot,=ax0.plot([],[],"o",color=COLORS["ped"],ms=9); cardot,=ax0.plot([],[],"s",color=COLORS["driver"],ms=9)
    ax0.set_aspect("equal",adjustable="datalim"); ax0.set_xlabel("World X (m)"); ax0.set_ylabel("World Z (m)"); ax0.set_title("Authoritative VR trajectories (31.003 s recording)",loc="left",weight="bold"); ax0.legend(loc="upper right",ncol=3)
    draw_gantt(ax1,rows,macros,duration,compact_macros=True); line=ax1.axvline(0,color="#111827",lw=2.2,zorder=10)
    title=fig.suptitle("Driver–Pedestrian Interaction Timeline — 0.0 s",fontsize=17,weight="bold")
    fig.text(.99,.01,f"Source SHA-256 {source_hash()}",ha="right",fontsize=7)
    fig.subplots_adjust(left=.23,right=.98,bottom=.08,top=.91)
    frame_times=np.arange(0,duration,1/15)
    def update(i):
        t=frame_times[i]; idx=int(np.argmin(np.abs(m.time.to_numpy()-t)))
        ped_span=m.iloc[:idx+1]; peddot.set_data([m.ped_x.iloc[idx]],[m.ped_z.iloc[idx]]); cardot.set_data([m.driver_x.iloc[idx]],[m.driver_z.iloc[idx]])
        line.set_xdata([t,t]); title.set_text(f"Driver–Pedestrian Interaction Timeline — {t:0.1f} s")
        return peddot,cardot,line,title
    anim=FuncAnimation(fig,update,frames=len(frame_times),interval=1000/15,blit=False)
    path=OUT/"synchronized_vr_encounter_and_timeline.mp4"
    anim.save(path,writer=FFMpegWriter(fps=15,codec="libx264",bitrate=3500,extra_args=["-pix_fmt","yuv420p"])); plt.close(fig); return path


def write_docs_legacy(m,event,macros,micro,zone,video_path):
    source_cols={
        "Time":"ScenarioTime", "Pedestrian position":"B VR Pos X/Y/Z", "Pedestrian orientation":"B VR Rot X/Y/Z",
        "Head orientation":"B Bone Rot Head X/Y/Z", "Hands":"B Bone Pos LeftHand X/Y/Z; B Bone Pos RightHand X/Y/Z",
        "Shoulders":"B Bone Pos LeftShoulder X/Y/Z; B Bone Pos RightShoulder X/Y/Z", "Driver position":"A car Pos X/Y/Z",
        "Driver orientation":"A car Rot X/Y/Z", "Driver speed":"derived planar driver speed (A velocity is also recorded)",
        "Driver acceleration":"derived planar acceleration; A accel is recorded control input", "Driver signaling":"A indicators (all missing)"}
    methods=f"""# PedNYC1 Scenario 3 presentation workflow

## Verified input

- Exact raw input: `{RAW.relative_to(ROOT)}`
- Exact processed input: `{DECODED.relative_to(ROOT)}` ({len(m)} rows)
- Common time axis: ScenarioTime minus first sample, 0.000000–{m.time.max():.6f} s.
- Macro output: `{MACRO.relative_to(ROOT)}` ({len(macros)} unchanged segments).
- Micro output: `{MICRO.relative_to(ROOT)}` ({len(micro)} unchanged segments).
- Conflict-zone definition: `{ZONE.relative_to(ROOT)}`; circle centered at ({zone['center_x_m']:.6f}, {zone['center_z_m']:.6f}) m, radius {zone['radius_m']:.1f} m, robust median of recording-level observed path intersections. Occupancy is recomputed from this unchanged definition and the canonical synchronized rows because the existing PedNYC1 event row does not match this recording.

## Input columns

""" + "\n".join(f"- **{k}:** {v}" for k,v in source_cols.items()) + f"""

## Workflow and methods

Pedestrian motion uses the validated `B VR Pos X/Z` coordinates, never avatar position. Pedestrian and driver planar speeds are consecutive X/Z displacement divided by positive ScenarioTime Δt and then a centered {SMOOTH_SPEED}-sample rolling mean. Acceleration is the derivative of that smoothed speed and then a centered {SMOOTH_ACCEL}-sample rolling mean. Distance is synchronized planar Euclidean distance and uses the same {SMOOTH_SPEED}-sample mean. Head-turn rate unwraps head yaw before differentiation; head/body difference is the wrapped head-yaw minus VR-body-yaw angle.

Candidate hand raises are the raw conditions `B Bone Pos LeftHand Y > B Bone Pos LeftShoulder Y` and `B Bone Pos RightHand Y > B Bone Pos RightShoulder Y`. No smoothing or minimum duration was added, so brief tracking noise may appear. These are candidates, not confirmed gestures.

The 12 m proximity threshold, pedestrian motion/head thresholds, speed-dip rule, and possible-yielding rule are reused from microsegmentation v6. `Possible yielding` means a microsegment is within 12 m and its pedestrian motion tag is slowing, pausing, or speed dip with recovery; it does not establish intent. Driver measured-state rows reuse configured acceleration thresholds: ≥ +0.35 m/s² speeding up, ≤ −0.25 m/s² slowing down, otherwise steady while speed is ≥0.15 m/s. No extra duration filter is applied.

Pedestrian conflict-zone occupancy: {event.pedestrian_entry_time:.6f}–{event.pedestrian_exit_time:.6f} s. Driver occupancy: {event.vehicle_entry_time:.6f}–{event.vehicle_exit_time:.6f} s. These are the first-to-last sampled circle-membership times on the canonical trajectories.

## Limitations

- `A indicators` is missing for all {len(m)} canonical samples. No numeric-to-direction mapping exists to verify, so no No/Left/Right labels were assigned and signal rows are omitted from the Gantt.
- The existing `outputs/conflict_zone_analysis/scenario_03/conflict_zone_events.csv` PedNYC1 row reports pedestrian 15.366–18.316 s and driver 23.033–23.846 s, but direct circle membership for the canonical trajectories is pedestrian 19.697–22.869 s and driver 16.413–17.056 s. The presentation uses the latter, recomputed with the unchanged existing circle equation.
- No original screen-capture video matching the canonical 16:58:19 recording was located. The MP4 therefore animates the correct recorded VR pedestrian/driver trajectories beside the synchronized Gantt; it does not substitute the older 13:48:59 capture.
- The canonical report records a failed regression against older Scenario 3 ground truth (boundaries and labels differed). This presentation faithfully uses the current canonical macro/micro files and does not claim equivalence to the older run.

## Synchronization checks

- All figures and the MP4 use 0.000000–{m.time.max():.6f} s.
- Macro boundaries were read, not recomputed or changed.
- Micro intervals were read, not renamed in analytical CSVs; presentation-only labels are applied at render time.
- Conflict entries/exits were recomputed with the unchanged validated circle definition and checked against sample-level canonical trajectories; the mismatched existing event row is retained untouched and reported above.
"""
    (OUT/"methods_summary.md").write_text(methods,encoding="utf-8")

    definitions=[
        ["Pedestrian speed","Pedestrian","B VR Pos X/Z; ScenarioTime",f"planar displacement/Δt; centered {SMOOTH_SPEED}-sample mean","Observed"],
        ["Pedestrian acceleration","Pedestrian","smoothed pedestrian speed; ScenarioTime",f"Δspeed/Δt; centered {SMOOTH_ACCEL}-sample mean","Observed"],
        ["Head movement","Pedestrian","head-turn rate; head/body yaw difference",f"micro v6: mean |rate| ≥25 deg/s OR max |rate| ≥50 deg/s OR max |yaw difference| ≥20 deg","Observed"],
        ["Low head movement","Pedestrian","same", "does not meet head-checking/head-active thresholds","Observed"],
        ["Left hand above shoulder","Pedestrian","left-hand Y; left-shoulder Y","hand Y > corresponding shoulder Y; raw; no minimum duration","Observed candidate"],
        ["Right hand above shoulder","Pedestrian","right-hand Y; right-shoulder Y","hand Y > corresponding shoulder Y; raw; no minimum duration","Observed candidate"],
        ["Driver nearby","Interaction","smoothed planar distance","distance < 12.0 m within microsegment/current sample","Observed"],
        ["Pedestrian inside conflict zone","Pedestrian","B VR Pos X/Z; scenario circle","distance to (11.846832, 58.392967) m ≤ 2.0 m; first contiguous visit","Observed"],
        ["Driver inside conflict zone","Driver","A car Pos X/Z; scenario circle","distance to (11.846832, 58.392967) m ≤ 2.0 m; first contiguous visit","Observed"],
        ["Driver speeding up","Driver","smoothed driver acceleration","acceleration ≥ +0.35 m/s²; raw condition; no duration filter","Observed"],
        ["Driver slowing down","Driver","smoothed driver acceleration","acceleration ≤ −0.25 m/s²; raw condition; no duration filter","Observed"],
        ["Driver steady speed","Driver","smoothed driver speed and acceleration","−0.25 < acceleration < +0.35 m/s² and speed ≥0.15 m/s; no duration filter","Observed"],
        ["Speed dip with recovery","Interaction","micro speed drop/recovery",">0.18 m/s and >22% drop; recovery ≥35% of drop; duration ≤1.5 s; mean speed >0.25 m/s","Inferred"],
        ["Possible yielding","Interaction","distance + pedestrian micro motion","minimum distance <12 m AND pedestrian slowing, pausing, or speed dip with recovery","Inferred; not intent"],
        ["Driver signaling","Driver","A indicators","unavailable: field missing for every canonical sample; no encoding assigned","Missing"],
    ]
    with (OUT/"definitions_table.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["Display label","Actor","Input features","Exact rule or threshold","Observed versus inferred status"]); w.writerows(definitions)
    verification={"time_origin_seconds":0.0,"duration_seconds":float(m.time.max()),"rows":len(m),"macro_boundaries_unchanged":True,"analytical_csvs_modified":False,"signal_available":False,"video_kind":"canonical VR trajectory animation" if video_path else "not rendered","pedestrian_zone_event":[event.pedestrian_entry_time,event.pedestrian_exit_time],"driver_zone_event":[event.vehicle_entry_time,event.vehicle_exit_time]}
    (OUT/"verification_report.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")


def write_docs(m,event,macros,micro,zone,video_path):
    methods=f"""# PedNYC1 Scenario 3 corrected presentation workflow

## Source and synchronization

- Single authoritative input: `{SOURCE}`
- SHA-256: `{source_hash()}`
- Source shape: 648 rows x 1,006 columns.
- Common time axis: `ScenarioTime - ScenarioTime[0]`, 0.000000 to {m.time.max():.6f} s.
- No source rows were removed. One shared dataframe supplies both actors and every signal.
- Derived metrics: `{DECODED.relative_to(ROOT)}`. This is a derivative, not an independent source.
- Macro output: `{MACRO.relative_to(ROOT)}` ({len(macros)} regenerated segments).
- Micro output: `{MICRO.relative_to(ROOT)}` ({len(micro)} regenerated segments).

## Authoritative columns

- Pedestrian position: `B VR Pos X`, `B VR Pos Z`; body yaw: `B VR Rot Y`.
- Head orientation: `B  Bone Rot Head X/Y/Z`.
- Hands: `B  Bone Pos LeftHand X/Y/Z`, `B  Bone Pos RightHand X/Y/Z`.
- Shoulders: `B  Bone Pos LeftShoulder X/Y/Z`, `B  Bone Pos RightShoulder X/Y/Z`.
- Driver position: `A car Pos X`, `A car Pos Z`; yaw: `A car Rot Y`; signaling: `A indicators`.

## Calculations

Planar speed is consecutive X/Z displacement divided by positive ScenarioTime delta, followed by a centered {SMOOTH_SPEED}-sample rolling mean. Acceleration is the derivative of smoothed speed, followed by a centered {SMOOTH_ACCEL}-sample mean. Distance is synchronized planar Euclidean distance with the same {SMOOTH_SPEED}-sample mean. Head-turn rate unwraps head yaw before differentiation; head/body difference is wrapped to [-180, 180] degrees.

Candidate hand raises are raw `hand Y > corresponding shoulder Y` conditions with no smoothing or minimum duration. Signaling is decoded directly: `LeftFalse_RightFalse` = No signal, `LeftTrue_RightFalse` = Left signal, `LeftFalse_RightTrue` = Right signal.

The existing conflict-zone circle is centered at ({zone['center_x_m']:.6f}, {zone['center_z_m']:.6f}) m with radius {zone['radius_m']:.1f} m. Newly computed occupancy is pedestrian {event.pedestrian_entry_time:.6f}-{event.pedestrian_exit_time:.6f} s and driver {event.vehicle_entry_time:.6f}-{event.vehicle_exit_time:.6f} s.

The 12 m proximity threshold, pedestrian micro rules, and possible-yielding rule are reused without modification. Possible yielding remains an unvalidated interpretation. Driver rows use existing acceleration thresholds: >= +0.35 m/s2 speeding up, <= -0.25 m/s2 slowing down, otherwise steady while speed >= 0.15 m/s.

No matching original screen capture was established from this CSV alone. The MP4 animates the authoritative VR trajectories and regenerated Gantt rather than substituting another recording.
"""
    (OUT/"methods_summary.md").write_text(methods,encoding="utf-8")
    definitions=[
        ["Pedestrian speed","Pedestrian","B VR Pos X/Z; ScenarioTime",f"planar displacement/delta-t; centered {SMOOTH_SPEED}-sample mean","Observed"],
        ["Pedestrian acceleration","Pedestrian","smoothed pedestrian speed; ScenarioTime",f"delta-speed/delta-t; centered {SMOOTH_ACCEL}-sample mean","Observed"],
        ["Head movement","Pedestrian","head-turn rate; head/body yaw difference","micro v6 head thresholds","Observed"],
        ["Left hand above shoulder","Pedestrian","left-hand Y; left-shoulder Y","hand Y > shoulder Y; raw; no minimum duration","Observed candidate"],
        ["Right hand above shoulder","Pedestrian","right-hand Y; right-shoulder Y","hand Y > shoulder Y; raw; no minimum duration","Observed candidate"],
        ["Driver nearby","Interaction","smoothed planar distance","distance < 12.0 m","Observed"],
        ["Pedestrian inside conflict zone","Pedestrian","B VR Pos X/Z; existing circle","distance to center <= 2.0 m","Observed"],
        ["Driver inside conflict zone","Driver","A car Pos X/Z; existing circle","distance to center <= 2.0 m","Observed"],
        ["Driver speeding up","Driver","smoothed driver acceleration","acceleration >= +0.35 m/s2","Observed"],
        ["Driver slowing down","Driver","smoothed driver acceleration","acceleration <= -0.25 m/s2","Observed"],
        ["Driver steady speed","Driver","driver speed and acceleration","-0.25 < acceleration < +0.35 m/s2 and speed >=0.15 m/s","Observed"],
        ["Speed dip with recovery","Interaction","micro speed drop/recovery",">0.18 m/s and >22% drop; recovery >=35%; duration <=1.5 s; mean speed >0.25 m/s","Inferred"],
        ["Possible yielding","Interaction","distance + pedestrian motion","distance <12 m AND slowing, pausing, or speed dip with recovery","Inferred; not intent"],
        ["Driver signaling","Driver","A indicators","False/False=No; True/False=Left; False/True=Right","Observed"],
    ]
    with (OUT/"definitions_table.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["Display label","Actor","Input features","Exact rule or threshold","Observed versus inferred status"]); w.writerows(definitions)
    verification={"authoritative_source":str(SOURCE),"authoritative_sha256":source_hash(),"time_origin_seconds":0.0,"duration_seconds":float(m.time.max()),"rows":len(m),"macro_micro_regenerated":True,"existing_outputs_modified":False,"signal_available":True,"video_kind":"authoritative VR trajectory animation" if video_path else "not rendered","pedestrian_zone_event":[event.pedestrian_entry_time,event.pedestrian_exit_time],"driver_zone_event":[event.vehicle_entry_time,event.vehicle_exit_time]}
    (OUT/"verification_report.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite existing presentation folder: {OUT}")
    OUT.mkdir(parents=True)
    subprocess.run([sys.executable, str(ROOT/"src/features/create_metrics_and_graph.py"), "--input-csv", str(SOURCE), "--output-csv", str(DECODED), "--segment-csv", str(OUT/"legacy_segments_from_authoritative.csv"), "--graph-dir", str(OUT/"pipeline_metric_diagnostics"), "--fourframe-dir", str(OUT/"pipeline_metric_diagnostics_copy")],check=True)
    MACRO.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/"src/segmentation/macro_segmentation.py"),"--input-csv",str(DECODED),"--output-dir",str(MACRO.parent),"--segments-csv",str(MACRO),"--plot-path",str(MACRO.parent/"macro_segments_authoritative.png")],check=True)
    subprocess.run([sys.executable,str(ROOT/"src/segmentation/micro_segmentation.py"),"--feature-csv",str(DECODED),"--macro-csv",str(MACRO),"--output-dir",str(OUT/"micro"),"--scenario-stem","PedNYC1_scenario3"],check=True)
    df=read_decoded(); m=metrics(df); macros=pd.read_csv(MACRO); micro=pd.read_csv(MICRO); zone=json.loads(ZONE.read_text(encoding="utf-8")); event=zone_event_from_trajectory(m, zone); duration=float(m.time.max())
    for path, frame in ((MACRO,macros),(MICRO,micro)):
        frame.insert(0,"source_sha256",source_hash()); frame.insert(0,"source_csv",str(SOURCE)); frame.to_csv(path,index=False)
    pd.DataFrame([{"source_csv":str(SOURCE),"source_sha256":source_hash(),"pedestrian_entry_time":event.pedestrian_entry_time,"pedestrian_exit_time":event.pedestrian_exit_time,"driver_entry_time":event.vehicle_entry_time,"driver_exit_time":event.vehicle_exit_time,"zone_center_x_m":zone["center_x_m"],"zone_center_z_m":zone["center_z_m"],"zone_radius_m":zone["radius_m"]}]).to_csv(OUT/"conflict_zone_events_authoritative.csv",index=False)
    pedestrian_figure(m,event); driver_figure(m,event); trajectory_figure(m,zone)
    rows=gantt_data(m,event,micro); gantt_figure(rows,macros,duration)
    video=render_video(m,rows,macros,zone,duration)
    write_docs(m,event,macros,micro,zone,video)
    print(OUT)


if __name__ == "__main__": main()
