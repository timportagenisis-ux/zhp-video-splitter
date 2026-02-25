#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shot_splitter_fixed.py

Shot splitter (FFmpeg scene detect + boundary-window false-cut merge + intra-shot refine + short-shot merge)
+ AUTO PROFILE (style detection + auto tune + soft weights)
+ PROFILE PRESETS (manual override)

No third-party deps. Windows-safe.

Outputs:
  - slots/<input_name>/*.mp4 (default output dir)
  - slots/<input_name>/shots.json (if --save_meta)
  - slots.zip (default or --zip_path)

Typical:
  python main.py input.mp4
  # equivalent to: python main.py -i input.mp4 --auto_profile -o slots/input

Manual profile:
  python main.py input.mp4 --save_meta --profile talking_head
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from array import array


# =========================
# Data models
# =========================


@dataclass
class CutPoint:
    t: float
    score: float
    reason: str = "hard"  # hard/transition/strong


# =========================
# subprocess helpers
# =========================


def run_cmd_text(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out or "", err or ""


def run_cmd_bytes(cmd: List[str]) -> Tuple[int, bytes, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_b, err_b = p.communicate()
    try:
        err = err_b.decode("utf-8", errors="ignore")
    except Exception:
        err = ""
    return p.returncode, out_b or b"", err


FFMPEG_PATH = "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe"
FFPROBE_PATH = "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0.1-essentials_build\\bin\\ffprobe.exe"


def require_ffmpeg():
    """Check if ffmpeg and ffprobe executables exist at the specified paths."""
    if not os.path.isfile(FFMPEG_PATH):
        raise RuntimeError(f"Missing dependency: ffmpeg not found at {FFMPEG_PATH}")
    if not os.path.isfile(FFPROBE_PATH):
        raise RuntimeError(f"Missing dependency: ffprobe not found at {FFPROBE_PATH}")


# =========================
# ffprobe
# =========================


def ffprobe_get_duration(input_path: str) -> float:
    cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    code, out, err = run_cmd_text(cmd)
    if code != 0 or not out.strip():
        raise RuntimeError(f"ffprobe duration failed.\nERR:\n{err}")
    return float(out.strip())


def ffprobe_get_fps(input_path: str) -> float:
    cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    code, out, _ = run_cmd_text(cmd)
    if code != 0 or not out.strip():
        return 30.0

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    rates = []
    for l in lines:
        if "/" in l:
            a, b = l.split("/", 1)
            try:
                rates.append(float(a) / float(b))
            except Exception:
                pass
        else:
            try:
                rates.append(float(l))
            except Exception:
                pass

    for r in rates:
        if 1.0 <= r <= 240.0:
            return r
    return 30.0


def ffprobe_get_wh(input_path: str) -> Tuple[int, int]:
    cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        input_path,
    ]
    code, out, err = run_cmd_text(cmd)
    if code != 0 or not out.strip():
        raise RuntimeError(f"ffprobe width/height failed.\nERR:\n{err}")
    w, h = out.strip().split("x")
    return int(w), int(h)


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x + 1


# =========================
# Scene detect (FFmpeg scene_score)
# =========================


def extract_scene_meta_text(
    input_path: str,
    th_hard: float,
    denoise: bool,
    blur: bool,
    t_limit: Optional[float] = None,
) -> str:
    """
    Runs ffmpeg select=gt(scene,th_hard),metadata=print and returns stderr text
    (metadata lines are printed to stderr).
    """
    vf_parts = []
    if denoise:
        vf_parts.append("hqdn3d")
    if blur:
        vf_parts.append("boxblur=1:1")
    vf_parts.append(f"select='gt(scene,{th_hard})'")
    vf_parts.append("metadata=print")
    vf = ",".join(vf_parts)

    cmd = [FFMPEG_PATH, "-hide_banner", "-y"]
    if t_limit and t_limit > 0:
        cmd += ["-t", f"{t_limit:.6f}"]
    cmd += ["-i", input_path, "-vf", vf, "-an", "-f", "null", "-"]

    code, _, err = run_cmd_text(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg scene detect failed.\nERR:\n{err}")
    return err


def parse_scene_meta(text: str) -> List[CutPoint]:
    cuts: List[CutPoint] = []
    pending_t: Optional[float] = None

    pts_re = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
    score_re = re.compile(r"lavfi\.scene_score=(\d+(?:\.\d+)?)")

    for line in text.splitlines():
        m_t = pts_re.search(line)
        if m_t:
            pending_t = float(m_t.group(1))
            continue
        m_s = score_re.search(line)
        if m_s and pending_t is not None:
            cuts.append(CutPoint(t=pending_t, score=float(m_s.group(1)), reason="hard"))
            pending_t = None

    cuts.sort(key=lambda c: c.t)
    return cuts


def jitter_merge(cuts: List[CutPoint], merge_gap: float) -> List[CutPoint]:
    """Merge candidates too close; keep higher score."""
    if not cuts:
        return []
    kept = [cuts[0]]
    for c in cuts[1:]:
        last = kept[-1]
        if c.t - last.t < merge_gap:
            if c.score > last.score:
                kept[-1] = c
        else:
            kept.append(c)
    return kept


def soft_transition_cluster(
    cuts: List[CutPoint], th_soft: float, window_sec: float = 1.0, min_count: int = 3
) -> List[CutPoint]:
    """
    Cluster multiple nearby peaks (flash/fade-like). Keep the strongest.
    """
    if not cuts:
        return []
    result: List[CutPoint] = []
    i = 0
    n = len(cuts)
    while i < n:
        c = cuts[i]
        if c.score < th_soft:
            result.append(c)
            i += 1
            continue

        start_t = c.t
        cluster = []
        j = i
        while j < n and (cuts[j].t - start_t) <= window_sec:
            if cuts[j].score >= th_soft:
                cluster.append(cuts[j])
            j += 1

        if len(cluster) >= min_count:
            peak = max(cluster, key=lambda x: x.score)
            peak.reason = "transition"
            result.append(peak)
            i = j
        else:
            result.append(c)
            i += 1

    result.sort(key=lambda x: x.t)
    return jitter_merge(result, merge_gap=0.001)


def enforce_min_shot_len(
    cuts: List[CutPoint], duration: float, min_shot_len: float, th_keep_strong: float
) -> List[CutPoint]:
    """
    Remove non-strong cuts that create too-short segments.
    """
    if not cuts:
        return []

    for c in cuts:
        if c.score >= th_keep_strong:
            c.reason = "strong"

    kept = sorted(cuts, key=lambda c: c.t)

    def bounds(_kept: List[CutPoint]) -> List[float]:
        ts = [c.t for c in _kept if 0.001 < c.t < duration - 0.001]
        ts.sort()
        return [0.0] + ts + [duration]

    changed = True
    while changed and kept:
        changed = False
        b = bounds(kept)

        remove_idx: Optional[int] = None

        # too-short left segment
        for seg_i in range(1, len(b) - 1):
            seg_len = b[seg_i] - b[seg_i - 1]
            if seg_len < min_shot_len:
                cut = kept[seg_i - 1]
                if cut.reason != "strong":
                    remove_idx = seg_i - 1
                    break

        # too-short right segment
        if remove_idx is None:
            for seg_i in range(1, len(b) - 1):
                seg_len = b[seg_i + 1] - b[seg_i]
                if seg_len < min_shot_len:
                    left_cut = kept[seg_i - 1]
                    right_cut = kept[seg_i] if seg_i < len(kept) else None
                    candidates = []
                    if left_cut.reason != "strong":
                        candidates.append((seg_i - 1, left_cut.score))
                    if right_cut and right_cut.reason != "strong":
                        candidates.append((seg_i, right_cut.score))
                    if candidates:
                        candidates.sort(key=lambda x: x[1])  # remove weaker
                        remove_idx = candidates[0][0]
                        break

        if remove_idx is not None:
            kept.pop(remove_idx)
            changed = True

    return kept


def build_shots_from_cuts(
    cuts: List[CutPoint], duration: float
) -> List[Tuple[float, float]]:
    ts = [c.t for c in sorted(cuts, key=lambda x: x.t)]
    cleaned = []
    for t in ts:
        if 0.001 < t < duration - 0.001:
            if not cleaned or abs(t - cleaned[-1]) > 1e-6:
                cleaned.append(t)
    b = [0.0] + cleaned + [duration]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


# =========================
# Raw frames + metrics (no numpy)
# =========================


def raw_frames_gray_window(
    input_path: str,
    start: float,
    end: float,
    sample_fps: int,
    out_w: int,
    out_h: int,
    blur: int,
) -> List[bytes]:
    dur = max(0.0, end - start)
    if dur <= 0:
        return []

    vf = f"fps={sample_fps},scale={out_w}:{out_h},boxblur={blur}:{blur},format=gray"
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{dur:.6f}",
        "-i",
        input_path,
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    code, out_b, _ = run_cmd_bytes(cmd)
    if code != 0 or not out_b:
        return []
    frame_size = out_w * out_h
    n = len(out_b) // frame_size
    if n <= 0:
        return []
    return [out_b[i * frame_size : (i + 1) * frame_size] for i in range(n)]


def mean_abs_diff_bytes(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    total = 0
    for x, y in zip(a, b):
        total += x - y if x >= y else y - x
    return total / len(a)


def avg_frame_diff_between_windows(
    left_frames: List[bytes], right_frames: List[bytes]
) -> Optional[float]:
    """
    Average absolute difference between per-pixel means of left window and right window.
    Uses integer accumulation via array('I') for speed without numpy.
    """
    if not left_frames or not right_frames:
        return None
    frame_size = len(left_frames[0])
    if any(len(f) != frame_size for f in left_frames + right_frames):
        return None

    def avg_acc(frames: List[bytes]) -> array:
        acc = array("I", [0]) * frame_size
        for fr in frames:
            mv = memoryview(fr)
            for i in range(frame_size):
                acc[i] += mv[i]
        return acc

    la = avg_acc(left_frames)
    ra = avg_acc(right_frames)
    ln = len(left_frames)
    rn = len(right_frames)

    total = 0.0
    for i in range(frame_size):
        lv = la[i] / ln
        rv = ra[i] / rn
        d = lv - rv
        total += d if d >= 0 else -d
    return total / frame_size


# =========================
# Boundary-window validation (merge false cuts)
# =========================


def filter_false_cuts_by_boundary_window(
    input_path: str,
    cuts: List[CutPoint],
    duration: float,
    out_w: int,
    out_h: int,
    gap: float,
    win: float,
    sample_fps: int,
    blur: int,
    merge_th: float,
) -> List[CutPoint]:
    if not cuts:
        return []

    kept: List[CutPoint] = []
    for c in cuts:
        t = c.t
        if t <= (gap + win) or (duration - t) <= (gap + win):
            kept.append(c)
            continue

        left = raw_frames_gray_window(
            input_path, t - (gap + win), t - gap, sample_fps, out_w, out_h, blur
        )
        right = raw_frames_gray_window(
            input_path, t + gap, t + (gap + win), sample_fps, out_w, out_h, blur
        )

        m = avg_frame_diff_between_windows(left, right)
        if m is None:
            kept.append(c)
            continue

        # If boundary diff is small -> likely same shot -> drop cut (merge)
        if m < merge_th:
            continue
        kept.append(c)

    kept.sort(key=lambda x: x.t)
    return kept


# =========================
# Intra-shot refine
# =========================


def robust_threshold(diffs: List[float], k_mad: float, abs_min: float) -> float:
    if not diffs:
        return abs_min
    s = sorted(diffs)
    med = s[len(s) // 2]
    dev = sorted([abs(x - med) for x in diffs])
    mad = dev[len(dev) // 2] + 1e-6
    return max(abs_min, med + k_mad * mad)


def refine_shots_intra(
    input_path: str,
    shots: List[Tuple[float, float]],
    out_w: int,
    out_h: int,
    sample_fps: int,
    blur: int,
    edge_ignore: float,
    k_mad: float,
    abs_min: float,
    min_new_len: float,
    min_gap: float,
) -> List[Tuple[float, float]]:
    refined: List[Tuple[float, float]] = []
    for s, e in shots:
        if e - s < 1.2:
            refined.append((s, e))
            continue

        frames = raw_frames_gray_window(
            input_path, s, e, sample_fps, out_w, out_h, blur
        )
        if len(frames) < 3:
            refined.append((s, e))
            continue

        diffs = [
            mean_abs_diff_bytes(frames[i - 1], frames[i]) for i in range(1, len(frames))
        ]
        thr = robust_threshold(diffs, k_mad=k_mad, abs_min=abs_min)

        cand = [i for i, v in enumerate(diffs, start=1) if v >= thr]
        if not cand:
            refined.append((s, e))
            continue

        keep_idx = []
        cluster = [cand[0]]
        for idx in cand[1:]:
            if (idx - cluster[-1]) / sample_fps <= min_gap:
                cluster.append(idx)
            else:
                best = max(cluster, key=lambda j: diffs[j - 1])
                keep_idx.append(best)
                cluster = [idx]
        best = max(cluster, key=lambda j: diffs[j - 1])
        keep_idx.append(best)

        cut_times = []
        for j in keep_idx:
            t = s + (j / sample_fps)
            if t - s < edge_ignore:
                continue
            if e - t < edge_ignore:
                continue
            cut_times.append(t)
        cut_times.sort()

        if not cut_times:
            refined.append((s, e))
            continue

        b = [s] + cut_times + [e]
        local: List[Tuple[float, float]] = []
        for i in range(len(b) - 1):
            ss, ee = b[i], b[i + 1]
            if ee - ss >= min_new_len:
                local.append((ss, ee))
            else:
                if local:
                    ps, pe = local[-1]
                    local[-1] = (ps, max(pe, ee))
                else:
                    local.append((ss, ee))
        refined.extend(local)

    overlap_eps = 1e-4
    refined.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    for s, e in refined:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s < pe - overlap_eps:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# =========================
# Merge short shots into next
# =========================


def merge_short_shots_with_next(
    shots: List[Tuple[float, float]], min_len: float
) -> List[Tuple[float, float]]:
    """
    Rule:
      If a shot duration < min_len, merge it into the NEXT shot.
      If it's the last shot and too short, merge into previous.
    """
    if not shots:
        return shots
    if min_len <= 0:
        return shots

    shots = [(float(s), float(e)) for s, e in shots]
    i = 0
    while i < len(shots):
        s, e = shots[i]
        dur = e - s
        if dur >= min_len:
            i += 1
            continue

        if i + 1 < len(shots):
            ns, ne = shots[i + 1]
            shots[i] = (s, ne)
            shots.pop(i + 1)
        else:
            if i - 1 >= 0:
                ps, pe = shots[i - 1]
                shots[i - 1] = (ps, e)
                shots.pop(i)
                i -= 1
            else:
                break

    out = []
    for s, e in shots:
        if e > s:
            out.append((s, e))
    return out


# =========================
# End safety pad (avoid next-shot frame leakage)
# =========================


def apply_end_safety_pad(
    shots: List[Tuple[float, float]], fps: float, min_keep: float = 0.10
) -> List[Tuple[float, float]]:
    if not shots:
        return shots
    one_frame = 1.0 / max(1.0, fps)
    out = []
    for i, (s, e) in enumerate(shots):
        if i == len(shots) - 1:
            out.append((s, e))
        else:
            e2 = e - one_frame
            if e2 <= s + min_keep:
                e2 = e
            out.append((s, e2))
    return out


# =========================
# Cutting + zip
# =========================


def ffmpeg_cut_segment(
    input_path: str,
    out_path: str,
    start: float,
    end: float,
    mode: str,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    start = max(0.0, float(start))
    end = max(start, float(end))
    dur = max(0.0, end - start)
    start_str = f"{start:.6f}"
    dur_str = f"{dur:.6f}"

    if mode == "copy":
        cmd = [
                FFMPEG_PATH,
                "-hide_banner",
                "-y",
                "-ss",
                start_str,
                "-i",
                input_path,
                "-t",
                dur_str,
                "-c",
                "copy",
                out_path,
            ]
    else:
        cmd = [
            FFMPEG_PATH,
            "-hide_banner",
            "-y",
            "-i",
            input_path,
            "-ss",
            start_str,
            "-t",
            dur_str,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            out_path,
        ]
    code, _, err = run_cmd_text(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg cut failed: {out_path}\nERR:\n{err}")


def zip_outputs(output_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(output_dir):
            for fn in files:
                if fn.lower().endswith(".mp4") or fn.lower() == "shots.json":
                    abs_path = os.path.join(root, fn)
                    rel_path = os.path.relpath(abs_path, os.path.dirname(output_dir))
                    z.write(abs_path, rel_path)


# =========================
# AUTO PROFILE (style detection + params)
# =========================


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    q = clamp(q, 0.0, 1.0)
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def analyze_motion_transition(
    input_path: str, seconds: float, sample_fps: int = 8, scale_w: int = 160
) -> Dict[str, Any]:
    """
    Fast heuristic:
      - fps downsample
      - grayscale
      - tblend difference
      - signalstats + metadata=print
    Uses lavfi.signalstats.YAVG as diff magnitude proxy.

    Returns:
      motion_mean, motion_p95, spike_rate, motion_score(0-100), transition_score(0-100)
    """
    yavg_re = re.compile(r"lavfi\.signalstats\.YAVG=(\d+(?:\.\d+)?)")

    vf = (
        f"fps={sample_fps},"
        f"scale={scale_w}:-2,"
        f"format=gray,"
        f"tblend=all_mode=difference,"
        f"signalstats,"
        f"metadata=print"
    )
    cmd = [FFMPEG_PATH, "-hide_banner", "-y"]
    if seconds and seconds > 0:
        cmd += ["-t", f"{seconds:.6f}"]
    cmd += ["-i", input_path, "-vf", vf, "-an", "-f", "null", "-"]

    code, _, err = run_cmd_text(cmd)
    if code != 0:
        raise RuntimeError(f"auto_profile motion analyze failed.\nERR:\n{err}")

    vals: List[float] = []
    for line in err.splitlines():
        m = yavg_re.search(line)
        if m:
            vals.append(float(m.group(1)))

    if len(vals) < 5:
        return {
            "motion_mean": 0.0,
            "motion_p95": 0.0,
            "spike_rate": 0.0,
            "motion_score": 0.0,
            "transition_score": 0.0,
            "raw": {"diff_yavg": vals},
        }

    vals_sorted = sorted(vals)
    mean_v = sum(vals) / len(vals)
    p95 = percentile(vals_sorted, 0.95)

    # spikes: above max(20, median + 3*MAD)
    med = vals_sorted[len(vals_sorted) // 2]
    dev = sorted([abs(x - med) for x in vals])
    mad = dev[len(dev) // 2] + 1e-6
    spike_th = max(20.0, med + 3.0 * mad)
    spikes = sum(1 for x in vals if x >= spike_th)
    spike_rate = spikes / len(vals)

    # Map to 0-100 scores
    motion_score = clamp((mean_v - 8.0) * 2.2, 0.0, 100.0)
    transition_score = clamp((p95 - 18.0) * 2.0 + spike_rate * 80.0, 0.0, 100.0)

    return {
        "motion_mean": mean_v,
        "motion_p95": p95,
        "spike_rate": spike_rate,
        "motion_score": motion_score,
        "transition_score": transition_score,
        "raw": {"diff_yavg": vals[:2000], "spike_th": spike_th},
    }


def analyze_cut_density(input_path: str, seconds: float) -> Dict[str, Any]:
    """
    Cut density proxy:
      scene detect at low threshold 0.20 and count cuts per minute.
    """
    t_limit = seconds if seconds and seconds > 0 else None
    meta_text = extract_scene_meta_text(
        input_path, th_hard=0.20, denoise=True, blur=False, t_limit=t_limit
    )
    cuts = parse_scene_meta(meta_text)

    dur = (
        max(1e-6, float(seconds))
        if seconds and seconds > 0
        else ffprobe_get_duration(input_path)
    )
    per_min = len(cuts) / dur * 60.0
    return {"cuts": len(cuts), "duration": dur, "cut_density_per_min": per_min}


def profile_presets() -> Dict[str, Dict[str, Any]]:
    """
    Hand-tuned presets (good starting points).
    Keys must match args fields.
    """
    return {
        "talking_head": {
            "th_hard": 0.35,
            "th_soft": 0.25,
            "boundary_merge_th": 15.0,
            "boundary_win": 0.22,
            "boundary_blur": 5,
            "intra_sample_fps": 8,
            "intra_abs_min": 45.0,
            "min_shot_merge_len": 0.50,
        },
        "drama_normal": {
            "th_hard": 0.30,
            "th_soft": 0.21,
            "boundary_merge_th": 12.0,
            "boundary_win": 0.20,
            "boundary_blur": 4,
            "intra_sample_fps": 12,
            "intra_abs_min": 35.0,
            "min_shot_merge_len": 0.30,
        },
        "street_motion": {
            "th_hard": 0.35,
            "th_soft": 0.25,
            "boundary_merge_th": 16.0,
            "boundary_win": 0.30,
            "boundary_blur": 6,
            "intra_sample_fps": 10,
            "intra_abs_min": 40.0,
            "min_shot_merge_len": 0.40,
        },
        "fastcut_transition": {
            "th_hard": 0.25,
            "th_soft": 0.18,
            "boundary_merge_th": 9.0,
            "boundary_win": 0.18,
            "boundary_blur": 3,
            "intra_sample_fps": 15,
            "intra_abs_min": 28.0,
            "min_shot_merge_len": 0.20,
        },
    }


def style_weights_from_scores(
    motion_score: float, transition_score: float, cut_density_per_min: float
) -> Dict[str, float]:
    """
    Soft weighting (not hard classification).
    Uses distance-to-archetype with exponential falloff -> weights sum to 1.
    """
    centers = {
        "talking_head": {"m": 15.0, "t": 15.0, "d": 5.0},
        "drama_normal": {"m": 35.0, "t": 30.0, "d": 15.0},
        "street_motion": {"m": 75.0, "t": 35.0, "d": 12.0},
        "fastcut_transition": {"m": 45.0, "t": 80.0, "d": 35.0},
    }
    sm, st, sd = 25.0, 25.0, 15.0

    def dist2(c):
        dm = (motion_score - c["m"]) / sm
        dt = (transition_score - c["t"]) / st
        dd = (cut_density_per_min - c["d"]) / sd
        return dm * dm + dt * dt + dd * dd

    raw = {name: math.exp(-dist2(c)) for name, c in centers.items()}
    s = sum(raw.values()) + 1e-9
    return {k: v / s for k, v in raw.items()}


def confidence_from_weights(weights: Dict[str, float]) -> float:
    return max(weights.values()) if weights else 0.0


def blend_params_by_weights(weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Weighted blend across presets.
    - float params: weighted average
    - int params: weighted average then round
    """
    presets = profile_presets()
    keys = set()
    for p in presets.values():
        keys |= set(p.keys())

    s = sum(max(0.0, v) for v in weights.values())
    if s <= 1e-9:
        return dict(presets["drama_normal"])

    wnorm = {k: max(0.0, v) / s for k, v in weights.items()}

    out: Dict[str, Any] = {}
    for k in keys:
        acc = 0.0
        for name, w in wnorm.items():
            if name in presets and k in presets[name]:
                acc += float(presets[name][k]) * w
        if k in ("boundary_blur", "intra_sample_fps"):
            out[k] = int(round(acc))
        else:
            out[k] = float(acc)
    return out


def apply_auto_params_if_default(
    args: argparse.Namespace, recommended: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Only override args.<k> if it is still equal to its default value.
    This respects explicit user settings.
    Returns dict of actually-applied overrides.
    """
    applied = {}
    for k, v in recommended.items():
        if not hasattr(args, k):
            continue
        if k in defaults and getattr(args, k) == defaults[k]:
            setattr(args, k, v)
            applied[k] = v
    return applied


# =========================
# Main
# =========================


def main():
    parser = argparse.ArgumentParser(
        description="Shot splitter (scene + boundary merge + intra refine + short merge) with auto style/profile."
    )

    # I/O
    parser.add_argument(
        "input_positional", nargs="?", help="Input video path (same as -i/--input)."
    )
    parser.add_argument("-i", "--input", help="Input video path.")
    parser.add_argument(
        "-o",
        "--output_dir",
        default="slots",
        help="Base output directory. Final output is <output_dir>/<input_name>.",
    )
    parser.add_argument("--zip_path", default="slots.zip")
    parser.add_argument("--mode", choices=["reencode", "copy"], default="reencode")
    parser.add_argument("--save_meta", action="store_true")

    # Auto profile + presets
    auto_profile_group = parser.add_mutually_exclusive_group()
    auto_profile_group.add_argument(
        "--auto_profile",
        dest="auto_profile",
        action="store_true",
        help="Analyze motion/transition/cut_density and recommend parameters (only overrides params still at defaults).",
    )
    auto_profile_group.add_argument(
        "--no_auto_profile",
        dest="auto_profile",
        action="store_false",
        help="Disable automatic profile analysis.",
    )
    parser.set_defaults(auto_profile=True)
    parser.add_argument(
        "--auto_seconds",
        type=float,
        default=12.0,
        help="Analyze only first N seconds for auto_profile. Default 12s. Use 0 for full video (slower).",
    )
    parser.add_argument(
        "--profile",
        choices=[
            "auto",
            "talking_head",
            "drama_normal",
            "street_motion",
            "fastcut_transition",
        ],
        default="auto",
        help="Style preset. 'auto' blends presets using detected weights.",
    )

    # Scene detection
    parser.add_argument("--th_hard", type=float, default=0.40)
    parser.add_argument("--th_soft", type=float, default=0.28)
    parser.add_argument("--min_shot_len", type=float, default=1.6)
    parser.add_argument("--merge_gap", type=float, default=0.25)
    parser.add_argument("--th_keep_strong", type=float, default=0.55)
    parser.add_argument("--no_denoise", action="store_true")
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--no_auto_relax", action="store_true")

    # Boundary-window merge
    parser.add_argument("--boundary_gap", type=float, default=0.05)
    parser.add_argument("--boundary_win", type=float, default=0.20)
    parser.add_argument("--boundary_sample_fps", type=int, default=6)
    parser.add_argument("--boundary_scale_w", type=int, default=64)
    parser.add_argument("--boundary_blur", type=int, default=4)
    parser.add_argument("--boundary_merge_th", type=float, default=12.0)

    # Intra refine
    parser.add_argument("--no_intra_refine", action="store_true")
    parser.add_argument("--intra_sample_fps", type=int, default=12)
    parser.add_argument("--intra_blur", type=int, default=2)
    parser.add_argument("--intra_edge_ignore", type=float, default=0.25)
    parser.add_argument("--intra_k_mad", type=float, default=6.0)
    parser.add_argument("--intra_abs_min", type=float, default=35.0)
    parser.add_argument("--min_new_shot_len", type=float, default=0.9)
    parser.add_argument("--intra_min_gap", type=float, default=0.5)

    # Short-shot merge
    parser.add_argument(
        "--min_shot_merge_len",
        type=float,
        default=0.30,
        help="If any shot duration < this, merge into NEXT shot (or previous if last).",
    )

    # End pad
    parser.add_argument("--no_end_pad", action="store_true")

    # Encode
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--audio_bitrate", default="128k")

    args = parser.parse_args()

    # Allow positional input so `python main.py input.mp4` works by default.
    if not args.input:
        args.input = args.input_positional
    elif args.input_positional and args.input != args.input_positional:
        parser.error(
            "Input provided twice with different values: positional INPUT and -i/--input."
        )

    if not args.input:
        parser.error("Missing input. Use positional INPUT or -i/--input.")

    # Output directory rule:
    # create a subfolder under output_dir named after input file stem.
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    if not input_name:
        input_name = "output"

    out_tail = os.path.basename(os.path.normpath(args.output_dir))
    if out_tail != input_name:
        args.output_dir = os.path.join(args.output_dir, input_name)

    # Defaults used to decide what is "user-set" vs "still default"
    defaults = {
        "th_hard": 0.40,
        "th_soft": 0.28,
        "boundary_merge_th": 12.0,
        "boundary_win": 0.20,
        "boundary_blur": 4,
        "intra_sample_fps": 12,
        "intra_abs_min": 35.0,
        "min_shot_merge_len": 0.30,
    }

    require_ffmpeg()
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    duration = ffprobe_get_duration(args.input)
    fps = ffprobe_get_fps(args.input)
    src_w, src_h = ffprobe_get_wh(args.input)

    # =========================
    # Apply manual profile preset (non-destructive; only overrides defaults)
    # =========================
    presets = profile_presets()
    profile_info = None
    if args.profile != "auto":
        recommended = dict(presets[args.profile])
        applied = apply_auto_params_if_default(args, recommended, defaults)
        profile_info = {
            "profile": args.profile,
            "params_recommended": recommended,
            "params_applied": applied,
        }
        print(f"[profile] using preset={args.profile}")
        if applied:
            print(
                f"[profile] applied (only defaults): {', '.join([f'{k}={applied[k]}' for k in applied])}"
            )
        else:
            print(
                "[profile] no overrides applied (you likely set these params manually)."
            )

    # =========================
    # AUTO PROFILE (analysis + blended preset)
    # =========================
    auto_info = None
    if args.auto_profile:
        analyze_seconds = (
            args.auto_seconds
            if args.auto_seconds and args.auto_seconds > 0
            else duration
        )

        mt = analyze_motion_transition(
            args.input, seconds=analyze_seconds, sample_fps=8, scale_w=160
        )
        cd = analyze_cut_density(args.input, seconds=analyze_seconds)

        weights = style_weights_from_scores(
            mt["motion_score"], mt["transition_score"], cd["cut_density_per_min"]
        )
        conf = confidence_from_weights(weights)

        if args.profile == "auto":
            style_top = max(weights, key=lambda name: weights[name])
            recommended = blend_params_by_weights(weights)
            weights_used = weights
            conf_used = conf
        else:
            # If user forced a profile, auto still reports weights but won't change preset choice.
            style_top = args.profile
            recommended = dict(presets[style_top])
            weights_used = {k: (1.0 if k == style_top else 0.0) for k in presets.keys()}
            conf_used = 1.0

        applied = apply_auto_params_if_default(args, recommended, defaults)

        auto_info = {
            "analyze_seconds": analyze_seconds,
            "motion_score": mt["motion_score"],
            "transition_score": mt["transition_score"],
            "cut_density_per_min": cd["cut_density_per_min"],
            "style_top": style_top,
            "weights": weights_used,
            "confidence": conf_used,
            "params_recommended": recommended,
            "params_applied": applied,
            "raw": {"motion": mt, "cut_density": cd},
        }

        print(
            f"[auto_profile] motion={mt['motion_score']:.1f} transition={mt['transition_score']:.1f} cut_density/min={cd['cut_density_per_min']:.1f}"
        )
        print(
            "[auto_profile] weights: "
            + ", ".join([f"{k}={weights[k]:.2f}" for k in sorted(weights.keys())])
        )
        print(f"[auto_profile] confidence={conf_used:.2f} style={style_top}")
        if applied:
            print(
                f"[auto_profile] applied (only defaults): {', '.join([f'{k}={applied[k]}' for k in applied])}"
            )
        else:
            print(
                "[auto_profile] no overrides applied (you likely set these params manually)."
            )

    # =========================
    # Prepare analysis sizes
    # =========================
    bw = args.boundary_scale_w
    bh = make_even(int(round(src_h * (bw / src_w))))
    iw = 160
    ih = make_even(int(round(src_h * (iw / src_w))))

    # =========================
    # 1) Scene detect
    # =========================
    meta_text = extract_scene_meta_text(
        args.input, args.th_hard, denoise=(not args.no_denoise), blur=args.blur
    )
    cuts = parse_scene_meta(meta_text)
    print(f"Parsed cuts: {len(cuts)} (th_hard={args.th_hard})")

    if (not args.no_auto_relax) and (not cuts):
        for th in [0.30, 0.25, 0.20, 0.18, 0.15, 0.12]:
            meta_text = extract_scene_meta_text(
                args.input, th, denoise=(not args.no_denoise), blur=args.blur
            )
            cuts = parse_scene_meta(meta_text)
            print(f"Parsed cuts: {len(cuts)} (auto th_hard={th})")
            if cuts:
                args.th_hard = th
                args.th_soft = min(args.th_soft, th * 0.70)
                break

    # =========================
    # 2) Cut post rules
    # =========================
    cuts = jitter_merge(cuts, merge_gap=args.merge_gap)
    cuts = soft_transition_cluster(cuts, th_soft=args.th_soft)
    cuts = jitter_merge(cuts, merge_gap=args.merge_gap)
    cuts = enforce_min_shot_len(cuts, duration, args.min_shot_len, args.th_keep_strong)

    # =========================
    # 3) Boundary-window validation (merge false cuts)
    # =========================
    cuts = filter_false_cuts_by_boundary_window(
        input_path=args.input,
        cuts=cuts,
        duration=duration,
        out_w=bw,
        out_h=bh,
        gap=args.boundary_gap,
        win=args.boundary_win,
        sample_fps=args.boundary_sample_fps,
        blur=args.boundary_blur,
        merge_th=args.boundary_merge_th,
    )

    # =========================
    # 4) Build initial shots
    # =========================
    shots = build_shots_from_cuts(cuts, duration)

    # =========================
    # 5) Intra refine
    # =========================
    if (not args.no_intra_refine) and shots:
        shots = refine_shots_intra(
            input_path=args.input,
            shots=shots,
            out_w=iw,
            out_h=ih,
            sample_fps=args.intra_sample_fps,
            blur=args.intra_blur,
            edge_ignore=args.intra_edge_ignore,
            k_mad=args.intra_k_mad,
            abs_min=args.intra_abs_min,
            min_new_len=args.min_new_shot_len,
            min_gap=args.intra_min_gap,
        )

    # =========================
    # 6) Merge too-short shots into next
    # =========================
    shots = merge_short_shots_with_next(shots, min_len=args.min_shot_merge_len)

    # =========================
    # 7) End pad (avoid next-shot frame leak)
    # =========================
    if (not args.no_end_pad) and shots:
        shots = apply_end_safety_pad(shots, fps=fps, min_keep=0.10)

    # =========================
    # Output meta
    # =========================
    meta: Dict[str, Any] = {
        "input": os.path.abspath(args.input),
        "duration": duration,
        "fps": fps,
        "analysis_sizes": {"boundary": {"w": bw, "h": bh}, "intra": {"w": iw, "h": ih}},
        "params_effective": vars(args),
        "profile": profile_info,
        "auto_profile": auto_info,
        "cuts": [{"t": c.t, "score": c.score, "reason": c.reason} for c in cuts],
        "shots": [],
    }

    # =========================
    # Write clips
    # =========================
    pad = max(3, int(math.log10(max(1, len(shots)))) + 1)
    for idx, (s, e) in enumerate(shots, start=1):
        out_name = f"shot_{idx:0{pad}d}_{s:.2f}-{e:.2f}.mp4"
        out_path = os.path.join(args.output_dir, out_name)
        ffmpeg_cut_segment(
            args.input,
            out_path,
            s,
            e,
            args.mode,
            args.crf,
            args.preset,
            args.audio_bitrate,
        )
        meta["shots"].append(
            {"id": idx, "start": s, "end": e, "duration": e - s, "file": out_name}
        )

    # Metadata JSON
    if args.save_meta:
        meta_path = os.path.join(args.output_dir, "shots.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # Zip
    zip_outputs(args.output_dir, args.zip_path)

    print(f"Done. Shots: {len(shots)}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    if args.save_meta:
        print(
            f"Metadata: {os.path.join(os.path.abspath(args.output_dir), 'shots.json')}"
        )
    print(f"Zip: {os.path.abspath(args.zip_path)}")


if __name__ == "__main__":
    main()
