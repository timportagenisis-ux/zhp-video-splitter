#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFmpeg-based shot splitter (real-person short video) - optimized final.

Fixes:
- Windows-safe: capture metadata=print from stderr (no metadata=print:file=D:\... issues)
- Robust parsing: pair pts_time with next lavfi.scene_score
- Better cutting accuracy:
  - use duration (-t) instead of -to
  - for reencode: place -ss AFTER -i (accurate)
  - apply end safety pad: subtract 1 frame from each segment end (except last)
- Auto relax thresholds if no cuts found
- Optional fallback time split
- Output zip: shots.zip (contains clips + shots.json)

Requires: ffmpeg + ffprobe in PATH
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
from typing import List, Tuple, Optional


@dataclass
class CutPoint:
    t: float
    score: float
    reason: str = "hard"  # hard / transition / strong


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def require_ffmpeg():
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            raise RuntimeError(
                f"Missing dependency: {bin_name} not found in PATH. "
                f"Install FFmpeg and ensure ffmpeg/ffprobe are available."
            )


def get_duration_sec(input_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    code, out, err = run_cmd(cmd)
    if code != 0 or not out.strip():
        raise RuntimeError(f"ffprobe failed to get duration.\nCMD: {' '.join(cmd)}\nERR:\n{err}")
    return float(out.strip())


def get_fps(input_path: str) -> float:
    """
    Get FPS from video stream.
    Prefer avg_frame_rate, fallback to r_frame_rate.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    code, out, err = run_cmd(cmd)
    if code != 0 or not out.strip():
        # fallback: assume 30
        return 30.0

    # output is two lines: avg_frame_rate then r_frame_rate (maybe)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    rates = []
    for l in lines:
        # format like "30/1" or "30000/1001"
        if "/" in l:
            num, den = l.split("/", 1)
            try:
                num_f = float(num)
                den_f = float(den)
                if den_f != 0:
                    rates.append(num_f / den_f)
            except ValueError:
                pass
        else:
            try:
                rates.append(float(l))
            except ValueError:
                pass

    # pick a reasonable fps
    for r in rates:
        if 1.0 <= r <= 240.0:
            return r
    return 30.0


def extract_candidate_cuts_text(
    input_path: str,
    th_hard: float,
    use_denoise: bool = True,
    add_blur: bool = False,
) -> str:
    """
    Run ffmpeg scene detection and return metadata=print output as text (stderr).
    """
    vf_parts = []
    if use_denoise:
        vf_parts.append("hqdn3d")
    if add_blur:
        vf_parts.append("boxblur=1:1")
    vf_parts.append(f"select='gt(scene,{th_hard})'")
    vf_parts.append("metadata=print")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-vf", vf,
        "-an",
        "-f", "null", "-"
    ]
    code, out, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg scene detect failed.\nCMD: {' '.join(cmd)}\nERR:\n{err}")
    return err  # metadata printed here


def parse_cuts_text(text: str) -> List[CutPoint]:
    """
    Pair pts_time with next lavfi.scene_score line.
    """
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
            score = float(m_s.group(1))
            cuts.append(CutPoint(t=pending_t, score=score, reason="hard"))
            pending_t = None

    cuts.sort(key=lambda c: c.t)
    return cuts


def jitter_merge(cuts: List[CutPoint], merge_gap: float) -> List[CutPoint]:
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
    cuts: List[CutPoint],
    th_soft: float,
    window_sec: float = 1.0,
    min_count: int = 3
) -> List[CutPoint]:
    """
    If >= min_count soft-qualified cuts within window, keep only peak (max score).
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
            if peak.reason != "strong":
                peak.reason = "transition"
            result.append(peak)
            i = j
        else:
            result.append(c)
            i += 1

    result.sort(key=lambda x: x.t)
    result = jitter_merge(result, merge_gap=0.001)
    return result


def enforce_min_shot_len(
    cuts: List[CutPoint],
    duration: float,
    min_shot_len: float,
    th_keep_strong: float
) -> List[CutPoint]:
    """
    Remove weak cuts that create too-short shots, unless strong.
    """
    if not cuts:
        return []

    for c in cuts:
        if c.score >= th_keep_strong:
            c.reason = "strong"

    kept = sorted(cuts, key=lambda c: c.t)

    def boundaries(_kept: List[CutPoint]) -> List[float]:
        ts = [c.t for c in _kept if 0.001 < c.t < duration - 0.001]
        ts.sort()
        return [0.0] + ts + [duration]

    changed = True
    while changed and kept:
        changed = False
        b = boundaries(kept)

        remove_idx: Optional[int] = None

        # segment ends at kept[i-1]
        for seg_i in range(1, len(b) - 1):
            seg_len = b[seg_i] - b[seg_i - 1]
            if seg_len < min_shot_len:
                cut = kept[seg_i - 1]
                if cut.score < th_keep_strong:
                    remove_idx = seg_i - 1
                    break

        # segment starts at a cut: remove weaker neighbor
        if remove_idx is None:
            for seg_i in range(1, len(b) - 1):
                seg_len = b[seg_i + 1] - b[seg_i]
                if seg_len < min_shot_len:
                    left_cut = kept[seg_i - 1]
                    right_cut = kept[seg_i] if seg_i < len(kept) else None

                    candidates = []
                    if left_cut.score < th_keep_strong:
                        candidates.append((seg_i - 1, left_cut.score))
                    if right_cut and right_cut.score < th_keep_strong:
                        candidates.append((seg_i, right_cut.score))

                    if candidates:
                        candidates.sort(key=lambda x: x[1])
                        remove_idx = candidates[0][0]
                        break

        if remove_idx is not None:
            kept.pop(remove_idx)
            changed = True

    return kept


def build_shots_from_cuts(cuts: List[CutPoint], duration: float) -> List[Tuple[float, float]]:
    times = [c.t for c in sorted(cuts, key=lambda x: x.t)]
    cleaned = []
    for t in times:
        if 0.001 < t < duration - 0.001:
            if not cleaned or abs(t - cleaned[-1]) > 1e-6:
                cleaned.append(t)
    b = [0.0] + cleaned + [duration]
    shots = []
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]
        if e > s:
            shots.append((s, e))
    return shots


def apply_end_safety_pad(
    shots: List[Tuple[float, float]],
    fps: float,
    min_keep: float = 0.10
) -> List[Tuple[float, float]]:
    """
    Prevent 'next shot first frame leaks into previous shot':
    subtract 1 frame from end of each segment (except last).
    """
    if not shots:
        return shots
    one_frame = 1.0 / max(1.0, fps)

    padded = []
    for i, (s, e) in enumerate(shots):
        if i == len(shots) - 1:
            padded.append((s, e))
            continue
        e2 = e - one_frame
        # ensure not inverted / too short
        if e2 <= s + min_keep:
            e2 = max(s + min_keep, e - (one_frame * 0.5))
            if e2 <= s + 0.01:
                e2 = e  # give up, keep original
        padded.append((s, e2))
    return padded


def build_time_fallback_shots(duration: float, seg_sec: float) -> List[Tuple[float, float]]:
    if seg_sec <= 0:
        return [(0.0, duration)]
    shots = []
    t = 0.0
    while t < duration - 1e-6:
        e = min(duration, t + seg_sec)
        shots.append((t, e))
        t = e
    return shots


def ffmpeg_cut_segment(
    input_path: str,
    out_path: str,
    start: float,
    end: float,
    mode: str,
    crf: int,
    preset: str,
    audio_bitrate: str
) -> None:
    """
    Cutting strategy:
    - reencode (accurate): -i first, then -ss, then -t (duration)
    - copy (fast): keyframe-limited, may leak frames; use only when you accept that.
    """
    start = max(0.0, float(start))
    end = max(start, float(end))
    dur = max(0.0, end - start)

    start_str = f"{start:.6f}"
    dur_str = f"{dur:.6f}"

    if mode == "copy":
        # Fast but keyframe-limited; less accurate boundaries.
        # Use -ss before -i for speed, and -t for duration.
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-ss", start_str,
            "-i", input_path,
            "-t", dur_str,
            "-c", "copy",
            out_path
        ]
    else:
        # Accurate: decode from exact timestamp (slower but correct)
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", input_path,
            "-ss", start_str,
            "-t", dur_str,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            out_path
        ]

    code, out, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg cut failed: {out_path}\nCMD: {' '.join(cmd)}\nERR:\n{err}")


def zip_outputs(output_dir: str, zip_path: str) -> None:
    """
    Create zip containing all mp4 + shots.json under output_dir.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(output_dir):
            for fn in files:
                if fn.lower().endswith(".mp4") or fn.lower() == "shots.json":
                    abs_path = os.path.join(root, fn)
                    rel_path = os.path.relpath(abs_path, os.path.dirname(output_dir))
                    z.write(abs_path, rel_path)


def main():
    parser = argparse.ArgumentParser(description="FFmpeg-based shot splitter (optimized, zip output).")
    parser.add_argument("-i", "--input", required=True, help="Input video file path")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for shot clips")
    parser.add_argument("--zip_path", default="shots.zip", help="Output zip path (default: shots.zip)")
    parser.add_argument("--mode", choices=["reencode", "copy"], default="reencode",
                        help="Cut mode: reencode (accurate) or copy (fast, keyframe-limited)")
    parser.add_argument("--save_meta", action="store_true", help="Save shots metadata as shots.json")

    # Scene detection + rules params (your current run: th_hard=0.30 th_soft=0.21)
    parser.add_argument("--th_hard", type=float, default=0.40, help="Hard cut threshold for scene score")
    parser.add_argument("--th_soft", type=float, default=0.28, help="Soft threshold for transition clustering")
    parser.add_argument("--min_shot_len", type=float, default=1.6, help="Minimum shot length in seconds")
    parser.add_argument("--merge_gap", type=float, default=0.25, help="Merge gap for nearby cuts (seconds)")
    parser.add_argument("--th_keep_strong", type=float, default=0.55, help="Strong cut keep threshold")

    parser.add_argument("--no_denoise", action="store_true", help="Disable hqdn3d denoise before scene detection")
    parser.add_argument("--blur", action="store_true", help="Add light blur before scene detection")

    parser.add_argument("--no_auto_relax", action="store_true", help="Disable auto relax threshold fallback")
    parser.add_argument("--fallback_time_sec", type=float, default=0.0,
                        help="If still no cuts found, split by fixed seconds (e.g., 2.5). 0=disabled")

    # End safety pad
    parser.add_argument("--no_end_pad", action="store_true",
                        help="Disable end safety pad (subtract 1 frame from each segment end except last)")

    # Reencode params
    parser.add_argument("--crf", type=int, default=18, help="CRF for x264 when reencode mode")
    parser.add_argument("--preset", default="veryfast", help="x264 preset when reencode mode")
    parser.add_argument("--audio_bitrate", default="128k", help="Audio bitrate when reencode mode")

    args = parser.parse_args()

    require_ffmpeg()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    duration = get_duration_sec(input_path)
    fps = get_fps(input_path)

    def detect_with_threshold(th: float) -> List[CutPoint]:
        text = extract_candidate_cuts_text(
            input_path=input_path,
            th_hard=th,
            use_denoise=not args.no_denoise,
            add_blur=args.blur
        )
        return parse_cuts_text(text)

    cuts = detect_with_threshold(args.th_hard)
    print(f"Parsed cuts: {len(cuts)} (th_hard={args.th_hard})")

    if (not args.no_auto_relax) and (not cuts):
        for th in [0.30, 0.25, 0.20, 0.18, 0.15, 0.12]:
            cuts = detect_with_threshold(th)
            print(f"Parsed cuts: {len(cuts)} (auto th_hard={th})")
            if cuts:
                args.th_hard = th
                args.th_soft = min(args.th_soft, th * 0.70)
                break

    used_fallback = False
    if not cuts and args.fallback_time_sec > 0:
        used_fallback = True
        shots = build_time_fallback_shots(duration, args.fallback_time_sec)
    else:
        cuts = jitter_merge(cuts, merge_gap=args.merge_gap)
        cuts = soft_transition_cluster(cuts, th_soft=args.th_soft, window_sec=1.0, min_count=3)
        cuts = jitter_merge(cuts, merge_gap=args.merge_gap)
        cuts = enforce_min_shot_len(
            cuts=cuts,
            duration=duration,
            min_shot_len=args.min_shot_len,
            th_keep_strong=args.th_keep_strong
        )
        shots = build_shots_from_cuts(cuts, duration)

    # 🔧 Critical fix for your “shot tail leaks next shot” issue:
    # subtract 1 frame from each end (except last)
    if (not args.no_end_pad) and (not used_fallback) and shots:
        shots = apply_end_safety_pad(shots, fps=fps, min_keep=0.10)

    meta = {
        "input": os.path.abspath(input_path),
        "duration": duration,
        "fps": fps,
        "used_fallback_time_split": used_fallback,
        "params": {
            "th_hard": args.th_hard,
            "th_soft": args.th_soft,
            "min_shot_len": args.min_shot_len,
            "merge_gap": args.merge_gap,
            "th_keep_strong": args.th_keep_strong,
            "mode": args.mode,
            "no_denoise": bool(args.no_denoise),
            "blur": bool(args.blur),
            "fallback_time_sec": args.fallback_time_sec,
            "end_pad_enabled": (not args.no_end_pad)
        },
        "cuts": [{"t": c.t, "score": c.score, "reason": c.reason} for c in cuts],
        "shots": []
    }

    pad = max(3, int(math.log10(max(1, len(shots)))) + 1)

    for idx, (s, e) in enumerate(shots, start=1):
        out_name = f"shot_{idx:0{pad}d}_{s:.2f}-{e:.2f}.mp4"
        out_path = os.path.join(args.output_dir, out_name)

        ffmpeg_cut_segment(
            input_path=input_path,
            out_path=out_path,
            start=s,
            end=e,
            mode=args.mode,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate
        )

        meta["shots"].append({
            "id": idx,
            "start": s,
            "end": e,
            "duration": e - s,
            "file": out_name
        })

    if args.save_meta:
        meta_path = os.path.join(args.output_dir, "shots.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # Create zip
    zip_outputs(args.output_dir, args.zip_path)

    print(f"Done. Shots: {len(shots)}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    if args.save_meta:
        print(f"Metadata: {os.path.join(os.path.abspath(args.output_dir), 'shots.json')}")
    print(f"Zip: {os.path.abspath(args.zip_path)}")
    if used_fallback:
        print(f"Note: Used fallback time split: every {args.fallback_time_sec}s")


if __name__ == "__main__":
    main()