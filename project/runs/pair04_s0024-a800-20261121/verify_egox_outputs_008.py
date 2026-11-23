#!/usr/bin/env python3
"""Verify recovered EgoX MP4s and export the generated right-hand view."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


TAKES = ("pair04_s0024", "pair01_s0288", "pair02_s0000", "pair05_s0024")
FPS = 30
FRAME_COUNT = 49
EGO_SIZE = 448
EXO_WIDTH = 784
CROP = f"crop={EGO_SIZE}:{EGO_SIZE}:{EXO_WIDTH}:0"


def run(command):
    result = subprocess.run(command, capture_output=True, timeout=120)
    if result.returncode or result.stderr.strip():
        error = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"{command[0]} returned {result.returncode}. {error[:2000]}")
    return result.stdout


def ffmpeg(video, *arguments):
    return run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n", "-err_detect", "explode",
        "-i", str(video), "-map", "0:v:0", *arguments,
    ])


def probe(video, width, *, raw_mux=False):
    result = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v", "-show_frames", "-show_streams",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,time_base,duration_ts,pix_fmt:"
        "frame=pts,duration,pkt_duration,width,height",
        "-of", "json", str(video),
    ]))
    streams = result.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream, found {len(streams)}")
    stream, = streams
    if (stream["codec_name"], stream["width"], stream["height"], Fraction(stream["avg_frame_rate"])) != (
        "h264", width, EGO_SIZE, FPS
    ):
        raise ValueError(f"Unexpected codec, dimensions, or frame rate for {video.name}: {stream}")
    frames = result.get("frames", [])
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} decoded frames, found {len(frames)}")
    time_base = Fraction(stream["time_base"])
    frame_duration = Fraction(1, FPS)
    pts = []
    for index, frame in enumerate(frames):
        timestamp = int(frame["pts"]) * time_base
        decoded_duration = int(frame.get("duration", frame.get("pkt_duration"))) * time_base
        if (frame["width"], frame["height"], timestamp, decoded_duration) != (
            width, EGO_SIZE, index * frame_duration, frame_duration
        ):
            raise ValueError(f"Unexpected dimensions or timing at frame {index}: {frame}")
        pts.append(str(timestamp))
    duration = int(stream["duration_ts"]) * time_base
    duration_error = duration - FRAME_COUNT * frame_duration
    # The official raw MP4 writer quantizes its edit-list duration to milliseconds.
    allowed_error = Fraction(1, 1000) if raw_mux else 0
    if abs(duration_error) > allowed_error:
        raise ValueError(f"Unexpected stream duration: {duration} s")
    return {
        "stream": stream, "frames": len(frames), "pts_seconds": pts,
        "frame_duration_seconds": str(frame_duration), "duration_seconds": str(duration),
        "stream_duration_error_seconds": str(duration_error),
    }


def decode_gray(video, *, source_crop=False):
    filters = f"{CROP},format=gray" if source_crop else "format=gray"
    data = ffmpeg(video, "-vf", filters, "-pix_fmt", "gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-")
    frame_bytes = EGO_SIZE * EGO_SIZE
    if len(data) != FRAME_COUNT * frame_bytes:
        raise ValueError(f"Unexpected decoded gray payload length for {video.name}: {len(data)} bytes")
    return [data[offset:offset + frame_bytes] for offset in range(0, len(data), frame_bytes)]


def check_content(frames):
    standard_deviations = []
    digests = []
    for index, frame in enumerate(frames):
        histogram = Counter(frame)
        mean = sum(level * count for level, count in histogram.items()) / len(frame)
        stddev = math.sqrt(sum((level - mean) ** 2 * count for level, count in histogram.items()) / len(frame))
        if stddev <= 1.0 or max(histogram) - min(histogram) <= 8:
            raise ValueError(f"Frame {index} is flat or blank (grayscale standard deviation {stddev:.3f})")
        standard_deviations.append(stddev)
        digests.append(hashlib.sha256(frame).hexdigest())
    distinct = len(set(digests))
    if distinct < 2:
        raise ValueError("All decoded grayscale frames are identical")
    return {
        "minimum_frame_gray_stddev": round(min(standard_deviations), 6),
        "distinct_gray_frames": distinct,
        "adjacent_duplicate_gray_frames": sum(a == b for a, b in zip(digests, digests[1:])),
    }


def check_crop(source_frames, export_frames):
    frame_mae = [
        sum(abs(a - b) for a, b in zip(source, export, strict=True)) / len(source)
        for source, export in zip(source_frames, export_frames, strict=True)
    ]
    if max(frame_mae) > 12:
        raise ValueError(f"Export diverges from the source crop (maximum frame mean gray error {max(frame_mae):.3f})")
    return {"mean_gray_error": round(sum(frame_mae) / len(frame_mae), 6), "max_frame_gray_error": round(max(frame_mae), 6)}


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, payload):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.close()
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def make_contact_sheet(video, output):
    selection = r"select=eq(n\,0)+eq(n\,24)+eq(n\,48),scale=336:336,tile=3x1:padding=8:margin=8:color=white"
    ffmpeg(video, "-vf", selection, "-fps_mode", "passthrough", "-frames:v", "1", "-q:v", "3", "-update", "1", str(output))
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("FFmpeg did not create a contact sheet")


def verify_take(take, raw, output_dir, contact_sheets):
    target = output_dir / f"{take}_ego.mp4"
    receipt_path = output_dir / f"{take}.receipt.json"
    sheet = output_dir / f"{take}.contact.jpg"
    raw_hash = file_sha256(raw)
    raw_probe = probe(raw, EXO_WIDTH + EGO_SIZE, raw_mux=True)
    source_frames = decode_gray(raw, source_crop=True)
    source_content = check_content(source_frames)
    existing = target.exists()
    if existing:
        if not receipt_path.is_file():
            raise ValueError(f"Existing export {target.name} has no verification receipt. Use a new output directory.")
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous["raw"]["sha256"] != raw_hash or previous["export"]["sha256"] != file_sha256(target):
            raise ValueError(f"Existing export {target.name} or its source has changed. Use a new output directory.")

    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staged_video = Path(staging) / target.name
        video = target if existing else staged_video
        if not existing:
            ffmpeg(
                raw, "-vf", CROP, "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-enc_time_base", "1:30",
                "-video_track_timescale", "30000", "-movie_timescale", "30000", "-movflags", "+faststart", str(video),
            )
        output_probe = probe(video, EGO_SIZE)
        export_frames = decode_gray(video)
        output_content = check_content(export_frames)
        fidelity = check_crop(source_frames, export_frames)
        if file_sha256(raw) != raw_hash:
            raise ValueError(f"Raw video changed during verification: {raw.name}")
        export_hash = file_sha256(video)
        staged_sheet = Path(staging) / sheet.name
        contact = None
        if contact_sheets:
            if sheet.exists():
                prior_contact = previous.get("contact_sheet") if existing else None
                if not prior_contact or prior_contact["sha256"] != file_sha256(sheet):
                    raise ValueError(f"Existing contact sheet {sheet.name} conflicts with this export. Use a new output directory.")
            else:
                make_contact_sheet(video, staged_sheet)
            contact = {"file": sheet.name, "frame_indices": [0, 24, 48], "sha256": file_sha256(sheet if sheet.exists() else staged_sheet)}
        if not existing:
            os.link(staged_video, target)
        if staged_sheet.exists():
            os.link(staged_sheet, sheet)
        receipt = {
            "take": take,
            "status": "verified",
            "raw": {"file": raw.name, "sha256": raw_hash, "probe": raw_probe, "right_crop_content": source_content},
            "export": {"file": target.name, "sha256": export_hash, "probe": output_probe, "content": output_content},
            "crop": {"x": EXO_WIDTH, "y": 0, "width": EGO_SIZE, "height": EGO_SIZE, "libx264_crf": 18, **fidelity},
            "contact_sheet": contact if contact_sheets else (previous.get("contact_sheet") if existing else None),
        }
        write_json(receipt_path, receipt)
    return {"take": take, "status": "verified", "video": target.name, "receipt": receipt_path.name, "sha256": export_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="Directory containing each recovered <take>.mp4")
    parser.add_argument("output_dir", type=Path, help="Directory for verified <take>_ego.mp4, receipts and summary")
    parser.add_argument("--contact-sheets", action="store_true", help="Also export frames 0, 24 and 48 as a JPEG per take")
    options = parser.parse_args()
    raw_dir = options.raw_dir.resolve()
    output_dir = options.output_dir.resolve()
    if not raw_dir.is_dir() or raw_dir == output_dir:
        parser.error("RAW_DIR must be an existing directory distinct from OUTPUT_DIR")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for take in TAKES:
        raw = raw_dir / f"{take}.mp4"
        if not raw.exists():
            result = {"take": take, "status": "missing", "expected_file": raw.name}
        else:
            try:
                result = verify_take(take, raw, output_dir, options.contact_sheets)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                result = {"take": take, "status": "rejected", "error": str(error)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    statuses = Counter(result["status"] for result in results)
    complete = statuses["verified"] == len(TAKES)
    write_json(output_dir / "verification.json", {"complete": complete, "counts": dict(statuses), "takes": results})
    if complete:
        return 0
    return 1 if statuses["rejected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
