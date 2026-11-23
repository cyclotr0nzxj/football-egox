#!/usr/bin/env python3
"""Verify or recheck a completed Pair05 GGA comparison arm."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


TAKE = "pair05_s0024"
WIDTH, HEIGHT = 1232, 448
FPS, FRAME_COUNT = 30, 49
RECEIPT_NAME = "verified-result.json"


def run_video_tool(command):
    result = subprocess.run(command, capture_output=True, timeout=90)
    if result.returncode or result.stderr.strip():
        error = result.stderr.decode(errors="replace").strip()[:2000]
        raise ValueError(f"{Path(command[0]).name} returned {result.returncode}: {error}")
    return result.stdout


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_video(video, ffprobe, ffmpeg):
    before = sha256(video)
    probe = json.loads(run_video_tool([
        ffprobe, "-v", "error", "-select_streams", "v", "-show_frames", "-show_streams",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,time_base:frame=pts,width,height",
        "-of", "json", str(video),
    ]))
    stream, = probe["streams"]
    stream_format = (stream["codec_name"], stream["width"], stream["height"], Fraction(stream["avg_frame_rate"]))
    if stream_format != ("h264", WIDTH, HEIGHT, FPS):
        raise ValueError(f"Unexpected video codec, dimensions or frame rate: {stream_format}")
    frames = probe["frames"]
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} decoded frames, found {len(frames)}")
    time_base = Fraction(stream["time_base"])
    pts = []
    for index, frame in enumerate(frames):
        timestamp = int(frame["pts"]) * time_base
        if (frame["width"], frame["height"], timestamp) != (WIDTH, HEIGHT, Fraction(index, FPS)):
            raise ValueError(f"Unexpected dimensions or timestamp at frame {index}: {frame}")
        pts.append(str(timestamp))

    run_video_tool([
        ffmpeg, "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-i", str(video),
        "-map", "0:v:0", "-f", "null", "-",
    ])
    if sha256(video) != before:
        raise ValueError(f"Video changed during verification: {video.name}")
    return {
        "file": video.name, "sha256": before, "codec": "h264", "width": WIDTH, "height": HEIGHT,
        "fps": FPS, "frames": FRAME_COUNT, "pts_seconds": pts,
    }


def verify_audit(path):
    payload = path.read_bytes()
    report = json.loads(payload)
    load, fuse = report["events"]
    if (load["operation"], fuse["operation"]) != ("load_lora_weights", "fuse_lora"):
        raise ValueError("Runtime audit must record loading and fusing the LoRA in order.")
    if load["kwargs"]["weight_name"] != "pytorch_lora_weights.safetensors" or fuse["kwargs"] != {
        "components": ["transformer"], "lora_scale": 1.0,
    }:
        raise ValueError("Runtime audit does not match the official LoRA weight file or fusion settings.")
    for event in (load, fuse):
        active = event["active_adapters"]
        ranks = event["peft_config"]
        if not active or set(active) != set(ranks) or any(config["r"] != 256 for config in ranks.values()):
            raise ValueError("Runtime audit must show an active rank-256 LoRA adapter.")

    elapsed = report["elapsed_seconds"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Runtime audit has an invalid elapsed time.")
    resources = {name: report[name] for name in (
        "max_rss_kib", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes",
    )}
    if any(type(value) is not int or value <= 0 for value in resources.values()):
        raise ValueError("Runtime audit has an invalid peak RAM or CUDA memory value.")
    if resources["max_cuda_allocated_bytes"] > resources["max_cuda_reserved_bytes"]:
        raise ValueError("Peak allocated CUDA memory exceeds peak reserved CUDA memory.")
    return {
        "file": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
        "elapsed_seconds": elapsed, **resources,
    }


def publish_receipt(path, receipt):
    # Publish the finished receipt atomically and refuse to replace prior evidence.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--complete", action="store_true", help="Publish the first successful receipt.")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.fingerprint):
        parser.error("--fingerprint must be a lowercase SHA-256 digest")

    receipt_path = args.out / RECEIPT_NAME
    try:
        if args.complete:
            if os.path.lexists(receipt_path):
                raise ValueError(f"A result receipt already exists at {receipt_path}")
        else:
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (previous["arm"], previous["fingerprint"]) != (args.arm, args.fingerprint):
                raise ValueError("Stored result receipt has a different arm or experiment fingerprint.")

        exit_code = int((args.out / "inference-exit-code.txt").read_text(encoding="ascii").strip())
        if exit_code != 0:
            raise ValueError(f"Inference exited with {exit_code}; keep its outputs for diagnosis.")
        receipt = {
            "status": "verified", "arm": args.arm, "take": TAKE, "fingerprint": args.fingerprint,
            "inference_exit_code": exit_code,
            "runtime_audit": verify_audit(args.out / "runtime_audit.json"),
            "raw": verify_video(args.out / f"{TAKE}.mp4", args.ffprobe, args.ffmpeg),
        }
        if args.complete:
            publish_receipt(receipt_path, receipt)
        elif previous != receipt:
            raise ValueError("Stored result receipt differs from current video or audit evidence.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError,
            subprocess.SubprocessError) as error:
        print(f"PAIR01_ARM_{args.arm}_VERIFICATION_FAILED {error}", file=sys.stderr)
        return 1

    print(f"PAIR01_ARM_{args.arm}_VERIFIED {receipt['raw']['sha256']} {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
