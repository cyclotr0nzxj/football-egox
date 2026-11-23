"""Verify the single B run and original full-width video."""

import argparse
import json
import math
from pathlib import Path
import re

from pair05_clip_gate import CALLS, SAMPLED_CALLS
from verify_ab_008 import publish_receipt, verify_audit, verify_video


INPUTS = {"latents", "condition", "image", "prompt", "negative_prompt", "timesteps", "geometry", "rays"}


def read_evidence(path):
    events = []
    for line in path.read_bytes().splitlines():
        line = line.strip()
        for marker in (b"PAIR05_CLIP_INITIAL", b"PAIR05_CLIP_FINISH"):
            if line.startswith(marker + b" "):
                events.append((marker, json.loads(line.partition(b" ")[2])))
    if [marker for marker, _ in events] != [b"PAIR05_CLIP_INITIAL", b"PAIR05_CLIP_FINISH"]:
        raise ValueError("Expected one ordered set of B-only evidence.")
    inputs, trace = events[0][1], events[1][1]
    if inputs.keys() != INPUTS or any(not re.fullmatch("[0-9a-f]{64}", value) for value in inputs.values()):
        raise ValueError("Required initialization digests are absent.")
    if ((trace["image_calls"], trace["foreground_calls"]) != (CALLS, CALLS)
            or set(map(int, trace["samples"])) != set(SAMPLED_CALLS)):
        raise ValueError("The candidate lacks full image and foreground attention coverage.")
    for before, after in trace["samples"].values():
        if not math.isfinite(before) or before <= 0 or after != 0:
            raise ValueError("Image contribution does not match the B-only gate.")
    return {"initial": inputs, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    out = args.out
    if ((out / "fingerprint.txt").read_text().strip() != args.fingerprint
            or (out / "inference-exit-code.txt").read_text().strip() != "0"):
        raise ValueError("Run identity or successful exit is missing.")
    evidence = read_evidence(out / "inference.log")
    receipt = {"status": "verified", "arm": "B", "take": "pair05_s0024", "fingerprint": args.fingerprint,
               **evidence, "runtime_audit": verify_audit(out / "runtime_audit.json"),
               "raw": verify_video(out / "pair05_s0024.mp4", args.ffprobe, args.ffmpeg)}
    path = out / "verified-result.json"
    if args.complete:
        publish_receipt(path, receipt)
    elif json.loads(path.read_text()) != receipt:
        raise ValueError("Stored receipt differs from current evidence.")
    print("PAIR05_CLIP_B_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
