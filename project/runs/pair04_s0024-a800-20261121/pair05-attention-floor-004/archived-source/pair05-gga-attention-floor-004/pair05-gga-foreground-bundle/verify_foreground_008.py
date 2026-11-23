"""Verify or recover the isolated Pair05 foreground result."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from stage_foreground_008 import FROZEN
from verify_floor_diagnostics import EXPECTED_CALLS, verify_samples
from verify_ab_008 import RECEIPT_NAME, TAKE, publish_receipt, verify_audit, verify_video


PLAN = Path("plan/pair05-gga-routes.npz")
MARKER = re.compile(rb"Pair05 foreground GGA attention calls: ([1-9][0-9]*)")


def verify_log(log):
    payload = log.read_bytes()
    # tqdm often writes a final carriage return just before infer.py's summary.
    markers = [MARKER.fullmatch(line.strip()) for line in payload.splitlines()
               if line.strip().startswith(b"Pair05 foreground GGA attention calls:")]
    if len(markers) != 1 or markers[0] is None:
        raise ValueError("The inference log must record exactly one positive foreground GGA attention count.")
    count = int(markers[0][1])
    if count != EXPECTED_CALLS[-1]:
        raise ValueError(f"Expected {EXPECTED_CALLS[-1]} foreground attention calls, received {count}.")
    diagnostics = verify_samples(payload)
    return {"attention_calls": count, "plan_sha256": FROZEN[PLAN], "floor_diagnostics": diagnostics,
            "inference_log": log.name, "inference_log_sha256": hashlib.sha256(payload).hexdigest()}


def verify_result(out, fingerprint, ffprobe, ffmpeg, complete):
    receipt_path = out / RECEIPT_NAME
    if (out / "fingerprint.txt").read_text(encoding="ascii").strip() != fingerprint:
        raise ValueError("The stored attempt belongs to a different bundle or runtime.")
    if (out / "inference-exit-code.txt").read_text(encoding="ascii").strip() != "0":
        raise ValueError("Inference did not exit successfully.")
    if complete and os.path.lexists(receipt_path):
        raise ValueError("A result receipt already exists.")
    foreground = verify_log(out / "inference.log")
    receipt = {
        "status": "verified", "arm": "foreground", "take": TAKE, "fingerprint": fingerprint,
        "inference_exit_code": 0, "foreground_gga": foreground,
        "runtime_audit": verify_audit(out / "runtime_audit.json"),
        "raw": verify_video(out / f"{TAKE}.mp4", ffprobe, ffmpeg),
    }
    if complete:
        publish_receipt(receipt_path, receipt)
    elif json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("The stored receipt differs from current inference evidence.")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.fingerprint):
        parser.error("--fingerprint must be a lowercase SHA-256 digest")
    try:
        result = verify_result(args.out, args.fingerprint, args.ffprobe, args.ffmpeg, args.complete)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError,
            subprocess.SubprocessError) as error:
        print(f"PAIR05_FOREGROUND_VERIFICATION_FAILED {error}", file=sys.stderr)
        return 1
    print(f"PAIR05_FOREGROUND_VERIFIED {result['raw']['sha256']} "
          f"attention_calls={result['foreground_gga']['attention_calls']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
