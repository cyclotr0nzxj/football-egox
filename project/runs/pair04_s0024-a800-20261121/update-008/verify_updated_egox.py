#!/usr/bin/env python3
"""Verify the recovered three-take EgoX batch and export its generated views."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_egox_outputs_008 import file_sha256, verify_take, write_json


TAKES = ("pair04_s0024", "pair05_s0024", "pair01_s0288")


def read_batch_result(path):
    payload = path.read_bytes()
    batch = json.loads(payload)
    rows = batch["takes"]
    if tuple(row["take"] for row in rows) != TAKES:
        raise ValueError("Batch result must contain exactly the three updated takes in launch order.")
    for row in rows:
        if row["file"] != f"{row['take']}.mp4" or row["status"] not in {"output_verified", "output_unverified"}:
            raise ValueError(f"Invalid remote result for {row['take']}.")
        if row["status"] == "output_verified" and not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError(f"Missing or invalid remote SHA-256 for {row['take']}.")

    status = batch["status"]
    exit_code = batch["inference_exit_code"]
    adapter = batch["adapter_evidence"]
    complete = (
        exit_code == 0 and adapter == "runtime_audit_verified"
        and all(row["status"] == "output_verified" for row in rows)
    )
    if type(exit_code) is not int or status != ("complete" if complete else "incomplete"):
        raise ValueError("Remote batch status, exit code, or adapter evidence is inconsistent.")
    return rows, {
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": status,
        "inference_exit_code": exit_code,
        "adapter_evidence": adapter,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="Directory containing recovered <take>.mp4 files")
    parser.add_argument("output_dir", type=Path, help="Directory for verified <take>_ego.mp4 files and receipts")
    parser.add_argument("--batch-result", type=Path, required=True, help="Recovered batch-result.json from the launch")
    parser.add_argument("--contact-sheets", action="store_true", help="Also export frames 0, 24 and 48 as a JPEG per take")
    options = parser.parse_args()
    raw_dir = options.raw_dir.resolve()
    output_dir = options.output_dir.resolve()
    if not raw_dir.is_dir() or raw_dir == output_dir:
        parser.error("RAW_DIR must be an existing directory distinct from OUTPUT_DIR")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required on PATH")

    summary = output_dir / "verification.json"
    try:
        rows, batch = read_batch_result(options.batch_result)
        if summary.exists():
            previous = json.loads(summary.read_text(encoding="utf-8"))
            if previous.get("batch_result", {}).get("sha256") != batch["sha256"]:
                raise ValueError("OUTPUT_DIR contains a different verification report. Use a new directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.error(str(error))

    results = []
    for row in rows:
        take = row["take"]
        raw = raw_dir / row["file"]
        if not raw.exists():
            result = {"take": take, "status": "missing", "expected_file": raw.name}
        else:
            try:
                if row["status"] != "output_verified":
                    raise ValueError("The remote batch has no verified output for this take.")
                if not raw.is_file() or file_sha256(raw) != row["sha256"]:
                    raise ValueError("Recovered MP4 does not match the remote SHA-256.")
                result = verify_take(take, raw, output_dir, options.contact_sheets)
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as error:
                result = {"take": take, "status": "rejected", "error": str(error)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)

    counts = Counter(result["status"] for result in results)
    complete = batch["status"] == "complete" and counts["verified"] == len(TAKES)
    write_json(summary, {"complete": complete, "counts": dict(counts), "batch_result": batch, "takes": results})
    if complete:
        return 0
    return 1 if counts["rejected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
