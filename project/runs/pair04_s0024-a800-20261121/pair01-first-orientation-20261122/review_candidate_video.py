#!/usr/bin/env python3
"""Verify and prepare a local review of the recovered Pair01 candidate video."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUN_ROOT / "pair01-gga-ab"))
from review_pair01_ab import (  # noqa: E402
    REFERENCES, TAKE, check_reference, read_remote_receipt, review_frames,
)
from verify_egox_outputs_008 import (  # noqa: E402
    EGO_SIZE, EXO_WIDTH, FPS, FRAME_COUNT, file_sha256, verify_take, write_json,
)


WINDOWS = {
    "F00-F04": tuple(range(0, 5)),
    "F20-F28": tuple(range(20, 29)),
    "F36-F43": tuple(range(36, 44)),
}
FRAME_INDICES = [index for window in WINDOWS.values() for index in window]
FIRST_EXPORT = f"four-egox-outputs-008/verified/{TAKE}_ego.mp4"
FIRST_EXPORT_SHA256 = "515214b8d487c6edec53a7f5b8a0d39f1ab2bdb866b098759c099b9fd3512a11"
FIXTURE_RAW_SHA256 = "c7cceb1c8a8e7587a735f9cd15e0e133a9527874fd7b67b10e899b90ce50ac58"
FIXTURE_PRIOR_SHA256 = REFERENCES["Geometry prior"][1]


def read_frames(path, sha, width, fps):
    if file_sha256(path) != sha:
        raise ValueError(f"Unexpected SHA-256 for {path}.")
    check_reference(path, width, fps)
    frames = review_frames(path, width, FRAME_INDICES)
    if file_sha256(path) != sha:
        raise ValueError(f"Video changed during verification: {path}.")
    return frames


def make_sheets(stage, columns, smoke):
    labels = ("Original EXO (25 fps)", "Formal prior (30 fps)",
              "First generated export (30 fps)", "Candidate export (30 fps)")
    if smoke:
        labels = (labels[0], "Historical prior (30 fps)", labels[2], "Historical arm A (30 fps)")
    widths = (EXO_WIDTH, EGO_SIZE, EGO_SIZE, EGO_SIZE)
    title_font = ImageFont.load_default(size=28)
    label_font = ImageFont.load_default(size=22)
    margin, gap, header, row_height = 16, 12, 116, EGO_SIZE + 40
    width = 2 * margin + sum(widths) + gap * (len(widths) - 1)
    for name, indices in WINDOWS.items():
        sheet = Image.new("RGB", (width, header + len(indices) * row_height + margin), "#101720")
        draw = ImageDraw.Draw(sheet)
        title = f"Pair01  {name}"
        if smoke:
            title += "  |  LOCAL SMOKE TEST WITH HISTORICAL FILES"
        draw.text((margin, 14), title, font=title_font, fill="white")
        draw.text((margin, 51), "Aligned by zero-based source frame index.", font=label_font, fill="#cbd5e1")
        x = margin
        for label, column_width in zip(labels, widths, strict=True):
            draw.text((x, 86), label, font=label_font, fill="white")
            x += column_width + gap
        for row, index in enumerate(indices):
            y = header + row * row_height
            draw.text((margin, y), f"F{index:02d}", font=label_font, fill="white")
            x = margin
            for frames, column_width in zip(columns, widths, strict=True):
                sheet.paste(frames[index], (x, y + 28))
                x += column_width + gap
        target = stage / f"{name}.png"
        sheet.save(target)
        yield {"file": target.name, "frame_indices": list(indices), "sha256": file_sha256(target)}


def write_manual_form(stage, video, smoke):
    lines = ["# Pair01 画面验收", ""]
    if smoke:
        lines.extend(["> 本机脚本自检。此处使用历史 A 的先验和成片。", ""])
    lines.extend([
        "每行按同一原始帧号对照。EXO 时刻为帧号除以 25，其他列为帧号除以 30。",
        "",
        "- [F00–F04](F00-F04.png) 核对开场是否保住首版构图，白鞋队友的位置和红鞋佩戴者的视角是否正确。",
        "- [F20–F28](F20-F28.png) 逐帧对照先验与成片，记录低头方向、球和红鞋的相对位置、触球是否清楚。",
        "- [F36–F43](F36-F43.png) 分别记录先验与成片中可辨球形的数量。遮挡或看不清时记「不明」，附帧号。",
        f"- [完整 49 帧成片]({video}) 逐帧查看全片，记录其它区间的身份混淆、球消失、重复球或跳变。",
        "",
        "先锁定结论，再使用 Nano 做独立盲评。首版来自四段批次，用于对照可见的构图和动作。",
        "",
        "| 帧号 | 正式先验 | 成片 |",
        "| --- | --- | --- |",
    ])
    lines.extend(f"| F{index:02d} | 待检查 | 待检查 |" for index in FRAME_INDICES)
    lines.extend(["", "全片额外观察与帧号　待填写", "", "结论　待人工验收", ""])
    (stage / "manual-review.md").write_text("\n".join(lines), encoding="utf-8")


def review(options):
    output = options.output_dir.absolute()
    if os.path.lexists(output):
        raise ValueError("Choose a new output directory to preserve existing evidence.")

    remote = read_remote_receipt(options.receipt, "A")
    if options.expected_fingerprint and remote["fingerprint"] != options.expected_fingerprint:
        raise ValueError("Remote receipt fingerprint does not match the expected run.")
    if options.raw.name != f"{TAKE}.mp4" or file_sha256(options.raw) != remote["raw_sha256"]:
        raise ValueError("Recovered raw MP4 file name or SHA-256 differs from the remote receipt.")
    prior_sha = file_sha256(options.prior)
    if options.expected_prior_sha256 and prior_sha != options.expected_prior_sha256:
        raise ValueError("Formal prior SHA-256 does not match the expected render.")
    historical_raw = remote["raw_sha256"] == FIXTURE_RAW_SHA256
    historical_prior = prior_sha == FIXTURE_PRIOR_SHA256
    if options.smoke_test and not (historical_raw and historical_prior):
        raise ValueError("The smoke test requires the pinned historical arm A and its formal prior.")
    if not options.smoke_test and (historical_raw or historical_prior):
        raise ValueError("Historical arm A or its prior was supplied. Use the candidate files or --smoke-test.")

    exo_relative, exo_sha, exo_width, exo_fps = REFERENCES["EXO"]
    columns = [
        read_frames(RUN_ROOT / exo_relative, exo_sha, exo_width, exo_fps),
        read_frames(options.prior, prior_sha, EGO_SIZE, FPS),
        read_frames(RUN_ROOT / FIRST_EXPORT, FIRST_EXPORT_SHA256, EGO_SIZE, FPS),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pair01-candidate-review-", dir=output.parent) as directory:
        stage = Path(directory)
        result = verify_take(TAKE, options.raw, stage, contact_sheets=False)
        receipt = json.loads((stage / result["receipt"]).read_text(encoding="utf-8"))
        raw = receipt["raw"]
        if (raw["sha256"], raw["probe"]["pts_seconds"]) != (remote["raw_sha256"], remote["raw_pts_seconds"]):
            raise ValueError("Local raw SHA-256 or exact frame PTS differs from the remote receipt.")
        columns.append(read_frames(stage / result["video"], result["sha256"], EGO_SIZE, FPS))
        sheets = list(make_sheets(stage, columns, options.smoke_test))
        write_manual_form(stage, result["video"], options.smoke_test)
        summary = {
            "take": TAKE,
            "mode": "local_historical_smoke" if options.smoke_test else "recovered_candidate",
            "numeric_verification": "passed",
            "manual_visual_verdict": "ineligible_smoke_test" if options.smoke_test else "pending",
            "frame_mapping": {"exo_fps": exo_fps, "render_fps": FPS, "frames": FRAME_COUNT,
                              "alignment": "same_zero_based_source_frame_index"},
            "references": {
                "exo": {"file_relative_to_run": exo_relative, "sha256": exo_sha},
                "first_generated_export": {"file_relative_to_run": FIRST_EXPORT, "sha256": FIRST_EXPORT_SHA256},
                "formal_prior": {"file": str(options.prior), "sha256": prior_sha,
                                 "expected_sha256_checked": options.expected_prior_sha256 is not None},
            },
            "remote_receipt": remote,
            "expected_fingerprint_checked": options.expected_fingerprint is not None,
            "verified_export": result,
            "sheets": sheets,
            "manual_review": "manual-review.md",
        }
        write_json(stage / "verification.json", summary)
        output.mkdir()
        published = []
        try:
            for source in sorted(stage.iterdir(), key=lambda path: path.name == "verification.json"):
                target = output / source.name
                os.link(source, target)
                published.append(target)
        except OSError:
            for target in published:
                target.unlink()
            output.rmdir()
            raise
    print(f"Pair01 file checks passed. Visual verdict {summary['manual_visual_verdict']}. {output}")


def digest(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("Expected a lowercase SHA-256 digest.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Recovered Pair01 raw GPU MP4")
    parser.add_argument("--receipt", type=Path, required=True, help="Remote arm A verified-result.json")
    parser.add_argument("--prior", type=Path, required=True, help="Formal prior used by the GPU run")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new local evidence directory")
    parser.add_argument("--expected-fingerprint", type=digest, help="Run fingerprint from preflight")
    parser.add_argument("--expected-prior-sha256", type=digest, help="Formal prior SHA-256 from render verification")
    parser.add_argument("--smoke-test", action="store_true", help="Test using the pinned historical arm A and prior")
    options = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required by the shared EgoX verification script.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required on PATH.")
    try:
        review(options)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError,
            subprocess.SubprocessError) as error:
        parser.exit(1, f"Pair01 review failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
