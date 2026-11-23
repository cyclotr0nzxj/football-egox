#!/usr/bin/env python3
"""Verify recovered Pair01 GGA output for A alone or both A and B."""

import argparse
from fractions import Fraction
import hashlib
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
sys.path.insert(0, str(RUN_ROOT))
from verify_egox_outputs_008 import (  # noqa: E402
    EGO_SIZE, EXO_WIDTH, FPS, FRAME_COUNT, ffmpeg, file_sha256, run, verify_take, write_json,
)


TAKE = "pair01_s0288"
ARMS = {"A": "opencv_original", "B": "aria_fisheye_inverse"}
RAW_NAME = f"{TAKE}.mp4"
GROUPS = {
    "F00": (0,),
    "F20-F28": tuple(range(20, 29)),
    "F36-F43": tuple(range(36, 44)),
}
REFERENCES = {
    "EXO": (
        f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4",
        "94a34d2fdccee5594be5cbc768d7c7656b065d10c5d347c80d5dd3d5a05ee2ab",
        EXO_WIDTH, 25,
    ),
    "Geometry prior": (
        f"update-008/output-008/priors/{TAKE}-final.mp4",
        "7846541ec6c90cda675a4f97b51b96cf162a364488cc99814c15a4bd18f87b97",
        EGO_SIZE, FPS,
    ),
}
HISTORICAL_RAW_SHA256 = "ba53df7e4b0cfb15eae5d374c974b8296749a373971a6c77ed25deffd9f57c2a"


def read_remote_receipt(path, arm):
    payload = path.read_bytes()
    receipt = json.loads(payload)
    if (receipt["status"], receipt["arm"], receipt["take"]) != ("verified", arm, TAKE):
        raise ValueError(f"Remote receipt does not identify a verified Pair01 arm {arm}.")
    if type(receipt["inference_exit_code"]) is not int or receipt["inference_exit_code"] != 0:
        raise ValueError(f"Remote receipt for {arm} does not record a successful inference exit.")
    raw = receipt["raw"]
    for digest in (receipt["fingerprint"], raw["sha256"], receipt["runtime_audit"]["sha256"]):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Remote receipt for {arm} contains an invalid SHA-256 digest.")
    expected_video = (RAW_NAME, "h264", EXO_WIDTH + EGO_SIZE, EGO_SIZE, FPS, FRAME_COUNT)
    if tuple(raw[field] for field in ("file", "codec", "width", "height", "fps", "frames")) != expected_video:
        raise ValueError(f"Remote receipt for {arm} contains unexpected raw video metadata.")
    return {
        "receipt_file": path.name,
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "fingerprint": receipt["fingerprint"],
        "reported_runtime_audit_sha256": receipt["runtime_audit"]["sha256"],
        "raw_sha256": raw["sha256"],
        "raw_pts_seconds": raw["pts_seconds"],
    }


def check_reference(path, width, fps):
    probe = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v", "-show_streams", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,time_base:frame=pts,width,height",
        "-of", "json", str(path),
    ]))
    stream, = probe["streams"]
    if (stream["width"], stream["height"], Fraction(stream["avg_frame_rate"])) != (width, EGO_SIZE, fps):
        raise ValueError(f"Unexpected reference stream dimensions or frame rate for {path.name}.")
    if len(probe["frames"]) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} reference frames in {path.name}.")
    time_base = Fraction(stream["time_base"])
    for index, frame in enumerate(probe["frames"]):
        timestamp = int(frame["pts"]) * time_base
        if (frame["width"], frame["height"], timestamp) != (width, EGO_SIZE, Fraction(index, fps)):
            raise ValueError(f"Unexpected reference dimensions or PTS at F{index:02d} in {path.name}.")


def review_frames(video, width, indices=None):
    if indices is None:
        indices = [index for group in GROUPS.values() for index in group]
    selection = "select=" + "+".join(f"eq(n\\,{index})" for index in indices)
    data = ffmpeg(video, "-vf", selection, "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-f", "rawvideo", "-")
    frame_bytes = width * EGO_SIZE * 3
    if len(data) != frame_bytes * len(indices):
        raise ValueError(f"Unexpected selected-frame payload length for {video.name}.")
    return {
        index: Image.frombytes("RGB", (width, EGO_SIZE), data[offset:offset + frame_bytes])
        for offset, index in zip(range(0, len(data), frame_bytes), indices, strict=True)
    }


def make_sheets(stage, columns, arms, smoke):
    labels = ["EXO 25 fps", "Geometry prior 30 fps", "A  OpenCV original"]
    if "B" in arms:
        labels.append("B  Aria inverse")
    if smoke:
        labels[2:] = [f"{arm}  Historical copy" for arm in arms]
    title_font = ImageFont.load_default(size=26)
    label_font = ImageFont.load_default(size=20)
    gap, margin = 12, 16
    widths = (EXO_WIDTH, EGO_SIZE) + (EGO_SIZE,) * len(arms)
    width = sum(widths) + gap * (len(widths) - 1) + 2 * margin
    for group, indices in GROUPS.items():
        height = 128 + len(indices) * (EGO_SIZE + 36)
        sheet = Image.new("RGB", (width, height), "#101720")
        draw = ImageDraw.Draw(sheet)
        title = f"Pair01  {group}"
        if smoke:
            title += "  |  CPU SMOKE TEST: HISTORICAL CLIP"
        draw.text((margin, 14), title, font=title_font, fill="#ffffff")
        draw.text((margin, 52), "Each row uses the same zero-based source frame index.", font=label_font, fill="#cbd5e1")
        x = margin
        for label, column_width in zip(labels, widths, strict=True):
            draw.text((x, 86), label, font=label_font, fill="#e2e8f0")
            x += column_width + gap
        for row, index in enumerate(indices):
            top = 118 + row * (EGO_SIZE + 36)
            draw.text((margin, top), f"F{index:02d}", font=label_font, fill="#ffffff")
            x = margin
            for column, column_width in zip(columns, widths, strict=True):
                sheet.paste(column[index], (x, top + 26))
                x += column_width + gap
        target = stage / f"{group}.png"
        sheet.save(target)
        yield {"file": target.name, "sha256": file_sha256(target), "frame_indices": list(indices)}


def write_manual_form(stage, arms, smoke):
    compare = "B" in arms
    lines = [f"# Pair01 {'A/B' if compare else 'A'} 画面验收", ""]
    if smoke:
        note = "A、B 均为同一历史原片的副本" if compare else "A 为历史原片的副本"
        lines.extend([f"> CPU 流程自检。{note}，仅用于校验脚本。", ""])
    outputs = "A、B" if compare else "A"
    footage = "两组完整的 49 帧" if compare else "A 的完整 49 帧"
    lines.extend([
        f"按零起始帧号对照 EXO、冻结的正式几何先验及 {outputs}。EXO 时刻为帧号除以 25，其他列为帧号除以 30。",
        "",
        "- [F00](F00.png) 检查队友是不是出现在先验所示方向，能否确认画面属于红鞋球员的视角。",
        "- [F20–F28](F20-F28.png) 逐帧检查是否低头朝球，并记录红鞋球员脚和球的相对位置。",
        "- [F36–F43](F36-F43.png) 逐帧记录可区分的球形数量。遮挡或看不清时填「不明」，附帧号和依据。",
        f"- 再逐帧查看{footage}，记录上述区间外的重复球形或身份混淆及对应帧号。",
        "",
        ("每组分别判断。" if compare else "") + "确认多出一个球形时拒收。整段画面无法确认红鞋球员视角或低头朝球时也拒收。逐帧记录遮挡或辨识不清的情况。",
        "视角候选及结论锁定后才使用 Nano 做独立盲评。",
    ])
    for arm in arms:
        lines.extend([
            "", f"## {arm}", "",
            f"[完整 49 帧](./{arm}_ego.mp4)", "",
            "| 帧号 | 观察记录 |", "| --- | --- |",
        ])
        lines.extend(f"| F{index:02d} | 待检查 |" for indices in GROUPS.values() for index in indices)
        lines.extend([
            "", "全片额外观察与帧号　待填写", "", "身份判断　待检查", "", "朝向判断　待检查",
            "", "重复球形判断　待检查", "", "结论　待人工验收",
        ])
    (stage / "manual-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def review(options):
    raws = {"A": options.raw_a}
    remotes = {"A": read_remote_receipt(options.receipt_a, "A")}
    compare = options.raw_b is not None
    if compare:
        raws["B"] = options.raw_b
        remotes["B"] = read_remote_receipt(options.receipt_b, "B")
        if remotes["A"]["fingerprint"] != remotes["B"]["fingerprint"]:
            raise ValueError("Arms A and B have different experiment fingerprints.")
        if raws["A"].samefile(raws["B"]):
            raise ValueError("Provide a separately recovered raw MP4 for each arm.")
    for arm, raw in raws.items():
        if raw.name != RAW_NAME or file_sha256(raw) != remotes[arm]["raw_sha256"]:
            raise ValueError(f"Recovered raw MP4 for {arm} does not match the remote file name and SHA-256.")
    if options.smoke_test and any(remote["raw_sha256"] != HISTORICAL_RAW_SHA256 for remote in remotes.values()):
        raise ValueError("The CPU smoke test requires the pinned historical Pair01 raw video for every supplied arm.")

    reference_evidence = {}
    columns = []
    for label, (relative, expected_hash, width, fps) in REFERENCES.items():
        path = RUN_ROOT / relative
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Frozen {label} reference SHA-256 does not match.")
        check_reference(path, width, fps)
        columns.append(review_frames(path, width))
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Frozen {label} reference changed during decoding.")
        reference_evidence[label] = {"file_relative_to_run": relative, "sha256": expected_hash, "fps": fps}

    output = options.output_dir.absolute()
    if os.path.lexists(output):
        raise ValueError("Choose a new output directory to preserve existing evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pair01-review-", dir=output.parent) as directory:
        stage = Path(directory)
        results = {}
        for arm, raw in raws.items():
            result = verify_take(arm, raw, stage, contact_sheets=False)
            local = json.loads((stage / result["receipt"]).read_text(encoding="utf-8"))
            remote = remotes[arm]
            if (local["raw"]["sha256"], local["raw"]["probe"]["pts_seconds"]) != (
                    remote["raw_sha256"], remote["raw_pts_seconds"]):
                raise ValueError(f"Verified local raw SHA-256 or PTS for {arm} differs from the remote receipt.")
            columns.append(review_frames(stage / result["video"], EGO_SIZE))
            results[arm] = {"method": "historical_copy" if options.smoke_test else ARMS[arm],
                            "remote_receipt": remote, **result}
        sheets = list(make_sheets(stage, columns, raws, options.smoke_test))
        write_manual_form(stage, raws, options.smoke_test)
        if options.smoke_test:
            mode = "historical_duplicate_cpu_smoke" if compare else "historical_single_cpu_smoke"
        else:
            mode = "recovered_remote_ab" if compare else "recovered_remote_a"
        summary = {
            "take": TAKE,
            "mode": mode,
            "numeric_verification": "passed",
            "manual_visual_verdict": "ineligible_smoke_test" if options.smoke_test else "pending",
            "fingerprint": remotes["A"]["fingerprint"],
            "references": reference_evidence,
            "arms": results,
            "sheets": sheets,
            "manual_review": "manual-review.md",
        }
        if compare:
            summary["raw_sha256_equal"] = remotes["A"]["raw_sha256"] == remotes["B"]["raw_sha256"]
        write_json(stage / "verification.json", summary)
        output.mkdir()
        published = []
        try:
            for source in sorted(stage.iterdir(), key=lambda path: path.name == "verification.json"):
                target = output / source.name
                os.link(source, target)
                published.append(target)
        except OSError:
            for path in published:
                path.unlink()
            output.rmdir()
            raise
    print(f"Pair01 numeric verification passed. Visual verdict {summary['manual_visual_verdict']}. {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-a", type=Path, required=True)
    parser.add_argument("--receipt-a", type=Path, required=True)
    parser.add_argument("--raw-b", type=Path)
    parser.add_argument("--receipt-b", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new local evidence directory")
    parser.add_argument("--smoke-test", action="store_true", help="Label historical MP4 copies as a CPU smoke test")
    options = parser.parse_args()
    if (options.raw_b is None) != (options.receipt_b is None):
        parser.error("Supply --raw-b and --receipt-b together for A/B review.")
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required by the shared EgoX verification script.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required on PATH.")
    try:
        review(options)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Pair01 review failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
