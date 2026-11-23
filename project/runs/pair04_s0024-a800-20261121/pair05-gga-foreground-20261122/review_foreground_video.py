#!/usr/bin/env python3
"""Verify the recovered Pair05 foreground GGA output and prepare its visual review."""

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

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from review_pair01_ab import check_reference, review_frames  # noqa: E402
from verify_ab_008 import verify_audit  # noqa: E402
from verify_egox_outputs_008 import (  # noqa: E402
    EGO_SIZE, EXO_WIDTH, FPS, FRAME_COUNT, file_sha256, verify_take, write_json,
)


TAKE = "pair05_s0024"
EXO_FPS = 25
KEY_FRAMES = (12, 24)
RAW_NAME = f"{TAKE}.mp4"
HISTORICAL_RUN = "pair05-aria-008/output-008-20261122"
HISTORICAL_RAW_SHA = "c75206919368fe2a07e0a68ba8e693cc296c58fcb77209c20c3a10256f21279d"
HISTORICAL_RECEIPT_SHA = "9dc62a0746f05856734e456dd09f2186ac519a86dd698caae1ea40dbf5985a6d"
PLAN_SHA = "3cd25a102a6ecf4006add800b293dcf40b82b296392ff652aeea308fcd0f501e"
ATTENTION_MARKER = b"Pair05 foreground GGA attention calls:"
REFERENCES = {
    "EXO": (f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4",
            "902b187de0af168bdbc6ccf3ec4a8720d43298a65906b77e5e7d7e7e7e3f9fd7", EXO_WIDTH, EXO_FPS),
    "Locked geometry prior": (f"update-008/output-008/priors/{TAKE}-final.mp4",
                              "c6ee04f0be3d48bcfc7564576fedfa137d2ad63f7f46ccdae711980b8713eeaa", EGO_SIZE, FPS),
    "Historical Aria output": (f"{HISTORICAL_RUN}/verified/{TAKE}_ego.mp4",
                               "189252e559002d8f61db322e7b9fd0bb4de915b41cb3a0a53b5741123514b04c", EGO_SIZE, FPS),
}


def verify_remote(options):
    payload = options.receipt.read_bytes()
    receipt_sha = hashlib.sha256(payload).hexdigest()
    receipt = json.loads(payload)
    arm = "B" if options.smoke_test else "foreground"
    if (receipt["status"], receipt["arm"], receipt["take"]) != ("verified", arm, TAKE):
        raise ValueError(f"The receipt does not identify a verified Pair05 {arm} result.")
    if type(receipt["inference_exit_code"]) is not int or receipt["inference_exit_code"] != 0:
        raise ValueError("The receipt does not record a successful inference exit.")
    if options.expected_fingerprint and receipt["fingerprint"] != options.expected_fingerprint:
        raise ValueError("The receipt does not match the expected run fingerprint.")

    raw = receipt["raw"]
    fields = ("file", "codec", "width", "height", "fps", "frames")
    expected = (RAW_NAME, "h264", EXO_WIDTH + EGO_SIZE, EGO_SIZE, FPS, FRAME_COUNT)
    if tuple(raw[field] for field in fields) != expected:
        raise ValueError("Unexpected raw video metadata in the receipt.")
    if raw["pts_seconds"] != [str(Fraction(index, FPS)) for index in range(FRAME_COUNT)]:
        raise ValueError("The receipt does not contain all 49 expected frame PTS in order.")
    if options.raw.name != RAW_NAME or file_sha256(options.raw) != raw["sha256"]:
        raise ValueError("Recovered raw MP4 name or SHA-256 differs from the receipt.")
    if options.smoke_test and (receipt_sha, raw["sha256"]) != (HISTORICAL_RECEIPT_SHA, HISTORICAL_RAW_SHA):
        raise ValueError("The smoke test requires the pinned historical Aria receipt and raw MP4.")
    if not options.smoke_test and raw["sha256"] == HISTORICAL_RAW_SHA:
        raise ValueError("The historical Aria MP4 is only eligible for the smoke test.")
    if verify_audit(options.runtime_audit) != receipt["runtime_audit"]:
        raise ValueError("The local LoRA/CUDA runtime audit differs from the remote receipt.")

    if not options.smoke_test:
        foreground = receipt["foreground_gga"]
        if foreground["plan_sha256"] != PLAN_SHA:
            raise ValueError("The receipt does not identify the pinned Pair05 foreground GGA plan.")
        log = options.inference_log.read_bytes()
        if (options.inference_log.name, hashlib.sha256(log).hexdigest()) != (
            foreground["inference_log"], foreground["inference_log_sha256"]
        ):
            raise ValueError("The local inference log filename or SHA-256 differs from the remote receipt.")
        markers = [line.strip() for line in log.splitlines() if line.strip().startswith(ATTENTION_MARKER)]
        count = foreground["attention_calls"]
        if type(count) is not int or count <= 0 or markers != [ATTENTION_MARKER + f" {count}".encode("ascii")]:
            raise ValueError("The inference log must report the same positive attention count as the receipt once.")

    return receipt, receipt_sha


def read_frames(path, sha, width, fps, indices):
    if file_sha256(path) != sha:
        raise ValueError(f"Unexpected SHA-256 for {path}.")
    check_reference(path, width, fps)
    frames = review_frames(path, width, indices)
    if file_sha256(path) != sha:
        raise ValueError(f"Video changed during verification: {path}.")
    return frames


def sheet_record(target, indices):
    return {"file": target.name, "frame_indices": list(indices), "sha256": file_sha256(target)}


def make_key_sheets(stage, references, candidate, smoke):
    title_font = ImageFont.load_default(size=26)
    label_font = ImageFont.load_default(size=21)
    margin, gap, top = 16, 12, 122
    widths = [width for _, _, width, _ in REFERENCES.values()] + [EGO_SIZE]
    labels = [*REFERENCES, "Historical Aria fixture" if smoke else "Foreground GGA candidate"]
    width = sum(widths) + (len(widths) - 1) * gap + 2 * margin
    for index in KEY_FRAMES:
        sheet = Image.new("RGB", (width, top + EGO_SIZE + margin), "#101720")
        draw = ImageDraw.Draw(sheet)
        title = f"Pair05  F{index:02d}" + ("  |  LOCAL SMOKE TEST" if smoke else "")
        draw.text((margin, 14), title, font=title_font, fill="white")
        draw.text((margin, 53), f"Same source index. EXO {index / EXO_FPS:.3f}s | Render {index / FPS:.3f}s",
                  font=label_font, fill="#cbd5e1")
        x = margin
        for label, column_width, frames in zip(labels, widths, [*references, candidate], strict=True):
            draw.text((x, 88), label, font=label_font, fill="white")
            sheet.paste(frames[index], (x, top))
            x += column_width + gap
        target = stage / f"F{index:02d}.png"
        sheet.save(target)
        yield sheet_record(target, [index])


def make_full_sheets(stage, candidate, smoke):
    font = ImageFont.load_default(size=21)
    columns, margin, gap, header, caption = 4, 16, 12, 57, 27
    width = columns * EGO_SIZE + (columns - 1) * gap + 2 * margin
    for start, stop in ((0, 12), (12, 24), (24, 36), (36, FRAME_COUNT)):
        indices = range(start, stop)
        rows = (len(indices) + columns - 1) // columns
        height = header + rows * (EGO_SIZE + caption) + (rows - 1) * gap + margin
        sheet = Image.new("RGB", (width, height), "#101720")
        draw = ImageDraw.Draw(sheet)
        name = f"F{start:02d}-F{stop - 1:02d}"
        method = "historical Aria fixture (local smoke test)" if smoke else "foreground GGA candidate"
        draw.text((margin, 14), f"Pair05  {name}  |  {method}", font=font, fill="white")
        for offset, index in enumerate(indices):
            row, col = divmod(offset, columns)
            x = margin + col * (EGO_SIZE + gap)
            y = header + row * (EGO_SIZE + caption + gap)
            draw.text((x, y), f"F{index:02d}", font=font, fill="white")
            sheet.paste(candidate[index], (x, y + caption))
        target = stage / f"full-{name}.png"
        sheet.save(target)
        yield sheet_record(target, indices)


def write_manual_form(stage, video, full_sheets, smoke):
    lines = ["# Pair05 画面验收", ""]
    if smoke:
        lines.extend(["> 本机自检复用历史 Aria 成片，此表不用于验收新候选。", ""])
    lines.extend([
        "四列按同一零起始源帧号配对。EXO 时刻为帧号除以 25，其余列时刻为帧号除以 30。",
        "",
        "F12、F24 都要同时看到正前方的白鞋队友、队友脚边的一颗球，以及相应的蓝色标志盘。F12 核对至少一个盘，F24 对照一个清楚的盘和另一个只露出一部分的盘。球门可以在背景，不能主导画面。把看不清的项目记为「不明」，并写下依据。",
        "",
        "| 同帧对照 | 队友和脚球位置 | 可辨球数 | 标志盘 | 球门是否主导 | 本帧判断和依据 |",
        "| --- | --- | --- | --- | --- | --- |",
        "| [F12](F12.png) | 待查 | 待查 | 待查 | 待查 | 待查 |",
        "| [F24](F24.png) | 待查 | 待查 | 待查 | 待查 | 待查 |",
        "",
        f"播放[完整 49 帧]({video})，再按帧查看下列全段图。记录人物重复、队友身份或位置跳变、球消失或重影、球门重新占据主要视野的区间。",
        "",
    ])
    for item in full_sheets:
        indices = item["frame_indices"]
        lines.append(f"- [F{indices[0]:02d}–F{indices[-1]:02d}]({item['file']})　待查")
    lines.extend([
        "",
        "F12 和 F24 均达到同帧条件，且逐帧复查全段无重复人物或球、无球门持续主导，才可判为通过。遮挡、辨识不清或前景连续性异常应注明帧号。先在此锁定判断，再使用 Nano 做独立盲评。",
        "",
        "全段异常与帧号　待填写",
        "",
        "结论　历史自检，不参与验收" if smoke else "结论　待人工验收",
        "",
    ])
    (stage / "manual-review.md").write_text("\n".join(lines), encoding="utf-8")


def review(options):
    output = options.output_dir.absolute()
    if os.path.lexists(output):
        raise ValueError("Choose a new output directory to preserve existing evidence.")
    remote, receipt_sha = verify_remote(options)
    reference_evidence = {}
    references = []
    for label, (relative, sha, width, fps) in REFERENCES.items():
        references.append(read_frames(RUN / relative, sha, width, fps, KEY_FRAMES))
        reference_evidence[label] = {"file_relative_to_run": relative, "sha256": sha, "fps": fps}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pair05-review-", dir=output.parent) as directory:
        stage = Path(directory)
        result = verify_take(TAKE, options.raw, stage, contact_sheets=False)
        local = json.loads((stage / result["receipt"]).read_text(encoding="utf-8"))["raw"]
        if (local["sha256"], local["probe"]["pts_seconds"]) != (
            remote["raw"]["sha256"], remote["raw"]["pts_seconds"]
        ):
            raise ValueError("Local raw SHA-256 or frame PTS differs from the remote receipt.")
        candidate = read_frames(stage / result["video"], result["sha256"], EGO_SIZE, FPS, range(FRAME_COUNT))
        key_sheets = list(make_key_sheets(stage, references, candidate, options.smoke_test))
        full_sheets = list(make_full_sheets(stage, candidate, options.smoke_test))
        write_manual_form(stage, result["video"], full_sheets, options.smoke_test)
        verdict = "ineligible_smoke_test" if options.smoke_test else "pending"
        summary = {
            "take": TAKE,
            "mode": "local_historical_smoke" if options.smoke_test else "recovered_foreground_candidate",
            "numeric_verification": "passed",
            "manual_visual_verdict": verdict,
            "frame_mapping": {"exo_fps": EXO_FPS, "render_fps": FPS, "frames": FRAME_COUNT,
                              "alignment": "same_zero_based_source_frame_index"},
            "references": reference_evidence,
            "remote_receipt": {"file": options.receipt.name, "sha256": receipt_sha,
                               "fingerprint": remote["fingerprint"], "arm": remote["arm"],
                               "raw_sha256": remote["raw"]["sha256"],
                               "runtime_audit": remote["runtime_audit"],
                               "foreground_gga": remote.get("foreground_gga")},
            "verified_export": result,
            "key_frame_sheets": key_sheets,
            "full_clip_sheets": full_sheets,
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
    print(f"Pair05 file checks passed. Visual verdict {verdict}. {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Recovered pair05_s0024.mp4")
    parser.add_argument("--receipt", type=Path, required=True, help="Recovered verified-result.json")
    parser.add_argument("--runtime-audit", type=Path, required=True, help="Recovered runtime_audit.json")
    parser.add_argument("--inference-log", type=Path, help="Recovered inference.log from the new run")
    parser.add_argument("--expected-fingerprint", help="SHA-256 fingerprint recorded at preflight")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new local evidence directory")
    parser.add_argument("--smoke-test", action="store_true", help="Use the pinned historical Aria fixture")
    options = parser.parse_args()
    if not options.smoke_test and (not options.inference_log or not options.expected_fingerprint):
        parser.error("Supply --inference-log and --expected-fingerprint for a recovered foreground run.")
    if options.smoke_test and options.inference_log:
        parser.error("The historical smoke receipt does not record an inference log hash.")
    if options.expected_fingerprint and not re.fullmatch(r"[0-9a-f]{64}", options.expected_fingerprint):
        parser.error("--expected-fingerprint must be a lowercase SHA-256 digest.")
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required by the shared EgoX verification script.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required on PATH.")
    try:
        review(options)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError,
            subprocess.SubprocessError) as error:
        parser.exit(1, f"Pair05 review failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
