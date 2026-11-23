#!/usr/bin/env python3
"""Export a frame-aligned Pair03 comparison after freezing the geometry preview."""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
PROJECT_ROOT = RUN_ROOT.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUN_ROOT))
from verify_egox_outputs_008 import file_sha256, run, write_json  # noqa: E402

FPS = 25
FRAME_COUNT = 49
PREVIEW_SIZE = (448, 448)
NANO_SIZE = (3840, 2160)
DISPLAY_NANO = (768, 432)
EXO_SIZE = (784, 448)
KEYFRAMES = (0, 12, 24, 36, 48)
SYNC_RECEIPT = PROJECT_ROOT / "records/user_ego_trim_20261019/03_export.json"
EXO_RECEIPT = RUN_ROOT / "pair05-left-diagnosis-20261122/pair03-exo/exo-check.json"
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
BACKGROUND = "#10151c"


def check_digest(path, expected):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}.")
    return actual


def probe_visible_frames(path, size, *, clip_start_only=False):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if clip_start_only:
        command += ["-read_intervals", "%+3"]
    command += [
        "-show_frames", "-show_streams", "-show_entries",
        "stream=codec_name,width,height,time_base:frame=pts,width,height",
        "-of", "json", str(path),
    ]
    data = json.loads(run(command))
    stream, = data["streams"]
    if (stream["width"], stream["height"]) != size:
        raise ValueError(f"Unexpected dimensions in {path.name}.")
    time_base = Fraction(stream["time_base"])
    rows = [frame for frame in data["frames"] if int(frame["pts"]) >= 0]
    if clip_start_only:
        rows = rows[:FRAME_COUNT]
    pts = [int(row["pts"]) * time_base for row in rows]
    if pts != [Fraction(index, FPS) for index in range(FRAME_COUNT)]:
        raise ValueError(f"Visible frames F00–F48 in {path.name} do not follow exact i/25 PTS.")
    if any((row["width"], row["height"]) != size for row in rows):
        raise ValueError(f"Unexpected frame size in {path.name}.")
    return {"codec": stream["codec_name"], "size": list(size), "frames": len(rows),
            "pts_seconds": [str(timestamp) for timestamp in pts]}


def decode_visible(path, size):
    width, height = size
    filters = f"trim=start=0,trim=end_frame={FRAME_COUNT},scale={width}:{height}:flags=lanczos,setsar=1"
    decoded = run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-copyts", "-threads", "2",
        "-err_detect", "explode", "-i", str(path), "-map", "0:v:0", "-an", "-vf", filters,
        "-frames:v", str(FRAME_COUNT), "-pix_fmt", "rgb24", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-",
    ])
    bytes_per_frame = width * height * 3
    if len(decoded) != FRAME_COUNT * bytes_per_frame:
        raise ValueError(f"Incorrect number of decoded RGB frames in {path.name}.")
    return [Image.frombytes("RGB", size, decoded[i:i + bytes_per_frame])
            for i in range(0, len(decoded), bytes_per_frame)]


def compose_frame(panels, labels, index, fonts):
    margin = gap = 12
    header = 54
    width = 2 * margin + gap * (len(panels) - 1) + sum(panel.width for panel in panels)
    body_height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, header + body_height + margin), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    heading, timestamp = fonts
    x = margin
    for panel, label in zip(panels, labels, strict=True):
        draw.text((x, 13), label, font=heading, fill="#f0f3f7")
        canvas.paste(panel, (x, header + (body_height - panel.height) // 2))
        x += panel.width + gap
    text = f"F{index:02d} · {index / FPS:.2f} s"
    draw.text((width - margin, 17), text, anchor="rt", font=timestamp, fill="#becbd9")
    return canvas


def create_visuals(stage, columns, labels):
    fonts = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 19)
    video = stage / "pair03-geometry-vs-nano.mp4"
    contact = stage / "pair03-geometry-vs-nano-keyframes.jpg"
    first = compose_frame([column[0] for column in columns], labels, 0, fonts)
    width, height = first.size
    key_images = []
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "17", "-pix_fmt", "yuv420p", "-video_track_timescale", "25000",
        "-movflags", "+faststart", str(video),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
        try:
            for index, panels in enumerate(zip(*columns, strict=True)):
                frame = first if index == 0 else compose_frame(panels, labels, index, fonts)
                encoder.stdin.write(frame.tobytes())
                if index in KEYFRAMES:
                    key_images.append(frame)
            encoder.stdin.close()
        except BrokenPipeError as error:
            raise ValueError(encoder.stderr.read().decode(errors="replace")) from error
        errors = encoder.stderr.read().decode(errors="replace")
        if encoder.wait(timeout=60) or errors.strip():
            raise ValueError(f"Failed to encode comparison: {errors}")

    sheet = Image.new("RGB", (width, height * len(key_images)), BACKGROUND)
    for row, key_frame in enumerate(key_images):
        sheet.paste(key_frame, (0, row * height))
    sheet.save(contact, quality=94, optimize=True)
    return video, contact, probe_visible_frames(video, (width, height))


def build(options):
    preview = options.preview.resolve()
    output = options.output_dir.absolute()
    if os.path.lexists(output):
        raise ValueError("Choose a new output directory to preserve existing comparisons.")
    # Verify the preview freeze before opening any Nano bytes.
    preview_hash = check_digest(preview, options.preview_sha256)
    preview_probe = probe_visible_frames(preview, PREVIEW_SIZE)
    preview_images = decode_visible(preview, PREVIEW_SIZE)

    sync = json.loads(SYNC_RECEIPT.read_text(encoding="utf-8"))
    pair = sync["pair"]
    if (pair["id"], pair["source_start_frame"], sync["visible_frames"], sync["visible_pts_match_reference"]) != (
        "03", 87, 280, True,
    ):
        raise ValueError("Historical Pair03 synchronization evidence does not match the expected trim.")
    nano = PROJECT_ROOT / pair["reference_ego"]
    nano_hash = check_digest(nano, sync["reference_sha256"])
    nano_probe = probe_visible_frames(nano, NANO_SIZE, clip_start_only=True)
    nano_images = decode_visible(nano, DISPLAY_NANO)

    labels = ["第一视角几何预览", "Nano 实拍 · 第一视角"]
    columns = [preview_images, nano_images]
    checks = [(preview, preview_hash), (nano, nano_hash)]
    if options.exo is not None:
        exo = options.exo.resolve()
        exo_receipt = json.loads(EXO_RECEIPT.read_text(encoding="utf-8"))
        exo_hash = check_digest(exo, exo_receipt["sha256"])
        probe_visible_frames(exo, EXO_SIZE)
        columns.insert(0, decode_visible(exo, EXO_SIZE))
        labels.insert(0, "R8 实拍 · 第三人称")
        checks.append((exo, exo_hash))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pair03-comparison-", dir=output.parent) as temporary:
        stage = Path(temporary)
        video, contact, export_probe = create_visuals(stage, columns, labels)
        for path, digest in checks:
            check_digest(path, digest)
        receipt = stage / "comparison-check.json"
        write_json(receipt, {
            "status": "synchronized_pair03_comparison_verified",
            "alignment": {"frame_mapping": "preview F[n] to trimmed Nano visible F[n]",
                          "base_aligned_clip_start_frame": pair["source_start_frame"],
                          "nominal_fps": FPS, "historical_sync_uncertainty_frames": 1,
                          "nano_role": "evaluation_only_after_preview_sha256_verified",
                          "sync_receipt": str(SYNC_RECEIPT.relative_to(PROJECT_ROOT))},
            "source_preview": {"file": preview.name, "sha256": preview_hash, **preview_probe},
            "source_nano": {"file": pair["reference_ego"], "sha256": nano_hash,
                            "display_size": list(DISPLAY_NANO), "cropped": False, **nano_probe},
            "comparison_video": {"file": video.name, "sha256": file_sha256(video), **export_probe},
            "keyframe_sheet": {"file": contact.name, "sha256": file_sha256(contact), "frames": list(KEYFRAMES)},
            "panels_left_to_right": labels,
        })
        output.mkdir()
        for artifact in (video, contact, receipt):
            os.link(artifact, output / artifact.name)
    print(output / video.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview", type=Path)
    parser.add_argument("--preview-sha256", required=True, help="SHA-256 of the frozen preview")
    parser.add_argument("--output-dir", type=Path, default=HERE / "comparison")
    parser.add_argument("--exo", type=Path, help="Optionally add the verified 49-frame EXO panel")
    options = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", options.preview_sha256):
        parser.error("--preview-sha256 must be a lowercase SHA-256 digest.")
    build(options)


if __name__ == "__main__":
    main()
