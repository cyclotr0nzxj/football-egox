"""Verify the recovered B run and align it to the historical 004 by frame index."""

import argparse
from contextlib import suppress
from pathlib import Path
import json
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from review_pair01_ab import check_reference, file_sha256, review_frames
from verify_ab_008 import verify_audit, verify_video
from verify_clip_b import read_evidence


TAKE = "pair05_s0024"
EXO = RUN / f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4"
PRIOR = RUN / f"update-008/output-008/priors/{TAKE}-final.mp4"
OLD = RUN / f"clone-004-20261122/pair05/verified/{TAKE}_ego.mp4"
NAMES = ("R8 EXO", "Locked prior", "004 historical", "008 B")
KEYS = (12, 24)


def references(path, expected, width, fps):
    if file_sha256(path) != expected:
        raise ValueError(f"The historical input changed: {path.name}")
    check_reference(path, width, fps)
    return review_frames(path, width, range(49))


def verify_result(source):
    result = json.loads((source / "verified-result.json").read_text())
    if ((result["status"], result["arm"], result["take"]) != ("verified", "B", TAKE)
            or result["fingerprint"] != (source / "fingerprint.txt").read_text().strip()
            or (source / "inference-exit-code.txt").read_text().strip() != "0"):
        raise ValueError("This is not the completed B attempt.")
    evidence = read_evidence(source / "inference.log")
    for key, value in evidence.items():
        if result[key] != value:
            raise ValueError(f"Recovered receipt mismatch at {key}.")
    if result["runtime_audit"] != verify_audit(source / "runtime_audit.json"):
        raise ValueError("The recovered runtime audit differs from its receipt.")
    if result["raw"] != verify_video(source / f"{TAKE}.mp4", "ffprobe", "ffmpeg"):
        raise ValueError("The recovered video differs from its receipt.")
    return result


def frame_row(index, panes):
    font = ImageFont.load_default(size=20)
    with Image.new("RGB", (2128, 484), "#101621") as row:
        draw = ImageDraw.Draw(row)
        offset = 0
        for label, frames in zip(NAMES, panes, strict=True):
            tile = frames[index]
            draw.text((offset + 10, 8), f"F{index:02}  {label}", font=font, fill="white")
            row.paste(tile, (offset, 36))
            offset += tile.width
        return row.copy()


def encode_rows(path, rows):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "2128x484", "-framerate", "30", "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-video_track_timescale",
               "30000", "-movflags", "+faststart", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for row in rows:
                process.stdin.write(row.tobytes())
        finally:
            with suppress(BrokenPipeError):
                process.stdin.close()
        if process.wait() != 0:
            raise ValueError("Could not encode the same-index comparison.")


def run(source, output):
    result = verify_result(source)
    before = references(EXO, "902b187de0af168bdbc6ccf3ec4a8720d43298a65906b77e5e7d7e7e7e3f9fd7", 784, 25)
    geometry = references(PRIOR, "c6ee04f0be3d48bcfc7564576fedfa137d2ad63f7f46ccdae711980b8713eeaa", 448, 30)
    old_receipt = json.loads((OLD.parent / "verification.json").read_text())["verified_export"]
    if old_receipt["take"] != TAKE or old_receipt["video"] != OLD.name:
        raise ValueError("Expected the historically verified 004 Pair05 export.")
    old = references(OLD, old_receipt["sha256"], 448, 30)
    if result["raw"]["sha256"] == file_sha256(OLD.parent.parent / f"{TAKE}.mp4"):
        raise ValueError("An old 004 original cannot be accepted as a B result.")

    output.mkdir(exist_ok=False)
    movie = output / "Pair05-B.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source / f"{TAKE}.mp4"),
                    "-an", "-vf", "crop=448:448:784:0", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "14", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000",
                    "-movflags", "+faststart", str(movie)], check=True)
    check_reference(movie, 448, 30)
    new = review_frames(movie, 448, range(49))
    panes = (before, geometry, old, new)
    key_sheet = Image.new("RGB", (2128, 968))
    for row, index in enumerate(KEYS):
        key_sheet.paste(frame_row(index, panes), (0, row * 484))
    key_sheet.save(output / "F12-F24.png")
    encode_rows(output / "Pair05-same-frames.mp4", (frame_row(index, panes) for index in range(49)))
    sizes = (224, 224)
    full_sheet = Image.new("RGB", (sizes[0] * 7, (sizes[1] + 20) * 7), "#101621")
    caption = ImageDraw.Draw(full_sheet)
    for index, frame in new.items():
        y, x = divmod(index, 7)
        top = y * (sizes[1] + 20)
        caption.text((x * sizes[0] + 7, top + 4), f"F{index:02}", fill="white")
        full_sheet.paste(frame.resize(sizes, Image.Resampling.LANCZOS), (x * sizes[0], top + 20))
    full_sheet.save(output / "Pair05-B-all-frames.jpg", quality=94)
    (output / "local-verification.json").write_text(json.dumps({
        "fingerprint": result["fingerprint"], "candidate_raw_sha256": result["raw"]["sha256"],
        "historical_004_sha256": file_sha256(OLD), "frames": 49,
        "display_fps": 30, "alignment": "source frame index 0..48",
        "outputs": {path.name: file_sha256(path) for path in sorted(output.iterdir())},
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
