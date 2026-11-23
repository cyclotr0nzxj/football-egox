#!/usr/bin/env python3
"""Read-only validation of the Pair04 EXO-only input before geometry estimation."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/self_capture/pair04_s0024"
EXO_PATH = Path("example/self_capture/videos/pair04_s0024/exo.mp4")
EXO_SHA256 = "c96a61109c50b3890c4b610ba4b1bc6ad5eb3be1ef40bfeab72656cbc682c1d4"
SOURCE_PATH = Path("验证视频/裁剪后验证exo/MVI_5693.MP4")
SOURCE_SHA256 = "9c4f10825e9b6e2936ac87e4a1d1e45473f6259b6184d77b262981d9400ab2a4"
COUNT, SOURCE_FPS, RENDER_FPS = 49, 25, 30
SIZE = (784, 448)
CROP = (30, 0, 3780, 2160)
SCALE = Fraction(SIZE[0], CROP[2])
AFFINE = (
    (SCALE, 0, SCALE * (Fraction(1, 2) - CROP[0]) - Fraction(1, 2)),
    (0, SCALE, SCALE * (Fraction(1, 2) - CROP[1]) - Fraction(1, 2)),
    (0, 0, 1),
)


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isolated_exo(input_root: Path) -> Path:
    """Allow only the EXO file and ordinary Finder sidecars in its ancestors."""
    directory = input_root
    for part in EXO_PATH.parts:
        require(not directory.is_symlink() and directory.is_dir(), f"Expected input directory: {directory}")
        for entry in directory.iterdir():
            if entry.name == part:
                continue
            require(entry.name == ".DS_Store" and not entry.is_symlink() and entry.is_file(),
                    f"Unexpected staging entry: {entry.relative_to(input_root)}")
        directory /= part
    require(not directory.is_symlink() and directory.is_file(), "Expected regular EXO clip in staging tree")
    return directory


def validate_provenance(local_required: bool) -> dict:
    file = STAGE / "staging_manifest.json"
    source = ROOT / SOURCE_PATH
    manifest_verified = False
    source_verified = False
    if local_required or file.exists():
        require(not file.is_symlink() and file.is_file(), "Local staging manifest missing or unsafe")
        manifest = json.loads(file.read_text(encoding="utf-8"))
        require(manifest["schema_version"] == 1, "Unsupported staging manifest schema")
        require(manifest["exo_inference_rgb_path"] == (Path("egox_input") / EXO_PATH).as_posix(),
                "Manifest EXO input path mismatch")
        require(manifest["verification"]["exo"]["sha256"] == EXO_SHA256, "Manifest EXO digest mismatch")
        require(manifest["sources"]["exo"]["path"] == SOURCE_PATH.as_posix()
                and manifest["sources"]["exo"]["sha256"] == SOURCE_SHA256, "Manifest EXO source mismatch")
        spec = manifest["preprocessing_spec"]
        require(spec["pair_id"] == "04" and spec["exo_source_size"] == [3840, 2160]
                and spec["exo_crop_xywh"] == list(CROP) and spec["exo_output_size"] == list(SIZE)
                and spec["fps"] == SOURCE_FPS and spec["frame_count"] == COUNT,
                "Manifest EXO preprocessing mismatch")
        transform = manifest["spatial_transform"]
        require(Fraction(transform["exo_scale_exact"]) == SCALE, "Manifest EXO scale mismatch")
        matrix = transform["image_coordinate_transform_only_not_camera_K"]
        require(len(matrix) == 3 and all(len(row) == 3 for row in matrix)
                and all(math.isclose(value, expected, abs_tol=1e-12, rel_tol=0)
                        for row, expected_row in zip(matrix, AFFINE) for value, expected in zip(row, expected_row)),
                "Manifest EXO affine mismatch")
        mapping = manifest["frame_mapping"]
        require(len(mapping) == COUNT, "Manifest frame mapping length mismatch")
        for index, frame in enumerate(mapping):
            require(frame["model_frame"] == index and frame["short_clip_visible_frame"] == 24 + index
                    and frame["base_aligned_frame"] == 123 + index
                    and frame["historical_raw_exo_frame"] == 154 + index
                    and Fraction(str(frame["short_clip_pts_seconds"])) == Fraction(24 + index, SOURCE_FPS)
                    and Fraction(str(frame["model_pts_seconds"])) == Fraction(index, SOURCE_FPS),
                    f"Manifest frame mapping mismatch at {index}")
        manifest_verified = True
    if local_required or source.exists():
        require(not source.is_symlink() and source.is_file(), "Local EXO crop source missing or unsafe")
        require(sha256(source) == SOURCE_SHA256, "Local EXO crop source digest mismatch")
        source_verified = True
    return {"staging_manifest_checked": manifest_verified, "local_boxed_source_hash_checked": source_verified,
            "crop_source_size": [3840, 2160], "crop_xywh": list(CROP), "scale": str(SCALE),
            "pixel_center_affine_not_camera_intrinsics": [[str(value) for value in row] for row in AFFINE]}


def validate_video(path: Path) -> dict:
    require(sha256(path) == EXO_SHA256, "Staged EXO SHA-256 mismatch")
    probe = ["ffprobe", "-v", "error", "-show_streams", "-show_frames", "-show_entries",
             "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,time_base,start_pts,duration_ts:"
             "frame=media_type,pts,duration,pkt_duration,width,height", "-of", "json", str(path)]
    result = subprocess.run(probe, check=True, capture_output=True, text=True, timeout=60)
    require(not result.stderr.strip(), "ffprobe reported an error decoding the staged EXO clip")
    info = json.loads(result.stdout)
    require(len(info["streams"]) == 1, "Expected one video stream and no additional streams")
    stream = info["streams"][0]
    require(stream["codec_type"] == "video" and stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p",
            "Unexpected video stream format")
    require((stream["width"], stream["height"]) == SIZE, "Unexpected EXO frame size")
    require(Fraction(stream["r_frame_rate"]) == Fraction(stream["avg_frame_rate"]) == SOURCE_FPS,
            "Expected 25 fps CFR")
    time_base = Fraction(stream["time_base"])
    require(int(stream["nb_frames"]) == len(info["frames"]) == COUNT, "Expected 49 frames")
    require(int(stream["start_pts"]) == 0 and int(stream["duration_ts"]) * time_base == Fraction(COUNT, SOURCE_FPS),
            "Stream start or duration mismatch")
    for index, frame in enumerate(info["frames"]):
        require(frame["media_type"] == "video" and (frame["width"], frame["height"]) == SIZE,
                f"Unexpected decoded frame at {index}")
        require(int(frame["pts"]) * time_base == Fraction(index, SOURCE_FPS), f"Unexpected PTS at frame {index}")
        duration = frame.get("duration", frame.get("pkt_duration"))
        if duration is not None:
            require(int(duration) * time_base == Fraction(1, SOURCE_FPS), f"Unexpected duration at frame {index}")
    decode = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-hwaccel", "none",
              "-i", str(path), "-map", "0:v:0", "-an", "-f", "null", "-"]
    subprocess.run(decode, check=True, capture_output=True, timeout=60)
    return {"sha256": EXO_SHA256, "size": list(SIZE), "frames": COUNT, "fps": SOURCE_FPS,
            "time_base_seconds": str(time_base), "duration_seconds": str(Fraction(COUNT, SOURCE_FPS)),
            "full_decode": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, help="Initial EXO-only staging tree; defaults to the local Pair04 stage")
    args = parser.parse_args()
    input_root = (args.input_root or STAGE / "egox_input").absolute()
    try:
        exo = isolated_exo(input_root)
        provenance = validate_provenance(local_required=args.input_root is None)
        video = validate_video(exo)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        print(f"Pair04 input validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "passed", "exo_relative_path": EXO_PATH.as_posix(), "input_isolation": "exo_only",
        "video": video, "provenance": provenance,
        "timing": {"policy": "same frame index, different playback FPS; no interpolation or frame dropping",
                   "common_tick_seconds": "1/150", "exo_fps": SOURCE_FPS, "planned_render_fps": RENDER_FPS,
                   "map": [{"exo_frame": i, "render_frame": i, "exo_pts_ticks": 6 * i, "render_pts_ticks": 5 * i,
                            "short_clip_frame": i + 24, "aligned_base_frame": i + 123,
                            "historical_raw_exo_frame": i + 154} for i in range(COUNT)],
                   "planned_egox_gga_frame_indices": list(range(0, COUNT, 4))},
        "geometry_estimation": "not_run", "prior_rendering": "not_run", "egox_inference": "not_run",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
