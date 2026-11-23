#!/usr/bin/env python3
"""Extract only the week-8 report's selected visible frames without overwrites.

Frame numbers start after the MP4 edit-list/preroll boundary.  Historical raw
EXO frame numbers come from the verified trim manifest; they do not imply that
the missing raw backup has been recovered.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = [0, 10, 48, 50, 130, 150, 170, 200, 230, 270, 280, 292]
SIZE = (960, 540)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != SIZE:
                raise ValueError(f"Expected {SIZE}, found {image.size}: {path}")
    except Exception as exc:
        raise ValueError(f"Existing image is invalid; refusing to overwrite {path}: {exc}") from exc
    return True


def probe_visible(path: Path) -> tuple[dict, list[dict]]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_packets", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,time_base:packet=pts_time,flags",
        "-of", "json", str(path),
    ]
    result = json.loads(subprocess.check_output(command))
    packets = sorted(
        [packet for packet in result["packets"]
         if Fraction(packet["pts_time"]) >= 0 and "D" not in packet.get("flags", "")],
        key=lambda packet: Fraction(packet["pts_time"]),
    )
    if len({packet["pts_time"] for packet in packets}) != len(packets):
        raise ValueError(f"Repeated visible packet PTS requires a different frame mapping: {path}")
    return result["streams"][0], packets


def filter_expression(frames: list[int]) -> str:
    # trim drops hidden negative-PTS preroll before select's n starts at zero.
    selected = "+".join(f"eq(n,{frame})" for frame in frames)
    return f"trim=start=0,select='{selected}',scale=960:540"


def extract_missing(source: Path, destination: Path, frames: list[int]) -> int:
    missing = [frame for frame in frames if not valid_image(destination / f"{frame:04d}.jpg")]
    if not missing:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="football-week08-frames-") as temporary:
        temporary = Path(temporary)
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filter_expression(missing),
            "-fps_mode", "vfr", "-q:v", "2", "-start_number", "0",
            str(temporary / "%04d.jpg"),
        ]
        subprocess.run(command, check=True)
        produced = sorted(temporary.glob("*.jpg"))
        if len(produced) != len(missing):
            raise ValueError(f"Expected {len(missing)} selected images, got {len(produced)} from {source}")
        for frame, temporary_image in zip(missing, produced):
            valid_image(temporary_image)
            target = destination / f"{frame:04d}.jpg"
            # Exclusive create protects existing files even if another process
            # produced a frame after the initial existence check.
            try:
                with target.open("xb") as handle, temporary_image.open("rb") as original:
                    shutil.copyfileobj(original, handle)
            except FileExistsError:
                valid_image(target)
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("records/user_ego_trim_20261019/trim_manifest.json"))
    parser.add_argument("--pair", default="04")
    parser.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=Path("output/week08/frames"), help="Parent of scene04/exo and scene04/ego")
    args = parser.parse_args()
    frames = sorted(set(args.frames))
    if not frames or frames[0] < 0:
        raise ValueError("Frame numbers must be non-negative.")
    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_id = args.pair.zfill(2)
    matches = [pair for pair in manifest["pairs"] if pair["id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest pair {pair_id}, found {len(matches)}")
    pair = matches[0]
    if frames[-1] >= pair["visible_frame_count"]:
        raise ValueError(f"Frame {frames[-1]} is outside the {pair['visible_frame_count']} visible frames.")
    output = resolve(args.output) / ("scene" + pair_id)
    source_paths = {"exo": resolve(pair["output_exo"]), "ego": resolve(pair["reference_ego"])}
    streams, packets = {}, {}
    for view, source in source_paths.items():
        streams[view], packets[view] = probe_visible(source)
        if len(packets[view]) != pair["visible_frame_count"]:
            raise ValueError(f"{view}: visible packet count differs from trim manifest.")
    exo_pts = [packet["pts_time"] for packet in packets["exo"]]
    ego_pts = [packet["pts_time"] for packet in packets["ego"]]
    if exo_pts != ego_pts:
        raise ValueError("EXO and EGO visible PTS differ; refusing to assume synchronization.")
    base_start = int(pair["source_start_frame"])
    raw_exo_start = int(pair["original_exo_start_frame"])
    start_time = Fraction(str(pair["start_seconds"]))
    mapping = {
        "schema_version": 1,
        "pair_id": pair_id,
        "trim_manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "image_size": list(SIZE),
        "selected_visible_frames": frames,
        "preroll_policy": "ffmpeg trim=start=0 precedes select; negative PTS and discard packets are excluded from the visible-frame index.",
        "filter_for_all_selected_frames": filter_expression(frames),
        "historical_raw_exo_backup_available_now": resolve(pair["source_exo_backup"]).is_file(),
        "historical_raw_exo_note": "Raw EXO frame indices below are historical coordinates from the trim manifest. The raw backup was absent when this course experiment was prepared; these numbers are not evidence that raw footage was recovered.",
        "source_videos": {
            view: {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   "sha256": sha256(path), "probe": streams[view]}
            for view, path in source_paths.items()
        },
        "frames": [
            {
                "visible_frame": frame,
                "source_pts_seconds": float(Fraction(packets["exo"][frame]["pts_time"])),
                "source_pts_exact": packets["exo"][frame]["pts_time"],
                "source_pts_match_between_views": True,
                "base_aligned_frame": base_start + frame,
                "base_aligned_pts_seconds": float(start_time + Fraction(packets["exo"][frame]["pts_time"])),
                "historical_raw_exo_frame": raw_exo_start + frame,
                "historical_raw_exo_frame_is_history_only": True,
                "images": {view: f"{view}/{frame:04d}.jpg" for view in source_paths},
            }
            for frame in frames
        ],
    }
    # Check the mapping before modifying the output directory. Existing JPEGs
    # and an already identical mapping are reused; a conflicting mapping fails.
    mapping_path = output / "frame_mapping.json"
    serialized = json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    if mapping_path.exists() and mapping_path.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"Existing mapping differs; refusing to overwrite {mapping_path}")
    extracted = {}
    for view, source in source_paths.items():
        extracted[view] = extract_missing(source, output / view, frames)
    if not mapping_path.exists():
        output.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(json.dumps({"mapping": str(mapping_path), "selected_frames_per_view": len(frames), "new_images": extracted, "existing_images_reused": {view: len(frames) - count for view, count in extracted.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
