#!/usr/bin/env python3
"""Stage a 49-frame self-capture EXO smoke clip and an isolated Nano reference.

This prepares RGB videos only. It does not estimate geometry, render priors,
write model metadata, or run EgoX. Existing conflicting files are never replaced.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAIR = "04"
START, STOP, FPS = 24, 73, 25
COUNT = STOP - START
EXO_REL = Path("egox_input/example/self_capture/videos/pair04_s0024/exo.mp4")
NANO_REL = Path("reference/nano.mp4")
SPEC = {
    "pair_id": PAIR, "visible_frame_start_inclusive": START,
    "visible_frame_stop_exclusive": STOP, "frame_count": COUNT, "fps": FPS,
    "exo_source_size": [3840, 2160], "exo_crop_xywh": [30, 0, 3780, 2160],
    "exo_output_size": [784, 448], "nano_output_size": [960, 540],
    "encoder": "libx264", "crf": 18, "pixel_format": "yuv420p", "audio": False,
}


def resolve(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def probe_source(path: Path) -> tuple[dict, list[dict]]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_packets", "-show_entries",
               "stream=codec_name,width,height,pix_fmt,r_frame_rate,color_range,color_space:packet=pts_time,flags", "-of", "json", str(path)]
    result = json.loads(subprocess.check_output(command))
    packets = sorted([p for p in result["packets"] if Fraction(p["pts_time"]) >= 0 and "D" not in p.get("flags", "")], key=lambda p: Fraction(p["pts_time"]))
    return result["streams"][0], packets


def verify_clip(path: Path, size: tuple[int, int]) -> dict:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_frames", "-show_entries",
               "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration:frame=media_type,pts_time,width,height", "-of", "json", str(path)]
    result = json.loads(subprocess.check_output(command))
    assert len(result["streams"]) == 1, f"Expected video only: {path}"
    video = result["streams"][0]
    frames = [f for f in result["frames"] if f["media_type"] == "video"]
    assert video["codec_type"] == "video" and video["codec_name"] == "h264"
    assert (video["width"], video["height"]) == size
    assert video["pix_fmt"] == "yuv420p"
    assert Fraction(video["r_frame_rate"]) == FPS and Fraction(video["avg_frame_rate"]) == FPS
    assert len(frames) == COUNT and int(video["nb_frames"]) == COUNT
    expected_pts = [Fraction(n, FPS) for n in range(COUNT)]
    assert [Fraction(f["pts_time"]) for f in frames] == expected_pts
    assert abs(float(video["duration"]) - COUNT / FPS) < 1e-6
    # Fully decode both streams with errors promoted to a nonzero exit status.
    decode = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "4", "-i", str(path), "-map", "0:v:0", "-an", "-f", "null", "-"]
    subprocess.run(decode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {"status": "passed", "decoded_frames": len(frames), "size": list(size), "fps": FPS,
            "first_pts_seconds": float(expected_pts[0]), "last_pts_seconds": float(expected_pts[-1]),
            "duration_seconds": COUNT / FPS, "all_frame_pts_seconds": [float(p) for p in expected_pts],
            "full_decode_command": decode, "sha256": sha256(path)}


def export(source: Path, destination: Path, spatial_filter: str, output: Path, label: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".partial.mp4")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace existing file: {destination}")
    filters = f"trim=start=0,trim=start_frame={START}:end_frame={STOP},setpts=N/({FPS}*TB),{spatial_filter},format=yuv420p"
    command = ["ffmpeg", "-nostdin", "-v", "info", "-n", "-hwaccel", "videotoolbox", "-threads", "4", "-filter_threads", "2", "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters, "-frames:v", str(COUNT), "-r", str(FPS), "-fps_mode", "cfr",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p",
               "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-map_metadata", "-1", "-movflags", "+faststart", str(partial)]
    print(f"Encoding {label}: visible frames {START}..{STOP-1}", flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (output / (label + "_encode.log")).write_text(result.stderr, encoding="utf-8")
    result.check_returncode()
    size = (784, 448) if label == "exo" else (960, 540)
    evidence = verify_clip(partial, size)
    # Atomic, exclusive publication also protects against an unexpected file
    # being created between the preflight check and completion of encoding.
    os.link(partial, destination)
    partial.unlink()
    evidence["full_decode_command"][evidence["full_decode_command"].index(str(partial))] = str(destination)
    evidence["encoding_command"] = command
    return evidence


def contact_sheet(exo: Path, destination: Path) -> None:
    indices = [0, 24, 48]
    with tempfile.TemporaryDirectory(prefix="exo-smoke-contact-") as directory:
        pattern = str(Path(directory) / "%02d.png")
        command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(exo), "-map", "0:v:0", "-vf", "select='eq(n,0)+eq(n,24)+eq(n,48)'", "-fps_mode", "vfr", "-start_number", "0", pattern]
        subprocess.run(command, check=True)
        font_path = Path("/System/Library/Fonts/Monaco.ttf")
        font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default(size=17)
        canvas = Image.new("RGB", (784 * 3, 492), "#18212b")
        draw = ImageDraw.Draw(canvas)
        for column, index in enumerate(indices):
            image_path = Path(directory) / f"{column:02d}.png"
            with Image.open(image_path) as image:
                assert image.size == (784, 448)
                canvas.paste(image.convert("RGB"), (column * 784, 44))
            text = f"EXO model F{index} | pair04 F{START+index} | clip PTS {(START+index)/FPS:.2f}s"
            draw.text((column * 784 + 10, 12), text, fill="white", font=font)
        if destination.exists():
            raise FileExistsError(destination)
        canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("records/user_ego_trim_20261019/trim_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/self_capture/pair04_s0024"))
    args = parser.parse_args()
    manifest_path, output = resolve(args.manifest), resolve(args.output)
    pair = next(p for p in json.loads(manifest_path.read_text(encoding="utf-8"))["pairs"] if p["id"] == PAIR)
    sources = {"exo": resolve(pair["output_exo"]), "nano": resolve(pair["reference_ego"])}
    source_records, packets = {}, {}
    for label, source in sources.items():
        stream, packets[label] = probe_source(source)
        assert (stream["width"], stream["height"]) == (3840, 2160)
        assert Fraction(stream["r_frame_rate"]) == FPS
        assert len(packets[label]) == pair["visible_frame_count"]
        source_records[label] = {"path": str(source.relative_to(ROOT)), "sha256": sha256(source), "stream": stream}
    assert [p["pts_time"] for p in packets["exo"]] == [p["pts_time"] for p in packets["nano"]]
    manifest_file = output / "staging_manifest.json"
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert existing["preprocessing_spec"] == SPEC and existing["sources"] == source_records, "Existing staging conflicts with current sources or specification."
        for label, relative, size in [("exo", EXO_REL, (784, 448)), ("nano", NANO_REL, (960, 540))]:
            check = verify_clip(output / relative, size)
            assert check["sha256"] == existing["verification"][label]["sha256"], "Existing output changed."
        assert sha256(output / "exo_contact_sheet.png") == existing["contact_sheet_sha256"]
        print(json.dumps({"status": "verified_existing", "manifest": str(manifest_file)}, ensure_ascii=False))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Staging directory is nonempty without a completed manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verification = {
        "exo": export(sources["exo"], output / EXO_REL, "crop=3780:2160:30:0,scale=784:448", output, "exo"),
        "nano": export(sources["nano"], output / NANO_REL, "scale=960:540", output, "nano"),
    }
    sheet = output / "exo_contact_sheet.png"
    contact_sheet(output / EXO_REL, sheet)
    scale = Fraction(784, 3780)
    assert scale == Fraction(448, 2160)
    staging = {
        "schema_version": 1, "preprocessing_spec": SPEC, "trim_manifest": str(manifest_path.relative_to(ROOT)), "sources": source_records,
        "exo_inference_rgb_path": str(EXO_REL), "nano_reference_path": str(NANO_REL),
        "reference_isolation": "Nano is an evaluation/reference video outside egox_input. No Nano pixels are used for EXO window selection, prompt selection, priors, model metadata, or model input.",
        "frame_mapping": [{"model_frame": n, "short_clip_visible_frame": START+n,
                           "short_clip_pts_seconds": float(Fraction(packets["exo"][START+n]["pts_time"])),
                           "model_pts_seconds": n/FPS,
                           "base_aligned_frame": int(pair["source_start_frame"])+START+n,
                           "historical_raw_exo_frame": int(pair["original_exo_start_frame"])+START+n}
                          for n in range(COUNT)],
        "historical_raw_exo_backup_available_now": resolve(pair["source_exo_backup"]).is_file(),
        "historical_raw_note": "The original raw backup is absent. Original EXO indices are historical coordinates recorded during synchronization, not recovered raw footage.",
        "spatial_transform": {
            "coordinate_convention": "Integer (u,v) denotes the center of the top-left pixel at (0,0); crop rectangle uses half-open source bounds.",
            "exo_crop_source_bounds_xyxy": [30, 0, 3810, 2160], "exo_scale_exact": str(scale), "exo_scale_xy": [float(scale), float(scale)],
            "exo_pixel_center_forward": "u_out=(u_source-30+0.5)*(784/3780)-0.5; v_out=(v_source+0.5)*(448/2160)-0.5",
            "exo_pixel_center_inverse": "u_source=(u_out+0.5)/(784/3780)+30-0.5; v_source=(v_out+0.5)/(448/2160)-0.5",
            "image_coordinate_transform_only_not_camera_K": [[float(scale), 0, float(scale)*(0.5-30)-0.5], [0, float(scale), float(scale)*0.5-0.5], [0, 0, 1]],
            "nano_reference_scale_xy": [0.25, 0.25], "nano_pixel_center_forward": "u_out=(u_source+0.5)*0.25-0.5; v_out=(v_source+0.5)*0.25-0.5",
            "resize_kernel": "FFmpeg scale default bicubic", "encoding_note": "Both clips are H.264 CRF18 8-bit yuv420p limited-range BT.709 derivatives. Nano source is 10-bit and is converted to 8-bit for this reference clip.",
        },
        "verification": verification, "contact_sheet": "exo_contact_sheet.png", "contact_sheet_sha256": sha256(sheet),
        "contact_sheet_provenance": "Decoded only from staged EXO model frames 0,24,48. No Nano frame was used for this sheet.",
        "missing_inputs": ["Estimated scene depth for the selected EXO frames", "Calibrated or estimated camera intrinsics and source camera poses", "Per-frame target ego camera trajectory in the same coordinate system", "Renderer-produced geometric ego priors and their valid masks", "Verified official inference metadata and text conditioning"],
        "execution_status": {"rgb_staging": "completed_and_fully_decoded", "geometry_estimation": "not_run", "ego_prior_rendering": "not_run", "self_capture_egox_generation": "not_run", "model_metadata_created": False},
    }
    write_json(manifest_file, staging)
    print(json.dumps({"status": "staged_and_verified", "manifest": str(manifest_file), "frames_each": COUNT, "first_short_frame": START, "last_short_frame": STOP-1, "exo_path": str(output/EXO_REL), "nano_reference": str(output/NANO_REL)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
