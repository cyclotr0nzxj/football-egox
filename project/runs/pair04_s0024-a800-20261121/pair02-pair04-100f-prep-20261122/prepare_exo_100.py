"""Build 100-frame EXO-only inputs and overlapping 49-frame EgoX windows."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_self_capture_smoke as original


WIDTH, HEIGHT = 784, 448
WINDOW = 49
OFFSETS = (0, 28, 52)
CROP = "crop=3780:2160:30:0,scale=784:448"
PAIRS = {
    "Pair02": (
        "pair02_s0000", 0, "MVI_5689.MP4",
        RUN / "multiclip-prep/pair02_s0000/example/self_capture/videos/pair02_s0000/exo.mp4",
    ),
    "Pair04": (
        "pair04_s0024", 24, "MVI_5693.MP4",
        ROOT / "data/self_capture/pair04_s0024/egox_input/example/self_capture/videos/pair04_s0024/exo.mp4",
    ),
}


def pixels(path, count):
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                                   "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(count, HEIGHT, WIDTH, 3)


def diff(a, b):
    errors = np.abs(a.astype(np.int16) - b).mean(axis=(1, 2, 3))
    return {"frames": len(errors), "mean_abs_rgb_difference": float(errors.mean()),
            "max_frame_mean_abs_rgb_difference": float(errors.max())}


def export_window(source, target, first, count):
    original.START, original.STOP, original.COUNT = first, first + count, count
    logs = target.parent / "logs" / target.stem
    logs.mkdir(parents=True)
    evidence = original.export(source, target, CROP, logs, "exo")
    return {"source_first_frame": first, "source_last_frame": first + count - 1,
            "frames": count, "sha256": evidence["sha256"], "verified_pts_seconds": evidence["all_frame_pts_seconds"],
            "full_decode": evidence["status"]}


def run(output):
    frozen = {row["name"]: row for row in json.loads((RUN / "multiclip-evidence/validated-inputs.json").read_text())}
    output.mkdir(exist_ok=False)
    records = {}
    for name, (take, first, filename, baseline) in PAIRS.items():
        source = ROOT / "验证视频/裁剪后验证exo" / filename
        info, packets = original.probe_source(source)
        if (info["width"], info["height"], Fraction(info["r_frame_rate"])) != (3840, 2160, 25):
            raise ValueError(f"Source dimensions or rate changed: {name}")
        visible_pts = [Fraction(packet["pts_time"]) for packet in packets[first:first + 101]]
        if visible_pts != [Fraction(index, 25) for index in range(first, first + 101)]:
            raise ValueError(f"The EXO frames do not have the expected contiguous time grid: {name}")
        if original.sha256(baseline) != frozen[take]["exo_sha256"]:
            raise ValueError(f"Previously checked EXO changed: {name}")
        stage = output / name
        stage.mkdir()
        master_path = stage / "exo-100.mp4"
        master = export_window(source, master_path, first, 100)
        master_pixels = pixels(master_path, 100)
        original_pixels = pixels(baseline, WINDOW)
        base_alignment = diff(original_pixels, master_pixels[:WINDOW])
        if base_alignment["max_frame_mean_abs_rgb_difference"] > 5:
            raise ValueError(f"The existing baseline and new EXO master differ too much: {name}, {base_alignment}")
        windows = [{
            "offset": 0, "source_first_frame": first, "source_last_frame": first + 48, "frames": WINDOW,
            "reuse_existing_exo_and_model_result": True, "exo_path": str(baseline.relative_to(ROOT)),
            "exo_sha256": frozen[take]["exo_sha256"], "alignment_to_master": base_alignment,
        }]
        for offset in OFFSETS[1:]:
            target = stage / f"exo-window-{first + offset:04d}.mp4"
            details = export_window(source, target, first + offset, WINDOW)
            overlap_count = min(WINDOW, 100 - offset)
            alignment = diff(pixels(target, WINDOW)[:overlap_count], master_pixels[offset:offset + overlap_count])
            if alignment["max_frame_mean_abs_rgb_difference"] > 5:
                raise ValueError(f"Extended window does not match the master: {name}, {offset}, {alignment}")
            windows.append({"offset": offset, "exo_path": str(target.relative_to(output)),
                            **details, "alignment_to_master": alignment})
        records[name] = {"source": str(source.relative_to(ROOT)), "source_visible_frames": len(packets),
                         "master_exo": str(master_path.relative_to(output)), **master, "windows": windows,
                         "final_report_selected_version": "first 49-frame result"}
    receipt = {"source_fps": 25, "delivery_frames": 100, "delivery_seconds_at_source_rate": 4.0,
               "model_window_frames": WINDOW, "extra_source_tail_frame_for_model_window": 1,
               "phase_aligned_offsets": OFFSETS, "geometry_generation": "pending",
               "scope": "EXO only. No EGO ground truth was opened by this script.", "pairs": records}
    original.write_json(output / "exo-100-verification.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output.resolve())
