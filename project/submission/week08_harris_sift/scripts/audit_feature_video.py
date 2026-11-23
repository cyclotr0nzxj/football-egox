#!/usr/bin/env python3
"""Check dynamic feature-video evidence without modifying producer artifacts.

PTS, row inventory, parameter equality and hashes are read independently. Five
frames per view are redetected with the same fixed detector helper; that part is
explicitly a same-function spot check, not an independent detector algorithm.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import course_feature_detection as features


ROOT = Path(__file__).resolve().parents[1]
FRAMES, FPS = 311, 25
SAMPLES = (0, 50, 155, 200, 310)
VIEWS, METHODS = ("exo", "ego"), ("harris", "sift")
PARAMETER_KEYS = ("seed", "opencv_threads", "harris", "sift",
                  "unique_position_rounding_decimals", "minimum_position_distance_px",
                  "maximum_retained_positions", "common_visibility_margin_px")
VIDEO_NAME = "动态特征检测_双视角对照.mp4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def probe_video(path: Path, size: tuple[int, int], expect_audio: bool) -> dict:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,duration:frame=pts_time",
        "-of", "json", str(path),
    ]))
    stream = data["streams"][0]
    pts = [Fraction(frame["pts_time"]) for frame in data["frames"]]
    require(pts == [Fraction(n, FPS) for n in range(FRAMES)], f"Unexpected visible frame PTS: {path}")
    require((stream["width"], stream["height"]) == size, f"Unexpected dimensions: {path}")
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p", f"Unexpected codec: {path}")
    require(Fraction(stream["r_frame_rate"]) == FPS and Fraction(stream["avg_frame_rate"]) == FPS,
            f"Unexpected frame rate: {path}")
    require(abs(float(stream["duration"]) - FRAMES / FPS) < 1e-6, f"Unexpected duration: {path}")
    audio = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_streams",
        "-show_entries", "stream=codec_name", "-of", "json", str(path),
    ]))["streams"]
    require(bool(audio) == expect_audio, f"Unexpected audio presence: {path}")
    if expect_audio:
        require(len(audio) == 1 and audio[0]["codec_name"] == "aac", f"Expected one AAC stream: {path}")
    return {"path": str(path), "decoded_frame_count": len(pts), "dimensions": list(size),
            "fps": FPS, "duration_seconds": FRAMES / FPS, "first_pts_seconds": 0.0,
            "last_pts_seconds": (FRAMES - 1) / FPS, "all_pts_equal_n_over_25": True,
            "aac_audio_present": expect_audio, "sha256": digest(path)}


def decode_samples(path: Path, selected_frames: tuple[int, ...] = SAMPLES) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    require(capture.isOpened(), f"Cannot decode {path}")
    selected = {}
    count = 0
    try:
        while True:
            okay, image = capture.read()
            if not okay:
                break
            if count in selected_frames:
                selected[count] = image.copy()
            count += 1
    finally:
        capture.release()
    require(count == FRAMES and set(selected) == set(selected_frames), f"Wrong sequential decode count: {path}")
    return selected


def recorded_path(value: str, dynamic: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Both the original project and portable package record root-relative paths.
    # A bare output filename is relative to the dynamic evidence directory.
    for candidate in (ROOT / path, dynamic / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve recorded file {value!r} under {ROOT} or {dynamic}")


def audit(dynamic: Path, static_results: Path, exo: Path | None = None,
          ego: Path | None = None, video: Path | None = None,
          frames_root: Path | None = None) -> dict:
    manifest = read(dynamic / "manifest.json")
    producer = read(dynamic / "audit.json")
    static = read(static_results / "metrics.json")
    require(producer["status"] == "passed", "Producer audit has not passed")
    require(manifest["frame_count"] == FRAMES and manifest["fps"] == FPS
            and manifest["display_duration_seconds"] == FRAMES / FPS, "Manifest timeline differs")
    require(manifest["working_resolution"] == [960, 540]
            and manifest["display_resolution"] == [1536, 960], "Manifest image sizes differ")
    expected_parameters = {key: static["parameters"][key] for key in PARAMETER_KEYS}
    require(manifest["parameters"] == expected_parameters, "Dynamic parameters differ from static experiment")
    require({key: features.PARAMETERS[key] for key in PARAMETER_KEYS} == expected_parameters,
            "Current detector helper differs from recorded parameters")
    require(Path(manifest["output_video"]).name == VIDEO_NAME, "Unexpected output filename")

    supplied = {"exo": exo, "ego": ego}
    working = {view: supplied[view].resolve() if supplied[view] is not None
               else recorded_path(manifest["work_inputs"][view]["path"], dynamic) for view in VIEWS}
    probes = {view: probe_video(working[view], (960, 540), view == "ego") for view in VIEWS}
    output_video = video.resolve() if video is not None else recorded_path(manifest["output_video"], dynamic)
    probes["comparison"] = probe_video(output_video, (1536, 960), True)
    for view in VIEWS:
        require(probes[view]["sha256"] == manifest["work_inputs"][view]["sha256"],
                f"Working-copy hash differs: {view}")
    require(probes["comparison"]["sha256"] == manifest["output_video_sha256"], "Output video hash differs")

    fields = ["frame_index", "time_seconds", "view", "method", "raw_count",
              "unique_position_count", "retained_count", "detection_ms"]
    with (dynamic / "counts.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, "Unexpected CSV schema")
        rows = list(reader)
    require(len(rows) == FRAMES * 4, "Expected 1244 detection rows")
    keyed = {}
    for row in rows:
        key = (int(row["frame_index"]), row["view"], row["method"])
        require(key not in keyed, f"Duplicate CSV key: {key}")
        require(Fraction(row["time_seconds"]) == Fraction(key[0], FPS), f"Incorrect CSV timestamp: {key}")
        raw, unique, retained = (int(row[name]) for name in ("raw_count", "unique_position_count", "retained_count"))
        require(0 <= retained <= unique <= raw and retained <= 500, f"Invalid point counts: {key}")
        require(np.isfinite(float(row["detection_ms"])) and float(row["detection_ms"]) >= 0,
                f"Invalid descriptive time: {key}")
        keyed[key] = row
    require(set(keyed) == {(n, view, method) for n in range(FRAMES) for view in VIEWS for method in METHODS},
            "CSV does not contain every frame/view/method exactly once")
    mapping = manifest["frame_mapping"]
    require(len(mapping) == FRAMES, "Wrong mapping length")
    origin = manifest["origin"]
    for n, row in enumerate(mapping):
        require(row["frame_index"] == n and row["source_short_visible_frame"] == n
                and Fraction(str(row["time_seconds"])) == Fraction(n, FPS), f"Mapping differs at F{n}")
        if "base_aligned_start_frame" in origin:
            require(row["base_aligned_frame"] == int(origin["base_aligned_start_frame"]) + n,
                    f"Base-frame mapping differs at F{n}")
            require(row["historical_raw_exo_frame"] == int(origin["historical_raw_exo_start_frame"]) + n,
                    f"Historical mapping differs at F{n}")
    require(producer["counts_rows"] == FRAMES * 4 and producer["expected_counts_rows"] == FRAMES * 4,
            "Producer row audit differs")

    # Preserve the producer's actual numerical input: saved H.264 working copies.
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    working_sample_indices = tuple(sorted(set(SAMPLES) | set(range(47, 54)) | set(range(197, 204))))
    samples = {view: decode_samples(path, working_sample_indices) for view, path in working.items()}
    output_samples = decode_samples(output_video)
    static_timing_checks = []
    for view in VIEWS:
        for n in (50, 200):
            reference = next(item for item in static["images"]
                             if item["view"] == view and item["visible_frame"] == n)
            if frames_root is not None:
                path = frames_root.resolve() / view / f"{n:04d}.jpg"
            else:
                candidates = [ROOT / folder / view / f"{n:04d}.jpg" for folder in (
                    "data/frames/scene04", "output/week08/frames/scene04")]
                candidates.append(Path(reference["path"]))
                path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            require(digest(path) == reference["sha256"], f"Static JPEG reference hash differs: {path}")
            jpeg = cv2.imread(str(path))
            require(jpeg is not None and jpeg.shape == (540, 960, 3), "Cannot read static reference")
            errors = {offset: float(np.abs(samples[view][n + offset].astype(np.float32)
                                           - jpeg.astype(np.float32)).mean()) for offset in range(-3, 4)}
            best_offset = min(errors, key=errors.get)
            require(best_offset == 0, f"Working frame/JPEG timing disagrees: {view}/F{n}, best offset={best_offset}")
            static_timing_checks.append({"view": view, "static_visible_frame": n,
                                         "reference_sha256": reference["sha256"],
                                         "working_frame_offsets_checked": [-3, -2, -1, 0, 1, 2, 3],
                                         "pixel_mae_by_offset_diagnostic_only": errors,
                                         "best_matching_working_frame_offset": best_offset,
                                         "same_action_timestamp_check": "passed"})
    mask = features.valid_mask(np.eye(3))
    text_font = features.font(21)
    sample_checks, preview_checks, expected_canvases = [], [], {}
    for n in SAMPLES:
        canvas = Image.new("RGB", (1536, 960), "#18212b")
        draw = ImageDraw.Draw(canvas)
        for column, view in enumerate(VIEWS):
            image = samples[view][n]
            require(image.shape == (540, 960, 3), "Working-copy decode dimensions differ")
            for panel_row, method in enumerate(METHODS):
                result = features.detect(image, mask, method)
                row = keyed[(n, view, method)]
                values = {"raw_count": result["statistics"]["raw_count"],
                          "unique_position_count": result["statistics"]["unique_position_count"],
                          "retained_count": len(result["points"])}
                require(all(int(row[key]) == value for key, value in values.items()),
                        f"Same-function redetection differs: F{n}/{view}/{method}")
                points = result["points"][:, :2]
                require(len({tuple(p) for p in np.round(points, 3)}) == len(points), "Duplicate sample centers")
                if len(points) > 1:
                    distances = np.linalg.norm(points[:, None] - points[None], axis=2)
                    np.fill_diagonal(distances, np.inf)
                    require(float(distances.min()) >= 5.0 - 1e-9, "Sample spacing below 5px")
                painted = features.point_picture(image, result, method)
                painted = cv2.resize(painted, (768, 432), interpolation=cv2.INTER_AREA)
                canvas.paste(Image.fromarray(cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)),
                             (column * 768, panel_row * 480 + 48))
                label = (f"{method.upper()} | {'R8 / EXO' if view == 'exo' else 'Nano / EGO'} | "
                         f"F{n:03d} | {n/FPS:.2f}s | N={len(points)}")
                draw.text((column * 768 + 10, panel_row * 480 + 13), label, fill="white", font=text_font)
                sample_checks.append({"frame_index": n, "time_seconds": n / FPS,
                                      "view": view, "method": method, **values,
                                      "csv_equals_same_function_redetection": True,
                                      "unique_centers_and_5px_spacing": "passed"})
        # Re-encode the independently rebuilt canvas with the stated JPEG settings.
        encoded = BytesIO()
        canvas.save(encoded, format="JPEG", quality=88)
        with Image.open(BytesIO(encoded.getvalue())) as image:
            expected_preview = np.asarray(image)
        preview_path = dynamic / "previews" / f"frame_{n:04d}.jpg"
        with Image.open(preview_path) as image:
            require(image.size == (1536, 960), f"Preview dimensions differ at F{n}")
            stored_preview = np.asarray(image)
        require(np.array_equal(expected_preview, stored_preview), f"Preview pixels/labels/overlays differ at F{n}")
        expected_canvas = np.asarray(canvas)
        expected_canvases[n] = expected_canvas
        actual_video_rgb = cv2.cvtColor(output_samples[n], cv2.COLOR_BGR2RGB)
        mae = float(np.abs(actual_video_rgb.astype(np.float32) - expected_canvas.astype(np.float32)).mean())
        # A loose integrity check for the lossy output; not a vision-quality metric.
        require(mae < 20.0, f"Large encoded frame/layout disagreement at F{n}: MAE={mae}")
        preview_checks.append({"frame_index": n, "preview": str(preview_path),
                               "preview_exactly_equals_rebuilt_jpeg_pixels": True,
                               "encoded_video_rgb_mae_to_preencode_canvas_diagnostic": mae,
                               "loose_integrity_threshold_mae": 20.0})
    for n in SAMPLES:
        rgb = cv2.cvtColor(output_samples[n], cv2.COLOR_BGR2RGB).astype(np.float32)
        errors = {frame: float(np.abs(rgb - canvas.astype(np.float32)).mean())
                  for frame, canvas in expected_canvases.items()}
        require(min(errors, key=errors.get) == n, f"Encoded sample resembles a different sampled timestamp: F{n}")

    return {
        "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only video/CSV/parameter/PTS verification and same-function redetection on F0/F50/F155/F200/F310. Only the requested audit JSON is written.",
        "passed_counts": {"videos_with_311_decoded_pts": 3, "synchronized_pts_per_video": FRAMES,
                          "unique_csv_frame_view_method_rows": len(rows), "mapping_rows": len(mapping),
                          "same_function_detection_cases": len(sample_checks), "exact_rebuilt_jpeg_previews": len(preview_checks),
                          "static_jpeg_neighbor_timing_checks": len(static_timing_checks)},
        "parameters_equal_static_results_and_current_helper": True,
        "parameters": expected_parameters, "video_checks": probes,
        "sample_redetection": sample_checks, "sample_overlay_checks": preview_checks,
        "static_jpeg_timing_checks": static_timing_checks,
        "source_input_note": manifest["static_vs_video_input_note"],
        "source_timeline_note": origin.get("timeline_note", "Portable working copies; raw-source preroll not rechecked here."),
        "input_integrity": {"manifest_sha256": digest(dynamic / "manifest.json"),
                            "counts_sha256": digest(dynamic / "counts.csv"),
                            "producer_audit_sha256": digest(dynamic / "audit.json"),
                            "static_metrics_sha256": digest(static_results / "metrics.json"),
                            "detector_helper_sha256": digest(Path(features.__file__)),
                            "auditor_sha256": digest(Path(__file__))},
        "visual_review": {"status": "not_performed_by_script", "note": "A reviewer may append actual preview observations separately; numerical image checks are not a human review."},
        "limitations": [
            "Detection spot checks use the same fixed detect/point_picture functions as the producer; this is not independent validation of Harris/SIFT algorithms.",
            "All 311 timestamps and 1244 row keys are checked; detection counts and overlays are recomputed only at five selected frames per view.",
            "Preview JPEGs are display illustrations. Actual numerical inputs are the two saved H.264 working copies, so counts need not equal the earlier static JPEG experiment.",
            "F50/F200 versus their +/-3 working-video neighbors use pixel MAE only to check frame-number consistency; no point-count equality to static JPEGs or subframe synchronization accuracy is inferred.",
            "The lossy output-frame MAE threshold is a loose layout/integrity diagnostic, not a detector, synchronization-accuracy or image-quality benchmark.",
            "The pre-existing clap synchronization uncertainty remains approximately +/-1 original frame. Equal exported PTS do not independently remove that uncertainty.",
            "These are independent framewise detections: no tracking, point identity persistence, cross-view correspondence or video repeatability is evaluated.",
            "Source edit-list endpoint is about 3.984ms longer than the 12.44s working copies; the last displayed frame duration is standardized, not preserved exactly.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=ROOT / "output/week08_harris_sift/dynamic")
    parser.add_argument("--static-results", type=Path, default=ROOT / "output/week08_harris_sift/results")
    parser.add_argument("--exo", type=Path, help="Override recorded EXO working-copy path")
    parser.add_argument("--ego", type=Path, help="Override recorded EGO working-copy path")
    parser.add_argument("--video", type=Path, help="Override recorded comparison-video path")
    parser.add_argument("--frames-root", type=Path, help="Root containing the original static EXO/EGO JPEG references")
    parser.add_argument("--output", type=Path, default=ROOT / "output/week08_harris_sift/independent_dynamic_audit.json")
    args = parser.parse_args()
    try:
        result = audit(args.dynamic.resolve(), args.static_results.resolve(), args.exo, args.ego,
                       args.video, args.frames_root)
        code = 0
    except Exception as error:
        result = {"status": "failed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error)}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed_counts": result.get("passed_counts"),
                      "error": result.get("error"), "output": str(args.output)}, ensure_ascii=False))
    sys.exit(code)


if __name__ == "__main__":
    main()
