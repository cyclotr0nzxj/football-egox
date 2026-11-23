#!/usr/bin/env python3
"""Render independent per-frame Harris/SIFT detections on synchronized video.

--prepare-inputs creates the saved 960x540 working videos from the existing
pair04 trim manifest. Portable reproduction instead supplies --exo and --ego.
No tracking, correspondence estimation or video repeatability is implemented.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

import course_feature_detection as features


ROOT = Path(__file__).resolve().parents[1]
COUNT, FPS, DURATION = 311, 25, 311/25
VIDEO_NAME = "动态特征检测_双视角对照.mp4"
PREVIEWS = {0, 50, 155, 200, 310}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def path_record(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def probe(path: Path, decode_frames: bool = False) -> dict:
    mode = "-show_frames" if decode_frames else "-show_packets"
    item_entries = "frame=media_type,pts_time" if decode_frames else "packet=codec_type,pts_time,flags"
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", mode, "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration,sample_rate:"+item_entries,
        "-of", "json", str(path)]))


def video_stream(data: dict) -> dict:
    return next(s for s in data["streams"] if s["codec_type"] == "video")


def visible_pts(data: dict) -> list[Fraction]:
    if "frames" in data:
        return [Fraction(frame["pts_time"]) for frame in data["frames"] if frame["media_type"] == "video"]
    return sorted(Fraction(p["pts_time"]) for p in data["packets"] if p["codec_type"] == "video" and Fraction(p["pts_time"]) >= 0 and "D" not in p.get("flags", ""))


def verify(path: Path, size: tuple[int, int], audio: bool) -> dict:
    data = probe(path, decode_frames=True)
    stream = video_stream(data)
    pts = visible_pts(data)
    assert (stream["width"], stream["height"]) == size
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert Fraction(stream["r_frame_rate"]) == FPS and Fraction(stream["avg_frame_rate"]) == FPS
    assert len(pts) == COUNT and pts == [Fraction(n, FPS) for n in range(COUNT)]
    assert abs(float(stream["duration"])-DURATION) < 1e-6
    audio_streams = [s for s in data["streams"] if s["codec_type"] == "audio"]
    assert (len(audio_streams) > 0) == audio
    if audio:
        assert audio_streams[0]["codec_name"] == "aac"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "4", "-i", str(path), "-map", "0:v:0"]
    if audio:
        command += ["-map", "0:a:0"]
    command += ["-f", "null", "-"]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    return {"status": "passed", "decoded_frame_count": len(pts), "size": list(size), "fps": FPS,
            "display_duration_seconds": DURATION, "first_pts_seconds": float(pts[0]), "last_pts_seconds": float(pts[-1]),
            "all_frame_pts_match_n_over_25": True, "aac_audio_present": audio, "sha256": sha256(path)}


def prepare(manifest_path: Path, output: Path, log) -> tuple[dict, dict]:
    pair = next(p for p in json.loads(manifest_path.read_text(encoding="utf-8"))["pairs"] if p["id"] == "04")
    sources = {"exo": resolve(pair["output_exo"]), "ego": resolve(pair["reference_ego"])}
    original_records = {}
    source_pts = None
    for view, source in sources.items():
        data = probe(source)
        pts = visible_pts(data)
        assert pts == [Fraction(n, FPS) for n in range(COUNT)]
        assert len(pts) == pair["visible_frame_count"]
        if source_pts is not None:
            assert pts == source_pts
        source_pts = pts
        original_records[view] = {"path": path_record(source), "sha256": sha256(source), "visible_frame_count": len(pts), "packet_count_including_preroll": len([p for p in data["packets"] if p["codec_type"] == "video"]), "video_stream": video_stream(data)}
        destination = output/(view+".mp4")
        partial = output/(view+".partial.mp4")
        command = ["ffmpeg", "-nostdin", "-v", "warning", "-n"]
        if sys.platform == "darwin":
            command += ["-hwaccel", "videotoolbox"]
        command += ["-threads", "4", "-filter_threads", "2", "-i", str(source), "-map", "0:v:0"]
        if view == "ego":
            command += ["-map", "0:a:0"]
        else:
            command += ["-an"]
        command += ["-vf", "trim=start=0,trim=end_frame=311,setpts=N/(25*TB),scale=960:540,format=yuv420p",
            "-frames:v", str(COUNT), "-r", str(FPS), "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-maxrate", "3M", "-bufsize", "6M", "-threads", "4", "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        if view == "ego":
            command += ["-af", "atrim=start=0:end=12.44,asetpts=PTS-STARTPTS,apad,atrim=end=12.44", "-c:a", "aac", "-b:a", "128k", "-ar", "48000"]
        command += ["-t", str(DURATION), "-map_metadata", "-1", "-movflags", "+faststart", str(partial)]
        log.write("WORKING COPY COMMAND "+json.dumps(command, ensure_ascii=False)+"\n"); log.flush()
        print("Preparing saved working copy", view, flush=True)
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log, check=True)
        verify(partial, (960, 540), audio=view == "ego")
        os.link(partial, destination); partial.unlink()
    origin = {"mode": "prepared_from_existing_user_trim", "trim_manifest": path_record(manifest_path), "pair_id": "04", "original_files": original_records,
              "base_aligned_start_frame": pair["source_start_frame"], "historical_raw_exo_start_frame": pair["original_exo_start_frame"],
              "historical_raw_backup_available_now": resolve(pair["source_exo_backup"]).is_file(),
              "history_note": "Raw backup is absent; original EXO frame numbers are historical coordinates, not recovered raw video.",
              "timeline_note": "Discard/negative-PTS preroll is removed with trim=start=0. All 311 visible frames and n/25 timestamps are retained; last-frame display duration is standardized to 0.040s (source edit-list endpoint is about 3.984ms longer)."}
    return {view: output/(view+".mp4") for view in sources}, origin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs", action="store_true", help="Prepare 960x540 working copies from the existing pair04 manifest")
    parser.add_argument("--manifest", type=Path, default=Path("records/user_ego_trim_20261019/trim_manifest.json"))
    parser.add_argument("--exo", type=Path, help="Saved 960x540 EXO working video for portable reproduction")
    parser.add_argument("--ego", type=Path, help="Saved 960x540 EGO working video with AAC audio")
    parser.add_argument("--output", type=Path, default=Path("output/week08_harris_sift/dynamic"))
    args = parser.parse_args()
    if args.prepare_inputs and (args.exo or args.ego):
        parser.error("--prepare-inputs cannot be combined with --exo/--ego")
    if not args.prepare_inputs and not (args.exo and args.ego):
        parser.error("Supply --prepare-inputs or both --exo and --ego")
    output = resolve(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cv.setNumThreads(1); cv.setRNGSeed(42)
    started = time.perf_counter()
    with (output/"run.log").open("x", encoding="utf-8") as log:
        log.write("Independent framewise Harris/SIFT detection; no tracking or correspondences.\n"); log.flush()
        if args.prepare_inputs:
            working, origin = prepare(resolve(args.manifest), output, log)
        else:
            working = {"exo": resolve(args.exo), "ego": resolve(args.ego)}
            origin = {"mode": "provided_saved_working_copies", "original_files": {}, "history_note": "Original user videos are not required or checked in this portable reproduction; recorded input hashes refer to the supplied working copies."}
        inputs_audit = {view: verify(path, (960, 540), audio=view == "ego") for view, path in working.items()}
        before_hashes = {view: evidence["sha256"] for view, evidence in inputs_audit.items()}
        captures = {view: cv.VideoCapture(str(path)) for view, path in working.items()}
        assert all(cap.isOpened() for cap in captures.values())
        partial = output/"dynamic_features.partial.mp4"
        command = ["ffmpeg", "-nostdin", "-v", "warning", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", "1536x960", "-framerate", "25", "-i", "pipe:0", "-i", str(working["ego"]),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-maxrate", "6M", "-bufsize", "12M", "-threads", "4", "-pix_fmt", "yuv420p", "-c:a", "copy", "-frames:v", str(COUNT), "-r", "25", "-fps_mode", "cfr", "-t", str(DURATION), "-map_metadata", "-1", "-movflags", "+faststart", str(partial)]
        log.write("COMPARISON ENCODE COMMAND "+json.dumps(command, ensure_ascii=False)+"\n"); log.flush()
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        text_font = features.font(21)
        mask = features.valid_mask(np.eye(3))
        fields = ["frame_index", "time_seconds", "view", "method", "raw_count", "unique_position_count", "retained_count", "detection_ms"]
        counts_path = output/"counts.csv"
        rows = []
        (output/"previews").mkdir()
        try:
            with counts_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for n in range(COUNT):
                    canvas = Image.new("RGB", (1536, 960), "#18212b")
                    draw = ImageDraw.Draw(canvas)
                    for column, view in enumerate(["exo", "ego"]):
                        okay, image = captures[view].read()
                        assert okay and image.shape == (540, 960, 3), f"Missing decoded frame {n} in {view}"
                        for row, method in enumerate(["harris", "sift"]):
                            detected = features.detect(image, mask, method)
                            painted = features.point_picture(image, detected, method)
                            painted = cv.resize(painted, (768, 432), interpolation=cv.INTER_AREA)
                            panel = Image.fromarray(cv.cvtColor(painted, cv.COLOR_BGR2RGB))
                            canvas.paste(panel, (column*768, row*480+48))
                            count = len(detected["points"])
                            label = f"{method.upper()} | {'R8 / EXO' if view == 'exo' else 'Nano / EGO'} | F{n:03d} | {n/FPS:.2f}s | N={count}"
                            draw.text((column*768+10, row*480+13), label, fill="white", font=text_font)
                            stats = detected["statistics"]
                            record = {"frame_index": n, "time_seconds": n/FPS, "view": view, "method": method, "raw_count": stats["raw_count"], "unique_position_count": stats["unique_position_count"], "retained_count": count, "detection_ms": stats["detection_and_selection_ms"]}
                            writer.writerow(record); rows.append(record)
                    if n in PREVIEWS:
                        canvas.save(output/"previews"/f"frame_{n:04d}.jpg", quality=88)
                    encoder.stdin.write(canvas.tobytes())
                    if n % 25 == 0 or n == COUNT-1:
                        message = f"Frame {n:03d}/{COUNT-1} | elapsed {time.perf_counter()-started:.1f}s"
                        print(message, flush=True); log.write(message+"\n"); log.flush(); handle.flush()
                assert all(not cap.read()[0] for cap in captures.values()), "Working copy has extra decoded frames"
            encoder.stdin.close()
            assert encoder.wait() == 0, "FFmpeg comparison encoding failed; inspect run.log"
        finally:
            for cap in captures.values():
                cap.release()
            if encoder.poll() is None:
                encoder.kill(); encoder.wait()
        output_audit = verify(partial, (1536, 960), audio=True)
        final = output/VIDEO_NAME
        os.link(partial, final); partial.unlink()
        work_unchanged = all(sha256(path) == before_hashes[view] for view, path in working.items())
        originals_unchanged = all(sha256(resolve(item["path"])) == item["sha256"] for item in origin["original_files"].values()) if origin["original_files"] else None
        assert work_unchanged and originals_unchanged is not False
        assert len(rows) == COUNT*2*2
        assert len({(r["frame_index"], r["view"], r["method"]) for r in rows}) == len(rows)
        assert all(0 <= r["retained_count"] <= 500 for r in rows)
        parameters = {key: features.PARAMETERS[key] for key in ["seed", "opencv_threads", "harris", "sift", "unique_position_rounding_decimals", "minimum_position_distance_px", "maximum_retained_positions", "common_visibility_margin_px"]}
        mapping = [{"frame_index": n, "time_seconds": n/FPS, "source_short_visible_frame": n,
                    **({"base_aligned_frame": int(origin["base_aligned_start_frame"])+n, "historical_raw_exo_frame": int(origin["historical_raw_exo_start_frame"])+n} if "base_aligned_start_frame" in origin else {})} for n in range(COUNT)]
        manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "frame_count": COUNT, "fps": FPS, "display_duration_seconds": DURATION,
            "working_resolution": [960, 540], "display_resolution": [1536, 960], "layout": "left EXO/R8, right EGO/Nano; upper Harris, lower SIFT; each image 768x432 with its own 48px label band", "audio": "Saved Nano working-copy AAC track copied into the comparison video",
            "parameters": parameters, "versions": {"opencv": cv.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}, "origin": origin,
            "work_inputs": {view: {"path": path_record(path), "sha256": before_hashes[view], "role": "saved re-encoded 960x540 working copy; numerical detections are decoded from this file"} for view, path in working.items()},
            "output_video": VIDEO_NAME, "output_video_sha256": output_audit["sha256"], "counts_file": "counts.csv", "frame_mapping": mapping,
            "previews": [f"previews/frame_{n:04d}.jpg" for n in sorted(PREVIEWS)], "detection_semantics": "Each view and frame is independently detected with exactly the static experiment's parameters and position-selection rules. No trails, tracking, cross-view correspondence or inter-frame repeatability.",
            "static_vs_video_input_note": "The earlier static experiment used saved JPEGs. This dynamic experiment uses the saved H.264 working copies. Compression/decoding differences can change retained point counts; they are not forced to match the JPEG experiment.",
            "working_copy_encoding": "960x540 H.264 CRF22, maxrate3M, yuv420p; Nano AAC128k. These compressed working copies are the reproducible numerical inputs, not temporary in-memory original-video frames.",
            "runtime_seconds_descriptive": time.perf_counter()-started, "encoder_command": command}
        write_json(output/"manifest.json", manifest)
        previews_check = []
        for path in sorted((output/"previews").glob("*.jpg")):
            with Image.open(path) as im:
                assert im.size == (1536, 960); im.verify()
            previews_check.append(path.name)
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        assert final.stat().st_size < 15_000_000 and size < 29_000_000
        audit = {"status": "passed", "working_video_full_decode": inputs_audit, "comparison_video_full_decode": output_audit, "opencv_decoded_frames_per_view": COUNT, "all_three_video_pts_sequences_equal": True, "counts_rows": len(rows), "expected_counts_rows": COUNT*4, "unique_frame_view_method_rows": True, "retained_points_within_500_budget": True, "working_video_hashes_unchanged": work_unchanged, "original_video_hashes_unchanged": originals_unchanged, "original_hash_check_scope": "Existing user trim inputs" if origin["original_files"] else "Original user inputs unavailable in portable reproduction; saved working-copy hashes checked instead", "preview_jpegs_verified": previews_check, "output_video_bytes": final.stat().st_size, "directory_bytes_before_audit_readme": size}
        write_json(output/"audit.json", audit)
        (output/"README.md").write_text("# Dynamic Harris / SIFT evidence\n\nPair04: all 311 visible frames at25fps, 12.44s. The saved 960x540 H.264 exo.mp4 and ego.mp4 working copies are the actual numerical inputs; detection does not use unsaved original-video frames. Nano AAC audio is preserved in the four-panel output.\n\nLeft: R8/EXO; right: Nano/EGO; top: Harris; bottom: SIFT. Every panel shows its method, view, frame index, time and retained-center count. Frames/views are independently detected with the static experiment's exact parameters: seed42, one OpenCV thread, Harris2/3/0.04 with1% positive maximum and3x3 NMS; SIFT500/3/0.04/10/1.6; duplicate centers removed, minimum5px spacing, maximum500 positions,16px boundary margin. There is no tracking, point trail, cross-view correspondence or inter-frame R. Grass texture, motion blur and compression effects are retained. Count differences from old static JPEGs are expected and not corrected.\n\ncounts.csv has1244 frame-view-method rows. manifest.json records parameters, versions, input SHA-256 and timeline mappings; audit.json records full decoding,311 timestamps in all three videos and unchanged source hashes. Historical raw EXO indices are references only because the raw backup is absent. Five JPEG previews are display illustrations, not detector inputs. Timing is descriptive, not a speed benchmark.\n\nPortable reproduction after installing the repository dependencies and FFmpeg; start from the repository root and choose an empty destination:\n\n```sh\npython scripts/build_feature_video.py --exo data/video_pair04/exo.mp4 --ego data/video_pair04/ego.mp4 --output dynamic_reproduced\n```\n\nThe original-project preparation command is `python scripts/build_feature_video.py --prepare-inputs --output output/week08_harris_sift/dynamic`; it reads the existing trim manifest, excludes hidden negative-PTS preroll, and saves working copies before detection. It is not required for portable reproduction.\n", encoding="utf-8")
        log.write("AUDIT PASSED "+json.dumps({"rows": len(rows), "video_bytes": final.stat().st_size, "originals_unchanged": originals_unchanged})+"\n"); log.flush()
        print(json.dumps({"status": "completed_and_verified", "output": str(output), "frame_count": COUNT, "counts_rows": len(rows), "video_bytes": final.stat().st_size, "runtime_seconds": time.perf_counter()-started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
