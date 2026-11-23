#!/usr/bin/env python3
"""Export frame-aligned R8/Nano clips; the comparison uses the EGO audio track.

Run from any directory: python scripts/export_aligned_capture.py --manifest PATH
Original files are read only. FFmpeg autorotation is retained, then rotation
metadata is cleared. Video is decoded and re-encoded, never stream-copied.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESERVE = 300 * 1024 * 1024
SPECS = {
    "ego": (3840, 2160, "hevc", "yuv420p10le", 24_000_000),
    "exo": (3840, 2160, "h264", "yuv420p", 24_000_000),
    "comparison": (1920, 540, "h264", "yuv420p", 4_000_000),
}


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def probe(path):
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format",
               "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def stream(data, kind):
    return next((s for s in data["streams"] if s["codec_type"] == kind), None)


def rotation(video):
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            return float(side["rotation"])
    return float(video.get("tags", {}).get("rotate", 0))


def validate(path, kind, fps, count):
    data = probe(path)
    video, audio = stream(data, "video"), stream(data, "audio")
    width, height, codec, pixels, _ = SPECS[kind]
    errors = []
    if video is None or audio is None:
        raise ValueError(f"{path}: missing video or audio stream")
    for key, expected in [("width", width), ("height", height),
                          ("codec_name", codec), ("pix_fmt", pixels),
                          ("color_range", "tv"), ("color_space", "bt709"),
                          ("color_transfer", "bt709"), ("color_primaries", "bt709")]:
        if video.get(key) != expected:
            errors.append(f"{key}={video.get(key)!r}, expected {expected!r}")
    if int(video.get("nb_frames", -1)) != count:
        errors.append(f"nb_frames={video.get('nb_frames')}, expected {count}")
    for key in ("avg_frame_rate", "r_frame_rate"):
        if Fraction(video.get(key, "0/1")) != fps:
            errors.append(f"{key}={video.get(key)}, expected {fps}")
    duration = count / fps
    if abs(float(video.get("duration", -1)) - duration) > 0.002:
        errors.append(f"video duration={video.get('duration')}, expected {duration}")
    if abs(float(video.get("start_time", 0))) > 0.002:
        errors.append("video does not start at zero")
    if rotation(video) % 360 != 0:
        errors.append(f"nonzero output rotation={rotation(video)}")
    if audio.get("codec_name") != "aac" or int(audio.get("sample_rate", 0)) != 48000:
        errors.append("audio must be AAC at 48000 Hz")
    # AAC packet padding permits at most one frame plus a small rounding margin.
    if abs(float(audio.get("duration", -1)) - duration) > 1024 / 48000 + 0.005:
        errors.append(f"audio duration={audio.get('duration')}, expected {duration}")
    if abs(float(audio.get("start_time", 0))) > 0.025:
        errors.append("audio does not start near zero")
    if errors:
        raise ValueError(f"Invalid output {path}: " + "; ".join(errors))
    return data


def source_info(path, kind, fps, start, count):
    if not path.is_file():
        raise ValueError(f"Missing source: {path}")
    data = probe(path)
    video, audio = stream(data, "video"), stream(data, "audio")
    if video is None or audio is None:
        raise ValueError(f"{path}: source must contain video and audio")
    if (video["width"], video["height"]) != (3840, 2160):
        raise ValueError(f"{path}: expected 3840x2160 source")
    if Fraction(video["avg_frame_rate"]) != fps or Fraction(video["r_frame_rate"]) != fps:
        raise ValueError(f"{path}: expected constant {fps} fps")
    if int(video.get("nb_frames", -1)) < start + count:
        raise ValueError(f"{path}: insufficient source frames or unknown frame count")
    expected_range = "pc" if kind == "exo" else "tv"
    if video.get("color_range") != expected_range:
        raise ValueError(f"{path}: expected {expected_range} color range")
    if rotation(video) % 180 != 0:
        raise ValueError(f"{path}: portrait rotation would change requested dimensions")
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "video": video, "audio": audio,
            "source_display_rotation_degrees": rotation(video),
            "orientation": "FFmpeg default autorotate applied; output rotation cleared"}


def audio_filter(info, start, count, fps):
    begin, end, duration = start / fps, (start + count) / fps, count / fps
    offset = float(info["audio"].get("start_time", 0)) - float(info["video"].get("start_time", 0))
    return (f"asetpts=PTS-STARTPTS+({offset:.9f})/TB,"
            f"atrim=start={begin:.9f}:end={end:.9f},asetpts=PTS-({begin:.9f})/TB,"
            f"aresample=48000:async=1:first_pts=0,apad,atrim=end={duration:.9f}")


def encode_options(kind, fps, count):
    result = ["-r", str(fps), "-fps_mode", "cfr", "-frames:v", str(count),
              "-c:v", "hevc_videotoolbox" if kind == "ego" else "h264_videotoolbox",
              "-allow_sw", "0", "-b:v", str(SPECS[kind][4]),
              "-pix_fmt", "p010le" if kind == "ego" else "nv12",
              "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
              "-color_primaries", "bt709", "-c:a", "aac", "-b:a", "192k",
              "-ar", "48000", "-t", f"{count / fps:.9f}",
              "-map_metadata", "-1", "-metadata:s:v:0", "rotate=0",
              "-movflags", "+faststart"]
    if kind == "ego":
        result += ["-profile:v", "main10", "-tag:v", "hvc1"]
    return result


def make_labels(path):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGBA", (1920, 54), (0, 0, 0, 175))
    font_path = Path("/System/Library/Fonts/Helvetica.ttc")
    if font_path.is_file():
        font = ImageFont.truetype(str(font_path), 30)
    else:
        try:
            font = ImageFont.load_default(size=30)
        except TypeError:
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    draw.text((24, 9), "EXO/R8", font=font, fill="white")
    draw.text((984, 9), "EGO/Nano", font=font, fill="white")
    image.save(path)


def export_pair(pair, manifest, manifest_path):
    pair_id, fps = str(pair["id"]), int(manifest["fps"])
    if fps != 25 or not pair_id.isdigit():
        raise ValueError("This capture exporter expects fps=25 and a numeric pair id")
    count = pair["frame_count"]
    starts = {kind: pair[f"{kind}_start_frame"] for kind in ("ego", "exo")}
    if type(count) is not int or count <= 0 or any(type(v) is not int or v < 0 for v in starts.values()):
        raise ValueError("Frame counts and starting frames must be positive/nonnegative integers")
    sources = {kind: source_info(resolve(pair[kind]), kind, fps, starts[kind], count)
               for kind in ("ego", "exo")}
    directory = resolve(manifest["output_root"]) / pair_id
    directory.mkdir(parents=True, exist_ok=True)
    targets = {kind: directory / (f"{kind}_aligned.mp4" if kind != "comparison" else "comparison.mp4")
               for kind in SPECS}
    signature = {"fps": fps, "frame_count": count, "starts": starts,
                 "sources": {k: {x: v[x] for x in ("path", "size", "mtime_ns")}
                             for k, v in sources.items()}, "comparison_audio": "ego"}
    log_path = directory / "export_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        if log.get("signature") != signature:
            raise ValueError(f"{log_path}: existing export settings differ; refusing overwrite")
    else:
        if any(path.exists() for path in targets.values()):
            raise ValueError(f"{directory}: existing output has no provenance log; refusing reuse")
        log = {"created_at": now(), "manifest": str(manifest_path), "pair_id": pair_id,
               "signature": signature, "sources": sources, "outputs": {}, "commands": [],
               "notes": ["Comparison audio: EGO/Nano close microphone.",
                         "Frame trim uses full decoding; no input seek, interpolation or speed change.",
                         "Metadata ffprobe checks are not a substitute for full decode or visual review."]}
    log["status"] = "running"
    write_json(log_path, log)
    pending = []
    for kind, path in targets.items():
        if path.exists():
            metadata = validate(path, kind, fps, count)
            log["outputs"][kind] = {"path": str(path), "status": "reused", "probe": metadata}
            print(f"[{pair_id}] reused {path.name}", flush=True)
        else:
            pending.append(kind)
    estimate = int(sum((SPECS[k][4] + 192_000) * (count / fps) / 8 * 1.10 for k in pending))
    free = shutil.disk_usage(directory).free
    log["space_check"] = {"free_bytes": free, "estimated_output_bytes": estimate,
                          "reserve_bytes": RESERVE, "checked_at": now()}
    if free < estimate + RESERVE:
        raise ValueError(f"Insufficient free space: need {estimate + RESERVE:,} bytes, have {free:,}")
    write_json(log_path, log)
    labels = directory / ".comparison_labels.partial.png"
    labels_created = False
    try:
        for kind in pending:
            target = targets[kind]
            temporary = target.with_name(target.stem + ".partial.mp4")
            if temporary.exists():
                raise ValueError(f"Existing partial output requires inspection: {temporary}")
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-n",
                       "-filter_threads", "2", "-filter_complex_threads", "2"]
            if kind in ("exo", "ego"):
                info, start = sources[kind], starts[kind]
                vf = f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS"
                if kind == "exo":
                    vf += ",scale=in_range=full:out_range=tv:in_color_matrix=bt709:out_color_matrix=bt709"
                vf += ",setsar=1,format=" + ("p010le" if kind == "ego" else "nv12")
                command += ["-hwaccel", "videotoolbox", "-threads", "4", "-i", info["path"],
                            "-map", "0:v:0", "-map", "0:a:0",
                            "-vf", vf, "-af", audio_filter(info, start, count, fps)]
            else:
                if labels.exists():
                    raise ValueError(f"Existing temporary label image: {labels}")
                make_labels(labels)
                labels_created = True
                graph = ("[0:v]scale=960:540:in_range=tv:out_range=tv,setsar=1,format=yuv420p[left];"
                         "[1:v]scale=960:540:in_range=tv:out_range=tv,setsar=1,format=yuv420p[right];"
                         "[left][right]hstack=inputs=2[stack];"
                         "[stack][2:v]overlay=0:0:shortest=1,"
                         "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709,"
                         "format=nv12[out]")
                command += ["-hwaccel", "videotoolbox", "-threads", "4", "-i", str(targets["exo"]),
                            "-hwaccel", "videotoolbox", "-threads", "4", "-i", str(targets["ego"]),
                            "-loop", "1", "-framerate", str(fps), "-i", str(labels),
                            "-filter_complex", graph, "-map", "[out]", "-map", "1:a:0"]
            command += encode_options(kind, fps, count) + [str(temporary)]
            ffmpeg_log = directory / f"{kind}_ffmpeg_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.log"
            entry = {"kind": kind, "argv": command, "started_at": now(), "stderr_log": str(ffmpeg_log)}
            log["commands"].append(entry)
            write_json(log_path, log)
            print(f"[{pair_id}] encoding {target.name} ({count} frames)", flush=True)
            with ffmpeg_log.open("x") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            entry.update({"finished_at": now(), "returncode": result.returncode})
            write_json(log_path, log)
            if result.returncode:
                raise RuntimeError(f"ffmpeg failed; inspect {ffmpeg_log}; partial retained: {temporary}")
            metadata = validate(temporary, kind, fps, count)
            # Hard link creates the final name without replacing an existing file.
            os.link(temporary, target)
            temporary.unlink()
            log["outputs"][kind] = {"path": str(target), "status": "exported", "probe": metadata}
            write_json(log_path, log)
            print(f"[{pair_id}] verified {target.name}", flush=True)
        log.update({"status": "complete", "completed_at": now()})
        write_json(log_path, log)
        print(f"[{pair_id}] complete; comparison audio=EGO/Nano", flush=True)
    except Exception as error:
        log.update({"status": "failed", "error": str(error), "failed_at": now()})
        write_json(log_path, log)
        raise
    finally:
        if labels_created and labels.exists():
            labels.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair", help="Export just this pair id (e.g. 01)")
    args = parser.parse_args()
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RuntimeError(f"Required executable is unavailable: {program}")
    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    pairs = manifest["pairs"]
    ids = [str(p["id"]) for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pair ids in manifest")
    if args.pair:
        pairs = [p for p in pairs if str(p["id"]) == args.pair]
        if not pairs:
            raise ValueError(f"Unknown pair id: {args.pair}")
    for pair in pairs:
        export_pair(pair, manifest, manifest_path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
