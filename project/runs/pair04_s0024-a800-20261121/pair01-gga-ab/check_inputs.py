"""Verify the exact Pair01 inference inputs used by both A/B variants."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess


TAKE = "pair01_s0288"
SELF_CAPTURE = Path("example/self_capture")
EXO = SELF_CAPTURE / "videos" / TAKE / "exo.mp4"
PRIOR = SELF_CAPTURE / "ego_priors" / TAKE / "ego_Prior.mp4"
DEPTHS = SELF_CAPTURE / "depth_maps" / TAKE
FRAMES = 49


def sha256(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check_video(path, width, fps, ffprobe):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_frames", "-show_entries",
         "stream=width,height,avg_frame_rate,time_base:frame=width,height,pts",
         "-of", "json", str(path)],
        check=True, capture_output=True, timeout=30,
    )
    info = json.loads(result.stdout)
    stream, = info["streams"]
    if result.stderr or (stream["width"], stream["height"], Fraction(stream["avg_frame_rate"])) != (width, 448, fps):
        raise ValueError(f"Unexpected video dimensions, frame rate, or decode error: {path}")
    time_base = Fraction(stream["time_base"])
    frames = [(f["width"], f["height"], int(f["pts"]) * time_base) for f in info["frames"]]
    if frames != [(width, 448, Fraction(i, fps)) for i in range(FRAMES)]:
        raise ValueError(f"Expected 49 decoded frames with exact frame timestamps: {path}")


def validate(bundle, media_root, ffprobe):
    metadata = json.loads((bundle / "meta.json").read_text())
    row, = metadata["test_datasets"]
    keys = {"exo_path", "ego_prior_path", "prompt", "camera_intrinsics",
            "camera_extrinsics", "ego_intrinsics", "ego_extrinsics"}
    if (set(metadata) != {"test_datasets"} or set(row) != keys
            or row["exo_path"] != f"./{EXO}" or row["ego_prior_path"] != f"./{PRIOR}"
            or len(row["ego_extrinsics"]) != FRAMES):
        raise ValueError("Expected only the frozen EXO, camera, prompt, and prior for Pair01.")
    receipt, = json.loads((bundle / "validated-inputs.json").read_text())
    depth_hashes = receipt["depth_sha256"]
    if receipt["name"] != TAKE or receipt["frames"] != FRAMES or len(depth_hashes) != FRAMES:
        raise ValueError("Incorrect input receipt for Pair01.")

    depths = sorted((media_root / DEPTHS).glob("*.npy"))
    indices = [re.sub(r"\D", "", path.stem) for path in depths]
    if len(indices) != FRAMES or any(not index or int(index) != i for i, index in enumerate(indices)):
        raise ValueError("Expected exactly 49 depth files in inference order.")
    frozen = [(media_root / EXO, receipt["exo_sha256"]), (media_root / PRIOR, receipt["prior_sha256"])]
    frozen.extend(zip(depths, depth_hashes))
    for path, expected in frozen:
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(path) != expected:
            raise ValueError(f"Frozen SHA-256 mismatch: {path}")
    check_video(media_root / EXO, 784, 25, ffprobe)
    check_video(media_root / PRIOR, 448, 30, ffprobe)
    return {"take": TAKE, "frames": FRAMES, "depths_verified": len(depths),
            "exo_sha256": receipt["exo_sha256"], "prior_sha256": receipt["prior_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        result = validate(Path(__file__).resolve().parent, args.media_root, args.ffprobe)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Pair01 input check failed: {error}\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
