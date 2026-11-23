"""Rebuild the Pair01 startup bundle after a CPU check of the real inputs."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from check_inputs import DEPTHS, EXO, FRAMES, PRIOR, TAKE, sha256, validate


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT = RUN.parents[1]
REPLAY = RUN / "update-008/output-008"
SOURCE_META_SHA = "451ca56c98508b88bf486322c0704ea1c270736326e7d4c6a4bfa9d0b67ad6aa"
SOURCE_RECEIPT_SHA = "d5571e208a00a9dde3a62f8a762e630c9078ece7b2f3703a3078b4b6b9204448"
APPROVED_POSE_SHA = "e4f801e2d5a56c7d2c50da904fede6ade4d4572e45546f010063688ff51819c6"
SCRIPTS = (
    "aria_gga.py", "check_inputs.py", "prepare_infer.py", "run_pair01_gga_ab_008.sh",
    "check_runtime_008.py", "verify_ab_008.py",
)


def read_frozen(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected historical source hash: {path}")
    return json.loads(data)


def select_pair01(package):
    history = read_frozen(REPLAY / "logs/meta.json", SOURCE_META_SHA)
    receipts = read_frozen(REPLAY / "logs/validated-inputs.json", SOURCE_RECEIPT_SHA)
    approved = read_frozen(RUN / f"ball-view-cpu-review/{TAKE}-ball-view-meta.json", APPROVED_POSE_SHA)
    merged, = [row for row in history["test_datasets"] if row["exo_path"] == f"./{EXO}"]
    if approved != {"test_datasets": [merged]}:
        raise ValueError("Historically executed metadata and approved camera metadata differ.")
    row, = [item for item in receipts if item["name"] == TAKE]
    (package / "meta.json").write_text(json.dumps({"test_datasets": [merged]}, indent=2) + "\n")
    (package / "validated-inputs.json").write_text(json.dumps([row], indent=2) + "\n")
    return row["depth_sha256"]


def stage_local_media(work, expected_depths, take=TAKE):
    sys.path.insert(0, str(RUN / "ball-view-cpu-review"))
    from build_ball_previews import load_depth
    import numpy as np

    media = work / "local-media"
    exo = EXO.parent.with_name(take) / EXO.name
    prior = PRIOR.parent.with_name(take) / PRIOR.name
    depths = DEPTHS.with_name(take)
    for relative, source in [
        (exo, RUN / "multiclip-prep" / take / exo),
        (prior, REPLAY / f"priors/{take}-final.mp4"),
    ]:
        target = media / relative
        target.parent.mkdir(parents=True)
        target.symlink_to(source)
    archive = RUN / f"multiclip-evidence/{take}/geometry/vipe_results/{take}/depth/exo.zip"
    directory = media / depths
    directory.mkdir(parents=True)
    with ZipFile(archive) as original:
        for frame in range(FRAMES):
            data = BytesIO()
            np.save(data, load_depth(original, frame), allow_pickle=False)
            payload = data.getvalue()
            if hashlib.sha256(payload).hexdigest() != expected_depths[frame]:
                raise ValueError(f"Decoded ViPE depth differs from the certified NPY at frame {frame}.")
            (directory / f"{frame:05}.npy").write_bytes(payload)
    return media


def prepare(output):
    with tempfile.TemporaryDirectory(prefix="pair01-gga-ab-") as directory:
        work = Path(directory)
        package = work / "pair01-gga-ab-bundle"
        package.mkdir()
        depth_hashes = select_pair01(package)
        media = stage_local_media(work, depth_hashes)
        checked = validate(package, media, "ffprobe")
        for script in SCRIPTS:
            shutil.copyfile(HERE / script, package / script)
        projection = subprocess.run(
            [sys.executable, "-B", str(package / "prepare_infer.py"),
             "--official-infer", str(PROJECT / "vendor/EgoX/infer.py"),
             "--meta", str(package / "meta.json"), "--check-only"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        checked["projection"] = json.loads(projection.stdout)
        subprocess.run(["bash", "-n", str(package / "run_pair01_gga_ab_008.sh")], check=True)
        files = sorted(package.iterdir())
        (package / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
        with ZipFile(output, mode="x") as bundle:
            for path in sorted(package.iterdir()):
                entry = ZipInfo(f"{package.name}/{path.name}")
                bundle.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
    return dict(checked, file=output.name, sha256=sha256(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "pair01-gga-ab-bundle.zip")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.output), indent=2))
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Pair01 package not prepared: {error.stderr or error}\n")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Pair01 package not prepared: {error}\n")


if __name__ == "__main__":
    main()
