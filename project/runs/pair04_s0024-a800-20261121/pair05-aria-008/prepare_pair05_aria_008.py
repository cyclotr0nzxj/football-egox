"""Package and CPU-check the previously approved Pair05 inputs for one Aria GGA run."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PAIR01 = RUN / "pair01-gga-ab"
sys.path.insert(0, str(PAIR01))
from prepare_bundle import (  # noqa: E402
    PROJECT, REPLAY, SOURCE_META_SHA, SOURCE_RECEIPT_SHA, read_frozen, sha256, stage_local_media,
)

TAKE = "pair05_s0024"
ORIGINAL_ZIP = PAIR01 / "pair01-gga-ab-run-a-bundle.zip"
ORIGINAL_SHA = "aa198537fe87f4948ef28b236293cd9f889f1deb9e6e19ed1f0f5ee4357554ab"
APPROVED_POSE_SHA = "c2422ed6f6ecb0c150b8be612bc7c927cc39f5ec70cdd5a0593d8811d8d5846c"
ORIGINAL_PREFIX = "pair01-gga-ab-bundle"
PACKAGE = "pair05-aria-bundle"
LAUNCHER = "run_pair05_aria_008.sh"


def freeze_inputs(package):
    approved = read_frozen(RUN / f"ball-view-cpu-review/{TAKE}-ball-view-meta.json", APPROVED_POSE_SHA)
    sources = read_frozen(REPLAY / "logs/meta.json", SOURCE_META_SHA)["test_datasets"]
    entry, = [row for row in sources if row["exo_path"] == f"./example/self_capture/videos/{TAKE}/exo.mp4"]
    if approved != {"test_datasets": [entry]}:
        raise ValueError("Pair05 approved tracking pose differs from the historical render metadata.")
    receipts = read_frozen(REPLAY / "logs/validated-inputs.json", SOURCE_RECEIPT_SHA)
    receipt, = [item for item in receipts if item["name"] == TAKE]
    (package / "meta.json").write_text(json.dumps(approved, indent=2) + "\n")
    (package / "validated-inputs.json").write_text(json.dumps([receipt], indent=2) + "\n")
    return receipt["depth_sha256"]


def stage_scripts(package):
    if sha256(ORIGINAL_ZIP) != ORIGINAL_SHA:
        raise ValueError("The previously executed launcher bundle has an unexpected hash.")
    with ZipFile(ORIGINAL_ZIP) as original:
        def member(name):
            return original.read(f"{ORIGINAL_PREFIX}/{name}")

        for name in ("aria_gga.py", "check_runtime_008.py"):
            (package / name).write_bytes(member(name))
        for name in ("check_inputs.py", "prepare_infer.py", "verify_ab_008.py"):
            source = member(name).decode("utf-8")
            if source.count('TAKE = "pair01_s0288"') != 1:
                raise ValueError(f"Unexpected take declaration in {name}.")
            (package / name).write_text(source.replace("pair01_s0288", TAKE).replace("Pair01", "Pair05"))

        runner, tail = member("run_pair01_gga_ab_008.sh").decode("utf-8").split("\nrun_arm A\n")
        if tail.count("\nrun_arm B\n") != 1:
            raise ValueError("Unexpected historical A/B driver.")
        runner = (runner.replace(" && $mode != run-a", "").replace("preflight|run|run-a", "preflight|run")
                  .replace("pair01-gga-ab-results", "pair05-aria-results").replace("pair01_s0288", TAKE))
        runner += '\nrun_arm B\nprintf \'EGOX_PAIR05_COMPLETE %s %s\\n\' "$fingerprint" "$out_root"\n'
        (package / LAUNCHER).write_text(runner)


def prepare(output):
    with tempfile.TemporaryDirectory(prefix="pair05-aria-") as directory:
        work = Path(directory)
        package = work / PACKAGE
        package.mkdir()
        expected_depths = freeze_inputs(package)
        stage_scripts(package)
        media = stage_local_media(work, expected_depths, take=TAKE)
        inputs = subprocess.check_output(
            [sys.executable, "-B", str(package / "check_inputs.py"), "--media-root", str(media)],
            text=True, timeout=90,
        )
        projection = subprocess.check_output(
            [sys.executable, "-B", str(package / "prepare_infer.py"),
             "--official-infer", str(PROJECT / "vendor/EgoX/infer.py"),
             "--meta", str(package / "meta.json"), "--check-only"],
            text=True, timeout=30,
        )
        subprocess.run(["bash", "-n", str(package / LAUNCHER)], check=True)
        (package / "SHA256SUMS").write_text("".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(package.iterdir())
        ))
        with ZipFile(output, mode="x") as bundle:
            for path in sorted(package.iterdir()):
                bundle.writestr(ZipInfo(f"{PACKAGE}/{path.name}"), path.read_bytes(), compress_type=ZIP_DEFLATED)
    return {"inputs": json.loads(inputs), "projection": json.loads(projection), "sha256": sha256(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "pair05-aria-bundle.zip")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
