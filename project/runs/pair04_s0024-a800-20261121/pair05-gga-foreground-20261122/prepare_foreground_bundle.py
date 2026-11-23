"""Package and check the frozen Pair05 foreground run entirely offline."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from prepare_bundle import PROJECT, sha256, stage_local_media  # noqa: E402


TAKE = "pair05_s0024"
PACKAGE = "pair05-gga-foreground-bundle"
LAUNCHER = "run_pair05_gga_foreground_008.sh"
PLAN = Path("plan/pair05-gga-routes.npz")
ARIA_PREFIX = "pair05-aria-bundle"
ARIA_BUNDLE = RUN / "pair05-aria-008" / f"{ARIA_PREFIX}.zip"
ARIA_SHA = "d546f888d70e0762d05d4bcea584682225ba6e7edd729c1980f4b142b7757d8a"
ARIA_FILES = ("aria_gga.py", "meta.json", "validated-inputs.json", "check_inputs.py",
              "check_runtime_008.py", "prepare_infer.py", "verify_ab_008.py")
FOREGROUND_FILES = (LAUNCHER, "stage_foreground_008.py", "verify_foreground_008.py",
                    "check_foreground_cuda.py")
DEFAULT_TORCH = Path("/usr/local/Caskroom/miniforge/base/envs/dino_wm_local/bin/python")


def local_command(command, *, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or f"Check failed: {command}")
    return result.stdout.strip()


def prepare(output, torch_python):
    if sha256(ARIA_BUNDLE) != ARIA_SHA:
        raise ValueError("The previously executed Pair05 Aria bundle differs from the frozen reference.")
    with tempfile.TemporaryDirectory(prefix="pair05-foreground-") as directory:
        work = Path(directory)
        package = work / PACKAGE
        package.mkdir()
        with ZipFile(ARIA_BUNDLE) as prior:
            for name in ARIA_FILES:
                (package / name).write_bytes(prior.read(f"{ARIA_PREFIX}/{name}"))
        for name in FOREGROUND_FILES:
            shutil.copyfile(HERE / name, package / name)
        shutil.copytree(HERE / "staged", package / "staged", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (package / PLAN).parent.mkdir()
        shutil.copyfile(HERE / PLAN, package / PLAN)

        receipt, = json.loads((package / "validated-inputs.json").read_bytes())
        media = stage_local_media(work, receipt["depth_sha256"], take=TAKE)
        official = PROJECT / "vendor/EgoX"
        verify_code = runpy.run_path(str(package / "check_runtime_008.py"))["verify_code"]
        verify_code(official, PROJECT / "scripts/instrumented_infer.py")
        inputs = json.loads(local_command(
            [sys.executable, "-I", "-B", str(package / "check_inputs.py"), "--media-root", str(media)]))
        projection = json.loads(local_command(
            [sys.executable, "-B", str(package / "prepare_infer.py"),
             "--official-infer", str(official / "infer.py"), "--meta", str(package / "meta.json"), "--check-only"]))
        stage = work / "local-foreground-stage"
        stage_result = local_command(
            [sys.executable, "-B", str(package / "stage_foreground_008.py"),
             "--official", str(official), "--media-root", str(media), "--stage", str(stage)])
        structure = local_command(
            [str(torch_python), "-I", "-B", str(package / "check_foreground_cuda.py"), "--check-structure",
             "--stage", str(stage), "--plan", str(stage / PLAN), "--meta", str(stage / "meta.json")], cwd=stage)
        local_command(["bash", "-n", str(package / LAUNCHER)])

        files = sorted(path for path in package.rglob("*") if path.is_file())
        checksums = {path.relative_to(package).as_posix(): sha256(path) for path in files}
        checksum_file = package / "SHA256SUMS"
        checksums_data = "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()).encode()
        checksum_file.write_bytes(checksums_data)
        temporary = work / f"{PACKAGE}.zip"
        with ZipFile(temporary, mode="x") as bundle:
            for path in sorted(files + [checksum_file]):
                entry = ZipInfo(f"{PACKAGE}/{path.relative_to(package).as_posix()}")
                bundle.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
        with ZipFile(temporary) as bundle:
            archived = {name: hashlib.sha256(bundle.read(f"{PACKAGE}/{name}")).hexdigest() for name in checksums}
            if (bundle.testzip() is not None or archived != checksums or len(bundle.namelist()) != len(files) + 1
                    or bundle.read(f"{PACKAGE}/{checksum_file.name}") != checksums_data):
                raise ValueError("ZIP checksum or file list differs from the prepared inputs.")
        with output.open("xb") as stream, temporary.open("rb") as source:
            shutil.copyfileobj(source, stream)

    return {"bundle": output.name, "sha256": sha256(output), "archive_files": len(files) + 1,
            "metadata_sha256": checksums["meta.json"], "input_receipt_sha256": checksums["validated-inputs.json"],
            "launch_manifest_sha256": hashlib.sha256(checksums_data).hexdigest(),
            "plan_sha256": checksums[PLAN.as_posix()], "source_aria_bundle_sha256": ARIA_SHA,
            "inputs": inputs, "matched_aria_projection_reference": projection, "staging": stage_result,
            "plan_structure": structure,
            "verified": ["official_infer_and_30_core_files", "official_observational_wrapper", "launcher_syntax",
                         "archive_contents_and_crc"], "cuda_executed": False, "inference_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / f"{PACKAGE}.zip")
    parser.add_argument("--torch-python", type=Path, default=DEFAULT_TORCH)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.output, args.torch_python.resolve(strict=True)), indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Pair05 foreground package not prepared: {error}\n")


if __name__ == "__main__":
    main()
