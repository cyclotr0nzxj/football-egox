"""Package the reviewed camera candidate and replay the preflight gates locally."""

import argparse
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT = RUN.parents[1]
REFERENCE = RUN / "multiclip-evidence"
AB = RUN / "pair01-gga-ab"
UPDATE = RUN / "update-008"
sys.path.insert(0, str(AB))
sys.path.insert(0, str(UPDATE))
sys.path.insert(0, str(RUN / "ball-view-cpu-review"))
from build_ball_previews import load_depth
from check_inputs import DEPTHS, EXO, PRIOR, TAKE, check_video, sha256
from stage_update_008 import OLD_RECEIPT_SHA, RENDERER_SHA, require_digest, verify_prior

BUNDLE_NAME = "pair01-first-orientation-bundle"
FIRST_META_SHA = "bf649f277c5987098a9815e17cb7cc8bf1a3fde1632d78af7d0451aaa343f88c"
CANDIDATE_SHA = "b34c2e94d40c5f1ad587e2663d2607443e7c69599543cedbc3ea4f130bf523c4"
PREVIOUS_BUNDLE_SHA = "0dac0e67ba0fc80dba5d44f0d23157b4b69b2dfcf452eb46d08ec49a8612cc5a"


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text())


def build(package, fixture):
    metadata = HERE / "results/camera-candidate.json"
    receipts = REFERENCE / "validated-inputs.json"
    require_digest(REFERENCE / "meta.json", FIRST_META_SHA)
    require_digest(receipts, OLD_RECEIPT_SHA)
    require_digest(metadata, CANDIDATE_SHA)
    require_digest(UPDATE / "update-bundle.zip", PREVIOUS_BUNDLE_SHA)
    require_digest(PROJECT / "records/official_renderer_geometry.py", RENDERER_SHA)
    first = read(REFERENCE / "meta.json")["test_datasets"][1]
    candidate, = read(metadata)["test_datasets"]
    if (set(candidate) != set(first) or {key for key in first if candidate[key] != first[key]} != {"ego_extrinsics"}
            or candidate["ego_extrinsics"][:4] != first["ego_extrinsics"][:4]
            or candidate["exo_path"] != f"./{EXO}" or candidate["ego_prior_path"] != f"./{PRIOR}"):
        raise ValueError("The candidate must use the same first-run inputs and opening matrices.")
    original, revised = [np.asarray(row["ego_extrinsics"]) for row in (first, candidate)]
    for pose in (original, revised):
        if pose.shape != (49, 3, 4) or not np.allclose(pose[:, :, :3] @ pose[:, :, :3].transpose(0, 2, 1), np.eye(3)):
            raise ValueError("Expected 49 valid camera rotations.")
    centers = [-pose[:, :, :3].transpose(0, 2, 1) @ pose[:, :, 3:] for pose in (original, revised)]
    delta = float(np.abs(centers[0] - centers[1]).max())
    if delta > 1e-8 or not np.allclose(np.linalg.det(revised[:, :, :3]), 1):
        raise ValueError("Camera position or handedness changed.")

    receipt, = [row for row in read(receipts) if row["name"] == TAKE]
    with ZipFile(UPDATE / "update-bundle.zip") as old:
        geometry, = [row for row in json.loads(old.read("update-bundle/plan.json")) if row["name"] == TAKE]
    depth_root = REFERENCE / f"{TAKE}/geometry/vipe_results/{TAKE}"
    # The stock renderer can also read the ViPE intrinsics text export.
    geometry["geometry_sha256"]["intrinsics/exo_camera.txt"] = "cd0772f72835c4da35e4833e4d7bd53f0e6f6093e50f1fa82f1522064b08d682"
    for relative, expected in geometry["geometry_sha256"].items():
        require_digest(depth_root / relative, expected)
    if geometry["source_exo_sha256"] != receipt["exo_sha256"] or len(receipt["depth_sha256"]) != 49:
        raise ValueError("Pair01 original source and depth receipts disagree.")
    exo = RUN / "multiclip-prep" / TAKE / EXO
    first_prior = REFERENCE / TAKE / "teammate-prior" / TAKE / "ego_Prior.mp4"
    for path, expected in ((exo, receipt["exo_sha256"]), (first_prior, receipt["prior_sha256"])):
        require_digest(path, expected)
    check_video(exo, 784, 25, "ffprobe")
    verify_prior(first_prior, "ffprobe", "ffmpeg")
    (fixture / DEPTHS).mkdir(parents=True)
    with ZipFile(depth_root / "depth/exo.zip") as depths:
        for f, expected in enumerate(receipt["depth_sha256"]):
            data = BytesIO()
            np.save(data, load_depth(depths, f), allow_pickle=False)
            path = fixture / DEPTHS / f"{f:05}.npy"
            path.write_bytes(data.getvalue())
            require_digest(path, expected)
    for relative, path in ((EXO, exo), (PRIOR, first_prior), (Path("infer.py"), PROJECT / "vendor/EgoX/infer.py"),
                           (Path("core"), PROJECT / "vendor/EgoX/core")):
        dest = fixture / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(path)

    sources = [HERE / "run_candidate_008.py", AB / "check_inputs.py", AB / "check_runtime_008.py",
               AB / "verify_ab_008.py", UPDATE / "stage_update_008.py"]
    for source in sources:
        data = source.read_bytes()
        compile(data, str(source), "exec")
        (package / source.name).write_bytes(data)
    (package / "meta.json").write_bytes(metadata.read_bytes())
    plan = {"geometry": geometry, "first_input_receipt": receipt, "first_batch_meta_sha256": FIRST_META_SHA}
    write(package / "plan.json", plan)
    write(package / "bundle-hashes.json", {p.name: sha256(p) for p in sorted(package.iterdir())})
    return plan, delta


def smoke(work, package, fixture, plan):
    launcher = package / "run_candidate_008.py"
    subprocess.run([sys.executable, "-B", str(launcher), "verify-bundle"], check=True, stdout=subprocess.DEVNULL)
    spec = importlib.util.spec_from_file_location("candidate_launcher", launcher)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    compile(stage.CONDITIONING_CHECK, "<conditioning check>", "exec")
    prior_fixture = UPDATE / "output-008/priors" / f"{TAKE}-final.mp4"
    verify_prior(prior_fixture, "ffprobe", "ffmpeg")
    attempt = work / "attempt-local-fixture"
    attempt.mkdir()
    stage.stage_inputs(attempt / "work", fixture, prior_fixture, plan["first_input_receipt"], package / "meta.json", "ffprobe")
    visual = {key: True for key in stage.REVIEW_KEYS}
    prior = {"prior_sha256": sha256(prior_fixture)}
    gate = work / "review.json"
    for changes in ({}, {"prior_sha256": "0" * 64}, {"foot_ball_contact_visible": False}):
        if changes:
            gate.write_text(json.dumps(dict(visual, **prior, observations="fixture") | changes))
        try:
            stage.check_visual_review(gate, prior)
        except (OSError, ValueError):
            pass
        else:
            raise ValueError("The prior guard accepted missing or invalid evidence.")
    gate.write_text(json.dumps(dict(visual, **prior, observations="historical local fixture only")))
    assert stage.check_visual_review(gate, prior)["prior_sha256"] == prior["prior_sha256"]
    (attempt / "inference-start.json").write_text("{}\n")
    try:
        stage.infer(attempt, plan, {}, "", gate)
    except ValueError as error:
        if "already started" not in str(error):
            raise
    else:
        raise ValueError("The repeat guard would launch the same attempt twice.")
    metadata = package / "meta.json"
    good = metadata.read_bytes()
    metadata.write_bytes(good + b" ")
    rejected = subprocess.run([sys.executable, "-B", str(launcher), "verify-bundle"], capture_output=True, text=True)
    metadata.write_bytes(good)
    if rejected.returncode == 0 or "Changed staged/cache file" not in rejected.stderr:
        raise ValueError("The bundle did not reject changed metadata.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="pair01-candidate-local-") as folder:
        work = Path(folder)
        package = work / BUNDLE_NAME
        package.mkdir()
        fixture = work / "historical-media-fixture"
        plan, delta = build(package, fixture)
        smoke(work, package, fixture, plan)
        with ZipFile(args.output, mode="x", compression=ZIP_DEFLATED) as archive:
            for path in sorted(package.iterdir()):
                archive.writestr(ZipInfo(f"{BUNDLE_NAME}/{path.name}"), path.read_bytes(), compress_type=ZIP_DEFLATED)
    print(json.dumps({"status": "local_preflight_pass", "bundle_file": args.output.name,
                      "bundle_sha256": sha256(args.output), "bundle_bytes": args.output.stat().st_size,
                      "verified_depth_frames": 49, "renderer_assets_verified": len(plan["geometry"]["geometry_sha256"]),
                      "first_and_candidate_center_max_delta": delta,
                      "passed": ["frozen_inputs", "historical_fixture_stage", "visual_receipt_gate",
                                 "repeat_inference_gate", "tampered_bundle_rejected"],
                      "live_render_or_inference": False}, indent=2))


if __name__ == "__main__":
    main()
