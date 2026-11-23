"""Run an isolated Pair01 candidate in two phases on the authorized 008."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from check_inputs import DEPTHS, EXO, PRIOR, TAKE, validate
from stage_update_008 import BASE, RENDERER_SHA, digest, require_digest, validate_cache, verify_prior
from verify_ab_008 import publish_receipt

ROOT = BASE / "egox-shared-008"
FIRST = ROOT / "inference-four"
RUNS = ROOT / "pair01-first-orientation-results"
WEIGHTS = Path("/root/autodl-fs/football_egox/models")
RENDERER = BASE / "Renderer-sparse/scripts/render_vipe_pointcloud.py"
BINS = BASE / "renderer-env/bin"
PYTHON = ROOT / "egox-env/bin/python"
BUNDLE = Path(__file__).resolve().parent
SEED = 846514
REVIEW_FRAMES = (0, 4, *range(20, 29), *range(36, 44), 48)
REVIEW_KEYS = ("opening_white_shoe_teammate", "foot_ball_contact_visible", "single_visible_ball_F36_F43")

CONDITIONING_CHECK = """
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers.utils import load_video
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageDraw

prior, sheet_path = sys.argv[1:3]
review = json.loads(sys.argv[3])
frames = load_video(video=prior)
if len(frames) != 49 or any(im.size != (448, 448) or im.mode != 'RGB' for im in frames):
    raise ValueError('EgoX did not decode 49 RGB frames of 448x448')
# Stock core/inference/wan.py crops each whole frame. Its downstream processor
# takes the first 49 frames and uses 448x448, preserving their pixel positions.
frames = [im.crop((0, 0, *im.size)) for im in frames]
rgb = np.stack([np.asarray(im) for im in frames])
input_tensor = VideoProcessor(vae_scale_factor=8).preprocess(frames[:49], height=448, width=448).to(dtype=torch.float32)
recovered = ((input_tensor + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
if not np.array_equal(recovered, rgb):
    raise ValueError('EgoX spatial preprocessing changed prior pixels or frame order')
sheet = Image.new('RGB', (5 * 296, 4 * 316), '#171d25')
draw = ImageDraw.Draw(sheet)
for i, frame in enumerate(review):
    x, y = i % 5 * 296, i // 5 * 316
    sheet.paste(frames[frame].resize((280, 280)), (x + 8, y + 24))
    draw.text((x + 8, y + 6), f'F{frame:02}', fill='white')
with Path(sheet_path).open('xb') as output:
    sheet.save(output, format='PNG')
print(json.dumps({'count': len(frames), 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                  'preprocessed_pixels_match_in_order': True}))
"""


def read(path):
    return json.loads(path.read_text())


def verify_bundle():
    hashes = read(BUNDLE / "bundle-hashes.json")
    expected = {"run_candidate_008.py", "check_inputs.py", "check_runtime_008.py", "verify_ab_008.py",
                "stage_update_008.py", "meta.json", "plan.json"}
    if set(hashes) != expected:
        raise ValueError("Unexpected candidate bundle content.")
    for name, sha in hashes.items():
        require_digest(BUNDLE / name, sha)
    plan = read(BUNDLE / "plan.json")
    candidate, = read(BUNDLE / "meta.json")["test_datasets"]
    if (plan["geometry"]["name"] != TAKE or candidate["exo_path"] != f"./{EXO}"
            or candidate["ego_prior_path"] != f"./{PRIOR}" or len(candidate["ego_extrinsics"]) != 49):
        raise ValueError("Only the single frozen Pair01 candidate is supported.")
    return plan


def execute(args, *, timeout=120, env=None, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, timeout=timeout, env=env, **kwargs)


@contextmanager
def machine_lock():
    if socket.gethostname() != "autodl-container-cb894da57f-45e726a1":
        raise ValueError("Expected the authorized 008 container.")
    with (ROOT / ".egox-inference.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(["findmnt", "--mountpoint", WEIGHTS.parents[1].resolve()], stdout=subprocess.DEVNULL)
        gpu = execute(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.total,memory.used",
                       "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
        name, total, used = map(str.strip, gpu)
        processes = execute(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True).stdout.strip()
        if name != "NVIDIA A800 80GB PCIe" or total != "81920" or int(used) > 500 or processes:
            raise ValueError("Expected the idle A800 on 008.")
        yield


def offline_environment():
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", OMP_NUM_THREADS="8", PYTHONHASHSEED=str(SEED),
               IMAGEIO_FFMPEG_EXE=str(BINS / "ffmpeg"))
    return env


def preflight(plan, env):
    require_digest(FIRST / "meta.json", plan["first_batch_meta_sha256"])
    require_digest(RENDERER, RENDERER_SHA)
    frozen = validate_cache(ROOT, [plan["geometry"]])[TAKE]
    if frozen != plan["first_input_receipt"]:
        raise ValueError("First-run input receipt changed.")
    require_digest(FIRST / PRIOR, frozen["prior_sha256"])
    result = execute([PYTHON, "-I", "-B", BUNDLE / "check_runtime_008.py", ROOT, FIRST, WEIGHTS,
                      BINS / "ffmpeg", BINS / "ffprobe"], timeout=180, env=env, capture_output=True, text=True)
    print(result.stdout, end="", flush=True)
    fingerprint, = [line.split()[1] for line in result.stdout.splitlines()
                     if line.startswith("EGOX_RUNTIME_FINGERPRINT ")]
    driver = execute(["nvidia-smi", "-i", "0", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip()
    return hashlib.sha256(f"{digest(BUNDLE / 'bundle-hashes.json')}\n{fingerprint}\n{driver}".encode()).hexdigest()


def stage_inputs(work, first, prior, receipt, metadata, ffprobe):
    work.mkdir()
    for relative, source in ((EXO, first / EXO), (DEPTHS, first / DEPTHS), (PRIOR, prior),
                             (Path("infer.py"), first / "infer.py"), (Path("core"), first / "core")):
        link = work / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source.resolve(strict=True))
    (work / "meta.json").write_bytes(metadata.read_bytes())
    publish_receipt(work / "validated-inputs.json", [dict(receipt, prior_sha256=digest(prior))])
    validate(work, work, ffprobe)


def render(attempt, plan, env, fingerprint):
    renderer_env = dict(env, PYTHONPATH=str(RENDERER.parents[1]), PATH=f"{BINS}:{env['PATH']}",
                        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    smoke = """
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.points.rasterize_points import rasterize_points
points = Pointclouds(points=[torch.tensor([[0., 0., 1.]], device='cuda')])
indices, _, _ = rasterize_points(points, image_size=8, radius=0.4, points_per_pixel=1)
assert (indices == 0).any().item()
"""
    execute([BINS / "python", "-B", "-c", smoke], env=renderer_env, timeout=30)
    with (attempt / "render.log").open("x") as log:
        execute([BINS / "python", "-B", RENDERER, "--input_dir", BASE / plan["geometry"]["geometry_cache"],
                 "--out_dir", attempt / "prior", "--meta_json_path", BUNDLE / "meta.json", "--point_size", "5.0",
                 "--start_frame", "0", "--end_frame", "48", "--fish_eye_rendering", "--use_mean_bg"],
                env=renderer_env, timeout=360, stdout=log, stderr=subprocess.STDOUT)
    prior = attempt / "prior" / TAKE / "ego_Prior.mp4"
    verify_prior(prior, BINS / "ffprobe", BINS / "ffmpeg")
    source = plan["first_input_receipt"]
    if digest(prior) == source["prior_sha256"]:
        raise ValueError("The rendered candidate is byte-identical to the first prior.")
    stage_inputs(attempt / "work", FIRST, prior, source, BUNDLE / "meta.json", BINS / "ffprobe")
    decoded = execute([PYTHON, "-I", "-B", "-c", CONDITIONING_CHECK, prior, attempt / "prior-review.png",
                       json.dumps(REVIEW_FRAMES)], env=env, capture_output=True, text=True)
    conditioning = json.loads(decoded.stdout)
    receipt = {"status": "prior_technical_pass_visual_pending", "take": TAKE, "frames": 49, "fps": 30,
               "prior_sha256": digest(prior), "meta_sha256": digest(BUNDLE / "meta.json"),
               "runtime_fingerprint": fingerprint, "review_frames": list(REVIEW_FRAMES), "conditioning": conditioning,
               "review_image_sha256": digest(attempt / "prior-review.png")}
    publish_receipt(attempt / "prior-render.json", receipt)


def check_visual_review(approval, prior):
    visual = read(approval)
    if (visual.get("prior_sha256") != prior["prior_sha256"]
            or any(visual.get(key) is not True for key in REVIEW_KEYS)
            or not isinstance(visual.get("observations"), str) or not visual["observations"].strip()):
        raise ValueError("New prior needs a matching positive review of opening, contact, and single ball.")
    return visual


def infer(attempt, plan, env, fingerprint, approval):
    if (attempt / "inference-start.json").exists():
        raise ValueError("This attempt already started inference. Recover its evidence before deciding on a new run.")
    prior = read(attempt / "prior-render.json")
    work = attempt / "work"
    if (prior["status"] != "prior_technical_pass_visual_pending" or prior["take"] != TAKE
            or prior["runtime_fingerprint"] != fingerprint or prior["meta_sha256"] != digest(BUNDLE / "meta.json")):
        raise ValueError("Candidate render does not belong to this bundle and runtime.")
    if (prior["conditioning"]["count"] != 49 or prior["conditioning"]["preprocessed_pixels_match_in_order"] is not True
            or prior["review_frames"] != list(REVIEW_FRAMES)):
        raise ValueError("Candidate prior has no verified input-processor receipt.")
    require_digest(attempt / "prior-review.png", prior["review_image_sha256"])
    require_digest(work / "meta.json", prior["meta_sha256"])
    require_digest(work / "infer.py", digest(FIRST / "infer.py"))
    if (work / "core").resolve(strict=True) != (FIRST / "core").resolve(strict=True):
        raise ValueError("Staged EgoX core no longer matches the verified runtime.")
    source = plan["first_input_receipt"]
    if read(work / "validated-inputs.json") != [dict(source, prior_sha256=prior["prior_sha256"])]:
        raise ValueError("Candidate staged input receipt changed.")
    validate(work, work, str(BINS / "ffprobe"))
    visual = check_visual_review(approval, prior)
    run_fingerprint = hashlib.sha256(f"{fingerprint}\n{prior['prior_sha256']}".encode()).hexdigest()
    publish_receipt(attempt / "inference-start.json", {"fingerprint": run_fingerprint, "prior_review": visual})
    print(f"PAIR01_INFERENCE_STARTED {attempt} {run_fingerprint}", flush=True)
    for folder in ("cache", "tmp"):
        (attempt / folder).mkdir()
    env = dict(env, HF_HOME=str(attempt / "cache"), HF_HUB_CACHE=str(attempt / "cache/hub"),
               TRANSFORMERS_CACHE=str(attempt / "cache/hub"), XDG_CACHE_HOME=str(attempt / "cache"),
               TMPDIR=str(attempt / "tmp"), EGOX_AUDIT_PATH=str(attempt / "runtime_audit.json"))
    with (attempt / "inference.log").open("x") as log:
        completed = subprocess.run(["timeout", "--signal=TERM", "--kill-after=10s", "3600s", str(PYTHON), "-B",
                                    str(ROOT / "scripts/instrumented_infer.py"), "--meta_data_file", "./meta.json",
                                    "--model_path", str(WEIGHTS / "Wan2.1-I2V-14B-480P-Diffusers"), "--lora_path",
                                    str(WEIGHTS / "EgoX/pytorch_lora_weights.safetensors"), "--lora_rank", "256",
                                    "--out", str(attempt), "--seed", str(SEED), "--use_GGA", "--cos_sim_scaling_factor",
                                    "3.0", "--in_the_wild", "--idx", "-1"], cwd=work, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    with (attempt / "inference-exit-code.txt").open("x") as code:
        code.write(f"{completed.returncode}\n")
    completed.check_returncode()
    execute([PYTHON, "-I", "-B", BUNDLE / "verify_ab_008.py", "--arm", "A", "--out", attempt,
             "--fingerprint", run_fingerprint, "--ffprobe", BINS / "ffprobe", "--ffmpeg", BINS / "ffmpeg", "--complete"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify-bundle", "preflight", "render", "infer"))
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    if args.mode == "infer" and (args.attempt is None or args.visual_review is None):
        parser.error("infer requires --attempt and --visual-review from the new rendered prior")
    plan = verify_bundle()
    if args.mode == "verify-bundle":
        print("PAIR01_CANDIDATE_BUNDLE_VERIFIED")
        return
    env = offline_environment()
    with machine_lock():
        fingerprint = preflight(plan, env)
        print(f"PAIR01_CANDIDATE_PREFLIGHT_OK {fingerprint}", flush=True)
        if args.mode == "render":
            RUNS.mkdir(exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=RUNS))
            print(f"PAIR01_PRIOR_STARTED {attempt}", flush=True)
            render(attempt, plan, env, fingerprint)
            print(f"PAIR01_PRIOR_REVIEW_REQUIRED {attempt}", flush=True)
        elif args.mode == "infer":
            attempt = args.attempt.resolve(strict=True)
            if attempt.parent != RUNS.resolve() or not attempt.name.startswith("attempt-"):
                raise ValueError("Expected a new Pair01 attempt under the isolated results directory.")
            infer(attempt, plan, env, fingerprint, args.visual_review.resolve(strict=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"PAIR01_CANDIDATE_STOPPED {error}", file=sys.stderr)
        raise SystemExit(1) from error
