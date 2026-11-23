#!/usr/bin/env bash
# Once explicitly authorized, stream to the authenticated Pair04 A800 via ssh bash -s.
set -euo pipefail
umask 077
export LC_ALL=C

root=/root/autodl-tmp/football_egox/pair04-20261121
environment="$root/renderer-env"
archive="$root/pair04_exo_stage.zip"
stage="$root/stage"
renderer="$root/Renderer-sparse"
moge="$root/MoGe-sparse"
run="$root/first_vipe"
lock="$root/.pair04-vipe-launch.lock"
video="$stage/exo_only/example/self_capture/videos/pair04_s0024/exo.mp4"
archive_sha=89b4ff66f68bb63ca8adc4aff136a58dc5916617382d9363b574cd4f0389bbf8
renderer_pin=17cdef688885c3ac6a4c371a2d620391cc28dab4
moge_pin=07444410f1e33f402353b99d6ccd26bd31e469e8

fail() {
    printf 'PAIR04_VIPE_BLOCKED %s\n' "$*" >&2
    exit 1
}

require_start_time() {
    local now
    now=$(date +%s)
    if (( deadline - now < 180 )); then
        fail 'Fewer than 180 seconds remain before 16:50 Asia/Shanghai.'
    fi
}

deadline=$(TZ=Asia/Shanghai date -d '2026-11-21 16:50:00' +%s)
require_start_time
cd -- "$root"
mkdir -- "$lock" || fail 'Another launch is active or requires manual review.'
trap 'rmdir -- "$lock"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -e "$run" || -L "$run" ]]; then
    fail 'first_vipe already exists. Its result or failed attempt requires manual review.'
fi
if [[ ! -f "$archive" || -L "$archive" ]]; then
    fail 'The authorized EXO ZIP is required on the server.'
fi
if [[ ! -x "$environment/bin/python" || ! -x "$environment/bin/vipe" ]]; then
    fail 'The isolated Renderer environment is incomplete.'
fi
unset PYTHONPATH PYTHONHOME
export PATH="$environment/bin:$PATH"
export CONDA_PREFIX="$environment" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$root/caches/hf" HF_HUB_CACHE="$root/caches/hf/hub" TORCH_HOME="$root/caches/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# Torch Hub and gdown use HTTP clients outside Hugging Face's offline setting.
export http_proxy=http://127.0.0.1:0 https_proxy=http://127.0.0.1:0 all_proxy=http://127.0.0.1:0
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" no_proxy='' NO_PROXY=''
python="$environment/bin/python"
vipe="$environment/bin/vipe"

witness="$root/vipe-finish-extended.log"
if [[ ! -f "$witness" || -L "$witness" ]] ||
    ! grep -Fxq 'VIPE_CUDA_KERNEL_OK NVIDIA A800 80GB PCIe' "$witness" ||
    ! grep -Fxq 'VIPE_CLI_AND_CUDA_OK' "$witness"; then
    fail 'The native A800 CUDA kernel and ViPE CLI witness is incomplete.'
fi
printf 'PAIR04_CUDA_WITNESS_OK vipe-finish-extended.log\n'

check_pin() {
    local checkout="$1" revision="$2"
    [[ -d "$checkout" && ! -L "$checkout" ]] || fail "Missing checkout: $checkout"
    [[ $(git --no-optional-locks -C "$checkout" rev-parse --show-toplevel) == "$checkout" ]] ||
        fail "Checkout has the wrong Git root: $checkout"
    [[ $(git --no-optional-locks -C "$checkout" rev-parse HEAD) == "$revision" ]] ||
        fail "Checkout is not at the pinned commit: $checkout"
    local tracked_status
    tracked_status=$(git --no-optional-locks -C "$checkout" status --porcelain --untracked-files=no) ||
        fail "Cannot verify tracked files: $checkout"
    [[ -z "$tracked_status" ]] || fail "Tracked files differ from the pin: $checkout"
}
check_pin "$renderer" "$renderer_pin"
check_pin "$moge" "$moge_pin"

check_gpu() {
    local gpu
    gpu=$(nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits) ||
        fail 'Cannot verify GPU zero.'
    if [[ ! "$gpu" =~ ^NVIDIA\ A800\ 80GB\ PCIe,[[:space:]]*([0-9]+),[[:space:]]*([0-9]+)$ ]] ||
        (( BASH_REMATCH[1] < 81920 || BASH_REMATCH[2] > 500 )); then
        fail "GPU zero is outside the authorized A800 and memory limits: $gpu"
    fi
    printf 'PAIR04_GPU_OK %s\n' "$gpu"
}
check_gpu
free_kib=$(df -Pk -- "$root" | awk 'NR == 2 {print $4}')
if [[ ! "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < 1048576 )); then
    fail 'Less than 1 GiB of available space or disk usage could not be verified.'
fi

cache_gate="$root/pair04_cache_gate.py"
[[ -f "$cache_gate" && ! -L "$cache_gate" ]] || fail 'Missing the existing offline cache verifier.'
cache_result=$(timeout --signal=TERM --kill-after=2s 120s /root/miniconda3/bin/python -I -B "$cache_gate" </dev/null) ||
    fail 'Offline cache verification failed or exceeded 120 seconds.'
if ! grep -Fxq 'VERIFIED_CHECKPOINT_AND_TOKENIZER_FILES 15 OF 15' <<<"$cache_result" ||
    ! grep -Fxq 'PAIR04_OFFLINE_CACHE_OK' <<<"$cache_result"; then
    fail 'The offline cache verifier did not attest all 15 assets.'
fi
printf 'VERIFIED_CHECKPOINT_AND_TOKENIZER_FILES 15 OF 15\nPAIR04_OFFLINE_CACHE_OK\n'

"$python" -I -B - "$archive" "$archive_sha" "$stage" <<'PY' || fail 'EXO ZIP or staging verification failed.'
import hashlib
import os
import stat
import sys
import zipfile
from pathlib import Path

archive, sha256, stage = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
members = {
    'exo_only/example/self_capture/videos/pair04_s0024/exo.mp4',
    'scripts/pair04_cpu_dry_run.py',
    'scripts/pair04_vipe_preflight.py',
}
parents = {parent for name in members for parent in Path(name).parents if parent != Path('.')}
with archive.open('rb') as stream:
    if os.fstat(stream.fileno()).st_size != 727441 or hashlib.sha256(stream.read()).hexdigest() != sha256:
        raise SystemExit('Unexpected EXO ZIP size or SHA-256.')
    stream.seek(0)
    with zipfile.ZipFile(stream) as bundle:
        items = bundle.infolist()
        if len(items) != len(members) or set(bundle.namelist()) != members:
            raise SystemExit('Unexpected EXO ZIP member names.')
        if any(stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG) for item in items):
            raise SystemExit('All EXO ZIP members must be regular files.')
        contents = {name: bundle.read(name) for name in members}

if stage.is_symlink():
    raise SystemExit('Unsafe stage directory.')
if stage.exists():
    expected_paths = {stage / name for name in members} | {stage / parent for parent in parents}
    if not stage.is_dir() or set(stage.rglob('*')) != expected_paths:
        raise SystemExit('Existing stage does not contain exactly the authorized files and directories.')
    for parent in parents:
        path = stage / parent
        if path.is_symlink() or not path.is_dir():
            raise SystemExit('Unsafe directory in existing stage.')
    for name, expected in contents.items():
        path = stage / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise SystemExit(f'Existing stage differs from the authorized ZIP: {name}')
else:
    stage.mkdir()
    for name, content in sorted(contents.items()):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output:
            output.write(content)
print('PAIR04_EXO_ZIP_AND_STAGE_OK')
PY

cpu_result=$("$python" -B "$stage/scripts/pair04_cpu_dry_run.py" --input-root "$stage/exo_only") ||
    fail 'EXO software decode and PTS validation failed.'
preflight_status=0
preflight_result=$("$python" -B "$stage/scripts/pair04_vipe_preflight.py" \
    --input-root "$stage/exo_only" --renderer "$renderer" --run-dir "$run") || preflight_status=$?

"$python" -I -B - "$moge" "$environment" "$cpu_result" "$preflight_result" "$preflight_status" <<'PY' || fail 'Renderer preflight failed.'
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

moge, environment = map(Path, sys.argv[1:3])
if Path(sys.prefix).resolve() != environment.resolve():
    raise SystemExit('Python is not running from the isolated Renderer environment.')
cpu = json.loads(sys.argv[3])
preflight = json.loads(sys.argv[4])
status = int(sys.argv[5])
video = cpu['video']
if (cpu['status'], cpu['input_isolation'], video['frames'], video['fps'], video['full_decode']) != (
    'passed', 'exo_only', 49, 25, 'passed'
):
    raise SystemExit('EXO preflight did not confirm a complete 49-frame software decode at 25 fps.')
checks = preflight['checks']
expected = {
    'exo_cpu_dry_run', 'new_run_dir', 'renderer_commit', 'renderer_tracked_files_clean', 'python_3_10',
    'torch_and_torchvision', 'pytorch3d_0_7_9', 'vipe_cli_from_active_env',
    'vipe_import_from_pinned_checkout', 'nvidia_smi_visible',
}
if (set(checks) != expected or preflight['geometry_started'] is not False or
        any(checks[key] is not True for key in expected - {'pytorch3d_0_7_9'})):
    raise SystemExit(f'Native ViPE preflight checks failed: {checks}')
if checks['pytorch3d_0_7_9'] is True and status == 0:
    print('PAIR04_PYTORCH3D_VERSION_OK')
elif checks['pytorch3d_0_7_9'] is False and status == 2:
    try:
        importlib.metadata.version('pytorch3d')
    except importlib.metadata.PackageNotFoundError:
        if importlib.util.find_spec('pytorch3d') is not None:
            raise SystemExit('PyTorch3D has importable code without matching package metadata.')
    else:
        raise SystemExit('An unexpected PyTorch3D version is installed.')
    print('PAIR04_PYTORCH3D_ABSENT_RENDERING_UNVERIFIED')
else:
    raise SystemExit('Unexpected preflight status.')

distribution = importlib.metadata.distribution('moge')
installed_sources = [path for path in distribution.files or () if path.parts[0] == 'moge' and path.suffix == '.py']
if distribution.version != '2.0.0' or Path('moge/model/v2.py') not in installed_sources:
    raise SystemExit('The installed MoGe wheel must provide version 2.0.0 and its v2 module.')
for relative in installed_sources:
    installed = distribution.locate_file(relative)
    if not installed.resolve().is_relative_to(environment) or installed.read_bytes() != (moge / relative).read_bytes():
        raise SystemExit(f'Installed MoGe source differs from the pinned checkout: {relative}')
print('PAIR04_SOURCE_AND_CPU_PREFLIGHT_OK frames=49 fps=25')
PY

check_gpu
require_start_time
budget=$(( deadline - $(date +%s) - 15 ))
if (( budget > 890 )); then
    budget=890
fi
mkdir -- "$run" || fail 'Could not reserve a fresh output directory.'
printf 'started_at=%s\ntimeout_seconds=%s\nkill_after_seconds=10\n' "$(date -Is)" "$budget" > "$run/launch-status.txt"
printf 'PAIR04_VIPE_STARTED timeout_seconds=%s log=%s\n' "$budget" "$run/vipe.log"
vipe_status=0
timeout --signal=TERM --kill-after=10s "${budget}s" "$vipe" infer "$video" \
    --output "$run/vipe_results" --start_frame 0 --end_frame 48 \
    --assume_fixed_camera_pose --pipeline lyra </dev/null > "$run/vipe.log" 2>&1 || vipe_status=$?
printf 'finished_at=%s\nvipe_exit_code=%s\n' "$(date -Is)" "$vipe_status" >> "$run/launch-status.txt"
if (( vipe_status != 0 )); then
    printf 'PAIR04_VIPE_FAILED exit_code=%s log=%s\n' "$vipe_status" "$run/vipe.log" >&2
    exit "$vipe_status"
fi

artifacts="$run/vipe_results/pair04_s0024"
verify_budget=$(( deadline - $(date +%s) - 2 ))
if (( verify_budget <= 0 )); then
    fail 'ViPE exited successfully. Artifact validation remains pending because the deadline was reached.'
fi
timeout --signal=TERM --kill-after=1s "${verify_budget}s" "$python" -I -B - "$artifacts" \
    > "$run/artifact-check.json" 2> "$run/artifact-check.stderr" <<'PY' || fail "ViPE exited successfully, but artifact validation did not pass. See $run/artifact-check.stderr."
import json
import subprocess
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from vipe.utils.io import read_depth_artifacts

root = Path(sys.argv[1])
paths = {kind: root / kind / f'exo.{suffix}' for kind, suffix in (
    ('rgb', 'mp4'), ('pose', 'npz'), ('intrinsics', 'npz'), ('depth', 'zip'),
)}
for kind, path in paths.items():
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f'Missing or empty ViPE {kind} artifact: {path}')
receipt = root / 'vipe/exo_info.pkl'
if not receipt.is_file() or receipt.stat().st_size <= 0:
    raise SystemExit('Missing ViPE export completion file.')

count = 49
frames = np.arange(count)
probe = subprocess.run([
    'ffprobe', '-v', 'error', '-show_frames', '-select_streams', 'v:0', '-show_entries',
    'stream=codec_name,width,height,avg_frame_rate,time_base:frame=pts,width,height', '-of', 'json', str(paths['rgb']),
], check=True, capture_output=True, text=True)
if probe.stderr.strip():
    raise SystemExit('ffprobe reported an error decoding the ViPE RGB frames.')
rgb_result = json.loads(probe.stdout)
rgb = rgb_result['streams'][0]
shape = (rgb['height'], rgb['width'])
if (shape != (448, 784) or rgb['codec_name'] != 'h264' or len(rgb_result['frames']) != count or
        Fraction(rgb['avg_frame_rate']) != 25):
    raise SystemExit('RGB export must contain 49 decoded 784x448 H.264 frames at 25 fps.')
time_base = Fraction(rgb['time_base'])
for index, frame in enumerate(rgb_result['frames']):
    if (frame['height'], frame['width']) != shape or int(frame['pts']) * time_base != Fraction(index, 25):
        raise SystemExit(f'RGB frame {index} differs from the expected PTS or image grid.')

def indexed_values(kind, dimensions):
    with np.load(paths[kind], allow_pickle=False) as artifact:
        data = artifact['data']
        if not np.array_equal(artifact['inds'], frames) or data.shape != dimensions or not np.isfinite(data).all():
            raise SystemExit(f'Unexpected ViPE {kind} frame indices, shape, or values.')
    return data.astype(np.float64)

pose = indexed_values('pose', (count, 4, 4))
rotation = pose[:, :3, :3]
rotation_t = np.swapaxes(rotation, -2, -1)
if (not np.allclose(pose[:, 3], [0, 0, 0, 1], atol=1e-4, rtol=0) or
        not np.allclose(rotation_t @ rotation, np.eye(3), atol=1e-4, rtol=0) or
        not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)):
    raise SystemExit('ViPE C2W pose is not a valid rigid homogeneous transform.')
relative_rotation = rotation_t[0] @ rotation
relative_angle = np.degrees(np.arccos(np.clip((np.trace(relative_rotation, axis1=1, axis2=2) - 1) / 2, -1, 1)))
relative_translation = np.linalg.norm(pose[:, :3, 3] - pose[0, :3, 3], axis=1)

intrinsics = indexed_values('intrinsics', (count, 4))
if (intrinsics[:, :2] <= 0).any():
    raise SystemExit('Nonpositive estimated focal lengths.')
camera_type = root / 'intrinsics/exo_camera.txt'
if camera_type.read_text().splitlines() != [f'{index}: PINHOLE' for index in frames]:
    raise SystemExit('Expected PINHOLE camera type for all 49 indexed frames.')

with zipfile.ZipFile(paths['depth']) as archive:
    if sorted(archive.namelist()) != [f'{index:05d}.exr' for index in frames] or archive.testzip() is not None:
        raise SystemExit('Depth ZIP does not contain exactly 49 valid, indexed EXRs.')
depth_quality = []
for frame, depth in read_depth_artifacts(paths['depth']):
    if frame != len(depth_quality) or tuple(depth.shape) != shape:
        raise SystemExit('Depth frame indices or grid differ from the RGB export.')
    z = depth.detach().cpu().numpy()
    valid = z[np.isfinite(z) & (z > 0)]
    if not valid.size or valid.min() == valid.max():
        raise SystemExit(f'No nonconstant finite positive depth at frame {frame}.')
    depth_quality.append(float(valid.size / z.size))
if len(depth_quality) != count:
    raise SystemExit('ViPE depth reader did not decode all 49 EXRs.')

print(json.dumps({
    'status': 'raw_vipe_exports_verified', 'frames': count, 'fps': 25,
    'rgb_size': [shape[1], shape[0]], 'artifact_bytes': {kind: path.stat().st_size for kind, path in paths.items()},
    'c2w_pose_max_rotation_degrees_from_first_frame': float(relative_angle.max()),
    'c2w_pose_max_translation_from_first_frame_in_vipe_units': float(relative_translation.max()),
    'c2w_pose_exactly_constant': bool(np.all(pose == pose[0])),
    'intrinsics_fx_fy_cx_cy_min': intrinsics.min(axis=0).tolist(),
    'intrinsics_fx_fy_cx_cy_max': intrinsics.max(axis=0).tolist(),
    'depth_finite_positive_fraction_by_frame': depth_quality,
}, indent=2))
PY
printf 'artifact_check=passed\n' >> "$run/launch-status.txt"
printf 'PAIR04_VIPE_RAW_ARTIFACTS_OK frames=49 fps=25 report=%s\n' "$run/artifact-check.json"
