#!/usr/bin/env bash
set -euo pipefail
umask 077

# Preserve the verified four-clip launcher settings for this three-clip batch.

base=/root/autodl-tmp/football_egox/pair04-20261121
run_root="$base/egox-shared-008"
weights=/root/autodl-fs/football_egox/models
python="$run_root/egox-env/bin/python"
ffprobe="$base/renderer-env/bin/ffprobe"
stage_name=${1:-inference-update}
if [[ $# -gt 1 || $stage_name != inference-update ]]; then
    printf 'Usage: %s [inference-update]\n' "$0" >&2
    exit 1
fi
takes=(pair04_s0024 pair05_s0024 pair01_s0288)
budget=8100
out="$run_root/update-egox-output"

exec 9>"$run_root/.egox-inference.lock"
if ! flock -n 9; then
    printf 'Another EgoX launcher is active in this run directory.\n' >&2
    exit 1
fi

cd "$run_root/$stage_name"
test -L infer.py
test -L core
test -d core
test -f "$run_root/scripts/instrumented_infer.py"
test "$(sha256sum infer.py | cut -d' ' -f1)" = 99199fac22858821d356f0ec3b8577495290dd579a8d80368de1fca68d559bed
if [[ -e $out || -L $out ]]; then
    printf 'Output directory already exists: %s\n' "$out" >&2
    exit 1
fi
timeout --signal=TERM --kill-after=5s 90s "$python" -I -B - "${takes[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

takes = sys.argv[1:]
meta_path = Path('meta.json')
receipt_path = Path('validated-inputs.json')

def sha256(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()

manifest = Path('SHA256SUMS').read_text().splitlines()
expected_names = {'meta.json', 'validated-inputs.json'}
matches = [re.fullmatch(r'[0-9a-f]{64} [ *](meta\.json|validated-inputs\.json)', line)
           for line in manifest]
if len(matches) != 2 or not all(matches) or {match[1] for match in matches} != expected_names:
    raise SystemExit('SHA256SUMS must contain exactly the certified metadata and input receipt.')
subprocess.run(['sha256sum', '--check', '--strict', 'SHA256SUMS'], check=True)

receipts = json.loads(receipt_path.read_text())
if [receipt['name'] for receipt in receipts] != takes:
    raise SystemExit('Validated input receipt does not match the three-take order.')
metadata = json.loads(meta_path.read_text())
if set(metadata) != {'test_datasets'}:
    raise SystemExit('Expected only the official test_datasets metadata key.')
entries = metadata['test_datasets']
if len(entries) != len(takes):
    raise SystemExit('Metadata must contain the exact expected number of takes.')
keys = {'exo_path', 'ego_prior_path', 'prompt', 'camera_intrinsics', 'camera_extrinsics',
        'ego_intrinsics', 'ego_extrinsics'}
for entry, take, receipt in zip(entries, takes, receipts, strict=True):
    exo = f'./example/self_capture/videos/{take}/exo.mp4'
    prior = f'./example/self_capture/ego_priors/{take}/ego_Prior.mp4'
    if (set(entry) != keys or entry['exo_path'] != exo or entry['ego_prior_path'] != prior
            or len(entry['ego_extrinsics']) != 49):
        raise SystemExit(f'EXO-only metadata does not match the fixed inputs for {take}.')
    if receipt['frames'] != 49 or len(receipt['depth_sha256']) != 49:
        raise SystemExit(f'Incomplete validated input receipt for {take}.')
    depths = sorted(Path(f'example/self_capture/depth_maps/{take}').glob('*.npy'))
    indices = [''.join(char for char in path.stem if char.isdigit()) for path in depths]
    if len(depths) != 49 or any(not value or int(value) != i for i, value in enumerate(indices)):
        raise SystemExit(f'Expected exactly 49 indexed input depths for {take}.')
    frozen = [(Path(exo), receipt['exo_sha256']), (Path(prior), receipt['prior_sha256'])]
    frozen.extend(zip(depths, receipt['depth_sha256'], strict=True))
    for path, digest in frozen:
        if sha256(path) != digest:
            raise SystemExit(f'Input checksum mismatch: {path}.')
print('EGOX_INPUTS_VERIFIED', *takes, flush=True)
PY

findmnt --mountpoint "$(readlink -f /root/autodl-fs)" >/dev/null
gpu_state=$(nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)
if [[ ! $gpu_state =~ ^NVIDIA\ A800\ 80GB\ PCIe,[[:space:]]*81920,[[:space:]]*([0-9]+)$ ]] || (( BASH_REMATCH[1] > 500 )); then
    printf 'EgoX requires the authorized idle A800. Observed %s\n' "$gpu_state" >&2
    exit 1
fi
grep -Fxq ALL_MODELS_VERIFIED "$run_root/download-shared.log"
"$run_root/egox-env/bin/python" -I -B - "$run_root" "$weights" <<'PY'
import importlib.metadata
import json
import pathlib
import sys

import torch
from safetensors import safe_open

root, models = map(pathlib.Path, sys.argv[1:])
receipts = json.loads((root / 'records/model_verification.json').read_text())
plan = json.loads((root / 'records/download_plan.json').read_text())
expected = {}
for model in plan['models']:
    folder = model['repo'].split('/')[-1]
    for item in model['files']:
        expected[f'models/{folder}/{item["rfilename"]}'] = item
if len(receipts) != 35 or {row['path'] for row in receipts} != set(expected):
    raise SystemExit('The manifest of 35 verified official files is required.')
for row in receipts:
    item = expected[row['path']]
    location = models / pathlib.Path(row['path']).relative_to('models')
    digest = item.get('lfs', {}).get('sha256')
    if (not row['official_hash_verified'] or row['bytes'] != item['size'] or
            not location.is_file() or location.stat().st_size != item['size'] or
            (digest and row['sha256'] != digest)):
        raise SystemExit(f'Model verification receipt mismatch: {location.name}')

for package, version in [('diffusers', '0.34.0'), ('transformers', '4.49.0'),
                         ('accelerate', '1.5.2'), ('peft', '0.17.1')]:
    if importlib.metadata.version(package) != version:
        raise SystemExit(f'Wrong {package} version.')
if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4' or not torch.cuda.is_available():
    raise SystemExit('EgoX requires the previously tested CUDA-enabled PyTorch 2.5.1 environment.')
gpu = torch.cuda.get_device_properties(0)
if 'A800' not in gpu.name or gpu.total_memory < 79 * 2**30:
    raise SystemExit(f'Wrong GPU: {gpu.name}')
matrix = torch.eye(8, device='cuda')
if not torch.equal(matrix @ matrix, matrix):
    raise SystemExit('EgoX CUDA smoke test failed.')
torch.cuda.synchronize()
cgroup = pathlib.Path('/sys/fs/cgroup')
if (cgroup / 'memory.max').is_file():
    limit = cgroup / 'memory.max'
    usage = cgroup / 'memory.current'
    inactive_key = 'inactive_file'
else:
    cgroup = cgroup / 'memory'
    limit = cgroup / 'memory.limit_in_bytes'
    usage = cgroup / 'memory.usage_in_bytes'
    inactive_key = 'total_inactive_file'
limit_bytes = int(limit.read_text())
usage_bytes = int(usage.read_text())
memory_stats = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
reclaimable = int(memory_stats[inactive_key])
if limit_bytes - usage_bytes + reclaimable < 110 * 2**30:
    raise SystemExit('Host RAM available to the inference container is below 110 GiB.')
print('EGOX_CGROUP_RAM_BYTES', limit_bytes, usage_bytes, reclaimable)
adapter = models / 'EgoX/pytorch_lora_weights.safetensors'
with safe_open(adapter, framework='pt', device='cpu') as tensors:
    if not tensors.keys():
        raise SystemExit('Could not mmap the official adapter on shared storage.')
    row = tensors.get_slice(tensors.keys()[0])[0:1]
    if not bool(torch.isfinite(row.float()).all()):
        raise SystemExit('Cannot read a finite tensor slice from shared storage.')
print('EGOX_GPU_ENV_AND_MODEL_RECEIPTS_OK', flush=True)
PY

mkdir -- "$out" "$out/tmp" "$out/cache"
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=8
export EGOX_AUDIT_PATH="$out/runtime_audit.json" HF_HOME="$out/cache" TMPDIR="$out/tmp"
export IMAGEIO_FFMPEG_EXE="$base/renderer-env/bin/ffmpeg"
printf 'EGOX_STARTED %s\n' "$(date -Is)"
exit_code=0
timeout --signal=TERM --kill-after=10s "${budget}s" "$python" -B "$run_root/scripts/instrumented_infer.py" \
    --meta_data_file ./meta.json \
    --model_path "$weights/Wan2.1-I2V-14B-480P-Diffusers" \
    --lora_path "$weights/EgoX/pytorch_lora_weights.safetensors" \
    --lora_rank 256 --out "$out" --seed 846514 --use_GGA \
    --cos_sim_scaling_factor 3.0 --in_the_wild --idx -1 >"$out/inference.log" 2>&1 || exit_code=$?
printf '%s\n' "$exit_code" >"$out/inference-exit-code.txt"

"$python" -I -B - "$out" "$ffprobe" "$exit_code" "${takes[@]}" <<'PY'
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

out = Path(sys.argv[1])
ffprobe, exit_code, *takes = sys.argv[2:]
exit_code = int(exit_code)
results = []

def write_json(path, doc):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(doc, indent=2) + '\n')
    temporary.replace(path)

for take in takes:
    video = out / f'{take}.mp4'
    row = {'take': take, 'file': video.name}
    try:
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_streams',
                                 '-show_entries', 'stream=width,height,avg_frame_rate,time_base:frame=pts,width,height',
                                 '-of', 'json', str(video)], capture_output=True, text=True, timeout=45, check=True)
        probe = json.loads(result.stdout)
        stream, = probe['streams']
        frames = probe['frames']
        if result.stderr or (
            stream['width'], stream['height'], Fraction(stream['avg_frame_rate']), len(frames)
        ) != (1232, 448, 30, 49):
            raise ValueError('Output must decode without errors to 49 frames at 1232×448 and 30 fps.')
        if not all(
            (f['width'], f['height'], int(f['pts']) * Fraction(stream['time_base'])) == (1232, 448, Fraction(i, 30))
            for i, f in enumerate(frames)
        ):
            raise ValueError('Output frame dimensions or presentation timestamps differ.')
        row.update(status='output_verified', sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                   frames=49, width=1232, height=448, fps=30)
        (out / f'{take}-probe.json').write_text(result.stdout)
        print('EGOX_RESULT_VERIFIED', take, '49 frames', '1232x448', '30fps', row['sha256'], flush=True)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        row.update(status='output_unverified', error=str(error))
    write_json(out / f'{take}-result.json', row)
    results.append(row)

def valid_adapter(events):
    return [event.get('operation') for event in events] == ['load_lora_weights', 'fuse_lora'] and all(
        event.get('active_adapters') and event.get('peft_config')
        and all(config.get('r') == 256 for config in event['peft_config'].values())
        for event in events
    )

audit_status = 'unverified'
try:
    audit = json.loads((out / 'runtime_audit.json').read_text())
    if valid_adapter(audit['events']):
        audit_status = 'runtime_audit_verified'
except (OSError, ValueError, KeyError, TypeError, AttributeError):
    pass
if audit_status == 'unverified':
    try:
        prefix = 'ADAPTER_AUDIT '
        with (out / 'inference.log').open(errors='replace') as stream:
            events = [json.loads(line[len(prefix):]) for line in stream if line.startswith(prefix)]
        if valid_adapter(events):
            audit_status = 'inference_log_verified'
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass

failed = [row['take'] for row in results if row['status'] != 'output_verified']
complete = not exit_code and not failed and audit_status == 'runtime_audit_verified'
write_json(out / 'batch-result.json', {
    'status': 'complete' if complete else 'incomplete',
    'inference_exit_code': exit_code,
    'adapter_evidence': audit_status,
    'takes': results,
})
if not complete:
    raise SystemExit(
        f'Incomplete generation (exit {exit_code}, adapter {audit_status}); '
        f'clips without verified output: {failed}.'
    )
PY
