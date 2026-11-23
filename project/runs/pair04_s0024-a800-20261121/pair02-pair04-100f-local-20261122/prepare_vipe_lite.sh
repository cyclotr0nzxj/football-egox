#!/usr/bin/env bash
# Explicitly authorized run only: EXO ViPE geometry and overlap checks; no EgoX.
set -euo pipefail
umask 077
archive=${1:?Pass the uploaded exo-geometry-prep.zip path}
base=/root/autodl-tmp/football_egox/pair04-20261121
python="$base/renderer-env/bin/python"
vipe="$base/renderer-env/bin/vipe"
expected=477153e0d753ba961e3324197db198e78d32165dfdd07c0e16050184aca64743
[[ -f "$archive" && ! -L "$archive" && -x "$python" && -x "$vipe" ]] || { echo 'PREP_BLOCKED missing archive or renderer environment' >&2; exit 2; }
[[ $(sha256sum "$archive" | cut -d' ' -f1) == "$expected" ]] || { echo 'PREP_BLOCKED archive checksum' >&2; exit 2; }
gpu=$(nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)
[[ $gpu =~ ^NVIDIA\ A800\ 80GB\ PCIe,[[:space:]]*81920,[[:space:]]*([0-9]+)$ ]] && (( BASH_REMATCH[1] <= 500 )) || { echo "PREP_BLOCKED unexpected or occupied GPU: $gpu" >&2; exit 2; }
free_kib=$(df -Pk -- "$base" | awk 'NR == 2 {print $4}')
[[ $free_kib =~ ^[0-9]+$ ]] && (( free_kib >= 5242880 )) || { echo 'PREP_BLOCKED fewer than 5 GiB free' >&2; exit 2; }
"$python" -I -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(2,device="cuda"); assert (x+x).sum().item()==4' || { echo 'PREP_BLOCKED CUDA execution failed' >&2; exit 2; }
exec 9>>"$base/egox-shared-008/.egox-inference.lock"
flock -n 9 || { echo 'PREP_BLOCKED another pipeline holds the GPU lock' >&2; exit 2; }
run="$base/pair02-pair04-100f-geometry-prep"
[[ ! -e "$run" ]] || { echo 'PREP_BLOCKED output already exists; inspect before retry' >&2; exit 2; }
mkdir "$run"
"$python" -I - "$archive" "$run" <<'PY'
from pathlib import Path
import hashlib,json,sys,zipfile
archive,root=map(Path,sys.argv[1:])
with zipfile.ZipFile(archive) as z:
 checks=json.loads(z.read('SHA256.json'))
 if set(z.namelist())!=set(checks)|{'SHA256.json'}: raise ValueError('Unexpected package members')
 for name,digest in checks.items():
  path=Path(name)
  if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe package path')
  contents=z.read(name)
  if hashlib.sha256(contents).hexdigest()!=digest: raise ValueError(f'Corrupt {name}')
  dest=root/path; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(contents)
PY
# Restore only the four explicitly named legacy assets after verifying their hashes.
"$python" -I - "$run" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=json.loads((root/'legacy-reuse.json').read_text())
expected={'legacy/Pair02/depth/exo.zip','legacy/Pair04/depth/exo.zip','legacy/Pair02/prior.mp4','legacy/Pair04/prior.mp4'}
if set(rows)!=expected: raise ValueError('Unexpected reused asset set')
for name,row in rows.items():
 source=Path(row['path']); h=hashlib.sha256()
 with source.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
 if h.hexdigest()!=row['sha256']: raise ValueError('Remote legacy asset hash mismatch: '+name)
 dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as src,dest.open('xb') as out: shutil.copyfileobj(src,out)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
 if h.hexdigest()!=row['sha256']: raise ValueError('Copied asset differs: '+name)
print('REMOTE_LEGACY_ASSETS_VERIFIED')
PY
# Check the complete pinned EgoX environment before spending time on ViPE.
"$base/egox-shared-008/egox-env/bin/python" -I -B "$run/check_runtime_008.py" \
 "$base/egox-shared-008" "$base/egox-shared-008/inference-four" \
 /root/autodl-fs/football_egox/models "$base/renderer-env/bin/ffmpeg" "$base/renderer-env/bin/ffprobe" \
 >"$run/runtime-preflight.log" 2>&1 || { echo 'PREP_BLOCKED EgoX runtime check; see runtime-preflight.log' >&2; exit 2; }
export PATH="$base/renderer-env/bin:$PATH" PYTHONPATH="$base/Renderer-sparse"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME="$base/caches/hf" HF_HUB_CACHE="$base/caches/hf/hub" TORCH_HOME="$base/caches/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export http_proxy=http://127.0.0.1:0 https_proxy=http://127.0.0.1:0 all_proxy=http://127.0.0.1:0
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" no_proxy='' NO_PROXY=''
process() {
 local name=$1 reference=$2 old_start=$3 new_start=$4
 local input="$run/windows/$name/exo.mp4" output="$run/windows/$name/geometry"
 echo "VIPE_START $name $(date -Is)"
 if ! timeout --signal=TERM --kill-after=10s 900s "$vipe" infer "$input" --output "$output" \
   --start_frame 0 --end_frame 48 --assume_fixed_camera_pose --pipeline lyra >"$run/windows/$name/vipe.log" 2>&1; then
  echo "PREP_BLOCKED ViPE failed $name" >&2; return 2
 fi
 local new="$output/$name"
 [[ -d "$new" ]] || { echo "PREP_BLOCKED missing artifact directory $name" >&2; return 2; }
 if ! "$python" -B "$run/check_overlap_geometry.py" "$reference" "$new" --old-start "$old_start" --new-start "$new_start" --output "$run/windows/$name/overlap.json"; then
  echo "PREP_BLOCKED overlap check $name" >&2; return 2
 fi
 echo "VIPE_OVERLAP_CHECKED $name $(date -Is)"
}
failures=0
if process pair02_s0028 "$run/legacy/Pair02" 0 28; then
 process pair02_s0052 "$run/windows/pair02_s0028/geometry/pair02_s0028" 28 52 || failures=1
else
 failures=1
fi
if process pair04_s0052 "$run/legacy/Pair04" 24 52; then
 process pair04_s0076 "$run/windows/pair04_s0052/geometry/pair04_s0052" 52 76 || failures=1
else
 failures=1
fi
(( failures == 0 )) || { echo "VIPE_PREP_INCOMPLETE: check logs and overlap receipts" >&2; exit 2; }
echo 'VIPE_PREP_COMPLETE: 3D head, camera motion, rendered prior and EgoX remain to be verified.'
