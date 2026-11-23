#!/usr/bin/env bash
# Bounded retry after both first windows estimated different intrinsics.
# The old EXO estimate is a fixed reference, not measured ground truth.
set -uo pipefail
base=/root/autodl-tmp/football_egox/pair04-20261121
run="$base/pair02-pair04-100f-geometry-prep"
py="$base/renderer-env/bin/python"
vipe="$base/renderer-env/bin/vipe"
[[ -d "$run" && ! -e "$run/fixed-k-attempt" && ! -e "$run/continue_fixed_008.py" ]] || { echo 'FIXED_K_BLOCKED existing or missing run' >&2; exit 2; }
[[ -x "$py" && -x "$vipe" ]] || { echo 'FIXED_K_BLOCKED renderer environment missing' >&2; exit 2; }
archive="$base/egox-shared-008/pair05-vae-66fc463f.FlNTz4tU/exo-geometry-ready.zip"
"$py" -I - "$run" "$archive" <<'PY_CHECK' || { echo 'FIXED_K_BLOCKED input integrity' >&2; exit 2; }
import hashlib,json,sys,zipfile
from pathlib import Path
run,archive=map(Path,sys.argv[1:])
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
if digest(archive)!='477153e0d753ba961e3324197db198e78d32165dfdd07c0e16050184aca64743':raise ValueError('Input archive hash differs')
with zipfile.ZipFile(archive) as z:
 checks=json.loads(z.read('SHA256.json'))
 if set(z.namelist())!=set(checks)|{'SHA256.json'}:raise ValueError('Unexpected archive members')
for name,expected in checks.items():
 rel=Path(name)
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
 path=run/rel
 if not path.is_file() or path.is_symlink() or digest(path)!=expected:raise ValueError('Input differs: '+name)
reuse=json.loads((run/'legacy-reuse.json').read_text())
expected={'legacy/Pair02/depth/exo.zip','legacy/Pair04/depth/exo.zip','legacy/Pair02/prior.mp4','legacy/Pair04/prior.mp4'}
if set(reuse)!=expected:raise ValueError('Unexpected legacy assets')
for name,row in reuse.items():
 path=run/name
 if not path.is_file() or path.is_symlink() or digest(path)!=row['sha256']:raise ValueError('Legacy asset differs: '+name)
print('FIXED_K_INPUTS_VERIFIED',flush=True)
PY_CHECK
gpu=$(nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits) || exit 2
[[ $gpu =~ ^NVIDIA\ A800\ 80GB\ PCIe,[[:space:]]*81920,[[:space:]]*([0-9]+)$ ]] && (( BASH_REMATCH[1] <= 500 )) || { echo "FIXED_K_BLOCKED unexpected or occupied GPU: $gpu" >&2; exit 2; }
free_kib=$(df -Pk -- "$base" | awk 'NR == 2 {print $4}')
[[ $free_kib =~ ^[0-9]+$ ]] && (( free_kib >= 5242880 )) || { echo 'FIXED_K_BLOCKED fewer than 5 GiB free' >&2; exit 2; }
exec 9>>"$base/egox-shared-008/.egox-inference.lock"
flock -n 9 || { echo 'FIXED_K_BLOCKED GPU lock' >&2; exit 2; }
"$base/egox-shared-008/egox-env/bin/python" -I -B "$run/check_runtime_008.py" \
 "$base/egox-shared-008" "$base/egox-shared-008/inference-four" \
 /root/autodl-fs/football_egox/models "$base/renderer-env/bin/ffmpeg" "$base/renderer-env/bin/ffprobe" || { echo 'FIXED_K_BLOCKED runtime preflight' >&2; exit 2; }
mkdir "$run/fixed-k-attempt" || exit 2
"$py" - "$run" <<'PY' || exit 2
from pathlib import Path
import sys
r=Path(sys.argv[1]);src=r/'continue_008.py';dst=r/'continue_fixed_008.py'
s=src.read_text()
replacements = {
 "geom=win/'geometry'/name": "geom=win/'geometry-fixed'/name",
 "win/'overlap.json'": "win/'overlap-fixed.json'",
}
if dst.exists(): raise ValueError('Existing fixed copy')
for old, new in replacements.items():
 if s.count(old)!=1: raise ValueError('Unexpected continuation script: '+old)
 s=s.replace(old,new)
dst.write_text(s)
PY
[[ -f "$run/continue_fixed_008.py" ]] || exit 2
export PATH="$base/renderer-env/bin:$PATH" PYTHONPATH="$base/Renderer-sparse"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME="$base/caches/hf" HF_HUB_CACHE="$base/caches/hf/hub" TORCH_HOME="$base/caches/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export http_proxy=http://127.0.0.1:0 https_proxy=http://127.0.0.1:0 all_proxy=http://127.0.0.1:0
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" no_proxy='' NO_PROXY=''
process() {
 local pair=$1 name=$2 old=$3 old_start=$4 new_start=$5
 local win="$run/windows/$name" out="$run/windows/$name/geometry-fixed" k
 [[ ! -e "$out" && ! -e "$win/overlap-fixed.json" ]] || { echo "FIXED_K_BLOCKED existing result $name" >&2; return 1; }
 k=$("$py" - "$run/legacy/$pair/intrinsics/exo.npz" <<'PY'
import json,numpy as np,sys
v=np.load(sys.argv[1])['data'][0]
print(json.dumps([[float(v[0]),0,float(v[2])],[0,float(v[1]),float(v[3])],[0,0,1]],separators=(',',':')))
PY
) || return 1
 echo "FIXED_K_VIPE_START $name $(date -Is)"
 if ! timeout --signal=TERM --kill-after=10s 900s "$vipe" infer "$win/exo.mp4" --output "$out" --start_frame 0 --end_frame 48 --assume_fixed_camera_pose --pipeline lyra --use_exo_intrinsic_gt "$k" >"$win/vipe-fixed.log" 2>&1; then echo "FIXED_K_VIPE_FAILED $name" >&2; return 1; fi
 if ! "$py" -B "$run/check_overlap_geometry.py" "$old" "$out/$name" --old-start "$old_start" --new-start "$new_start" --output "$win/overlap-fixed.json"; then echo "FIXED_K_OVERLAP_FAILED $name" >&2; return 1; fi
 if ! "$py" - "$win/overlap-fixed.json" "$win/overlap-frame-guard.json" <<'PY_GUARD'
import json,math,sys
from pathlib import Path
result=json.loads(Path(sys.argv[1]).read_text())
rows=result['heldout_frames']
if not rows:raise ValueError('Missing held-out frame evidence')
metrics=[(float(r['median_relative_error']),float(r['p90_relative_error'])) for r in rows]
passed=all(math.isfinite(a) and math.isfinite(b) and a<=0.20 and b<=0.40 for a,b in metrics)
report={'passed':passed,'heldout_frames':len(metrics),'max_frame_median_relative_error':max(a for a,b in metrics),'max_frame_p90_relative_error':max(b for a,b in metrics),'gate':'Every held-out frame: median <= 20%, pixel p90 <= 40%; provisional consistency screen; spatial and visual seam checks remain.'}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report),flush=True)
raise SystemExit(0 if passed else 2)
PY_GUARD
 then echo "FIXED_K_FRAME_GUARD_FAILED $name" >&2; return 1; fi
 echo "FIXED_K_OVERLAP_PASS $name $(date -Is)"
}
fail=0
if process Pair02 pair02_s0028 "$run/legacy/Pair02" 0 28; then
 process Pair02 pair02_s0052 "$run/windows/pair02_s0028/geometry-fixed/pair02_s0028" 28 52 || fail=1
else fail=1; fi
if process Pair04 pair04_s0052 "$run/legacy/Pair04" 24 52; then
 process Pair04 pair04_s0076 "$run/windows/pair04_s0052/geometry-fixed/pair04_s0052" 52 76 || fail=1
else fail=1; fi
flock -u 9
for pair in Pair02 Pair04; do
 if [[ $pair == Pair02 ]]; then a=pair02_s0028; b=pair02_s0052; else a=pair04_s0052; b=pair04_s0076; fi
 if [[ -f "$run/windows/$a/overlap-fixed.json" && -f "$run/windows/$b/overlap-fixed.json" && -f "$run/windows/$a/overlap-frame-guard.json" && -f "$run/windows/$b/overlap-frame-guard.json" ]] && "$py" -c 'import json,sys;assert all(json.load(open(x))["passed_provisional_gate"] is True for x in sys.argv[1:3]);assert all(json.load(open(x))["passed"] is True for x in sys.argv[3:])' "$run/windows/$a/overlap-fixed.json" "$run/windows/$b/overlap-fixed.json" "$run/windows/$a/overlap-frame-guard.json" "$run/windows/$b/overlap-frame-guard.json"; then
  if ! "$py" -B "$run/continue_fixed_008.py" all --run "$run" --pair "$pair"; then echo "FIXED_K_PAIR_INCOMPLETE $pair" >&2; fail=1; fi
 else echo "FIXED_K_PAIR_SKIPPED $pair" >&2; fail=1; fi
done
archive="$base/pair02-pair04-fixed-k-evidence-$(date +%Y%m%d-%H%M%S).zip"
"$py" -B "$run/collect_008.py" --run "$run" --output "$archive" || exit 2
echo "FIXED_K_EVIDENCE_ARCHIVE $archive"
(( fail == 0 )) || exit 2
echo 'FIXED_K_ALL_WINDOWS_TECHNICALLY_VERIFIED'
