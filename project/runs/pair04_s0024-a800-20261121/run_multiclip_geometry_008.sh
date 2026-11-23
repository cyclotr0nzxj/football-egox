#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/root/autodl-tmp/football_egox/pair04-20261121
root="$base/multiclip-008"
name=${1:?clip identifier required}
digest=${2:?EXO checksum required}
[[ $name =~ ^pair(01|02|05)_s[0-9]{4}$ && $digest =~ ^[a-f0-9]{64}$ ]]
clip="$root/$name"
input="$clip/example/self_capture/videos/$name/exo.mp4"
export PATH="$base/renderer-env/bin:$PATH" PYTHONPATH="$base/Renderer-sparse"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME="$base/caches/hf" HF_HUB_CACHE="$base/caches/hf/hub" TORCH_HOME="$base/caches/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export http_proxy=http://127.0.0.1:0 https_proxy=http://127.0.0.1:0 all_proxy=http://127.0.0.1:0
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" no_proxy='' NO_PROXY=''
python="$base/renderer-env/bin/python"

test ! -e "$clip/geometry"
test "$input" = "$(find "$clip/example" -type f -print)"
"$python" -I -B "$root/validate_multiclip_exo.py" "$input" "$digest" > "$clip/exo-check.json"
gpu=$(nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)
if [[ ! $gpu =~ ^NVIDIA\ A800\ 80GB\ PCIe,[[:space:]]*81920,[[:space:]]*([0-9]+)$ ]] || (( BASH_REMATCH[1] > 47500 )); then
    printf 'Insufficient GPU space for ViPE. Observed %s\n' "$gpu" >&2
    exit 1
fi
mkdir -- "$clip/geometry"
printf 'VIPE_MULTICLIP_STARTED %s %s\n' "$name" "$(date -Is)"
timeout --signal=TERM --kill-after=10s 900s vipe infer "$input" \
    --output "$clip/geometry/vipe_results" --start_frame 0 --end_frame 48 \
    --assume_fixed_camera_pose --pipeline lyra > "$clip/geometry/vipe.log" 2>&1
"$python" -I -B "$base/check-vipe-artifacts-008.py" "$clip/geometry/vipe_results/$name" > "$clip/geometry/artifact-check.json"
printf 'VIPE_MULTICLIP_VERIFIED %s %s\n' "$name" "$(date -Is)"
