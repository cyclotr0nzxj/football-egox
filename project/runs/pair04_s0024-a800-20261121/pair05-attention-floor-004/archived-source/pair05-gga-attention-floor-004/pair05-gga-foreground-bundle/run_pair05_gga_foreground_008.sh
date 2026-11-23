#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:-preflight}
if [[ $# -gt 1 || ( $mode != preflight && $mode != run ) ]]; then
    printf 'Usage: %s [preflight|run]\n' "$0" >&2
    exit 2
fi
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base=/root/autodl-tmp/football_egox/pair04-20261121
run_root="$base/egox-shared-008"
stock="$run_root/inference-update"
out_root="$run_root/pair05-gga-foreground-results-004-floor25"
weights=/root/autodl-fs/football_egox/models
python="$run_root/egox-env/bin/python"
ffprobe="$base/renderer-env/bin/ffprobe"
ffmpeg="$base/renderer-env/bin/ffmpeg"
audit="$run_root/scripts/instrumented_infer.py"
seed=846514
run_timeout=3600
plan_relative=plan/pair05-gga-routes.npz

fail() {
    printf 'PAIR05_FOREGROUND_FAILED %s\n' "$*" >&2
    exit 1
}

check_idle_a800() {
    local state processes
    state=$(nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)
    if [[ ! $state =~ ^NVIDIA\ A800\ 80GB\ PCIe,[[:space:]]*81920,[[:space:]]*([0-9]+)$ ]] ||
        (( BASH_REMATCH[1] > 500 )); then
        fail "Expected the idle original A800 80GB PCIe. Observed $state."
    fi
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] || fail 'Another CUDA computation is active.'
}

assemble() {
    timeout --signal=TERM --kill-after=5s 120s "$python" -B "$bundle/stage_foreground_008.py" \
        --official "$stock" --media-root "$stock" --stage "$1"
}

verify_result() {
    local attempt=$1
    shift
    timeout --signal=TERM --kill-after=5s 120s "$python" -B "$bundle/verify_foreground_008.py" \
        --out "$attempt" --fingerprint "$fingerprint" --ffprobe "$ffprobe" --ffmpeg "$ffmpeg" "$@"
}

[[ $(hostname) == autodl-container-d293479255-5411ceb1 ]] || fail 'Expected the authorized AutoDL 004 container.'
[[ -x $python && -x $ffprobe && -x $ffmpeg ]] || fail 'The existing EgoX or renderer environment is missing.'
[[ -f $run_root/.egox-inference.lock ]] || fail 'The existing EgoX inference lock file is missing.'
exec 9<"$run_root/.egox-inference.lock"
flock -n 9 || fail 'Another EgoX launcher holds the inference lock.'
findmnt --mountpoint "$(readlink -f /root/autodl-fs)" >/dev/null || fail 'Expected the original shared model mount.'
check_idle_a800

unset PYTHONPATH PYTHONHOME
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled OMP_NUM_THREADS=8 PYTHONHASHSEED="$seed" IMAGEIO_FFMPEG_EXE="$ffmpeg"

(cd -- "$bundle" && sha256sum -c --strict SHA256SUMS)
inputs=$(timeout --signal=TERM --kill-after=5s 120s "$python" -I -B "$bundle/check_inputs.py" \
    --media-root "$stock" --ffprobe "$ffprobe")
printf '%s\n' "$inputs"
runtime=$(timeout --signal=TERM --kill-after=5s 180s "$python" -I -B "$bundle/check_runtime_008.py" \
    "$run_root" "$stock" "$weights" "$ffmpeg" "$ffprobe")
printf '%s\n' "$runtime"
runtime_hash=$(awk '/^EGOX_RUNTIME_FINGERPRINT / {print $2}' <<< "$runtime")
[[ $runtime_hash =~ ^[a-f0-9]{64}$ ]] || fail 'Missing verified stock runtime fingerprint.'
projection=$(timeout --signal=TERM --kill-after=5s 120s "$python" -B "$bundle/prepare_infer.py" \
    --official-infer "$stock/infer.py" --meta "$bundle/meta.json" --check-only)
printf '%s\n' "$projection"

driver=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader,nounits)
bundle_hash=$(sha256sum "$bundle/SHA256SUMS" | cut -d' ' -f1)
fingerprint=$(printf '%s\n' "$bundle_hash" "$runtime_hash" "$driver" | sha256sum | cut -d' ' -f1)
preflight_dir=$(mktemp -d "$run_root/.pair05-foreground-preflight-XXXXXXXX")
trap 'rm -rf -- "$preflight_dir"' EXIT
stage="$preflight_dir/work"
assemble "$stage"
cuda=$(cd -- "$stage" && timeout --signal=TERM --kill-after=5s 120s "$python" -I -B "$bundle/check_foreground_cuda.py" \
    --stage "$stage" --plan "$stage/$plan_relative" --meta "$stage/meta.json")
printf '%s\n' "$cuda"
[[ ${cuda##*$'\n'} == 'PAIR05_GGA_FLOOR_CUDA_OK '* ]] || fail 'Missing floor CUDA preflight evidence.'
preflight_marker="PAIR05_FOREGROUND_PREFLIGHT_OK $fingerprint seed=$seed timeout_s=$run_timeout"
printf '%s\n' "$preflight_marker"
[[ $mode != preflight ]] || exit 0

[[ ! -L $out_root ]] || fail "Result root is a symbolic link: $out_root."
mkdir -p -- "$out_root"
shopt -s nullglob
attempts=("$out_root"/attempt-*)
if (( ${#attempts[@]} > 1 )); then
    fail "Expected at most one attempt in $out_root."
fi
if (( ${#attempts[@]} == 1 )); then
    attempt=${attempts[0]}
    [[ -d $attempt && ! -L $attempt ]] || fail "Unexpected attempt path: $attempt."
    if [[ -f $attempt/verified-result.json && ! -L $attempt/verified-result.json ]]; then
        verify_result "$attempt"
    elif [[ -f $attempt/inference-exit-code.txt && $(<"$attempt/inference-exit-code.txt") == 0 ]]; then
        verify_result "$attempt" --complete
    else
        fail "Existing attempt did not finish successfully. Evidence retained at $attempt."
    fi
    printf 'PAIR05_FOREGROUND_COMPLETE %s %s\n' "$fingerprint" "$attempt"
    exit 0
fi

check_idle_a800
disk_free=$(df -Pk "$out_root" | awk 'END {print $4}')
[[ $disk_free =~ ^[0-9]+$ ]] && (( disk_free >= 2097152 )) || fail 'Less than 2 GiB of local scratch is available.'
timeout --signal=TERM --kill-after=5s 120s "$python" -I -B "$bundle/check_inputs.py" \
    --media-root "$preflight_dir/work" --ffprobe "$ffprobe"
attempt=$(mktemp -d "$out_root/attempt-XXXXXXXX")
stage="$attempt/work"
mkdir -- "$attempt/cache" "$attempt/tmp"
mv -- "$preflight_dir/work" "$stage"
printf '%s\n' "$fingerprint" >"$attempt/fingerprint.txt"
printf '%s\n' "$inputs" "$runtime" "$projection" "$cuda" "$preflight_marker" >"$attempt/preflight.log"
cd -- "$stage"
{
    sha256sum infer.py aria_gga.py pair05_gga_routes.py meta.json validated-inputs.json "$plan_relative"
    find core -name '*.py' -type f -print | LC_ALL=C sort | xargs sha256sum
} >"$attempt/executed-code-and-meta.sha256"

export HF_HOME="$attempt/cache" HF_HUB_CACHE="$attempt/cache/hub"
export TRANSFORMERS_CACHE="$attempt/cache/hub" XDG_CACHE_HOME="$attempt/cache"
export TMPDIR="$attempt/tmp" EGOX_AUDIT_PATH="$attempt/runtime_audit.json"
command=("$python" -B "$audit" --meta_data_file ./meta.json
    --model_path "$weights/Wan2.1-I2V-14B-480P-Diffusers"
    --lora_path "$weights/EgoX/pytorch_lora_weights.safetensors"
    --lora_rank 256 --out "$attempt" --seed "$seed" --use_GGA
    --cos_sim_scaling_factor 3.0 --in_the_wild --idx -1 --foreground_gga_plan "$stage/$plan_relative")
{
    printf 'started_at=%s\nseed=%s\ndriver=%s\n' "$(date -Is)" "$seed" "$driver"
    printf 'command='; printf ' %q' "${command[@]}"; printf '\n'
} >"$attempt/launch.log"
printf 'PAIR05_FOREGROUND_STARTED %s %s\n' "$(date -Is)" "$attempt"
exit_code=0
timeout --signal=TERM --kill-after=10s "${run_timeout}s" "${command[@]}" >"$attempt/inference.log" 2>&1 || exit_code=$?
printf '%s\n' "$exit_code" >"$attempt/inference-exit-code.txt"
printf 'ended_at=%s\nexit_code=%s\n' "$(date -Is)" "$exit_code" >>"$attempt/launch.log"
[[ $exit_code == 0 ]] || fail "Inference exited with $exit_code. Evidence retained at $attempt."
verify_result "$attempt" --complete
printf 'PAIR05_FOREGROUND_COMPLETE %s %s\n' "$fingerprint" "$attempt"
