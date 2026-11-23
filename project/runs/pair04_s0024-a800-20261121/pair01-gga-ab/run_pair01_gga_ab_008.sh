#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:-preflight}
if [[ $# -gt 1 || ( $mode != preflight && $mode != run && $mode != run-a ) ]]; then
    printf 'Usage: %s [preflight|run|run-a]\n' "$0" >&2
    exit 2
fi
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base=/root/autodl-tmp/football_egox/pair04-20261121
run_root="$base/egox-shared-008"
old="$run_root/inference-update"
out_root="$run_root/pair01-gga-ab-results"
weights=/root/autodl-fs/football_egox/models
python="$run_root/egox-env/bin/python"
ffprobe="$base/renderer-env/bin/ffprobe"
ffmpeg="$base/renderer-env/bin/ffmpeg"
audit="$run_root/scripts/instrumented_infer.py"
take=pair01_s0288
seed=846514
arm_timeout=3600

fail() {
    printf 'EGOX_AB_FAILED %s\n' "$*" >&2
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

[[ $(hostname) == autodl-container-cb894da57f-45e726a1 ]] || fail 'Expected the authorized AutoDL 008 container.'
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
timeout --signal=TERM --kill-after=5s 120s "$python" -I -B "$bundle/check_inputs.py" \
    --media-root "$old" --ffprobe "$ffprobe"
runtime=$(timeout --signal=TERM --kill-after=5s 180s "$python" -I -B "$bundle/check_runtime_008.py" \
    "$run_root" "$old" "$weights" "$ffmpeg" "$ffprobe")
printf '%s\n' "$runtime"
runtime_hash=$(awk '/^EGOX_RUNTIME_FINGERPRINT / {print $2}' <<< "$runtime")
[[ $runtime_hash =~ ^[a-f0-9]{64}$ ]] || fail 'Missing verified runtime fingerprint.'
driver=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader,nounits)
bundle_hash=$(sha256sum "$bundle/SHA256SUMS" | cut -d' ' -f1)
fingerprint=$(printf '%s\n' "$bundle_hash" "$runtime_hash" "$driver" | sha256sum | cut -d' ' -f1)
timeout --signal=TERM --kill-after=5s 120s "$python" -B "$bundle/prepare_infer.py" \
    --official-infer "$old/infer.py" --meta "$bundle/meta.json" --check-only
printf 'EGOX_AB_PREFLIGHT_OK %s seed=%s arm_timeout_s=%s\n' "$fingerprint" "$seed" "$arm_timeout"
[[ $mode != preflight ]] || exit 0

[[ ! -L $out_root ]] || fail "Result root is a symbolic link: $out_root."
mkdir -p -- "$out_root"
shopt -s nullglob

verify_arm() {
    local arm=$1 out=$2
    shift 2
    timeout --signal=TERM --kill-after=5s 120s "$python" -I -B "$bundle/verify_ab_008.py" \
        --arm "$arm" --out "$out" --fingerprint "$fingerprint" --ffprobe "$ffprobe" --ffmpeg "$ffmpeg" "$@"
}

run_arm() (
    local arm=$1 arm_root="$out_root/$1" attempt stage code exit_code
    [[ ! -L $arm_root ]] || fail "Arm directory is a symbolic link: $arm_root."
    mkdir -p -- "$arm_root"
    for attempt in "$arm_root"/attempt-*; do
        [[ -d $attempt && ! -L $attempt ]] || fail "Unexpected attempt path: $attempt."
        if [[ -e $attempt/verified-result.json ]]; then
            verify_arm "$arm" "$attempt"
            printf 'EGOX_ARM_REUSED %s %s\n' "$arm" "$attempt"
            return
        fi
        if [[ -f $attempt/inference-exit-code.txt && $(<"$attempt/inference-exit-code.txt") == 0 ]]; then
            [[ -f $attempt/fingerprint.txt && $(<"$attempt/fingerprint.txt") == "$fingerprint" ]] ||
                fail "Existing successful process for $arm belongs to a different runtime or bundle."
            verify_arm "$arm" "$attempt" --complete
            printf 'EGOX_ARM_RECOVERED %s %s\n' "$arm" "$attempt"
            return
        fi
    done

    check_idle_a800
    local disk_free
    disk_free=$(df -Pk "$out_root" | awk 'END {print $4}')
    [[ $disk_free =~ ^[0-9]+$ ]] && (( disk_free >= 2097152 )) || fail 'Less than 2 GiB of local scratch is available.'
    attempt=$(mktemp -d "$arm_root/attempt-XXXXXXXX")
    stage="$attempt/work"
    mkdir -- "$stage" "$attempt/cache" "$attempt/tmp" "$attempt/projection"
    printf '%s\n' "$fingerprint" >"$attempt/fingerprint.txt"
    timeout --signal=TERM --kill-after=5s 120s "$python" -B "$bundle/prepare_infer.py" \
        --official-infer "$old/infer.py" --meta "$bundle/meta.json" --output-dir "$attempt/projection" \
        >"$attempt/projection.log" 2>&1
    case "$arm" in
        A) code=infer_a.py ;;
        B) code=infer_b.py ;;
    esac
    cp -- "$attempt/projection/$code" "$stage/infer.py"
    local -a code_files=(infer.py meta.json)
    if [[ $arm == B ]]; then
        cp -- "$attempt/projection/aria_gga.py" "$stage/aria_gga.py"
        code_files+=(aria_gga.py)
    fi
    cp -- "$bundle/meta.json" "$stage/meta.json"
    ln -s -- "$(readlink -f "$old/core")" "$stage/core"

    local exo="example/self_capture/videos/$take/exo.mp4"
    local prior="example/self_capture/ego_priors/$take/ego_Prior.mp4"
    local depths="example/self_capture/depth_maps/$take"
    local relative depth
    for relative in "$exo" "$prior"; do
        mkdir -p -- "$stage/$(dirname -- "$relative")"
        ln -s -- "$(readlink -f "$old/$relative")" "$stage/$relative"
    done
    mkdir -p -- "$stage/$depths"
    for depth in "$old/$depths"/*.npy; do
        ln -s -- "$(readlink -f "$depth")" "$stage/$depths/${depth##*/}"
    done
    cd -- "$stage"
    sha256sum "${code_files[@]}" >"$attempt/executed-code-and-meta.sha256"

    export HF_HOME="$attempt/cache" HF_HUB_CACHE="$attempt/cache/hub"
    export TRANSFORMERS_CACHE="$attempt/cache/hub" XDG_CACHE_HOME="$attempt/cache"
    export TMPDIR="$attempt/tmp" EGOX_AUDIT_PATH="$attempt/runtime_audit.json"
    local -a command=("$python" -B "$audit" --meta_data_file ./meta.json
        --model_path "$weights/Wan2.1-I2V-14B-480P-Diffusers"
        --lora_path "$weights/EgoX/pytorch_lora_weights.safetensors"
        --lora_rank 256 --out "$attempt" --seed "$seed" --use_GGA
        --cos_sim_scaling_factor 3.0 --in_the_wild --idx -1)
    {
        printf 'arm=%s\nstarted_at=%s\nseed=%s\ndriver=%s\n' "$arm" "$(date -Is)" "$seed" "$driver"
        printf 'command='; printf ' %q' "${command[@]}"; printf '\n'
    } >"$attempt/launch.log"
    printf 'EGOX_ARM_STARTED %s %s %s\n' "$arm" "$(date -Is)" "$attempt"
    exit_code=0
    timeout --signal=TERM --kill-after=10s "${arm_timeout}s" "${command[@]}" >"$attempt/inference.log" 2>&1 || exit_code=$?
    printf '%s\n' "$exit_code" >"$attempt/inference-exit-code.txt"
    printf 'ended_at=%s\nexit_code=%s\n' "$(date -Is)" "$exit_code" >>"$attempt/launch.log"
    [[ $exit_code == 0 ]] || fail "Arm $arm quit with $exit_code. Evidence retained at $attempt."
    verify_arm "$arm" "$attempt" --complete
    printf 'EGOX_ARM_COMPLETE %s %s %s\n' "$arm" "$(date -Is)" "$attempt"
)

run_arm A
if [[ $mode == run-a ]]; then
    printf 'EGOX_A_COMPLETE %s %s\n' "$fingerprint" "$out_root"
    exit 0
fi
run_arm B
printf 'EGOX_AB_COMPLETE %s %s\n' "$fingerprint" "$out_root"
