#!/usr/bin/env bash
# Only after explicit user confirmation that the authorized 008 is running.
set -uo pipefail
bundle=${1:?EXO preparation ZIP path required}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
base=/root/autodl-tmp/football_egox/pair04-20261121
run="$base/pair02-pair04-100f-geometry-prep"
[[ ! -e "$run" ]] || { echo "PIPELINE_BLOCKED existing run; inspect before any retry" >&2; exit 2; }
if ! bash "$script_dir/prepare_vipe_lite.sh" "$bundle"; then
  echo 'GEOMETRY_INCOMPLETE: eligible independent pairs can still continue' >&2
fi
[[ -d "$run" && -f "$run/runtime-preflight.log" ]] && grep -q "^EGOX_RUNTIME_FINGERPRINT " "$run/runtime-preflight.log" || { echo 'PIPELINE_BLOCKED no preflighted run' >&2; exit 2; }
failures=0
for pair in Pair02 Pair04; do
  if [[ $pair == Pair02 ]]; then first=pair02_s0028; second=pair02_s0052; else first=pair04_s0052; second=pair04_s0076; fi
  if [[ ! -f "$run/windows/$first/overlap.json" || ! -f "$run/windows/$second/overlap.json" ]]; then
    echo "PAIR_SKIPPED $pair incomplete geometry" >&2; failures=1; continue
  fi
  if ! "$base/renderer-env/bin/python" -B "$run/continue_008.py" all --run "$run" --pair "$pair"; then
    echo "PAIR_INCOMPLETE $pair inspect run before retry" >&2; failures=1
  fi
done
archive="$base/pair02-pair04-100f-evidence-$(date +%Y%m%d-%H%M%S).zip"
if ! "$base/renderer-env/bin/python" -B "$run/collect_008.py" --run "$run" --output "$archive"; then
  echo 'EVIDENCE_COLLECTION_FAILED inspect cloud state' >&2; exit 2
fi
echo "EVIDENCE_ARCHIVE $archive"
(( failures == 0 )) || exit 2
echo 'ALL_WINDOWS_TECHNICALLY_VERIFIED; retrieve evidence and perform semantic and seam review before shutdown.'
