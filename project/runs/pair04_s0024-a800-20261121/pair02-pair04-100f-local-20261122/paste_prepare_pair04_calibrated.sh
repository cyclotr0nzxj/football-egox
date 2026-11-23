(
set -euo pipefail
base=/root/autodl-tmp/football_egox/pair04-20261121
run="$base/pair02-pair04-100f-geometry-prep"
out="$base/pair04-inverse-depth-20261122"
test ! -e "$out"
py="$base/renderer-env/bin/python"
stage="$run/prepare_pair04_inverse_depth.py"
partial=$(mktemp "$run/.pair04-calib.XXXXXXXX")
trap 'rm -f -- "$partial"' EXIT
base64 -d <<'PY_EGOX' | gzip -d > "$partial"
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
hiHWeRqGiLbDkA7tOyCE3p3/AGpqcdtMHQAA
PY_EGOX
printf '%s  %s\n' f94196155fc750663ecb673032fb1fb889f1417b46ce54da898671beccd8004d "$partial" | sha256sum -c -
if [ ! -e "$stage" ]; then ln -- "$partial" "$stage"; fi
printf '%s  %s\n' f94196155fc750663ecb673032fb1fb889f1417b46ce54da898671beccd8004d "$stage" | sha256sum -c -
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENCV_IO_ENABLE_OPENEXR=1 "$py" -B "$stage" --source "$run" --inputs "$run" --legacy "$run/legacy/Pair04" --output "$out"
"$py" - "$run" "$out" <<'PY_PATCH'
from pathlib import Path
import shutil, sys
run,out = map(Path,sys.argv[1:])
old_name='pair02-pair04-100f-geometry-prep'
new_name=out.name
driver=(run/'continue_fixed_008.py').read_text()
assert driver.count(old_name)==1 and driver.count("win/'overlap-fixed.json'")==1
with (out/'continue_fixed_008.py').open('x') as f: f.write(driver.replace(old_name,new_name))
for name in ('stage_geometry.py','check_overlap_geometry.py','assemble_extension.py','verify_egox_outputs_008.py','check_runtime_008.py','collect_008.py'):
 shutil.copy2(run/name,out/name)
PY_PATCH
printf 'PAIR04_BRANCH_READY=%s\n' "$out"
)
