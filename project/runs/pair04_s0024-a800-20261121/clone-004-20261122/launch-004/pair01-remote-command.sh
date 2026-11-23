#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname) == autodl-container-d293479255-5411ceb1 ]]
cd /root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008
zip=pair01-first-orientation-bundle-004.zip
printf '%s  %s\n' 4469cdf673321b6f8d5d93558e08635f4783027d019efe5d3d176998c5f86ca6 "$zip" | sha256sum -c --strict
work=$(mktemp -d "$PWD/.launch-004-pair01-XXXXXXXX")
unzip -q "$zip" -d "$work"
printf 'PAIR004_LAUNCH_DIR %s\n' "$work"
./egox-env/bin/python -B "$work/pair01-first-orientation-bundle/run_candidate_008.py" render
