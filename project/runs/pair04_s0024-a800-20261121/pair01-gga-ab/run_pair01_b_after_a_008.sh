#!/usr/bin/env bash
set -euo pipefail
umask 077

# Reuse the exact launcher's checks and B commands without ever dispatching A.
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /path/to/the/original/pair01-gga-ab-bundle\n' "$0" >&2
    exit 2
fi
frozen_bundle=$(cd -- "$1" && pwd -P)
frozen_launcher="$frozen_bundle/run_pair01_gga_ab_008.sh"
(
    cd -- "$frozen_bundle"
    printf '%s\n' \
        'aad159ddd1a6b5844ba508217829c39712cdca35070a6022632fa5c27704ea93  SHA256SUMS' \
        'aa614afba60e098f31b2db6532b2c6308c3ee9371cfb4d71a96d11536157927e  run_pair01_gga_ab_008.sh' |
        sha256sum -c -
)

# The previously executed manifest must stay intact so its fingerprint matches A.
b_launcher=$(mktemp "$frozen_bundle/.pair01-run-b-XXXXXXXX")
trap 'rm -f -- "$b_launcher"' EXIT
awk '
    /^run_arm A$/ { found = 1; exit }
    { print }
    END { if (!found) exit 1 }
' "$frozen_launcher" >"$b_launcher"
cat >>"$b_launcher" <<'B_ONLY'
baseline_fingerprint=39160604a212244d3baf24590b82e37baf693686a3a114ada53e752250515e4e
[[ $fingerprint == "$baseline_fingerprint" ]] || fail 'Runtime or frozen bundle differs from recovered A.'
baseline="$out_root/A/attempt-j9f1V6y5"
[[ -d $baseline && ! -L $baseline ]] || fail 'The original A attempt is unavailable.'
printf '%s  %s\n' c52f27a0af2d213e24bcc242ce056dbb6c35308cc9f96b2b18e41fe089da8f42 \
    "$baseline/verified-result.json" | sha256sum -c -
verify_arm A "$baseline"
printf 'EGOX_ARM_REUSED A %s\n' "$baseline"
run_arm B
printf 'EGOX_B_COMPLETE %s %s\n' "$fingerprint" "$out_root"
B_ONLY
bash -n "$b_launcher"
bash "$b_launcher" run
