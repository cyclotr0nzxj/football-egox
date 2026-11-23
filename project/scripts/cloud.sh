#!/bin/bash
set -euo pipefail
# Session authentication is held by OpenSSH; no credentials are stored here.
exec ssh -p 23955 -o Hostname=192.0.2.1 \
  -o HostKeyAlias=example.invalid \
  -o ControlPath=/tmp/football-egox-ssh-%C \
  -o ConnectTimeout=15 root@example.invalid "$@"
