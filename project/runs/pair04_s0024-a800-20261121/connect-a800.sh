#!/usr/bin/env bash
# Run interactively on the Mac during this instance window.
set -euo pipefail
control_dir=/tmp/pair04-a800.dWQKiW
test -d "$control_dir" && test -O "$control_dir" && test ! -L "$control_dir"
exec /usr/bin/ssh -MN -p 45845 -S "$control_dir/ctl" \
  -o ControlMaster=yes -o ControlPersist=no -o ServerAliveInterval=15 \
  -o StrictHostKeyChecking=ask -o UserKnownHostsFile="$control_dir/known_hosts" \
  -o HostName=192.0.2.1 -o 'HostKeyAlias=[example.invalid]:45845' \
  -o 'ProxyCommand=/usr/bin/nc -4 -n -b en0 -G 8 %h %p' \
  root@example.invalid
