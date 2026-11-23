#!/usr/bin/env bash
# Interactive SSH only. Do not store passwords, keys, or host-key bypasses here.
set -euo pipefail
if [[ $# -lt 1 || ! $1 =~ ^[0-9]{1,5}$ ]] || (( 10#$1 < 1 || 10#$1 > 65535 )); then
  echo 'Usage: connect_008_physical.sh CURRENT_SSH_PORT_FROM_AUTODL [REMOTE_COMMAND ...]' >&2
  exit 2
fi
port=$1
shift
host=example.invalid
interface=en0
if ! /sbin/ifconfig "$interface" | /usr/bin/grep -q 'status: active'; then
  echo 'Physical en0 is not active; recheck the current network first.' >&2
  exit 2
fi
if ! /usr/sbin/netstat -rn -f inet | /usr/bin/grep -E '^default[[:space:]].*[[:space:]]en0([[:space:]]|$)' >/dev/null; then
  echo 'No physical en0 default route; recheck the current network first.' >&2
  exit 2
fi
ip=$(/usr/bin/curl -fsS --connect-timeout 6 --max-time 10 -H 'accept: application/dns-json' \
  'https://cloudflare-dns.com/dns-query?name=example.invalid&type=A' | \
  /usr/bin/python3 -c 'import ipaddress,json,sys; d=json.load(sys.stdin); a=[x["data"] for x in d.get("Answer",[]) if x.get("type")==1]; assert a; i=ipaddress.ip_address(a[0]); assert i.version==4 and i.is_global; print(i)')
alias="[$host]:$port"
known_hosts="$HOME/.ssh/known_hosts"
historical_hosts=/tmp/pair01-008-recover.EvkKRx/known_hosts
if /usr/bin/ssh-keygen -F "$alias" -f "$known_hosts" >/dev/null 2>&1; then
  :
elif [[ -f "$historical_hosts" && ! -L "$historical_hosts" && -O "$historical_hosts" ]] && /usr/bin/ssh-keygen -F "$alias" -f "$historical_hosts" >/dev/null 2>&1; then
  known_hosts=$historical_hosts
else
  echo "No trusted host key recorded for $alias. Verify the current instance host key through an independent trusted channel before connecting." >&2
  exit 2
fi
printf 'Connecting to %s port %s via %s; strict host-key verification remains enabled.\n' "$host" "$port" "$interface" >&2
exec /usr/bin/ssh -p "$port" -o "HostName=$ip" -o "HostKeyAlias=$alias" \
  -o "UserKnownHostsFile=$known_hosts" -o StrictHostKeyChecking=yes \
  -o 'ProxyCommand=/usr/bin/nc -4 -n -b en0 -G 8 %h %p' "root@$host" "$@"
