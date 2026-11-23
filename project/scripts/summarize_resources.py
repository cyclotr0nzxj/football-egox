"""Summarize actual inference telemetry without starting any GPU computation."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/root/autodl-tmp/football_egox')
args = parser.parse_args()
root = Path(args.root)
attempts = json.loads((root/'records/attempts.json').read_text())
summaries = []
for attempt in attempts:
    directory = Path(attempt['output_directory'])
    rows = [json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
    gpu = []
    for row in rows:
        if 'gpu_csv' in row:
            values = row['gpu_csv'].split(',')
            gpu.append({'used_mib': float(values[1]), 'util_percent': float(values[3]), 'power_watts': float(values[4])})
    audit_path = directory/'runtime_audit.json'
    summary = {'attempt': attempt['attempt'], 'status': attempt['status'],
        'started_utc': attempt['started_utc'], 'ended_utc': attempt.get('ended_utc'),
        'elapsed_seconds': attempt.get('elapsed_seconds'), 'telemetry_samples': len(rows),
        'sample_interval_seconds': 2,
        'peak_sampled_gpu_used_mib': max((g['used_mib'] for g in gpu), default=None),
        'peak_sampled_gpu_util_percent': max((g['util_percent'] for g in gpu), default=None),
        'peak_sampled_gpu_power_watts': max((g['power_watts'] for g in gpu), default=None),
        'peak_sampled_container_memory_bytes': max((r['cgroup_memory_bytes'] for r in rows if 'cgroup_memory_bytes' in r), default=None),
        'runtime_audit': json.loads(audit_path.read_text()) if audit_path.exists() else None,
        'limits': 'nvidia-smi and cgroup peaks are sampled at two-second intervals; cgroup memory includes page cache and other container processes. Runtime CUDA peaks describe this inference process. No billing price was provided, so monetary cost is not estimated.'}
    summaries.append(summary)
(root/'records/resource_summary.json').write_text(json.dumps(summaries, indent=2))
print(json.dumps(summaries, indent=2))
