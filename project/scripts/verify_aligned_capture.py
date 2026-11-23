#!/usr/bin/env python3
"""Fully decode aligned exports and record frame-count/integrity evidence."""
import argparse
import json
import pathlib
import subprocess
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--pair')
    parser.add_argument('--wait-for-exports', action='store_true')
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    results = []
    for pair in manifest['pairs']:
        if args.pair and pair['id'] != args.pair:
            continue
        folder = ROOT / manifest['output_root'] / pair['id']
        if args.wait_for_exports:
            deadline = time.monotonic() + 1800
            while True:
                log_path = folder / 'export_log.json'
                log = json.loads(log_path.read_text()) if log_path.exists() else {}
                if log.get('status') == 'complete':
                    break
                if log.get('status') == 'failed':
                    raise RuntimeError('Export failed: ' + str(log.get('error')))
                if time.monotonic() > deadline:
                    raise TimeoutError('Timed out waiting for export: ' + pair['id'])
                time.sleep(3)
        for name in ['ego_aligned.mp4', 'exo_aligned.mp4', 'comparison.mp4']:
            path = folder / name
            print('Verifying', pair['id'], name, flush=True)
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror',
                       '-hwaccel', 'videotoolbox', '-i', str(path),
                       '-map', '0:v:0', '-map', '0:a:0',
                       '-fps_mode', 'passthrough', '-progress', 'pipe:1',
                       '-nostats', '-f', 'null', '-']
            result = subprocess.run(command, capture_output=True, text=True)
            counts = [int(line.split('=', 1)[1]) for line in result.stdout.splitlines()
                      if line.startswith('frame=')]
            decoded_frames = counts[-1] if counts else None
            passed = (result.returncode == 0 and decoded_frames == pair['frame_count'])
            entry = dict(pair=pair['id'], path=str(path.relative_to(ROOT)),
                         expected_frames=pair['frame_count'], decoded_frames=decoded_frames,
                         full_decode_passed=passed, command=command,
                         stderr=result.stderr, size_bytes=path.stat().st_size,
                         checked_at=datetime.now(timezone.utc).isoformat())
            results.append(entry)
            (folder / (name + '.decode_check.json')).write_text(
                json.dumps(entry, ensure_ascii=False, indent=2) + '\n')
            if not passed:
                raise RuntimeError('Decode/frame count verification failed: ' + str(path))
    report = manifest_path.parent / ('decode_verification' + ('_' + args.pair if args.pair else '') + '.json')
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('PASS:', len(results), 'videos;', report, flush=True)


if __name__ == '__main__':
    main()
