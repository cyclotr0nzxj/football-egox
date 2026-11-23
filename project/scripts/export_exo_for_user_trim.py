#!/usr/bin/env python3
"""Transfer a verified user EGO trim to its previously aligned EXO video."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def probe(path, packets=False):
    cmd = ['ffprobe', '-v', 'error', '-show_streams', '-show_format']
    if packets:
        cmd += ['-select_streams', 'v:0', '-show_packets', '-show_data_hash', 'sha256',
                '-show_entries', 'packet=pts_time,dts_time,duration_time,flags,data_hash']
    return json.loads(subprocess.check_output(cmd + ['-of', 'json', str(path)]))


def visible(data):
    return sorted([p for p in data['packets']
                   if float(p['pts_time']) >= 0 and 'D' not in p.get('flags', '')],
                  key=lambda p: Fraction(p['pts_time']))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def export(pair, record):
    source, reference, output = (ROOT / pair[k] for k in
                                 ['source_exo', 'reference_ego', 'output_exo'])
    assert not output.exists(), 'Refusing to overwrite ' + str(output)
    output.parent.mkdir(exist_ok=True)
    partial = output.with_name(output.stem + '.partial.mp4')
    assert not partial.exists()
    expected = visible(probe(reference, True))
    assert len(expected) == pair['visible_frame_count']
    input_hash = sha256(reference)
    start = pair['start_seconds']
    duration, last = pair['duration_seconds'], pair['last_frame_pts']
    # Drop only excess trailing packets. No picture payload is changed by these
    # bitstream filters. Match the reference's fractional last-frame hold time.
    bsf = (f"noise=drop='gt(pts*tb,{last + .00001:.9f})',"
           f"setts=duration='if(gt(PTS*TB,{last - .00001:.9f}),"
           f"round(({duration:.12f}-PTS*TB)/TB),DURATION)'")
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-ss', str(start),
               '-i', str(source), '-t', f'{duration:.12f}', '-map', '0:v:0',
               '-map', '0:a:0', '-c:v', 'copy', '-bsf:v', bsf,
               '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
               '-af', f'apad,atrim=end={duration:.12f}',
               '-avoid_negative_ts', 'disabled', '-use_editlist', '1',
               '-movflags', '+faststart', '-video_track_timescale', '12800',
               '-movie_timescale', '128000', str(partial)]
    receipt = {'pair': pair, 'command': command, 'reference_sha256': input_hash,
               'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(record / (pair['id'] + '_export.json'), receipt)
    print(pair['id'], 'exporting', output.name, flush=True)
    subprocess.run(command, check=True)
    result = probe(partial, True)
    actual = visible(result)
    assert len(actual) == len(expected), 'Visible frame count differs'
    assert [p['pts_time'] for p in actual] == [p['pts_time'] for p in expected], 'PTS differs'
    video = next(s for s in result['streams'] if s['codec_type'] == 'video')
    assert (video['width'], video['height']) == (3840, 2160)
    exact_duration = Fraction(video['duration_ts']) * Fraction(video['time_base'])
    assert abs(float(exact_duration) - duration) <= 1 / 12800
    assert abs(float(result['format']['duration']) - duration) <= .00005
    source_packets = probe(source, True)['packets']
    source_map = {round(float(p['pts_time']), 6): p for p in source_packets}
    for p in actual:
        source_packet = source_map[round(float(p['pts_time']) + start, 6)]
        assert p['data_hash'] == source_packet['data_hash'], 'Video payload changed'
    assert sha256(reference) == input_hash, 'Reference EGO changed'
    os.link(partial, output)
    partial.unlink()
    receipt.update(completed_at=datetime.now(timezone.utc).isoformat(),
                   status='exported_and_packet_verified', visible_frames=len(actual),
                   visible_pts_match_reference=True, video_packets_unchanged=True,
                   output_sha256=sha256(output), probe=result)
    write_json(record / (pair['id'] + '_export.json'), receipt)
    print(pair['id'], 'PASS:', len(actual), 'visible frames;', float(exact_duration), 'seconds', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    path = ROOT / args.manifest
    manifest = json.loads(path.read_text())
    for pair in manifest['pairs']:
        export(pair, path.parent)


if __name__ == '__main__':
    main()
