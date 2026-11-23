#!/usr/bin/env python3
"""Render and verify the four signed-off three-view comparison videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from hashlib import file_digest
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/pair04_s0024-a800-20261121'
SYNC = RUN / 'four-egox-outputs-008/synced-comparison-25fps'
GEOMETRY = RUN / 'geometry-preview-delivery-20261122'
FRAMES = 49
FPS = 25
PANEL = 448
GAP = 12
HEADER = 86
WIDTH = 3 * PANEL + 4 * GAP
HEIGHT = HEADER + PANEL + GAP
GEOMETRY_LABELS = {'01': '几何预览 · 低头看球', '05': '几何预览 · 跟球'}


def command(args, **options):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, **options).stdout


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def decode(path, start, edge):
    color = 'scale={0}:{0}:force_original_aspect_ratio=decrease:force_divisible_by=2'
    selection = f'trim=start_frame={start}:end_frame={start + FRAMES},'
    layout = color.format(edge) + f',pad={edge}:{edge}:(ow-iw)/2:(oh-ih)/2:color=0x171c23,setsar=1'
    filters = selection + layout if edge == PANEL else 'setsar=1'
    raw = command([
        'ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '4', '-i', str(path),
        '-an', '-vf', filters, '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'
    ])
    dimensions = (FRAMES, edge if edge == PANEL else HEIGHT, edge, 3)
    if len(raw) != int(np.prod(dimensions)):
        raise ValueError(f'Unexpected decoded frame count or geometry: {path}')
    return np.frombuffer(raw, np.uint8).reshape(dimensions)


def verify_output(path, panes):
    probe = json.loads(command([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,time_base,duration_ts:frame=pts',
        '-of', 'json', str(path)
    ]))
    stream, = probe['streams']
    time_base = Fraction(stream['time_base'])
    timestamps = [int(frame['pts']) * time_base for frame in probe['frames']]
    if (stream['codec_name'] != 'h264' or (stream['width'], stream['height']) != (WIDTH, HEIGHT)
            or Fraction(stream['avg_frame_rate']) != FPS
            or int(stream['duration_ts']) * time_base != Fraction(FRAMES, FPS)
            or timestamps != [Fraction(i, FPS) for i in range(FRAMES)]):
        raise ValueError(f'Unexpected output frame count, size, timing or codec: {path}')
    actual = decode(path, 0, WIDTH)
    errors = []
    for index, expected in enumerate(panes):
        x = GAP + index * (PANEL + GAP)
        decoded = actual[:, HEADER:HEADER + PANEL, x:x + PANEL]
        per_frame_mae = np.abs(decoded.astype(np.int16) - expected).mean(axis=(1, 2, 3))
        if float(per_frame_mae.max()) > 6:
            raise ValueError(f'Panel pixels deviate from expected frame sequence: {path}, panel {index}')
        errors.append(round(float(per_frame_mae.max()), 3))
    return actual, errors


def render(row, geometry_records, output, font_file, font_index):
    pair_id = row['pair']
    pair = 'Pair' + pair_id
    start, end = row['original_visible_frame_range']
    exo, nano = [ROOT / filename for filename in row['original_reference_files']]
    if pair_id in GEOMETRY_LABELS:
        geometry = geometry_records[pair]
        result = GEOMETRY / geometry['output']
        expected_hash = geometry['sha256']
        result_type = 'geometry_preview'
        result_label = GEOMETRY_LABELS[pair_id]
        provenance = geometry['source']
    else:
        result = ROOT / row['generated']
        expected_hash = row['generated_sha256']
        result_type = 'egox_generation'
        result_label = 'EgoX 转换'
        provenance = row['generated']
    if end - start + 1 != FRAMES or sha256(result) != expected_hash:
        raise ValueError(f'Source selection or frozen result changed: {pair}')
    panes = [decode(exo, start, PANEL), decode(nano, start, PANEL), decode(result, 0, PANEL)]
    labels = ['第三人称 · R8', '实拍第一人称 · Nano', result_label]
    title_font = ImageFont.truetype(font_file, 22, index=font_index)
    label_font = ImageFont.truetype(font_file, 19, index=font_index)
    frames = []
    for i in range(FRAMES):
        canvas = Image.new('RGB', (WIDTH, HEIGHT), '#10141a')
        draw = ImageDraw.Draw(canvas)
        draw.text((GAP, 13), pair, font=title_font, fill='#ffffff')
        draw.text((WIDTH - GAP, 14), f'源帧 {start + i:03d}', anchor='rt',
                  font=title_font, fill='#b7c1cb')
        for index, (frames_in, label) in enumerate(zip(panes, labels, strict=True)):
            x = GAP + index * (PANEL + GAP)
            draw.text((x, 54), label, font=label_font, fill='#d6dee7')
            canvas.paste(Image.fromarray(frames_in[i]), (x, HEADER))
        frames.append(canvas.tobytes())
    path = output / 'videos' / f'{pair}.mp4'
    command([
        'ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS), '-i', '-',
        '-c:v', 'libx264', '-threads', '4', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-video_track_timescale', '25000', '-movflags', '+faststart', str(path)
    ], input=b''.join(frames))
    del frames
    actual, errors = verify_output(path, panes)
    still = output / 'stills' / f'{pair}.jpg'
    Image.fromarray(actual[24]).save(still, quality=90, optimize=True)
    return {'pair': pair, 'source_frame_range_in_synchronized_clips': [start, end],
            'frames': FRAMES, 'fps': FPS, 'duration_seconds': FRAMES / FPS,
            'result_type': result_type, 'video': f'videos/{path.name}',
            'video_sha256': sha256(path), 'still': f'stills/{still.name}',
            'still_source_frame': start + 24, 'still_sha256': sha256(still),
            'panel_sources_sha256': [sha256(exo), sha256(nano), expected_hash],
            'max_frame_rgb_mae_per_panel_vs_input_0_to_255': errors,
            'verified': ['full_decode', 'all_presentation_timestamps', 'ordered_source_panels']}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', required=True)
    parser.add_argument('--font-index', type=int, default=0)
    args = parser.parse_args()
    output = args.output
    for name in ('videos', 'stills'):
        (output / name).mkdir(parents=True, exist_ok=True)
    sync = json.loads((SYNC / 'verification.json').read_text())
    geometry = json.loads((GEOMETRY / 'verification.json').read_text())
    rows = sorted(sync['rows'], key=lambda row: row['pair'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda row: render(row, geometry, output, args.font, args.font_index), rows))
    manifest = {'panel_order': ['R8 exocentric input', 'Nano egocentric reference', 'EgoX or geometry'],
                'physical_sync_uncertainty_frames': sync['physical_exo_nano_sync_uncertainty_frames'],
                'clips': [row for row, _ in results]}
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    for row, source in results:
        print(row['pair'], row['source_frame_range_in_synchronized_clips'],
              row['max_frame_rgb_mae_per_panel_vs_input_0_to_255'], source)


if __name__ == '__main__':
    main()
