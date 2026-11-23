#!/usr/bin/env python3
"""Replay the pinned Pair05 prior with an audited EXO pixel exclusion."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np


TAKE = 'pair05_s0024'
HEIGHT, WIDTH = 448, 784
FRAME_NAMES = tuple(f'{i:05d}.png' for i in range(49))
RENDERER_SHA256 = '177721241c123019e7f1932639c59efb510e314514f561ef9d54ce122f6a1ee8'
EXCLUSION_SHA256 = '6b74ac3711c5336dd4ba7f4cba8f30419417bfa8343a37fe5766623bb307ffa3'
ARTIFACT_SHA256 = {
    'mask/exo.zip': '7747332adc5416a8fc64ad3f7254164bafaed49b8ba8a7513e515a3e553d59d3',
    'rgb/exo.mp4': '4c32794c02e33b182c96baffe89cd07a76f99b7cd8d99fa02d851d2e73e8f925',
    'depth/exo.zip': '755ff85728c051cc197d26bc871bacf16d054d5a7e66d40dd79006215dbb082e',
    'pose/exo.npz': 'abdff3a87c4a92928d04c60eb95801eebd8092ed1408432355baefd95678407d',
    'intrinsics/exo.npz': 'c2072b4762d9ee70baa0a6406cbb0f41d64f1f232b161e0d86fc483e3bd1e744',
    'intrinsics/exo_camera.txt': 'cd0772f72835c4da35e4833e4d7bd53f0e6f6093e50f1fa82f1522064b08d682',
}


def read_verified(path, expected_sha256):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return data


def read_mask(archive, frame_name):
    pixels = cv2.imdecode(np.frombuffer(archive.read(frame_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.shape != (HEIGHT, WIDTH) or pixels.dtype != np.uint8:
        raise ValueError(f'Expected a {WIDTH}x{HEIGHT} uint8 mask: {archive.filename}/{frame_name}')
    return pixels


def load_exclusions(geometry, exclusion_zip):
    if geometry.name != TAKE:
        raise ValueError(f'Expected the archived ViPE geometry directory {TAKE}')
    for relative_path, sha256 in ARTIFACT_SHA256.items():
        read_verified(geometry / relative_path, sha256)

    masks = []
    retained = []
    with ZipFile(exclusion_zip) as excluded_zip, ZipFile(geometry / 'mask/exo.zip') as labels_zip:
        for archive in (excluded_zip, labels_zip):
            if tuple(sorted(archive.namelist())) != FRAME_NAMES:
                raise ValueError(f'Expected exactly frames 00000.png to 00048.png: {archive.filename}')
        for frame_name in FRAME_NAMES:
            raw_exclusion = read_mask(excluded_zip, frame_name)
            labels = read_mask(labels_zip, frame_name)
            excluded = raw_exclusion == 255
            if np.any((raw_exclusion != 0) & ~excluded) or not excluded.any() or np.any(labels[excluded] != 1):
                raise ValueError(f'Exclusion must be a nonempty binary subset of wearer instance 1: {frame_name}')
            lower_body = (labels == 1) & ~excluded
            if not lower_body[::2, ::2].any() or not (labels == 2)[::2, ::2].any():
                raise ValueError(f'Expected retained wearer and second player at renderer sampling stride: {frame_name}')
            masks.append(excluded)
            retained.append(int(lower_body.sum()))
    return masks, retained


def patch_renderer(source):
    edits = (
        (
            '        # Keep only background (instance == 0) when instance mask is available\n',
            '        depth_mask &= allowed_exo_pixels(frame_idx, depth, spatial_subsample)\n\n',
        ),
        (
            '        final_mask = static_mask & depth_mask\n',
            '        depth_mask &= torch.as_tensor(allowed_exo_pixels(frame_idx, depth), device=depth.device)\n',
        ),
        (
            '        # Keep only dynamic instances (instance != 0)\n',
            '        depth_mask &= allowed_exo_pixels(frame_idx, depth, spatial_subsample)\n\n',
        ),
    )
    for anchor, filter_line in edits:
        if source.count(anchor) != 1:
            raise ValueError(f'Renderer source seam changed: {anchor.strip()}')
        source = source.replace(anchor, filter_line + anchor, 1)
    return source


def validate_meta(path):
    samples = json.loads(path.read_text())['test_datasets']
    matching = [sample for sample in samples if Path(sample['exo_path']).parent.name == TAKE]
    if len(matching) != 1:
        raise ValueError(f'Expected exactly one {TAKE} camera entry in {path}')
    for key, expected_shape in (('camera_intrinsics', (3, 3)), ('camera_extrinsics', (3, 4)),
                                ('ego_intrinsics', (3, 3)), ('ego_extrinsics', (49, 3, 4))):
        matrix = np.asarray(matching[0][key], dtype=np.float64)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f'Invalid {key} in {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', required=True, type=Path)
    parser.add_argument('--exclusion-zip', required=True, type=Path)
    parser.add_argument('--input_dir', required=True, type=Path)
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--meta_json_path', required=True, type=Path)
    parser.add_argument('--point_size', type=float, choices=[5.0], default=5.0)
    parser.add_argument('--start_frame', type=int, choices=[0], default=0)
    parser.add_argument('--end_frame', type=int, choices=[48], default=48)
    parser.add_argument('--fish_eye_rendering', action='store_true', required=True)
    parser.add_argument('--use_mean_bg', action='store_true', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    renderer = args.renderer.resolve(strict=True)
    code = compile(patch_renderer(read_verified(renderer, RENDERER_SHA256).decode('utf-8')), str(renderer), 'exec')
    read_verified(args.exclusion_zip, EXCLUSION_SHA256)
    masks, retained = load_exclusions(args.input_dir, args.exclusion_zip)
    validate_meta(args.meta_json_path)
    target = args.out_dir / TAKE
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Renderer destination already exists: {target}')

    print(json.dumps({
        'take': TAKE,
        'check_only': args.check_only,
        'renderer_sha256': RENDERER_SHA256,
        'exclusion_sha256': EXCLUSION_SHA256,
        'excluded_pixels_per_frame': [int(mask.sum()) for mask in masks],
        'retained_wearer_pixels_per_frame': retained,
    }), flush=True)
    if args.check_only:
        return

    def allowed_exo_pixels(frame_index, depth, stride=1):
        if tuple(depth.shape) != (HEIGHT, WIDTH) or not 0 <= frame_index < len(masks):
            raise ValueError(f'Invalid render sample at frame {frame_index}: {tuple(depth.shape)}')
        return ~masks[frame_index][::stride, ::stride]

    target.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(renderer.parent))
    sys.argv = [str(renderer), '--input_dir', str(args.input_dir), '--out_dir', str(args.out_dir),
                '--meta_json_path', str(args.meta_json_path), '--point_size', str(args.point_size),
                '--start_frame', str(args.start_frame), '--end_frame', str(args.end_frame),
                '--fish_eye_rendering', '--use_mean_bg']
    exec(code, {'__name__': '__main__', '__file__': str(renderer), 'allowed_exo_pixels': allowed_exo_pixels})


if __name__ == '__main__':
    main()
