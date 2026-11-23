#!/usr/bin/env python3
"""Render the audited Pair05 EXO ball pixels on the per-frame dynamic layer."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZipFile

import cv2
import numpy as np


BASE_WRAPPER_SHA256 = '9549de452ec842831c8a038d3494c601e71a4b664433cf99dda7d880b776a019'
BALL_LABEL = 7
EXR_Z_CHANNEL = b'Z\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00'


def load_base_wrapper(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_WRAPPER_SHA256:
        raise ValueError(f'SHA-256 mismatch: {path}')
    wrapper = ModuleType('render_pair05')
    exec(compile(source, str(path), 'exec'), vars(wrapper))
    return wrapper


def load_ball_masks(base, geometry, ball_zip, expected_sha256, exclusions):
    raw_zip = ball_zip.read_bytes()
    sha256 = hashlib.sha256(raw_zip).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f'SHA-256 mismatch: {ball_zip}')

    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    depth_names = tuple(name.replace('.png', '.exr') for name in base.FRAME_NAMES)
    candidates, marked = [], []
    with (ZipFile(BytesIO(raw_zip)) as ball_archive,
          ZipFile(geometry / 'mask/exo.zip') as label_archive,
          ZipFile(geometry / 'depth/exo.zip') as depth_archive):
        if tuple(sorted(ball_archive.namelist())) != base.FRAME_NAMES:
            raise ValueError(f'Expected exactly frames 00000.png to 00048.png: {ball_zip}')
        if tuple(sorted(depth_archive.namelist())) != depth_names:
            raise ValueError('Expected the 49 indexed Pair05 depth frames')
        for name, depth_name, exclusion in zip(base.FRAME_NAMES, depth_names, exclusions, strict=True):
            mask = base.read_mask(ball_archive, name)
            visible = mask == 255
            if np.any((mask != 0) & ~visible):
                raise ValueError(f'Expected binary 0/255 ball mask: {name}')
            if np.any(visible & exclusion):
                raise ValueError(f'Ball mask overlaps the wearer exclusion: {name}')

            labels = base.read_mask(label_archive, name)
            if np.any(labels == BALL_LABEL):
                raise ValueError(f'ViPE already uses label {BALL_LABEL}: {name}')

            # OpenCV reads grayscale Y. ViPE stores one Z channel with the same pixel layout.
            raw_depth = depth_archive.read(depth_name)
            if raw_depth.count(EXR_Z_CHANNEL) != 1:
                raise ValueError(f'Expected one ViPE Z channel: {depth_name}')
            depth = cv2.imdecode(np.frombuffer(raw_depth.replace(EXR_Z_CHANNEL, b'Y' + EXR_Z_CHANNEL[1:], 1),
                                              dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.shape != visible.shape or depth.dtype != np.float32:
                raise ValueError(f'Invalid EXO depth shape or format: {depth_name}')

            marked.append(int(visible.sum()))
            candidates.append(visible & (labels == 0) & np.isfinite(depth) & (depth > 0))

    if not any(mask[::2, ::2].any() for mask in candidates):
        raise ValueError('No marked background pixels with viable depth at renderer sampling stride 2')
    return candidates, marked, sha256


def patch_ball_labels(source):
    edits = (
        ('        # Keep only background (instance == 0) when instance mask is available\n',
         '        instance_mask = promote_ball_pixels(frame_idx, instance_mask, depth_mask, spatial_subsample)\n'),
        ('        # Keep only background (instance_id == 0)\n',
         '        depth_mask = reliable_depth_mask_range(depth)\n'
         '        instance_mask = promote_ball_pixels(frame_idx, instance_mask, depth_mask)\n'),
        ('        # Keep only dynamic instances (instance != 0)\n',
         '        instance_mask = promote_ball_pixels(frame_idx, instance_mask, depth_mask, spatial_subsample)\n'),
    )
    for anchor, insertion in edits:
        if source.count(anchor) != 1:
            raise ValueError(f'Renderer source seam changed: {anchor.strip()}')
        source = source.replace(anchor, insertion + anchor, 1)

    moved_filter = '        # Mask out invalid depth values\n        depth_mask = reliable_depth_mask_range(depth)\n'
    if source.count(moved_filter) != 1:
        raise ValueError('Renderer mean-background depth filter seam changed')
    return source.replace(moved_filter, '', 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-wrapper', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'prior/render_pair05.py')
    parser.add_argument('--renderer', required=True, type=Path)
    parser.add_argument('--exclusion-zip', required=True, type=Path)
    parser.add_argument('--ball-zip', required=True, type=Path)
    parser.add_argument('--ball-sha256')
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

    base = load_base_wrapper(args.base_wrapper)
    renderer = args.renderer.resolve(strict=True)
    source = base.read_verified(renderer, base.RENDERER_SHA256).decode('utf-8')
    code = compile(patch_ball_labels(base.patch_renderer(source)), str(renderer), 'exec')
    base.read_verified(args.exclusion_zip, base.EXCLUSION_SHA256)
    exclusions, retained = base.load_exclusions(args.input_dir, args.exclusion_zip)
    candidates, marked, ball_sha256 = load_ball_masks(base, args.input_dir, args.ball_zip,
                                                      args.ball_sha256, exclusions)
    base.validate_meta(args.meta_json_path)
    target = args.out_dir / base.TAKE
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Renderer destination already exists: {target}')

    print(json.dumps({
        'take': base.TAKE,
        'check_only': args.check_only,
        'renderer_sha256': base.RENDERER_SHA256,
        'exclusion_sha256': base.EXCLUSION_SHA256,
        'ball_sha256': ball_sha256,
        'ball_label': BALL_LABEL,
        'marked_pixels_per_frame': marked,
        'depth_supported_background_ball_pixels_per_frame': [int(mask.sum()) for mask in candidates],
        'stride2_candidate_pixels_per_frame': [int(mask[::2, ::2].sum()) for mask in candidates],
        'excluded_wearer_pixels_per_frame': [int(mask.sum()) for mask in exclusions],
        'retained_wearer_pixels_per_frame': retained,
    }), flush=True)
    if args.check_only:
        return

    import torch

    dynamic_counts = {}

    def allowed_exo_pixels(frame_index, depth, stride=1):
        if tuple(depth.shape) != (base.HEIGHT, base.WIDTH) or not 0 <= frame_index < len(exclusions):
            raise ValueError(f'Invalid render sample at frame {frame_index}: {tuple(depth.shape)}')
        return ~exclusions[frame_index][::stride, ::stride]

    def promote_ball_pixels(frame_index, labels, depth_mask, stride=1):
        if labels is None or tuple(labels.shape) != (base.HEIGHT, base.WIDTH):
            raise ValueError(f'Invalid renderer instance mask at frame {frame_index}')
        if not 0 <= frame_index < len(candidates):
            raise ValueError(f'Unexpected renderer frame index: {frame_index}')
        sampled = candidates[frame_index][::stride, ::stride]
        if sampled.shape != tuple(depth_mask.shape):
            raise ValueError(f'Mismatched renderer depth mask at frame {frame_index}')
        y, x = np.nonzero(sampled)
        if isinstance(depth_mask, np.ndarray):
            valid = depth_mask[y, x]
            y, x = y[valid], x[valid]
        y = torch.as_tensor(y, device=labels.device)
        x = torch.as_tensor(x, device=labels.device)
        if isinstance(depth_mask, torch.Tensor):
            valid = depth_mask[y, x]
            y, x = y[valid], x[valid]
        y, x = y * stride, x * stride
        if stride == 2:
            dynamic_counts[frame_index] = y.numel()
        if not y.numel():
            return labels
        if torch.any(labels[y, x] != 0):
            raise ValueError(f'Renderer labels disagree with the verified EXO archive at frame {frame_index}')
        promoted = labels.clone()
        promoted[y, x] = BALL_LABEL
        return promoted

    target.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(renderer.parent))
    sys.argv = [str(renderer), '--input_dir', str(args.input_dir), '--out_dir', str(args.out_dir),
                '--meta_json_path', str(args.meta_json_path), '--point_size', str(args.point_size),
                '--start_frame', str(args.start_frame), '--end_frame', str(args.end_frame),
                '--fish_eye_rendering', '--use_mean_bg']
    exec(code, {'__name__': '__main__', '__file__': str(renderer), 'allowed_exo_pixels': allowed_exo_pixels,
                'promote_ball_pixels': promote_ball_pixels})
    counts = [dynamic_counts.get(i, 0) for i in range(len(candidates))]
    if len(dynamic_counts) != len(candidates) or not any(counts):
        raise ValueError('Incomplete dynamic rendering or renderer depth filter rejected all ball pixels')
    print(json.dumps({'take': base.TAKE, 'reliable_dynamic_ball_pixels_per_frame': counts}), flush=True)


if __name__ == '__main__':
    main()
