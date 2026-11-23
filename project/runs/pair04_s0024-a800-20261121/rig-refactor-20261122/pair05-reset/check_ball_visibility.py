"""Explain the observed loss of Pair05 ball pixels at the target Z-buffer."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from rebuild_pair05 import (RUN, PROJECT, TAKE, HEIGHT, WIDTH, SIZE, WEARER, backproject, closest_surface,
                           coefficients, load_depth, observed_objects, project, read_geometry, read_json,
                           read_mask, rebuild_points, require_hash, review_frames)


def audit(results):
    keyframes = (12, 24)
    geometry = RUN / 'multiclip-evidence' / TAKE / 'geometry/vipe_results' / TAKE
    receipt = read_json(results / 'rebuild-receipt.json')
    source = RUN / 'multiclip-prep' / TAKE
    exo = source / f'example/self_capture/videos/{TAKE}/exo.mp4'
    require_hash(exo, receipt['source_sha256']['EXO'])
    frames = review_frames(exo, WIDTH, keyframes)
    heads = read_json(source / 'exo_head_anchors.json')['frames']
    old_intrinsics = read_geometry(geometry)
    old_up = np.asarray(receipt['input_fitted_plane']['unit_up'])
    old_height = receipt['input_fitted_plane']['height_vipe_units']
    distortion = np.concatenate([coefficients(PROJECT / 'records/official_renderer_geometry.py',
                                             f'{term}_distortion_coeffs')
                                 for term in ('radial', 'tangential', 'thinPrism')])
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    py, px = np.mgrid[:SIZE, :SIZE]
    circle = np.hypot(px - SIZE / 2, py - SIZE / 2).ravel() <= SIZE / 2
    evidence = {}
    variants = {}
    for name in ('nominal', 'alternate'):
        camera = results / f'{name}-camera.json'
        require_hash(camera, receipt['products'][camera.name])
        variants[name] = read_json(camera)
    with (ZipFile(geometry / 'depth/exo.zip') as depths, ZipFile(geometry / 'mask/exo.zip') as masks,
          ZipFile(RUN / 'update-008/ball-recovery/pair05-ball.zip') as balls):
        for frame in keyframes:
            instances = read_mask(masks, frame)
            ball = read_mask(balls, frame) == 255
            labels = observed_objects(np.asarray(frames[frame]), instances, ball).ravel()
            ball_distance = cv2.distanceTransform((~ball).astype(np.uint8), cv2.DIST_L2, 5).ravel()
            head_y = heads[frame]['y']
            cutoff = head_y + .62 * (np.nonzero(instances == WEARER)[0].max() - head_y)
            wearer_torso = ((instances == WEARER) & (yy < cutoff)).ravel()
            frame_key = f'F{frame:02}'
            evidence[frame_key] = {'source_ball_pixels': int(ball.sum())}
            for name, camera in variants.items():
                intrinsics = np.asarray(camera['exo_intrinsics'])
                up = np.asarray(camera['ground_up'])
                rays = backproject(xx, yy, np.ones_like(xx), intrinsics).reshape(-1, 3)
                basis = (*tuple(intrinsics[:2] / old_intrinsics[:2]), 1.)
                points, valid, _ = rebuild_points(load_depth(depths, frame), rays, old_up * basis, old_height, up)
                source, destinations, z = project(points, np.asarray(camera['ego_extrinsics'][frame]),
                                                 np.asarray(camera['ego_intrinsics']), distortion,
                                                 np.flatnonzero(valid & ~wearer_torso))
                winner, front_z = closest_surface(source, destinations, z)
                ball_source = (labels[source] == 2) & circle[destinations]
                ball_dest = np.unique(destinations[ball_source])
                hit = labels[winner[ball_dest]] == 2
                occluded = ball_dest[~hit]
                ball_front_z = np.full_like(front_z, np.inf)
                np.minimum.at(ball_front_z, destinations[ball_source], z[ball_source])
                if int(hit.sum()) != receipt['metrics'][frame_key][name]['objects']['ball']['pixels']:
                    raise ValueError(f'Same-frame repeat disagrees with {frame_key} {name}.')
                gap = ball_front_z[occluded] - front_z[occluded]
                distance = ball_distance[winner[occluded]]
                evidence[frame_key][name] = {
                    'source_ball_pixels_in_target_circle': int(ball_source.sum()),
                    'unique_ball_pixels_before_visibility': int(len(ball_dest)),
                    'ball_pixels_after_visibility': int(hit.sum()),
                    'occluder_source_instance_histogram': np.bincount(instances.ravel()[winner[occluded]]).tolist(),
                    'occluder_distance_to_mask_px': np.quantile(distance, [0, .5, 1]).tolist() if distance.size else None,
                    'estimated_front_gap_m': np.quantile(gap, [0, .5, 1]).tolist() if gap.size else None,
                }
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    details = audit(args.results)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(details, handle, indent=2, allow_nan=False)
        handle.write('\n')
