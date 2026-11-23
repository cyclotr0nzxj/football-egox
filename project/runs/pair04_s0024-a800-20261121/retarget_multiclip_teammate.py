"""Keep the verified virtual camera position and aim it at the visible teammate."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_multiclip_virtual_view import COUNTER_ROLL, backproject, read_labels, smooth_track, unit
from vipe.utils.io import read_depth_artifacts, read_instance_artifacts


def torso_anchor(mask, instance):
    count, components, stats, _ = cv2.connectedComponentsWithStats((mask == instance).astype(np.uint8))
    if count < 2:
        raise ValueError('The visible teammate is missing.')
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left, top, width, height, area = map(int, stats[idx])
    if area < 1000 or height < 60 or width < 15:
        raise ValueError('Too little teammate mask is visible to determine a torso anchor.')
    interior = (components[top:top + height, left:left + width] == idx).astype(np.uint8)
    distance = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    row, column = np.mgrid[:height, :width]
    eligible = ((row >= 0.20 * height) & (row <= 0.47 * height) &
                (column >= 0.25 * width) & (column <= 0.75 * width))
    score = np.where(eligible, distance, 0)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    if score[y, x] < 3.0:
        raise ValueError('The upper-body region does not support an interior anchor.')
    return int(left + x), int(top + y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('viewpoint', type=Path)
    parser.add_argument('teammate_instance', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    labels = read_labels(args.artifacts / 'mask/exo.txt')
    report = json.loads((args.viewpoint / 'trajectory-check.json').read_text())
    meta = json.loads((args.viewpoint / 'meta.json').read_text())
    team = args.teammate_instance
    if labels.get(team) != 'person' or team == report['head_instance_id']:
        raise ValueError('Target must be a distinct observed person in the EXO masks.')
    camera = meta['test_datasets'][0]
    k = np.array(camera['camera_intrinsics'])
    intrinsics = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
    centers = np.array(report['head_centers_vipe_units'])
    up = unit(np.array(report['ground_up']))
    masks = read_instance_artifacts(args.artifacts / 'mask/exo.zip')
    depths = read_depth_artifacts(args.artifacts / 'depth/exo.zip')
    targets, pixels = [], []
    for i, ((j, mask), (k, depth)) in enumerate(zip(masks, depths, strict=True)):
        if i != j or i != k:
            raise ValueError('Mask and depth frames do not match.')
        mask, depth = mask.numpy(), depth.numpy()
        x, y = torso_anchor(mask, team)
        patch = np.s_[y - 2:y + 3, x - 2:x + 3]
        values = depth[patch][mask[patch] == team]
        if len(values) < 8 or not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f'Target torso has no supported depth at F{i}.')
        pixels.append([x, y])
        targets.append(backproject(x, y, float(np.median(values)), intrinsics))
    if len(targets) != len(centers) or len(targets) != 49:
        raise ValueError('Expected a target track for all 49 frames.')
    targets = smooth_track(np.array(targets))
    heights = targets @ up + report['fitted_exo_camera_height_vipe_units']
    if heights.min() < 0.45 or heights.max() > 2.6 or np.max(np.linalg.norm(np.diff(targets, axis=0), axis=1)) > 0.35:
        raise ValueError('ViPE target path is geometrically unstable.')

    poses, directions, distances = [], [], []
    for target, center in zip(targets, centers, strict=True):
        forward = unit(target - center)
        right = unit(np.cross(forward, up))
        down = unit(np.cross(forward, right))
        rotation = COUNTER_ROLL @ np.stack((right, down, forward))
        aim = rotation @ (target - center)
        if not np.isclose(np.linalg.det(rotation), 1) or not np.allclose(aim[:2], 0, atol=1e-8) or aim[2] < 1.5:
            raise ValueError('Target is not well supported in front of the virtual camera.')
        poses.append(np.column_stack((rotation, -rotation @ center)).tolist())
        directions.append(forward)
        distances.append(float(aim[2]))
    directions = np.array(directions)
    angles = np.degrees(np.arccos(np.clip(np.sum(directions[1:] * directions[:-1], axis=1), -1, 1)))
    if np.max(angles) > 7:
        raise ValueError('Virtual teammate look-at direction jumps between frames.')
    camera['ego_extrinsics'] = poses
    camera['prompt'] = 'A soccer player on an outdoor field plays the ball with a teammate ahead.'
    receipt = {
        'method': 'Verified head centers unchanged. Virtual orientation follows an EXO-visible teammate torso using ViPE instance and Z.',
        'target_instance_id': team,
        'target_pixels_xy': pixels,
        'target_centers_vipe_units': targets.tolist(),
        'minimum_target_forward_depth_vipe_units': min(distances),
        'maximum_step_rotation_degrees': float(np.max(angles)),
    }
    args.output.mkdir()
    for name, data in [('meta.json', meta), ('retarget-check.json', receipt)]:
        (args.output / name).write_text(json.dumps(data, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
