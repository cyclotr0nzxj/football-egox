"""Build a fixed virtual look-at trajectory from EXO-only annotations and ViPE exports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from vipe.utils.io import read_depth_artifacts, read_instance_artifacts, read_rgb_artifacts


FRAMES = 49
HEIGHT, WIDTH = 448, 784
TARGET_PIXEL = (650, 184)
TARGET_INSTANCE = 1
DRIBBLER_INSTANCE = 2


def backproject(x, y, z, intrinsics):
    fx, fy, cx, cy = intrinsics
    return np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=-1)


def surface_depth(depth, mask, pixel, instance):
    x, y = pixel
    region = np.s_[y - 2:y + 3, x - 2:x + 3]
    samples = depth[region][mask[region] == instance]
    if not np.isfinite(samples).all() or len(samples) < 8:
        raise ValueError(f'Insufficient instance {instance} depth at {pixel}.')
    return float(np.median(samples)), int(len(samples))


def fit_ground(depth, mask, rgb, intrinsics):
    y, x = np.mgrid[247:428:5, 45:741:6]
    colors = rgb[y, x]
    green = ((colors[..., 1] - colors[..., 0] > 0.015) &
             (colors[..., 1] - colors[..., 2] > 0.015))
    grass = green & (mask[y, x] == 0)
    cloud = backproject(x[grass], y[grass], depth[y[grass], x[grass]], intrinsics)
    if len(cloud) < 1000:
        raise ValueError('Insufficient visible grass to estimate the camera up direction.')
    center = np.median(cloud, axis=0)
    for _ in range(3):
        _, _, basis = np.linalg.svd(cloud - center, full_matrices=False)
        up = basis[-1]
        if up[1] > 0:
            up = -up
        keep = abs((cloud - center) @ up) < 0.06
        if keep.mean() < 0.8:
            raise ValueError('Visible grass does not support a consistent local plane.')
        cloud = cloud[keep]
        center = np.mean(cloud, axis=0)
    return up, float(-up @ center)


def smooth_track(track):
    padded = np.pad(track, ((2, 2), (0, 0)), mode='edge')
    return sum(weight * padded[i:i + FRAMES] for i, weight in enumerate((1, 2, 3, 2, 1))) / 9


def unit(vector):
    size = np.linalg.norm(vector)
    if size < 1e-6:
        raise ValueError('Virtual camera has an undefined viewing axis.')
    return vector / size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('source_exo', type=Path)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compensate-official-roll', action='store_true')
    args = parser.parse_args()

    annotation = json.loads(args.annotations.read_text())
    source_hash = hashlib.sha256(args.source_exo.read_bytes()).hexdigest()
    if annotation['video_sha256'] != source_hash or (annotation['width'], annotation['height']) != (WIDTH, HEIGHT):
        raise ValueError('Annotations do not match the exact EXO file and grid.')
    frames = annotation['frames']
    if [frame['frame'] for frame in frames] != list(range(FRAMES)):
        raise ValueError('Each EXO frame needs an independent image-plane annotation.')
    with np.load(args.artifacts / 'pose/exo.npz', allow_pickle=False) as pose:
        if not np.array_equal(pose['inds'], np.arange(FRAMES)) or not np.array_equal(pose['data'], np.tile(np.eye(4), (FRAMES, 1, 1))):
            raise ValueError('The artifact coordinate frame must be the fixed, identity EXO camera.')
    with np.load(args.artifacts / 'intrinsics/exo.npz', allow_pickle=False) as calibration:
        intrinsics = calibration['data'][0].astype(float)
        if not np.array_equal(calibration['inds'], np.arange(FRAMES)) or not np.all(calibration['data'] == intrinsics):
            raise ValueError('This trajectory builder requires constant indexed EXO intrinsics.')

    heads, targets, planes, head_counts = [], [], [], []
    streams = zip(read_depth_artifacts(args.artifacts / 'depth/exo.zip'),
                  read_instance_artifacts(args.artifacts / 'mask/exo.zip'),
                  read_rgb_artifacts(args.artifacts / 'rgb/exo.mp4'), frames, strict=True)
    for frame, ((depth_id, depth), (mask_id, mask), (rgb_id, rgb), anchor) in enumerate(streams):
        if (depth_id, mask_id, rgb_id) != (frame, frame, frame):
            raise ValueError('Geometry and annotation indices differ.')
        x, y = int(anchor['x']), int(anchor['y'])
        if not (2 <= x < WIDTH - 2 and 2 <= y < HEIGHT - 2):
            raise ValueError('Head point is outside the usable image.')
        depth, mask, rgb = depth.numpy(), mask.numpy(), rgb.numpy()
        if depth.shape != (HEIGHT, WIDTH) or mask.shape != depth.shape or not (np.isfinite(depth).all() and (depth > 0).all()):
            raise ValueError(f'Invalid ViPE depth or instance map at frame {frame}.')
        head_z, samples = surface_depth(depth, mask, (x, y), DRIBBLER_INSTANCE)
        target_z, _ = surface_depth(depth, mask, TARGET_PIXEL, TARGET_INSTANCE)
        heads.append(backproject(x, y, head_z, intrinsics))
        targets.append(backproject(*TARGET_PIXEL, target_z, intrinsics))
        head_counts.append(samples)
        if frame in (0, 12, 24, 36, 48):
            planes.append(fit_ground(depth, mask, rgb, intrinsics))

    head_track = np.array(heads)
    target = np.median(targets, axis=0)
    track = smooth_track(head_track)
    up = unit(np.median([plane[0] for plane in planes], axis=0))
    camera_height = float(np.median([plane[1] for plane in planes]))
    heights = track @ up + camera_height
    if heights.min() < 0.6 or heights.max() > 2.3 or np.max(np.linalg.norm(np.diff(track, axis=0), axis=1)) > 0.35:
        raise ValueError('Head height or frame-to-frame motion is inconsistent with this clip.')
    extrinsics = []
    # Renderer and EgoX both use clockwise quarter-turn image coordinates.
    # Counter-roll the camera basis so both inputs describe the same upright image.
    counter_roll = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    for center in track:
        forward = unit(target - center)
        right = unit(np.cross(forward, up))
        down = unit(np.cross(forward, right))
        rotation = np.stack((right, down, forward))
        if args.compensate_official_roll:
            rotation = counter_roll @ rotation
        if not np.isclose(np.linalg.det(rotation), 1.0) or (rotation @ (target - center))[2] < 0.5:
            raise ValueError('Invalid OpenCV camera basis or look-at target.')
        extrinsics.append(np.column_stack((rotation, -rotation @ center)).tolist())

    fx, fy, cx, cy = intrinsics
    # Nominal equidistant horizontal FOV, before the renderer's fixed distortion terms.
    fov_degrees, center = 95.0, 224.0
    focal = center / np.deg2rad(fov_degrees * 0.5)
    meta = {'test_datasets': [{
        'exo_path': './example/self_capture/videos/pair04_s0024/exo.mp4',
        'ego_prior_path': './example/self_capture/ego_priors/pair04_s0024/ego_Prior.mp4',
        'prompt': 'A soccer player dribbles a ball past blue markers on an outdoor field with a teammate ahead.',
        'camera_intrinsics': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        'camera_extrinsics': np.eye(4)[:3].tolist(),
        'ego_intrinsics': [[focal, 0, center], [0, focal, center], [0, 0, 1]],
        'ego_extrinsics': extrinsics,
    }]}
    report = {
        'source_exo_sha256': source_hash,
        'frames': FRAMES,
        'method': 'EXO-annotated head surface, instance-masked ViPE Z, five-frame triangular smoothing, virtual look at teammate upper torso',
        'world_frame': 'Identity ViPE EXO camera, units estimated by ViPE',
        'nominal_equidistant_fov_degrees': fov_degrees,
        'official_clockwise_image_rotation_compensated': args.compensate_official_roll,
        'teammate_pixel_xy': TARGET_PIXEL,
        'minimum_head_depth_samples': min(head_counts),
        'fitted_exo_camera_height_vipe_units': camera_height,
        'ground_up': up.tolist(),
        'smoothed_head_height_range_vipe_units': [float(heights.min()), float(heights.max())],
        'head_centers_vipe_units': track.tolist(),
        'source_pts_seconds': [i / 25 for i in range(FRAMES)],
        'render_pts_seconds': [i / 30 for i in range(FRAMES)],
    }
    args.output.mkdir()
    for file_name, data in [('meta.json', meta), ('trajectory-check.json', report)]:
        (args.output / file_name).write_text(json.dumps(data, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
