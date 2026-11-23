"""Build an upright virtual camera from ViPE geometry and EXO-only head anchors."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


FRAMES = 49
HEIGHT, WIDTH = 448, 784
MIN_PATCH_SAMPLES = 8
GROUND_FRAMES = (0, 12, 24, 36, 48)
COUNTER_ROLL = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


def pixel_xy(point):
    x, y = point
    if type(x) is not int or type(y) is not int or not (2 <= x < WIDTH - 2 and 2 <= y < HEIGHT - 2):
        raise ValueError(f'Annotation pixel {point} must be an integer at least two pixels from the image edge.')
    return x, y


def read_geometry(root):
    with np.load(root / 'pose/exo.npz', allow_pickle=False) as pose:
        if not np.array_equal(pose['inds'], np.arange(FRAMES)) or not np.array_equal(
            pose['data'], np.broadcast_to(np.eye(4), (FRAMES, 4, 4))
        ):
            raise ValueError('The indexed ViPE poses must define a fixed identity EXO world frame.')
    with np.load(root / 'intrinsics/exo.npz', allow_pickle=False) as calibration:
        data = calibration['data']
        if (
            not np.array_equal(calibration['inds'], np.arange(FRAMES))
            or data.shape != (FRAMES, 4)
            or not np.isfinite(data).all()
            or not np.all(data == data[0])
        ):
            raise ValueError('Expected 49 finite and constant indexed EXO intrinsics.')
        intrinsics = data[0].astype(float)
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0 or not np.allclose((cx, cy), (WIDTH / 2, HEIGHT / 2), atol=0.5, rtol=0):
        raise ValueError('EXO intrinsics do not match the centered EgoX input grid.')
    return intrinsics


def read_labels(path):
    labels = {}
    for line in path.read_text().splitlines():
        instance, label = line.split(':', 1)
        index = int(instance)
        if index in labels or index < 0 or not label.strip():
            raise ValueError(f'Invalid instance label: {line!r}')
        labels[index] = label.strip().lower()
    if labels.get(0) != 'background' or 'person' not in labels.values():
        raise ValueError('ViPE instance labels must identify background and people.')
    return labels


def backproject(x, y, z, intrinsics):
    fx, fy, cx, cy = intrinsics
    return np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=-1)


def project(points, intrinsics):
    fx, fy, cx, cy = intrinsics
    x, y, z = points.T
    if not np.isfinite(points).all() or np.any(z <= 0):
        raise ValueError('Invalid EXO reprojection depth.')
    return np.column_stack((fx * x / z + cx, fy * y / z + cy))


def fit_ground(depth, mask, rgb, intrinsics):
    y, x = np.mgrid[247:428:5, 45:741:6]
    colors = rgb[y, x]
    green = (colors[..., 1] - colors[..., 0] > 0.015) & (colors[..., 1] - colors[..., 2] > 0.015)
    grass = green & (mask[y, x] == 0)
    cloud = backproject(x[grass], y[grass], depth[y[grass], x[grass]], intrinsics)
    if len(cloud) < 1000:
        raise ValueError('Insufficient visible grass for the ground plane.')
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
    return up, float(-up @ center), int(len(cloud))


def read_frame_evidence(root, source, anchors, target, intrinsics):
    from vipe.utils.io import read_depth_artifacts, read_instance_artifacts, read_rgb_artifacts

    head_patches, target_patches, planes, rgb_differences = [], [], [], []
    frames = zip(
        read_depth_artifacts(root / 'depth/exo.zip'),
        read_instance_artifacts(root / 'mask/exo.zip'),
        read_rgb_artifacts(root / 'rgb/exo.mp4'),
        read_rgb_artifacts(source),
        anchors,
        strict=True,
    )
    for index, ((depth_id, depth), (mask_id, mask), (rgb_id, rgb), (source_id, source_rgb), anchor) in enumerate(frames):
        if (depth_id, mask_id, rgb_id, source_id) != (index,) * 4:
            raise ValueError(f'Geometry, RGB and annotation indices differ at frame {index}.')
        depth, mask, rgb, source_rgb = (image.numpy() for image in (depth, mask, rgb, source_rgb))
        if (
            depth.shape != (HEIGHT, WIDTH)
            or mask.shape != depth.shape
            or rgb.shape != (HEIGHT, WIDTH, 3)
            or source_rgb.shape != rgb.shape
            or not np.isfinite(depth).all()
            or np.any(depth <= 0)
            or not np.issubdtype(mask.dtype, np.integer)
        ):
            raise ValueError(f'Invalid depth, mask or RGB grid at frame {index}.')
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32) / 255
        if np.issubdtype(source_rgb.dtype, np.integer):
            source_rgb = source_rgb.astype(np.float32) / 255
        rgb_mae = float(np.mean(np.abs(rgb[::4, ::4] - source_rgb[::4, ::4])))
        if not np.isfinite(rgb_mae) or rgb_mae > 0.05:
            raise ValueError(f'ViPE RGB does not match the staged source at frame {index}: MAE {rgb_mae:.4f}.')
        rgb_differences.append(rgb_mae)
        for (x, y), patches in ((anchor, head_patches), (target, target_patches)):
            region = np.s_[y - 2:y + 3, x - 2:x + 3]
            patches.append((depth[region].copy(), mask[region].copy()))
        if index in GROUND_FRAMES:
            planes.append(fit_ground(depth, mask, rgb, intrinsics))
    return head_patches, target_patches, planes, rgb_differences


def select_stable_instance(patches, candidates, role):
    masks = np.array([mask for _, mask in patches])
    counts = {instance: np.sum(masks == instance, axis=(1, 2)) for instance in candidates}
    stable = [instance for instance, votes in counts.items() if np.all(votes >= MIN_PATCH_SAMPLES)]
    if not stable:
        supports = {instance: int(np.sum(votes >= MIN_PATCH_SAMPLES)) for instance, votes in counts.items()}
        raise ValueError(f'{role} has no stable instance across all 49 frames. Support by instance: {supports}.')
    # Prefer the mask at the actual anchor, then use the total patch coverage.
    def confidence(instance):
        return int(np.sum(masks[:, 2, 2] == instance)), int(np.sum(counts[instance]))

    stable.sort(key=confidence, reverse=True)
    if len(stable) > 1 and confidence(stable[0])[0] - confidence(stable[1])[0] < 5:
        raise ValueError(f'{role} instance is ambiguous: {[(i, confidence(i)) for i in stable]}.')
    instance = stable[0]
    samples = np.array([np.median(depth[mask == instance]) for depth, mask in patches])
    return instance, samples, counts[instance], confidence(instance)[0]


def smooth_track(track):
    padded = np.pad(track, ((2, 2), (0, 0)), mode='edge')
    return sum(weight * padded[i:i + FRAMES] for i, weight in enumerate((1, 2, 3, 2, 1))) / 9


def unit(vector):
    size = np.linalg.norm(vector)
    if not np.isfinite(size) or size < 1e-6:
        raise ValueError('Virtual camera has an undefined viewing axis.')
    return vector / size


def make_view(name, anchors, target_pixel, intrinsics, labels, heads, targets, planes, rgb_differences):
    people = sorted(instance for instance, label in labels.items() if label == 'person')
    head_id, head_depth, head_counts, head_center_frames = select_stable_instance(heads, people, 'Head')
    target_candidates = [instance for instance in people if instance != head_id]
    target_candidates.append(0)
    target_id, target_depth, target_counts, target_center_frames = select_stable_instance(
        targets, target_candidates, 'Target',
    )

    x, y = np.array(anchors).T
    raw_track = backproject(x, y, head_depth, intrinsics)
    track = smooth_track(raw_track)
    head_pixels = project(track, intrinsics)
    head_errors = np.linalg.norm(head_pixels - anchors, axis=1)
    if np.max(head_errors) > 13:
        worst = int(np.argmax(head_errors))
        raise ValueError(f'Smoothed head misses EXO anchor by {head_errors[worst]:.2f}px at F{worst}.')

    up = unit(np.median([plane[0] for plane in planes], axis=0))
    camera_height = float(np.median([plane[1] for plane in planes]))
    plane_angles = np.degrees(np.arccos(np.clip([plane[0] @ up for plane in planes], -1, 1)))
    if (
        not 0.5 <= camera_height <= 3.0
        or np.max(plane_angles) > 12
        or np.max(np.abs([plane[1] - camera_height for plane in planes])) > 0.45
    ):
        raise ValueError('The sampled ground planes disagree on EXO scale or orientation.')
    heights = track @ up + camera_height
    raw_steps = np.linalg.norm(np.diff(raw_track, axis=0), axis=1)
    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    if heights.min() < 0.6 or heights.max() > 2.3 or np.max(raw_steps) > 0.75 or np.max(steps) > 0.35:
        raise ValueError(
            f'Implausible head trajectory: height range {heights.min():.3f}–{heights.max():.3f}, '
            f'max raw/smoothed steps {np.max(raw_steps):.3f}/{np.max(steps):.3f} ViPE units.'
        )

    target_track = backproject(*target_pixel, target_depth, intrinsics)
    target = np.median(target_track, axis=0)
    target_deviation = np.linalg.norm(target_track - target, axis=1)
    target_heights = target_track @ up + camera_height
    if np.percentile(target_deviation, 90) > 0.55 or np.max(target_deviation) > 1.0:
        raise ValueError(f'The fixed target is unstable; max movement {np.max(target_deviation):.3f} ViPE units.')
    if target_id == 0 and np.max(np.abs(target_heights)) > 0.3:
        raise ValueError('The selected background target is unsupported by the ground plane.')

    extrinsics, target_forward_z = [], []
    for center in track:
        offset = target - center
        forward = unit(offset)
        right = unit(np.cross(forward, up))
        down = unit(np.cross(forward, right))
        rotation = COUNTER_ROLL @ np.stack((right, down, forward))
        aim = rotation @ offset
        if (
            not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1.0)
            or not np.allclose(aim[:2], 0, atol=1e-8)
            or aim[2] < 2.0
        ):
            raise ValueError('The look-at must project to the virtual center at least 2 ViPE units in front.')
        target_forward_z.append(float(aim[2]))
        extrinsics.append(np.column_stack((rotation, -rotation @ center)).tolist())

    fx, fy, cx, cy = intrinsics
    center = 224.0
    focal = center / np.deg2rad(95.0 / 2)
    meta = {'test_datasets': [{
        'exo_path': f'./example/self_capture/videos/{name}/exo.mp4',
        'ego_prior_path': f'./example/self_capture/ego_priors/{name}/ego_Prior.mp4',
        'prompt': 'A soccer player dribbles a ball on an outdoor field.',
        'camera_intrinsics': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        'camera_extrinsics': np.eye(4)[:3].tolist(),
        'ego_intrinsics': [[focal, 0, center], [0, focal, center], [0, 0, 1]],
        'ego_extrinsics': extrinsics,
    }]}
    report = {
        'name': name,
        'frames': FRAMES,
        'method': 'EXO head anchors, ViPE Z restricted to a stable person mask, five-frame smoothing, fixed virtual look-at',
        'world_frame': 'Identity ViPE EXO camera, with estimated ViPE units',
        'nominal_equidistant_fov_degrees': 95.0,
        'official_clockwise_image_rotation_compensated': True,
        'head_instance_id': head_id,
        'head_anchor_in_instance_frames': head_center_frames,
        'head_depth_samples_by_frame': head_counts.tolist(),
        'minimum_head_depth_samples': int(head_counts.min()),
        'head_depth_vipe_units_by_frame': head_depth.tolist(),
        'head_centers_vipe_units': track.tolist(),
        'head_raw_centers_vipe_units': raw_track.tolist(),
        'head_reprojected_pixels_by_frame': head_pixels.tolist(),
        'head_reprojection_error_px_by_frame': head_errors.tolist(),
        'max_head_reprojection_error_px': float(np.max(head_errors)),
        'max_head_raw_step_vipe_units': float(np.max(raw_steps)),
        'max_head_smoothed_step_vipe_units': float(np.max(steps)),
        'smoothed_head_height_range_vipe_units': [float(heights.min()), float(heights.max())],
        'fitted_exo_camera_height_vipe_units': camera_height,
        'ground_up': up.tolist(),
        'ground_planes': [
            {'frame': frame, 'up': plane[0].tolist(), 'camera_height': plane[1], 'inliers': plane[2],
             'angle_from_median_degrees': float(angle)}
            for frame, plane, angle in zip(GROUND_FRAMES, planes, plane_angles, strict=True)
        ],
        'target_pixel_xy': target_pixel,
        'target_kind': 'stationary_person' if target_id else 'ground_landmark',
        'target_instance_id': target_id,
        'target_anchor_in_instance_frames': target_center_frames,
        'target_depth_samples_by_frame': target_counts.tolist(),
        'target_depth_vipe_units_by_frame': target_depth.tolist(),
        'target_vipe_units': target.tolist(),
        'target_deviation_vipe_units_by_frame': target_deviation.tolist(),
        'target_height_vipe_units_by_frame': target_heights.tolist(),
        'target_virtual_depth_vipe_units_by_frame': target_forward_z,
        'minimum_target_virtual_depth_vipe_units': min(target_forward_z),
        'source_rgb_mae_by_frame': rgb_differences,
        'source_pts_seconds': [i / 25 for i in range(FRAMES)],
        'render_pts_seconds': [i / 30 for i in range(FRAMES)],
    }
    return meta, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('source_exo', type=Path)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    annotation = json.loads(args.annotations.read_text())
    name = annotation['name']
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{1,63}', name):
        raise ValueError('Invalid clip name.')
    source_hash = hashlib.sha256(args.source_exo.read_bytes()).hexdigest()
    if annotation['video_sha256'] != source_hash or (annotation['width'], annotation['height']) != (WIDTH, HEIGHT):
        raise ValueError('Annotations do not match the exact EXO file and image grid.')
    rows = annotation['frames']
    if [row['frame'] for row in rows] != list(range(FRAMES)):
        raise ValueError('Expected one annotated EXO head pixel per frame, in display order.')
    anchors = np.array([pixel_xy((row['x'], row['y'])) for row in rows], dtype=int)
    target = pixel_xy(annotation['target_pixel'])
    intrinsics = read_geometry(args.artifacts)
    labels = read_labels(args.artifacts / 'mask/exo.txt')
    evidence = read_frame_evidence(args.artifacts, args.source_exo, anchors, target, intrinsics)
    meta, report = make_view(name, anchors, target, intrinsics, labels, *evidence)
    report['source_exo_sha256'] = source_hash
    args.output.mkdir()
    for filename, data in (('meta.json', meta), ('trajectory-check.json', report)):
        (args.output / filename).write_text(json.dumps(data, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
