"""Reproject cached Pair04 geometry after aligning downward gaze with the EXO body heading."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[1]))

from build_ball_previews import (
    COUNTER_ROLL, EVIDENCE, FPS, FRAMES, HEIGHT, SIZE, WIDTH, backproject, decode,
    exclude_upper_body, inputs, load_depth, load_json, projection, rasterize, unit,
)
from build_multiclip_virtual_view import project, read_geometry
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TAKE = 'pair04_s0024'
SAMPLE_FRAMES = (0, 24, 48)
FRONT_DISTANCE_IN_HEAD_HEIGHTS = 0.65
HIGHLIGHT = (60, 228, 252)


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def probe(path, dimensions):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
               '-show_entries', 'stream=width,height,avg_frame_rate,time_base:frame=best_effort_timestamp',
               '-of', 'json', str(path)]
    report = json.loads(subprocess.check_output(command))
    stream, = report['streams']
    time_base = Fraction(stream['time_base'])
    pts = [int(frame['best_effort_timestamp']) * time_base for frame in report['frames']]
    if ((stream['width'], stream['height']) != dimensions or Fraction(stream['avg_frame_rate']) != FPS
            or pts != [Fraction(frame, FPS) for frame in range(FRAMES)]):
        raise ValueError(f'{path.name} is missing expected dimensions, frame count, or display timestamps.')
    return [float(value) for value in pts]


def load_exo(path):
    capture = cv2.VideoCapture(str(path))
    images = []
    try:
        for frame in range(FRAMES):
            ok, image = capture.read()
            if not ok or image.shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f'Staged EXO frame {frame} is missing or has the wrong shape.')
            images.append(image)
        if capture.read()[0]:
            raise ValueError('Staged EXO contains more than 49 images.')
    finally:
        capture.release()
    return images


def cone_heading(images, centers, intrinsics, up, ground_offset):
    # The stationary cyan cones are visible in a temporal median despite player occlusion.
    x0, y0, x1, y1 = 296, 247, 508, 286
    median = np.median([image[y0:y1, x0:x1] for image in images[::4]], axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, (74, 57, 100), (100, 255, 255))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cyan)
    contacts = []
    for label, (_, _, width, height, area) in enumerate(stats[1:], start=1):
        if 8 <= width <= 16 and 3 <= height <= 9 and 20 <= area <= 80:
            rows, cols = np.nonzero(labels == label)
            contacts.append([cols.mean() + x0, rows.max() + y0])
    pixels = np.asarray(sorted(contacts))
    if pixels.shape != (4, 2):
        raise ValueError('Expected four observed cone contact points in the staged Pair04 EXO.')
    rays = backproject(pixels[:, 0], pixels[:, 1], np.ones(len(pixels)), intrinsics)
    altitude = rays @ up
    if np.any(altitude >= 0):
        raise ValueError('A cone contact ray does not intersect the ground in front of EXO.')
    cones = rays * (-ground_offset / altitude)[:, None]
    centered_cones = cones - cones.mean(axis=0)
    _, _, axes = np.linalg.svd(centered_cones, full_matrices=False)
    # The player repeatedly faces and steps toward increasing EXO x in the clip.
    heading = axes[0] * np.sign(axes[0] @ (cones[-1] - cones[0]))
    heading = unit(heading - (heading @ up) * up)
    residuals = np.linalg.norm(centered_cones - (centered_cones @ heading)[:, None] * heading, axis=1)
    travel = centers[-1] - centers[0]
    travel = unit(travel - (travel @ up) * up)
    return heading, pixels, residuals, float(np.degrees(np.arccos(np.clip(travel @ heading, -1, 1))))


def make_view(centers, heading, up, ground_offset):
    heights = centers @ up + ground_offset
    distances = heights * FRONT_DISTANCE_IN_HEAD_HEIGHTS
    ground_under_head = centers - heights[:, None] * up
    targets = ground_under_head + distances[:, None] * heading
    forward = np.stack([unit(offset) for offset in targets - centers])
    right = unit(np.cross(heading, up))
    upright = np.stack((np.broadcast_to(right, forward.shape), np.cross(forward, right), forward), axis=1)
    official = COUNTER_ROLL @ upright
    extrinsics = np.concatenate((official, -(official @ centers[..., None])), axis=2)
    assert np.allclose(official @ official.transpose(0, 2, 1), np.eye(3), atol=1e-12, rtol=0)
    assert np.allclose(np.linalg.det(official), 1, atol=1e-12, rtol=0)
    assert np.allclose(targets @ up + ground_offset, 0, atol=1e-12, rtol=0)
    return upright, extrinsics, ground_under_head, targets, distances


def inverse_centers(extrinsics):
    rotation, translation = extrinsics[:, :, :3], extrinsics[:, :, 3:]
    return -(rotation.transpose(0, 2, 1) @ translation).squeeze(axis=2)


def check_centers(extrinsics, centers):
    error = float(np.max(np.abs(inverse_centers(extrinsics) - centers)))
    if error > 1e-10:
        raise ValueError('The virtual camera center differs from the sealed EXO-derived head position.')
    return error


def angle_ranges(axes, heading, up):
    sideways = unit(np.cross(up, heading))
    yaw = np.degrees(np.arctan2(axes @ sideways, axes @ heading))
    pitch_down = -np.degrees(np.arcsin(np.clip(axes @ up, -1, 1)))
    return {'yaw_relative_to_cones_degrees': [float(yaw.min()), float(yaw.max())],
            'pitch_down_degrees': [float(pitch_down.min()), float(pitch_down.max())]}


def header():
    strip = Image.new('RGB', (3 * SIZE, 48), '#151c26')
    draw = ImageDraw.Draw(strip)
    font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 22)
    labels = ('第三人称 · 正前方落点', '跟球版 · CPU 投影', '正前方俯视 · CPU 投影')
    for index, label in enumerate(labels):
        draw.text((index * SIZE + 14, 10), label, font=font, fill='white')
    return np.asarray(strip)[..., ::-1].copy()


def source_panel(image, frame, eye, origin, target):
    panel = image.copy()
    start, finish = tuple(np.rint(origin).astype(int)), tuple(np.rint(target).astype(int))
    cv2.arrowedLine(panel, start, finish, HIGHLIGHT, 2, cv2.LINE_AA, tipLength=0.2)
    cv2.circle(panel, tuple(np.rint(eye).astype(int)), 9, HIGHLIGHT, 2, cv2.LINE_AA)
    exo_panel = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    exo_panel[96:352] = cv2.resize(panel, (SIZE, 256), interpolation=cv2.INTER_AREA)
    cv2.putText(exo_panel, f'F{frame:02}  /  {frame / FPS:.2f}s', (13, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return exo_panel


def render_preview(path, images, geometry, centers, instance, intrinsics,
                   focal, previous, candidate, old_ball_px, ground_under_head, targets):
    bar = header()
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx.ravel(), yy.ravel(), np.ones(HEIGHT * WIDTH), intrinsics)
    head_px = project(centers, intrinsics)
    origin_px = project(ground_under_head, intrinsics)
    landing_px = project(targets, intrinsics)
    coverages, excluded_counts = [], []
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-video_size', f'{3 * SIZE}x{SIZE + len(bar)}', '-framerate', str(FPS), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-video_track_timescale', '25000', '-movflags', '+faststart', str(path)]
    with ZipFile(geometry / 'depth/exo.zip') as depths, ZipFile(geometry / 'mask/exo.zip') as masks:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for frame, bgr in enumerate(images):
                depth = load_depth(depths, frame)
                labels = decode(masks.read(f'{frame:05}.png'))
                if labels.shape != depth.shape or not np.issubdtype(labels.dtype, np.integer):
                    raise ValueError(f'Invalid wearer mask for frame {frame}.')
                excluded = exclude_upper_body(labels, instance, head_px[frame])
                excluded_counts.append(int(excluded.sum()))
                kept = ~excluded.ravel()
                points = rays[kept] * depth.ravel()[kept, None]
                colors = bgr.reshape(-1, 3)[kept]
                before, area_before = rasterize(points, colors, centers[frame], previous[frame], focal)
                after, area_after = rasterize(points, colors, centers[frame], candidate[frame], focal)
                coverages.append([area_before, area_after])
                cv2.drawMarker(before, tuple(np.rint(old_ball_px[frame]).astype(int)), HIGHLIGHT,
                               cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                cv2.drawMarker(after, (SIZE // 2, SIZE // 2), HIGHLIGHT,
                               cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                source = source_panel(bgr, frame, head_px[frame], origin_px[frame], landing_px[frame])
                surface = np.vstack((bar, np.hstack((source, before, after))))
                encoder.stdin.write(surface.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('CPU comparison encoding failed.')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
    return np.asarray(coverages), excluded_counts


def contact_sheet(preview, output):
    capture = cv2.VideoCapture(str(preview))
    frames = []
    try:
        for index in SAMPLE_FRAMES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot read encoded comparison frame {index}.')
            frames.append(frame)
    finally:
        capture.release()
    if not cv2.imwrite(str(output), np.vstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise OSError(f'Cannot write {output.name}.')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def build(output):
    cv2.setNumThreads(2)
    source, geometry, trajectory, default_instance = inputs(TAKE)
    exo = source / 'example/self_capture/videos' / TAKE / 'exo.mp4'
    sealed = load_json(trajectory)
    exo_sha = sha256(exo)
    if exo_sha != sealed['source_exo_sha256']:
        raise ValueError('EXO SHA-256 differs from the sealed source for this head track.')
    source_pts = probe(exo, (WIDTH, HEIGHT))
    images = load_exo(exo)
    intrinsics = read_geometry(geometry)
    original, = [row for row in load_json(EVIDENCE / 'meta.json')['test_datasets'] if f'/{TAKE}/' in row['exo_path']]
    failed, = load_json(SCRIPT.parents[1] / f'{TAKE}-ball-view-meta.json')['test_datasets']
    k = np.asarray(original['camera_intrinsics'])
    if not np.array_equal(intrinsics, (k[0, 0], k[1, 1], k[0, 2], k[1, 2])):
        raise ValueError('The sealed input metadata does not match ViPE K.')
    old_ball_px, = [row['nominal_ball_projection_by_frame']
                    for row in load_json(SCRIPT.parents[1] / 'preview-check.json')['takes']
                    if row['take'] == TAKE]
    centers = np.asarray(sealed['head_centers_vipe_units'])
    if centers.shape != (FRAMES, 3) or not np.isfinite(centers).all():
        raise ValueError('Expected exactly 49 finite sealed head positions.')
    original_pose, failed_pose = (np.asarray(row['ego_extrinsics']) for row in (original, failed))
    check_centers(original_pose, centers)
    check_centers(failed_pose, centers)
    up = unit(np.asarray(sealed['ground_up']))
    ground_offset = sealed['fitted_exo_camera_height_vipe_units']
    heading, cone_px, residuals, travel_error = cone_heading(images, centers, intrinsics, up, ground_offset)
    upright, pose, origin, targets, distances = make_view(centers, heading, up, ground_offset)
    candidate = dict(original, ego_extrinsics=pose.tolist())
    meta_path = output / 'pair04_s0024-front-view-meta.json'
    write_json(meta_path, {'test_datasets': [candidate]})
    serialized, = load_json(meta_path)['test_datasets']
    if {key: value for key, value in serialized.items() if key != 'ego_extrinsics'} != {
            key: value for key, value in original.items() if key != 'ego_extrinsics'}:
        raise ValueError('Metadata fields outside the target extrinsics changed.')
    center_error = check_centers(np.asarray(serialized['ego_extrinsics']), centers)
    focal = candidate['ego_intrinsics'][0][0]
    targets_local = (upright @ (targets - centers)[..., None]).squeeze(axis=2)
    projected = projection(targets_local, focal)
    if not ((targets_local[:, 2] > 0).all() and np.allclose(projected, SIZE / 2, atol=1e-10, rtol=0)):
        raise ValueError('Body-front ground target does not project to the optical center.')

    preview = output / 'pair04_s0024-front-view-comparison.mp4'
    instance = sealed.get('head_instance_id', default_instance)
    before = COUNTER_ROLL.T @ failed_pose[:, :, :3]
    area, excluded = render_preview(preview, images, geometry, centers, instance, intrinsics,
                                    focal, before, upright, old_ball_px, origin, targets)
    preview_pts = probe(preview, (3 * SIZE, SIZE + 48))
    sheet = output / 'pair04_s0024-front-view-contact.jpg'
    contact_sheet(preview, sheet)
    write_json(output / 'pair04_s0024-front-view-check.json', {
        'take': TAKE,
        'frames': FRAMES,
        'evidence': 'staged Pair04 EXO, sealed camera and head track, cached ViPE depth and masks',
        'exo_sha256': exo_sha,
        'sealed_head_track_sha256': sha256(trajectory),
        'cached_depth_sha256': sha256(geometry / 'depth/exo.zip'),
        'cached_mask_sha256': sha256(geometry / 'mask/exo.zip'),
        'head_centers_vipe_units': sealed['head_centers_vipe_units'],
        'max_serialized_camera_center_coordinate_error_vipe_units': center_error,
        'wearer_vipe_instance': instance,
        'cone_contact_pixels': cone_px.tolist(),
        'cone_fit_max_residual_vipe_units': float(residuals.max()),
        'horizontal_heading_exo_xyz': heading.tolist(),
        'cone_vs_head_travel_heading_angle_degrees': travel_error,
        'target_distance_in_head_heights': FRONT_DISTANCE_IN_HEAD_HEIGHTS,
        'ground_target_distance_range_vipe_units': [float(distances.min()), float(distances.max())],
        'ground_target_exo_xyz_by_frame': targets.tolist(),
        'ground_target_optical_center_error_px': float(np.max(abs(projected - SIZE / 2))),
        'original_teammate_view': angle_ranges(original_pose[:, 2, :3], heading, up),
        'previous_ball_view': angle_ranges(before[:, 2], heading, up),
        'front_down_view': angle_ranges(upright[:, 2], heading, up),
        'removed_upper_wearer_surface_pixels_by_frame': excluded,
        'usable_raster_fraction_mean_ball_and_front': area.mean(axis=0).tolist(),
        'usable_raster_fraction_min_ball_and_front': area.min(axis=0).tolist(),
        'source_pts_seconds': source_pts,
        'preview_pts_seconds': preview_pts,
        'future_renderer_pts_seconds': [frame / 30 for frame in range(FRAMES)],
        'preview_full_decode': 'passed',
        'contact_sheet_display_frames': list(SAMPLE_FRAMES),
        'artifact_sha256': {path.name: sha256(path) for path in (meta_path, preview, sheet)},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SCRIPT.parent / 'artifacts',
                        help='new output directory, created on successful verification')
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix='.front-view-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'artifacts'
        staged.mkdir()
        build(staged)
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)


if __name__ == '__main__':
    main()
