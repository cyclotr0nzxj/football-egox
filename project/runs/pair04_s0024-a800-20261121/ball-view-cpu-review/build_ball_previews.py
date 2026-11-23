"""Reproject observed EXO pixels to compare two view directions at the same head positions."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path(__file__).resolve().parent
RUN = OUTPUT.parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN))
from build_multiclip_virtual_view import COUNTER_ROLL, backproject, smooth_track, unit

EVIDENCE = RUN / 'multiclip-evidence'
TAKES = ('pair04_s0024', 'pair01_s0288', 'pair05_s0024')
FRAMES = 49
HEIGHT, WIDTH = 448, 784
SIZE = 448
FPS = 25
# These ZIPs have one half-float Z channel. OpenCV's grayscale path reads Y.
EXR_Z_CHANNEL = b'Z\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00'


def load_json(path):
    return json.loads(path.read_text())


def decode(data):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def load_depth(archive, frame):
    raw = archive.read(f'{frame:05}.exr')
    if raw.count(EXR_Z_CHANNEL) != 1:
        raise ValueError('Expected exactly one ViPE Z channel.')
    depth = decode(raw.replace(EXR_Z_CHANNEL, b'Y' + EXR_Z_CHANNEL[1:], 1))
    if depth.shape != (HEIGHT, WIDTH) or not np.isfinite(depth).all() or np.min(depth) <= 0:
        raise ValueError(f'Invalid observed ViPE depth at frame {frame}.')
    return depth


def inputs(take):
    if take == 'pair04_s0024':
        pair04 = RUN / 'geometry-008'
        return (ROOT / 'data/self_capture/pair04_s0024/egox_input', pair04 / take,
                pair04 / 'viewpoint-008-upright/trajectory-check.json', 2)
    evidence = EVIDENCE / take
    geometry = evidence / 'geometry/vipe_results' / take
    return RUN / 'multiclip-prep' / take, geometry, evidence / 'viewpoint/trajectory-check.json', None


def corrected_orientations(anchors, centers, k, up, plane_offset):
    frames, x, y, radius = np.asarray([entry[:4] for entry in anchors], dtype=float).T
    if not np.array_equal(frames, np.arange(0, FRAMES, 4)):
        raise ValueError('Ball landmarks must span the sealed 49-frame window.')
    pixels = np.column_stack([np.interp(np.arange(FRAMES), frames, values) for values in (x, y, radius)])
    x, y, r = pixels.T
    rays = backproject(x, y, np.ones(FRAMES), k)
    # For gaze only, use the rolling-ball plane at approximately one image-derived radius.
    divisor = rays @ up - r / np.sqrt(k[0] * k[1])
    if np.max(divisor) >= 0:
        raise ValueError('Ball rays must hit the ground in front of the EXO camera.')
    ball = smooth_track(rays * (-plane_offset / divisor)[:, None])
    gaze = np.asarray([unit(target - eye) for eye, target in zip(centers, ball, strict=True)])
    initial_heading = unit(gaze[0] - (gaze[0] @ up) * up)
    side = unit(np.cross(up, initial_heading))
    desired_yaw = np.unwrap(np.arctan2(gaze @ side, gaze @ initial_heading))
    # A ball passing underneath the wearer can make the raw horizontal aim flip.
    # Limit the per-frame yaw in both directions, since this is an offline preview.
    forward_yaw, reverse_yaw = desired_yaw.copy(), desired_yaw.copy()
    step = np.deg2rad(6)
    for index in range(1, FRAMES):
        before = forward_yaw[index - 1]
        forward_yaw[index] = np.clip(forward_yaw[index], before - step, before + step)
        backward = FRAMES - 1 - index
        after = reverse_yaw[backward + 1]
        reverse_yaw[backward] = np.clip(reverse_yaw[backward], after - step, after + step)
    yaw = (forward_yaw + reverse_yaw) / 2
    heading = np.cos(yaw[:, None]) * initial_heading + np.sin(yaw[:, None]) * side
    pitch = np.arctan2(gaze @ up, np.sum(gaze * heading, axis=1)) + np.deg2rad(8)
    pitch = smooth_track(smooth_track(np.maximum(pitch, np.deg2rad(-82))[:, None]))
    forward = np.cos(pitch) * heading + np.sin(pitch) * up
    right = np.asarray([unit(np.cross(direction, up)) for direction in forward])
    return pixels, ball, np.stack((right, np.cross(forward, right), forward), axis=1)


def projection(points, focal):
    x, y, z = points.T
    radial = np.hypot(x, y)
    angle = np.arctan2(radial, z)
    scale = np.divide(focal * angle, radial, out=np.zeros_like(radial), where=radial > 1e-8)
    return np.column_stack((x * scale + SIZE / 2, y * scale + SIZE / 2))


def exclude_upper_body(mask, instance, head_xy):
    rows, cols = np.where(mask == instance)
    x, y = head_xy
    supported = (np.abs(cols - x) < 23) & (np.abs(rows - y) < 23)
    if supported.sum() < 30:
        raise ValueError('The wearer mask does not support the sealed head center.')
    cutoff = rows.min() + 0.60 * (rows.max() - rows.min())
    return (mask == instance) & (np.arange(HEIGHT)[:, None] < cutoff)


def rasterize(points, colors, center, rotation, focal):
    local = (points - center) @ rotation.T
    forward = local[:, 2] > 0.20
    local, colors = local[forward], colors[forward]
    uv = np.rint(projection(local, focal)).astype(np.int32)
    valid = ((uv >= 0) & (uv < SIZE)).all(1)
    uv, local, colors = uv[valid], local[valid], colors[valid]
    screen_indices = uv[:, 1] * SIZE + uv[:, 0]
    ranges = np.linalg.norm(local, axis=1)
    nearest = np.full(SIZE * SIZE, np.inf)
    np.minimum.at(nearest, screen_indices, ranges)
    hits = ranges == nearest[screen_indices]
    pixels = np.zeros((SIZE * SIZE, 3), dtype=np.uint8)
    pixels[screen_indices[hits]] = colors[hits]
    written = np.isfinite(nearest).reshape(SIZE, SIZE)
    pixels = pixels.reshape(SIZE, SIZE, 3)
    # Close isolated sampling cracks with a single nearest-range neighboring sample.
    gap = np.where(written, nearest.reshape(SIZE, SIZE), np.inf)
    shown = pixels.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = np.roll(nearest.reshape(SIZE, SIZE), (dy, dx), axis=(0, 1))
        use = (~written) & (neighbor < gap)
        if dy:
            use[0 if dy == 1 else -1, :] = False
        if dx:
            use[:, 0 if dx == 1 else -1] = False
        shown[use] = np.roll(pixels, (dy, dx), axis=(0, 1))[use]
        gap[use] = neighbor[use]
    return shown, float(np.mean(np.isfinite(gap)))


def header():
    bar = Image.new('RGB', (SIZE * 3, 48), '#171e29')
    draw = ImageDraw.Draw(bar)
    font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 22)
    for position, label in enumerate(('第三人称 · 球位置', '原朝向 · CPU 投影', '跟球朝向 · CPU 投影')):
        draw.text((position * SIZE + 12, 10), label, font=font, fill='white')
    return np.asarray(bar)[..., ::-1].copy()


def build(take, anchors, previous, receipt, bar):
    source, geometry, trajectory, fallback_id = inputs(take)
    exo = source / 'example/self_capture/videos' / take / 'exo.mp4'
    if hashlib.sha256(exo.read_bytes()).hexdigest() != receipt['exo_sha256']:
        raise ValueError(f'{take} EXO differs from sealed input.')
    report = load_json(trajectory)
    instance = report.get('head_instance_id', fallback_id)
    camera = previous.copy()
    centers = np.asarray(report['head_centers_vipe_units'])
    old_extrinsics = np.asarray(camera['ego_extrinsics'])
    if not np.allclose(old_extrinsics[:, :, :3] @ centers[:, :, None], -old_extrinsics[:, :, 3:], atol=1e-8):
        raise ValueError(f'{take} original and proposed camera positions do not match.')
    old = COUNTER_ROLL.T @ old_extrinsics[:, :, :3]
    k = np.asarray(camera['camera_intrinsics'])
    k = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
    up = unit(np.asarray(report['ground_up']))
    landmarks, ball, revised = corrected_orientations(anchors, centers, k, up, report['fitted_exo_camera_height_vipe_units'])
    focal = camera['ego_intrinsics'][0][0]
    relative = (revised @ (ball - centers)[:, :, None]).squeeze(2)
    aim_px = projection(relative, focal)
    if not (relative[:, 2] > 0).all() or not ((aim_px > 20) & (aim_px < SIZE - 20)).all():
        raise ValueError(f'{take} ball aim leaves the proved virtual image bounds.')
    official = COUNTER_ROLL @ revised
    camera['ego_extrinsics'] = np.concatenate((official, -(official @ centers[:, :, None])), axis=2).tolist()
    assert np.allclose(np.linalg.det(official), 1) and np.allclose(official @ official.transpose(0, 2, 1), np.eye(3))

    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx.flatten(), yy.flatten(), np.ones(HEIGHT * WIDTH), k)
    visible_pixels, removed_pixels = [], []
    video = OUTPUT / f'{take}-aim-preview.mp4'
    if video.exists():
        raise FileExistsError(video)
    capture = cv2.VideoCapture(str(exo))
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'bgr24',
               '-video_size', f'{SIZE * 3}x{SIZE + bar.shape[0]}', '-framerate', str(FPS), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-video_track_timescale', '25000', '-movflags', '+faststart', str(video)]
    with ZipFile(geometry / 'depth/exo.zip') as depth_file, ZipFile(geometry / 'mask/exo.zip') as mask_file:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for frame in range(FRAMES):
                ok, rgb = capture.read()
                if not ok or rgb.shape != (HEIGHT, WIDTH, 3):
                    raise ValueError(f'{take} input frame {frame} is missing.')
                depth = load_depth(depth_file, frame)
                labels = decode(mask_file.read(f'{frame:05}.png'))
                eye = centers[frame]
                head_xy = [k[0] * eye[0] / eye[2] + k[2], k[1] * eye[1] / eye[2] + k[3]]
                excluded = exclude_upper_body(labels, instance, head_xy)
                removed_pixels.append(int(excluded.sum()))
                keep = ~excluded.flatten()
                points = rays[keep] * depth.flatten()[keep, None]
                colors = rgb.reshape(-1, 3)[keep]
                first, coverage_before = rasterize(points, colors, eye, old[frame], focal)
                second, coverage_after = rasterize(points, colors, eye, revised[frame], focal)
                visible_pixels.append([coverage_before, coverage_after])
                annotation = rgb.copy()
                px, py, radius = np.rint(landmarks[frame]).astype(int)
                cv2.circle(annotation, (px, py), radius + 8, (70, 233, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(second, np.rint(aim_px[frame]).astype(int), (70, 233, 255), cv2.MARKER_CROSS, 13, 2)
                scaled = cv2.resize(annotation, (SIZE, 256), interpolation=cv2.INTER_AREA)
                source_view = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                source_view[96:352] = scaled
                surface = np.vstack((bar, np.hstack((source_view, first, second))))
                encoder.stdin.write(surface.tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Preview encoding failed.')
    capture.release()

    pitch = np.degrees(np.arcsin(np.clip(revised[:, 2] @ up, -1, 1)))
    old_pitch = np.degrees(np.arcsin(np.clip(old[:, 2] @ up, -1, 1)))
    step_cosine = (np.einsum('ijk,ijk->i', revised[:-1], revised[1:]) - 1) / 2
    step = np.degrees(np.arccos(np.clip(step_cosine, -1, 1)))
    (OUTPUT / f'{take}-ball-view-meta.json').write_text(json.dumps({'test_datasets': [camera]}, indent=2) + '\n')
    return {'take': take, 'frames': FRAMES, 'wearer_vipe_instance_id': instance, 'camera_centers_unchanged': True,
            'old_pitch_degrees': [float(old_pitch.min()), float(old_pitch.max())],
            'ball_follow_pitch_degrees': [float(pitch.min()), float(pitch.max())],
            'max_step_degrees': float(step.max()), 'upper_body_pixels_excluded': removed_pixels,
            'projected_coverage_before_and_after': np.mean(visible_pixels, axis=0).tolist(),
            'ball_pixel_by_frame': landmarks.tolist(), 'nominal_ball_projection_by_frame': aim_px.tolist()}


def main():
    cv2.setNumThreads(2)
    old_cameras = {row['exo_path'].split('/')[-2]: row for row in load_json(EVIDENCE / 'meta.json')['test_datasets']}
    receipts = {row['name']: row for row in load_json(EVIDENCE / 'validated-inputs.json')}
    anchors = load_json(OUTPUT / 'ball-anchors.json')
    bar = header()
    metrics = []
    for take in TAKES:
        row = build(take, anchors[take], old_cameras[take], receipts[take], bar)
        metrics.append(row)
        print(take, 'pitch old/revised', row['old_pitch_degrees'], row['ball_follow_pitch_degrees'],
              'max direction step', row['max_step_degrees'], flush=True)
    (OUTPUT / 'preview-check.json').write_text(json.dumps({'cpu_diagnostic_only': True, 'takes': metrics}, indent=2) + '\n')


if __name__ == '__main__':
    main()
