"""Compare the approved Pair01 viewpoint with and without observed grass underfill."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT.parent))
from build_ball_previews import (COUNTER_ROLL, EVIDENCE, FRAMES, HEIGHT, SIZE, WIDTH,
                                 backproject, cv2, decode, exclude_upper_body, inputs,
                                 load_depth, load_json, np, projection, rasterize, unit)
from PIL import Image, ImageDraw, ImageFont


def ground_reference(observations, ball_track, planes):
    colors = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    support = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for (exo, z, labels), (x, y, radius) in zip(observations, ball_track, strict=True):
        moving = cv2.dilate((labels != 0).astype(np.uint8), np.ones((7, 7), np.uint8))
        cv2.circle(moving, (int(round(x)), int(round(y))), int(np.ceil(radius)) + 6, 1, -1)
        grass = (moving == 0) & (np.abs(z * planes[0] + planes[1]) <= 0.12)
        colors[grass] += exo[grass]
        support[grass] += 1
    average = colors / np.maximum(support[..., None], 1)
    return np.rint(average).astype(np.uint8), (support >= 6).astype(np.uint8)


def screen_rays(focal):
    y, x = np.mgrid[:SIZE, :SIZE]
    x, y = (x - SIZE / 2) / focal, (y - SIZE / 2) / focal
    theta = np.hypot(x, y)
    scale = np.divide(np.sin(theta), theta, out=np.ones_like(theta), where=theta > 0)
    return np.stack((x * scale, y * scale, np.cos(theta)), axis=2)


def project_ground(eye, rotation, local_rays, up, datum, k, texture, support):
    world = local_rays @ rotation
    pitch = world @ up
    eye_height = eye @ up + datum
    distance = np.divide(-eye_height, pitch, out=np.zeros_like(pitch), where=pitch < -1e-7)
    points = eye + distance[..., None] * world
    z = points[:, :, 2]
    intrinsic_z = np.maximum(z, 1e-7)
    fx, fy, cx, cy = k
    px = (fx * points[:, :, 0] / intrinsic_z + cx).astype(np.float32)
    py = (fy * points[:, :, 1] / intrinsic_z + cy).astype(np.float32)
    safe = (distance > 0) & (distance <= eye_height * 4) & (z > 0)
    # An interpolated color needs support at all four surrounding EXO pixels.
    found = cv2.remap(support.astype(np.float32), px, py, cv2.INTER_LINEAR) >= 0.999
    grass = cv2.remap(texture, px, py, cv2.INTER_LINEAR)
    return grass, safe & found


def foreground_guard(points, eye, rotation, focal):
    local = (points - eye) @ rotation.T
    uv = np.rint(projection(local[local[:, 2] > 0.20], focal)).astype(int)
    uv = uv[((uv >= 0) & (uv < SIZE)).all(1)]
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[uv[:, 1], uv[:, 0]] = 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool)


def banner():
    image = Image.new('RGB', (SIZE * 2, 48), '#171e29')
    font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 22)
    draw = ImageDraw.Draw(image)
    for offset, label in ((0, '原低头停球 · CPU 投影'), (SIZE, '以已观测草地补投影空洞')):
        draw.text((offset + 12, 10), label, font=font, fill='white')
    return np.asarray(image)[..., ::-1].copy()


def main():
    take = 'pair01_s0288'
    path, geometry, trajectory, _ = inputs(take)
    clip = path / f'example/self_capture/videos/{take}/exo.mp4'
    chosen_pose = OUTPUT.parent / f'{take}-ball-view-meta.json'
    proof = next(row for row in load_json(EVIDENCE / 'validated-inputs.json') if row['name'] == take)
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == proof['exo_sha256']
    sequence = next(row for row in load_json(OUTPUT.parent / 'preview-check.json')['takes'] if row['take'] == take)
    track = np.asarray(sequence['ball_pixel_by_frame'])
    ball_xy = np.rint(sequence['nominal_ball_projection_by_frame']).astype(int)
    camera = load_json(chosen_pose)['test_datasets'][0]
    angles = np.asarray(camera['ego_extrinsics'])
    rotations = COUNTER_ROLL.T @ angles[:, :, :3]
    centers = -(angles[:, :, :3].transpose(0, 2, 1) @ angles[:, :, 3:]).squeeze(2)
    reference = load_json(trajectory)
    assert np.allclose(centers, reference['head_centers_vipe_units'], atol=1e-8)
    up = unit(np.asarray(reference['ground_up']))
    datum = reference['fitted_exo_camera_height_vipe_units']
    actor = reference['head_instance_id']
    k = np.asarray(camera['camera_intrinsics'])[([0, 1, 0, 1], [0, 1, 2, 2])]
    focal = camera['ego_intrinsics'][0][0]
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx.flatten(), yy.flatten(), np.ones(HEIGHT * WIDTH), k)
    exo_plane = (rays @ up).reshape(HEIGHT, WIDTH)
    local_rays = screen_rays(focal)
    capture = cv2.VideoCapture(str(clip))
    observations = []
    with ZipFile(geometry / 'depth/exo.zip') as depth_file, ZipFile(geometry / 'mask/exo.zip') as label_file:
        for f in range(FRAMES):
            ok, rgb = capture.read()
            assert ok and rgb.shape == (HEIGHT, WIDTH, 3)
            observations.append((rgb, load_depth(depth_file, f), decode(label_file.read(f'{f:05}.png'))))
    capture.release()
    texture, support = ground_reference(observations, track, (exo_plane, datum))
    heading = banner()
    target = OUTPUT / 'pair01-leg-comparison.mp4'
    receipt = OUTPUT / 'verification.json'
    if target.exists() or receipt.exists():
        raise FileExistsError(target)
    ffmpeg = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pixel_format', 'bgr24',
              '-video_size', f'{SIZE * 2}x{SIZE + heading.shape[0]}', '-framerate', '25', '-i', '-',
              '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
              '-video_track_timescale', '25000', '-movflags', '+faststart', str(target)]
    filled_counts = []
    with subprocess.Popen(ffmpeg, stdin=subprocess.PIPE) as encoder:
        for idx, ((rgb, depth, labels), eye, rotation) in enumerate(zip(observations, centers, rotations, strict=True)):
            points = rays * depth.flatten()[:, None]
            eye_xy = eye[:2] / eye[2] * k[:2] + k[2:]
            allowed = ~exclude_upper_body(labels, actor, eye_xy).flatten()
            original, _ = rasterize(points[allowed], rgb.reshape(-1, 3)[allowed], eye, rotation, focal)
            px, py, size = track[idx]
            ball = (xx - px) ** 2 + (yy - py) ** 2 <= (size + 3) ** 2
            visible_dynamic = ((labels != 0) | ball).flatten() & allowed
            protected = foreground_guard(points[visible_dynamic], eye, rotation, focal)
            ground, found = project_ground(eye, rotation, local_rays, up, datum, k, texture, support)
            fill = (~original.any(axis=2)) & (~protected) & found
            filled_counts.append(int(fill.sum()))
            repaired = original.copy()
            repaired[fill] = ground[fill]
            assert np.array_equal(repaired[~fill], original[~fill])
            for frame in (original, repaired):
                cv2.drawMarker(frame, ball_xy[idx], (70, 233, 255), cv2.MARKER_CROSS, 13, 2)
            encoder.stdin.write(np.vstack((heading, np.hstack((original, repaired)))).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Failed to encode the comparison.')
    receipt.write_text(json.dumps({
        'take': take, 'frames': FRAMES, 'mode': 'CPU geometric comparison from EXO only',
        'approved_camera_sha256': hashlib.sha256(chosen_pose.read_bytes()).hexdigest(),
        'actual_EXO_eligible_grass_px': int(support.sum()),
        'new_ground_pixels_by_frame': filled_counts,
        'existing_nonblack_and_guarded_foreground_preserved': True,
    }, indent=2) + '\n')


if __name__ == '__main__':
    cv2.setNumThreads(2)
    main()
