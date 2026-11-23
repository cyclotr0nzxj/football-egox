"""Compare two ball-following pitches at the same observed wearer head positions."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT.parent))
from build_ball_previews import (COUNTER_ROLL, EVIDENCE, FRAMES, HEIGHT, SIZE, WIDTH,
                                 backproject, corrected_orientations, cv2, decode, exclude_upper_body,
                                 inputs, load_depth, load_json, np, projection, rasterize, unit)
from PIL import Image, ImageDraw, ImageFont


def compose_banner():
    image = Image.new('RGB', (2 * SIZE, 42), '#171e29')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 21)
    for idx, label in enumerate(('原跟球预览', '球位靠下 · 镜头抬高 10°')):
        draw.text((idx * SIZE + 12, 8), label, font=font, fill='white')
    return np.asarray(image)[..., ::-1].copy()


def main():
    take = 'pair05_s0024'
    prior = AUDIT.parent
    video = AUDIT / 'pair05-pitch-comparison.mp4'
    mask_zip = AUDIT / 'pair05-wearer-upper-body.zip'
    camera_path = AUDIT / 'pair05-ball-low-meta.json'
    audit_path = AUDIT / 'verification.json'
    for path in (video, mask_zip, camera_path, audit_path):
        if path.exists():
            raise FileExistsError(path)

    base, geometry, trajectory, _ = inputs(take)
    source = base / 'example/self_capture/videos' / take / 'exo.mp4'
    view = load_json(prior / f'{take}-ball-view-meta.json')['test_datasets'][0]
    provenance = next(row for row in load_json(EVIDENCE / 'validated-inputs.json') if row['name'] == take)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['exo_sha256']
    report = load_json(trajectory)
    wearer = report['head_instance_id']
    centers = np.asarray(report['head_centers_vipe_units'])
    up = unit(np.asarray(report['ground_up']))
    k = np.asarray(view['camera_intrinsics'])[([0, 1, 0, 1], [0, 1, 2, 2])]
    anchors = load_json(prior / 'ball-anchors.json')[take]
    _, ball, current = corrected_orientations(anchors, centers, k, up, report['fitted_exo_camera_height_vipe_units'])
    staged = np.asarray(view['ego_extrinsics'])
    assert np.allclose(staged[:, :, :3], COUNTER_ROLL @ current, atol=1e-8)
    raise_by = np.deg2rad(10.0)
    right, down, forward = current.transpose(1, 0, 2)
    raised_view = forward * np.cos(raise_by) - down * np.sin(raise_by)
    candidate = np.stack((right, np.cross(raised_view, right), raised_view), axis=1)
    rotations = np.stack((current, candidate))
    official_candidate = COUNTER_ROLL @ candidate
    candidate_extrinsics = np.concatenate((official_candidate, -(official_candidate @ centers[:, :, None])), axis=2)
    recovered_centers = -(candidate_extrinsics[:, :, :3].transpose(0, 2, 1) @ candidate_extrinsics[:, :, 3:])
    assert np.allclose(recovered_centers[:, :, 0], centers, atol=1e-8)
    assert np.allclose(candidate @ candidate.transpose(0, 2, 1), np.eye(3), atol=1e-8)
    assert np.allclose(np.linalg.det(candidate), 1)

    focal = view['ego_intrinsics'][0][0]
    ball_relative = (rotations @ (ball - centers)[:, :, None]).squeeze(3)
    mapped_ball = np.asarray([projection(arm, focal) for arm in ball_relative])
    assert (ball_relative[..., 2] > 0).all() and ((mapped_ball > 18) & (mapped_ball < SIZE - 18)).all()
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx.flatten(), yy.flatten(), np.ones(HEIGHT * WIDTH), k)
    banner = compose_banner()
    observed_coverage, upper_coverage, exclude_counts, keep_leg_counts = [], [], [], []
    capture = cv2.VideoCapture(str(source))
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pixel_format', 'bgr24',
               '-video_size', f'{2 * SIZE}x{SIZE + banner.shape[0]}', '-framerate', '25', '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-video_track_timescale', '25000', '-movflags', '+faststart', str(video)]
    with ZipFile(geometry / 'depth/exo.zip') as depths, ZipFile(geometry / 'mask/exo.zip') as masks, \
            ZipFile(mask_zip, 'x', compression=ZIP_DEFLATED) as exported:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for idx, eye in enumerate(centers):
                valid, frame = capture.read()
                assert valid and frame.shape == (HEIGHT, WIDTH, 3)
                depth = load_depth(depths, idx)
                labels = decode(masks.read(f'{idx:05}.png'))
                head = (eye[:2] / eye[2]) * k[:2] + k[2:]
                excluded = exclude_upper_body(labels, wearer, head)
                retained = (labels == wearer) & ~excluded
                assert (labels[excluded] == wearer).all() and retained.any()
                exclude_counts.append(int(excluded.sum()))
                keep_leg_counts.append(int(retained.sum()))
                ok, png = cv2.imencode('.png', excluded.astype(np.uint8) * 255)
                assert ok
                exported.writestr(f'{idx:05}.png', png.tobytes())
                points = rays[~excluded.flatten()] * depth.flatten()[~excluded.flatten(), None]
                colors = frame.reshape(-1, 3)[~excluded.flatten()]
                render, coverage, upper = [], [], []
                for arm, rotation in enumerate(rotations[:, idx]):
                    image, occupied = rasterize(points, colors, eye, rotation, focal)
                    coverage.append(occupied)
                    upper.append(float(np.mean(np.any(image[:SIZE // 2] != 0, axis=2))))
                    xy = np.rint(mapped_ball[arm, idx]).astype(int)
                    cv2.drawMarker(image, xy, (70, 233, 255), cv2.MARKER_CROSS, 13, 2)
                    render.append(image)
                observed_coverage.append(coverage)
                upper_coverage.append(upper)
                encoder.stdin.write(np.vstack((banner, np.hstack(render))).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Comparison encoding failed.')
    capture.release()

    view['ego_extrinsics'] = candidate_extrinsics.tolist()
    camera_path.write_text(json.dumps({'test_datasets': [view]}, indent=2) + '\n')
    slopes = -np.degrees(np.arcsin(np.clip(rotations[:, :, 2] @ up, -1, 1)))
    audit_path.write_text(json.dumps({
        'mode': 'CPU reprojection using only EXO data, no diffusion', 'frames': FRAMES,
        'head_centers_max_abs_difference': float(np.max(np.abs(recovered_centers[:, :, 0] - centers))),
        'pitch_down_degrees_before_after': np.column_stack((slopes.min(axis=1), slopes.max(axis=1))).tolist(),
        'ball_screen_y_range_before_after': np.column_stack((mapped_ball[:, :, 1].min(axis=1), mapped_ball[:, :, 1].max(axis=1))).tolist(),
        'actual_projection_coverage_before_after': np.mean(observed_coverage, axis=0).tolist(),
        'top_half_nonblack_before_after': np.mean(upper_coverage, axis=0).tolist(),
        'wearer_instance': wearer, 'excluded_upper_body_px_per_frame': exclude_counts,
        'retained_lower_body_px_per_frame': keep_leg_counts,
    }, indent=2) + '\n')


if __name__ == '__main__':
    cv2.setNumThreads(2)
    main()
