import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path.cwd()
RUN = ROOT / 'runs/pair04_s0024-a800-20261121'
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN / 'pair05-foreground-match-20261122'))
import build_foreground_match as base

parser = argparse.ArgumentParser(description="Reproject Pair05 with fixed head centers on CPU.")
parser.add_argument("--timeline", action="store_true", help="Check teammate and ball at every GGA slot, at frozen yaw only.")
args = parser.parse_args()
FRAMES = tuple(range(0, 49, 4)) if args.timeline else base.FRAMES
ANGLES = (-15, -10, -5, 0, 5, 10, 15, 20)
if args.timeline:
    ANGLES = (0,)
TAKE = base.TAKE
path = RUN / f'ball-view-cpu-review/{TAKE}-ball-view-meta.json'
verification = json.loads((RUN / 'pair05-foreground-match-20261122/results/verification.json').read_text())
assert base.sha256(path) == verification['input_hashes']['meta']
meta, = json.loads(path.read_text())['test_datasets']
old_meta, = json.loads((RUN / f'multiclip-evidence/{TAKE}/teammate-view/meta.json').read_text())['test_datasets']
up = np.array(json.loads((RUN / f'multiclip-evidence/{TAKE}/viewpoint/trajectory-check.json').read_text())['ground_up'])
up /= np.linalg.norm(up)
exo = RUN / f'multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4'
geometry = RUN / f'multiclip-evidence/{TAKE}/geometry/vipe_results/{TAKE}'
mask_path = geometry / 'mask/exo.zip'
ball_path = RUN / 'update-008/ball-recovery/pair05-ball.zip'
exclude_path = RUN / 'ball-view-cpu-review/pair05-composition-review/pair05-wearer-upper-body.zip'
renderer = ROOT / 'records/official_renderer_geometry.py'
for identity, file in (('exo', exo), ('balls', ball_path), ('instances', mask_path), ('wearer_exclusion', exclude_path), ('renderer', renderer)):
    assert base.sha256(file) == verification['input_hashes'][identity]
distortion = np.concatenate([base.coefficients(renderer, f'{field}_distortion_coeffs')
                             for field in ('radial', 'tangential', 'thinPrism')])
report = json.loads((RUN / 'update-008/output-008/logs/validated-inputs.json').read_text())
receipt, = [row for row in report if row['name'] == TAKE]
images = base.review_frames(exo, base.WIDTH, FRAMES)
w2c = np.asarray(meta['camera_extrinsics'])
k = np.asarray(meta['ego_intrinsics'])
ys, xs = np.mgrid[:base.HEIGHT, :base.WIDTH]
ray = np.stack((xs, ys, np.ones_like(xs)), axis=-1) @ np.linalg.inv(meta['camera_intrinsics']).T
cross = np.cross(np.broadcast_to(up, (3, 3)), np.eye(3)).T
sy, sx = np.mgrid[:base.SIZE, :base.SIZE]
inside = ((sx - k[0, 2])**2 + (sy - k[1, 2])**2 <= (base.SIZE / 2)**2).ravel()

def horizontal(vector):
    vector = vector - (vector @ up) * up
    return vector / np.linalg.norm(vector)

def left_bearing(forward, direction):
    forward = horizontal(forward)
    direction = horizontal(direction)
    return round(float(np.degrees(np.arctan2(direction @ np.cross(up, forward), direction @ forward))), 2)

def pose_yaw(pose, yaw):
    angle = np.deg2rad(yaw)
    q = np.cos(angle) * np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * np.outer(up, up)
    center = np.linalg.solve(pose[:, :3], -pose[:, 3])
    rotation = pose[:, :3] @ q.T
    updated = np.column_stack((rotation, -rotation @ center))
    assert np.allclose(np.linalg.solve(rotation, -updated[:, 3]), center, atol=1e-12)
    assert abs(left_bearing(pose[2, :3], updated[2, :3]) - yaw) < 0.001
    return updated, center


def source_foreground(frame, rgb, masks, balls):
    instances = base.read_mask(masks, frame)
    ball = base.read_mask(balls, frame) == 255
    if frame in base.FRAMES:
        return base.foreground_labels(rgb, instances, ball)
    # Only the two detailed frames have validated disk labels.
    labels = np.zeros(instances.shape, dtype=np.int16)
    labels[instances == 2] = base.TEAMMATE
    labels[ball & (instances == 0)] = base.BALL
    return labels


results = {}
fig = Image.new('RGB', (5 * 224, 2 * (224 + 26)), '#162132')
draw = ImageDraw.Draw(fig)
with (ZipFile(mask_path) as masks, ZipFile(ball_path) as balls, ZipFile(exclude_path) as exclusions,
      ZipFile(geometry / 'depth/exo.zip') as depths):
    for index, frame in enumerate(FRAMES):
        rgb = np.asarray(images[frame])
        source_labels = source_foreground(frame, rgb, masks, balls)
        exclude = base.read_mask(exclusions, frame) == 255
        depth = base.load_depth(depths, frame)
        frozen_bytes = io.BytesIO()
        np.save(frozen_bytes, depth, allow_pickle=False)
        assert hashlib.sha256(frozen_bytes.getvalue()).hexdigest() == receipt['depth_sha256'][frame]
        scene = (ray * depth[..., None]).reshape(-1, 3)
        world = np.linalg.solve(w2c[:, :3], (scene - w2c[:, 3]).T).T
        frozen = np.asarray(meta['ego_extrinsics'][frame])
        prior = np.asarray(old_meta['ego_extrinsics'][frame])
        candidate = np.flatnonzero(~exclude.ravel())
        bearing_rows = {}
        take_results = {'old_teammate_camera_left_degrees': left_bearing(frozen[2, :3], prior[2, :3]), 'views': {}}
        for angle in ANGLES:
            pose, center = pose_yaw(frozen, angle)
            source, destination, z = base.project(world, pose, k, distortion, candidate)
            surface, _ = base.closest_surface(source, destination, z)
            hit = surface >= 0
            labels = np.zeros_like(surface, dtype=np.int16)
            labels[hit] = source_labels.ravel()[surface[hit]]
            if (labels == base.TEAMMATE).any():
                labels, _ = base.trusted_teammate(labels, surface, source_labels)
            labels[~inside] = 0
            row = {}
            names = base.NAMES if frame in base.FRAMES else {1: 'teammate', 2: 'ball'}
            for label, name in names.items():
                yy, xx = np.where(labels.reshape(base.SIZE, base.SIZE) == label)
                row[name] = [len(xx), [float(np.median(xx)), float(np.median(yy))] if len(xx) else None]
                if angle == 0 and frame in base.FRAMES:
                    prior_obs = verification['measurements'][f'F{frame:02}']['objects'][name]
                    assert len(xx) == prior_obs['target_visible_pixels']
                    if len(xx):
                        assert row[name][1] == prior_obs['center_xy']
                        samples = surface[labels == label]
                        bearing_rows[name] = left_bearing(frozen[2, :3], np.median(world[samples], axis=0) - center)
            row['nominal_circle_coverage'] = round(float((hit & inside).sum() / inside.sum()), 4)
            if frame in base.FRAMES:
                row['three_way_visible'] = bool(
                    row['teammate'][0] and row['ball'][0]
                    and any(row[name][0] for name in list(base.NAMES.values())[2:])
                )
            take_results['views'][angle] = row
            angles_plot = (-10, 0, 5, 10, 15)
            if not args.timeline and angle in angles_plot:
                tile = np.zeros((base.SIZE * base.SIZE, 3), dtype=np.uint8)
                tile[hit & inside] = rgb.reshape(-1, 3)[surface[hit & inside]]
                for label, color in base.COLORS.items():
                    relevant = labels == label
                    tile[relevant] = ((tile[relevant].astype(int) + np.asarray(color)) / 2).astype(np.uint8)
                tile = Image.fromarray(tile.reshape(base.SIZE, base.SIZE, 3)).resize((224, 224))
                x = 224 * angles_plot.index(angle)
                y = index * (224 + 26)
                draw.text((x + 8, y + 5), f'F{frame:02}  LEFT {angle:+} deg', fill='white')
                fig.paste(tile, (x, y+26))
        take_results['frozen_visible_object_left_bearing_degrees'] = bearing_rows
        results[f'F{frame:02}'] = take_results
if not args.timeline:
    with (OUTPUT / 'yaw.jpg').open('xb') as figure:
        fig.save(figure, format='JPEG', quality=91)
output_name = 'timeline.json' if args.timeline else 'yaw.json'
with (OUTPUT / output_name).open('x') as output:
    output.write(json.dumps(results, indent=2) + '\n')
for frame, entry in results.items():
    print(frame, 'teammate camera signed left', entry['old_teammate_camera_left_degrees'], 'visible rays signed left', entry['frozen_visible_object_left_bearing_degrees'])
    for angle, row in entry['views'].items():
        print(f'{angle:+d}', row)
