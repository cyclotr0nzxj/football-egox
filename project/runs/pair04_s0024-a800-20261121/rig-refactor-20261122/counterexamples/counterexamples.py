from pathlib import Path
from io import BytesIO
import hashlib
import json
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

root = Path.cwd()
run = root / 'runs/pair04_s0024-a800-20261121'
sys.path.insert(0, str(run / 'pair05-foreground-match-20261122'))
import build_foreground_match as base

output = Path(sys.argv[1])

def read_json(path):
    return json.loads(path.read_text())

def verify(path, expected_sha):
    if base.sha256(path) != expected_sha:
        raise ValueError(f'Hash mismatch: {path.name}')

learned = read_json(run / 'pair01-pose-audit-20261122/calibration/field-calibration.json')
ex01 = run / 'multiclip-prep/pair01_s0288/example/self_capture/videos/pair01_s0288/exo.mp4'
verify(ex01, learned['exo_sha256'])
base.check_reference(ex01, 784, 25)
rgb01 = np.asarray(base.review_frames(ex01, 784, [28])[28])
old_ball = next(row for row in read_json(run / 'ball-view-cpu-review/ball-anchors.json')['pair01_s0288'] if row[0] == 28)
height = learned['camera_height_reference_m']
ball_radius = 0.11
toe = (597, 283)
toe_rgb = rgb01[toe[1], toe[0]].tolist()
assert toe_rgb[0] > toe_rgb[1] + 50 and toe_rgb[0] > toe_rgb[2] + 50


def contact_case(calibration):
    focal = calibration['focal_px']
    up = np.asarray(calibration['ground_up'])

    def projected_ray(pixel):
        return np.array([(pixel[0] - 392) / focal, (pixel[1] - 224) / focal, 1.0])

    def pixel_at(point):
        return np.array([focal * point[0] / point[2] + 392, focal * point[1] / point[2] + 224])

    ball_ray = projected_ray(old_ball[1:3])
    ball_center = ball_ray * (ball_radius - height) / (up @ ball_ray)
    toe_ray = projected_ray(toe)
    toe_ray /= np.linalg.norm(toe_ray)
    closest = ball_center @ toe_ray
    ray_dist = np.linalg.norm(ball_center - closest * toe_ray)
    assert ray_dist < ball_radius
    near_surface = closest - np.sqrt(ball_radius**2 - ray_dist**2)
    toe_points = np.array([near_surface, near_surface - 0.16])[:, None] * toe_ray
    repro = np.stack([pixel_at(point) for point in toe_points])
    assert np.max(np.abs(repro - toe)) < 1e-9
    clearance = np.linalg.norm(toe_points - ball_center, axis=1) - ball_radius
    assert abs(clearance[0]) < 1e-10 and clearance[1] > 0.10
    return {
        'r8_equal_shoe_reprojection_px_xy': repro.tolist(),
        'approx_ball_center_m': ball_center.tolist(),
        'surface_clearance_m': clearance.tolist(),
        'shoe_point_height_above_ground_m': (toe_points @ up + height).tolist(),
    }


sensitivity = [contact_case(variant) for variant in learned['sensitivity_variants']]
summary = {
    'Pair01_F28': {
        'scope': 'Local shoe-point depth counterexample, not whole-frame or rigid shoe reconstruction. Nominal field calibration and approximate ball radius are assumptions.',
        'r8_ball_pixel_anchor': old_ball[1:3], 'r8_red_edge_pixel': list(toe), 'r8_red_edge_rgb': toe_rgb,
        'approx_ball_radius_m': ball_radius, 'shoe_point_perturbation_along_exo_ray_m': 0.16,
        **contact_case(learned['nominal']),
        'calibration_sweep_cases': len(sensitivity),
        'calibration_sweep_noncontact_gap_minmax_m': [
            reducer(item['surface_clearance_m'][1] for item in sensitivity) for reducer in (min, max)
        ],
    },
    'Pair05': {},
}

exo = run / 'multiclip-prep/pair05_s0024/example/self_capture/videos/pair05_s0024/exo.mp4'
prior = run / 'update-008/output-008/priors/pair05_s0024-final.mp4'
pose = run / 'ball-view-cpu-review/pair05_s0024-ball-view-meta.json'
geometry = run / 'multiclip-evidence/pair05_s0024/geometry/vipe_results/pair05_s0024'
excluded = run / 'ball-view-cpu-review/pair05-composition-review/pair05-wearer-upper-body.zip'
renderer = root / 'records/official_renderer_geometry.py'
old_ego = run / 'clone-004-20261122/pair05/verified/pair05_s0024_ego.mp4'
match_root = run / 'pair05-foreground-match-20261122/results'
hashes = read_json(match_root / 'verification.json')['input_hashes']
for path, key in ((exo, 'exo'), (prior, 'prior'), (pose, 'meta'), (excluded, 'wearer_exclusion'), (renderer, 'renderer')):
    verify(path, hashes[key])
plan_hash = read_json(run / 'clone-004-20261122/pair05/verified/verification.json')['verified_export']['sha256']
verify(old_ego, plan_hash)
for path, width, fps in ((exo, 784, 25), (prior, 448, 30), (old_ego, 448, 30)):
    base.check_reference(path, width, fps)
frames = (12, 24)
exo_images = base.review_frames(exo, 784, frames)
prior_images = base.review_frames(prior, 448, frames)
old_images = base.review_frames(old_ego, 448, frames)
meta, = read_json(pose)['test_datasets']
receipt, = [item for item in read_json(run / 'update-008/output-008/logs/validated-inputs.json') if item['name'] == base.TAKE]
distortion = np.concatenate([base.coefficients(renderer, f'{name}_distortion_coeffs')
                             for name in ('radial', 'tangential', 'thinPrism')])
y, x = np.mgrid[:448, :784]
ray = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(meta['camera_intrinsics']).T
w2c = np.asarray(meta['camera_extrinsics'])
k = np.asarray(meta['ego_intrinsics'])
y, x = np.mgrid[:448, :448]
in_circle = (x - k[0, 2]) ** 2 + (y - k[1, 2]) ** 2 <= 224**2
central_upper_region = in_circle & (x >= 112) & (x < 336) & (y >= 32) & (y < 224)
columns = ('Same-frame EXO projection', 'Foreground lock + old result', 'All-support lock + old result', 'All-support lock + prior')
thumb, pad, label = 320, 12, 22
panel = Image.new('RGB', (4 * thumb + 5 * pad, label + (thumb + pad) * 2 + pad), '#151f2a')
text = ImageDraw.Draw(panel)
with (ZipFile(geometry / 'depth/exo.zip') as depths, ZipFile(excluded) as exclusions,
      np.load(match_root / 'foreground-routing.npz', allow_pickle=False) as routing):
    for row, frame in enumerate(frames):
        prefix = f'F{frame:02}'
        rgb = np.asarray(exo_images[frame]).reshape(-1, 3)
        old_rgb = np.asarray(old_images[frame]).copy()
        prior_rgb = np.asarray(prior_images[frame]).copy()
        model_inputs = base.load_depth(depths, frame)
        encoded = BytesIO()
        np.save(encoded, model_inputs, allow_pickle=False)
        assert hashlib.sha256(encoded.getvalue()).hexdigest() == receipt['depth_sha256'][frame]
        camera_points = (ray * model_inputs[..., None]).reshape(-1, 3)
        world = np.linalg.solve(w2c[:, :3], (camera_points - w2c[:, 3]).T).T
        candidates = np.flatnonzero(base.read_mask(exclusions, frame).ravel() == 0)
        src, dst, depth = base.project(world, np.asarray(meta['ego_extrinsics'][frame]), k, distortion, candidates)
        source_map, _ = base.closest_surface(src, dst, depth)
        source_map = source_map.reshape(448, 448)
        known = (source_map >= 0) & in_circle
        fg_map = routing[f'{prefix}_source_index']
        fg = fg_map >= 0
        assert (fg <= known).all() and np.array_equal(source_map[fg], fg_map[fg])
        observed = np.zeros_like(old_rgb)
        observed[known] = rgb[source_map[known]]
        foreground_only = old_rgb.copy()
        foreground_only[fg] = observed[fg]
        all_support = old_rgb.copy()
        all_support[known] = observed[known]
        observed_and_prior = prior_rgb.copy()
        observed_and_prior[known] = observed[known]
        assert np.max(np.abs(all_support[known].astype(float) - observed[known])) == 0
        assert np.array_equal(all_support[known], observed_and_prior[known])
        assert np.array_equal(all_support[fg], foreground_only[fg])
        unknown = in_circle & ~known
        delta = np.abs(all_support.astype(float) - observed_and_prior.astype(float)).mean(axis=2)
        body = routing[f'{prefix}_target_label'] == base.TEAMMATE
        body_y, body_x = np.nonzero(body)
        # A translated second silhouette can occupy untouched unknown pixels.
        candidates = []
        for dy in (-96, -80, -64):
            for dx in (-96, -80, -64, 64, 80, 96):
                yy, xx = body_y + dy, body_x + dx
                in_grid = (yy >= 0) & (yy < 448) & (xx >= 0) & (xx < 448)
                can_copy = in_grid.copy()
                can_copy[in_grid] = unknown[yy[in_grid], xx[in_grid]]
                candidates.append((int(can_copy.sum()), dx, dy))
        alias_count, alias_dx, alias_dy = max(candidates)
        duplicated = observed.copy()
        yy, xx = body_y + alias_dy, body_x + alias_dx
        in_grid = (yy >= 0) & (yy < 448) & (xx >= 0) & (xx < 448)
        near_unknown = in_grid.copy()
        near_unknown[in_grid] = unknown[yy[in_grid], xx[in_grid]]
        duplicated[yy[near_unknown], xx[near_unknown]] = observed[body][near_unknown]
        assert np.array_equal(duplicated[known], observed[known])
        summary['Pair05'][prefix] = {
            'valid_circle_pixels': int(in_circle.sum()), 'real_foreground_pixels': int(fg.sum()),
            'real_foreground_valid_circle_fraction': float(fg.sum() / in_circle.sum()),
            'same_frame_visible_observed_valid_circle_fraction': float(known.sum() / in_circle.sum()),
            'central_upper_unknown_fraction': float((central_upper_region & unknown).sum() / central_upper_region.sum()),
            'unknown_pixel_rgb_mae_255_real_prior_vs_old_completions': float(delta[unknown].mean()),
            'central_upper_unknown_rgb_mae_255_real_prior_vs_old_completions': float(delta[central_upper_region & unknown].mean()),
            'shared_observed_pixel_rgb_maxdiff': float(delta[known].max()),
            'synthetic_duplicate_teammate_copy_unknown_pixels': alias_count,
            'synthetic_duplicate_teammate_shift_xy': [alias_dx, alias_dy],
        }
        for index, image in enumerate((observed, foreground_only, all_support, observed_and_prior)):
            start_x = pad + index * (thumb + pad)
            start_y = label + pad + row * (thumb + pad)
            if row == 0:
                text.text((start_x, 8), columns[index], fill='white')
            tile = Image.fromarray(image).resize((thumb, thumb), Image.Resampling.LANCZOS)
            panel.paste(tile, (start_x, start_y))
            text.text((start_x + 4, start_y + 4), prefix, fill='white')
summary['Pair05']['scope'] = 'Pixel-level adversarial composites only. Exact timing, hashes and same-frame source IDs are verified. Difference outside measured support is non-identifiability, not a model run.'
panel.save(output / 'pair05-compositing-counterexample.jpg', quality=93)
(output / 'measured-counterexamples.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
