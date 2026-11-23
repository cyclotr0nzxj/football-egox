"""Compare the pinned GGA and rendered fisheye camera equations on CPU."""

import ast
import json
from pathlib import Path
import re

import cv2
import numpy as np


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[4]


def coefficients(path, variable):
    source = path.read_text()
    expression = re.search(rf'\b{variable}\s*=\s*np\.array\((\[[-+.eE0-9,\s\[\]]+\])\)', source)
    if expression is None:
        raise ValueError(f'Could not identify {variable} in {path.name}.')
    return np.asarray(ast.literal_eval(expression[1])).reshape(-1)


def aria_project(ab, radial, tangential, prism):
    # FishEyeCameras v0.7.9 _project_points_batch, including nonradial terms.
    radius = np.linalg.norm(ab, axis=1)
    theta = np.arctan(radius)
    powers = theta[:, None] ** (2 * np.arange(1, 7))
    ratio = np.divide(theta, radius, out=np.ones_like(radius), where=radius > 1e-8)
    uv = ab * (ratio * (1 + powers @ radial))[:, None]
    norm2 = np.sum(uv**2, axis=1)
    distort = uv + 2 * (uv @ tangential)[:, None] * uv + norm2[:, None] * tangential
    return distort + np.column_stack((norm2, norm2**2)) @ prism.reshape(2, 2).T


def invert_aria(observations, radial, tangential, prism):
    radius = np.linalg.norm(observations, axis=1)
    ratio = np.divide(np.tan(radius), radius, out=np.ones_like(radius), where=radius > 1e-8)
    guess = observations * ratio[:, None]
    step = 1e-5
    basis = np.eye(2) * step
    for _ in range(8):
        here = aria_project(guess, radial, tangential, prism)
        jacobian = np.stack([(aria_project(guess + axis, radial, tangential, prism) -
                              aria_project(guess - axis, radial, tangential, prism)) / (2 * step)
                             for axis in basis], axis=2)
        diff = here - observations
        guess -= np.linalg.solve(jacobian, diff[:, :, None]).squeeze(2)
    return guess


def main():
    code = PROJECT / 'vendor/EgoX/infer.py'
    renderer = PROJECT / 'records/official_renderer_geometry.py'
    radial, tangential, prism = [coefficients(renderer, f'{name}_distortion_coeffs') for name in
                                 ('radial', 'tangential', 'thinPrism')]
    actual_gga_coeffs = coefficients(code, 'distortion_coeffs')
    assert np.array_equal(np.concatenate((radial, tangential, prism)), actual_gga_coeffs)
    meta = json.loads((OUT.parent / 'logs/meta.json').read_text())['test_datasets']
    k = np.asarray(meta[0]['ego_intrinsics'], dtype=np.float32)
    assert all(np.allclose(row['ego_intrinsics'], k) for row in meta)
    assert float(k[0, 2]) == float(k[1, 2]) == 224 and k[0, 0] == k[1, 1]
    scale = k.copy()
    scale[:2, :3] /= 8
    vv, uu = np.mgrid[:56, :56]
    latent_uv = np.column_stack((uu.ravel(), vv.ravel())).astype(np.float32)
    inferred = cv2.undistortPoints(latent_uv.reshape(-1, 1, 2), scale, actual_gga_coeffs.astype(np.float32),
                                  R=np.eye(3), P=np.eye(3)).reshape(-1, 2).astype(np.float64)
    # Renderer row-vector direction after its axis conversion is (y, -x, 1).
    in_renderer = np.column_stack((inferred[:, 1], -inferred[:, 0]))
    reprojection = 224 - float(k[0, 0]) * aria_project(in_renderer, radial, tangential, prism)
    grid_centered = np.column_stack((448 - 8 * latent_uv[:, 1], 8 * latent_uv[:, 0]))
    displacement = np.linalg.norm(reprojection - grid_centered, axis=1)
    radius_in_image = np.linalg.norm(grid_centered - 224, axis=1)
    inside = radius_in_image <= 224
    center_pixel = radius_in_image == 0
    assert displacement[center_pixel].max() < 1e-5
    selected = displacement[inside]
    matched_target = (224 - grid_centered[inside]) / float(k[0, 0])
    matched_rays = invert_aria(matched_target, radial, tangential, prism)
    matched_projection = 224 - float(k[0, 0]) * aria_project(matched_rays, radial, tangential, prism)
    matched_residual = np.linalg.norm(matched_projection - grid_centered[inside], axis=1)
    assert np.isfinite(matched_residual).all() and matched_residual.max() < 1e-3
    result = {
        'metric': 'Pixel residual after current GGA inverse and exact Aria forward equations on an ideal continuous grid; excludes discrete rot90 index, point rasterization, and attention pooling.',
        'physical_image_width': 448, 'latent_grid': [56, 56], 'nominal_circle_samples': int(inside.sum()),
        'median_px': float(np.median(selected)), 'p90_px': float(np.percentile(selected, 90)),
        'max_px': float(selected.max()), 'optical_center_px': float(displacement[center_pixel].max()),
        'matched_aria_inverse_p90_px': float(np.percentile(matched_residual, 90)),
        'matched_aria_inverse_max_px': float(matched_residual.max()),
    }
    output = OUT / 'lens-check.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
