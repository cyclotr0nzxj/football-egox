"""Isolate target pixel rounding in Pair05 without changing camera, depth or labels."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rig-refactor-20261122/pair05-reset"))
from rebuild_pair05 import (  # noqa: E402
    HEIGHT, PROJECT, RUN, SIZE, TAKE, WEARER, WIDTH, backproject, coefficients,
    load_depth, read_geometry, read_json, read_mask, rebuild_points, require_hash,
)
from build_foreground_match import project_aria  # noqa: E402


def project_unrounded(points, candidates, pose, intrinsics, distortion):
    transformed = points[candidates] @ pose[:, :3].T + pose[:, 3]
    front = np.isfinite(transformed).all(axis=1) & (transformed[:, 2] > 1e-5)
    source, xyz = candidates[front], transformed[front]
    planes = np.column_stack((xyz[:, 1], -xyz[:, 0])) / xyz[:, 2, None]
    xy = intrinsics[:2, 2] - intrinsics[0, 0] * project_aria(planes, distortion)
    return source, xy, xyz[:, 2]


def sample(source, xy, depth, ball, scale):
    target_size = SIZE * scale
    destination = np.rint(xy * scale).astype(np.int32)
    mx, my = destination.T
    inside = ((mx >= 0) & (mx < target_size) & (my >= 0) & (my < target_size)
              & (((mx - target_size / 2)**2 + (my - target_size / 2)**2) <= (target_size / 2)**2))
    inside_source = source[inside]
    d = my[inside] * target_size + mx[inside]
    selected_z = depth[inside]
    ordering = np.lexsort((inside_source, selected_z, d))
    _, first = np.unique(d[ordering], return_index=True)
    winner = ordering[first]
    winner_ball = ball.ravel()[inside_source[winner]]
    final_inside_positions = np.rint(xy[inside][winner][winner_ball]).astype(np.int32)
    return {
        "target_scale": scale,
        "ball_source_samples_in_view": int(ball.ravel()[inside_source].sum()),
        "ball_subpixels_before_zbuffer": int(np.unique(d[ball.ravel()[inside_source]]).size),
        "ball_subpixels_after_zbuffer": int(winner_ball.sum()),
        "original_grid_cells_with_surviving_ball_subpixel": int(np.unique(final_inside_positions, axis=0).shape[0]),
    }


def audit():
    rebuilt = HERE.parent / "rig-refactor-20261122/pair05-reset"
    recorded = read_json(rebuilt / "visibility-counterexample.json")
    geometry = RUN / "multiclip-evidence" / TAKE / "geometry/vipe_results" / TAKE
    reconstruction = rebuilt / "results-sole-filtered"
    receipt = read_json(reconstruction / "rebuild-receipt.json")
    old_k = read_geometry(geometry)
    old_up = np.asarray(receipt["input_fitted_plane"]["unit_up"])
    old_height = receipt["input_fitted_plane"]["height_vipe_units"]
    heads = read_json(RUN / "multiclip-prep" / TAKE / "exo_head_anchors.json")["frames"]
    lens = PROJECT / "records/official_renderer_geometry.py"
    require_hash(lens, receipt["source_sha256"]["distortion_source"])
    distortion = np.concatenate([coefficients(lens, f"{term}_distortion_coeffs")
                                 for term in ("radial", "tangential", "thinPrism")])
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    cameras = {}
    for name in ("nominal", "alternate"):
        path = reconstruction / f"{name}-camera.json"
        require_hash(path, receipt["products"][path.name])
        cameras[name] = read_json(path)
    masks_path = geometry / "mask/exo.zip"
    balls_path = RUN / "update-008/ball-recovery/pair05-ball.zip"
    require_hash(masks_path, receipt["source_sha256"]["instance_masks"])
    require_hash(balls_path, receipt["source_sha256"]["ball_masks"])
    depth_receipt = next(row for row in read_json(RUN / "update-008/output-008/logs/validated-inputs.json")
                         if row["name"] == TAKE)
    if depth_receipt["exo_sha256"] != receipt["source_sha256"]["EXO"]:
        raise ValueError("Wrong depth/EXO pairing.")
    results = {}
    with (ZipFile(geometry / "depth/exo.zip") as depth_zip, ZipFile(masks_path) as masks,
          ZipFile(balls_path) as balls):
        for frame in (12, 24):
            key = f"F{frame:02}"
            instances = read_mask(masks, frame)
            ball = read_mask(balls, frame) == 255
            baseline_depth = load_depth(depth_zip, frame)
            encoded = BytesIO()
            np.save(encoded, baseline_depth, allow_pickle=False)
            if hashlib.sha256(encoded.getvalue()).hexdigest() != depth_receipt["depth_sha256"][frame]:
                raise ValueError(f"Depth checksum mismatch at {key}.")
            head_y = heads[frame]["y"]
            wearer_upper = (instances == WEARER) & (yy < head_y + .62 *
                           (np.nonzero(instances == WEARER)[0].max() - head_y))
            results[key] = {}
            for name, camera in cameras.items():
                k = np.asarray(camera["exo_intrinsics"])
                rays = backproject(xx, yy, np.ones_like(xx), k).reshape(-1, 3)
                points, valid, _ = rebuild_points(
                    baseline_depth, rays, old_up * (*tuple(k[:2] / old_k[:2]), 1.0), old_height,
                    np.asarray(camera["ground_up"]),
                )
                source, xy, z = project_unrounded(
                    points, np.flatnonzero(valid & ~wearer_upper.ravel()),
                    np.asarray(camera["ego_extrinsics"][frame]), np.asarray(camera["ego_intrinsics"]),
                    distortion,
                )
                variants = [sample(source, xy, z, ball, multiplier) for multiplier in (1, 2, 4)]
                expected = recorded[key][name]
                one = variants[0]
                if (one["ball_subpixels_before_zbuffer"], one["ball_subpixels_after_zbuffer"]) != (
                        expected["unique_ball_pixels_before_visibility"], expected["ball_pixels_after_visibility"]):
                    raise ValueError("Baseline rasterization does not reproduce the frozen receipt.")
                results[key][name] = variants
    return {
        "frames": results,
        "interpretation": "4x is a subdivision diagnostic with the old EXO ball label. A coarse "
                          "cell carrying one surviving subpixel is not a visible full-size ball pixel.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = audit()
    with args.output.open("x") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
