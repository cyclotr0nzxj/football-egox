"""Compare the two available EXO ball-depth cues before ordering target samples."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rig-refactor-20261122/pair05-reset"))
from rebuild_pair05 import (  # noqa: E402
    HEIGHT, RUN, TAKE, WIDTH, backproject, load_depth, read_geometry,
    read_json, read_mask, rebuild_points, require_hash,
)


FRAMES = (12, 24)
RADIUS_HYPOTHESIS_M = 0.11


def ball_shape_cues(mask, intrinsics, up, camera_height):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rim = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(rim) < 5:
        raise ValueError("Not enough observed ball rim pixels to fit a silhouette.")
    center, diameters, _ = cv2.fitEllipse(rim)
    center_ray = backproject(*center, 1.0, intrinsics)
    direction = center_ray / np.linalg.norm(center_ray)
    rim_rays = backproject(*rim.T, np.ones(len(rim)), intrinsics)
    rim_direction = rim_rays / np.linalg.norm(rim_rays, axis=-1, keepdims=True)
    angles = np.arccos(np.clip(rim_direction @ direction, -1.0, 1.0))
    angular_radius = np.median(angles)
    ground_inverse_range = (up @ direction) / (RADIUS_HYPOTHESIS_M - camera_height)
    if angular_radius <= 0 or ground_inverse_range <= 0:
        raise ValueError("The candidate ray must hit the approximate ball center height.")
    return {
        "source_center_px": list(center),
        "observed_ellipse_diameters_px": list(diameters),
        "angular_radius_quantiles_degrees": np.degrees(np.quantile(angles, (.1, .5, .9))).tolist(),
        "ray_Z_from_nominal_radius_and_ellipse_m": float(
            direction[2] * RADIUS_HYPOTHESIS_M / np.sin(angular_radius)
        ),
        "ray_Z_from_nominal_radius_and_ground_m": float(direction[2] / ground_inverse_range),
    }


def compare(reviewed_masks=None):
    reset = HERE.parent / "rig-refactor-20261122/pair05-reset/results-sole-filtered"
    receipt = read_json(reset / "rebuild-receipt.json")
    geometry = RUN / "multiclip-evidence" / TAKE / "geometry/vipe_results" / TAKE
    depth_path = geometry / "depth/exo.zip"
    ball_path = RUN / "update-008/ball-recovery/pair05-ball.zip"
    require_hash(ball_path, receipt["source_sha256"]["ball_masks"])
    depth_receipt = next(row for row in read_json(RUN / "update-008/output-008/logs/validated-inputs.json")
                         if row["name"] == TAKE)
    if depth_receipt["exo_sha256"] != receipt["source_sha256"]["EXO"]:
        raise ValueError("The depth archive receipt belongs to a different EXO clip.")
    cameras = {}
    for variant in ("nominal", "alternate"):
        path = reset / f"{variant}-camera.json"
        require_hash(path, receipt["products"][path.name])
        cameras[variant] = read_json(path)
    old_intrinsics = read_geometry(geometry)
    old_up = np.asarray(receipt["input_fitted_plane"]["unit_up"])
    old_height = receipt["input_fitted_plane"]["height_vipe_units"]
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    results = {}
    mask_overrides = {}
    if reviewed_masks is not None:
        repair = read_json(reviewed_masks.parent / "repair-receipt.json")
        require_hash(reviewed_masks, repair["products"][reviewed_masks.name])
        if repair["exo_sha256"] != receipt["source_sha256"]["EXO"]:
            raise ValueError("The revised masks use a different EXO clip.")
        with np.load(reviewed_masks, allow_pickle=False) as archive:
            for frame in FRAMES:
                mask_overrides[frame] = archive[f"F{frame:02}_reviewed_ball"]
    with ZipFile(depth_path) as depths, ZipFile(ball_path) as balls:
        for frame in FRAMES:
            mask = read_mask(balls, frame) == 255
            if reviewed_masks is not None:
                revised = mask_overrides[frame]
                if revised.dtype != bool or revised.shape != mask.shape or (mask & ~revised).any():
                    raise ValueError(f"Invalid same-frame contour revision at F{frame:02}.")
                mask = revised
            results[f"F{frame:02}"] = {}
            depth = load_depth(depths, frame)
            encoded = BytesIO()
            np.save(encoded, depth, allow_pickle=False)
            if hashlib.sha256(encoded.getvalue()).hexdigest() != depth_receipt["depth_sha256"][frame]:
                raise ValueError(f"Depth checksum mismatch at F{frame:02}.")
            for name, camera in cameras.items():
                intrinsics = np.asarray(camera["exo_intrinsics"])
                up = np.asarray(camera["ground_up"])
                basis = (*tuple(intrinsics[:2] / old_intrinsics[:2]), 1.0)
                rays = backproject(x, y, np.ones_like(x), intrinsics).reshape(-1, 3)
                points, valid, _ = rebuild_points(depth, rays, old_up * basis, old_height, up)
                observed = mask.ravel() & valid
                shape = ball_shape_cues(mask, intrinsics, up, camera["camera_height_hypothesis_m"])
                shape["raw_projective_ball_Z_quantiles_m"] = np.quantile(
                    points[observed, 2], (.1, .5, .9)
                ).tolist()
                shape["source_pixels"] = int(observed.sum())
                results[f"F{frame:02}"][name] = shape
    return {
        **({"mask_source": reviewed_masks.name} if reviewed_masks is not None else {}),
        "radius_hypothesis_m": RADIUS_HYPOTHESIS_M,
        "interpretation": "The ellipse represents reviewed visible EXO samples, and may be partial. "
                          "The ground and radius estimates are sensitivity cues, not camera or contact calibration.",
        "frames": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewed-masks", type=Path)
    args = parser.parse_args()
    values = compare(args.reviewed_masks)
    with args.output.open("x") as handle:
        json.dump(values, handle, indent=2, allow_nan=False)
        handle.write("\n")
