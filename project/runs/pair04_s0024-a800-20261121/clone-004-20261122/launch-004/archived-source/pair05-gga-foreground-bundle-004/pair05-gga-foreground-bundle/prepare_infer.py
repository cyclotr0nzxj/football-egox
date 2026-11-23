"""Stage the pinned EgoX entrypoints and verify the Pair05 lens change on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from aria_gga import project_aria


OFFICIAL_INFER_SHA256 = "99199fac22858821d356f0ec3b8577495290dd579a8d80368de1fca68d559bed"
OFFICIAL_INVERSE = (
    "            normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))\n"
)
ARIA_INVERSE = (
    "            from aria_gga import undistort_aria_for_gga\n"
    "            normalized_points = undistort_aria_for_gga(pixel_coords_cv, K, D)\n"
)
TAKE = "pair05_s0024"
GRID = 56
PIXELS_PER_RAY = 8
IMAGE = GRID * PIXELS_PER_RAY


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def read_projection_data(metadata_bytes):
    rows = json.loads(metadata_bytes)["test_datasets"]
    if len(rows) != 1 or rows[0].get("exo_path") != f"./example/self_capture/videos/{TAKE}/exo.mp4":
        raise ValueError(f"Expected a frozen single-take metadata file for {TAKE}.")

    row = rows[0]
    k = np.asarray(row["ego_intrinsics"], dtype=np.float32)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Expected a finite 3×3 ego intrinsic matrix.")
    scaled_k = k.copy()
    scaled_k[:2, :3] /= PIXELS_PER_RAY

    extrinsics = np.asarray(row["ego_extrinsics"], dtype=np.float32)
    if extrinsics.shape != (49, 3, 4) or not np.isfinite(extrinsics).all():
        raise ValueError("Expected 49 finite 3×4 ego extrinsic matrices.")
    rotations = extrinsics[::4, :3, :3]
    gram = rotations @ rotations.transpose(0, 2, 1)
    if not np.allclose(gram, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1):
        raise ValueError("The 13 sampled ego rotations must be right-handed and orthonormal.")
    return scaled_k, rotations


def run_lens_block(source, grid, scaled_k):
    start = source.index("            import cv2\n")
    end = source.index("            normalized_points = torch.from_numpy", start)
    # Execute only the original/replaced inverse block using the exact staged bytes.
    code = "\n".join(line[12:] for line in source[start:end].splitlines())
    scope = {"np": np, "pixel_coords_cv": grid, "K": scaled_k}
    exec(compile(code, "<staged GGA inverse block>", "exec"), scope)
    rays = scope["normalized_points"]
    if rays.shape != (GRID * GRID, 1, 2) or rays.dtype != np.float32 or not np.isfinite(rays).all():
        raise ValueError("The inverse did not return finite float32 rays with EgoX's expected shape.")
    return rays[:, 0], scope["D"].reshape(-1).astype(np.float64)


def screen_errors(camera_xy, distortion, scaled_k, rotations, target_screen):
    # Repeat the pinned normalization, W2C row-vector multiplication and clockwise grid rotation.
    camera_rays = np.column_stack((camera_xy, np.ones(len(camera_xy), dtype=np.float32)))
    camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)
    world_rays = camera_rays @ rotations
    world_grid = np.rot90(world_rays.reshape(13, GRID, GRID, 3), k=-1, axes=(1, 2))
    original_camera_rays = world_grid.reshape(13, -1, 3) @ np.linalg.inv(rotations.astype(np.float64))

    # Renderer row-vector axis conversion places (y, -x, z) at Aria lens input.
    renderer_axes = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    renderer_rays = original_camera_rays.reshape(-1, 3) @ renderer_axes
    renderer_xy = renderer_rays[:, :2] / renderer_rays[:, 2:3]
    focal = PIXELS_PER_RAY * float(scaled_k[0, 0])
    projected = IMAGE / 2 - focal * project_aria(renderer_xy, distortion).reshape(13, GRID, GRID, 2)
    return np.linalg.norm(projected - target_screen, axis=-1)


def error_summary(errors, valid_circle):
    nominal_errors = errors[:, valid_circle]
    return {
        "nominal_circle_median_px": float(np.median(nominal_errors)),
        "nominal_circle_p90_px": float(np.percentile(nominal_errors, 90)),
        "nominal_circle_max_px": float(nominal_errors.max()),
        "outside_nominal_circle_max_px": float(errors[:, ~valid_circle].max()),
        "full_grid_max_px": float(errors.max()),
    }


def audit_projection(official, patched, metadata_bytes):
    k, rotations = read_projection_data(metadata_bytes)
    y, x = np.mgrid[:GRID, :GRID]
    grid = np.stack((x, y), axis=-1).reshape(-1, 1, 2).astype(np.float32)
    a_rays, a_distortion = run_lens_block(official, grid, k)
    b_rays, b_distortion = run_lens_block(patched, grid, k)
    if not np.array_equal(a_distortion, b_distortion):
        raise ValueError("The two inverse blocks used different Aria coefficients.")

    # Propagate the continuous diagnostic target with the same index permutation as the rays.
    # Physical latent-cell centers, rasterization and attention pooling are outside this metric.
    target_before_rotation = PIXELS_PER_RAY * np.stack((GRID - y, x), axis=-1)
    target_screen = np.rot90(target_before_rotation, k=-1, axes=(0, 1))
    if not np.array_equal(target_screen, PIXELS_PER_RAY * np.stack((x + 1, y), axis=-1)):
        raise ValueError("The continuous reference target was associated with the wrong rotated grid cells.")
    target_radius = np.linalg.norm(target_screen - IMAGE / 2, axis=-1)
    valid_circle = target_radius <= IMAGE / 2
    center = (grid[:, 0] == GRID // 2).all(axis=-1)
    if a_rays[center].tolist() != [[0, 0]] or b_rays[center].tolist() != [[0, 0]]:
        raise ValueError("The alternative solver moved the pinned optical center ray.")
    rotated_center = np.argwhere(np.rot90(center.reshape(GRID, GRID), k=-1))
    if rotated_center.tolist() != [[GRID // 2, GRID // 2 - 1]]:
        raise ValueError("The discrete rotation placed the optical center in the wrong grid cell.")
    corner_indices = [0, GRID - 1, GRID * (GRID - 1), GRID * GRID - 1]
    if np.sign(b_rays[corner_indices]).tolist() != [[-1, -1], [1, -1], [-1, 1], [1, 1]]:
        raise ValueError("The four matched corner rays have the wrong pre-rotation signs.")

    a_metrics = error_summary(screen_errors(a_rays, a_distortion, k, rotations, target_screen), valid_circle)
    b_metrics = error_summary(screen_errors(b_rays, b_distortion, k, rotations, target_screen), valid_circle)
    if not 38.0 < a_metrics["nominal_circle_p90_px"] < 38.4:
        raise ValueError("The pinned A result did not reproduce the historical Pair05 lens audit.")
    if b_metrics["full_grid_max_px"] >= 1e-3:
        raise ValueError("The matched B inverse did not pass the full-grid reprojection check.")
    return {
        "metric": "Continuous Aria screen reprojection of the staged GGA rays; excludes physical latent-cell centers, point rasterization and attention pooling.",
        "take": TAKE,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "metadata_sha256": digest(metadata_bytes),
        "image_width_px": IMAGE,
        "ray_grid": [GRID, GRID],
        "nominal_circle_cells": int(valid_circle.sum()),
        "outside_nominal_circle_cells": int((~valid_circle).sum()),
        "outside_circle_policy": "Solve the same front-facing Aria equation in B without discarding, folding, substituting or rearranging corner rays; full-grid validity is algebraic.",
        "b_max_angle_from_optical_axis_deg": float(np.degrees(np.arctan(np.linalg.norm(b_rays, axis=1))).max()),
        "optical_center_ray_xy": [0, 0],
        "optical_center_after_rot90_row_col": rotated_center[0].tolist(),
        "discrete_grid_mapping": "Upstream clockwise rotation maps pre-rotation (row=y, col=x) to (row=x, col=55-y). The existing continuous metric uses (screen_u=448-8y, screen_v=8x) before the same array permutation.",
        "pose_frames": list(range(0, 49, 4)),
        "a_official": a_metrics,
        "b_matched_aria": b_metrics,
        "exo_point_map_code_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-infer", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    official_bytes = args.official_infer.read_bytes()
    if digest(official_bytes) != OFFICIAL_INFER_SHA256:
        raise ValueError("EgoX infer.py does not match the pinned upstream SHA-256.")
    official = official_bytes.decode("utf-8")
    if official.count(OFFICIAL_INVERSE) != 1:
        raise ValueError("Could not isolate exactly one upstream GGA inverse assignment.")
    patched = official.replace(OFFICIAL_INVERSE, ARIA_INVERSE, 1)
    patched_bytes = patched.encode("utf-8")
    compile(patched, "infer_b.py", "exec")
    aria_bytes = Path(__file__).with_name("aria_gga.py").read_bytes()

    receipt = audit_projection(official, patched, args.meta.read_bytes())
    files = {"infer_a.py": official_bytes, "infer_b.py": patched_bytes, "aria_gga.py": aria_bytes}
    receipt["source_sha256"] = {name: digest(contents) for name, contents in files.items()}
    receipt_bytes = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    if args.output_dir is not None:
        args.output_dir.mkdir(exist_ok=True)
        if args.output_dir.is_symlink() or any(args.output_dir.iterdir()):
            raise ValueError("Projection stage must be a fresh or empty non-symlink directory.")
        files["projection_receipt.json"] = receipt_bytes
        for name, contents in files.items():
            with (args.output_dir / name).open("xb") as out:
                out.write(contents)
    print(receipt_bytes.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
