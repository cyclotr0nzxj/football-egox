"""Reconstruct and replay an EXO-only Pair01 candidate using a separate calibrated rig."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
sys.path[:0] = [str(RUN / subdir) for subdir in (
    "pair01-pose-audit-20261122", "pair05-foreground-match-20261122", ""
)]

from audit_pose import (  # noqa: E402
    FRAMES, HEIGHT, SIZE, WIDTH, GEOMETRY, PREP, SCENE, TAKE, backproject, center_reprojection,
    check_reference, coefficients, load_depth, optical_centers, read_json, read_mask, review_frames,
    sha256, smooth_track,
)
from build_foreground_match import closest_surface, project_aria  # noqa: E402
from check_field_calibration import estimate_head, unit_height_plane  # noqa: E402
from review_orientation import camera_ext  # noqa: E402

CONTACT_FRAMES = tuple(range(20, 29))
KEYS = (0, 12, 20, 24, 28, 36, 48)
WEARER = 2
TEAMMATE = 1
ROWS, COLS = np.mgrid[:HEIGHT, :WIDTH]
YY, XX = np.mgrid[:SIZE, :SIZE]
FISHEYE_CIRCLE = ((XX - SIZE / 2) ** 2 + (YY - SIZE / 2) ** 2 <= (SIZE / 2) ** 2).ravel()


def unit(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


class GroundRemap:
    """Preserve pixel rays and map the old supported plane to the new plane."""

    def __init__(self, old_focal, new_focal, old_up, old_height, new_up, new_height):
        self.ray_change = np.r_[old_focal / new_focal, 1.]
        moved_normal = old_up / self.ray_change
        normal_scale = np.linalg.norm(moved_normal)
        moved_up = moved_normal / normal_scale
        self.scale = new_height / (old_height / normal_scale)
        self.gradient = (moved_up - new_up) / new_height

    def transform(self, points):
        scaled = points * self.ray_change * self.scale
        divisor = np.einsum("...i,i->...", scaled, self.gradient) + 1
        valid = np.isfinite(divisor) & (divisor > .05)
        moved = np.full_like(scaled, np.nan)
        np.divide(scaled, divisor[..., None], where=valid[..., None], out=moved)
        return moved, valid

    def transport_axes(self, centers, rotations):
        scaled_centers = centers * self.ray_change * self.scale
        scaled_axes = rotations * self.ray_change * self.scale
        divisor = scaled_centers @ self.gradient + 1
        induced = (scaled_axes @ self.gradient)[..., None] * scaled_centers[:, None, :]
        direction = scaled_axes * divisor[:, None, None] - induced
        forward = unit(direction[:, 2])
        right = unit(direction[:, 0] - np.sum(direction[:, 0] * forward, axis=1)[:, None] * forward)
        return np.stack((right, np.cross(forward, right), forward), axis=1)


def nearest_surface(world, eligible, w2c, focal_matrix, distortion):
    indices = np.flatnonzero(eligible)
    view = world[indices] @ w2c[:, :3].T + w2c[:, 3]
    # Candidates beyond this generous cone cannot reach the 95-degree display circle.
    bounded = ((view[:, 2] > 1e-3) & (np.max(np.abs(view[:, :2]), axis=1) <= 3 * view[:, 2]))
    view, indices = view[bounded], indices[bounded]
    aria = np.column_stack((view[:, 1], -view[:, 0])) / view[:, 2, None]
    uv = np.rint(focal_matrix[:2, 2] - focal_matrix[0, 0] * project_aria(aria, distortion)).astype(int)
    inside = ((uv >= 0) & (uv < SIZE)).all(axis=1)
    x, y = uv[inside].T
    destinations = y * SIZE + x
    inside_circle = FISHEYE_CIRCLE[destinations]
    projected = indices[inside][inside_circle]
    front, _ = closest_surface(projected, destinations[inside_circle], view[inside, 2][inside_circle])
    return projected, front


def ball_core(mark):
    if mark is None:
        return np.zeros((HEIGHT, WIDTH), dtype=bool)
    x, y, radius, min_x = mark
    core = (COLS - x) ** 2 + (ROWS - y) ** 2 <= radius ** 2
    return core if min_x is None else core & (COLS >= min_x)


def red_shoes(rgb, instances, head_y):
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    return ((red > 110) & (red - green > 50) & (red - blue > 32)
            & (instances == WEARER) & (ROWS > head_y + 60))


def torso_mask(instances):
    rows = np.nonzero(instances == TEAMMATE)[0]
    if len(rows) < 50:
        raise ValueError("Insufficient same-frame white-shoe teammate body support.")
    upper, lower = np.quantile(rows, (.20, .55))
    return (instances == TEAMMATE) & (ROWS >= upper) & (ROWS <= lower)


def blend(left, right, fraction):
    weight = np.clip(fraction, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    return left * (1 - weight) + right * weight


def look_target(frame, teammate, foot_center, observed_ball):
    if frame <= 19:
        return blend(teammate, foot_center, (frame - 12) / 7)
    if frame <= 28:
        if observed_ball is None:
            raise ValueError("Missing same-frame observed ball in stop window.")
        return .32 * foot_center + .68 * observed_ball
    return blend(foot_center, teammate, (frame - 41) / 7)


def camera_axes(centers, targets, teammate_targets, up):
    gaze = unit(smooth_track(unit(targets - centers)))
    vertical = gaze @ up
    horizontal = gaze - vertical[:, None] * up
    lengths = np.linalg.norm(horizontal, axis=1)
    for index in np.flatnonzero(lengths < 1e-6):
        backup = teammate_targets[index] - centers[index]
        horizontal[index] = backup - (backup @ up) * up
    pitch = np.maximum(np.arctan2(vertical, lengths), np.deg2rad(-78))
    forward = unit(horizontal) * np.cos(pitch)[:, None] + up * np.sin(pitch)[:, None]
    right = unit(np.cross(forward, up))
    return np.stack((right, np.cross(forward, right), forward), axis=1)


def render(world, eligible, pose, lens, distortion, rgb, classes):
    projected, surface = nearest_surface(world, eligible, pose, lens, distortion)
    visible = surface >= 0
    image = np.zeros((SIZE * SIZE, 3), dtype=np.uint8)
    image[visible] = rgb.reshape(-1, 3)[surface[visible]]
    semantic = np.zeros(SIZE * SIZE, np.uint8)
    semantic[visible] = classes[surface[visible]]
    result = {"direct_coverage_of_circle": float(visible.sum() / FISHEYE_CIRCLE.sum())}
    for index, name in ((1, "ball_core"), (2, "red_shoe"), (3, "teammate_torso")):
        y, x = np.nonzero((semantic == index).reshape(SIZE, SIZE))
        result[name] = {
            "eligible_source_pixels": int(np.count_nonzero(classes[eligible] == index)),
            "projected_in_circle_before_depth_competition": int(np.count_nonzero(classes[projected] == index)),
            "front_surface_pixels": len(x),
            "target_bounds_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None,
        }
    return image.reshape(SIZE, SIZE, 3), result, semantic, surface


def make_grid(output, frames, images, observations):
    labels = ("原 EXO / 人工球点", "旧跟球朝向", "同焦距改地面", "独立重建候选")
    thumb = 220
    row = thumb + 8
    sheet = Image.new("RGB", (thumb * 4 + 25, row * len(KEYS) + 27), "#171d24")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 16)
    for index, label in enumerate(labels):
        draw.text((5 + index * (thumb + 5), 4), label, font=font, fill="white")
    for index, number in enumerate(KEYS):
        y = 25 + index * row
        image = frames[number].copy()
        if number in observations:
            x, ball_y, radius, _ = observations[number]
            ImageDraw.Draw(image).ellipse((x-radius, ball_y-radius, x+radius, ball_y+radius),
                                         outline="#fff269", width=2)
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        sheet.paste(image, (5, y + (thumb - image.height) // 2))
        draw.text((8, y + 3), f"F{number:02}", font=font, fill="white")
        for arm, pixels in enumerate(images[number], start=1):
            sheet.paste(Image.fromarray(pixels).resize((thumb, thumb), Image.Resampling.LANCZOS),
                        (5 + arm * (thumb + 5), y))
    sheet.save(output, quality=94)


def video_writer(path):
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
        "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "15", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def main(output):
    calibration_dir = RUN / "pair01-pose-audit-20261122"
    calibration_path = calibration_dir / "calibration/field-calibration.json"
    prior_report = read_json(calibration_dir / "results/audit.json")
    calibration = read_json(calibration_path)
    trajectory = read_json(SCENE / "viewpoint/trajectory-check.json")
    old_meta_path = RUN / "pair01-first-orientation-20261122/results/camera-candidate.json"
    old, = read_json(old_meta_path)["test_datasets"]
    observations = read_json(HERE / "ball_observations.json")
    exo = PREP / old["exo_path"]
    expected_digest = sha256(exo)
    if any(data["exo_sha256"] != expected_digest for data in (observations, prior_report, calibration)):
        raise ValueError("Geometry inputs do not identify the same frozen EXO.")
    if not np.allclose(old["camera_extrinsics"], np.eye(4)[:3]):
        raise ValueError("EXO must be the common identity camera frame.")
    obs = {int(frame): row for frame, row in observations["observations"].items()}
    if tuple(sorted(obs)) != CONTACT_FRAMES or trajectory["head_instance_id"] != WEARER:
        raise ValueError("Unexpected EXO annotations.")
    depth_path = GEOMETRY / "depth/exo.zip"
    mask_path = GEOMETRY / "mask/exo.zip"
    exclusion_path = RUN / "update-008/ball-recovery/pair01-wearer-upper-body.zip"
    for path, key in ((depth_path, "vipe_depth_archive_sha256"), (mask_path, "vipe_mask_archive_sha256")):
        if sha256(path) != prior_report[key]:
            raise ValueError(f"Pair01 geometry archive mismatch: {path.name}")
    check_reference(exo, WIDTH, 25)
    frames = review_frames(exo, WIDTH, range(FRAMES))
    old_k = np.asarray(old["camera_intrinsics"])[[0, 1, 0, 1], [0, 1, 2, 2]]
    fx, fy, cx, cy = old_k
    if not np.isclose(fx, fy):
        raise ValueError("The field calibration assumes equal XY focal lengths.")
    lens = np.asarray(old["ego_intrinsics"])
    old_pose = np.asarray(old["ego_extrinsics"])
    centers = optical_centers(old_pose)
    up = np.asarray(trajectory["ground_up"])
    old_height = trajectory["fitted_exo_camera_height_vipe_units"]
    field_height = calibration["camera_height_reference_m"]
    rebuilt_calibration = calibration["nominal"]
    fixed_focal, = [variant for variant in calibration["sensitivity_variants"]
                    if np.isclose(variant["focal_px"], fx) and np.isclose(variant["horizon_slope"], .025)]
    fixed_up = np.asarray(fixed_focal["ground_up"])
    rebuilt_focal = rebuilt_calibration["focal_px"]
    rebuilt_up = np.asarray(rebuilt_calibration["ground_up"])
    rebuilt_k = np.asarray((rebuilt_focal, rebuilt_focal, cx, cy))
    fixed_map = GroundRemap(old_k[:2], old_k[:2], up, old_height, fixed_up, field_height)
    rebuilt_map = GroundRemap(old_k[:2], rebuilt_k[:2], up, old_height, rebuilt_up, field_height)
    fixed_centers, fixed_valid = fixed_map.transform(centers)
    if not fixed_valid.all():
        raise ValueError("Ground correction invalidates the control track.")
    fixed_axes = fixed_map.transport_axes(centers, old_pose[:, :, :3])
    fixed_poses = np.concatenate((fixed_axes, -(fixed_axes @ fixed_centers[..., None])), axis=2)
    head_marks = read_json(PREP / "exo-head-annotations.json")
    if head_marks["video_sha256"] != expected_digest:
        raise ValueError("Head source marks differ from the selected EXO.")
    head_xy = np.asarray([(row["x"], row["y"]) for row in head_marks["frames"]])
    red_shoe_marks = prior_report["per_frame_red_shoe_components"]
    feet = np.asarray([unit_height_plane([shoe["pixel_xy"] for shoe in row], rebuilt_k, rebuilt_up).mean(axis=0)
                       for row in red_shoe_marks])
    head_rays = backproject(*head_xy.T, np.ones(FRAMES), rebuilt_k)
    rebuilt_centers = estimate_head(red_shoe_marks, head_rays, rebuilt_k, rebuilt_up)
    source_error = np.linalg.norm(center_reprojection(rebuilt_centers, rebuilt_k) - head_xy, axis=1)
    if source_error.max() > 2:
        raise ValueError("The rebuilt head track no longer preserves its source image anchor.")
    renderer = RUN.parents[1] / "records/official_renderer_geometry.py"
    distortion = np.concatenate([coefficients(renderer, f"{term}_distortion_coeffs")
                                 for term in ("radial", "tangential", "thinPrism")])
    rays = backproject(COLS.ravel(), ROWS.ravel(), np.ones(HEIGHT * WIDTH), old_k)
    markers = np.asarray(calibration["blue_disc_markers_perimeter_order_xy"])
    marker_rays = backproject(*markers.T, np.ones(len(markers)), old_k)
    old_markers = marker_rays * (-old_height / (marker_rays @ up))[:, None]
    marker_tests = {}
    for name, remap, target_k, target_up in (
        ("same_focal", fixed_map, old_k, fixed_up), ("rebuilt", rebuilt_map, rebuilt_k, rebuilt_up)
    ):
        moved, valid = remap.transform(old_markers)
        plane_error = float(np.max(np.abs(moved @ target_up + field_height)))
        reprojection_error = float(np.max(np.abs(center_reprojection(moved, target_k) - markers)))
        if not valid.all() or plane_error > 1e-10 or reprojection_error > 1e-10:
            raise ValueError("The remap failed its plane or source-ray invariant.")
        marker_tests[name] = {"max_plane_error_m": plane_error, "max_source_pixel_error_px": reprojection_error}

    targets, teammates = [], []
    with ZipFile(depth_path) as depths, ZipFile(mask_path) as masks:
        for index in range(FRAMES):
            old_world = rays * load_depth(depths, index).ravel()[:, None]
            world, valid = rebuilt_map.transform(old_world)
            instance = read_mask(masks, index)
            team = torso_mask(instance).ravel() & valid
            observed = ball_core(obs.get(index)).ravel() & valid
            teammate = np.median(world[team], axis=0)
            ball = np.median(world[observed], axis=0) if observed.any() else None
            teammates.append(teammate)
            targets.append(look_target(index, teammate, feet[index], ball))
    rebuilt_poses = camera_ext(camera_axes(rebuilt_centers, np.asarray(targets), np.asarray(teammates), rebuilt_up),
                               rebuilt_centers)
    output.mkdir(parents=True, exist_ok=False)
    preview = output / "reconstruction-preview.mp4"
    results, key_images, support_maps = {}, {}, {}
    with (ZipFile(depth_path) as depths, ZipFile(mask_path) as masks, ZipFile(exclusion_path) as exclusions,
          video_writer(preview) as encoder):
        for index in range(FRAMES):
            rgb = np.asarray(frames[index])
            instance = read_mask(masks, index)
            upper = read_mask(exclusions, index).ravel() != 0
            core = ball_core(obs.get(index))
            if np.any(core.ravel() & upper):
                raise ValueError(f"The F{index:02d} reviewed ball core intersects the excluded upper body.")
            classes = np.zeros((HEIGHT, WIDTH), np.uint8)
            classes[torso_mask(instance)] = 3
            classes[red_shoes(rgb, instance, head_xy[index, 1])] = 2
            classes[core] = 1
            classes = classes.ravel()
            old_world = rays * load_depth(depths, index).ravel()[:, None]
            fixed_world, valid_fixed = fixed_map.transform(old_world)
            new_world, valid_new = rebuilt_map.transform(old_world)
            batches = (
                ("old_aim", old_world, ~upper, old_pose[index]),
                ("same_focal_field", fixed_world, valid_fixed & ~upper, fixed_poses[index]),
                ("rebuilt", new_world, valid_new & ~upper, rebuilt_poses[index]),
            )
            key_images[index] = []
            results[index] = {}
            for name, world, eligible, pose in batches:
                pixels, receipt, target_labels, source_indices = render(
                    world, eligible, pose, lens, distortion, rgb, classes
                )
                results[index][name] = receipt
                if index in KEYS:
                    key_images[index].append(pixels)
                if name == "rebuilt":
                    encoder.stdin.write(pixels.tobytes())
                    support_maps[f"F{index:02d}_target_label"] = target_labels.reshape(SIZE, SIZE)
                    support_maps[f"F{index:02d}_source_index"] = np.where(
                        target_labels > 0, source_indices, -1
                    ).reshape(SIZE, SIZE)
        encoder.stdin.close()
        if encoder.wait():
            raise ValueError("FFmpeg failed to encode the rebuilt preview.")
    check_reference(preview, SIZE, Fraction(25))
    make_grid(output / "same-frame-comparison.jpg", frames, key_images, obs)
    support_path = output / "same-frame-support.npz"
    np.savez_compressed(support_path, **support_maps)
    report = {
        "scope": "CPU EXO-only geometry sensitivity and candidate, no EgoX or visual gate clearance.",
        "inputs_sha256": {"exo": expected_digest, "vipe_depth": sha256(depth_path), "vipe_instance": sha256(mask_path),
                          "wearer_exclusion": sha256(exclusion_path), "ground_calibration": sha256(calibration_path),
                          "old_comparison_pose": sha256(old_meta_path), "ball_observations": sha256(HERE / "ball_observations.json")},
        "ray_and_plane_invariants": marker_tests,
        "head_source_pixel_error_max": float(source_error.max()),
        "max_axis_transport_deg": float(np.rad2deg(np.arccos(np.clip(
            np.sum(old_pose[:, 2, :3] * fixed_poses[:, 2, :3], axis=-1), -1, 1))).max()),
        "ground_height_m_assumption": field_height,
        "rig": {
            "exo_focal_px": rebuilt_focal, "exo_principal_point_px": [cx, cy],
            "ground_up": rebuilt_up.tolist(), "ego_centers_nominal_field_m": rebuilt_centers.tolist(),
            "ego_world_to_camera_renderer_convention": rebuilt_poses.tolist(),
            "aim": "White-shoe teammate torso at opening, smoothly shifts toward feet during F13–19, observed ball-and-feet during F20–28, returns to teammate at the end. Pitch limit 78 degrees down.",
        },
        "preview_sha256": sha256(preview),
        "preview_frame_count": FRAMES, "preview_fps": 25,
        "same_frame_support_sha256": sha256(support_path),
        "support_map_labels": {"0": "unknown or other", "1": "observed ball core F20–F28 only",
                               "2": "visible red-shoe source pixels", "3": "white-shoe teammate torso"},
        "support_map_source_index": "Row-major index in same 784x448 source frame, -1 where label is 0.",
        "same_frame_depth_evidence": results,
    }
    (output / "reconstruction-receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory for the local replay.")
    main(parser.parse_args().output)
