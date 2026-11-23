"""Compare first Pair01 pose against independent same-frame red-shoe support."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
for folder in ("", "ball-view-cpu-review", "pair01-gga-ab", "pair05-foreground-match-20261122",
               "pair01-first-orientation-20261122", "update-008/output-008/diagnosis"):
    sys.path.insert(0, str(RUN / folder))

from audit_lens import coefficients
from build_ball_previews import COUNTER_ROLL, FRAMES, HEIGHT, SIZE, WIDTH, load_depth
from build_ball_previews import corrected_orientations
from build_foreground_match import closest_surface, project, read_mask
from build_multiclip_virtual_view import backproject, smooth_track
from review_orientation import anchor_image, camera_ext
from review_pair01_ab import check_reference, review_frames


TAKE = "pair01_s0288"
KEY_FRAMES = (0, 20, 24, 28, 40, 48)
SCENE = RUN / "multiclip-evidence" / TAKE
PREP = RUN / "multiclip-prep" / TAKE
GEOMETRY = SCENE / "geometry/vipe_results" / TAKE


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def optical_centers(extrinsics):
    return np.linalg.solve(extrinsics[:, :, :3], -extrinsics[:, :, 3:]).squeeze(-1)


def find_shoe_candidates(rgb, mask, head_y, wearer, rgb_depth, intrinsics, up, height):
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    feet_region = np.arange(HEIGHT)[:, None] > head_y + 60
    shoes = (red > 110) & (red - green > 50) & (red - blue > 32) & (mask == wearer) & feet_region
    _, labels, stats, _ = cv2.connectedComponentsWithStats(shoes.astype(np.uint8), connectivity=8)
    largest = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1
    result = []
    for instance in largest:
        area = int(stats[instance, cv2.CC_STAT_AREA])
        if area < 7:
            continue
        y, x = np.nonzero(labels == instance)
        bottom = int(y.max())
        sole_x = float(np.median(x[y >= bottom - 2]))
        ray = backproject(sole_x, bottom, 1.0, intrinsics)
        if ray @ up >= 0:
            raise ValueError("A red-shoe candidate does not meet the ground in front of EXO.")
        ground_point = (-height / (ray @ up)) * ray
        sample = float(rgb_depth[bottom, round(sole_x)])
        result.append({
            "pixel_xy": [sole_x, bottom],
            "red_component_area_px": area,
            "plane_intersection": ground_point.tolist(),
            "depth_at_bottom_sample": sample,
            "predicted_bottom_height": float((ray * sample) @ up + height),
        })
    return result


def center_reprojection(points, intrinsics):
    fx, fy, cx, cy = intrinsics
    return np.column_stack((fx * points[:, 0] / points[:, 2] + cx,
                            fy * points[:, 1] / points[:, 2] + cy))


def summary(values):
    return {name: float(value) for name, value in
            zip(("min", "median", "max"), np.quantile(values, (0, 0.5, 1)), strict=True)}


def make_sheet(path, frames, heads, candidates, ball_screen, poses_and_names, ego_k, distortion, rays, depths,
               upper_body_mask_path):
    thumb = 250
    exo_thumb = 350
    margin = 12
    label_height = 32
    row_height = thumb + 26
    sheet = Image.new("RGB", (exo_thumb + 4 * thumb + 6 * margin, label_height + len(KEY_FRAMES) * row_height),
                      "#17212c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 17)
    labels = ("R8 源画面 · 原头点与红鞋取样", "首版机位与朝向", "只改机位代理值", "只改朝向", "机位代理值与新朝向")
    for x, label in zip([margin] + [exo_thumb + (j + 2) * margin + j * thumb for j in range(4)], labels):
        draw.text((x, 4), label, fill="white", font=font)
    with ZipFile(upper_body_mask_path) as masks:
        for index, frame in enumerate(KEY_FRAMES):
            y = label_height + index * row_height
            rgb = np.asarray(frames[frame])
            annotated = frames[frame].copy()
            origin = ImageDraw.Draw(annotated)
            x0, y0 = heads[frame]
            origin.ellipse((x0 - 5, y0 - 5, x0 + 5, y0 + 5), outline="#ffeb33", width=2)
            for candidate in candidates[frame]:
                x1, y1 = candidate["pixel_xy"]
                origin.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), outline="#4df5ff", width=2)
            scaled = annotated.resize((exo_thumb, round(HEIGHT * exo_thumb / WIDTH)), Image.Resampling.LANCZOS)
            sheet.paste(scaled, (margin, y + (thumb - scaled.height) // 2))
            draw.text((margin, y + 5), f"F{frame:02}", fill="white", font=font)
            world = rays * depths[frame].ravel()[:, None]
            retained = np.flatnonzero(read_mask(masks, frame).ravel() == 0)
            for j, (name, transforms) in enumerate(poses_and_names):
                src, dst, z = project(world, transforms[frame], ego_k, distortion, retained)
                surface, _ = closest_surface(src, dst, z)
                image = np.zeros((SIZE * SIZE, 3), dtype=np.uint8)
                visible = surface >= 0
                image[visible] = rgb.reshape(-1, 3)[surface[visible]]
                panel = Image.fromarray(image.reshape(SIZE, SIZE, 3)).resize((thumb, thumb), Image.Resampling.LANCZOS)
                x = exo_thumb + (j + 2) * margin + j * thumb
                sheet.paste(panel, (x, y))
                circle = ball_screen[name][frame]
                if np.isfinite(circle).all() and np.linalg.norm(circle - SIZE / 2) < SIZE / 2:
                    bx, by = circle * thumb / SIZE
                    draw.ellipse((x + bx - 5, y + by - 5, x + bx + 5, y + by + 5), outline="#ffeb33", width=2)
    sheet.save(path, quality=93, optimize=True)


def audit(output):
    first, = read_json(SCENE / "teammate-view/meta.json")["test_datasets"]
    batch = read_json(RUN / "multiclip-evidence/meta.json")["test_datasets"]
    if first != batch[1] or TAKE not in first["exo_path"] or not np.allclose(first["camera_extrinsics"], np.eye(4)[:3]):
        raise ValueError("The Pair01 first camera does not match the executed batch metadata.")
    updated, = read_json(RUN / "ball-view-cpu-review" / f"{TAKE}-ball-view-meta.json")["test_datasets"]
    orientation_only, = read_json(
        RUN / "pair01-first-orientation-20261122/results/camera-candidate.json"
    )["test_datasets"]
    first_pose = np.asarray(first["ego_extrinsics"])
    previous_pose = np.asarray(updated["ego_extrinsics"])
    proposed_pose = np.asarray(orientation_only["ego_extrinsics"])
    original_centers = optical_centers(first_pose)
    previous_centers = optical_centers(previous_pose)
    proposed_centers = optical_centers(proposed_pose)
    reference = read_json(SCENE / "viewpoint/trajectory-check.json")
    if not np.allclose(original_centers, reference["head_centers_vipe_units"]):
        raise ValueError("First-run optical centers differ from the frozen head track.")
    intrinsics = np.asarray(first["camera_intrinsics"])[[0, 1, 0, 1], [0, 1, 2, 2]]
    ego_k = np.asarray(first["ego_intrinsics"])
    up = np.asarray(reference["ground_up"])
    up /= np.linalg.norm(up)
    plane_height = reference["fitted_exo_camera_height_vipe_units"]
    wearer = reference["head_instance_id"]
    marks = read_json(PREP / "exo-head-annotations.json")
    heads = np.asarray([[frame["x"], frame["y"]] for frame in marks["frames"]])
    exo = PREP / first["exo_path"]
    first_video_sha = sha256(exo)
    if first_video_sha != marks["video_sha256"]:
        raise ValueError("Head marks are not tied to this EXO video.")
    check_reference(exo, WIDTH, 25)
    frames = review_frames(exo, WIDTH, range(FRAMES))
    shoe_candidates = []
    depth_frames = {}
    anchors_on_wearer = []
    depth_path = GEOMETRY / "depth/exo.zip"
    mask_path = GEOMETRY / "mask/exo.zip"
    upper_body_mask_path = RUN / "update-008/ball-recovery/pair01-wearer-upper-body.zip"
    with ZipFile(depth_path) as depths, ZipFile(mask_path) as masks:
        for frame, (x, y) in enumerate(heads):
            rgb = np.asarray(frames[frame])
            mask = read_mask(masks, frame)
            depth = load_depth(depths, frame)
            if frame in KEY_FRAMES:
                depth_frames[frame] = depth
            anchors_on_wearer.append(bool(mask[y, x] == wearer))
            shoe_candidates.append(find_shoe_candidates(rgb, mask, y, wearer, depth, intrinsics, up, plane_height))
    if not all(shoe_candidates) or not all(anchors_on_wearer):
        raise ValueError("A wearer head or visible red shoe could not be supported in all 49 frames.")

    # A sole supported plane proxy supplies Z only and retains the original EXO head ray.
    foot_center = np.asarray([
        np.mean([candidate["plane_intersection"] for candidate in candidates], axis=0)
        for candidates in shoe_candidates
    ])
    head_rays = backproject(heads[:, 0], heads[:, 1], np.ones(FRAMES), intrinsics)
    horizontal_rays = head_rays - (head_rays @ up)[:, None] * up
    horizontal_feet = foot_center - (foot_center @ up)[:, None] * up
    proxy_z = (horizontal_feet * horizontal_rays).sum(axis=1) / (horizontal_rays**2).sum(axis=1)
    proxy_centers = smooth_track(head_rays * proxy_z[:, None])
    shift = np.linalg.norm(proxy_centers - original_centers, axis=1)
    anchors = read_json(RUN / "ball-view-cpu-review/ball-anchors.json")[TAKE]
    _, ball, _ = corrected_orientations(anchors, original_centers, intrinsics, up, plane_height)
    teammate = np.asarray(read_json(SCENE / "teammate-view/retarget-check.json")["target_centers_vipe_units"])
    renderer = RUN.parents[1] / "records/official_renderer_geometry.py"
    distortion = np.concatenate([
        coefficients(renderer, f"{name}_distortion_coeffs") for name in ("radial", "tangential", "thinPrism")
    ])
    first_axes = COUNTER_ROLL.T @ first_pose[:, :, :3]
    proposed_axes = COUNTER_ROLL.T @ proposed_pose[:, :, :3]
    experiments = (
        ("first", original_centers, first_axes),
        ("position_proxy_only", proxy_centers, first_axes),
        ("orientation_only", original_centers, proposed_axes),
        ("position_proxy_and_orientation", proxy_centers, proposed_axes),
    )
    visibility, ball_screen, all_poses = {}, {}, []
    for name, centers, axes in experiments:
        ball_uv = anchor_image(ball, centers, axes, ego_k, distortion)
        team_uv = anchor_image(teammate, centers, axes, ego_k, distortion)
        ball_inside = np.linalg.norm(ball_uv - SIZE / 2, axis=1) < SIZE / 2
        team_inside = np.linalg.norm(team_uv - SIZE / 2, axis=1) < SIZE / 2
        ball_screen[name] = ball_uv
        all_poses.append((name, camera_ext(axes, centers)))
        visibility[name] = {
            "ball_center_inside_count": int(ball_inside.sum()),
            "ball_and_teammate_centers_inside_count": int((ball_inside & team_inside).sum()),
            "ball_center_inside_F20_F28_count": int(ball_inside[20:29].sum()),
            "ball_center_inside_frames": np.flatnonzero(ball_inside).tolist(),
        }
    if not (visibility["first"]["ball_center_inside_count"] == 8 and
            visibility["orientation_only"]["ball_center_inside_count"] == 49):
        raise ValueError("Audit could not reproduce previously frozen first and orientation-only arms.")
    z_difference = proxy_centers[:, 2] - original_centers[:, 2]
    first_ray = first_axes[:, 2]
    first_ball = ball - original_centers
    ball_offset = np.rad2deg(np.arccos(np.clip(
        np.sum(first_ray * first_ball, axis=1) / np.linalg.norm(first_ball, axis=1), -1, 1)))
    report = {
        "frame_count": FRAMES,
        "exo_sha256": first_video_sha,
        "vipe_depth_archive_sha256": sha256(depth_path),
        "vipe_mask_archive_sha256": sha256(mask_path),
        "first_pose_matches_executed_batch": True,
        "wearer_vipe_instance": wearer,
        "head_anchors_in_wearer_mask_frames": sum(anchors_on_wearer),
        "first_vs_previous_low_center_delta_max": float(np.linalg.norm(previous_centers - original_centers, axis=1).max()),
        "first_vs_orientation_candidate_center_delta_max": float(
            np.linalg.norm(proposed_centers - original_centers, axis=1).max()),
        "first_head_height_vipe_units": summary(original_centers @ up + plane_height),
        "proxy_head_height_vipe_units": summary(proxy_centers @ up + plane_height),
        "first_head_image_error_px": summary(np.linalg.norm(center_reprojection(original_centers, intrinsics) - heads, axis=1)),
        "proxy_head_image_error_px": summary(np.linalg.norm(center_reprojection(proxy_centers, intrinsics) - heads, axis=1)),
        "frames_with_two_red_shoe_components": sum(len(items) == 2 for items in shoe_candidates),
        "proxy_center_delta_vipe_units": summary(shift),
        "proxy_z_minus_first_z_vipe_units": summary(z_difference),
        "proxy_center_delta_F20_F28_vipe_units": summary(shift[20:29]),
        "first_axis_to_ball_F20_F28_degrees": summary(ball_offset[20:29]),
        "same_camera_lens_projection_visibility": visibility,
        "proxy_limit": "Ground intersections on visible red shoe bottoms and a vertical-body approximation. Foot contacts, monocular scale and real optical center are not independently known.",
        "first_centers": original_centers.tolist(),
        "proxy_centers": proxy_centers.tolist(),
        "per_frame_red_shoe_components": shoe_candidates,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx.ravel(), yy.ravel(), np.ones(HEIGHT * WIDTH), intrinsics)
    make_sheet(output / "pose-comparison.jpg", frames, heads, shoe_candidates, ball_screen,
               all_poses, ego_k, distortion, rays, depth_frames, upper_body_mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
