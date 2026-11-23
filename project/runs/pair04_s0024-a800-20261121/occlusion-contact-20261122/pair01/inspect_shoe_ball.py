"""Trace the visible Pair01 shoes and ball and test a same-ray ball-depth correction."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
RIG = RUN / "rig-refactor-20261122/pair01"
sys.path.insert(0, str(RIG))
from rebuild_pair01 import (  # noqa: E402
    CONTACT_FRAMES, GEOMETRY, HEIGHT, PREP, SCENE, SIZE, WIDTH,
    COLS, ROWS, GroundRemap, backproject, ball_core, check_reference,
    coefficients, load_depth, read_json, read_mask, red_shoes, render,
    review_frames, sha256, torso_mask,
)

FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
VISIBLE_RADIUS = 7.0
UNCERTAIN_OUTER_RADIUS = 9.0
ASSUMED_RADII = (0.10, 0.11, 0.12)
MARKER_RGB = "#ffe07a"


def pixel_rays(x, y, intrinsic):
    vector = backproject(x, y, np.ones_like(x, dtype=float), intrinsic)
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def feasible_sphere(pixel, intrinsic, radius, up, camera_height):
    """Choose the closest feasible angular size to the reviewed 7-pixel contour."""
    center_ray = pixel_rays(pixel[0], pixel[1], intrinsic)
    apparent_radii = np.linspace(VISIBLE_RADIUS - 1, VISIBLE_RADIUS + 1, 81)
    direction = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x_boundary = pixel[0] + apparent_radii[:, None] * np.cos(direction)
    y_boundary = pixel[1] + apparent_radii[:, None] * np.sin(direction)
    boundary = pixel_rays(x_boundary, y_boundary, intrinsic)
    angles = np.arccos(np.clip(boundary @ center_ray, -1, 1))
    centers = radius / np.sin(np.median(angles, axis=1))
    centers = centers[:, None] * center_ray
    clearance = centers @ up + camera_height - radius
    supported = np.flatnonzero(clearance >= 0)
    if not len(supported):
        return None
    index = supported[np.argmin(np.abs(apparent_radii[supported] - VISIBLE_RADIUS))]
    return centers[index], float(apparent_radii[index]), float(clearance[index])


def sphere_near_points(rays, center, radius):
    along = rays @ center
    discriminant = along**2 - (center @ center - radius**2)
    if (discriminant <= 0).any():
        raise ValueError("A trusted ball pixel falls outside the proposed sphere silhouette.")
    distance = along - np.sqrt(discriminant)
    if (distance <= 0).any():
        raise ValueError("A sphere surface point falls behind EXO.")
    return distance[:, None] * rays


def source_shoe_relation(red_mask, core, ball_xy):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask.astype(np.uint8), 8)
    components = []
    ball_pixels = np.column_stack(np.nonzero(core)[::-1])
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 6:
            continue
        pixels = np.column_stack(np.nonzero(labels == index)[::-1])
        distance = np.linalg.norm(pixels - ball_xy, axis=1)
        core_gap = np.linalg.norm(pixels[:, None] - ball_pixels[None, :], axis=2)
        components.append({
            "red_pixels": int(len(pixels)),
            "centroid_xy": pixels.mean(axis=0).tolist(),
            "nearest_to_ball_center_xy": pixels[distance.argmin()].tolist(),
            "distance_to_ball_center_px": float(distance.min()),
            "minimum_distance_to_reviewed_ball_core_px": float(core_gap.min()),
        })
    components.sort(key=lambda row: row["centroid_xy"][0])
    if len(components) != 2:
        raise ValueError("The source support does not identify exactly two visible red-shoe regions.")
    nearest = min(range(len(components)), key=lambda index: components[index]["distance_to_ball_center_px"])
    return components, nearest


def target_relation(labels, source_indices, nearest_shoe_xy):
    label_image = labels.reshape(SIZE, SIZE)
    result = {}
    ball, shoes = (np.column_stack(np.nonzero(label_image == index)[::-1]) for index in (1, 2))
    for name, points in (("ball_core", ball), ("shoes", shoes)):
        result[name] = {
            "pixels": int(len(points)),
            "center_xy": points.mean(axis=0).tolist() if len(points) else None,
            "bounds_xyxy": np.r_[points.min(axis=0), points.max(axis=0)].tolist() if len(points) else None,
            "p90_distance_to_median_px": float(np.quantile(np.linalg.norm(
                points - np.median(points, axis=0), axis=1), .9)) if len(points) else None,
        }
    nearest_shoe_source = nearest_shoe_xy[1] * WIDTH + nearest_shoe_xy[0]
    locations = np.flatnonzero((source_indices == nearest_shoe_source) & (labels == 2))
    if len(locations) > 1:
        raise ValueError("A single source shoe pixel cannot project to several target pixels.")
    shoe_y, shoe_x = divmod(int(locations[0]), SIZE) if len(locations) else (None, None)
    result["source_nearest_shoe_pixel_target_xy"] = [shoe_x, shoe_y] if len(locations) else None
    result["ball_minus_nearest_shoe_target_dxdy_px"] = (
        (ball.mean(axis=0) - np.asarray((shoe_x, shoe_y))).tolist()
        if len(ball) and len(locations) else None
    )
    return result


def source_group_card(rgb, ball_xy, shoes, near_index, frame, path_labels):
    crop_box = (511, 250, 626, 303)
    width, height = 460, 212
    tile = Image.new("RGB", (width, height + 39), "#101921")
    crop = Image.fromarray(rgb).crop(crop_box).resize((width, height), Image.Resampling.NEAREST)
    tile.paste(crop, (0, 22))
    draw = ImageDraw.Draw(tile)
    font, small = path_labels
    draw.text((5, 1), f"F{frame:02} · R8 原图局部", font=font, fill="white")
    x, y = (np.asarray(ball_xy) - crop_box[:2]) * 4 + (0, 22)
    radius = VISIBLE_RADIUS * 4
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=MARKER_RGB, width=1)
    near = shoes[near_index]
    sx, sy = (np.asarray(near["nearest_to_ball_center_xy"]) - crop_box[:2]) * 4 + (0, 22)
    draw.line((sx, sy, x, y), width=1, fill="#91e2ff")
    draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), outline="#91e2ff", width=2)
    state = "边缘相贴或交叠" if frame == 28 else "像面可见间隔"
    draw.text((5, height + 23), f"{state} · 触球不确定", font=small, fill="white")
    return tile


def virtual_group_card(baseline, corrected, frame, font):
    labels = ("原深度 · 同帧直接投影", "球深度约束 · 未知球缘留空")
    tile = Image.new("RGB", (448, 249), "#101921")
    draw = ImageDraw.Draw(tile)
    draw.text((5, 1), f"F{frame:02}  虚拟视角局部", font=font, fill="white")
    for index, (title, image) in enumerate(zip(labels, (baseline, corrected), strict=True)):
        x = index * 224
        crop = Image.fromarray(image).crop((108, 116, 340, 348))
        tile.paste(crop.resize((224, 224), Image.Resampling.LANCZOS), (x, 23))
        draw.text((x + 3, 24), title, font=font, fill="white", stroke_width=1, stroke_fill="black")
    return tile


def run(output):
    receipt_path = RIG / "results/reconstruction-receipt.json"
    baseline = read_json(receipt_path)
    metadata = read_json(RIG / "ball_observations.json")
    rig = baseline["rig"]
    old_meta, = read_json(RUN / "pair01-first-orientation-20261122/results/camera-candidate.json")["test_datasets"]
    exo = PREP / old_meta["exo_path"]
    paths = {
        "exo": exo, "vipe_depth": GEOMETRY / "depth/exo.zip",
        "vipe_instance": GEOMETRY / "mask/exo.zip",
        "wearer_exclusion": RUN / "update-008/ball-recovery/pair01-wearer-upper-body.zip",
        "old_comparison_pose": RUN / "pair01-first-orientation-20261122/results/camera-candidate.json",
        "ball_observations": RIG / "ball_observations.json",
    }
    for key, path in paths.items():
        if baseline["inputs_sha256"][key] != sha256(path):
            raise ValueError(f"The input {path.name} differs from the frozen rebuild.")
    if sha256(exo) != metadata["exo_sha256"]:
        raise ValueError("Reviewed ball center and decoded EXO do not share a source.")
    support_path = RIG / "results/same-frame-support.npz"
    if sha256(support_path) != baseline["same_frame_support_sha256"]:
        raise ValueError("The frozen rebuilt support map has changed.")
    check_reference(exo, WIDTH, 25)
    frames = review_frames(exo, WIDTH, CONTACT_FRAMES)
    head = read_json(PREP / "exo-head-annotations.json")["frames"]
    original_k = np.asarray(old_meta["camera_intrinsics"])[[0, 1, 0, 1], [0, 1, 2, 2]]
    focal = rig["exo_focal_px"]
    intrinsic = np.asarray((focal, focal, *rig["exo_principal_point_px"]))
    old_rays = backproject(COLS.ravel(), ROWS.ravel(), np.ones(WIDTH * HEIGHT), original_k)
    trajectory = read_json(SCENE / "viewpoint/trajectory-check.json")
    up = np.array(rig["ground_up"])
    height = baseline["ground_height_m_assumption"]
    remap = GroundRemap(original_k[:2], intrinsic[:2], np.array(trajectory["ground_up"]),
                       trajectory["fitted_exo_camera_height_vipe_units"], up, height)
    lens = np.array(old_meta["ego_intrinsics"])
    renderer = RUN.parents[1] / "records/official_renderer_geometry.py"
    distortion = np.concatenate([coefficients(renderer, f"{term}_distortion_coeffs")
                                 for term in ("radial", "tangential", "thinPrism")])
    poses = np.array(rig["ego_world_to_camera_renderer_convention"])
    label_fonts = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 12)
    thumb_width, thumb_height = 460, 251
    source_sheet = Image.new("RGB", (3 * thumb_width, 3 * thumb_height), "#101921")
    geometry_sheet = Image.new("RGB", (3 * 448, 3 * 249), "#101921")
    output.mkdir(parents=True, exist_ok=False)
    data = []
    with (ZipFile(paths["vipe_depth"]) as depths, ZipFile(paths["vipe_instance"]) as masks,
          ZipFile(paths["wearer_exclusion"]) as exclusions, np.load(support_path, allow_pickle=False) as support):
        for frame in CONTACT_FRAMES:
            frame_tag = f"F{frame:02}"
            rgb = np.asarray(frames[frame])
            instances = read_mask(masks, frame)
            upper = read_mask(exclusions, frame).ravel() != 0
            mark = metadata["observations"][str(frame)]
            core = ball_core(mark)
            red_mask = red_shoes(rgb, instances, head[frame]["y"])
            if np.any(red_mask & core) or np.any(core.ravel() & upper):
                raise ValueError(f"{frame_tag} contains overlapping source object identities.")
            shoe_components, nearest = source_shoe_relation(red_mask, core, mark[:2])
            nearest_xy = shoe_components[nearest]["nearest_to_ball_center_xy"]
            classes = np.zeros((HEIGHT, WIDTH), np.uint8)
            classes[torso_mask(instances)] = 3
            classes[red_mask] = 2
            classes[core] = 1
            classes = classes.ravel()
            x_ball, y_ball = mark[:2]
            near_silhouette = ((COLS - x_ball) ** 2 + (ROWS - y_ball) ** 2 <= UNCERTAIN_OUTER_RADIUS**2)
            uncertain_rim = near_silhouette.ravel() & (classes == 0) & (instances.ravel() == 0)
            original_depth = load_depth(depths, frame).ravel()
            world, valid = remap.transform(old_rays * original_depth[:, None])
            eligible = valid & ~upper
            original_rgb, _, original_labels, original_sources = render(
                world, eligible, poses[frame], lens, distortion, rgb, classes
            )
            recorded = support[f"{frame_tag}_target_label"].ravel()
            if (not np.array_equal(recorded, original_labels)
                    or not np.array_equal(support[f"{frame_tag}_source_index"].ravel()[recorded > 0],
                                          original_sources[recorded > 0])):
                raise ValueError(f"{frame_tag} does not reproduce the frozen prior.")
            same_scene_without_rim = eligible & ~uncertain_rim
            if not np.array_equal(same_scene_without_rim[classes > 0], eligible[classes > 0]):
                raise ValueError("The uncertainty guard discarded confirmed ball, shoe or torso observations.")
            variants = []
            y, x = np.nonzero(core)
            core_rays = pixel_rays(x, y, intrinsic)
            corrected_rgb = original_rgb
            for radius in ASSUMED_RADII:
                solution = feasible_sphere(mark[:2], intrinsic, radius, up, height)
                if solution is None:
                    variants.append({"radius_m_assumption": radius, "state": "no_feasible_solution"})
                    continue
                center, apparent_radius, clearance = solution
                corrected_world = world.copy()
                new_ball_surface = sphere_near_points(core_rays, center, radius)
                reprojection = new_ball_surface[:, :2] / new_ball_surface[:, 2, None]
                reprojection = reprojection * intrinsic[:2] + intrinsic[2:]
                if not np.allclose(reprojection, np.column_stack((x, y)), rtol=0, atol=1e-8):
                    raise ValueError(f"{frame_tag} sphere depth changed actual EXO pixel locations.")
                corrected_world[core.ravel()] = new_ball_surface
                moved = np.any(corrected_world != world, axis=1) & valid
                if np.any(moved != core.ravel()) or np.any(~valid[core.ravel()]):
                    raise ValueError(f"{frame_tag} correction changed unverified source pixels.")
                _, _, unchecked_labels, unchecked_sources = render(
                    corrected_world, eligible, poses[frame], lens, distortion, rgb, classes
                )
                pixels, _, target_labels, source_indices = render(
                    corrected_world, same_scene_without_rim, poses[frame], lens, distortion, rgb, classes
                )
                if not np.all(core.ravel()[source_indices[target_labels == 1]]):
                    raise ValueError("The depth correction introduced ball pixels without same-frame RGB.")
                if uncertain_rim[source_indices[source_indices >= 0]].any():
                    raise ValueError("The corrected render retained an unconfirmed ball rim source pixel.")
                variants.append({
                    "radius_m_assumption": radius,
                    "state": "tested",
                    "apparent_radius_px": apparent_radius,
                    "center_exo_coordinates_m_assumption": center.tolist(),
                    "ground_clearance_m_assumption": clearance,
                    "without_rim_guard": target_relation(unchecked_labels, unchecked_sources, nearest_xy),
                    **target_relation(target_labels, source_indices, nearest_xy),
                })
                if radius == 0.11:
                    corrected_rgb = pixels
            position = frame - CONTACT_FRAMES[0]
            source_sheet.paste(source_group_card(rgb, mark[:2], shoe_components, nearest, frame, label_fonts),
                               (position % 3 * thumb_width, position // 3 * thumb_height))
            geometry_sheet.paste(virtual_group_card(original_rgb, corrected_rgb, frame, label_fonts[1]),
                                 (position % 3 * 448, position // 3 * 249))
            data.append({
                "frame": frame, "source_ball_center_xy": mark[:2], "source_ball_core_pixels": int(core.sum()),
                "red_shoes_sorted_by_source_x": shoe_components, "nearest_red_shoe_index": nearest,
                "source_2d_relation": "near_or_overlapping_boundary" if frame == 28 else "separated_contours",
                "physical_contact_status": "uncertain",
                "held_out_unknown_source_pixels_near_ball": int(np.count_nonzero(eligible & uncertain_rim)),
                "baseline": target_relation(original_labels, original_sources, nearest_xy),
                "ball_only_depth_variants": variants,
            })
    source_sheet.save(output / "r8-shoe-and-ball.jpg", quality=96, subsampling=0)
    geometry_sheet.save(output / "geometry-same-frame.jpg", quality=96, subsampling=0)
    report = {
        "inputs_sha256": {k: sha256(v) for k, v in paths.items()},
        "prior_receipt_sha256": sha256(receipt_path),
        "source_frame_pts": "frame_index / 25",
        "image_radius_px_reviewed_approximately": VISIBLE_RADIUS,
        "unknown_rim_guard_radius_px": UNCERTAIN_OUTER_RADIUS,
        "mode": "EXO observation with assumed sphere near-depth on reviewed same-frame ball core only",
        "frames": data,
    }
    (output / "shoe-ball-review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
