"""Recheck Pair05 F12/F24 ball identity using visible same-frame EXO contours."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
REBUILD = RUN / "rig-refactor-20261122/pair05-reset"
BASE = REBUILD / "results-sole-filtered"
sys.path.insert(0, str(REBUILD))
from rebuild_pair05 import (  # noqa: E402
    BALL, COLORS, EXO_FPS, HEIGHT, NAMES, PROJECT, SIZE, TAKE, WEARER, WIDTH,
    backproject, check_reference, closest_surface, coefficients, load_depth,
    observed_objects, project, project_frame, read_geometry, read_json, read_mask,
    rebuild_points, require_hash, review_frames, sha256,
)

KEYFRAMES = (12, 24)
VARIANTS = ("nominal", "alternate")
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def validate_additions(original_ball, revised_ball, original_labels, instances, controls):
    added = revised_ball & ~original_ball
    if np.any(original_ball & ~revised_ball):
        raise ValueError("Reviewed expansion deleted a sealed original ball pixel.")
    if np.any(added & ((instances != 0) | (original_labels != 0))):
        raise ValueError("Proposed ball pixel overlaps an already labeled foreground surface.")
    for control in controls:
        x, y = control["xy"]
        if instances[y, x] != control["source_instance"] or revised_ball[y, x]:
            raise ValueError(f"Non-ball control was reclassified: {control['name']}.")


def reviewed_mask(original_ball, original_labels, instances, annotation):
    x_y = np.asarray(annotation["visible_polygon_xy"], dtype=np.int32)
    if np.any((x_y < 0) | (x_y >= [WIDTH, HEIGHT])):
        raise ValueError("The reviewed polygon goes outside the source image.")
    polygon = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillPoly(polygon, [x_y], 1)
    revised = original_ball | (polygon != 0)
    controls = annotation["negative_controls"]
    validate_additions(original_ball, revised, original_labels, instances, controls)

    tightened = original_ball | (cv2.erode(polygon, np.ones((3, 3), np.uint8)) != 0)
    validate_additions(original_ball, tightened, original_labels, instances, controls)
    for control in controls:
        x, y = control["xy"]
        negative_mask = revised.copy()
        negative_mask[y, x] = True
        try:
            validate_additions(original_ball, negative_mask, original_labels, instances, controls)
        except ValueError:
            continue
        raise ValueError(f"Accepting the non-ball control should fail: {control['name']}.")
    return revised, tightened


def update_source_labels(labels, original_ball, revised_ball):
    updated = labels.copy()
    updated[revised_ball & ~original_ball] = BALL
    return updated


def named_metrics(report):
    return {
        name: {**item, "pixels": int(item["pixels"])}
        for name, item in report["objects"].items()
    }


def compare_view(frame, camera, rgb, instance, depth, old_labels, new_labels, tight_labels, old_intrinsics,
                 old_up, old_height, head_xy, distortion, baseline, grid):
    intrinsics = np.asarray(camera["exo_intrinsics"])
    up = np.asarray(camera["ground_up"])
    rays = backproject(*grid, np.ones_like(grid[0]), intrinsics).reshape(-1, 3)
    old_basis = (*tuple(intrinsics[:2] / old_intrinsics[:2]), 1.)
    points, valid, _ = rebuild_points(depth, rays, old_up * old_basis, old_height, up)
    pose, ego = np.asarray(camera["ego_extrinsics"][frame]), np.asarray(camera["ego_intrinsics"])
    shared = (points, valid, rgb)
    geometry = (instance, head_xy, pose, ego, distortion)
    before = project_frame(*shared, old_labels, *geometry)
    after = project_frame(*shared, new_labels, *geometry)
    smaller = project_frame(*shared, tight_labels, *geometry)
    before_rgb, before_labels, before_index, before_metrics = before
    after_rgb, after_labels, after_index, after_metrics = after
    if not np.array_equal(before_rgb, after_rgb) or not np.array_equal(before_rgb, smaller[0]):
        raise ValueError("Identity-only repair changed rendered RGB.")
    if not np.array_equal(before_labels, baseline["target_label"]) or not np.array_equal(
        before_index, baseline["source_index"]
    ):
        raise ValueError(f"Baseline provenance does not reproduce at F{frame:02}.")
    unchanged_ids = (before_labels != 0) & (before_labels != BALL)
    if np.any(before_labels[unchanged_ids] != after_labels[unchanged_ids]) or np.any(
        before_index[unchanged_ids] != after_index[unchanged_ids]
    ):
        raise ValueError("A verified teammate or disc pixel was replaced.")

    y_grid = grid[1]
    wearer_base = np.nonzero(instance == WEARER)[0].max()
    cutoff = head_xy[1] + .62 * (wearer_base - head_xy[1])
    wearer_torso = (instance == WEARER) & (y_grid < cutoff)
    source, destination, z = project(
        points, pose, ego, distortion, np.flatnonzero(valid & ~wearer_torso.ravel())
    )
    winner, _ = closest_surface(source, destination, z)
    result_positions = np.flatnonzero(after_labels.ravel() == BALL)
    if not np.array_equal(after_index.ravel()[result_positions], winner[result_positions]):
        raise ValueError("A repaired ball pixel is not the original nearest-depth winner.")
    if np.any(instance.ravel()[winner[result_positions]] != 0):
        raise ValueError("A previously labeled person source pixel was reassigned to ball.")

    y_target, x_target = np.mgrid[:SIZE, :SIZE]
    circle = (np.hypot(x_target - ego[0, 2], y_target - ego[1, 2]) <= SIZE / 2).ravel()
    original_bits, revised_bits = old_labels.ravel(), new_labels.ravel()
    old_ball_dst = np.unique(destination[(original_bits[source] == BALL) & circle[destination]])
    new_ball_dst = np.unique(destination[(revised_bits[source] == BALL) & circle[destination]])
    previous_losses = old_ball_dst[original_bits[winner[old_ball_dst]] != BALL]
    reclassified = previous_losses[revised_bits[winner[previous_losses]] == BALL]
    remaining = new_ball_dst[revised_bits[winner[new_ball_dst]] != BALL]
    if np.any(winner[remaining] < 0) or np.any(winner[previous_losses] < 0):
        raise ValueError("A supposedly occluded target has no competing source pixel.")
    source_people = instance.ravel()[winner[remaining]] > 0
    touches = {
        "old_candidate_ray_positions": int(len(old_ball_dst)),
        "new_candidate_ray_positions": int(len(new_ball_dst)),
        "old_losses_reclassified_as_same_ball": int(len(reclassified)),
        "old_losses_total": int(len(previous_losses)),
        "remaining_occluded_positions": int(len(remaining)),
        "remaining_positions_with_measured_person_winner": int(source_people.sum()),
        "remaining_competitors_are_not_certified_occluders": True,
        "old_loss_source_xy": [[int(p % WIDTH), int(p // WIDTH)] for p in winner[previous_losses]],
        "reclassified_loss_source_xy": [[int(p % WIDTH), int(p // WIDTH)] for p in winner[reclassified]],
    }
    simultaneous, counts = {}, named_metrics(after_metrics)
    for name, mask in (
        ("before", before_labels), ("reviewed", after_labels), ("one_pixel_inward", smaller[1])
    ):
        simultaneous[name] = bool(np.any(mask == 1) and np.any(mask == BALL) and np.any(mask >= 3))
    detail = {
        "before": named_metrics(before_metrics),
        "reviewed": counts,
        "one_pixel_inward": named_metrics(smaller[3]),
        "simultaneous_measured_support": simultaneous,
        "projected_rgb_identical": bool(np.array_equal(before_rgb, after_rgb)),
        "original_teammate_and_disc_pixels_identical": True,
        "depth_competition": touches,
    }
    return before, after, detail


def enlarged_crop(rgb, old_mask, reviewed, annotation):
    ys, xs = np.nonzero(reviewed)
    x0, x1 = int(xs.min() - 10), int(xs.max() + 14)
    y0, y1 = int(ys.min() - 10), int(ys.max() + 15)
    roi = np.s_[y0:y1, x0:x1]
    tile = Image.fromarray(rgb[roi])
    panels = []
    for mask, color in ((None, None), (old_mask, "#e599ff"), (reviewed, "#61e5eb")):
        displayed = tile.resize((244, round(244 * tile.height / tile.width)), Image.Resampling.NEAREST)
        if mask is not None:
            marked = cv2.resize(mask[roi].astype(np.uint8), displayed.size, interpolation=cv2.INTER_NEAREST)
            boundary = cv2.subtract(marked, cv2.erode(marked, np.ones((3, 3), dtype=np.uint8)))
            rows, cols = np.nonzero(boundary)
            overlay = ImageDraw.Draw(displayed)
            overlay.point(list(zip(cols.tolist(), rows.tolist())), fill=color)
            for control in annotation["negative_controls"]:
                x, y = control["xy"]
                if x0 <= x < x1 and y0 <= y < y1:
                    px = (x + .5 - x0) * displayed.width / (x1 - x0)
                    py = (y + .5 - y0) * displayed.height / (y1 - y0)
                    overlay.ellipse((px - 3, py - 3, px + 3, py + 3), outline="#ffe49b", width=2)
        panels.append(displayed)
    return panels


def draw_crop_sheet(path, rows):
    pitch = 260
    font = ImageFont.truetype(FONT, 18)
    sheet = Image.new("RGB", (3 * pitch + 12, 588), "#142031")
    draw = ImageDraw.Draw(sheet)
    for col, text in enumerate(("R8 原像素放大", "旧球掩码（紫）", "复划球轮廓（蓝）")):
        draw.text((12 + col * pitch, 9), text, font=font, fill="white")
    for pos, (frame, row) in enumerate(rows.items()):
        rgb, old_mask, revised, annotation = row
        y = 48 + 270 * pos
        for col, crop in enumerate(enlarged_crop(rgb, old_mask, revised, annotation)):
            sheet.paste(crop, (12 + col * pitch, y))
        draw.text((17, y + 4), f"F{frame:02}", font=font, fill="white", stroke_width=1, stroke_fill="black")
    sheet.save(path)


def draw_projection_sheet(path, rows):
    size, pitch = 248, 264
    font = ImageFont.truetype(FONT, 17)
    sheet = Image.new("RGB", (pitch * 3 + 12, 1120), "#142031")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("几何复投影 RGB", "旧前景证据", "复划后的前景证据")):
        draw.text((12 + pitch * column, 9), title, font=font, fill="white")
    for row, ((frame, name), triplet) in enumerate(rows.items()):
        before, after, detail = triplet
        y = 50 + row * 266
        for col, view in enumerate((before, before, after)):
            x = 12 + col * pitch
            rgb, labels, _, _ = view
            if col:
                annotated = np.zeros_like(rgb)
                annotated[labels > 0] = rgb[labels > 0]
                rgb = annotated
            panel = Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS)
            sheet.paste(panel, (x, y))
            if col:
                for label, name_key in NAMES.items():
                    target = detail["before" if col == 1 else "reviewed"][name_key]
                    if target["center_xy"] is None:
                        continue
                    px, py = np.asarray(target["center_xy"]) * size / SIZE
                    draw.ellipse((x + px - 6, y + py - 6, x + px + 6, y + py + 6),
                                 outline=COLORS[label], width=1)
                draw.text((x + 8, y + size - 24), f"球 {np.count_nonzero(labels == BALL)} px",
                          font=font, fill="#ffcf5e")
            else:
                title = f"F{frame:02}   " + ("名义标定" if name == "nominal" else "另一标定")
                draw.text((x + 8, y + 7), title, font=font, fill="white")
    sheet.save(path)


def draw_ball_closeups(path, rows):
    pitch, strip, half_extent = 180, 176, 22
    sheet = Image.new("RGB", (3 * pitch + 12, 4 * (strip + 32) + 50), "#142031")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 16)
    titles = ("原 RGB 放大", "旧球证据", "复划后的球证据")
    for col, title in enumerate(titles):
        draw.text((12 + pitch * col, 9), title, font=font, fill="white")
    for row, ((frame, name), (before, after, detail)) in enumerate(rows.items()):
        x, y = map(int, detail["reviewed"]["ball"]["center_xy"])
        tile = np.s_[y - half_extent:y + half_extent, x - half_extent:x + half_extent]
        top = 40 + row * (strip + 32)
        for col, view in enumerate((before, before, after)):
            material = view[0][tile].copy()
            if col:
                material[view[1][tile] != BALL] = 0
            panel = Image.fromarray(material).resize((strip, strip), Image.Resampling.NEAREST)
            sheet.paste(panel, (12 + pitch * col, top))
        title = f"F{frame:02} " + ("名义标定" if name == "nominal" else "另一标定")
        draw.text((12, top + strip + 3), title, font=font, fill="white")
    sheet.save(path)


def run(output):
    annotations_path = HERE / "ball-contours.json"
    annotations = read_json(annotations_path)["frames"]
    receipt = read_json(BASE / "rebuild-receipt.json")
    source = RUN / "multiclip-prep" / TAKE
    geometry = RUN / "multiclip-evidence" / TAKE / f"geometry/vipe_results/{TAKE}"
    exo = source / f"example/self_capture/videos/{TAKE}/exo.mp4"
    instance_path = geometry / "mask/exo.zip"
    ball_path = RUN / "update-008/ball-recovery/pair05-ball.zip"
    renderer = PROJECT / "records/official_renderer_geometry.py"
    for name, path in (("EXO", exo), ("instance_masks", instance_path), ("ball_masks", ball_path),
                       ("distortion_source", renderer)):
        require_hash(path, receipt["source_sha256"][name])
    original_source = next(row for row in read_json(RUN / "update-008/output-008/logs/validated-inputs.json")
                           if row["name"] == TAKE)
    if receipt["source_sha256"]["EXO"] != original_source["exo_sha256"]:
        raise ValueError("The original depth set belongs to a different EXO clip.")
    anchors = read_json(source / "exo_head_anchors.json")["frames"]
    old_intrinsics = read_geometry(geometry)
    old_up = np.asarray(receipt["input_fitted_plane"]["unit_up"])
    old_height = receipt["input_fitted_plane"]["height_vipe_units"]
    distortion = np.concatenate([coefficients(renderer, f"{part}_distortion_coeffs")
                                 for part in ("radial", "tangential", "thinPrism")])
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    check_reference(exo, WIDTH, EXO_FPS)
    frames = review_frames(exo, WIDTH, KEYFRAMES)
    output.mkdir(parents=True, exist_ok=False)

    cameras, baselines = {}, {}
    for name in VARIANTS:
        camera = BASE / f"{name}-camera.json"
        require_hash(camera, receipt["products"][camera.name])
        cameras[name] = read_json(camera)
        baseline = BASE / f"{name}-provenance.npz"
        require_hash(baseline, receipt["products"][baseline.name])
        with np.load(baseline, allow_pickle=False) as mappings:
            baselines[name] = {f"F{frame:02}_{suffix}": mappings[f"F{frame:02}_{suffix}"]
                               for frame in KEYFRAMES for suffix in ("target_label", "source_index")}

    source_summary, projections, crop_rows, maps, source_layers = {}, {}, {}, {}, {}
    with (ZipFile(geometry / "depth/exo.zip") as depth_zip, ZipFile(instance_path) as instance_zip,
          ZipFile(ball_path) as ball_zip):
        for frame, image in frames.items():
            key = f"F{frame:02}"
            rgb = np.asarray(image)
            depth = load_depth(depth_zip, frame)
            serialized = BytesIO()
            np.save(serialized, depth, allow_pickle=False)
            if hashlib.sha256(serialized.getvalue()).hexdigest() != original_source["depth_sha256"][frame]:
                raise ValueError(f"The frozen input depth changed at {key}.")
            instances = read_mask(instance_zip, frame)
            old_mask = read_mask(ball_zip, frame) == 255
            old_labels = observed_objects(rgb, instances, old_mask)
            annotation = annotations[key]
            revised_mask, tightened = reviewed_mask(old_mask, old_labels, instances, annotation)
            revised_labels = update_source_labels(old_labels, old_mask, revised_mask)
            tightened_labels = update_source_labels(old_labels, old_mask, tightened)
            add_y, add_x = np.nonzero(revised_mask & ~old_mask)
            stats = {
                "old_ball_source_pixels": int(old_mask.sum()),
                "reviewed_ball_source_pixels": int(revised_mask.sum()),
                "added_ball_source_pixels_xy": np.column_stack((add_x, add_y)).tolist(),
                "negative_controls_retained_and_forced_relabel_rejected": annotation["negative_controls"],
                "variants": {},
            }
            source_summary[key] = stats
            source_layers[f"{key}_old_ball"] = old_mask
            source_layers[f"{key}_reviewed_ball"] = revised_mask
            crop_rows[frame] = rgb, old_mask, revised_mask, annotation
            head = anchors[frame]
            for name in VARIANTS:
                past = {suffix: baselines[name][f"{key}_{suffix}"] for suffix in ("target_label", "source_index")}
                view = compare_view(frame, cameras[name], rgb, instances, depth, old_labels, revised_labels,
                                    tightened_labels, old_intrinsics, old_up, old_height,
                                    (head["x"], head["y"]), distortion, past, (xx, yy))
                before, corrected, detail = view
                projections[(frame, name)] = view
                Image.fromarray(corrected[0]).save(output / f"{key}-{name}-rgb.png")
                maps.setdefault(name, {})[f"{key}_target_label"] = corrected[1]
                maps[name][f"{key}_source_index"] = corrected[2]
                stats["variants"][name] = detail
    for name, arrays in maps.items():
        np.savez_compressed(output / f"{name}-provenance.npz", **arrays)
    np.savez_compressed(output / "source-ball-masks.npz", **source_layers)
    draw_crop_sheet(output / "exo-contour-review.png", crop_rows)
    draw_projection_sheet(output / "projection-review.png", projections)
    draw_ball_closeups(output / "projected-ball-closeups.png", projections)
    result = {
        "status": "LOCAL_SOURCE_IDENTITY_REPAIRED",
        "exo_sha256": receipt["source_sha256"]["EXO"],
        "contour_sha256": sha256(annotations_path),
        "camera_sha256": {name: receipt["products"][f"{name}-camera.json"] for name in VARIANTS},
        "invariants": "Same-frame source IDs, frozen cameras, depth, projected RGB and teammate/disc identities.",
        "frames": source_summary,
        "products": {file.name: sha256(file) for file in sorted(output.iterdir())},
        "scope": "Two-frame observed source identity correction; no depth or image correction and no unobserved surface.",
    }
    with (output / "repair-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)
