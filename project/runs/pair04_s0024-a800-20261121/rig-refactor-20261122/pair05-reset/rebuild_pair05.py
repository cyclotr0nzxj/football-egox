"""Rebuild Pair05 from EXO landmarks and same-frame measured depths."""

import argparse
import hashlib
from io import BytesIO
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
PROJECT = RUN.parents[1]
for name in ("", "pair01-pose-audit-20261122", "pair05-foreground-match-20261122"):
    sys.path.insert(0, str(RUN / name))

from audit_pose import find_shoe_candidates
from build_foreground_match import (DISC_SOURCE, closest_surface, coefficients, load_depth, project, read_mask,
                                  trusted_teammate)
from build_multiclip_virtual_view import COUNTER_ROLL, backproject, fit_ground, read_geometry, smooth_track, unit
from review_pair01_ab import check_reference, review_frames


TAKE = "pair05_s0024"
FRAMES, WIDTH, HEIGHT, SIZE = 49, 784, 448, 448
EXO_FPS, PREVIEW_FPS = 25, 30
KEY_FRAMES = (0, 12, 24, 36, 48)
WEARER, TEAMMATE = 1, 2
TARGET_TEAMMATE, BALL = 1, 2
METRIC_CAMERA_HEIGHT = 1.23
HORIZON_ROLL = .025
COLORS = {1: "#72f09c", 2: "#ffcf5e", 3: "#978afc", 4: "#55c8ff", 5: "#d884f4", 6: "#5af2dd"}
NAMES = {1: "teammate", 2: "ball", **{i: f"disc_{i-2}" for i in range(3, 7)}}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_hash(path, expected):
    value = sha256(path)
    if value != expected:
        raise ValueError(f"Unexpected source digest: {path.name}")
    return value


def line_observations(image):
    """Fit the two visible white stripes in independently inspected EXO windows."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lines = []
    for guess_slope, guess_intercept in ((-.120, 300), (.120, 148)):
        centers = []
        for x in range(657, 783, 3):
            guess = int(guess_slope * x + guess_intercept)
            ys = np.arange(guess - 5, guess + 6)
            weights = np.maximum(gray[ys, x].astype(float) - 126, 0)
            if weights.sum() > 10:
                centers.append((x, float(np.average(ys, weights=weights))))
        if len(centers) < 30:
            raise ValueError("A field stripe has insufficient observed pixels.")
        centers = np.asarray(centers)
        line = np.polyfit(*centers.T, 1)
        error = float(np.sqrt(np.mean((np.polyval(line, centers[:, 0]) - centers[:, 1]) ** 2)))
        if error > 1.5:
            raise ValueError(f"Field line residual {error:.2f}px is too large.")
        lines.append({"line": line.tolist(), "samples": len(centers), "rmse_px": error})
    return lines


def solve_plane(stripes, focal, roll, cx, cy):
    """Compute the horizon under the explicit perpendicular-lines hypothesis."""
    rays = []
    for slope, intercept in stripes:
        x = np.polynomial.Polynomial([-roll * cx - intercept, 1]) / (slope - roll)
        y = slope * x + intercept
        rays.append(((x - cx) / focal, (y - cy) / focal))
    ray_a, ray_b = rays
    roots = (ray_a[0] * ray_b[0] + ray_a[1] * ray_b[1] + 1).roots()
    observable = []
    for root in roots:
        if abs(root.imag) > 1e-8:
            continue
        horizon = float(root.real)
        if all(y > horizon + roll * (x - cx) for x, y in DISC_SOURCE):
            observable.append(horizon)
    if len(observable) != 1:
        raise ValueError(f"Field lines and the visible discs cannot identify a horizon from {roots}.")
    horizon = observable[0]
    return horizon, unit(np.array([roll, -1, (horizon - cy) / focal]))


def observed_objects(rgb, instances, ball):
    labels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    labels[instances == TEAMMATE] = TARGET_TEAMMATE
    if np.any(ball & (instances != 0)):
        raise ValueError("The sealed ball mask conflicts with a source instance mask.")
    labels[ball] = BALL
    for label, (x, y) in enumerate(DISC_SOURCE, 3):
        patch = np.s_[y-5:y+6, x-7:x+8]
        r, g, b = rgb[patch].astype(np.int16).transpose(2, 0, 1)
        cyan = (g - r > 40) & (b - r > 35) & (g >= 115) & (instances[patch] == 0) & ~ball[patch]
        _, components = cv2.connectedComponents(cyan.astype(np.uint8))
        component = components[5, 7]
        if component and np.count_nonzero(components == component) >= 5:
            labels[patch][components == component] = label
    return labels


def rebuild_points(depth, rays, old_plane_in_new_rays, old_height, new_up):
    """Warp the depth field and plane in the same EXO ray basis.

    X'=X/d, d=(h_old + (n_old - n_new)·X)/h_new. Then
    n_new·X' + h_new = (n_old·X + h_old)/d, at unchanged EXO pixels.
    Invalid rays are unobserved instead of clipped to invented depths.
    """
    points = rays * depth.ravel()[:, None]
    denominator = ((old_plane_in_new_rays - new_up) @ points.T + old_height) / METRIC_CAMERA_HEIGHT
    supported = np.isfinite(denominator) & (denominator > 0.10)
    result = np.full_like(points, np.nan)
    result[supported] = points[supported] / denominator[supported, None]
    return result, supported, denominator


def subject_targets(points, valid, labels):
    labels = labels.ravel()
    y, x = np.nonzero(labels.reshape(HEIGHT, WIDTH) == TARGET_TEAMMATE)
    y0, y1 = np.quantile(y, (.08, .92))
    torso = (y0 + .21 * (y1 - y0) <= y) & (y <= y0 + .60 * (y1 - y0))
    x0, x1 = np.quantile(x[torso], (.12, .88))
    torso &= (x >= x0) & (x <= x1)
    teammates = (y * WIDTH + x)[torso]
    teammates = teammates[valid[teammates]]
    if len(teammates) < 30:
        raise ValueError("No measured teammate torso can define a target.")
    objects = {TARGET_TEAMMATE: np.median(points[teammates], axis=0)}
    for label in range(2, 7):
        mask = (labels == label) & valid
        if mask.sum() >= 5:
            objects[label] = np.median(points[mask], axis=0)
    return objects


def make_subject_pose(rays, feet, targets, up):
    flat_rays = rays - (rays @ up)[:, None] * up
    flat_feet = feet - (feet @ up)[:, None] * up
    head_z = np.sum(flat_feet * flat_rays, axis=1) / np.sum(flat_rays**2, axis=1)
    centers = rays * smooth_track(head_z[:, None])
    if np.min(head_z) <= 0:
        raise ValueError("A shoe-constrained head depth is behind the EXO camera.")

    forwards = []
    for center, objects in zip(centers, targets, strict=True):
        teammate = unit(objects[TARGET_TEAMMATE] - center)
        ball = unit(objects[BALL] - center) if BALL in objects else teammate
        discs = [target for label, target in objects.items() if label >= 3]
        nearby = sorted(discs, key=lambda point: np.linalg.norm(point - center))[:2]
        disc_direction = unit(np.mean(nearby, axis=0) - center) if nearby else ball
        forwards.append(unit(.55 * teammate + .35 * ball + .1 * disc_direction))
    orientations = []
    for forward in smooth_track(np.asarray(forwards)):
        forward = unit(forward)
        right = unit(np.cross(forward, up))
        orientations.append(COUNTER_ROLL @ np.stack((right, np.cross(forward, right), forward)))
    rotation = np.asarray(orientations)
    if not np.allclose(np.linalg.det(rotation), 1):
        raise ValueError("A reconstructed camera is not a proper rotation.")
    return centers, np.concatenate((rotation, -(rotation @ centers[..., None])), axis=2)


def project_frame(points, valid, rgb, labels, instances, head, pose, ego_k, distortion):
    # Reconstruct the wearer's exclusion from this frame's own instance components.
    wearer_rows = np.nonzero(instances == WEARER)[0]
    cutoff = head[1] + .62 * (int(wearer_rows.max()) - head[1])
    wearer_torso = (instances == WEARER) & (np.arange(HEIGHT)[:, None] < cutoff)
    candidates = np.flatnonzero(valid & ~wearer_torso.ravel())
    source, destination, depth = project(points, pose, ego_k, distortion, candidates)
    winner, _ = closest_surface(source, destination, depth)

    py, px = np.mgrid[:SIZE, :SIZE]
    inside_circle = np.hypot(px.ravel() - ego_k[0, 2], py.ravel() - ego_k[1, 2]) <= SIZE / 2
    seen = (winner >= 0) & inside_circle
    label_map = np.zeros(SIZE * SIZE, dtype=np.int16)
    label_map[seen] = labels.ravel()[winner[seen]]
    if np.any(label_map == TARGET_TEAMMATE):
        try:
            label_map, _ = trusted_teammate(label_map, winner, labels)
        except ValueError:
            label_map[label_map == TARGET_TEAMMATE] = 0
    trusted = label_map > 0
    foreground_sources = np.where(trusted, winner, -1).astype(np.int32)
    if not np.array_equal(labels.ravel()[foreground_sources[trusted]], label_map[trusted]):
        raise ValueError("Projected object pixels lost their same-frame source provenance.")

    rgb_out = np.zeros((SIZE * SIZE, 3), dtype=np.uint8)
    rgb_out[seen] = rgb.reshape(-1, 3)[winner[seen]]
    metrics = {"direct_coverage": float(seen.sum() / inside_circle.sum()), "objects": {}}
    for label, name in NAMES.items():
        yy, xx = np.nonzero(label_map.reshape(SIZE, SIZE) == label)
        largest = 0
        if len(xx):
            _, _, stats, _ = cv2.connectedComponentsWithStats(
                (label_map.reshape(SIZE, SIZE) == label).astype(np.uint8), connectivity=8)
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        metrics["objects"][name] = {"pixels": len(xx), "largest_contiguous_component": largest,
                                    "center_xy": [float(np.median(xx)), float(np.median(yy))] if len(xx) else None}
    return (rgb_out.reshape(SIZE, SIZE, 3), label_map.reshape(SIZE, SIZE),
            foreground_sources.reshape(SIZE, SIZE), metrics)


def write_video(path, frames):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{SIZE}x{SIZE}", "-framerate", str(PREVIEW_FPS), "-i", "-", "-an", "-c:v",
           "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-video_track_timescale",
           "30000", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames), check=True)
    check_reference(path, SIZE, PREVIEW_FPS)


def save_sheet(path, frames, historical, results):
    tile, width_exo, label_height, margin = 248, 350, 32, 12
    width = width_exo + 3 * tile + 5 * margin
    sheet = Image.new("RGB", (width, label_height + len(KEY_FRAMES) * (tile + margin)), "#17212c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 15)
    xs = [margin, width_exo+2*margin, width_exo+tile+3*margin, width_exo+2*tile+4*margin]
    for x, title in zip(xs, ("R8 同帧源画面", "旧正式几何（比较）", "重新估计机位", "另一帧内标定假设"), strict=True):
        draw.text((x, 7), title, font=font, fill="white")
    for index, frame in enumerate(KEY_FRAMES):
        y = label_height + index * (tile + margin)
        exo = frames[frame].resize((width_exo, round(width_exo * HEIGHT / WIDTH)), Image.Resampling.LANCZOS)
        sheet.paste(exo, (xs[0], y + (tile - exo.height) // 2))
        draw.text((xs[0], y + 3), f"F{frame:02}", font=font, fill="white")
        sheet.paste(historical[frame].resize((tile, tile), Image.Resampling.LANCZOS), (xs[1], y))
        for x, result in zip(xs[2:], results, strict=True):
            panel = Image.fromarray(result["images"][frame]).resize((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(panel, (x, y))
            for label, obj in result["metrics"][frame]["objects"].items():
                if obj["center_xy"] is None:
                    continue
                identity = next(key for key, name in NAMES.items() if name == label)
                px, py = np.asarray(obj["center_xy"]) * tile / SIZE
                draw.ellipse((x + px-4, y+py-4, x+px+4, y+py+4), outline=COLORS[identity], width=1)
    sheet.save(path, quality=94, optimize=True)


def rebuild_variant(intrinsics, stripes, old_intrinsics, old_up, old_height, observations, heads,
                    distortion, ego_k, output, name, unbounded_shoe_control):
    fx, fy, cx, cy = intrinsics
    horizon, up = solve_plane(stripes, fx, HORIZON_ROLL, cx, cy)
    # Preserve the EXO measurement under x'=fx_old/fx_new*x, y'=fy_old/fy_new*y.
    old_plane_in_new_rays = old_up * (fx / old_intrinsics[0], fy / old_intrinsics[1], 1)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx, yy, np.ones_like(xx), intrinsics).reshape(-1, 3)
    head_rays = backproject(*heads.T, np.ones(FRAMES), intrinsics)
    feet, targets, points_and_valid, audits = [], [], [], []
    for frame, (rgb, instances, depth, labels) in enumerate(observations):
        if instances[heads[frame, 1], heads[frame, 0]] != WEARER:
            raise ValueError(f"The source head mark does not overlap the red-shoe instance at F{frame:02}.")
        lowest_wearer_y = np.nonzero(instances == WEARER)[0].max()
        shoe_instances = (instances if unbounded_shoe_control else
                          np.where(yy >= lowest_wearer_y - 12, instances, 0))
        shoe = find_shoe_candidates(rgb, shoe_instances, heads[frame, 1], WEARER, depth, intrinsics, up,
                                    METRIC_CAMERA_HEIGHT)
        if not shoe:
            raise ValueError(f"No red shoe survives the ground support criterion at F{frame:02}.")
        feet.append(np.mean([support["plane_intersection"] for support in shoe], axis=0))
        points, valid, denominator = rebuild_points(depth, rays, old_plane_in_new_rays, old_height, up)
        targets.append(subject_targets(points, valid, labels))
        points_and_valid.append((points, valid))
        source_ground = (instances.ravel() == 0) & (yy.ravel() > 255) & valid
        transformed_height = points[source_ground] @ up + METRIC_CAMERA_HEIGHT
        unwarped_height = (rays[source_ground] * depth.ravel()[source_ground, None]) @ old_plane_in_new_rays + old_height
        error = np.max(abs(transformed_height - unwarped_height / denominator[source_ground]))
        if error > 1e-9:
            raise ValueError("The depth warp does not map the two calibration planes consistently.")
        audits.append({"lowest_wearer_y": int(lowest_wearer_y),
                       "red_shoe_source_points": [entry["pixel_xy"] for entry in shoe],
                       "valid_depth_fraction": float(valid.mean()), "plane_identity_error": float(error),
                       "source_objects_with_supported_depth": sorted(targets[-1])})

    centers, poses = make_subject_pose(head_rays, np.asarray(feet), targets, up)
    images, metrics, maps = [], [], {}
    for frame, ((rgb, instance, _, labels), (points, valid), pose) in enumerate(zip(observations, points_and_valid, poses, strict=True)):
        image, label_map, source_map, result = project_frame(points, valid, rgb, labels, instance, heads[frame],
                                                            pose, ego_k, distortion)
        images.append(image)
        for suffix, data in (("target_label", label_map), ("source_index", source_map)):
            maps[f"F{frame:02}_{suffix}"] = data
        metrics.append(dict(result, frame=frame, **audits[frame]))

    np.savez_compressed(output / f"{name}-provenance.npz", **maps)
    write_video(output / f"{name}-prior.mp4", images)
    camera = {"status": "CPU_DIAGNOSTIC_ONLY", "take": TAKE, "exo_intrinsics": intrinsics.tolist(),
              "exo_extrinsics": np.eye(4)[:3].tolist(), "ego_intrinsics": ego_k.tolist(),
              "ego_extrinsics": poses.tolist(), "head_centers": centers.tolist(), "ground_up": up.tolist(),
              "camera_height_hypothesis_m": METRIC_CAMERA_HEIGHT, "horizon_y_at_center": horizon,
              "horizon_slope": HORIZON_ROLL, "head_heights": (centers @ up + METRIC_CAMERA_HEIGHT).tolist(),
              "source_frame_indices": list(range(24, 24 + FRAMES)), "source_pts_seconds": [f"{i}/{EXO_FPS}" for i in range(FRAMES)],
              "render_pts_seconds": [f"{i}/{PREVIEW_FPS}" for i in range(FRAMES)],
              "calibration": "With perpendicular white lines, centered principal point and nominal horizon slope."
              " The depth field and plane use one EXO projective warp. Not an independent metric calibration."}
    (output / f"{name}-camera.json").write_text(json.dumps(camera, indent=2, allow_nan=False) + "\n")
    return {"images": images, "metrics": metrics, "camera": camera}


def run(output, unbounded_shoe_control):
    scene, prep = RUN / "multiclip-evidence" / TAKE, RUN / "multiclip-prep" / TAKE
    geometry = scene / "geometry/vipe_results" / TAKE
    anchors = read_json(prep / "exo_head_anchors.json")
    exo = prep / f"example/self_capture/videos/{TAKE}/exo.mp4"
    depth_path, mask_path = geometry / "depth/exo.zip", geometry / "mask/exo.zip"
    ball_path = RUN / "update-008/ball-recovery/pair05-ball.zip"
    historical_path = RUN / f"update-008/output-008/priors/{TAKE}-final.mp4"
    source_hashes = read_json(RUN / "pair05-foreground-match-20261122/results/verification.json")["input_hashes"]
    source_receipt = next(row for row in read_json(RUN / "update-008/output-008/logs/validated-inputs.json") if row["name"] == TAKE)
    renderer = PROJECT / "records/official_renderer_geometry.py"
    proved = {"EXO": require_hash(exo, anchors["video_sha256"]),
              "instance_masks": require_hash(mask_path, source_hashes["instances"]),
              "ball_masks": require_hash(ball_path, source_hashes["balls"]),
              "historical_prior": require_hash(historical_path, source_hashes["prior"]),
              "distortion_source": require_hash(renderer, source_hashes["renderer"])}
    if proved["EXO"] != source_receipt["exo_sha256"]:
        raise ValueError("Depth and annotated RGB have different EXO source hashes.")
    if [head["frame"] for head in anchors["frames"]] != list(range(FRAMES)):
        raise ValueError("Head measurements do not span exactly the expected local frames.")
    heads = np.asarray([[head["x"], head["y"]] for head in anchors["frames"]])
    cv2.setNumThreads(4)
    check_reference(exo, WIDTH, EXO_FPS)
    check_reference(historical_path, SIZE, PREVIEW_FPS)
    frames = review_frames(exo, WIDTH, range(FRAMES))
    historical = review_frames(historical_path, SIZE, KEY_FRAMES)
    original_intrinsics = read_geometry(geometry)
    distortion = np.concatenate([coefficients(renderer, f"{term}_distortion_coeffs")
                                 for term in ("radial", "tangential", "thinPrism")])
    ego_focal = (SIZE / 2) / np.deg2rad(95 / 2)
    ego_k = np.asarray([[ego_focal, 0, SIZE / 2], [0, ego_focal, SIZE / 2], [0, 0, 1.]])

    observations, old_plane_samples = [], []
    lines = [line_observations(np.asarray(frames[i])) for i in (0, 12, 24, 36, 48)]
    stripes = np.median([[line["line"] for line in row] for row in lines], axis=0)
    with ZipFile(depth_path) as depths, ZipFile(mask_path) as masks, ZipFile(ball_path) as balls:
        for frame in range(FRAMES):
            depth = load_depth(depths, frame)
            buffer = BytesIO()
            np.save(buffer, depth, allow_pickle=False)
            if hashlib.sha256(buffer.getvalue()).hexdigest() != source_receipt["depth_sha256"][frame]:
                raise ValueError(f"Depth digest mismatch F{frame:02}.")
            rgb, instances = np.asarray(frames[frame]), read_mask(masks, frame)
            labels = observed_objects(rgb, instances, read_mask(balls, frame) == 255)
            observations.append((rgb, instances, depth, labels))
            if frame in KEY_FRAMES:
                old_plane_samples.append(fit_ground(depth, instances, rgb.astype(float) / 255, original_intrinsics))
    old_up = unit(np.median([sample[0] for sample in old_plane_samples], axis=0))
    old_height = float(np.median([sample[1] for sample in old_plane_samples]))
    _, _, cx, cy = original_intrinsics
    output.mkdir(parents=True, exist_ok=False)
    focal_hypotheses = {"nominal": np.array([531., 531., cx, cy]), "alternate": original_intrinsics.copy()}
    results = {name: rebuild_variant(intrinsics, stripes, original_intrinsics, old_up, old_height,
                                     observations, heads, distortion, ego_k, output, name,
                                     unbounded_shoe_control)
               for name, intrinsics in focal_hypotheses.items()}
    save_sheet(output / "rebuild-comparison.jpg", frames, historical, list(results.values()))

    principal_delta = np.linalg.norm(np.asarray(results["nominal"]["camera"]["head_centers"])
                                     - np.asarray(results["alternate"]["camera"]["head_centers"]), axis=1)
    test = {}
    for frame in range(FRAMES):
        per_variant = {name: result["metrics"][frame] for name, result in results.items()}
        centers = [result["objects"]["ball"]["center_xy"] for result in per_variant.values()]
        ball_delta = None if any(center is None for center in centers) else float(np.linalg.norm(np.diff(centers, axis=0)))
        test[f"F{frame:02}"] = {"ball_center_shift_px": ball_delta, **per_variant}
    receipt = {"status": "CPU_DIAGNOSTIC_ONLY", "source_sha256": proved,
               "foot_support_filter": "unbounded control" if unbounded_shoe_control else "within 12px of wearer base",
               "source_code_sha256": sha256(Path(__file__)),
               "all_49_sealed_depth_digests_verified": True, "frame_count": FRAMES,
               "source_indices": [24, 72], "source_fps": EXO_FPS, "preview_fps": PREVIEW_FPS,
               "input_fitted_plane": {"unit_up": old_up.tolist(), "height_vipe_units": old_height},
               "line_fits_at_F00_F12_F24_F36_F48": lines, "median_lines": stripes.tolist(),
               "pose_sensitivity": {"head_center_shift_approx_m_median": float(np.median(principal_delta)),
                                    "head_center_shift_approx_m_max": float(np.max(principal_delta))},
               "target_label_meanings": NAMES, "source_index": "Same frame, y * 784 + x; -1 marks no verified foreground.",
               "metrics": test, "products": {path.name: sha256(path) for path in output.iterdir()}}
    (output / "rebuild-receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--unbounded-shoe-control", action="store_true",
                        help="Reproduce the rejected red-hand-as-shoe failure for adversarial comparison.")
    args = parser.parse_args()
    run(args.output, args.unbounded_shoe_control)
