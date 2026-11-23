"""Recover an EXO-only Pair03 virtual camera and render a CPU geometry preview."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from estimate_depth import EXO_SHA, FRAMES, HEIGHT, WIDTH, read_exo, sha256

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_multiclip_virtual_view import backproject, project, smooth_track, unit  # noqa: E402

SIZE, FPS = 448, 25
CAMERA = [530.96, 530.96, WIDTH / 2, HEIGHT / 2]
GROUND_HEIGHT = 1.23
HORIZON_Y_AT_CENTER = 208.81
HORIZON_SLOPE = 0.025
UP = unit(np.array([HORIZON_SLOPE, -1, (HORIZON_Y_AT_CENTER - CAMERA[3]) / CAMERA[1]]))
FOV_DEG = 114
ROWS, COLS = np.mgrid[:HEIGHT, :WIDTH]
GRID = backproject(COLS, ROWS, np.ones((HEIGHT, WIDTH)), CAMERA)
GROUND_INV_Z = -(GRID @ UP) / GROUND_HEIGHT
OUT_Y, OUT_X = np.mgrid[:SIZE, :SIZE].astype(np.float32) - SIZE / 2
OUT_RADIUS = np.hypot(OUT_X, OUT_Y)
OUT_ANGLE = OUT_RADIUS * (np.deg2rad(FOV_DEG / 2) / (SIZE / 2))
OUT_FACTOR = np.divide(np.sin(OUT_ANGLE), OUT_RADIUS, out=np.zeros_like(OUT_RADIUS), where=OUT_RADIUS > 0)
OUT_RAYS = np.stack((OUT_X * OUT_FACTOR, OUT_Y * OUT_FACTOR, np.cos(OUT_ANGLE)), axis=-1)


def ray(pixel):
    return backproject(*pixel, 1.0, CAMERA)


def ground_point(pixel, height=0):
    direction = ray(pixel)
    distance = (height - GROUND_HEIGHT) / (UP @ direction)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError(f"Invalid ground intersection for an EXO observation: {pixel}")
    return direction * distance


def in_box(box, margin=0):
    left, top, right, bottom = box
    return ((COLS >= left - margin) & (COLS < right + margin)
            & (ROWS >= top - margin) & (ROWS < bottom + margin))


def calibrate_depth(raw, frames, anchors):
    depth = np.zeros_like(raw)
    audit = []
    indices = (ROWS % 4 == 0) & (COLS % 4 == 0)
    for index, (inverse, rgb, row) in enumerate(zip(raw, frames, anchors, strict=True)):
        excluded = in_box(row["wearer_box_xyxy"], 16) | in_box(row["teammate_box_xyxy"], 16)
        # The observed green field below the players is the calibration surface.
        red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
        field = ((green - red > 7) & (green - blue > 4) & (ROWS >= 274)
                 & (ROWS <= 442) & ~excluded)
        sample = field & indices
        if int(sample.sum()) < 1500:
            raise ValueError(f"Insufficient observable field at F{index:02d}.")
        points = np.column_stack((inverse[sample], np.ones(sample.sum())))
        truth = GROUND_INV_Z[sample]
        inliers = np.ones(len(truth), dtype=bool)
        for _ in range(4):
            slope, offset = np.linalg.lstsq(points[inliers], truth[inliers], rcond=None)[0]
            residual = truth - points @ [slope, offset]
            mad = np.median(np.abs(residual - np.median(residual)))
            inliers = np.abs(residual) <= max(0.018, 3.5 * mad)
        if slope <= 0 or inliers.mean() < 0.65:
            raise ValueError(f"MiDaS did not support a positive, stable ground fit at F{index:02d}.")
        linear = inverse * slope + offset
        # Keep actual surface pixels, and regularize clear grass toward the surveyed plane.
        compatible_grass = field & (np.abs(linear - GROUND_INV_Z) < 0.055)
        linear[compatible_grass] = GROUND_INV_Z[compatible_grass]
        z_valid = (linear >= 1 / 80.0) & (linear <= 1 / 0.5)
        depth[index, z_valid] = 1 / linear[z_valid]
        audit.append({
            "frame": index, "slope": float(slope), "offset": float(offset),
            "ground_samples": int(sample.sum()), "ground_fit_inlier_fraction": float(inliers.mean()),
            "ground_fit_holdout_median_abs_inverse_depth": float(
                np.median(np.abs(inverse[field & ~indices] * slope + offset - GROUND_INV_Z[field & ~indices]))
            ),
            "valid_depth_fraction": float(z_valid.mean()),
        })
    return depth, audit


def depth_near(depth, pixel, radius):
    x, y = np.rint(pixel).astype(int)
    patch = depth[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
    values = patch[patch > 0]
    return float(np.median(values)) if values.size else None


def subject_center(depth, head, feet):
    ground = ground_point(feet)
    direction = ray(head)
    lateral_ray = direction - (direction @ UP) * UP
    z_ground = (ground @ lateral_ray) / (lateral_ray @ lateral_ray)
    image_depth = depth_near(depth, head, 3)
    used_depth = z_ground
    if image_depth is not None:
        used_depth = 0.85 * z_ground + 0.15 * np.clip(image_depth, 0.8 * z_ground, 1.2 * z_ground)
    return direction * used_depth


def rotation_from(forward):
    forward = unit(forward)
    right = unit(np.cross(forward, UP))
    return np.stack((right, np.cross(forward, right), forward))


def pose_tracks(depth, annotations):
    raw_eye = np.stack([
        subject_center(z, row["wearer_head_xy"], row["wearer_ground_xy"])
        for z, row in zip(depth, annotations, strict=True)
    ])
    eye = smooth_track(raw_eye)
    team = smooth_track(np.stack([
        subject_center(z, row["teammate_torso_xy"], row["teammate_ground_xy"])
        for z, row in zip(depth, annotations, strict=True)
    ]))
    gaze = []
    ball_world = []
    for center, teammate, row in zip(eye, team, annotations, strict=True):
        position = row["ball_xy"]
        ball = None if position is None else ground_point(position, 0.11)
        ball_world.append(ball)
        toward_teammate = unit(teammate - center)
        if ball is None:
            # Without a source ball observation, look between the teammate and the ground.
            gaze.append(unit(toward_teammate - 0.42 * UP))
        else:
            gaze.append(unit(0.56 * toward_teammate + 0.44 * unit(ball - center)))
    axes = np.stack([rotation_from(vector) for vector in smooth_track(smooth_track(np.stack(gaze)))])
    head_errors = np.linalg.norm(project(eye, CAMERA) - np.array([row["wearer_head_xy"] for row in annotations]), axis=1)
    heights = eye @ UP + GROUND_HEIGHT
    steps = np.linalg.norm(np.diff(eye, axis=0), axis=1)
    if not np.allclose(axes @ axes.transpose(0, 2, 1), np.eye(3)) or not np.allclose(np.linalg.det(axes), 1):
        raise ValueError("Invalid virtual camera rotation.")
    if head_errors.max() > 12 or heights.min() < 0.75 or heights.max() > 2.2 or steps.max() > 0.4:
        raise ValueError(f"Head track failed range checks: {head_errors.max()}, {heights}, {steps.max()}.")
    return eye, raw_eye, team, ball_world, axes, head_errors, heights, steps


def person_mask(rgb, box, head):
    pad = 6
    x0, y0, x1, y1 = np.rint(box).astype(int)
    left, top, right, bottom = max(0, x0 - pad), max(0, y0 - pad), min(WIDTH, x1 + pad), min(HEIGHT, y1 + pad)
    crop = np.ascontiguousarray(rgb[top:bottom, left:right])
    segmentation = np.zeros(crop.shape[:2], np.uint8)
    segmentation[y0 - top:y1 - top, x0 - left:x1 - left] = cv2.GC_PR_FGD
    # Restrict hard foreground seeds to the visible head anchor.
    hx, hy = np.rint(head).astype(int) - [left, top]
    cv2.circle(segmentation, (hx, hy), 2, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, segmentation, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[top:bottom, left:right] = (segmentation == cv2.GC_FGD) | (segmentation == cv2.GC_PR_FGD)
    return mask


def source_objects(rgb, row, ball):
    wearer = person_mask(rgb, row["wearer_box_xyxy"], row["wearer_head_xy"])
    teammate = person_mask(rgb, row["teammate_box_xyxy"], row["teammate_head_xy"])
    both = wearer & teammate
    if both.any():
        # Ambiguous touching silhouettes are assigned to the nearer annotated axis.
        wearer_head, wearer_foot = row["wearer_head_xy"], row["wearer_ground_xy"]
        team_head, team_foot = row["teammate_head_xy"], row["teammate_ground_xy"]
        r, c = np.nonzero(both)
        wearer_x = np.interp(r, [wearer_head[1], wearer_foot[1]], [wearer_head[0], wearer_foot[0]])
        team_x = np.interp(r, [team_head[1], team_foot[1]], [team_head[0], team_foot[0]])
        team_closer = np.abs(c - team_x) < np.abs(c - wearer_x)
        wearer[r[team_closer], c[team_closer]] = False
        teammate[r[~team_closer], c[~team_closer]] = False
    labels = np.zeros((HEIGHT, WIDTH), np.uint8)
    labels[wearer] = 1
    labels[teammate] = 2
    if ball is not None and row["ball_status"] == "clear":
        bx, by = row["ball_xy"]
        radius = row.get("ball_radius_px_estimate") or np.clip(0.105 * CAMERA[0] / ball[2], 4, 12)
        # Use interior source texture only. Never paint an inferred or occluded ball.
        interior = (COLS - bx)**2 + (ROWS - by)**2 < (0.75 * radius)**2
        red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
        sneaker_red = (red - green > 35) & (red - blue > 20)
        labels[interior & ~sneaker_red] = 3
    return labels


def observed_grass(rgb, labels):
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    grass = (green - red > 6) & (green - blue > 4)
    white = (green >= 150) & (abs(red - green) < 35) & (abs(blue - green) < 35)
    grass_neighbor = cv2.dilate(grass.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
    foreground_neighbor = cv2.dilate((labels > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
    return (grass | (white & grass_neighbor)) & (ROWS > 229) & ~foreground_neighbor


def ground_texture(rgb, grass, eye, rotation):
    direction = OUT_RAYS @ rotation
    downward = direction @ UP
    distance = np.divide(-GROUND_HEIGHT - eye @ UP, downward,
                         out=np.zeros_like(downward), where=downward < -1e-5)
    position = eye + direction * distance[..., None]
    z = position[..., 2]
    x = np.divide(CAMERA[0] * position[..., 0], z, out=np.zeros_like(z), where=z > .1) + CAMERA[2]
    y = np.divide(CAMERA[1] * position[..., 1], z, out=np.zeros_like(z), where=z > .1) + CAMERA[3]
    x, y = x.astype(np.float32), y.astype(np.float32)
    observed = cv2.remap(grass.astype(np.uint8), x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    hit = (observed != 0) & (distance > .12) & (z > .1) & (OUT_RADIUS < SIZE / 2)
    texture = cv2.remap(rgb, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    texture[~hit] = 0
    ranges = np.where(hit, distance, np.inf)
    return texture, ranges, hit


def compute_frame(rgb, depth, row, eye, team, ball, rotation):
    labels = source_objects(rgb, row, ball)
    adjusted = depth.copy()
    # Match broad person depth to the visible foot-supported subject, while retaining local relief.
    for label, person_depth in ((1, eye[2]), (2, team[2])):
        mask = labels == label
        values = adjusted[mask]
        adjusted[mask] = np.where(values > 0, 0.35 * np.clip(values, person_depth - .40, person_depth + .40)
                                  + 0.65 * person_depth, person_depth)
    if ball is not None:
        adjusted[labels == 3] = ball[2]
    upper_limit = row["wearer_head_xy"][1] + 0.62 * (row["wearer_ground_xy"][1] - row["wearer_head_xy"][1])
    wear_upper = (labels == 1) & (ROWS < upper_limit)
    # GrabCut can miss portions of the wearer at near-identical background colors.
    wear_upper |= in_box(row["wearer_box_xyxy"]) & (ROWS < upper_limit) & (labels != 2) & (labels != 3)
    grass = observed_grass(rgb, labels) & ~wear_upper
    floor_image, floor_range, floor_hit = ground_texture(rgb, grass, eye, rotation)
    use = (adjusted > 0) & ~(wear_upper | grass)
    points = GRID[use] * adjusted[use, None]
    local = (points - eye) @ rotation.T
    radius = np.hypot(local[:, 0], local[:, 1])
    angle = np.arctan2(radius, local[:, 2])
    scale = np.divide((SIZE / 2) * angle, np.deg2rad(FOV_DEG / 2) * radius,
                      out=np.zeros_like(radius), where=radius > 1e-9)
    x = np.rint(SIZE / 2 + local[:, 0] * scale).astype(int)
    y = np.rint(SIZE / 2 + local[:, 1] * scale).astype(int)
    keep = ((local[:, 2] > 0.18) & (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
            & ((x - SIZE / 2)**2 + (y - SIZE / 2)**2 < (SIZE / 2)**2))
    dst = y[keep] * SIZE + x[keep]
    ranges = np.linalg.norm(local[keep], axis=1)
    order = np.lexsort((ranges, dst))
    _, first = np.unique(dst[order], return_index=True)
    nearest = order[first]
    winner = dst[nearest]
    nearer = ranges[nearest] <= floor_range.ravel()[winner] + 0.03
    nearest, winner = nearest[nearer], winner[nearer]
    image = floor_image.reshape(-1, 3)
    classes = np.zeros(SIZE * SIZE, np.uint8)
    image[winner] = rgb[use][keep][nearest]
    classes[winner] = labels[use][keep][nearest]
    z_buffer = floor_range.ravel().copy()
    z_buffer[winner] = ranges[nearest]
    image = image.reshape(SIZE, SIZE, 3)
    z_buffer = z_buffer.reshape(SIZE, SIZE)
    # One-pixel display splat fills sampling cracks with observed nearest colors.
    shown = image.copy()
    best = z_buffer.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor = np.roll(z_buffer, (dy, dx), axis=(0, 1))
        fill = (~np.isfinite(z_buffer)) & (neighbor < best)
        if dy:
            fill[0 if dy == 1 else -1, :] = False
        if dx:
            fill[:, 0 if dx == 1 else -1] = False
        shown[fill] = np.roll(image, (dy, dx), axis=(0, 1))[fill]
        best[fill] = neighbor[fill]
    return shown, {
        "source_teammate_pixels_estimated": int((labels == 2).sum()),
        "source_ball_pixels_observed": int((labels == 3).sum()),
        "teammate_projected_pixels_zbuffer": int((classes == 2).sum()),
        "ball_projected_pixels_zbuffer": int((classes == 3).sum()),
        "observed_grass_fraction": float(floor_hit.mean()),
        "direct_coverage_fraction": float(np.isfinite(z_buffer).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anchors = json.loads(args.anchors.read_text())["frames"]
    if [row["frame"] for row in anchors] != list(range(FRAMES)):
        raise ValueError("Pair03 EXO annotation indices are incomplete.")
    frames = read_exo()
    with np.load(args.depth, allow_pickle=False) as saved:
        inverse_depth = saved["inverse_depth"]
    if inverse_depth.shape != (FRAMES, HEIGHT, WIDTH) or not np.isfinite(inverse_depth).all():
        raise ValueError("The EXO inverse depth archive does not match the working clip.")
    cv2.setNumThreads(2)
    cv2.setRNGSeed(203)
    depth, ground_audit = calibrate_depth(inverse_depth, frames, anchors)
    eye, raw_eye, team, ball, axes, head_errors, heights, steps = pose_tracks(depth, anchors)
    args.output.mkdir(parents=True, exist_ok=False)
    depth_file = args.output / "approx_camera_z_depth.npz"
    np.savez_compressed(depth_file, depth=depth)
    output = args.output / "pair03-geometry-preview.mp4"
    visibility = []
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{SIZE}x{SIZE}", "-framerate", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-video_track_timescale", "25000", "-movflags", "+faststart", str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for index, rgb in enumerate(frames):
            canvas, audit = compute_frame(rgb, depth[index], anchors[index], eye[index],
                                          team[index], ball[index], axes[index])
            encoder.stdin.write(canvas.tobytes())
            visibility.append({"frame": index, "exo_ball_status": anchors[index]["ball_status"], **audit})
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Pair03 geometry preview encoding failed.")
    extrinsics = np.concatenate((axes, -(axes @ eye[..., None])), axis=2)
    trajectory = {
        "method": "EXO annotated head and feet, ground scale plus local MiDaS, five-frame head smoothing; virtual gaze aims at observed teammate and ball",
        "coordinate_system": "EXO OpenCV camera frame; approximate metres from measured camera height",
        "frame_times": [index / FPS for index in range(FRAMES)],
        "camera_intrinsics_fx_fy_cx_cy": CAMERA, "ground_up": UP.tolist(),
        "ground_height_assumed_m": GROUND_HEIGHT, "horizon_at_cx_y": HORIZON_Y_AT_CENTER,
        "horizon_dy_dx": HORIZON_SLOPE, "virtual_fisheye_diameter_fov_deg": FOV_DEG,
        "head_raw_centers": raw_eye.tolist(), "head_centers": eye.tolist(),
        "world_to_virtual_camera": extrinsics.tolist(),
        "head_reprojection_error_px": head_errors.tolist(),
        "head_height_approx_m": heights.tolist(), "head_step_approx_m": steps.tolist(),
        "teammate_torso_points": team.tolist(),
        "ball_center_points_when_exo_observable": [None if point is None else point.tolist() for point in ball],
    }
    track_file = args.output / "head-trajectory.json"
    track_file.write_text(json.dumps(trajectory, indent=2, allow_nan=False) + "\n")
    result = {
        "method": "CPU same-frame point reprojection with inverse texture sampling of visible EXO grass on the estimated ground plane",
        "exo_sha256": EXO_SHA, "annotations_sha256": sha256(args.anchors),
        "raw_inverse_depth_sha256": sha256(args.depth), "approx_camera_z_depth_sha256": sha256(depth_file),
        "head_trajectory_sha256": sha256(track_file), "preview_sha256": sha256(output),
        "frame_count": FRAMES, "fps": FPS, "ground_fit_per_frame": ground_audit,
        "source_object_pixel_support_per_frame": visibility,
        "visibility_note": "Labels are EXO-derived and approximate; direct z-buffer counts exclude display splats.",
    }
    (args.output / "preview-freeze.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Pair03 geometry preview {result['preview_sha256']}", flush=True)


if __name__ == "__main__":
    main()
