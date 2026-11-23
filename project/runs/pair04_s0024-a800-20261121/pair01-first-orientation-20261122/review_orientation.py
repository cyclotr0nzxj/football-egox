"""Evaluate a rotation-only Pair01 camera update against the first run, on CPU."""

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
for directory in ("ball-view-cpu-review", "pair01-gga-ab", "pair05-foreground-match-20261122",
                  "update-008/output-008/diagnosis"):
    sys.path.insert(0, str(RUN / directory))
from aria_gga import project_aria
from audit_lens import coefficients
from build_ball_previews import COUNTER_ROLL, FRAMES, HEIGHT, SIZE, WIDTH, load_depth
from build_ball_previews import backproject, corrected_orientations, load_json
from build_foreground_match import closest_surface, project, read_mask
from review_pair01_ab import review_frames

TAKE = "pair01_s0288"
FIRST_VIDEO_SHA = "515214b8d487c6edec53a7f5b8a0d39f1ab2bdb866b098759c099b9fd3512a11"
REVIEW_FRAMES = (0, 12, 20, 28, 40)


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def angle(a, b):
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1))


def turn(a, b, limit):
    theta = angle(a, b)
    delta = np.minimum(theta, limit)
    moved = np.sin(theta - delta)[..., None] * a + np.sin(delta)[..., None] * b
    return np.where(theta[..., None] > 1e-9, unit(moved + (theta < 1e-9)[..., None] * a), a)


def look_direction(gaze, up):
    right = unit(np.cross(gaze, up))
    return np.stack((right, np.cross(gaze, right), gaze), axis=1)


def candidate_look(first, ball_direction, up):
    # Keep opening frames, then turn just enough to bring the approaching ball into the lens.
    wanted = np.maximum(0, angle(first[:, 2], ball_direction) - np.deg2rad(29))
    aim = turn(first[:, 2], ball_direction, wanted)
    offsets = np.arange(-4, 5)
    sample_indices = np.clip(np.arange(FRAMES)[:, None] + offsets[None, :] + 1, 0, FRAMES - 1)
    gaze = unit(np.sum(aim[sample_indices] * (5 - np.abs(offsets))[None, :, None], axis=1))
    gaze[:4] = first[:4, 2]
    for f in range(4, FRAMES):
        gaze[f] = turn(gaze[f - 1], gaze[f], np.deg2rad(6))
    return look_direction(gaze, up)


def anchor_image(points, centers, rotations, k, distortion):
    local = (rotations @ (points - centers)[..., None]).squeeze(-1)
    front = local[:, 2] > 0
    uv = np.full((len(local), 2), np.nan)
    # Compensated for the same clockwise image rotation as the first run.
    uv[front] = k[:2, 2] - k[0, 0] * project_aria(-local[front, :2] / local[front, 2, None], distortion)
    return uv


def camera_ext(rotations, centers):
    live = COUNTER_ROLL @ rotations
    return np.concatenate((live, -(live @ centers[..., None])), axis=2)


def frame_sheet(output, baseline_video, first_prior, exo, archive, exclusions, rays, w2c, camera_k, distortion, ball_screen):
    first = review_frames(baseline_video, SIZE, REVIEW_FRAMES)
    prior = review_frames(first_prior, SIZE, REVIEW_FRAMES)
    cameras = review_frames(exo, WIDTH, REVIEW_FRAMES)
    width, height = 290, 290
    sheet = Image.new("RGB", (3 * width + 48, len(REVIEW_FRAMES) * (height + 24) + 80), "#18222c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 19)
    for i, name in enumerate(("首版成片", "首版几何先验", "新朝向 · 本机投影")):
        draw.text((12 + i * (width + 12), 10), name, font=font, fill="white")
    with ZipFile(archive) as depths, ZipFile(exclusions) as masks:
        for i, frame in enumerate(REVIEW_FRAMES):
            y = 48 + i * (height + 24)
            rgb = np.asarray(cameras[frame]).reshape(-1, 3)
            world = rays * load_depth(depths, frame).ravel()[:, None]
            candidates = np.flatnonzero(read_mask(masks, frame).ravel() == 0)
            src, dst, z = project(world, w2c[frame], camera_k, distortion, candidates)
            surface, _ = closest_surface(src, dst, z)
            image = np.zeros((SIZE * SIZE, 3), dtype=np.uint8)
            image[surface >= 0] = rgb[surface[surface >= 0]]
            diagnostic = Image.fromarray(image.reshape(SIZE, SIZE, 3))
            for col, panel in enumerate((first[frame], prior[frame], diagnostic)):
                x = 12 + col * (width + 12)
                sheet.paste(panel.resize((width, height), Image.Resampling.LANCZOS), (x, y))
            if np.isfinite(ball_screen[frame]).all():
                x, vy = 12 + 2 * (width + 12) + ball_screen[frame, 0] * width / SIZE, y + ball_screen[frame, 1] * height / SIZE
                draw.ellipse((x - 8, vy - 8, x + 8, vy + 8), outline="#ffd367", width=3)
            draw.text((20, y + 8), f"F{frame:02}", font=font, fill="white", stroke_width=2, stroke_fill="#18222c")
    draw.text((12, sheet.height - 30), "黄色圈为按 EXO 球标注估计的球心。右栏为几何预览。", font=font, fill="white")
    sheet.save(output / "orientation-comparison.jpg", quality=93)


def main(output):
    stage = RUN / f"multiclip-evidence/{TAKE}"
    source = stage / "teammate-view/meta.json"
    first, = load_json(source)["test_datasets"]
    batch = load_json(RUN / "multiclip-evidence/meta.json")["test_datasets"]
    assert first == batch[1] and TAKE in first["exo_path"]
    receipt, = [r for r in load_json(RUN / "multiclip-evidence/validated-inputs.json") if r["name"] == TAKE]
    prior = stage / f"teammate-prior/{TAKE}/ego_Prior.mp4"
    exo = RUN / f"multiclip-prep/{TAKE}" / first["exo_path"]
    baseline_video = RUN / f"four-egox-outputs-008/verified/{TAKE}_ego.mp4"
    for path, expected in ((exo, receipt["exo_sha256"]), (prior, receipt["prior_sha256"]),
                           (baseline_video, FIRST_VIDEO_SHA)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    track = load_json(stage / "viewpoint/trajectory-check.json")
    centers = np.array(track["head_centers_vipe_units"])
    up = unit(np.array(track["ground_up"]))
    k = np.array(first["camera_intrinsics"])
    assert np.allclose(first["camera_extrinsics"], np.eye(4)[:3])
    intrinsics = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
    ego_k = np.array(first["ego_intrinsics"])
    renderer = RUN.parents[1] / "records/official_renderer_geometry.py"
    distortion = np.concatenate([coefficients(renderer, f"{part}_distortion_coeffs")
                                 for part in ("radial", "tangential", "thinPrism")])
    original = np.array(first["ego_extrinsics"])
    original_look = COUNTER_ROLL.T @ original[:, :, :3]
    anchors = load_json(RUN / "ball-view-cpu-review/ball-anchors.json")[TAKE]
    _, ball, _ = corrected_orientations(anchors, centers, intrinsics, up,
                                        track["fitted_exo_camera_height_vipe_units"])
    teammate = np.array(load_json(stage / "teammate-view/retarget-check.json")["target_centers_vipe_units"])
    low = np.array(load_json(RUN / f"ball-view-cpu-review/{TAKE}-ball-view-meta.json")["test_datasets"][0]["ego_extrinsics"])
    pitch = np.arcsin(original_look[:, 2] @ up) - np.deg2rad(30)
    original_horizontal = unit(original_look[:, 2] - (original_look[:, 2] @ up)[:, None] * up)
    pitch_only = look_direction(np.cos(pitch)[:, None] * original_horizontal + np.sin(pitch)[:, None] * up, up)
    candidate = candidate_look(original_look, unit(ball - centers), up)
    pose = camera_ext(candidate, centers)
    pose[:4] = original[:4]
    assert np.allclose(pose[:, :, :3] @ centers[..., None], -pose[:, :, 3:])
    assert np.allclose(np.linalg.det(pose[:, :, :3]), 1)

    measurements = {}
    outputs = {}
    for name, look in (("first", original_look), ("pitch_only_down_30", pitch_only),
                       ("previous_low", COUNTER_ROLL.T @ low[:, :, :3]), ("candidate", candidate)):
        uv = [anchor_image(point, centers, look, ego_k, distortion) for point in (ball, teammate)]
        radius = [np.linalg.norm(image - SIZE / 2, axis=1) for image in uv]
        inside = [r <= SIZE / 2 for r in radius]
        measurements[name] = {"ball_center_inside_frames": np.flatnonzero(inside[0]).tolist(),
                              "teammate_torso_inside_frames": np.flatnonzero(inside[1]).tolist(),
                              "both_centers_inside_frames": np.flatnonzero(inside[0] & inside[1]).tolist(),
                              "ball_center_inside_F20_F28_count": int(np.sum(inside[0][20:29])),
                              "ball_center_minimum_circle_margin_px": float(SIZE / 2 - np.max(radius[0])),
                              "maximum_gaze_step_degrees": float(np.rad2deg(angle(look[1:, 2], look[:-1, 2])).max())}
        measurements[name]["ball_center_minimum_circle_margin_px"] = (measurements[name]["ball_center_minimum_circle_margin_px"]
                                                                      if np.isfinite(radius[0]).all() else None)
        outputs[name] = uv
    separation = np.rad2deg(angle(unit(ball - centers), unit(teammate - centers)))
    report = {"frames": FRAMES, "scope": "EXO-derived approximate geometric aim, not generated video",
              "identical_opening_frames": [0, 1, 2, 3], "unchanged_head_centers": True,
              "ball_to_teammate_center_angle_degrees_in_F20_F28": [float(x) for x in separation[20:29]],
              "measurements": measurements}
    output.mkdir(parents=True)
    (output / "camera-candidate.json").write_text(json.dumps({"test_datasets": [dict(first, ego_extrinsics=pose.tolist())]}, indent=2) + "\n")
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = backproject(xx.ravel(), yy.ravel(), np.ones(HEIGHT * WIDTH), intrinsics)
    archive = stage / f"geometry/vipe_results/{TAKE}/depth/exo.zip"
    masks = RUN / "update-008/ball-recovery/pair01-wearer-upper-body.zip"
    frame_sheet(output, baseline_video, prior, exo, archive, masks, rays, pose, ego_k, distortion, outputs["candidate"][0])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
