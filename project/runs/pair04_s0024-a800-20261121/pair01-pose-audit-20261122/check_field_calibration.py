"""Check the frozen Pair01 ground orientation against visible field markings."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont
from scipy.optimize import brentq

from audit_pose import PREP, SCENE, backproject, read_json, review_frames, sha256, smooth_track, summary


LINE_FRAMES = (0, 12, 24)
STRIPES = ((-.130, 351, 601), (.115, 207, 604))
DISC_MARKS = ((95, 286), (395, 267), (688, 307), (379, 395))
RECORDED_CAMERA_HEIGHT = 1.23


def field_stripes(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    results = []
    for starting_slope, starting_intercept, start_x in STRIPES:
        centers = []
        for x in range(start_x, 783, 3):
            predicted_y = int(starting_slope * x + starting_intercept)
            ys = np.arange(predicted_y - 5, predicted_y + 6)
            weights = np.maximum(gray[ys, x].astype(float) - 112, 0)
            if weights.sum() > 10:
                centers.append((x, (ys * weights).sum() / weights.sum()))
        centers = np.asarray(centers)
        line = np.polyfit(*centers.T, deg=1)
        residual = centers[:, 1] - np.polyval(line, centers[:, 0])
        results.append({
            "slope": float(line[0]), "intercept": float(line[1]),
            "white_samples": len(centers), "fit_rmse_px": float(np.sqrt((residual**2).mean())),
        })
    return results


def own_plane(stripes, focal, roll, cx, cy):
    def orthogonality(horizon_y):
        rays = []
        for slope, intercept in stripes:
            x = (horizon_y - roll * cx - intercept) / (slope - roll)
            y = slope * x + intercept
            rays.append(backproject(x, y, 1.0, [focal, focal, cx, cy]))
        return float(np.dot(*rays))

    horizon = brentq(orthogonality, 155, 238)
    up = np.array([roll, -1.0, (horizon - cy) / focal])
    return horizon, up / np.linalg.norm(up)


def unit_height_plane(xy, intrinsics, up):
    xy = np.asarray(xy)
    rays = backproject(xy[..., 0], xy[..., 1], np.ones(xy.shape[:-1]), intrinsics)
    denominator = rays @ up
    if np.any(denominator >= 0):
        raise ValueError("A visible ground mark lies above the estimated field horizon.")
    return rays * (-RECORDED_CAMERA_HEIGHT / denominator)[..., None]


def estimate_head(row_candidates, head_rays, intrinsics, up):
    feet = np.asarray([
        unit_height_plane([row["pixel_xy"] for row in candidates], intrinsics, up).mean(axis=0)
        for candidates in row_candidates
    ])
    ray_flat = head_rays - (head_rays @ up)[:, None] * up
    feet_flat = feet - (feet @ up)[:, None] * up
    depth = (feet_flat * ray_flat).sum(axis=1) / (ray_flat**2).sum(axis=1)
    return smooth_track(head_rays * depth[:, None])


def write_overlay(image, stripes, old_line, nominal_line, cx, output):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for slope, intercept in stripes:
        draw.line((605, slope * 605 + intercept, 783, slope * 783 + intercept), fill="#54ff87", width=2)
    for (height, roll), color in ((old_line, "#ffae6c"), (nominal_line, "#65fff1")):
        draw.line((0, height - roll * cx, 783, height + roll * (783 - cx)), fill=color, width=2)
    for x, y in DISC_MARKS:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#aa96ff", width=2)
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 15)
    draw.rectangle((6, 4, 436, 50), fill="#131b25")
    draw.text((14, 7), "绿：拟合白线 · 紫：场地蓝盘取样", fill="white", font=font)
    draw.text((14, 27), "橙：ViPE 平面消失线 · 青：白线直角假设下的消失线", fill="white", font=font)
    image.save(output, quality=94, optimize=True)


def audit(position_results, output):
    old = read_json(position_results)
    view, = read_json(SCENE / "teammate-view/meta.json")["test_datasets"]
    old_track = read_json(SCENE / "viewpoint/trajectory-check.json")
    exo = PREP / view["exo_path"]
    if sha256(exo) != old["exo_sha256"]:
        raise ValueError("The field check EXO differs from the position audit.")
    frames = review_frames(exo, 784, LINE_FRAMES)
    raw = {str(n): field_stripes(np.asarray(frames[n])) for n in LINE_FRAMES}
    stripes = np.median([
        [(stripe["slope"], stripe["intercept"]) for stripe in rows] for rows in raw.values()
    ], axis=0)
    camera_calibration = np.asarray(view["camera_intrinsics"])
    cx, cy = camera_calibration[:2, 2]
    old_focal = camera_calibration[0, 0]
    old_up = np.asarray(old_track["ground_up"])
    old_horizon = cy - old_focal * old_up[2] / old_up[1]
    old_roll = -old_up[0] / old_up[1]
    old_in_m = np.asarray(old["first_centers"]) * RECORDED_CAMERA_HEIGHT / old_track["fitted_exo_camera_height_vipe_units"]
    head_xy = np.asarray([[entry["x"], entry["y"]]
                          for entry in read_json(PREP / "exo-head-annotations.json")["frames"]])
    candidates = old["per_frame_red_shoe_components"]
    variants = []
    nominal = None
    for focal in (510.0, 531.0, old_focal, 600.0):
        for roll in (.010, .025, .040):
            height, up = own_plane(stripes, focal, roll, cx, cy)
            intrinsics = [focal, focal, cx, cy]
            markers = unit_height_plane(DISC_MARKS, intrinsics, up)
            ground_markers = markers - (markers @ up)[:, None] * up
            heads = estimate_head(candidates, backproject(*head_xy.T, np.ones(len(head_xy)), intrinsics), intrinsics, up)
            row = {
                "focal_px": float(focal), "horizon_slope": roll, "horizon_y_at_cx_px": height,
                "ground_up": up.tolist(),
                "pitch_deg": float(np.rad2deg(np.arcsin(-up[2]))),
                "disc_distances_horizontal_m": np.linalg.norm(ground_markers, axis=1).tolist(),
                "disc_perimeter_edge_lengths_m": np.linalg.norm(markers - np.roll(markers, 1, axis=0), axis=1).tolist(),
                "head_z_m": summary(heads[:, 2]),
                "head_height_m": summary(heads @ up + RECORDED_CAMERA_HEIGHT),
                "head_difference_from_old_same_height_scale_m": summary(np.linalg.norm(heads - old_in_m, axis=1)),
            }
            variants.append(row)
            if (focal, roll) == (531.0, .025):
                nominal = dict(row, head_centers_exo_camera_m=heads.tolist())
    if nominal is None:
        raise ValueError("The nominal calibration variant is missing.")
    result = {
        "assumptions": "The two observed white field lines are perpendicular, the principal point is centered, "
                       "the crop has near-pinhole geometry, and the support points are near the ground. "
                       "Labeled blue-disc correspondence and optical center are uncalibrated.",
        "exo_sha256": old["exo_sha256"],
        "field_line_fits_by_frame": raw,
        "field_line_fits_median": stripes.tolist(),
        "camera_height_reference_m": RECORDED_CAMERA_HEIGHT,
        "blue_disc_markers_perimeter_order_xy": [list(marker) for marker in DISC_MARKS],
        "recorded_vipe": {
            "image_focal_px": float(old_focal), "horizon_y_at_cx_px": float(old_horizon),
            "horizon_slope": float(old_roll), "pitch_deg": float(np.rad2deg(np.arcsin(-old_up[2]))),
            "head_height_scaled_to_reference_m": summary(old_in_m @ old_up + RECORDED_CAMERA_HEIGHT),
        },
        "nominal": nominal,
        "sensitivity_variants": variants,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "field-calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_overlay(frames[0], stripes, (old_horizon, old_roll),
                  (nominal["horizon_y_at_cx_px"], nominal["horizon_slope"]), cx, output / "field-calibration.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("position_results", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    audit(options.position_results, options.output)
