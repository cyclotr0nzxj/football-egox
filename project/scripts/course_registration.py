#!/usr/bin/env python3
"""Reproducible ORB matching and planar registration for the week-8 report.

The controlled warp has a known homography.  Its evaluation grid is fixed before
feature extraction and is never used to estimate the homography.  The temporal
experiment has no registration ground truth and therefore reports diagnostics,
not an accuracy score.  No EGO image is used by either experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SEED = 42
PARAMETERS = {
    "orb_nfeatures": 5000,
    "orb_scaleFactor": 1.2,
    "orb_nlevels": 8,
    "orb_fastThreshold": 12,
    "knn_k": 2,
    "lowe_ratio": 0.75,
    "ransac_reprojection_threshold_px": 3.0,
    "ransac_max_iterations": 5000,
    "ransac_confidence": 0.995,
    "grid_columns": 17,
    "grid_rows": 11,
    "grid_margin_fraction": 0.075,
}


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Cannot write image: {path}")


def caption(image: np.ndarray, text: str) -> np.ndarray:
    result = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    font_scale = 0.65
    label_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    if label_width > image.shape[1] - 32:
        font_scale *= (image.shape[1] - 32) / label_width
    cv2.putText(result, text, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def fixed_grid(width: int, height: int) -> np.ndarray:
    margin = PARAMETERS["grid_margin_fraction"]
    xs = np.linspace(margin * (width - 1), (1 - margin) * (width - 1), PARAMETERS["grid_columns"])
    ys = np.linspace(margin * (height - 1), (1 - margin) * (height - 1), PARAMETERS["grid_rows"])
    return np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float64)


def transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def summary(values: np.ndarray) -> dict:
    return {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "maximum_px": float(np.max(values)),
        "point_count": int(values.size),
    }


def keypoint_figure(image: np.ndarray, keypoints: list, label: str) -> np.ndarray:
    strongest = sorted(keypoints, key=lambda key: key.response, reverse=True)[:700]
    figure = cv2.drawKeypoints(image, strongest, None, color=(0, 230, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return caption(figure, f"{label}: {len(keypoints)} ORB keypoints (strongest {len(strongest)} shown)")


def match_and_register(source: np.ndarray, target: np.ndarray, folder: Path, controlled: bool) -> tuple[dict, np.ndarray | None]:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(SEED)
    cv2.setNumThreads(1)
    orb = cv2.ORB_create(
        nfeatures=PARAMETERS["orb_nfeatures"],
        scaleFactor=PARAMETERS["orb_scaleFactor"],
        nlevels=PARAMETERS["orb_nlevels"],
        fastThreshold=PARAMETERS["orb_fastThreshold"],
    )
    source_kp, source_desc = orb.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None)
    target_kp, target_desc = orb.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    prefix = "CONTROLLED" if controlled else "REAL TEMPORAL"
    write_image(folder / "source.png", caption(source, f"{prefix}: source frame"))
    write_image(folder / "target.png", caption(target, f"{prefix}: {'known synthetic perspective warp' if controlled else 'later frame, same EXO camera'}"))
    write_image(folder / "source_keypoints.png", keypoint_figure(source, source_kp, prefix + " source"))
    write_image(folder / "target_keypoints.png", keypoint_figure(target, target_kp, prefix + " target"))
    metrics = {
        "experiment_type": "controlled_synthetic_perspective" if controlled else "real_same_camera_temporal",
        "source_keypoints": len(source_kp),
        "target_keypoints": len(target_kp),
        "knn_query_count": 0,
        "ratio_test_match_count": 0,
        "ransac_inlier_count": 0,
        "ransac_inlier_ratio": 0.0,
        "status": "not_estimated",
    }
    if source_desc is None or target_desc is None:
        metrics["failure"] = "Descriptors unavailable."
        return metrics, None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_desc, target_desc, k=PARAMETERS["knn_k"])
    matches = [neighbors[0] for neighbors in knn if len(neighbors) == 2 and neighbors[0].distance < PARAMETERS["lowe_ratio"] * neighbors[1].distance]
    matches.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
    metrics["knn_query_count"] = len(knn)
    metrics["ratio_test_match_count"] = len(matches)
    shown = matches[:80]
    figure = cv2.drawMatches(source, source_kp, target, target_kp, shown, None, matchColor=(0, 230, 255), singlePointColor=(180, 180, 180), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    write_image(folder / "ratio_matches.png", caption(figure, f"{prefix}: ratio test {len(matches)} matches; strongest {len(shown)} shown"))
    if len(matches) < 4:
        metrics["failure"] = "Fewer than four ratio-test matches."
        return metrics, None
    source_points = np.float32([source_kp[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    target_points = np.float32([target_kp[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    estimated_h, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        PARAMETERS["ransac_reprojection_threshold_px"],
        maxIters=PARAMETERS["ransac_max_iterations"],
        confidence=PARAMETERS["ransac_confidence"],
    )
    if estimated_h is None or mask is None or not np.isfinite(estimated_h).all():
        metrics["failure"] = "RANSAC did not return a finite homography."
        return metrics, None
    estimated_h /= estimated_h[2, 2]
    inliers = mask.ravel().astype(bool)
    metrics.update({
        "status": "estimated",
        "H_source_to_target": estimated_h.tolist(),
        "ransac_inlier_count": int(inliers.sum()),
        "ransac_inlier_ratio": float(inliers.mean()),
    })
    # These fit residuals are explicitly diagnostics, not held-out accuracy.
    projected = transform(source_points.reshape(-1, 2).astype(np.float64), estimated_h)
    residuals = np.linalg.norm(projected - target_points.reshape(-1, 2), axis=1)
    metrics["inlier_fit_residual_diagnostic_only"] = summary(residuals[inliers])
    strongest_inliers = [match for match, is_inlier in zip(matches, inliers) if is_inlier][:100]
    ransac_figure = cv2.drawMatches(source, source_kp, target, target_kp, strongest_inliers, None, matchColor=(30, 230, 60), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    write_image(folder / "ransac_inliers.png", caption(ransac_figure, f"{prefix}: RANSAC {int(inliers.sum())}/{len(matches)} inliers; up to 100 shown"))
    np.savetxt(folder / "H_est.txt", estimated_h, fmt="%.12g")
    height, width = source.shape[:2]
    inverse_h = np.linalg.inv(estimated_h)
    registered = cv2.warpPerspective(target, inverse_h, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(target.shape[:2], 255, dtype=np.uint8), inverse_h, (width, height), flags=cv2.INTER_NEAREST) > 0
    overlay = cv2.addWeighted(source, 0.5, registered, 0.5, 0)
    overlay[~valid] = source[~valid]
    rows, columns = np.indices((height, width))
    board = (rows // 60 + columns // 60) % 2 == 0
    checker = source.copy()
    checker[board & valid] = registered[board & valid]
    write_image(folder / "registered_target.png", caption(registered, f"{prefix}: target mapped back to source with inverse H_est"))
    write_image(folder / "registration_overlay.png", caption(overlay, f"{prefix}: source / registered target 50% overlay"))
    write_image(folder / "registration_checkerboard.png", caption(checker, f"{prefix}: source / registered target checkerboard"))
    return metrics, estimated_h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--temporal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, temporal = load_image(args.source), load_image(args.temporal)
    if source.shape != temporal.shape:
        raise ValueError("Source and temporal frame must have identical dimensions.")
    args.output.mkdir(parents=True, exist_ok=True)
    height, width = source.shape[:2]
    source_corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target_corners = np.float32([[0.038 * width, 0.046 * height], [0.944 * width, 0.015 * height], [0.983 * width, 0.959 * height], [0.019 * width, 0.911 * height]])
    true_h = cv2.getPerspectiveTransform(source_corners, target_corners)
    synthetic = cv2.warpPerspective(source, true_h, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    controlled, controlled_h = match_and_register(source, synthetic, args.output / "controlled", controlled=True)
    controlled["H_true_source_to_target"] = true_h.tolist()
    controlled["accuracy_evaluation"] = "Fixed 17 x 11 image grid; points are independent of ORB features and are not used for fitting."
    np.savetxt(args.output / "controlled" / "H_true.txt", true_h, fmt="%.12g")
    grid = fixed_grid(width, height)
    if controlled_h is not None:
        expected = transform(grid, true_h)
        predicted = transform(grid, controlled_h)
        errors = np.linalg.norm(predicted - expected, axis=1)
        controlled["independent_grid_transfer_error"] = summary(errors)
        np.savetxt(args.output / "controlled" / "grid_evaluation.csv", np.column_stack([grid, expected, predicted, errors]), delimiter=",", header="source_x,source_y,true_target_x,true_target_y,estimated_target_x,estimated_target_y,error_px", comments="", fmt="%.8f")
        evaluation_figure = synthetic.copy()
        for true_point, estimated_point in zip(expected, predicted):
            t, e = tuple(np.rint(true_point).astype(int)), tuple(np.rint(estimated_point).astype(int))
            cv2.circle(evaluation_figure, t, 3, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.drawMarker(evaluation_figure, e, (230, 0, 230), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        write_image(args.output / "controlled" / "independent_grid.png", caption(evaluation_figure, "CONTROLLED: independent grid, yellow = known H; magenta = estimated H (no error magnification)"))
    temporal_metrics, temporal_h = match_and_register(source, temporal, args.output / "temporal", controlled=False)
    temporal_metrics["accuracy_evaluation"] = "Not available: this real pair has no ground-truth homography. Fit residual and identity displacement are diagnostics, not accuracy."
    if temporal_h is not None:
        displacements = np.linalg.norm(transform(grid, temporal_h) - grid, axis=1)
        temporal_metrics["estimated_grid_displacement_from_identity_diagnostic_only"] = summary(displacements)
    report = {
        "seed": SEED,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "parameters": PARAMETERS,
        "source_image": str(args.source.resolve()),
        "temporal_image": str(args.temporal.resolve()),
        "image_size": {"width": width, "height": height},
        "controlled": controlled,
        "temporal": temporal_metrics,
        "limitations": [
            "Controlled experiment is a synthetic planar warp of one real frame, not real camera/viewpoint conversion.",
            "Temporal experiment uses two EXO frames from the same fixed camera, not EGO-to-EXO registration.",
            "A single homography does not model general 3D viewpoint change or moving people and balls.",
            "Temporal homography has no ground truth and is not reported as real registration accuracy.",
        ],
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "controlled_status": controlled["status"], "temporal_status": temporal_metrics["status"], "controlled_grid_error": controlled.get("independent_grid_transfer_error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
