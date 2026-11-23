#!/usr/bin/env python3
"""Independently audit exported Harris/SIFT position-repeatability results.

This script does not import course_feature_detection or rerun its detectors.
It reconstructs the declared transforms with scalar trigonometry, rebuilds the
valid masks, and uses a simple augmenting-path matcher rather than the producer's
Hopcroft-Karp implementation. Only the requested audit JSON is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, MARGIN = 960, 540, 16
DISTANCE, BUDGET, TOLERANCE = 5.0, 500, 3.0
METHODS = ("harris", "sift")
CONDITIONS = ("identity", "rotation15", "rotation45", "scale075", "scale125", "gaussian")
IMAGE_IDS = {f"{view}_f{frame:04d}" for view in ("exo", "ego") for frame in (50, 200)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def known_transform(condition: str) -> np.ndarray:
    if condition in ("identity", "gaussian"):
        return np.eye(3)
    radians = math.radians({"rotation15": 15.0, "rotation45": 45.0}.get(condition, 0.0))
    scale = {"scale075": 0.75, "scale125": 1.25}.get(condition, 1.0)
    alpha, beta = scale * math.cos(radians), scale * math.sin(radians)
    cx, cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    return np.array([[alpha, beta, (1 - alpha) * cx - beta * cy],
                     [-beta, alpha, beta * cx + (1 - alpha) * cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points[:, :2], np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def lookup(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    pixels = np.rint(points[:, :2]).astype(int)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < WIDTH)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < HEIGHT))
    answer = np.zeros(len(points), dtype=bool)
    answer[inside] = mask[pixels[inside, 1], pixels[inside, 0]] > 0
    return answer


def rebuild_mask(homography: np.ndarray) -> np.ndarray:
    # Rebuild from the stated raster contract, not from a producer helper.
    warped = cv2.warpPerspective(np.full((HEIGHT, WIDTH), 255, np.uint8),
                                homography, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.erode(warped, np.ones((33, 33), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def augmenting_path_cardinality(adjacency: list[list[int]], right_count: int) -> int:
    """Maximum bipartite cardinality via independent depth-first augmentation."""
    right_owner = [-1] * right_count

    def augment(source: int, visited: set[int]) -> bool:
        for target in adjacency[source]:
            if target in visited:
                continue
            visited.add(target)
            if right_owner[target] < 0 or augment(right_owner[target], visited):
                right_owner[target] = source
                return True
        return False

    return sum(augment(source, set()) for source in range(len(adjacency)))


def check_positions(points: np.ndarray, context: str) -> None:
    require(points.ndim == 2 and points.shape[1] == 5, f"{context}: expected N x 5 positions")
    require(np.isfinite(points).all(), f"{context}: nonfinite position/attribute")
    require(len(points) <= BUDGET, f"{context}: more than 500 retained positions")
    require(len({tuple(p) for p in np.round(points[:, :2], 3)}) == len(points),
            f"{context}: duplicated rounded centers")
    if len(points) > 1:
        distances = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=2)
        np.fill_diagonal(distances, np.inf)
        require(float(distances.min()) >= DISTANCE - 1e-9, f"{context}: spacing below 5px")


def audit(results: Path, frames_root: Path) -> dict:
    metrics = read_json(results / "metrics.json")
    producer_audit = read_json(results / "audit.json")
    parameters = metrics["parameters"]
    for key, expected in {
        "image_size": [WIDTH, HEIGHT], "unique_position_rounding_decimals": 3,
        "minimum_position_distance_px": DISTANCE, "maximum_retained_positions": BUDGET,
        "common_visibility_margin_px": MARGIN, "repeatability_tolerance_target_px": TOLERANCE,
    }.items():
        require(parameters.get(key) == expected, f"Unexpected declared parameter {key}")
    require(set(metrics["conditions"]) == set(CONDITIONS), "Condition inventory differs")
    require({item["image_id"] for item in metrics["images"]} == IMAGE_IDS
            and len(metrics["images"]) == 4, "Expected the four declared EXO/EGO frames")

    source_hashes = []
    for item in metrics["images"]:
        path = frames_root / item["view"] / f"{item['visible_frame']:04d}.jpg"
        require(item["image_id"] == f"{item['view']}_f{item['visible_frame']:04d}", "Image ID mismatch")
        with Image.open(path) as image:
            require(image.size == (WIDTH, HEIGHT), f"Wrong source dimensions: {path}")
        actual = sha256(path)
        require(actual == item["sha256"], f"Source hash mismatch: {path}")
        source_hashes.append({"image_id": item["image_id"], "sha256": actual, "matches_record": True})

    transforms, masks = {}, {}
    for condition in CONDITIONS:
        homography = known_transform(condition)
        declared = np.asarray(metrics["conditions"][condition]["H_source_to_target"], dtype=float)
        require(np.max(np.abs(homography - declared)) < 1e-12, f"Incorrect H: {condition}")
        expected = rebuild_mask(homography)
        with Image.open(results / "valid_masks" / f"{condition}.png") as image:
            stored = np.asarray(image)
        require(np.array_equal(expected, stored), f"Valid-mask mismatch: {condition}")
        require(not stored[:MARGIN].any() and not stored[-MARGIN:].any()
                and not stored[:, :MARGIN].any() and not stored[:, -MARGIN:].any(),
                f"Outer 16px margin missing: {condition}")
        transforms[condition], masks[condition] = homography, expected
    identity_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    identity_mask[16:524, 16:944] = 255
    require(np.array_equal(masks["identity"], identity_mask), "Identity margin is incorrect")

    expected_keys = {(image_id, condition, method) for image_id in IMAGE_IDS
                     for condition in CONDITIONS for method in METHODS}
    metric_rows = {(r["image_id"], r["condition"], r["method"]): r for r in metrics["records"]}
    require(set(metric_rows) == expected_keys and len(metrics["records"]) == 48,
            "Expected 48 distinct metric rows")
    producer_rows = {(r["image_id"], r["condition"], r["method"]): r
                     for r in producer_audit["per_case_checks"]}
    require(set(producer_rows) == expected_keys and len(producer_audit["per_case_checks"]) == 48,
            "Producer audit inventory differs")

    reconstructed = []
    coordinate_digest = hashlib.sha256()
    coordinate_files = sorted((results / "coordinates").rglob("*.json"))
    require(len(coordinate_files) == 24, "Expected 24 coordinate JSON files")
    seen_coordinates, base_positions = set(), {}
    max_projection_difference = 0.0
    for path in coordinate_files:
        data = read_json(path)
        image_id, condition = data["image_id"], data["condition"]
        require((image_id, condition) not in seen_coordinates, "Duplicate coordinate case")
        require(image_id in IMAGE_IDS and condition in CONDITIONS, "Unknown coordinate case")
        seen_coordinates.add((image_id, condition))
        coordinate_digest.update(str(path.relative_to(results)).encode("utf-8"))
        coordinate_digest.update(b"\0" + path.read_bytes() + b"\0")
        homography, target_mask = transforms[condition], masks[condition]
        require(np.max(np.abs(homography - np.asarray(data["H_source_to_target"]))) < 1e-12,
                f"Coordinate H mismatch: {path}")
        require(set(data["methods"]) == set(METHODS), f"Unexpected methods: {path}")
        for method in METHODS:
            case = data["methods"][method]
            context = f"{image_id}/{condition}/{method}"
            source = np.asarray(case["source_retained_positions"], dtype=float).reshape(-1, 5)
            target = np.asarray(case["target_retained_positions"], dtype=float).reshape(-1, 5)
            check_positions(source, context + "/source")
            check_positions(target, context + "/target")
            base_key = (image_id, method)
            if base_key in base_positions:
                require(np.array_equal(source, base_positions[base_key]), f"Baseline differs: {context}")
            else:
                base_positions[base_key] = source
            require(lookup(masks["identity"], source).all() and lookup(target_mask, target).all(),
                    f"Retained center outside its detection mask: {context}")
            source_indices = np.flatnonzero(lookup(masks["identity"], source)
                                           & lookup(target_mask, project(source, homography)))
            target_indices = np.flatnonzero(lookup(target_mask, target)
                                           & lookup(masks["identity"], project(target, np.linalg.inv(homography))))
            require(source_indices.tolist() == case["common_source_retained_indices"],
                    f"Common source indices differ: {context}")
            require(target_indices.tolist() == case["common_target_retained_indices"],
                    f"Common target indices differ: {context}")
            projected = project(source[source_indices], homography)
            exported = np.asarray(case["projected_common_source_xy_target_pixels"]).reshape(-1, 2)
            projection_difference = float(np.max(np.abs(projected - exported))) if len(projected) else 0.0
            max_projection_difference = max(max_projection_difference, projection_difference)
            require(projection_difference < 1e-9, f"Exported projection differs: {context}")
            distance = np.linalg.norm(projected[:, None] - target[target_indices, :2][None], axis=2)
            adjacency = [np.flatnonzero(row <= TOLERANCE).tolist() for row in distance]
            count = augmenting_path_cardinality(adjacency, len(target_indices))
            matches = case["matching"]
            require(len(matches) == count, f"Stored matching is not maximum: {context}")
            require(len({p[0] for p in matches}) == count == len({p[1] for p in matches}),
                    f"Stored matching is not one-to-one: {context}")
            for left, right, recorded_distance in matches:
                require(0 <= left < len(source_indices) and 0 <= right < len(target_indices),
                        f"Match index outside common-visible set: {context}")
                require(abs(float(distance[left, right]) - recorded_distance) < 1e-9
                        and distance[left, right] <= TOLERANCE, f"Match exceeds 3 target pixels: {context}")
            ns, nt = len(source_indices), len(target_indices)
            denominator = min(ns, nt)
            value = count / denominator if denominator else None
            expected = {"source_retained_count": len(source), "target_retained_count": len(target),
                        "common_source_count": ns, "common_target_count": nt,
                        "one_to_one_match_count": count, "repeatability_denominator": denominator,
                        "repeatability": value}
            for label, row in [("metrics", metric_rows[(image_id, condition, method)]),
                               ("coordinate summary", case["summary"])]:
                for key, number in expected.items():
                    require(row[key] == number, f"{label} {key} mismatch: {context}")
            certificate = case["matching_certificate"]
            producer = producer_rows[(image_id, condition, method)]
            for row in (certificate, producer):
                require(row["matching_count"] == count and row["minimum_vertex_cover_size"] == count
                        and row["maximum_cardinality_certified"] is True
                        and row["all_threshold_edges_covered"] is True
                        and row["threshold_edge_count"] == sum(map(len, adjacency)),
                        f"Producer certificate disagrees with independent result: {context}")
            if condition == "identity":
                require(value == 1.0 and np.array_equal(source, target), f"Identity failed: {context}")
            reconstructed.append({"image_id": image_id, "condition": condition, "method": method,
                                  "Ns_common_source": ns, "Nt_common_target": nt, "M_matches": count,
                                  "repeatability_M_over_min": value,
                                  "source_coverage_M_over_Ns": count / ns if ns else None,
                                  "target_coverage_M_over_Nt": count / nt if nt else None,
                                  "balanced_coverage_2M_over_sum": 2 * count / (ns + nt) if ns + nt else None})

    macros = []
    require(len(metrics["macro_by_condition"]) == 12, "Expected 12 macro rows")
    require({(row["condition"], row["method"]) for row in metrics["macro_by_condition"]}
            == {(condition, method) for condition in CONDITIONS for method in METHODS},
            "Macro inventory differs")
    for row in metrics["macro_by_condition"]:
        selected = [r for r in reconstructed if r["condition"] == row["condition"] and r["method"] == row["method"]]
        values = [r["repeatability_M_over_min"] for r in selected if r["repeatability_M_over_min"] is not None]
        require(len(selected) == 4 and row["frame_count"] == 4
                and row["valid_frame_count"] == len(values)
                and row["null_frame_count"] == 4 - len(values), "Macro sample count differs")
        mean = sum(values) / len(values) if values else None
        require(mean == row["macro_mean_repeatability"] or
                (mean is not None and abs(mean - row["macro_mean_repeatability"]) < 1e-14),
                "Four-image unweighted macro mean differs")
        macros.append({"condition": row["condition"], "method": row["method"],
                       "valid_images": len(values), "null_images": 4 - len(values),
                       "macro_repeatability": mean,
                       **{key: sum(r[key] for r in selected) / 4 for key in (
                           "source_coverage_M_over_Ns", "target_coverage_M_over_Nt",
                           "balanced_coverage_2M_over_sum")}})

    selected_examples = {("ego_f0200", "rotation45"), ("ego_f0200", "scale075"),
                         ("ego_f0200", "gaussian"), ("exo_f0050", "rotation45")}
    return {
        "status": "passed", "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "No producer import or detector rerun. Scalar-trigonometry H reconstruction, mask rebuild, coordinate visibility checks, independent DFS augmenting-path maximum matching, four-image macro recomputation and source SHA-256 verification.",
        "inputs": {"results_directory": str(results), "frames_root": str(frames_root),
                   "metrics_sha256": sha256(results / "metrics.json"),
                   "producer_audit_sha256": sha256(results / "audit.json"),
                   "coordinate_collection_sha256": coordinate_digest.hexdigest(),
                   "auditor_sha256": sha256(Path(__file__)),
                   "numpy_version": np.__version__, "opencv_version": cv2.__version__},
        "passed_counts": {"source_image_hashes": 4, "known_transforms": 6,
                          "reconstructed_valid_masks": 6, "coordinate_files": 24,
                          "source_and_target_position_sets": 96, "common_visibility_cases": 48,
                          "maximum_cardinality_matchings": 48, "one_to_one_threshold_checks": 48,
                          "identity_controls": 8, "four_image_macro_means": 12,
                          "metric_rows": 48, "producer_certificate_rows": 48},
        "max_projection_difference_px": max_projection_difference,
        "source_hash_checks": source_hashes, "independent_macro_results": macros,
        "denominator_bias_examples": [r for r in reconstructed if (r["image_id"], r["condition"]) in selected_examples],
        "limitations": [
            "Audits exported retained positions; does not independently rerun or validate Harris/SIFT detection, response ranking, or descriptor computation.",
            "Mask reconstruction reuses OpenCV warp/erosion primitives; H construction and the matching algorithm are independent of the producer's Python implementation.",
            "Common visibility tests rounded centers with a 16px square erosion, not full SIFT support regions or continuous 16px clearance.",
            "R=M/min(Ns,Nt) can be high after severe point-set shrinkage; inspect Ns, Nt, M, both directional coverages and balanced coverage together.",
            "Tolerance is 3 target-image pixels; its source-pixel equivalent changes under scaling. All reported values are specific to the 5px/500-location selection policy.",
            "Four selected frames from one scene and synthetic image changes do not establish generalization, real cross-view accuracy, panorama quality, descriptor quality or scale/orientation accuracy.",
            "Single-run timings are not independently benchmarked. No cloud work, training or source-video modification occurs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "output/week08_harris_sift/results")
    parser.add_argument("--frames-root", type=Path, default=ROOT / "output/week08/frames/scene04")
    parser.add_argument("--output", type=Path, default=ROOT / "output/week08_harris_sift/independent_audit.json")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        result = audit(args.results.resolve(), args.frames_root.resolve())
        exit_code = 0
    except Exception as error:
        result = {"status": "failed", "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error)}
        exit_code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "passed_counts": result.get("passed_counts"), "error": result.get("error")}, ensure_ascii=False))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
