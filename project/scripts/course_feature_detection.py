#!/usr/bin/env python3
"""Compare Harris and SIFT position repeatability under known image changes.

Four real EXO/EGO frames supply the image content. All rotations/scales/blur
conditions are controlled synthetic changes, not new camera observations.
Descriptors are not computed or fitted; correspondences use the known H.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SEED, WIDTH, HEIGHT = 42, 960, 540
MARGIN, MIN_DISTANCE, MAX_POINTS, TOLERANCE = 16, 5.0, 500, 3.0
METHODS = ["harris", "sift"]
CONDITIONS = ["identity", "rotation15", "rotation45", "scale075", "scale125", "gaussian"]
PARAMETERS = {
    "seed": SEED, "opencv_threads": 1, "image_size": [WIDTH, HEIGHT],
    "harris": {"blockSize": 2, "ksize": 3, "k": 0.04, "positive_response_threshold_fraction": 0.01, "local_nms_window": 3},
    "sift": {"nfeatures": 500, "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
    "unique_position_rounding_decimals": 3, "minimum_position_distance_px": MIN_DISTANCE,
    "maximum_retained_positions": MAX_POINTS, "common_visibility_margin_px": MARGIN,
    "repeatability_tolerance_target_px": TOLERANCE, "gaussian_kernel": [9, 9], "gaussian_sigma": 2.0,
}
FONT_CANDIDATES = [
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "DejaVuSansMono.ttf",
]


def font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default(size=size)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        raise IOError(path)


def project(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.column_stack([points[:, :2], np.ones(len(points))]) @ h.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def mask_at(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    rounded = np.rint(points[:, :2]).astype(int)
    inside = (rounded[:, 0] >= 0) & (rounded[:, 0] < WIDTH) & (rounded[:, 1] >= 0) & (rounded[:, 1] < HEIGHT)
    answer = np.zeros(len(points), dtype=bool)
    answer[inside] = mask[rounded[inside, 1], rounded[inside, 0]] > 0
    return answer


def valid_mask(h: np.ndarray) -> np.ndarray:
    visible = cv.warpPerspective(np.full((HEIGHT, WIDTH), 255, np.uint8), h, (WIDTH, HEIGHT), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    # The square support excludes at least 16 target pixels around invalid warp
    # regions and all four image boundaries, rather than treating black as data.
    return cv.erode(visible, np.ones((2*MARGIN+1, 2*MARGIN+1), np.uint8), borderType=cv.BORDER_CONSTANT, borderValue=0)


def transforms() -> dict[str, np.ndarray]:
    center = ((WIDTH-1)/2, (HEIGHT-1)/2)
    answer = {"identity": np.eye(3), "gaussian": np.eye(3)}
    for name, angle, scale in [("rotation15", 15, 1), ("rotation45", 45, 1), ("scale075", 0, .75), ("scale125", 0, 1.25)]:
        answer[name] = np.vstack([cv.getRotationMatrix2D(center, angle, scale), [0, 0, 1]])
    return answer


def detect(image: np.ndarray, mask: np.ndarray, method: str) -> dict:
    cv.setRNGSeed(SEED)
    started = time.perf_counter()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    response = None
    if method == "harris":
        response = cv.cornerHarris(gray.astype(np.float32)/255.0, blockSize=2, ksize=3, k=.04)
        maximum = float(np.max(response[mask > 0]))
        threshold = .01 * max(maximum, 0.0)
        positive = (response > threshold) & (response > 0) & (mask > 0)
        raw_count = int(positive.sum())
        local_max = positive & (response == cv.dilate(response, np.ones((3, 3), np.uint8)))
        ys, xs = np.nonzero(local_max)
        candidates = [(float(x), float(y), float(response[y, x]), 3.0, -1.0) for x, y in zip(xs, ys)]
        nms_count = len(candidates)
    else:
        detector = cv.SIFT_create(**PARAMETERS["sift"])
        keypoints = detector.detect(gray, mask)
        raw_count = len(keypoints)
        candidates = [(float(k.pt[0]), float(k.pt[1]), float(k.response), float(k.size), float(k.angle)) for k in keypoints]
        nms_count = None
        threshold = None
    candidates.sort(key=lambda p: (-p[2], p[1], p[0], p[4], p[3]))
    # Multiple SIFT orientations at the same physical center are one position.
    unique, seen = [], set()
    for candidate in candidates:
        identity = (round(candidate[0], 3), round(candidate[1], 3))
        if identity not in seen:
            unique.append(candidate)
            seen.add(identity)
    retained = []
    for candidate in unique:
        if all((candidate[0]-p[0])**2 + (candidate[1]-p[1])**2 >= MIN_DISTANCE**2 for p in retained):
            retained.append(candidate)
            if len(retained) == MAX_POINTS:
                break
    elapsed = (time.perf_counter()-started)*1000
    points = np.asarray(retained, dtype=np.float64).reshape(-1, 5)
    return {"points": points, "response": response, "statistics": {
        "raw_count": raw_count, "after_local_nms_count": nms_count,
        "unique_position_count": len(unique), "retained_count": len(retained),
        "harris_threshold": threshold, "detection_and_selection_ms": elapsed,
    }}


def maximum_matching(source: np.ndarray, target: np.ndarray) -> tuple[list[tuple[int, int, float]], dict]:
    """Hopcroft-Karp; validate maximum cardinality using a vertex-cover proof."""
    distance = np.linalg.norm(source[:, None, :]-target[None, :, :], axis=2) if len(source) and len(target) else np.empty((len(source), len(target)))
    adjacency = [np.flatnonzero(row <= TOLERANCE).tolist() for row in distance]
    left = [-1]*len(source)
    right = [-1]*len(target)
    levels = [-1]*len(source)

    def bfs() -> bool:
        queue = deque()
        for u in range(len(source)):
            levels[u] = 0 if left[u] < 0 else -1
            if left[u] < 0:
                queue.append(u)
        found = False
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if right[v] < 0:
                    found = True
                elif levels[right[v]] < 0:
                    levels[right[v]] = levels[u]+1
                    queue.append(right[v])
        return found

    def dfs(u: int) -> bool:
        for v in adjacency[u]:
            if right[v] < 0 or (levels[right[v]] == levels[u]+1 and dfs(right[v])):
                left[u], right[v] = v, u
                return True
        levels[u] = -1
        return False

    while bfs():
        for u in range(len(source)):
            if left[u] < 0:
                dfs(u)
    matches = [(u, v, float(distance[u, v])) for u, v in enumerate(left) if v >= 0]
    # Konig certificate: an alternating traversal yields a covering vertex set
    # whose size equals the matching, certifying that no larger matching exists.
    reachable_left = {u for u, v in enumerate(left) if v < 0}
    reachable_right = set()
    queue = deque(reachable_left)
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if left[u] != v and v not in reachable_right:
                reachable_right.add(v)
                if right[v] >= 0 and right[v] not in reachable_left:
                    reachable_left.add(right[v])
                    queue.append(right[v])
    cover_left = set(range(len(source)))-reachable_left
    cover_right = reachable_right
    covered = all(u in cover_left or v in cover_right for u, neighbors in enumerate(adjacency) for v in neighbors)
    assert covered and len(cover_left)+len(cover_right) == len(matches)
    assert len({m[0] for m in matches}) == len(matches) == len({m[1] for m in matches})
    assert all(m[2] <= TOLERANCE for m in matches)
    certificate = {"maximum_cardinality_certified": True, "matching_count": len(matches), "minimum_vertex_cover_size": len(cover_left)+len(cover_right), "all_threshold_edges_covered": covered, "threshold_edge_count": sum(map(len, adjacency)), "maximum_matched_distance_target_px": max([m[2] for m in matches], default=None)}
    return matches, certificate


def point_picture(image: np.ndarray, result: dict, method: str) -> np.ndarray:
    if method == "sift":
        keypoints = [cv.KeyPoint(float(p[0]), float(p[1]), float(p[3]), float(p[4]), float(p[2])) for p in result["points"]]
        return cv.drawKeypoints(image, keypoints, None, color=(0, 230, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    canvas = image.copy()
    for p in result["points"]:
        cv.circle(canvas, tuple(np.rint(p[:2]).astype(int)), 2, (0, 70, 255), -1, cv.LINE_AA)
    return canvas


def positive_heatmap(response: np.ndarray, mask: np.ndarray) -> np.ndarray:
    maximum = max(float(response[mask > 0].max()), 1e-20)
    intensity = np.log1p(1000*np.clip(response/maximum, 0, 1))/np.log1p(1000)
    colored = cv.applyColorMap(np.uint8(np.rint(255*intensity)), cv.COLORMAP_TURBO)
    colored[mask == 0] = (25, 25, 25)
    return colored


def panel(canvas: Image.Image, image: np.ndarray, label: str, column: int, row: int, cell_width: int = 480, picture_height: int = 270) -> None:
    x, y = column*cell_width, row*(picture_height+44)
    im = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    im = ImageOps.contain(im, (cell_width, picture_height))
    canvas.paste(im, (x+(cell_width-im.width)//2, y+44+(picture_height-im.height)//2))
    draw = ImageDraw.Draw(canvas)
    text_font = font(16)
    if draw.textlength(label, font=text_font) > cell_width-12:
        text_font = font(13)
    draw.text((x+6, y+13), label, fill="white", font=text_font)


def visibility(source: np.ndarray, target: np.ndarray, h: np.ndarray, source_mask: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_common = mask_at(source_mask, source) & mask_at(target_mask, project(source[:, :2], h))
    target_common = mask_at(target_mask, target) & mask_at(source_mask, project(target[:, :2], np.linalg.inv(h)))
    return np.flatnonzero(source_common), np.flatnonzero(target_common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, default=Path("output/week08/frames/scene04"))
    parser.add_argument("--output", type=Path, default=Path("output/week08_harris_sift/results"))
    args = parser.parse_args()
    frame_root = args.frames_root if args.frames_root.is_absolute() else ROOT/args.frames_root
    output = args.output if args.output.is_absolute() else ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    cv.setNumThreads(1)
    cv.setRNGSeed(SEED)
    matrices = transforms()
    source_mask = valid_mask(np.eye(3))
    records, images_metadata, audits, cached = [], [], [], {}
    conditions_metadata = {condition: {"H_source_to_target": matrices[condition].tolist(), "is_controlled_synthetic_change": condition != "identity", "operation": condition} for condition in CONDITIONS}
    for view in ["exo", "ego"]:
        for index in [50, 200]:
            image_id = f"{view}_f{index:04d}"
            source_path = frame_root/view/f"{index:04d}.jpg"
            image = cv.imread(str(source_path))
            assert image is not None and image.shape == (HEIGHT, WIDTH, 3)
            metadata = {"image_id": image_id, "view": view, "visible_frame": index, "path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
            images_metadata.append(metadata)
            source_results = {method: detect(image, source_mask, method) for method in METHODS}
            cached[image_id] = {"source": image, "baseline": source_results, "conditions": {}}
            image_save(output/"source_heatmaps"/(image_id+".png"), positive_heatmap(source_results["harris"]["response"], source_mask))
            for condition in CONDITIONS:
                h = matrices[condition]
                transformed = cv.GaussianBlur(image, (9, 9), 2.0) if condition == "gaussian" else cv.warpPerspective(image, h, (WIDTH, HEIGHT), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
                mask = valid_mask(h)
                image_save(output/"valid_masks"/(condition+".png"), mask) if image_id == "exo_f0050" else None
                if condition != "identity":
                    image_save(output/"transformed_images"/image_id/(condition+".jpg"), transformed)
                coordinate = {"image_id": image_id, "condition": condition, "H_source_to_target": h.tolist(), "coordinate_columns": ["x", "y", "response", "size", "angle"], "coordinate_units": "pixel centers; matching distances are in target image pixels", "visibility_rule": "Both centers must belong to their eroded valid masks after forward/inverse mapping; 16-pixel square erosion excludes image and warp boundaries.", "methods": {}}
                cached[image_id]["conditions"][condition] = {"image": transformed, "methods": {}}
                for method in METHODS:
                    base = source_results[method]
                    changed = detect(transformed, mask, method)
                    common_source, common_target = visibility(base["points"], changed["points"], h, source_mask, mask)
                    projected = project(base["points"][common_source, :2], h)
                    target_points = changed["points"][common_target, :2]
                    matches, certificate = maximum_matching(projected, target_points)
                    denominator = min(len(common_source), len(common_target))
                    repeatability = len(matches)/denominator if denominator else None
                    record = {"image_id": image_id, "view": view, "visible_frame": index, "condition": condition, "method": method,
                              "source_raw_count": base["statistics"]["raw_count"], "source_unique_count": base["statistics"]["unique_position_count"], "source_retained_count": len(base["points"]),
                              "target_raw_count": changed["statistics"]["raw_count"], "target_unique_count": changed["statistics"]["unique_position_count"], "target_retained_count": len(changed["points"]),
                              "common_source_count": len(common_source), "common_target_count": len(common_target), "one_to_one_match_count": len(matches), "repeatability_denominator": denominator, "repeatability": repeatability,
                              "source_detection_and_selection_ms": base["statistics"]["detection_and_selection_ms"], "target_detection_and_selection_ms": changed["statistics"]["detection_and_selection_ms"],
                              "coordinate_file": f"coordinates/{image_id}/{condition}.json"}
                    records.append(record)
                    coordinate["methods"][method] = {"source_statistics": base["statistics"], "target_statistics": changed["statistics"], "source_retained_positions": base["points"].tolist(), "target_retained_positions": changed["points"].tolist(), "common_source_retained_indices": common_source.tolist(), "common_target_retained_indices": common_target.tolist(), "projected_common_source_xy_target_pixels": projected.tolist(), "matching_columns": ["common_source_index", "common_target_index", "distance_target_px"], "matching": matches, "matching_certificate": certificate, "summary": record}
                    cached[image_id]["conditions"][condition]["methods"][method] = {"detection": changed, "common_source": common_source, "common_target": common_target, "matches": matches, "record": record}
                    grid = np.array([[16, 16], [943, 16], [943, 523], [16, 523], [479.5, 269.5]], dtype=float)
                    roundtrip = float(np.max(np.linalg.norm(project(project(grid, h), np.linalg.inv(h))-grid, axis=1)))
                    assert roundtrip < 1e-9
                    assert mask_at(mask, projected).all()
                    assert mask_at(source_mask, project(target_points, np.linalg.inv(h))).all()
                    assert len({tuple(np.round(p[:2], 3)) for p in changed["points"]}) == len(changed["points"])
                    if len(changed["points"]) > 1:
                        distances = np.linalg.norm(changed["points"][:, None, :2]-changed["points"][None, :, :2], axis=2)
                        np.fill_diagonal(distances, np.inf)
                        assert float(distances.min()) >= MIN_DISTANCE-1e-9
                    if condition == "identity":
                        assert np.array_equal(image, transformed) and repeatability == 1.0 and len(matches) == denominator
                    audits.append({"image_id": image_id, "condition": condition, "method": method, "identity_check": "passed" if condition == "identity" else "not_applicable", "homography_roundtrip_max_px": roundtrip, "common_visibility_check": "passed", "unique_positions_and_min_distance_check": "passed", **certificate})
                write_json(output/"coordinates"/image_id/(condition+".json"), coordinate)
            print(image_id, "finished", flush=True)

    macro = []
    for condition in CONDITIONS:
        for method in METHODS:
            selected = [r for r in records if r["condition"] == condition and r["method"] == method]
            values = [r["repeatability"] for r in selected if r["repeatability"] is not None]
            macro.append({"condition": condition, "method": method, "frame_count": len(selected), "valid_frame_count": len(values), "null_frame_count": len(selected)-len(values), "macro_mean_repeatability": float(np.mean(values)) if values else None, "min_frame_repeatability": min(values) if values else None, "max_frame_repeatability": max(values) if values else None, "mean_target_detection_and_selection_ms_descriptive": float(np.mean([r["target_detection_and_selection_ms"] for r in selected]))})
    limitations = ["Four selected real frames from one short scene; results do not measure generalization.", "Rotations, scales and blur are controlled image-space changes, not new physical camera observations or genuine panorama stitching.", "SIFT orientations at identical centers count as one location. Position repeatability does not evaluate orientation, scale, descriptors or semantic correctness.", "Both methods use at most 500 retained locations and minimum spacing 5px. SIFT's internal nfeatures limit occurs before spatial deduplication, so retained counts can differ; more points is not a quality ranking.", "The 16px center margin cannot guarantee that every large SIFT support region stays entirely inside the visible domain.", "Timing is one descriptive CPU observation per condition including detection and spatial selection, not a repeated performance benchmark."]
    metrics = {"schema_version": 1, "run_timestamp_utc": datetime.now(timezone.utc).isoformat(), "opencv_version": cv.__version__, "numpy_version": np.__version__, "parameters": PARAMETERS, "images": images_metadata, "conditions": conditions_metadata, "repeatability_definition": "Maximum-cardinality one-to-one 2D point matches within 3 target pixels / min(common-visible source count, common-visible target count); null if denominator is zero. Condition means are unweighted macro averages over four frames.", "records": records, "macro_by_condition": macro, "limitations": limitations}
    write_json(output/"metrics.json", metrics)
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    for view in ["exo", "ego"]:
        canvas = Image.new("RGB", (1920, 628), "#18212b")
        for row, index in enumerate([50, 200]):
            item = cached[f"{view}_f{index:04d}"]
            panel(canvas, item["source"], f"{view.upper()} F{index} | original", 0, row)
            panel(canvas, positive_heatmap(item["baseline"]["harris"]["response"], source_mask), "Harris positive response | log scale", 1, row)
            for column, method in [(2, "harris"), (3, "sift")]:
                count = len(item["baseline"][method]["points"])
                panel(canvas, point_picture(item["source"], item["baseline"][method], method), f"{method.upper()} | {count} unique centers", column, row)
        canvas.save(output/("overview_"+view+".png"))
        transform_sheet = Image.new("RGB", (1920, 780), "#18212b")
        item = cached[f"{view}_f0050"]
        for column, condition in enumerate(CONDITIONS[1:]):
            changed = item["conditions"][condition]
            panel(transform_sheet, changed["image"], f"{view.upper()} F50 | {condition}", column, 0, 384, 216)
            for row, method in [(1, "harris"), (2, "sift")]:
                record = changed["methods"][method]["record"]
                value = record["repeatability"]
                label = f"{method.upper()} R={value:.3f} ({record['one_to_one_match_count']}/{record['repeatability_denominator']})" if value is not None else f"{method.upper()} R=null"
                panel(transform_sheet, point_picture(changed["image"], changed["methods"][method]["detection"], method), label, column, row, 384, 216)
        transform_sheet.save(output/("controlled_changes_"+view+".png"))

    nonidentity = [r for r in records if r["condition"] != "identity" and r["repeatability"] is not None]
    extremes = [max(nonidentity, key=lambda r: r["repeatability"]), min(nonidentity, key=lambda r: r["repeatability"])]
    canvas = Image.new("RGB", (1440, 628), "#18212b")
    for row, record in enumerate(extremes):
        item = cached[record["image_id"]]
        changed = item["conditions"][record["condition"]]
        method = record["method"]
        evidence = changed["methods"][method]
        source_points = item["baseline"][method]["points"][evidence["common_source"], :2]
        target_points = evidence["detection"]["points"][evidence["common_target"], :2]
        source_picture, target_picture = item["source"].copy(), changed["image"].copy()
        matched_source = {a for a, _, _ in evidence["matches"]}
        matched_target = {b for _, b, _ in evidence["matches"]}
        for n, p in enumerate(source_points):
            cv.circle(source_picture, tuple(np.rint(p).astype(int)), 3, (40, 230, 40) if n in matched_source else (0, 70, 255), 1, cv.LINE_AA)
        for n, p in enumerate(target_points):
            cv.circle(target_picture, tuple(np.rint(p).astype(int)), 3, (40, 230, 40) if n in matched_target else (0, 70, 255), 1, cv.LINE_AA)
        projected = project(source_points, matrices[record["condition"]])
        mapped_picture = changed["image"].copy()
        for p in projected:
            cv.drawMarker(mapped_picture, tuple(np.rint(p).astype(int)), (255, 210, 30), cv.MARKER_CROSS, 7, 1, cv.LINE_AA)
        rank = "HIGHEST OBSERVED" if row == 0 else "LOWEST OBSERVED"
        panel(canvas, source_picture, f"{rank}: {method.upper()} {record['image_id']}", 0, row)
        panel(canvas, mapped_picture, f"Known H projection | {record['condition']}", 1, row)
        panel(canvas, target_picture, f"Target R={record['repeatability']:.3f} | green=matched", 2, row)
    canvas.save(output/"repeatability_examples.png")

    summary = Image.new("RGB", (1400, 700), "white")
    draw = ImageDraw.Draw(summary)
    draw.text((35, 20), "Position repeatability | 4-frame macro average | 3 target pixels", fill="#17212d", font=font(23))
    for tick in range(6):
        y = 580-tick*100
        draw.line((95, y, 1350, y), fill="#d9e0e4", width=1)
        draw.text((30, y-10), f"{tick/5:.1f}", fill="#17212d", font=font(17))
    for index, condition in enumerate(CONDITIONS):
        x = 120+index*205
        for offset, method, color in [(0, "harris", "#d25d43"), (72, "sift", "#2878ae")]:
            record = next(r for r in macro if r["condition"] == condition and r["method"] == method)
            value = record["macro_mean_repeatability"]
            if value is not None:
                top = 580-value*500
                draw.rectangle((x+offset, top, x+offset+58, 580), fill=color)
                draw.text((x+offset-2, top-26), f"{value:.3f}", fill=color, font=font(17))
        draw.text((x-6, 596), condition, fill="#17212d", font=font(15))
    draw.text((95, 645), "HARRIS", fill="#d25d43", font=font(22))
    draw.text((260, 645), "SIFT", fill="#2878ae", font=font(22))
    draw.text((400, 648), "Controlled changes; no universal detector ranking", fill="#586573", font=font(17))
    summary.save(output/"repeatability_summary.png")

    # Independently re-read the exported CSV and recalculate all macro means.
    with (output/"metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    macro_checks = []
    for entry in macro:
        values = [float(r["repeatability"]) for r in csv_rows if r["condition"] == entry["condition"] and r["method"] == entry["method"] and r["repeatability"] != ""]
        recalculated = sum(values)/len(values) if values else None
        assert recalculated == entry["macro_mean_repeatability"] or abs(recalculated-entry["macro_mean_repeatability"]) < 1e-14
        macro_checks.append({"condition": entry["condition"], "method": entry["method"], "csv_macro_mean": recalculated, "status": "passed"})
    image_checks = []
    for path in sorted(output.rglob("*")):
        if path.suffix.lower() in [".png", ".jpg"]:
            with Image.open(path) as image:
                image.verify()
            image_checks.append(str(path.relative_to(output)))
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    assert total_bytes < 60_000_000
    write_json(output/"audit.json", {"status": "passed", "identity_cases_passed": sum(a["condition"] == "identity" for a in audits), "per_case_checks": audits, "macro_csv_recalculation": macro_checks, "decodable_image_count": len(image_checks), "decodable_images": image_checks, "bytes_before_audit_and_readme": total_bytes, "source_images_unchanged": all(hashlib.sha256(Path(im["path"]).read_bytes()).hexdigest() == im["sha256"] for im in images_metadata)})
    (output/"README.md").write_text('# Harris / SIFT controlled repeatability experiment\n\nFour real 960x540 frames: EXO and EGO F50/F200 from paired clip04. Rotations, scale changes and Gaussian blur are **controlled synthetic image changes**, not new viewpoints or panorama results.\n\nFixed detector settings are recorded in metrics.json. Harris uses blockSize2, ksize3, k0.04, 1% of maximum positive response and 3x3 local NMS. SIFT uses nfeatures500, nOctaveLayers3, contrastThreshold0.04, edgeThreshold10 and sigma1.6. Centers rounded to 0.001px are deduplicated before response-ranked 5px minimum spacing and the 500-location budget. SIFT orientation copies are not independent locations. Harris raw_count is the number of thresholded pixels before local NMS; SIFT raw_count is its detector keypoint count. These raw counts have different definitions and are not a detector ranking.\n\nEach controlled warp has a saved source-to-target 3x3 H. A warped valid mask is eroded by a 33x33 square, removing 16px around image/warp boundaries. Source and target centers are filtered by forward/inverse H in the common visible region. Positions are matched using maximum-cardinality one-to-one bipartite matching at 3 **target-image pixels**. The denominator is min(common source count, common target count); zero produces null. Four image scores are macro-averaged with null cases reported separately. This does not use SIFT descriptors or a fitted transform.\n\nmetrics.csv has 48 image-condition-method rows; coordinates/ records retained position coordinates, H, common-visible indices and one-to-one correspondences. audit.json verifies all 8 identity controls equal1, homography round trips, visibility, spacing/uniqueness, maximum matching with minimum vertex-cover certificates and independent CSV macro recalculation. Timing is one descriptive CPU measurement including selection, not a repeated benchmark.\n\nFigures: overview_exo.png / overview_ego.png show original, log-positive Harris response, Harris centers and SIFT keypoints; controlled_changes_exo.png / controlled_changes_ego.png show all five changes of F50; repeatability_examples.png shows the numerically highest and lowest nonidentity cases without changing parameters; repeatability_summary.png shows all condition means. A 16px center margin does not guarantee that large SIFT support patches avoid all invalid pixels. Four selected frames do not establish generalization.\n\nNumeric evaluation uses the in-memory uint8 transformed images, before any JPEG writing. transformed_images/*.jpg are lossy display assets only and must not be reloaded to reproduce the saved metrics. Plot fonts try Monaco, Menlo, DejaVu Sans Mono and Consolas before Pillow default; font differences do not affect detection or matching.\n\nReproduce from the project root with Python3.12, OpenCV4.11, NumPy and Pillow; use an empty output directory:\n\n```sh\npython scripts/course_feature_detection.py \\\n  --frames-root output/week08/frames/scene04 \\\n  --output output/week08_harris_sift/results_reproduced\n```\n', encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records), "macro": macro, "audit": "passed", "total_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
