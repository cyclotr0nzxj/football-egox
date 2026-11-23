#!/usr/bin/env python3
"""Audit same-frame Pair05 foreground support at each GGA slot on CPU."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
PROJECT = RUN.parents[1]
MATCH = RUN / "pair05-foreground-match-20261122"
ROUTES = RUN / "pair05-gga-foreground-20261122"
sys.path[:0] = [str(MATCH), str(ROUTES)]
import build_foreground_match as match
import build_routes as routes


FRAMES = tuple(range(0, 49, 4))
TAKE = match.TAKE
OBJECTS = (match.TEAMMATE, match.BALL, *range(3, 7))
OBJECT_NAMES = match.NAMES
FROZEN_PLAN_SHA256 = "3cd25a102a6ecf4006add800b293dcf40b82b296392ff652aeea308fcd0f501e"


def read_json(path):
    return json.loads(path.read_text())


def decode_inputs():
    geometry = RUN / f"multiclip-evidence/{TAKE}/geometry/vipe_results/{TAKE}"
    update = RUN / "update-008"
    paths = {
        "exo": RUN / f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4",
        "prior": update / f"output-008/priors/{TAKE}-final.mp4",
        "meta": RUN / f"ball-view-cpu-review/{TAKE}-ball-view-meta.json",
        "instances": geometry / "mask/exo.zip",
        "balls": update / "ball-recovery/pair05-ball.zip",
        "wearer_exclusion": RUN / "ball-view-cpu-review/pair05-composition-review/pair05-wearer-upper-body.zip",
        "renderer": PROJECT / "records/official_renderer_geometry.py",
        "foreground_verification": MATCH / "results/verification.json",
        "foreground_mapping": MATCH / "results/foreground-routing.npz",
        "deployed_route_plan": ROUTES / "plan/pair05-gga-routes.npz",
        "infer": PROJECT / "vendor/EgoX/infer.py",
    }
    reviewed_hash = match.require_hash(paths["foreground_verification"], routes.VERIFICATION_HASH)
    reviewed = read_json(paths["foreground_verification"])
    for frame in match.FRAMES:
        if any(row["admitted"] for row in reviewed["measurements"][f"F{frame:02}"]["rejected"].values()):
            raise ValueError("Historical foreground mapping did not pass its negative checks.")
    expected = {
        **reviewed["input_hashes"], "foreground_verification": reviewed_hash,
        "foreground_mapping": routes.ROUTING_HASH, "deployed_route_plan": FROZEN_PLAN_SHA256,
        "infer": routes.INFER_HASH,
    }
    hashes = {key: match.require_hash(path, expected[key]) for key, path in paths.items()}
    match.check_reference(paths["exo"], match.WIDTH, 25)
    match.check_reference(paths["prior"], match.SIZE, 30)
    images = match.review_frames(paths["exo"], match.WIDTH, FRAMES)
    priors = match.review_frames(paths["prior"], match.SIZE, FRAMES)
    metadata, = read_json(paths["meta"])["test_datasets"]
    receipt, = [row for row in read_json(update / "output-008/logs/validated-inputs.json") if row["name"] == TAKE]
    if (receipt["exo_sha256"], receipt["prior_sha256"]) != (hashes["exo"], hashes["prior"]):
        raise ValueError("The renderer receipt is not paired to the verified EXO and prior.")
    exo_pose = np.asarray(metadata["camera_extrinsics"])
    ys, xs = np.mgrid[:match.HEIGHT, :match.WIDTH]
    rays = np.stack((xs, ys, np.ones_like(xs)), axis=-1) @ np.linalg.inv(metadata["camera_intrinsics"]).T
    distortion = np.concatenate([
        match.coefficients(paths["renderer"], f"{field}_distortion_coeffs")
        for field in ("radial", "tangential", "thinPrism")
    ])
    observations = {}
    depth_hashes = {}
    with (ZipFile(paths["instances"]) as instances, ZipFile(paths["balls"]) as balls,
          ZipFile(paths["wearer_exclusion"]) as exclusions, ZipFile(geometry / "depth/exo.zip") as depths):
        for frame in FRAMES:
            rgb = np.asarray(images[frame])
            people = match.read_mask(instances, frame)
            excluded = match.read_mask(exclusions, frame) == 255
            ball = match.read_mask(balls, frame) == 255
            if not excluded.any() or (excluded & (people != 1)).any():
                raise ValueError(f"Invalid wearer exclusion at F{frame:02}.")
            depth = match.load_depth(depths, frame)
            saved = BytesIO()
            np.save(saved, depth, allow_pickle=False)
            depth_hashes[f"F{frame:02}"] = sha256(saved.getvalue()).hexdigest()
            if depth_hashes[f"F{frame:02}"] != receipt["depth_sha256"][frame]:
                raise ValueError(f"Frozen depth changed at F{frame:02}.")
            scene = (rays * depth[..., None]).reshape(-1, 3)
            world = np.linalg.solve(exo_pose[:, :3], (scene - exo_pose[:, 3]).T).T
            if frame in match.FRAMES:
                labels = match.foreground_labels(rgb, people, ball)
            else:
                labels = np.zeros(people.shape, np.int16)
                labels[people == 1] = match.WEARER
                labels[people == 2] = match.TEAMMATE
                labels[ball & (people == 0)] = match.BALL
            observations[frame] = {
                "rgb": rgb, "labels": labels, "excluded": excluded, "world": world, "depth": depth,
                "source_ball_mask_pixels": int(ball.sum()),
                "ball_pixels_conflicting_with_people": int((ball & (people != 0)).sum()),
            }
    return paths, hashes, depth_hashes, metadata, distortion, priors, observations


def foreground_at_frame(frame, obs, adjacent_frame, adjacent, pose, intrinsics, distortion):
    candidates = np.flatnonzero(~obs["excluded"].ravel())
    source_ids = obs["labels"].ravel()
    source, destination, z = match.project(obs["world"], pose, intrinsics, distortion, candidates)
    surface, front_z = match.closest_surface(source, destination, z)
    present = surface >= 0
    labels = np.zeros(surface.shape, np.int16)
    labels[present] = source_ids[surface[present]]
    confidence = {"before_confidence_filter_square": 0, "rejected_source_silhouette_pixels": 0,
                  "rejected_disconnected_target_pixels": 0}
    if (labels == match.TEAMMATE).any():
        try:
            labels, confidence = match.trusted_teammate(labels, surface, obs["labels"])
        except ValueError as error:
            if str(error) != "No reliable teammate sample survives the silhouette check.":
                raise
            confidence["rejected_source_silhouette_pixels"] = int((labels == match.TEAMMATE).sum())
            confidence["before_confidence_filter_square"] = confidence["rejected_source_silhouette_pixels"]
            labels[labels == match.TEAMMATE] = 0
    yy, xx = np.mgrid[:match.SIZE, :match.SIZE]
    circle = ((xx - intrinsics[0, 2])**2 + (yy - intrinsics[1, 2])**2 <= (match.SIZE / 2)**2).ravel()
    labels[~circle] = 0
    admitted = match.allows(frame, frame, source, destination, source_ids[source], labels[destination], surface)
    protected = np.full(surface.shape, -1, np.int32)
    protected[destination[admitted]] = source[admitted]
    if int(admitted.sum()) != np.count_nonzero(labels > 0):
        raise ValueError(f"Incomplete nearest-surface foreground support at F{frame:02}.")

    wrong_identity = (source_ids[source] > 0) & (labels[destination] > 0) & (labels[destination] != source_ids[source])
    hidden = (source_ids[source] > 0) & (z > front_z[destination] + 1e-4)
    if admitted[wrong_identity | hidden].any():
        raise ValueError(f"Wrong-identity or occluded pixel passed at F{frame:02}.")
    alternate_ids = adjacent["labels"].ravel()
    alternate = np.flatnonzero((alternate_ids > 0) & ~adjacent["excluded"].ravel())
    alt_src, alt_dst, _ = match.project(adjacent["world"], pose, intrinsics, distortion, alternate)
    counterfactual = match.allows(frame, frame, alt_src, alt_dst, alternate_ids[alt_src], labels[alt_dst], surface)
    rejected = match.allows(frame, adjacent_frame, alt_src, alt_dst,
                            alternate_ids[alt_src], labels[alt_dst], surface)
    if rejected.any():
        raise ValueError(f"Off-time source was admitted at F{frame:02}.")
    return labels, protected, {
        "projected_geometry_coverage_in_circle": float(present[circle].mean()),
        "teammate_confidence_filter": confidence,
        "negative_probes": {
            "wrong_identity_rejected": int(wrong_identity.sum()),
            "behind_nearest_surface_rejected": int(hidden.sum()),
            "adjacent_frame": adjacent_frame,
            "off_time_pixels_that_would_pass_without_frame_check": int(counterfactual.sum()),
            "off_time_pixels_admitted": int(rejected.sum()),
        },
    }


def pixel_blocks(index, width, block_cols):
    y, x = np.divmod(index, width)
    return (y // routes.PATCH) * block_cols + x // routes.PATCH


def budgets(frame, labels, protected, depth, metadata, infer):
    destination = np.flatnonzero(labels > 0)
    identity = labels[destination]
    source = protected[destination]
    key = pixel_blocks(source, match.WIDTH, routes.SOURCE_COLS)
    query = pixel_blocks(destination, match.SIZE, routes.EGO_COLS)
    chosen, mixed = routes.choose_samples(key, identity, query)
    mixed = np.array(mixed, dtype=np.int64)
    drop = np.isin(key, mixed)
    if drop[chosen].any():
        raise ValueError("A mixed source block survived the shared budget.")
    allowed_labels, allowed_protected = labels.copy(), protected.copy()
    # Preserve the published two-frame path byte-for-byte. In newly audited slots,
    # loss from a mixed source block is reported before compiling surviving branches.
    if frame not in match.FRAMES:
        allowed_labels[destination[drop]] = 0
        allowed_protected[destination[drop]] = -1
    prefix = f"F{frame:02}"
    if (allowed_labels > 0).any():
        evidence = {f"{prefix}_target_label": allowed_labels, f"{prefix}_source_index": allowed_protected}
        edge, cosine, samples, _ = routes.foreground_rows(frame, evidence, depth, metadata, infer)
    else:
        edge = np.empty((0, 4), np.int32)
        cosine = np.empty(0, np.float32)
        samples = np.empty((0, 4), np.int32)
    expected_pixels = np.column_stack((np.full(len(chosen), frame), destination[chosen],
                                       source[chosen], identity[chosen])).astype(np.int32)
    if not np.array_equal(samples, expected_pixels):
        raise ValueError(f"Sample budget is not consistent with the frozen compiler at F{frame:02}.")
    by_object = {}
    for obj in OBJECTS:
        raw = identity == obj
        if not raw.any():
            continue
        sample_hit = samples[:, 3] == obj
        edge_hit = edge[:, 1] == obj
        by_object[OBJECT_NAMES[obj]] = {
            "visible_target_pixels": int(raw.sum()),
            "visible_target_query_blocks": int(len(np.unique(query[raw]))),
            "pixels_rejected_by_mixed_source_blocks": int((raw & drop).sum()),
            "samples": int(sample_hit.sum()),
            "distinct_source_blocks": int(len(np.unique(edge[edge_hit, 2]))),
            "distinct_target_query_blocks": int(len(np.unique(edge[edge_hit, 0]))),
            "object_edges": int(edge_hit.sum()),
        }
        if sample_hit.any():
            y, x = np.divmod(samples[sample_hit, 1], match.SIZE)
            xy = np.column_stack((x, y))
            radii = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
            by_object[OBJECT_NAMES[obj]]["sample_radius_p90_from_target_median_px"] = float(np.quantile(radii, 0.9))
    source_blocks = pixel_blocks(samples[:, 2], match.WIDTH, routes.SOURCE_COLS)
    counts = np.unique(source_blocks, return_counts=True)[1]
    return edge, cosine, samples, {
        "mixed_source_blocks_excluded": len(mixed),
        "highest_source_block_sample_count": int(counts.max(initial=0)),
        "target_query_blocks": int(len(np.unique(edge[:, 0]))),
        "source_blocks": int(len(counts)),
        "object_edges": int(len(edge)),
        "objects": by_object,
    }


def validate_compiled(edges, cosine, samples):
    frame, dest, source, identity = samples.astype(np.int64).T
    time = frame // 4
    q = time * routes.ROWS * routes.EGO_COLS + pixel_blocks(dest, match.SIZE, routes.EGO_COLS)
    sy, sx = np.divmod(source, match.WIDTH)
    k = (time * routes.ROWS + sy // routes.PATCH) * routes.COLS + sx // routes.PATCH
    grouped, counts = np.unique(np.column_stack((q, identity, k)), axis=0, return_counts=True)
    if not np.array_equal(np.column_stack((grouped, counts)), edges):
        raise ValueError("Compiled edges do not match their exact surviving provenance.")
    if (len(np.unique(samples[:, (0, 1)], axis=0)) != len(samples)
            or len(np.unique(samples[:, (0, 2)], axis=0)) != len(samples)
            or len(np.unique(np.column_stack((k, identity)), axis=0)) != len(np.unique(k))
            or np.unique(k, return_counts=True)[1].max() > routes.BUDGET
            or not np.isfinite(cosine).all() or (np.abs(cosine) > 1).any()):
        raise ValueError("Global temporal plan violates identity or per-source budget rules.")


def coverage_report(report):
    teammate = report.setdefault("teammate", {})
    ball = report.setdefault("ball", {})
    ball_blocks = ball.get("distinct_target_query_blocks", 0)
    if not teammate.get("distinct_target_query_blocks") or not ball_blocks:
        return "missing_teammate_or_ball_route"
    if ball["sample_radius_p90_from_target_median_px"] > routes.PATCH:
        return "fragmented_ball_projection"
    if ball_blocks == 1 or ball["samples"] <= routes.BUDGET // 2:
        return "limited_ball_support"
    return "both_objects_supported"


def review_image(output, frames, by_frame, reports):
    tile = 168
    row_height = tile + 49
    col_width = 2 * tile + 12
    sheet = Image.new("RGB", (4 * col_width + 8, 41 + 4 * row_height), "#192530")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    draw.text((10, 12), "Frozen geometry / Routed EXO samples    Green: teammate  |  Yellow: ball  |  Blue: reviewed discs",
              font=font, fill="white")
    for index, frame in enumerate(FRAMES):
        x = 8 + (index % 4) * col_width
        y = 41 + (index // 4) * row_height
        image = frames[frame]
        overlay = (np.asarray(image).astype(np.float32) * 0.56).astype(np.uint8)
        for _, destination, _, obj in by_frame[frame]:
            yy, xx = divmod(int(destination), match.SIZE)
            overlay[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3] = match.COLORS[obj]
        sheet.paste(image.resize((tile, tile)), (x, y + 21))
        sheet.paste(Image.fromarray(overlay).resize((tile, tile)), (x + tile + 2, y + 21))
        report = reports[f"F{frame:02}"]
        before = "old two-slot target" if frame in match.FRAMES else "previously unrouted slot"
        draw.text((x, y), f"F{frame:02}  {before}", font=font, fill="white")
        ball = report["objects"]["ball"].get("samples", 0)
        draw.text((x, y + tile + 23), f"{report['target_query_blocks']} target blocks   {ball} ball samples",
                  font=font, fill="white")
    sheet.save(output / "timeline-support.jpg", quality=92)


def build(output):
    paths, hashes, depth_hashes, metadata, distortion, priors, observations = decode_inputs()
    intrinsics = np.asarray(metadata["ego_intrinsics"])
    infer = paths["infer"].read_text()
    by_frame, reports = {}, {}
    edges, cosines, samples = [], [], []
    legacy_edges, legacy_cosines, legacy_samples = [], [], []
    yy, xx = np.mgrid[:match.SIZE, :match.SIZE]
    upper = ((xx - 224)**2 + (yy - 224)**2 <= 224**2) & (yy < 224)
    with np.load(paths["foreground_mapping"], allow_pickle=False) as mapping:
        for index, frame in enumerate(FRAMES):
            obs = observations[frame]
            adjacent = FRAMES[index - 1] if index else FRAMES[1]
            pose = np.asarray(metadata["ego_extrinsics"][frame])
            labels, protected, report = foreground_at_frame(
                frame, obs, adjacent, observations[adjacent], pose, intrinsics, distortion,
            )
            prefix = f"F{frame:02}"
            if frame in match.FRAMES:
                if not (np.array_equal(labels, mapping[f"{prefix}_target_label"].ravel())
                        and np.array_equal(protected, mapping[f"{prefix}_source_index"].ravel())):
                    raise ValueError(f"Reprojected F{frame:02} differs from the historic pixel mapping.")
            edge, cosine, sample, budget = budgets(frame, labels, protected, obs["depth"], metadata, infer)
            edges.append(edge)
            cosines.append(cosine)
            samples.append(sample)
            if frame in match.FRAMES:
                legacy_edges.append(edge)
                legacy_cosines.append(cosine)
                legacy_samples.append(sample)
            by_frame[frame] = sample
            reports[prefix] = {
                "slot": frame // 4, "previously_foreground_routed": frame in match.FRAMES,
                "source_ball_mask_pixels": obs["source_ball_mask_pixels"],
                "source_ball_pixels_conflicting_with_people": obs["ball_pixels_conflicting_with_people"],
                "discs_reviewed_here": frame in match.FRAMES,
                "prior_near_black_fraction_upper_circle": float((np.asarray(priors[frame]).max(axis=2)[upper] <= 8).mean()),
                **report, **budget,
                "support_class": coverage_report(budget["objects"]),
            }

    edges, cosines, samples = [np.concatenate(values) for values in (edges, cosines, samples)]
    validate_compiled(edges, cosines, samples)
    with np.load(paths["deployed_route_plan"], allow_pickle=False) as plan:
        for name, groups in zip(("edges", "cosine", "samples"), (legacy_edges, legacy_cosines, legacy_samples), strict=True):
            if not np.array_equal(np.concatenate(groups), plan[name]):
                raise ValueError(f"The historic F12 and F24 {name} changed when recomputed.")
        old_unique = int(len(np.unique(plan["edges"][:, 0])))
    temporal_negatives = sum(row["negative_probes"]["off_time_pixels_that_would_pass_without_frame_check"]
                             for row in reports.values())
    if not temporal_negatives:
        raise ValueError("No positive candidate for the adjacent-frame negative test was observed.")

    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "timeline-support-diagnostic.npz"
    np.savez_compressed(artifact, edges=edges, cosine=cosines, samples=samples, frames=np.asarray(FRAMES, dtype=np.int32),
                        scope=np.asarray("cpu_timeline_audit_not_deployable"), metadata_sha256=np.asarray(hashes["meta"]),
                        exo_sha256=np.asarray(hashes["exo"]), prior_sha256=np.asarray(hashes["prior"]))
    report = {
        "scope": "Observed EXO same-frame geometry and route construction on CPU. No VAE or denoising.",
        "frozen_source_hashes": hashes, "per_slot_depth_hashes": depth_hashes,
        "old_two_slot_exclusive": {"frames": list(match.FRAMES), "target_query_blocks": old_unique,
                                    "recomputed_mapping_edges_cosine_samples_exact": True},
        "sample_columns": ["source_and_target_frame", "target_flat_pixel", "source_flat_pixel", "object_id"],
        "edge_columns": ["flat_ego_query", "object_id", "flat_exo_key", "sample_count"],
        "diagnostic_plan_sha256": match.sha256(artifact),
        "candidate_target_query_blocks": int(len(np.unique(edges[:, 0]))),
        "total_target_query_blocks": routes.TIMES * routes.ROWS * routes.EGO_COLS,
        "candidate_object_edges": int(len(edges)), "candidate_selected_samples": int(len(samples)),
        "max_samples_per_same_frame_source_block": max(row["highest_source_block_sample_count"] for row in reports.values()),
        "off_time_negative_cases": temporal_negatives,
        "support_class_rule": ("limited means one ball target block or at most eight retained ball samples; fragmented "
                               "means p90 radius exceeds one 16-pixel patch; heuristic evidence labels"),
        "frames": reports,
    }
    (output / "timeline-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    review_image(output, priors, by_frame, reports)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))
