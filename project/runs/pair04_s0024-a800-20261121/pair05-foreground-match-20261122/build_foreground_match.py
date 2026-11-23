"""Build traceable Pair05 foreground correspondences for two real frames on CPU."""

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
RUN = HERE.parent
for name in ("ball-view-cpu-review", "pair01-gga-ab", "update-008/output-008/diagnosis"):
    sys.path.insert(0, str(RUN / name))
from aria_gga import project_aria
from audit_lens import coefficients
from build_ball_previews import load_depth
from check_inputs import sha256
from review_pair01_ab import check_reference, review_frames

TAKE = "pair05_s0024"
FRAMES = (12, 24)
HEIGHT, WIDTH, SIZE = 448, 784, 448
PATCH = 16
TEAMMATE, BALL = 1, 2
WEARER = -1
DISC_SOURCE = ((189, 244), (226, 257), (260, 239), (305, 250))
NAMES = {1: "teammate", 2: "ball", **{i: f"disc_{i - 2}" for i in range(3, 7)}}
COLORS = {
    1: (114, 240, 156), 2: (255, 205, 94), 3: (151, 138, 252),
    4: (85, 200, 255), 5: (216, 132, 244), 6: (90, 242, 221),
}
PRIOR_REGIONS = {
    1: (165, 150, 281, 270), 2: (212, 237, 247, 277),
    4: (280, 420, 336, 448), 6: (149, 387, 210, 441),
}


def require_hash(path, expected):
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"Frozen input changed: {path}")
    return digest


def read_mask(archive, frame):
    result = cv2.imdecode(np.frombuffer(archive.read(f"{frame:05}.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if result is None or result.shape != (HEIGHT, WIDTH) or result.dtype != np.uint8:
        raise ValueError(f"Invalid mask at F{frame}.")
    return result


def foreground_labels(rgb, instances, ball):
    ids = np.zeros((HEIGHT, WIDTH), np.int16)
    ids[instances == 1] = WEARER
    ids[instances == 2] = TEAMMATE
    if not ball.any() or np.any(ball & (instances != 0)):
        raise ValueError("Ball mask conflicts with the frozen instance labels.")
    ids[ball] = BALL
    for label, (x, y) in enumerate(DISC_SOURCE, start=3):
        s = np.s_[y - 5:y + 6, x - 7:x + 8]
        red, green, blue = rgb[s].astype(np.int16).transpose(2, 0, 1)
        cyan = (green - red > 40) & (blue - red > 35) & (green >= 115) & (ids[s] == 0)
        _, blobs = cv2.connectedComponents(cyan.astype(np.uint8))
        center_component = blobs[5, 7]
        if center_component == 0:
            raise ValueError(f"Disc {label - 2} does not have a cyan component at its reviewed EXO center.")
        ids[s][blobs == center_component] = label
    return ids


def project(world, w2c, k, distortion, candidates):
    view = world[candidates] @ w2c[:, :3].T + w2c[:, 3]
    front = np.isfinite(view).all(axis=1) & (view[:, 2] > 1e-5)
    source = candidates[front]
    view = view[front]
    # Renderer converts OpenCV camera axes (x,y,z) to Aria (y,-x,z), then screen axes flip.
    plane = np.column_stack((view[:, 1], -view[:, 0])) / view[:, 2, None]
    xy = k[:2, 2] - k[0, 0] * project_aria(plane, distortion)
    pixels = np.rint(xy).astype(np.int32)
    in_view = ((pixels >= 0) & (pixels < SIZE)).all(axis=1)
    x, y = pixels[in_view].T
    return source[in_view], y * SIZE + x, view[in_view, 2]


def closest_surface(source, destination, depth):
    order = np.lexsort((source, depth, destination))
    _, first = np.unique(destination[order], return_index=True)
    winners = order[first]
    front_id = np.full(SIZE * SIZE, -1, np.int32)
    front_z = np.full(SIZE * SIZE, np.inf, np.float64)
    front_id[destination[winners]] = source[winners]
    front_z[destination[winners]] = depth[winners]
    return front_id, front_z


def trusted_teammate(labels, surface, source_labels):
    """Remove uncertain silhouette samples and isolated projected speckles."""
    known = (source_labels == TEAMMATE).astype(np.uint8)
    interior = cv2.erode(known, np.ones((3, 3), np.uint8)).ravel().astype(bool)
    points = np.flatnonzero(labels == TEAMMATE)
    safe = points[interior[surface[points]]]
    valid = np.zeros(SIZE * SIZE, dtype=np.uint8)
    valid[safe] = 1
    # The expanded raster groups nearby observations. Its added pixels never enter the result.
    connected = cv2.dilate(valid.reshape(SIZE, SIZE), np.ones((5, 5), np.uint8))
    _, regions = cv2.connectedComponents(connected)
    groups = regions.ravel()[safe]
    if not len(groups):
        raise ValueError("No reliable teammate sample survives the silhouette check.")
    main = np.bincount(groups).argmax()
    result = labels.copy()
    result[points] = 0
    result[safe[groups == main]] = TEAMMATE
    return result, {
        "before_confidence_filter_square": len(points),
        "rejected_source_silhouette_pixels": len(points) - len(safe),
        "rejected_disconnected_target_pixels": int((groups != main).sum()),
    }


def allows(frame, candidate_frame, source, destination, candidate_id, query_id, surface):
    """Accept observed foreground only at its own time, identity and nearest surface."""
    return ((candidate_frame == frame) & (query_id > 0) & (candidate_id == query_id)
            & (surface[destination] == source))


def patch_edges(destinations, sources, labels, world, camera_center):
    query_y, query_x = np.divmod(destinations, SIZE)
    key_y, key_x = np.divmod(sources, WIDTH)
    rows = np.column_stack((labels, query_y // PATCH, query_x // PATCH, key_y // PATCH, key_x // PATCH))
    unique, inverse, counts = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    # Align-corners-false 1/8 bilinear followed by 2x2 pooling uses these 16 source pixels.
    native = np.isin(key_y % 8, (3, 4)) & np.isin(key_x % 8, (3, 4))
    native_counts = np.bincount(inverse, weights=native.astype(float), minlength=len(counts))
    rays = world[sources] - camera_center
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, rays)
    directions = sums / counts[:, None]
    if not np.isfinite(directions).all() or (np.linalg.norm(directions, axis=1) < 0.9).any():
        raise ValueError("An object-specific route has missing or conflicting projected directions.")
    return np.column_stack((unique, counts, native_counts)).astype(np.int32), directions.astype(np.float32)


def object_summary(mask, source_ids, object_id, edges):
    yy, xx = np.nonzero(mask.reshape(SIZE, SIZE) == object_id)
    row = {
        "source_pixels": int((source_ids == object_id).sum()),
        "target_visible_pixels": len(xx),
        "object_specific_patch_edges": int((edges[:, 0] == object_id).sum()),
        "edges_with_exact_native_gga_source_sample": int(((edges[:, 0] == object_id) & (edges[:, 6] > 0)).sum()),
    }
    if len(xx):
        row["center_xy"] = np.round([np.median(xx), np.median(yy)], 1).tolist()
        if object_id in PRIOR_REGIONS:
            x0, y0, x1, y1 = PRIOR_REGIONS[object_id]
            row["fraction_in_independently_reviewed_prior_region"] = float(
                ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).mean()
            )
    return row


def replay_native_ball_edges(edges, labels, source, destination):
    """Inspect the actual fixed source samples for each observed ball edge."""
    sy, sx = np.divmod(source, WIDTH)
    dy, dx = np.divmod(destination, SIZE)
    sampled = np.isin(sy % 8, (3, 4)) & np.isin(sx % 8, (3, 4))
    is_ball = labels[source] == BALL
    reports = []
    for _, qy, qx, ky, kx, support, exact_native in edges[edges[:, 0] == BALL]:
        key = (sy // PATCH == ky) & (sx // PATCH == kx) & sampled
        query = (dy // PATCH == qy) & (dx // PATCH == qx)
        reports.append({
            "query_yx": [int(qy), int(qx)],
            "key_yx": [int(ky), int(kx)],
            "observed_ball_support": int(support),
            "native_samples_labelled_ball_in_source_block": int((key & is_ball).sum()),
            "native_ball_samples_projecting_to_query_block_before_depth_gate": int((key & query & is_ball).sum()),
            "other_native_samples_projecting_to_query_block_before_depth_gate": int((key & query & ~is_ball).sum()),
            "native_ball_samples_still_observed_after_gate": int(exact_native),
        })
    return reports


def audit_ball_sample_budget(ball_destinations, ball_sources):
    """Retain geometric ball edges with at most 16 observed pixels per source block."""
    sy, sx = np.divmod(ball_sources, WIDTH)
    dy, dx = np.divmod(ball_destinations, SIZE)
    keys = (sy // PATCH) * (WIDTH // PATCH) + sx // PATCH
    queries = (dy // PATCH) * (SIZE // PATCH) + dx // PATCH
    picked = []
    for key in np.unique(keys):
        group = np.flatnonzero(keys == key)
        _, first = np.unique(queries[group], return_index=True)
        required = group[first]
        if len(required) > 16:
            raise ValueError("A ball source block needs more than 16 distinct query samples.")
        remaining = np.setdiff1d(group, required)
        spare = min(16 - len(required), len(remaining))
        picked.extend(required.tolist())
        if spare:
            picked.extend(remaining[np.linspace(0, len(remaining) - 1, spare, dtype=int)].tolist())
    before = np.unique(np.column_stack((keys, queries)), axis=0)
    after = np.unique(np.column_stack((keys[picked], queries[picked])), axis=0)
    if not np.array_equal(before, after):
        raise ValueError("The bounded sample replay lost a visible ball edge.")
    return {
        "observed_input_pixels": len(ball_sources),
        "retained_under_same_budget": len(picked),
        "retained_object_edges": len(after),
        "maximum_samples_per_source_block": int(np.unique(keys[picked], return_counts=True)[1].max()),
    }


def analyze(frame, data, other, metadata, distortion):
    rgb, ids, excluded, world = data
    ids = ids.ravel()
    exo_visible = np.flatnonzero(~excluded.ravel())
    pose = np.asarray(metadata["ego_extrinsics"][frame])
    k = np.asarray(metadata["ego_intrinsics"])
    source, destination, z = project(world, pose, k, distortion, exo_visible)
    surface, front_z = closest_surface(source, destination, z)
    front_labels = np.zeros_like(surface, dtype=np.int16)
    mapped = surface >= 0
    front_labels[mapped] = ids[surface[mapped]]
    front_labels, teammate_confidence = trusted_teammate(front_labels, surface, data[1])
    square_labels = front_labels.copy()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    nominal_circle = ((xx - k[0, 2])**2 + (yy - k[1, 2])**2 <= (SIZE / 2)**2).ravel()
    front_labels[~nominal_circle] = 0
    admitted = allows(frame, frame, source, destination, ids[source], front_labels[destination], surface)
    d, s = destination[admitted], source[admitted]
    camera_center = np.linalg.solve(pose[:, :3], -pose[:, 3])
    edges, masked_directions = patch_edges(d, s, ids[s], world, camera_center)
    if len(d) != np.count_nonzero(front_labels > 0) or len(d) != len(np.unique(d)):
        raise ValueError(f"Foreground destination mapping is duplicated or missing at F{frame}.")
    protected = np.full_like(surface, -1)
    protected[d] = s
    objects = {name: object_summary(front_labels, ids, label, edges) for label, name in NAMES.items()}
    for label, name in NAMES.items():
        square = object_summary(square_labels, ids, label, edges)
        objects[name]["square_visible_pixels"] = square["target_visible_pixels"]
        if label in PRIOR_REGIONS:
            fraction = square.get("fraction_in_independently_reviewed_prior_region", 0.0)
            objects[name]["square_fraction_in_reviewed_region"] = fraction
            if fraction < 0.9:
                raise ValueError(f"Observed {name} does not align with prior at F{frame}: {square}")
        if label in (TEAMMATE, BALL, 6) and not objects[name]["target_visible_pixels"]:
            raise ValueError(f"Expected {name} in the usable circle at F{frame}.")

    at_target = front_labels[destination] > 0
    depth_behind = z > front_z[destination] + 1e-4
    wrong_identity = at_target & (ids[source] > 0) & (ids[source] != front_labels[destination])
    wrong_to_background = ~at_target & (ids[source] > 0)
    background_to_front = at_target & (ids[source] <= 0)
    foreground_behind = (ids[source] > 0) & depth_behind

    _, other_ids, other_excluded, other_world = other
    other_ids = other_ids.ravel()
    valid_other = np.flatnonzero((other_ids > 0) & ~other_excluded.ravel())
    other_source, other_dest, _ = project(other_world, pose, k, distortion, valid_other)
    competing_time = (other_ids[other_source] == front_labels[other_dest])
    other_frame = next(item for item in FRAMES if item != frame)
    untime = allows(frame, other_frame, other_source, other_dest, other_ids[other_source],
                    front_labels[other_dest], surface)
    other_relaxed = allows(frame, frame, other_source, other_dest, other_ids[other_source],
                           front_labels[other_dest], surface)
    wrong_target_ball = front_labels[d] != BALL
    ball_source = s[ids[s] == BALL]
    mismatch_count = (
        len(ball_source) if wrong_target_ball.any() else 0
    )
    if mismatch_count:
        proposed_target = np.resize(d[wrong_target_ball], len(ball_source))
        false_ball_matches = allows(frame, frame, ball_source, proposed_target, ids[ball_source],
                                    front_labels[proposed_target], surface)
    else:
        false_ball_matches = np.zeros(0, dtype=bool)
    negatives = {
        "observed_foreground_behind_nearer_surface": foreground_behind,
        "observed_wrong_foreground_identity_at_protected_pixel": wrong_identity,
        "observed_foreground_landing_on_unprotected_pixel": wrong_to_background,
        "observed_background_or_wearer_at_protected_pixel": background_to_front,
    }
    rejected = {}
    for name, candidate in negatives.items():
        if admitted[candidate].any():
            raise ValueError(f"Failure in negative check {name}.")
        rejected[name] = {"tested": int(candidate.sum()), "admitted": 0}
    if untime.any() or false_ball_matches.any() or not other_relaxed.any() or not mismatch_count:
        raise ValueError("Temporal or semantic gate failed its real-candidate negative check.")
    rejected["other_frame_same_identity_landing_at_protected_pixel"] = {
        "tested": int(competing_time.sum()), "would_pass_without_time_gate": int(other_relaxed.sum()), "admitted": 0,
    }
    rejected["real_ball_routed_to_other_object"] = {"tested": mismatch_count, "admitted": 0}
    if (ids[s] == 4).any() and (ids[s] == 6).any():
        disc_source = s[ids[s] == 4]
        disc_target = np.resize(d[ids[s] == 6], len(disc_source))
        swapped_discs = allows(frame, frame, disc_source, disc_target, ids[disc_source],
                               front_labels[disc_target], surface)
        if swapped_discs.any():
            raise ValueError("Two distinct marker disks were accepted as the same object.")
        rejected["real_disc_2_routed_to_disc_4"] = {"tested": len(disc_source), "admitted": 0}
    # Count per-patch distinct foreground identities, not just the number of edges.
    pairs = np.unique(edges[:, :3], axis=0)
    _, class_counts = np.unique(pairs[:, 1:], axis=0, return_counts=True)
    summary = {
        "objects": objects,
        "protected_target_pixels": len(d),
        "object_specific_patch_edges": len(edges),
        "object_specific_directions_verified": len(masked_directions),
        "ball_edges_fixed_sample_replay": replay_native_ball_edges(edges, ids, source, destination),
        "ball_only_sample_budget_replay": audit_ball_sample_budget(d[ids[s] == BALL], ball_source),
        "target_patches_with_multiple_foreground_ids": int((class_counts > 1).sum()),
        "teammate_confidence_filter": teammate_confidence,
        "rejected": rejected,
    }
    return summary, front_labels.reshape(SIZE, SIZE), protected.reshape(SIZE, SIZE), edges, masked_directions


def make_panel(img, labels):
    img = img.copy()
    for label, color in COLORS.items():
        where = labels == label
        if where.any():
            border = cv2.dilate(where.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
            img[border & ~where] = color
    return Image.fromarray(img)


def render_review(root, priors, decoded, results):
    x_offsets = (12, 808, 1268, 1728)
    sheet = Image.new("RGB", (2188, 980), "#151f2a")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for x, label in zip(x_offsets, ("EXO and source labels", "Frozen geometry", "Visible observed pixels", "Matched positions"), strict=True):
        draw.text((x, 9), label, font=font, fill="white")
    for row, frame in enumerate(FRAMES):
        labels, source_index = results[frame]
        rgb, source_labels, *_ = decoded[frame]
        y = 54 + row * 464
        canvas = np.full((SIZE, SIZE, 3), (17, 25, 35), np.uint8)
        keep = source_index >= 0
        canvas[keep] = rgb.reshape(-1, 3)[source_index[keep]]
        target_rgb = np.asarray(priors[frame])
        columns = (make_panel(rgb, source_labels), priors[frame],
                   make_panel(canvas, labels), make_panel(target_rgb, labels))
        for x, col in zip(x_offsets, columns, strict=True):
            draw.text((x, y - 23), f"F{frame:02}", font=font, fill="white")
            sheet.paste(col, (x, y))
    sheet.save(root / "foreground-review.jpg", quality=94)


def build(output):
    inputs = json.loads((RUN / "pair05-aria-008/cpu-readiness.json").read_text())["inputs"]
    geometry = RUN / f"multiclip-evidence/{TAKE}/geometry/vipe_results/{TAKE}"
    update = RUN / "update-008"
    prior = update / f"output-008/priors/{TAKE}-final.mp4"
    exo = RUN / f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4"
    meta = RUN / f"ball-view-cpu-review/{TAKE}-ball-view-meta.json"
    render_log = json.loads((update / "output-008/logs/pair05-render.log").read_text().splitlines()[0])
    ball_zip = update / "ball-recovery/pair05-ball.zip"
    exclude_zip = RUN / "ball-view-cpu-review/pair05-composition-review/pair05-wearer-upper-body.zip"
    mask_zip = geometry / "mask/exo.zip"
    hashes = {
        "exo": require_hash(exo, inputs["exo_sha256"]),
        "prior": require_hash(prior, inputs["prior_sha256"]),
        "meta": require_hash(meta, "c2422ed6f6ecb0c150b8be612bc7c927cc39f5ec70cdd5a0593d8811d8d5846c"),
        "instances": require_hash(mask_zip, "7747332adc5416a8fc64ad3f7254164bafaed49b8ba8a7513e515a3e553d59d3"),
        "balls": require_hash(ball_zip, render_log["ball_sha256"]),
        "wearer_exclusion": require_hash(exclude_zip, render_log["exclusion_sha256"]),
    }
    check_reference(exo, WIDTH, 25)
    check_reference(prior, SIZE, 30)
    exo_images = review_frames(exo, WIDTH, FRAMES)
    priors = review_frames(prior, SIZE, FRAMES)
    metadata, = json.loads(meta.read_text())["test_datasets"]
    w2c = np.asarray(metadata["camera_extrinsics"])
    ys, xs = np.mgrid[:HEIGHT, :WIDTH]
    ray = np.stack((xs, ys, np.ones_like(xs)), axis=-1) @ np.linalg.inv(metadata["camera_intrinsics"]).T
    rows = json.loads((update / "output-008/logs/validated-inputs.json").read_text())
    receipt, = [row for row in rows if row["name"] == TAKE]
    renderer = RUN.parents[1] / "records/official_renderer_geometry.py"
    hashes["renderer"] = require_hash(renderer, render_log["renderer_sha256"])
    distortion = np.concatenate([coefficients(renderer, f"{field}_distortion_coeffs")
                                 for field in ("radial", "tangential", "thinPrism")])
    decoded = {}
    with (ZipFile(mask_zip) as labels, ZipFile(ball_zip) as balls, ZipFile(exclude_zip) as exclusions,
          ZipFile(geometry / "depth/exo.zip") as depths):
        for frame in FRAMES:
            rgb = np.asarray(exo_images[frame])
            instances = read_mask(labels, frame)
            excluded = read_mask(exclusions, frame) == 255
            if not excluded.any() or np.any(excluded & (instances != 1)):
                raise ValueError(f"Invalid wearer exclusion mask at F{frame}.")
            ids = foreground_labels(rgb, instances, read_mask(balls, frame) == 255)
            depth = load_depth(depths, frame)
            saved = BytesIO()
            np.save(saved, depth, allow_pickle=False)
            if hashlib.sha256(saved.getvalue()).hexdigest() != receipt["depth_sha256"][frame]:
                raise ValueError(f"Frozen depth hash changed at F{frame}.")
            scene = (ray * depth[..., None]).reshape(-1, 3)
            world = np.linalg.solve(w2c[:, :3], (scene - w2c[:, 3]).T).T
            decoded[frame] = (rgb, ids, excluded, world)

    measurements, arrays, review_arrays = {}, {}, {}
    for frame, alternate in (FRAMES, FRAMES[::-1]):
        report, labels, source_index, edges, directions = analyze(frame, decoded[frame], decoded[alternate],
                                                                 metadata, distortion)
        prefix = f"F{frame:02}"
        measurements[prefix] = report
        review_arrays[frame] = (labels, source_index)
        arrays[f"{prefix}_target_label"] = labels
        arrays[f"{prefix}_source_index"] = source_index
        arrays[f"{prefix}_patch_edges"] = edges
        arrays[f"{prefix}_masked_directions"] = directions
    output.mkdir(parents=True)
    np.savez_compressed(output / "foreground-routing.npz", **arrays)
    render_review(output, priors, decoded, review_arrays)
    result = {
        "status": "passed_pair05_cpu_foreground_matching",
        "scope": "Two real EXO frames, observed depth and the frozen Aria forward projection. No denoising.",
        "input_hashes": hashes,
        "labels": NAMES,
        "absolute_source_index": "y*784+x, or -1 for pixels outside the protected foreground",
        "patch_edge_columns": ["object_id", "query_y", "query_x", "key_y", "key_x", "visible_support_pixels",
                               "exact_native_gga_source_sample_pixels"],
        "masked_directions": "Mean unit world directions for each edge, using only that object's surviving pixels.",
        "measurements": measurements,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(build(args.output), indent=2))
