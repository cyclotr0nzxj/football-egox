"""Compile the verified Pair05 matches into a shared-budget sparse GGA plan."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT = RUN.parents[1]
for folder in ("local-reassessment-20261122", "pair01-gga-ab", "ball-view-cpu-review"):
    sys.path.insert(0, str(RUN / folder))
from audit_gga_inputs import corrected_rays
from build_ball_previews import load_depth
from check_inputs import sha256
from review_pair01_ab import review_frames

FRAMES = (12, 24)
TAKE = "pair05_s0024"
SOURCE_WIDTH = 784
SIZE = 448
PATCH = 16
SOURCE_COLS, EGO_COLS, ROWS = 49, 28, 28
COLS, TIMES = SOURCE_COLS + EGO_COLS, 13
BUDGET = 16
META_HASH = "c2422ed6f6ecb0c150b8be612bc7c927cc39f5ec70cdd5a0593d8811d8d5846c"
ROUTING_HASH = "e665f13acd916a49082633ddda9adda8299f2b3e24e636a203eee904491788e2"
VERIFICATION_HASH = "11de61639b178c78106ed02b993433246d1891b3cd2d68492884b70d135c7e1f"
INFER_HASH = "99199fac22858821d356f0ec3b8577495290dd579a8d80368de1fca68d559bed"


def choose_samples(key, identity, query):
    """Reject mixed source blocks and reserve one sample per edge before filling the shared budget."""
    selected, mixed = [], []
    for block in np.unique(key):
        points = np.flatnonzero(key == block)
        if np.unique(identity[points]).size != 1:
            mixed.append(int(block))
            continue
        _, first = np.unique(query[points], return_index=True)
        reserved = points[first]
        if len(reserved) > BUDGET:
            raise ValueError(f"Source block {block} has more distinct edges than its budget.")
        remaining = np.setdiff1d(points, reserved)
        spare = min(len(remaining), BUDGET - len(reserved))
        selected.extend(reserved.tolist())
        if spare:
            selected.extend(remaining[np.linspace(0, len(remaining) - 1, spare, dtype=int)].tolist())
    return np.array(selected, dtype=np.int64), mixed


def foreground_rows(frame, data, depth, metadata, infer):
    labels = data[f"F{frame:02}_target_label"].ravel()
    provenance = data[f"F{frame:02}_source_index"].ravel()
    if not np.array_equal(labels > 0, provenance >= 0):
        raise ValueError(f"Unverified foreground provenance at F{frame}.")
    destination = np.flatnonzero(labels > 0)
    source = provenance[destination]
    identity = labels[destination]
    if len(np.unique(source)) != len(source) or (source >= SIZE * SOURCE_WIDTH).any():
        raise ValueError("Foreground provenance has duplicate or invalid source pixels.")
    py, px = np.divmod(destination, SIZE)
    sy, sx = np.divmod(source, SOURCE_WIDTH)
    query = (py // PATCH) * EGO_COLS + px // PATCH
    key = (sy // PATCH) * SOURCE_COLS + sx // PATCH
    chosen, mixed = choose_samples(key, identity, query)
    if not len(chosen):
        raise ValueError("No trusted foreground sample remains.")
    budget = np.bincount(key[chosen], minlength=ROWS * SOURCE_COLS)
    if budget.max() > BUDGET or any(budget[block] for block in mixed):
        raise ValueError("A mixed source block or oversubscribed block entered the plan.")
    if not np.array_equal(np.unique(np.column_stack((query, identity)), axis=0),
                          np.unique(np.column_stack((query[chosen], identity[chosen])), axis=0)):
        raise ValueError("The budget or mixed-block check removed an entire visible object branch.")

    xyz = np.column_stack((sx[chosen], sy[chosen], np.ones(len(chosen)))) @ np.linalg.inv(metadata["camera_intrinsics"]).T
    xyz *= depth[sy[chosen], sx[chosen], None]
    w2c = np.asarray(metadata["camera_extrinsics"])
    world = np.linalg.solve(w2c[:, :3], (xyz - w2c[:, 3]).T).T
    ego = np.asarray(metadata["ego_extrinsics"][frame])
    center = np.linalg.solve(ego[:, :3], -ego[:, 3])
    direction = world - center
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    routing, group, counts = np.unique(np.column_stack((query[chosen], identity[chosen], key[chosen])),
                                       axis=0, return_inverse=True, return_counts=True)
    pooled = np.zeros((len(routing), 3), np.float64)
    np.add.at(pooled, group, direction)
    pooled /= counts[:, None]
    # Same camera query rays as the pinned Aria arm. Replace only eligible EXO edge geometry.
    rays = corrected_rays(infer, metadata, frame).reshape(-1, 3)
    cosine = np.clip(np.sum(rays[routing[:, 0]] * pooled, axis=1), -1, 1)

    time = frame // 4
    key_y, key_x = np.divmod(routing[:, 2], SOURCE_COLS)
    q = time * ROWS * EGO_COLS + routing[:, 0]
    exo = (time * ROWS + key_y) * COLS + key_x
    evidence = np.column_stack((np.full(len(chosen), frame), destination[chosen], source[chosen],
                                identity[chosen])).astype(np.int32)
    edges = np.column_stack((q, routing[:, 1], exo, counts)).astype(np.int32)
    summary = {
        "time_slot": time,
        "input_visible_pixels": len(source),
        "selected_pixels": len(chosen),
        "maximum_samples_per_source_block": int(budget.max()),
        "source_blocks_at_budget": int((budget == BUDGET).sum()),
        "mixed_source_blocks_blocked_yx": [[y, x] for y, x in (divmod(item, SOURCE_COLS) for item in mixed)],
        "per_object": {
            str(int(obj)): {"selected_pixels": int((identity[chosen] == obj).sum()),
                            "routes": int((edges[:, 1] == obj).sum())}
            for obj in np.unique(identity)
        },
        "routed_query_patches": len(np.unique(q)),
        "object_branches": len(np.unique(edges[:, :2], axis=0)),
    }
    return edges, cosine.astype(np.float32), evidence, summary


def render_selected_points(output, prior, samples):
    frames = review_frames(prior, SIZE, FRAMES)
    colors = {1: (120, 246, 154), 2: (255, 207, 87), 4: (103, 185, 255), 6: (103, 185, 255)}
    sheet = Image.new("RGB", (932, 1000), "#15202b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=19)
    draw.text((12, 12), "Frozen geometry prior", font=font, fill="white")
    draw.text((472, 12), "Geometry samples routed into GGA", font=font, fill="white")
    for x, (value, label) in zip((472, 602, 676), ((1, "Teammate"), (2, "Ball"), (4, "Marker disc"))):
        draw.rectangle((x, 43, x + 10, 53), fill=colors[value])
        draw.text((x + 16, 37), label, font=font, fill="white")
    for row, frame in enumerate(FRAMES):
        y = 64 + row * 466
        prior_frame = frames[frame]
        rgb = (np.asarray(prior_frame).astype(np.float32) * 0.55).astype(np.uint8)
        for _, dst, _, identity in samples[samples[:, 0] == frame]:
            yy, xx = divmod(dst, SIZE)
            rgb[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] = colors[identity]
        sheet.paste(prior_frame, (12, y))
        sheet.paste(Image.fromarray(rgb), (472, y))
        draw.text((24, y + 13), f"F{frame:02}", font=font, fill="white", stroke_width=2, stroke_fill="#15202b")
    sheet.save(output / "routed-points.png")


def build(output):
    previous = RUN / "pair05-foreground-match-20261122/results"
    route_file = previous / "foreground-routing.npz"
    prior_report = previous / "verification.json"
    meta = RUN / f"ball-view-cpu-review/{TAKE}-ball-view-meta.json"
    infer = PROJECT / "vendor/EgoX/infer.py"
    for path, digest in ((route_file, ROUTING_HASH), (prior_report, VERIFICATION_HASH), (meta, META_HASH),
                         (infer, INFER_HASH)):
        if sha256(path) != digest:
            raise ValueError(f"Verified input changed: {path}")
    old = json.loads(prior_report.read_text())
    for frame in FRAMES:
        if any(row["admitted"] for row in old["measurements"][f"F{frame:02}"]["rejected"].values()):
            raise ValueError("Upstream negative correspondences did not pass validation.")
    metadata, = json.loads(meta.read_text())["test_datasets"]
    receipts = json.loads((RUN / "update-008/output-008/logs/validated-inputs.json").read_text())
    receipt, = [item for item in receipts if item["name"] == TAKE]
    depths = RUN / f"multiclip-evidence/{TAKE}/geometry/vipe_results/{TAKE}/depth/exo.zip"
    edges, weights, samples, summary = [], [], [], {}
    with np.load(route_file, allow_pickle=False) as data, ZipFile(depths) as archive:
        source = infer.read_text()
        for frame in FRAMES:
            depth = load_depth(archive, frame)
            serialized = BytesIO()
            np.save(serialized, depth, allow_pickle=False)
            if hashlib.sha256(serialized.getvalue()).hexdigest() != receipt["depth_sha256"][frame]:
                raise ValueError(f"Frozen depth payload changed at F{frame}.")
            edge, cos, support, frame_report = foreground_rows(frame, data, depth, metadata, source)
            edges.append(edge)
            weights.append(cos)
            samples.append(support)
            summary[f"F{frame:02}"] = frame_report
    output.mkdir(parents=True)
    plan = output / "pair05-gga-routes.npz"
    selected = np.concatenate(samples)
    np.savez_compressed(plan, edges=np.concatenate(edges), cosine=np.concatenate(weights),
                        samples=selected, frames=np.array(FRAMES, dtype=np.int32),
                        metadata_sha256=np.array(META_HASH), exo_sha256=np.array(receipt["exo_sha256"]),
                        prior_sha256=np.array(receipt["prior_sha256"]))
    report = {"plan_sha256": sha256(plan), "seed_geometry": "pinned Pair05 Aria arm",
              "edge_columns": ["flat_ego_query", "object_id", "flat_exo_key", "sample_count"],
              "sample_columns": ["source_and_target_frame", "destination_flat_pixel", "source_flat_pixel", "object_id"],
              "frames": summary}
    (output / "plan-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    prior = RUN / f"update-008/output-008/priors/{TAKE}-final.mp4"
    if sha256(prior) != receipt["prior_sha256"]:
        raise ValueError("Frozen Pair05 prior changed.")
    render_selected_points(output, prior, selected)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: build_routes.py NEW_OUTPUT_DIRECTORY")
    print(json.dumps(build(Path(sys.argv[1])), indent=2))
