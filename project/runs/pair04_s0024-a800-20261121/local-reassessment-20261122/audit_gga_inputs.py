"""Audit actual Pair01/05 source sampling and geometric attention candidates on CPU."""

from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT = RUN.parents[1]
for directory in ("ball-view-cpu-review", "pair01-gga-ab"):
    sys.path.insert(0, str(RUN / directory))
from build_ball_previews import load_depth
from check_inputs import sha256
from prepare_infer import OFFICIAL_INVERSE, ARIA_INVERSE, OFFICIAL_INFER_SHA256, run_lens_block

FRAMES = tuple(range(0, 49, 4))
HEIGHT = 448
WIDTH = 784
ROWS = HEIGHT // 16
COLS = WIDTH // 16


def load_json(path):
    return json.loads(path.read_text())


def masks(archive, frame):
    png = archive.read(f"{frame:05}.png")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image.shape != (HEIGHT, WIDTH):
        raise ValueError("Unexpected archived EXO mask dimensions.")
    return image > 0


def pool_spatial(grid, columns, reduce):
    return reduce(grid.reshape(*grid.shape[:-2], ROWS, 2, columns, 2), axis=(-3, -1))


def support(mask):
    # EgoX align_corners=False at 1/8 samples four neighbors per latent grid cell.
    sampled = np.sum([mask[dy::8, dx::8] for dy, dx in ((3, 3), (3, 4), (4, 3), (4, 4))], axis=0)
    return pool_spatial(sampled, COLS, np.sum)


def sampled_world(depth, k, c2w):
    result = np.zeros((56, 98, 3), dtype=np.float32)
    for dy, dx in [(3, 3), (3, 4), (4, 3), (4, 4)]:
        yy, xx = np.mgrid[dy:HEIGHT:8, dx:WIDTH:8].astype(np.float32)
        ray = np.stack(((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)), axis=-1)
        result += ray * depth[dy::8, dx::8, None]
    return (result * 0.25) @ c2w[:3, :3].T + c2w[:3, 3]


def unit(value):
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def pose_diff(take, updated):
    first = load_json(RUN / f"multiclip-evidence/{take}/teammate-view/meta.json")["test_datasets"][0]
    a, b = [np.asarray(meta["ego_extrinsics"]) for meta in (first, updated)]
    cosines = np.clip(np.sum(a[:, 2, :3] * b[:, 2, :3], axis=-1), -1, 1)
    centers = [np.linalg.solve(p[:, :, :3], -p[:, :, 3:]) for p in (a, b)]
    return {
        "max_camera_center_delta": float(np.abs(centers[0] - centers[1]).max()),
        "forward_angle_difference_median_deg": float(np.median(np.rad2deg(np.arccos(cosines)))),
        "same_prompt_and_source_K": first["prompt"] == updated["prompt"] and first["camera_intrinsics"] == updated["camera_intrinsics"],
    }


def corrected_rays(source, metadata, frame):
    y, x = np.mgrid[:56, :56]
    grid = np.stack((x, y), axis=-1).reshape(-1, 1, 2).astype(np.float32)
    k = np.asarray(metadata["ego_intrinsics"], dtype=np.float32)
    k[:2, :3] /= 8
    rays, _ = run_lens_block(source.replace(OFFICIAL_INVERSE, ARIA_INVERSE, 1), grid, k)
    w2c = np.asarray(metadata["ego_extrinsics"][frame], dtype=np.float32)
    rays = unit(np.column_stack((rays, np.ones(len(rays), dtype=np.float32)))) @ w2c[:, :3]
    rays = np.rot90(rays.reshape(56, 56, 3), k=-1)
    return pool_spatial(rays.transpose(2, 0, 1), 28, np.mean).transpose(1, 2, 0)


def inspect(source, metadata, receipt, anchors, take, frame):
    source_k = np.asarray(metadata["camera_intrinsics"], dtype=np.float32)
    k = source_k.copy()
    for axis, center in ((0, WIDTH / 2), (1, HEIGHT / 2)):
        k[axis, axis] *= center / k[axis, 2]
        k[axis, 2] = center
    inv_source = np.linalg.inv(np.vstack([metadata["camera_extrinsics"], [0, 0, 0, 1]])).astype(np.float32)
    inv_ego = np.linalg.inv(np.vstack([metadata["ego_extrinsics"][frame], [0, 0, 0, 1]])).astype(np.float32)
    ball_mask_zip = "pair05-ball" if take.startswith("pair05") else "pair01-wearer-upper-body"
    recordings = RUN / "update-008/ball-recovery"
    geometry = RUN / f"multiclip-evidence/{take}/geometry/vipe_results/{take}/depth/exo.zip"
    world, ball_counts, wearer_counts = [], [], []
    with ZipFile(geometry) as depths, ZipFile(recordings / f"{ball_mask_zip}.zip") as annotations:
        yy, xx = np.mgrid[:HEIGHT, :WIDTH]
        for i in FRAMES:
            depth = load_depth(depths, i)
            npy = BytesIO()
            np.save(npy, depth, allow_pickle=False)
            if hash_bytes(npy.getvalue()).hexdigest() != receipt["depth_sha256"][i]:
                raise ValueError(f"Frozen ViPE depth changed at {take} F{i}.")
            world.append(sampled_world(depth, k, inv_source))
            ax, ay, radius = anchors[i][1:4]
            if take.startswith("pair05"):
                ball = masks(annotations, i)
                wearer = np.zeros_like(ball)
            else:
                # Pair01 has an approximate manually inspected ball core, not a per-pixel ball segmentation.
                ball = (xx - ax)**2 + (yy - ay)**2 <= (radius * 0.65)**2
                wearer = masks(annotations, i)
            ball_counts.append(support(ball))
            wearer_counts.append(support(wearer))
            if i == frame:
                z = depth[ball]
                if not z.size:
                    raise ValueError("No observed ball depth at the chosen query frame.")
                d = float(np.median(z))
                cam_ray = np.array([(ax - k[0, 2]) / k[0, 0], (ay - k[1, 2]) / k[1, 1], 1])
                ball_world = (cam_ray * d) @ inv_source[:3, :3].T + inv_source[:3, 3]

    world = np.stack(world)
    ball_counts = np.stack(ball_counts)
    wearer_counts = np.stack(wearer_counts)
    vectors = unit(world - inv_ego[:3, 3]).transpose(0, 3, 1, 2)
    keys = pool_spatial(vectors, COLS, np.mean).transpose(0, 2, 3, 1)
    query_grid = corrected_rays(source, metadata, frame)
    true_ray = unit(ball_world - inv_ego[:3, 3])
    y, x = np.unravel_index(np.argmax(unit(query_grid) @ true_ray), (28, 28))
    query = query_grid[y, x]
    score = np.clip(keys @ query, -1, 1)
    # The CUDA transformer subsequently rounds pooled directions to bfloat16 and adds learned logits.
    geometric_weight = 1 + np.where(score > 0, 3 * score, score)
    near = (ball_counts > 0) & (score >= 0.98)
    protected_ball = ball_counts > 0
    wearer_gate = (wearer_counts >= 8) & ~protected_ball
    other_times = np.asarray(FRAMES)[:, None, None] != frame
    time_gate = protected_ball & other_times
    row = FRAMES.index(frame)
    return {
        "pose_comparison": pose_diff(take, metadata),
        "query_frame": frame,
        "query_token_yx": [int(y), int(x)],
        "same_frame_observed_ball_candidate_keys": int(protected_ball[row].sum()),
        "same_frame_ball_candidate_max_support_of_16": int(ball_counts[row].max()),
        "same_frame_ball_candidate_best_geometric_weight": float(geometric_weight[row][protected_ball[row]].max())
        if protected_ball[row].any() else None,
        "ball_candidate_source_frames_dot_at_least_0_98": [i for i, included in zip(FRAMES, near.any(axis=(1, 2))) if included],
        "all_source_eligible_keys": int(keys.shape[0] * keys.shape[1] * keys.shape[2]),
        "all_ego_ungated_keys": len(FRAMES) * ROWS * ROWS,
        "similar_other_time_ball_candidates_removed_by_time_gate": int((time_gate & near).sum()),
        "same_frame_ball_candidates_retained_by_time_gate": int((protected_ball & ~time_gate)[row].sum()),
        "pair01_wearer_gate": {
            "upper_body_dominant_source_keys": int((wearer_counts >= 8).sum()),
            "blocked_with_ball_protection": int(wearer_gate.sum()),
            "observed_ball_keys_blocked": int((wearer_gate & protected_ball).sum()),
        } if take.startswith("pair01") else None,
    }


def main():
    source_path = PROJECT / "vendor/EgoX/infer.py"
    if sha256(source_path) != OFFICIAL_INFER_SHA256:
        raise ValueError("Unexpected upstream infer.py.")
    source = source_path.read_text()
    logs = RUN / "update-008/output-008/logs"
    rows = load_json(logs / "meta.json")["test_datasets"]
    receipts = load_json(logs / "validated-inputs.json")
    marked = load_json(RUN / "ball-view-cpu-review/ball-anchors.json")
    result = {
        "scope": "NumPy float32 reconstruction of actual EXO samples and selected GGA geometric logits. No learned attention, denoising or GPU.",
        "ball_support": "Pair05 uses the observed-ball mask. Pair01 uses the inspected ball anchor's inner 65% radius.",
        "takes": {},
    }
    for take, frame in (("pair01_s0288", 36), ("pair05_s0024", 24)):
        metadata, = [r for r in rows if f"/{take}/" in r["exo_path"]]
        receipt, = [r for r in receipts if r["name"] == take]
        anchors = {entry[0]: entry for entry in marked[take]}
        result["takes"][take] = inspect(source, metadata, receipt, anchors, take, frame)
    dest = HERE / "gga-input-audit.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
