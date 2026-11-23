"""Audit direct source support of released EgoX sampling on sealed EXO evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair05-foreground-match-20261122"))
from build_foreground_match import (  # noqa: E402
    BALL, HEIGHT, SIZE, TEAMMATE, WIDTH, closest_surface, coefficients, load_depth, project, require_hash,
)
from build_ball_previews import COUNTER_ROLL, inputs  # noqa: E402


PATCH = 16
SOURCE_STRIDE = 8
GRID = {
    "official": tuple(offset for index in range(2) for offset in
                      (index * SOURCE_STRIDE + 3, index * SOURCE_STRIDE + 4)),
    "uniform": (1, 5, 9, 13),
}
SUPPORT_RADII = (1.0, 0.6, 0.35)
EXCLUDED_FRAMES = (12, 24)
POSE_DIR = RUN / "ball-view-cpu-review"
TAKES = ("pair01_s0288", "pair04_s0024", "pair05_s0024")
ROOT = RUN.parents[1]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def summarize(counts):
    """Counts are paired per target; baseline is the release footprint."""
    values = np.asarray(counts, dtype=int).reshape(-1, 2)
    return {
        "frames": len(values),
        "official_misses": int((values[:, 0] == 0).sum()),
        "uniform_misses": int((values[:, 1] == 0).sum()),
        "median_counts_official_uniform": np.median(values, axis=0).tolist() if len(values) else None,
        "uniform_more": int((values[:, 1] > values[:, 0]).sum()),
        "official_more": int((values[:, 0] > values[:, 1]).sum()),
    }


def phase_control():
    """All integer sub-patch phases with both layouts tiled across boundaries."""
    yy, xx = np.mgrid[-PATCH:2 * PATCH, -PATCH:2 * PATCH]
    distances = []
    for offsets in GRID.values():
        support = np.isin(yy % PATCH, offsets) & np.isin(xx % PATCH, offsets)
        distances.append([
            int(((xx[support] - x)**2 + (yy[support] - y)**2).min())
            for y in range(PATCH) for x in range(PATCH)
        ])
    squared = np.asarray(distances).T
    by_radius = {str(radius): summarize(squared <= radius**2) for radius in range(1, 9)}
    counterexamples = np.flatnonzero((squared[:, 0] <= 1) & (squared[:, 1] > 1))
    if not len(counterexamples):
        raise ValueError("Expected a phase where uniform misses a radius-one disk.")
    y, x = divmod(int(counterexamples[0]), PATCH)
    return {"by_radius_px": by_radius, "radius_one_uniform_miss_official_hit_center_xy": [x, y]}


def source_samples(xx, yy):
    grids = [np.isin(xx % PATCH, offsets) & np.isin(yy % PATCH, offsets)
             for offsets in GRID.values()]
    for mask in grids:
        per_patch = mask.reshape(HEIGHT // PATCH, PATCH, WIDTH // PATCH, PATCH).sum(axis=(1, 3))
        if not np.all(per_patch == 16):
            raise ValueError("A footprint breaks the equal budget.")
    return np.stack(grids).reshape(2, -1)


def projected_sources(frame, archived_depth, depths, view, distortion, xx, yy):
    depth = load_depth(archived_depth, frame)
    # Dataset receipts hash the uncompressed float32 tensor in NPY representation.
    buffer = io.BytesIO()
    np.save(buffer, depth, allow_pickle=False)
    if hashlib.sha256(buffer.getvalue()).hexdigest() != depths[frame]:
        raise ValueError(f"Frozen depth checksum mismatch at F{frame:02}.")
    camera_k = np.asarray(view["camera_intrinsics"])
    if not np.allclose(np.asarray(view["camera_extrinsics"]), np.eye(4)[:3]):
        raise ValueError("This audit expects a fixed EXO camera as world frame.")
    points = np.column_stack((((xx - camera_k[0, 2]) / camera_k[0, 0]).ravel(),
                              ((yy - camera_k[1, 2]) / camera_k[1, 1]).ravel(),
                              np.ones(xx.size))) * depth.reshape(-1, 1)
    pose = np.asarray(view["ego_extrinsics"][frame])
    source, destination, z = project(points, pose, np.asarray(view["ego_intrinsics"]),
                                     distortion, np.arange(len(points)))
    winners, _ = closest_surface(source, destination, z)
    gy, gx = np.mgrid[:SIZE, :SIZE]
    in_circle = ((gx - SIZE / 2)**2 + (gy - SIZE / 2)**2 <= (SIZE / 2)**2).ravel()
    return winners[in_circle & (winners >= 0)]


def inspect_take(take, receipt, anchors, distortion, xx, yy, samples):
    source, geometry, _, _ = inputs(take)
    exo = source / "example/self_capture/videos" / take / "exo.mp4"
    exo_digest = require_hash(exo, receipt["exo_sha256"])
    stream_info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(exo),
    ], text=True))["streams"][0]
    if stream_info != {"width": WIDTH, "height": HEIGHT, "r_frame_rate": "25/1", "nb_frames": "49"}:
        raise ValueError(f"{take} has unexpected EXO dimensions or timing.")
    if [item[0] for item in anchors] != list(range(0, 49, 4)):
        raise ValueError(f"{take} anchors no longer align to the staged temporal lattice.")
    camera_path = POSE_DIR / f"{take}-ball-view-meta.json"
    camera = read_json(camera_path)["test_datasets"][0]
    rows, excluded = [], []
    with ZipFile(geometry / "depth/exo.zip") as archived_depth:
        for frame, x, y, radius, status in anchors:
            if frame in EXCLUDED_FRAMES or status != "visible":
                excluded.append([frame, status])
                continue
            base_dist_sq = ((xx - x)**2 + (yy - y)**2).ravel()
            winners = projected_sources(frame, archived_depth, receipt["depth_sha256"], camera,
                                        distortion, xx, yy)
            row = {"frame": frame, "anchor_xy_radius": [x, y, radius], "support": {}}
            for scale in SUPPORT_RADII:
                disk = base_dist_sq <= (radius * scale)**2
                winner_in_disk = disk[winners]
                row["support"][str(scale)] = {
                    "disk_px": int(disk.sum()),
                    "source_counts_official_uniform": (samples & disk).sum(axis=1).tolist(),
                    "projected_disk_winners": int(winner_in_disk.sum()),
                    "winner_counts_official_uniform": (samples[:, winners] & winner_in_disk).sum(axis=1).tolist(),
                }
            rows.append(row)
    summary = {}
    for scale in map(str, SUPPORT_RADII):
        records = [row["support"][scale] for row in rows]
        summary[scale] = {
            "source": summarize([row["source_counts_official_uniform"] for row in records]),
            "projected": summarize([row["winner_counts_official_uniform"] for row in records
                                    if row["projected_disk_winners"] > 0]),
            "no_projected_disk_winner": sum(row["projected_disk_winners"] == 0 for row in records),
        }
    return {"exo_sha256": exo_digest, "virtual_camera_sha256": digest(camera_path), "input": stream_info,
            "included": rows, "excluded": excluded, "summary": summary}


def audit():
    lens = ROOT / "records/official_renderer_geometry.py"
    distortion = np.concatenate([coefficients(lens, f"{term}_distortion_coeffs")
                                 for term in ("radial", "tangential", "thinPrism")])
    anchors_path = POSE_DIR / "ball-anchors.json"
    anchors = read_json(anchors_path)
    receipts = {row["name"]: row for row in read_json(RUN / "multiclip-evidence/validated-inputs.json")}
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    samples = source_samples(xx, yy)
    takes = {take: inspect_take(take, receipts[take], anchors[take], distortion, xx, yy, samples)
             for take in TAKES}
    primary = [entry["summary"]["0.6"]["projected"] for entry in takes.values()]
    no_new_full_radius_miss = all(
        not any(row["support"]["1.0"]["winner_counts_official_uniform"][0] > 0
                and row["support"]["1.0"]["winner_counts_official_uniform"][1] == 0
                for row in entry["included"]) for entry in takes.values()
    )
    return {
        "official_code_ref": "43c095b552a8f09d3a81c0882491f52c5cf4e953",
        "positions_per_axis": GRID, "pixels_per_patch_per_method": 16,
        "anchors_sha256": digest(anchors_path), "lens_sha256": digest(lens),
        "protocol_sha256": digest(HERE / "protocol.md"),
        "synthetic_integer_phase_control": phase_control(),
        "takes": takes,
        "passed_substitution_gate": no_new_full_radius_miss and all(
            row["uniform_misses"] < row["official_misses"] for row in primary
        ),
        "interpretation": "Only traces direct RGB-pixel support of the released geometry branch. "
                          "Direction normalization, model attention, video encoding and video quality "
                          "are outside this audit.",
    }


def reviewed_pair05():
    """Read the two after-the-fact contour checks outside the cross-clip protocol."""
    folder = RUN / "occlusion-contact-20261122/pair05/results"
    receipt = read_json(folder / "repair-receipt.json")
    masks_file = folder / "source-ball-masks.npz"
    require_hash(masks_file, receipt["products"][masks_file.name])
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    samples = source_samples(xx, yy)
    results = {}
    with np.load(masks_file, allow_pickle=False) as masks:
        for rig in ("nominal", "alternate"):
            provenance = folder / f"{rig}-provenance.npz"
            require_hash(provenance, receipt["products"][provenance.name])
            results[rig] = {}
            with np.load(provenance, allow_pickle=False) as mapping:
                for frame in ("F12", "F24"):
                    source = mapping[f"{frame}_source_index"]
                    target_label = mapping[f"{frame}_target_label"]
                    ball = source[target_label == BALL]
                    old_mask = masks[f"{frame}_old_ball"].ravel()
                    corrected_mask = masks[f"{frame}_reviewed_ball"].ravel()
                    if np.any(ball < 0) or not np.all(corrected_mask[ball]):
                        raise ValueError(f"{frame} contains target balls without source support.")
                    results[rig][frame] = {
                        "full_target_ball_old_reviewed": [int(old_mask[ball].sum()), len(ball)],
                        "sampled_target_ball_old_official_uniform": (samples[:, ball] & old_mask[ball]).sum(axis=1).tolist(),
                        "sampled_target_ball_reviewed_official_uniform": samples[:, ball].sum(axis=1).tolist(),
                        "sampled_target_teammate_official_uniform": samples[:, source[target_label == TEAMMATE]].sum(axis=1).tolist(),
                        "sampled_target_all_discs_official_uniform": samples[:, source[target_label > BALL]].sum(axis=1).tolist(),
                    }
    return {"pairs": results, "interpretation": "Post hoc, two reviewed contours and frozen target winners. "
            "Counts test direct pixel support of geometric sampling, not generated video."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewed-pair05", action="store_true")
    arguments = parser.parse_args()
    result = reviewed_pair05() if arguments.reviewed_pair05 else audit()
    with arguments.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
