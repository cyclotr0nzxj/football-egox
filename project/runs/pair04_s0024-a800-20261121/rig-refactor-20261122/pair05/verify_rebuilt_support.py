#!/usr/bin/env python3
"""Verify observable Pair05 RGB support for a newly reconstructed rig."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_timeline import MATCH, RUN, TAKE, match, routes


FRAMES = tuple(range(49))
CHECK_FRAMES = (0, 12, 24, 36, 40, 48)


def source_ids(rgb, instances, ball):
    labels = np.zeros(instances.shape, dtype=np.int16)
    labels[instances == 2] = match.TEAMMATE
    labels[ball & (instances == 0)] = match.BALL
    # These are the four fixed EXO positions already reviewed in the original shot.
    # Only pixels in a cyan component touching the reviewed anchor are evidence.
    for label, (x, y) in enumerate(match.DISC_SOURCE, start=3):
        region = np.s_[y - 5:y + 6, x - 7:x + 8]
        red, green, blue = rgb[region].astype(np.int16).transpose(2, 0, 1)
        cyan = (green - red > 40) & (blue - red > 35) & (green >= 115) & (instances[region] == 0) & ~ball[region]
        _, components = cv2.connectedComponents(cyan.astype(np.uint8))
        center = components[5, 7]
        if center and np.count_nonzero(components == center) >= 5:
            labels[region][components == center] = label
    return labels


def object_evidence(destinations, rgb, rendered, obj):
    if not len(destinations):
        return {"validated_pixels": 0}
    yy, xx = np.divmod(destinations, match.SIZE)
    xy = np.column_stack((xx, yy))
    center = np.median(xy, axis=0)
    mask = np.zeros((match.SIZE, match.SIZE), dtype=np.uint8)
    mask[yy, xx] = 1
    # Group adjacent projection holes without counting any added pixel as evidence.
    neighborhood = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    _, components = cv2.connectedComponents(neighborhood)
    membership = components[yy, xx]
    counts = np.bincount(membership)
    color_error = np.abs(rgb.astype(np.int16) - rendered[yy, xx].astype(np.int16)).mean(axis=1)
    evidence = {
        "validated_pixels": len(destinations),
        "nearby_components": int(np.count_nonzero(counts)),
        "largest_nearby_component_fraction": float(counts.max() / len(destinations)),
        "target_center_xy_median": [float(k) for k in center],
        "radius_p90_from_target_median_px": float(np.quantile(np.linalg.norm(xy - center, axis=1), 0.9)),
        "visible_rgb_mae_median": float(np.median(color_error)),
    }
    if obj == match.BALL:
        evidence["rendered_rgb_near_black_fraction"] = float((rendered[yy, xx].max(axis=1) <= 8).mean())
    return evidence


def mapped_pixels(frame, arrays):
    labels = arrays[f"F{frame:02}_target_label"]
    index = arrays[f"F{frame:02}_source_index"]
    shape = (match.SIZE, match.SIZE)
    if (labels.shape != shape or index.shape != shape or labels.dtype.kind not in "iu"
            or index.dtype.kind != "i" or not set(np.unique(labels)).issubset({0, *match.NAMES})):
        raise ValueError(f"Unexpected provenance format at F{frame:02}.")
    if not np.array_equal(labels > 0, index >= 0) or (index[labels == 0] != -1).any():
        raise ValueError(f"Missing or extra explicit foreground source index at F{frame:02}.")
    destination = np.flatnonzero(labels.ravel() > 0)
    source = index.ravel()[destination]
    if (source >= match.HEIGHT * match.WIDTH).any() or len(np.unique(source)) != len(source):
        raise ValueError(f"Invalid or repeated same-frame EXO pixel at F{frame:02}.")
    return destination, source, labels.ravel()[destination]


def render_review(path, previews, verified):
    size = 210
    sheet = Image.new("RGB", (3 * (size * 2 + 10) + 10, 50 + 2 * (size + 36)), "#17232e")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    draw.text((10, 12), "Rebuilt geometry / independently verified EXO RGB support", font=font, fill="white")
    for index, frame in enumerate(CHECK_FRAMES):
        x = 10 + index % 3 * (size * 2 + 10)
        y = 50 + index // 3 * (size + 36)
        gray = (np.asarray(previews[frame]).astype(np.float32) * 0.55).astype(np.uint8)
        for _, dst, _, object_id, *_ in verified[frame]:
            yy, xx = divmod(int(dst), match.SIZE)
            gray[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3] = match.COLORS[object_id]
        draw.text((x, y), f"F{frame:02}", font=font, fill="white")
        sheet.paste(previews[frame].resize((size, size)), (x, y + 21))
        sheet.paste(Image.fromarray(gray).resize((size, size)), (x + size + 3, y + 21))
    sheet.save(path, quality=92)


def build(provenance, provenance_sha, preview, preview_sha, preview_fps, output):
    old = MATCH / "results/verification.json"
    match.require_hash(old, routes.VERIFICATION_HASH)
    original = json.loads(old.read_text())["input_hashes"]
    exo = RUN / f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4"
    masks = RUN / f"multiclip-evidence/{TAKE}/geometry/vipe_results/{TAKE}/mask/exo.zip"
    balls = RUN / "update-008/ball-recovery/pair05-ball.zip"
    for key, path in (("exo", exo), ("instances", masks), ("balls", balls)):
        match.require_hash(path, original[key])
    match.require_hash(provenance, provenance_sha)
    match.require_hash(preview, preview_sha)
    match.check_reference(exo, match.WIDTH, 25)
    match.check_reference(preview, match.SIZE, preview_fps)
    source_frames = match.review_frames(exo, match.WIDTH, FRAMES)
    angle_frames = match.review_frames(preview, match.SIZE, FRAMES)
    frames = {}
    verified = {}
    with np.load(provenance, allow_pickle=False) as arrays, ZipFile(masks) as people, ZipFile(balls) as ball:
        for frame in FRAMES:
            rgb = np.asarray(source_frames[frame])
            labels = source_ids(rgb, match.read_mask(people, frame), match.read_mask(ball, frame) == 255)
            target, source, identity = mapped_pixels(frame, arrays)
            match_id = labels.ravel()[source] == identity
            colors = rgb.reshape(-1, 3)[source[match_id]]
            anchors = np.column_stack((np.full(int(match_id.sum()), frame), target[match_id],
                                       source[match_id], identity[match_id], colors)).astype(np.int32)
            verified[frame] = anchors
            report = {"object_id_conflicts_by_label": {match.NAMES[k]: int(((identity == k) & ~match_id).sum())
                                                      for k in np.unique(identity) if ((identity == k) & ~match_id).any()},
                      "objects": {}}
            for obj, name in match.NAMES.items():
                selected = anchors[:, 3] == obj
                report["objects"][name] = object_evidence(anchors[selected, 1], anchors[selected, 4:],
                                                          np.asarray(angle_frames[frame]), obj)
            frames[f"F{frame:02}"] = report

    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "verified-observed-rgb.npz"
    np.savez_compressed(artifact, samples=np.concatenate(list(verified.values())),
                        exo_sha256=np.asarray(original["exo"]), provenance_sha256=np.asarray(provenance_sha),
                        preview_sha256=np.asarray(preview_sha))
    report = {
        "scope": "independent same-frame source identity check and sparse RGB observations; no denoising",
        "status": "CPU_DIAGNOSTIC_ONLY",
        "sample_columns": ["frame", "target_flat_pixel", "exo_flat_pixel", "object_id", "r", "g", "b"],
        "source_and_preview_pts": "same frame index, EXO i/25 and preview i/30 verified separately",
        "exo_sha256": original["exo"],
        "preview_sha256": preview_sha, "provenance_sha256": provenance_sha,
        "verified_observed_rgb_sha256": match.sha256(artifact),
        "frames_with_source_identity_conflicts": [key for key, value in frames.items() if value["object_id_conflicts_by_label"]],
        "frames_with_both_teammate_and_ball_pixels": [key for key, value in frames.items()
                                                      if all(value["objects"][name]["validated_pixels"] > 0 for name in ("teammate", "ball"))],
        "frames": frames,
    }
    (output / "rebuilt-support-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    render_review(output / "rebuilt-support-review.jpg", angle_frames, verified)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("rig_results", type=Path)
    cli.add_argument("variant", choices=("nominal", "alternate"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    receipt = json.loads((args.rig_results / "rebuild-receipt.json").read_text())
    baseline = MATCH / "results/verification.json"
    match.require_hash(baseline, routes.VERIFICATION_HASH)
    exo_sha = json.loads(baseline.read_text())["input_hashes"]["exo"]
    if (receipt["status"] != "CPU_DIAGNOSTIC_ONLY" or not receipt["all_49_sealed_depth_digests_verified"]
            or receipt["source_sha256"]["EXO"] != exo_sha
            or receipt["source_indices"] != [24, 72] or receipt["frame_count"] != len(FRAMES)
            or receipt["source_fps"] != 25 or receipt["preview_fps"] != 30):
        raise ValueError("Rebuilt rig receipt does not bind the expected frozen Pair05 shot.")
    provenance_name = f"{args.variant}-provenance.npz"
    preview_name = f"{args.variant}-prior.mp4"
    build(args.rig_results / provenance_name, receipt["products"][provenance_name],
          args.rig_results / preview_name, receipt["products"][preview_name], receipt["preview_fps"], args.output)
