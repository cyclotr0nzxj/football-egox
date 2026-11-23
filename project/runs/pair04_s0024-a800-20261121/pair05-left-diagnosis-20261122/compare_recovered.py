"""Compare the two recovered Pair05 outputs and quantify blank prior coverage."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

RUN = Path(__file__).resolve().parent.parent
GGA = RUN / "pair05-gga-foreground-20261122"
sys.path.insert(0, str(GGA))
from review_foreground_video import (  # noqa: E402
    EGO_SIZE, FPS, FRAME_COUNT, KEY_FRAMES, PLAN_SHA, REFERENCES, TAKE,
    file_sha256, read_frames,
)


def rgb_vectors(frames, index):
    return np.asarray(frames[index], dtype=np.float32)


def gradients(rgb, region):
    y, x = region
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)[y, x]
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)[y, x]
    return np.concatenate((dx.ravel(), dy.ravel()))


def measure():
    verified = RUN / "clone-004-20261122/pair05/verified"
    receipt = json.loads((verified / "verification.json").read_text())
    assert receipt["frame_mapping"]["alignment"] == "same_zero_based_source_frame_index"
    historical = REFERENCES["Historical Aria output"]
    prior = REFERENCES["Locked geometry prior"]
    candidate = receipt["verified_export"]
    indices = range(FRAME_COUNT)
    old_frames = read_frames(RUN / historical[0], *historical[1:], indices)
    new_frames = read_frames(verified / candidate["video"], candidate["sha256"], EGO_SIZE, FPS, indices)
    prior_frames = read_frames(RUN / prior[0], *prior[1:], KEY_FRAMES)
    plan_path = GGA / "plan/pair05-gga-routes.npz"
    assert file_sha256(plan_path) == PLAN_SHA == receipt["remote_receipt"]["foreground_gga"]["plan_sha256"]
    with np.load(plan_path, allow_pickle=False) as plan:
        samples = plan["samples"].copy()

    y, x = np.mgrid[:EGO_SIZE, :EGO_SIZE]
    circle = (x - 224)**2 + (y - 224)**2 <= 224**2
    upper = circle & (y < 224)
    lower = circle & ~upper
    fence = (slice(95, 280), slice(82, 366))
    rows = []
    for index in indices:
        old, new = rgb_vectors(old_frames, index), rgb_vectors(new_frames, index)
        error = np.abs(old - new).mean(axis=2)
        row = {
            "frame": index,
            "old_new_circle_rgb_mae_0_to_255": float(error[circle].mean()),
            "old_new_fence_region_gradient_correlation": float(
                np.corrcoef(gradients(old, fence), gradients(new, fence))[0, 1]
            ),
        }
        if index in KEY_FRAMES:
            destination = samples[samples[:, 0] == index, 1]
            patch_y, patch_x = np.divmod(destination, EGO_SIZE)
            routed_patches = np.zeros((28, 28), dtype=bool)
            routed_patches[patch_y // 16, patch_x // 16] = True
            routed = routed_patches[y // 16, x // 16] & circle
            blank = rgb_vectors(prior_frames, index).max(axis=2) <= 8
            row.update({
                "routed_patches": int(routed_patches.sum()),
                "old_new_routed_rgb_mae_0_to_255": float(error[routed].mean()),
                "old_new_other_rgb_mae_0_to_255": float(error[circle & ~routed].mean()),
                "prior_near_black_upper_circle_fraction": float(blank[upper].mean()),
                "prior_near_black_lower_circle_fraction": float(blank[lower].mean()),
            })
        rows.append(row)
    return {
        "comparison": "Recovered Aria output versus recovered foreground GGA output, same source index",
        "decoder_checks": "49 frames, exact dimensions/PTS, source hashes",
        "fence_region_xyxy_exclusive": [82, 95, 366, 280],
        "near_black_max_rgb_threshold": 8,
        "limitation": "Descriptive comparison; neither gradient similarity nor black pixels establish cause or semantic labels.",
        "median_fence_gradient_correlation": float(np.median([
            row["old_new_fence_region_gradient_correlation"] for row in rows
        ])),
        "median_circle_rgb_mae_0_to_255": float(np.median([
            row["old_new_circle_rgb_mae_0_to_255"] for row in rows
        ])),
        "per_frame": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = measure()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
