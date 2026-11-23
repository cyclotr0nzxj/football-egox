"""Export exact same-frame RGB observations and expose unsupported target pixels."""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from review_pair01_ab import check_reference, review_frames  # noqa: E402
from verify_egox_outputs_008 import file_sha256  # noqa: E402

SOURCE_W, SOURCE_H, TARGET = 784, 448, 448


def load_correspondences(path):
    mapped = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            match = re.fullmatch(r"F([0-4][0-9])_target_label", name)
            if match is None:
                continue
            index = int(match[1])
            labels = archive[name]
            source = archive[f"F{index:02}_source_index"]
            if (labels.shape != (TARGET, TARGET) or source.shape != labels.shape
                    or not np.issubdtype(labels.dtype, np.integer)
                    or not np.issubdtype(source.dtype, np.signedinteger)
                    or (labels < 0).any() or (source < -1).any()
                    or (source >= SOURCE_W * SOURCE_H).any() or ((labels > 0) & (source < 0)).any()):
                raise ValueError(f"Invalid object provenance at F{index:02}.")
            mapped[index] = labels, source
    if not mapped:
        raise ValueError("Missing same-frame correspondence maps.")
    return mapped


def observed_pixels(source_rgb, labels, source, alternate=None):
    if source_rgb.shape != (SOURCE_H, SOURCE_W, 3) or source_rgb.dtype != np.uint8:
        raise ValueError("Unexpected source RGB layout.")
    visible = labels > 0
    if alternate is not None:
        other_labels, other_indices = alternate
        visible &= (labels == other_labels) & (source == other_indices)
    output = np.zeros((TARGET, TARGET, 3), dtype=np.uint8)
    output[visible] = source_rgb.reshape(-1, 3)[source[visible]]
    return output, visible


def preserve_observations(candidate, exact_rgb, protected):
    if (candidate.shape != (TARGET, TARGET, 3) or candidate.shape != exact_rgb.shape
            or candidate.dtype != np.uint8 or exact_rgb.dtype != np.uint8
            or protected.dtype != bool or protected.shape != candidate.shape[:2]):
        raise ValueError("Candidate and observations must share one RGB pixel grid.")
    output = candidate.copy()
    output[protected] = exact_rgb[protected]
    return output


def audit_pixel_lock(candidate, observed, mask, output):
    if not np.array_equal(output[mask], observed[mask]):
        raise ValueError("The completion changed protected observations.")
    changed = np.any(candidate != output, axis=-1)
    if (changed & ~mask).any():
        raise ValueError("The pixel lock changed unsupported pixels.")
    return int(changed.sum())


def object_counts(labels, visible):
    return {str(value): int(np.count_nonzero((labels == value) & visible))
            for value in np.unique(labels) if value > 0}


def joint_source_support(labels, indices, alternative):
    other_labels, other_indices = alternative
    result = {}
    for identity in np.unique(labels):
        if identity == 0:
            continue
        first = np.flatnonzero(labels.ravel() == identity)
        second = np.flatnonzero(other_labels.ravel() == identity)
        src_a, order_a = np.unique(indices.ravel()[first], return_index=True)
        src_b, order_b = np.unique(other_indices.ravel()[second], return_index=True)
        known, at_a, at_b = np.intersect1d(src_a, src_b, assume_unique=True, return_indices=True)
        a_y, a_x = np.divmod(first[order_a[at_a]], TARGET)
        b_y, b_x = np.divmod(second[order_b[at_b]], TARGET)
        displacement = np.hypot(a_x - b_x, a_y - b_y)
        result[str(identity)] = {
            "common_source_pixels": int(known.size),
            "median_target_shift_px": float(np.median(displacement)) if known.size else None,
            "p90_target_shift_px": float(np.percentile(displacement, 90)) if known.size else None,
        }
    return result


def diagnostic_strip(output, frame, before, exact, visible, after):
    names = ("Prior attempt", "Observed EXO pixels", "After exact pixel lock")
    panel = Image.new("RGB", (TARGET * 3, TARGET + 32), "#141b27")
    draw = ImageDraw.Draw(panel)
    tiles = before, np.where(visible[..., None], exact, 0), after
    for column, (name, pixels) in enumerate(zip(names, tiles, strict=True)):
        x = column * TARGET
        draw.text((x + 10, 10), f"F{frame:02} | {name}", fill="white")
        panel.paste(Image.fromarray(pixels), (x, 32))
    panel.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exo", type=Path, required=True)
    parser.add_argument("--exo-sha256", required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--alternate-maps", type=Path)
    parser.add_argument("--diagnostic-old-generation", type=Path,
                        help="Use a prior generation strictly as a counterexample to pixel locking alone.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if file_sha256(args.exo) != args.exo_sha256:
        raise ValueError("EXO SHA-256 disagrees with the input receipt.")
    check_reference(args.exo, SOURCE_W, 25)
    maps = load_correspondences(args.maps)
    map_hash = file_sha256(args.maps)
    alternate_hash = file_sha256(args.alternate_maps) if args.alternate_maps else None
    if alternate_hash == map_hash:
        raise ValueError("An identical correspondence file is not an alternate hypothesis.")
    alternate = load_correspondences(args.alternate_maps) if args.alternate_maps else None
    indices = sorted(maps)
    if alternate is not None and set(indices) != set(alternate):
        raise ValueError("Calibration hypotheses need identical frame indices.")
    inputs = review_frames(args.exo, SOURCE_W, indices)
    candidate = None
    if args.diagnostic_old_generation:
        check_reference(args.diagnostic_old_generation, TARGET, 30)
        candidate = review_frames(args.diagnostic_old_generation, TARGET, indices)
    args.output.mkdir(parents=True, exist_ok=False)
    products = {}
    rows = []
    for index in indices:
        labels, source = maps[index]
        rgb, visible = observed_pixels(np.asarray(inputs[index]), labels, source,
                                       None if alternate is None else alternate[index])
        label_counts = object_counts(labels, visible)
        products[f"F{index:02}_rgb"] = rgb
        products[f"F{index:02}_valid"] = visible
        products[f"F{index:02}_source_index"] = np.where(visible, source, -1)
        products[f"F{index:02}_target_label"] = np.where(visible, labels, 0)
        row = {"frame": index, "target_pixels": int(visible.sum()), "by_object": label_counts,
               "primary_candidate_by_object": object_counts(labels, labels > 0)}
        if alternate is not None:
            row["common_source_with_target_displacement"] = joint_source_support(labels, source, alternate[index])
        if candidate is not None:
            before = np.asarray(candidate[index])
            after = preserve_observations(before, rgb, visible)
            row["changed_pixels_inside_lock"] = audit_pixel_lock(before, rgb, visible, after)
            row["changed_pixels_outside_lock"] = 0
            diagnostic_strip(args.output / f"F{index:02}-pixel-lock-counterexample.png",
                             index, before, rgb, visible, after)
        rows.append(row)
    data = args.output / "verified-pixels.npz"
    np.savez_compressed(data, **products)
    receipt = {
        "mode": "strict_same_source_pixel_agreement" if alternate else "single_projection_observation",
        "exo_sha256": args.exo_sha256, "maps_sha256": map_hash,
        "alternate_maps_sha256": alternate_hash,
        "data_sha256": file_sha256(data),
        "frames": rows,
        "diagnostic_old_generation_sha256": (
            file_sha256(args.diagnostic_old_generation) if candidate is not None else None
        ),
        "limits": [
            "Unknown pixels are a separate validity bit, including actual black observations.",
            "The map validates provenance and exact pixel preservation under the supplied projection hypotheses.",
            "Geometry agreement is not ground truth. Exact anchors alone make no claim about the rest of the frame.",
        ],
    }
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
