#!/usr/bin/env python3
"""Rebuild the three Week08 paper figures from immutable measured outputs.

No detector ranking, new samples, model training or inferential statistics are
introduced. Saved coordinate JSON is authoritative for all displayed centers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2 as cv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GRAY = "#2878A0", "#CA762F", "#747D85"
COLORS = {"harris": BLUE, "sift": ORANGE}
METHODS = ["harris", "sift"]
CONDITIONS = ["identity", "rotation15", "rotation45", "scale075", "scale125", "gaussian"]
CONDITION_LABELS = ["Identity", "Rotate\n15°", "Rotate\n45°", "Scale\n0.75×", "Scale\n1.25×", "Gaussian\nblur"]
IMAGE_IDS = ["exo_f0050", "exo_f0200", "ego_f0050", "ego_f0200"]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_path(frames, image_id):
    view, frame = image_id.split("_f")
    return frames / view / (frame + ".jpg")


def read_rgb(path):
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    assert image.shape == (540, 960, 3)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def label(ax, value, x_offset=0):
    ax.annotate(value, xy=(0, 1), xycoords="axes fraction", xytext=(x_offset, 10),
                textcoords="offset points", fontsize=9, fontweight="bold", va="bottom")


def hide_image_axes(ax):
    ax.set_axis_off()


def save(fig, output, name, require_matplotlib_panel_alignment):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=output / f"{name}.alignment.json",
          overlay_svg=output / f"{name}.alignment.svg", tolerance_pt=1.5,
          gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    # Fixed physical canvas; no tight bounding box changes the declared width.
    fig.savefig(output / f"{name}.svg")
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=300)
    plt.close(fig)


def plate(results, frames, output, align):
    fig, axes = plt.subplots(2, 4, figsize=(180 / 25.4, 86 / 25.4))
    fig.subplots_adjust(left=.025, right=.985, bottom=.15, top=.89, hspace=.56, wspace=.08)
    entries = []
    for row, image_id in enumerate(["exo_f0050", "ego_f0050"]):
        rgb = read_rgb(image_path(frames, image_id))
        payload = json.loads((results / "coordinates" / image_id / "identity.json").read_text())
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        response = cv.cornerHarris(gray.astype(np.float32) / 255., 2, 3, .04)
        mask = np.zeros(gray.shape, bool)
        mask[16:-16, 16:-16] = True
        maximum = max(float(response[mask].max()), 1e-20)
        intensity = np.log1p(1000 * np.clip(response / maximum, 0, 1)) / np.log1p(1000)
        intensity[~mask] = 0
        for col, ax in enumerate(axes[row]):
            label(ax, chr(97 + row * 4 + col))
            hide_image_axes(ax)
            if col == 1:
                heat = ax.imshow(intensity, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(rgb, interpolation="nearest")
            if col in [2, 3]:
                method = METHODS[col - 2]
                xy = np.array(payload["methods"][method]["source_retained_positions"])
                ax.scatter(xy[:, 0], xy[:, 1], s=4, facecolors="none", edgecolors=COLORS[method], linewidths=.43)
                ax.set_title(f"{method.capitalize() if method == 'harris' else 'SIFT'}  N={len(xy)}", fontsize=8, pad=19)
                entries.append({"panel": chr(97 + row * 4 + col), "image_id": image_id, "method": method, "displayed_center_count": len(xy)})
            elif col == 0:
                ax.set_title("R8 / EXO" if row == 0 else "Nano / EGO", fontsize=8, pad=19)
            else:
                ax.set_title("Harris response", fontsize=8, pad=19)
    fig.text(.025, .075, "F50 (2.00 s)   |   Full 960 × 540 inputs; no image crop or colour adjustment", fontsize=7.5)
    fig.text(.025, .025, "Response: log-positive, normalized per image (0 to 1); location overlays use saved coordinates", fontsize=7)
    save(fig, output, "figure1_feature_plate", align)
    write_json(output / "figure1_source_data.json", {"panels": entries, "response_display_only": "log1p(1000*clip(C/max(C_valid),0,1))/log1p(1000); magma palette; 16px invalid center margin", "response_display_recomputed": True})


def stability(rows, output, align):
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 110 / 25.4))
    fig.subplots_adjust(left=.12, right=.985, bottom=.20, top=.82, hspace=.65)
    points = []
    for ai, (ax, metric) in enumerate(zip(axes, ["repeatability", "source_retention"])):
        label(ax, chr(97 + ai), -12)
        ax.set_title("Location repeatability: r = M / min(Ns, Nt)" if ai == 0 else "Source retention: cs = M / Ns", loc="left", fontsize=8.5, pad=19)
        for ci, condition in enumerate(CONDITIONS):
            for mi, method in enumerate(METHODS):
                selected = [r for r in rows if r["condition"] == condition and r["method"] == method]
                selected.sort(key=lambda r: IMAGE_IDS.index(r["image_id"]))
                assert len(selected) == 4
                values = [float(r["repeatability"]) if metric == "repeatability" else int(r["one_to_one_match_count"]) / int(r["common_source_count"]) for r in selected]
                offset = -.17 if mi == 0 else .17
                jitter = np.array([-.065, -.022, .022, .065])
                ax.scatter(ci + offset + jitter, values, s=14, facecolors="none", edgecolors=COLORS[method], linewidths=.7, zorder=3)
                mean = float(np.mean(values))
                ax.plot([ci + offset - .09, ci + offset + .09], [mean, mean], color=COLORS[method], lw=2.2, zorder=4, solid_capstyle="butt")
                for r, value in zip(selected, values):
                    points.append({"metric": metric, "image_id": r["image_id"], "condition": condition, "method": method, "value": value, "macro_mean": mean})
        ax.set_ylim(-.03, 1.06)
        ax.set_xlim(-.5, 5.5)
        ax.set_xticks(range(6), CONDITION_LABELS, fontsize=7.5)
        ax.set_yticks([0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#E3E7EB", linewidth=.5, zorder=0)
        ax.tick_params(length=2.5, width=.6)
    handles = [Line2D([], [], color=COLORS[m], lw=2, label="Harris" if m == "harris" else "SIFT") for m in METHODS]
    handles += [Line2D([], [], marker="o", markersize=4, mfc="none", mec=GRAY, ls="none", label="One input image")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=3, fontsize=8, columnspacing=1.8, handlelength=1.8)
    fig.text(.12, .075, "Dots: 4 input images (2 views × 2 moments from one recording); bars: unweighted image mean", fontsize=7)
    fig.text(.12, .025, "Descriptive observations only; no independent-session replication or confidence interval", fontsize=7)
    save(fig, output, "figure2_stability", align)
    with (output / "figure2_source_data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def case(results, frames, output, align):
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 122 / 25.4), gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.10, right=.985, bottom=.15, top=.885, hspace=.55, wspace=.30)
    payload = json.loads((results / "coordinates/ego_f0200/rotation45.json").read_text())
    rgb = read_rgb(image_path(frames, "ego_f0200"))
    # The source-to-target transform is the recorded experimental transform.
    target = cv.warpPerspective(rgb, np.array(payload["H_source_to_target"]), (960, 540), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    summaries = {}
    for mi, method in enumerate(METHODS):
        ax = axes[0, mi]
        label(ax, chr(97 + mi), -12)
        ax.imshow(target, interpolation="nearest")
        d = payload["methods"][method]
        src = np.array(d["projected_common_source_xy_target_pixels"])
        matches = np.array(d["matching"])
        tgt = np.array(d["target_retained_positions"])[d["common_target_retained_indices"], :2]
        ax.scatter(src[:, 0], src[:, 1], s=4, marker="o", facecolors="none", edgecolors="#C5C9CC", linewidths=.35)
        matched = tgt[matches[:, 1].astype(int)]
        ax.scatter(matched[:, 0], matched[:, 1], s=5, facecolors="none", edgecolors=COLORS[method], linewidths=.65)
        hide_image_axes(ax)
        ax.set_title("Harris" if mi == 0 else "SIFT", loc="left", fontsize=8.5, color=COLORS[method], pad=19)
        s = d["summary"]
        ns, nt, count = [int(s[k]) for k in ["common_source_count", "common_target_count", "one_to_one_match_count"]]
        summaries[method] = {"Ns": ns, "Nt": nt, "M": count, "r": count / min(ns, nt), "cs": count / ns}
    for i, ax in enumerate(axes[1]):
        label(ax, chr(99 + i), -12)
        ax.tick_params(length=2.5, width=.6)
    ax = axes[1, 0]
    ax.set_title("Common-visible sets and matches", loc="left", fontsize=8.5, pad=19)
    for mi, method in enumerate(METHODS):
        values = [summaries[method][key] for key in ["Ns", "Nt", "M"]]
        x = np.arange(3) + (-.16 if mi == 0 else .16)
        ax.bar(x, values, width=.27, color=COLORS[method], zorder=2)
        for xx, yy in zip(x, values):
            ax.text(xx, yy + 8, str(yy), ha="center", va="bottom", fontsize=7.5, color=COLORS[method])
    ax.set_xticks(range(3), ["Source Ns", "Target Nt", "Matches M"], fontsize=7.5)
    ax.set_ylim(0, 450)
    ax.set_ylabel("Positions", fontsize=8)
    ax = axes[1, 1]
    ax.set_title("Two denominators, different conclusions", loc="left", fontsize=8.5, pad=19)
    for mi, method in enumerate(METHODS):
        values = [summaries[method][key] for key in ["r", "cs"]]
        x = np.arange(2) + (-.16 if mi == 0 else .16)
        ax.bar(x, values, width=.27, color=COLORS[method], zorder=2)
        for xx, yy in zip(x, values):
            ax.text(xx, yy + .035, f"{yy:.1%}", ha="center", va="bottom", fontsize=7.5, color=COLORS[method])
    ax.set_xticks(range(2), ["Repeatability r", "Source retention cs"], fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, .5, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.text(.10, .075, "EGO F200, 45° rotation   |   Grey: projected source centers; colour: matched target centers", fontsize=7.3)
    fig.text(.10, .025, "Source Ns / target Nt: common-visible sets; M: maximum one-to-one matches within 3 target pixels", fontsize=7)
    save(fig, output, "figure3_retention_case", align)
    write_json(output / "figure3_source_data.json", {"image_id": "ego_f0200", "condition": "rotation45", "methods": summaries, "coordinates": "results/coordinates/ego_f0200/rotation45.json"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "output/week08_harris_sift/results")
    p.add_argument("--frames-root", type=Path, default=ROOT / "output/week08/frames/scene04")
    p.add_argument("--output", type=Path, default=ROOT / "output/week08_harris_sift/paper_figures")
    p.add_argument("--qa-tools", type=Path, help="Directory containing the supplied audit_panel_alignment.py")
    p.add_argument("--figures", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3], help="Default rebuilds all three figures; selected rendering does not subset numeric data")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    qa = args.qa_tools or args.output / "qa_tools"
    if not (qa / "audit_panel_alignment.py").is_file():
        raise FileNotFoundError("Supply --qa-tools with the included figure alignment helper")
    sys.path.insert(0, str(qa.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                         "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .65,
                         "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white", "axes.labelcolor": "#26313B", "text.color": "#26313B"})
    with (args.results / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 48
    assert len({(r["image_id"], r["condition"], r["method"]) for r in rows}) == 48
    # Data contract precedes all plotting, and records every exclusion.
    write_json(args.output / "figure_contract.json", {
        "backend": "Python/matplotlib", "matplotlib_version": mpl.__version__, "opencv_version": cv.__version__, "numpy_version": np.__version__,
        "target": "Chinese course research article; Nature-informed graphics, not a Nature submission",
        "core_claim": "High minimum-denominator location repeatability can coexist with substantial loss of source locations.",
        "figure1": {"archetype": "image plate", "question": "Where are the retained centers in two synchronized F50 views?", "scope": "2 representative F50 images, chosen before this revision; F200 results remain fully represented in quantitative Figure2", "size_mm": [180, 86]},
        "figure2": {"archetype": "quantitative grid", "question": "How do repeatability and source retention respond under the same controlled conditions?", "input_rows": 48, "excluded_rows": 0, "n": "4 images, 2 views x 2 moments from one recording", "center": "unweighted macro mean across images", "spread": "all four observations shown; no inferential interval", "test": "none", "size_mm": [180, 110]},
        "figure3": {"archetype": "image plate + quant", "question": "Why can the Harris rotation45 case have high r but low source retention?", "selection": "existing diagnosed EGO F200 rotation45 case; not a representative population estimate", "size_mm": [180, 122]},
        "image_integrity": {"source_photos": "four saved 960x540 JPEG inputs", "crop": "none", "color_adjustment": "none", "response_only": "recomputed Harris response for display, per-image log-positive normalization, magma pseudocolour", "rotation_case": "recorded H applied in memory to original JPEG; no reloading lossy transformed JPEG", "point_overlay": "all saved centers relevant to the specified panel; no aesthetic subsampling"},
        "input_sha256": {"metrics.csv": sha(args.results / "metrics.csv"), **{i: sha(image_path(args.frames_root, i)) for i in IMAGE_IDS}},
        "exports": ["SVG with editable text", "PDF with embedded TrueType text", "PNG 300dpi"], "minimum_font_pt": 7,
        "limitations": ["Four related images do not establish generalization", "Position matching is not descriptor matching or tracking", "No statistical tests, seeds, confidence intervals, new independent sessions, or new model results"]})
    if 1 in args.figures:
        plate(args.results, args.frames_root, args.output, require_matplotlib_panel_alignment)
    if 2 in args.figures:
        stability(rows, args.output, require_matplotlib_panel_alignment)
    if 3 in args.figures:
        case(args.results, args.frames_root, args.output, require_matplotlib_panel_alignment)
    print(json.dumps({"status": "rendered", "output": str(args.output), "figures": args.figures, "matplotlib": mpl.__version__}))


if __name__ == "__main__":
    main()
