"""Verify a recovered Pair05 run and generate its historical same-frame comparison."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from review_pair01_ab import EGO_SIZE, EXO_WIDTH, check_reference, file_sha256, review_frames, verify_take, write_json

TAKE = "pair05_s0024"
FRAMES = (0, 12, 24, 36, 48)
HISTORICAL_HASH = "98fe4e765bb2376ca86581c5b49b3df6bdc2329d47e4fed1a68c46150a204db9"


def review(raw, receipt, output):
    expected = json.loads((HERE / "cpu-readiness.json").read_text())["inputs"]
    remote = json.loads(receipt.read_text())
    if (remote["status"], remote["arm"], remote["take"], remote["inference_exit_code"]) != (
        "verified", "B", TAKE, 0
    ) or raw.name != f"{TAKE}.mp4" or file_sha256(raw) != remote["raw"]["sha256"]:
        raise ValueError("Recovered video does not match the verified Pair05 result.")
    original = RUN / f"multiclip-prep/{TAKE}/example/self_capture/videos/{TAKE}/exo.mp4"
    prior = RUN / f"update-008/output-008/priors/{TAKE}-final.mp4"
    old = RUN / f"update-008/output-008/verified/{TAKE}_ego.mp4"
    references = (
        ("EXO", original, EXO_WIDTH, 25, expected["exo_sha256"]),
        ("Geometry prior", prior, EGO_SIZE, 30, expected["prior_sha256"]),
        ("Previous generation", old, EGO_SIZE, 30, HISTORICAL_HASH),
    )
    for _, path, width, fps, digest in references:
        if file_sha256(path) != digest:
            raise ValueError(f"Historical reference changed: {path}")
        check_reference(path, width, fps)
    output.mkdir()
    result = verify_take(TAKE, raw, output, contact_sheets=False)
    parsed = json.loads((output / result["receipt"]).read_text())
    if parsed["raw"]["sha256"] != remote["raw"]["sha256"] or parsed["raw"]["probe"]["pts_seconds"] != remote["raw"]["pts_seconds"]:
        raise ValueError("Recovered raw video timing or hashes do not match the cloud result.")
    sources = [(label, path, width) for label, path, width, _, _ in references]
    sources.append(("Aria correction", output / result["video"], EGO_SIZE))

    gap = 12
    row_height = EGO_SIZE + 24
    page_width = sum(width for _, _, width in sources) + gap * (len(sources) + 1)
    sheet = Image.new("RGB", (page_width, 55 + row_height * len(FRAMES)), "#161e27")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    x = gap
    for label, path, width in sources:
        draw.text((x, 12), label, fill="white", font=font)
        pictures = review_frames(path, width, FRAMES)
        for row, index in enumerate(FRAMES):
            top = 56 + row * row_height
            draw.text((x, top - 23), f"F{index:02d}", fill="white", font=font)
            sheet.paste(pictures[index], (x, top))
        x += width + gap
    sheet.save(output / "same-frame.jpg", quality=94)
    write_json(output / "verification.json", {
        "take": TAKE,
        "numeric_verification": "passed",
        "manual_visual_verdict": "pending",
        "historical_only": True,
        "frame_indices": list(FRAMES),
        "result": result,
        "remote_raw_sha256": remote["raw"]["sha256"],
    })
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    review(args.raw, args.receipt, args.output_dir)
