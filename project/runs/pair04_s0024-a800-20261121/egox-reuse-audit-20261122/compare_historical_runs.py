"""Decode selected source indices from the saved EXO, priors and EgoX renders."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from review_pair01_ab import check_reference, file_sha256, review_frames  # noqa: E402


PAIR01 = "pair01_s0288"
PAIR05 = "pair05_s0024"
ORIGINAL = "four-egox-outputs-008/verified"
UPDATE = "update-008/output-008"
AB = "pair01-gga-ab"
CLONE = "clone-004-20261122"
PAIRS = {
    PAIR01: (
        (0, 24, 36, 43),
        [
            ("R8 原片", f"multiclip-prep/{PAIR01}/example/self_capture/videos/{PAIR01}/exo.mp4"),
            ("首版正式先验", f"multiclip-evidence/{PAIR01}/teammate-prior/{PAIR01}/ego_Prior.mp4"),
            ("首版成片", f"{ORIGINAL}/{PAIR01}_ego.mp4"),
            ("低头正式先验", f"{UPDATE}/priors/{PAIR01}-final.mp4"),
            ("低头更新成片", f"{UPDATE}/verified/{PAIR01}_ego.mp4"),
            ("A 原投影成片", f"{AB}/output-a-008-20261122/verified/A_ego.mp4"),
            ("B 鱼眼修正成片", f"{AB}/output-b-008-20261122/verified/B_ego.mp4"),
            ("004 新正式先验", f"{CLONE}/pair01/prior/{PAIR01}/ego_Prior.mp4"),
        ],
    ),
    PAIR05: (
        (12, 24, 36, 48),
        [
            ("R8 原片", f"multiclip-prep/{PAIR05}/example/self_capture/videos/{PAIR05}/exo.mp4"),
            ("首版正式先验", f"multiclip-evidence/{PAIR05}/teammate-prior/{PAIR05}/ego_Prior.mp4"),
            ("首版成片", f"{ORIGINAL}/{PAIR05}_ego.mp4"),
            ("跟球正式先验", f"{UPDATE}/priors/{PAIR05}-final.mp4"),
            ("跟球更新成片", f"{UPDATE}/verified/{PAIR05}_ego.mp4"),
            ("鱼眼修正成片", f"pair05-aria-008/output-008-20261122/verified/{PAIR05}_ego.mp4"),
            ("004 前景 GGA", f"{CLONE}/pair05/verified/{PAIR05}_ego.mp4"),
        ],
    ),
}


def build(take, indices, rows, destination):
    label_width, width, row_height, header = 156, 310, 310, 46
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 17)
    sheet = Image.new("RGB", (label_width + width * len(indices), header + row_height * len(rows)), "#151D2A")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 14), take[:6].upper(), font=font, fill="#FFFFFF")
    for column, frame in enumerate(indices):
        draw.text((label_width + width * column + 8, 14), f"F{frame:02}", font=font, fill="#FFFFFF")
    receipt = []
    for row, (name, relative) in enumerate(rows):
        path = RUN / relative
        is_exo = name == "R8 原片"
        source_width, fps = (784, 25) if is_exo else (448, 30)
        check_reference(path, source_width, fps)
        frames = review_frames(path, source_width, indices)
        top = header + row * row_height
        draw.text((12, top + row_height // 2), name, font=font, fill="#FFFFFF")
        for column, frame in enumerate(indices):
            tile = ImageOps.contain(frames[frame], (width - 4, row_height - 4), Image.Resampling.LANCZOS)
            sheet.paste(tile, (label_width + width * column + 2, top + (row_height - tile.height) // 2))
        receipt.append({"label": name, "path": relative, "fps": fps, "sha256": file_sha256(path)})
    sheet.save(destination / f"{take[:6]}-same-index.jpg", quality=94, subsampling=0)
    return {"frame_indices": indices, "sources": receipt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    results = {take: build(take, indices, rows, args.output) for take, (indices, rows) in PAIRS.items()}
    (args.output / "source-receipt.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
