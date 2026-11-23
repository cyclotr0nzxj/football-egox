"""Materialize manually reviewed annotations for the frozen Pair03 EXO input."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parents[2]
EXO = RUN / "pair05-left-diagnosis-20261122/pair03-exo/exo.mp4"
EXO_SHA256 = "512a57b111ec1420e21def2814bdd17105d9a611f7e4e5da40f623c2e92a94b9"
FRAME_COUNT = 49
FRAME_SIZE = (784, 448)

# Independently read the 784x448 EXO contact sheets at two-frame spacing. All
# coordinates use the original staged frame, with x increasing to the right.
# Rows are wearer head, teammate head, teammate torso, planted red shoe,
# planted white shoe, approximate wearer box, approximate teammate box.
PERSON_KEYS = {
    0: ((515, 178), (451, 196), (437, 229), (531, 330), (423, 308), (501, 166, 563, 332), (413, 186, 464, 310)),
    2: ((512, 178), (451, 197), (436, 229), (533, 330), (423, 308), (498, 167, 560, 332), (414, 187, 464, 310)),
    4: ((508, 180), (451, 198), (435, 232), (533, 330), (423, 309), (495, 169, 557, 333), (412, 189, 463, 312)),
    6: ((501, 182), (450, 199), (434, 232), (530, 329), (424, 309), (490, 171, 548, 331), (413, 189, 462, 312)),
    8: ((497, 184), (449, 199), (432, 232), (520, 329), (423, 309), (482, 173, 545, 332), (409, 188, 461, 312)),
    10: ((493, 186), (448, 201), (430, 235), (520, 330), (420, 308), (478, 176, 545, 334), (409, 190, 460, 311)),
    12: ((489, 187), (446, 201), (428, 235), (518, 328), (415, 308), (465, 176, 537, 332), (404, 192, 458, 310)),
    14: ((485, 186), (446, 203), (425, 234), (510, 323), (414, 308), (460, 175, 531, 327), (403, 194, 456, 310)),
    16: ((477, 184), (438, 205), (425, 232), (502, 316), (410, 305), (456, 172, 526, 321), (399, 195, 450, 309)),
    18: ((473, 183), (435, 208), (422, 234), (483, 312), (409, 306), (450, 173, 519, 315), (397, 198, 448, 309)),
    20: ((473, 185), (432, 209), (418, 234), (473, 312), (410, 308), (447, 175, 529, 317), (399, 198, 447, 311)),
    22: ((472, 184), (435, 207), (418, 231), (472, 313), (412, 307), (443, 174, 530, 317), (401, 196, 450, 310)),
    24: ((471, 185), (438, 202), (419, 228), (473, 313), (411, 305), (444, 175, 520, 317), (399, 192, 450, 308)),
    26: ((469, 186), (434, 208), (417, 231), (471, 307), (414, 306), (445, 175, 514, 315), (401, 197, 450, 309)),
    28: ((468, 188), (434, 208), (415, 236), (501, 309), (413, 304), (450, 177, 513, 313), (398, 198, 450, 308)),
    30: ((467, 190), (431, 211), (410, 237), (500, 309), (414, 304), (453, 180, 512, 314), (396, 200, 447, 308)),
    32: ((464, 190), (431, 209), (408, 235), (499, 309), (409, 303), (451, 179, 507, 313), (389, 199, 444, 306)),
    34: ((460, 192), (430, 210), (407, 233), (465, 303), (409, 302), (447, 181, 505, 309), (389, 200, 444, 305)),
    36: ((455, 193), (427, 210), (402, 232), (466, 302), (405, 300), (438, 182, 500, 306), (383, 199, 440, 303)),
    38: ((452, 194), (424, 211), (398, 232), (466, 302), (398, 298), (435, 183, 495, 305), (378, 200, 438, 303)),
    40: ((450, 197), (417, 212), (393, 235), (466, 300), (390, 297), (434, 187, 495, 304), (373, 202, 432, 302)),
    42: ((447, 195), (414, 213), (391, 233), (468, 295), (391, 298), (440, 183, 491, 301), (371, 201, 428, 302)),
    44: ((444, 194), (409, 213), (385, 235), (474, 295), (389, 296), (433, 183, 486, 300), (368, 202, 426, 299)),
    46: ((440, 193), (407, 213), (384, 234), (474, 294), (383, 294), (430, 181, 484, 299), (365, 201, 424, 299)),
    48: ((436, 194), (406, 215), (382, 235), (471, 290), (381, 293), (427, 183, 481, 295), (364, 203, 421, 298)),
}

# A partial center is a visual estimate from the visible rim. For depth and
# color support, use the pixel on the visible part rather than an occluded center.
PARTIAL_BALL = {
    8: ((508, 312), (505, 310)),
    16: ((509, 304), (513, 302)), 17: ((509, 303), (513, 301)),
    28: ((480, 294), (482, 292)), 29: ((477, 293), (481, 293)),
    30: ((475, 293), (479, 293)), 31: ((473, 292), (477, 292)),
    40: ((460, 286), (457, 285)), 41: ((459, 286), (456, 285)),
    42: ((456, 285), (452, 285)), 43: ((455, 285), (452, 284)),
    44: ((455, 284), (452, 283)), 45: ((453, 284), (451, 282)),
}
CLEAR_BALL = {
    0: (519, 310), 1: (518, 310), 2: (518, 310), 3: (518, 310),
    4: (517, 310), 5: (515, 310), 6: (512, 310), 7: (510, 311),
    18: (509, 303), 19: (508, 302), 20: (507, 301), 21: (505, 301),
    22: (505, 300), 23: (504, 300), 24: (499, 298), 25: (494, 298),
    26: (487, 298), 27: (484, 297),
}

FIELD_NAMES = (
    "wearer_head_xy", "teammate_head_xy", "teammate_torso_xy", "wearer_ground_xy",
    "teammate_ground_xy", "wearer_box_xyxy", "teammate_box_xyxy",
)


def read_source():
    if hashlib.sha256(EXO.read_bytes()).hexdigest() != EXO_SHA256:
        raise ValueError("Frozen staged EXO hash differs.")
    capture = cv2.VideoCapture(str(EXO))
    frames = []
    try:
        while True:
            success, image = capture.read()
            if not success:
                break
            if image.shape != (FRAME_SIZE[1], FRAME_SIZE[0], 3):
                raise ValueError("Unexpected input dimensions.")
            frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} EXO frames, got {len(frames)}.")
    return frames


def person_points(frame):
    if frame in PERSON_KEYS:
        return PERSON_KEYS[frame]
    previous = PERSON_KEYS[frame - 1]
    following = PERSON_KEYS[frame + 1]
    return tuple(
        tuple(round((a + b) / 2) for a, b in zip(before, after, strict=True))
        for before, after in zip(previous, following, strict=True)
    )


def annotations():
    rows = []
    for frame in range(FRAME_COUNT):
        row = {
            "frame": frame,
            **dict(zip(FIELD_NAMES, person_points(frame), strict=True)),
            "person_provenance": "manual" if frame in PERSON_KEYS else "linear_between_adjacent_manual",
        }
        if frame in CLEAR_BALL:
            row.update(ball_xy=CLEAR_BALL[frame], ball_evidence_xy=CLEAR_BALL[frame], ball_status="clear")
        elif frame in PARTIAL_BALL:
            center, visible_pixel = PARTIAL_BALL[frame]
            row.update(ball_xy=center, ball_evidence_xy=visible_pixel, ball_status="partial")
        else:
            row.update(ball_xy=None, ball_evidence_xy=None, ball_status="occluded_or_unconfirmed")
        # Radius is an approximate support window from the apparent size of
        # clearly visible examples. It is not a fitted boundary.
        row["ball_radius_px_estimate"] = None if row["ball_xy"] is None else (8 if frame <= 24 else 7)
        rows.append(row)
    return {
        "name": "pair03",
        "width": FRAME_SIZE[0],
        "height": FRAME_SIZE[1],
        "video_sha256": EXO_SHA256,
        "source": "Pair03 cropped external R8 view, first 49 visible frames",
        "wearer_identity": "red shoes, dark helmet",
        "teammate_identity": "white shoes",
        "frames": rows,
    }


def save_review(frames, rows, output):
    selected = (0, 8, 12, 18, 24, 28, 36, 42, 48)
    crop = (352, 149, 574, 342)
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    contact = Image.new("RGB", (3 * width, 3 * height))
    for j, index in enumerate(selected):
        canvas = frames[index].copy()
        row = rows[index]
        draw = ImageDraw.Draw(canvas)
        for name, color in (("wearer", "#ff836b"), ("teammate", "#70e7d0")):
            draw.rectangle(row[name + "_box_xyxy"], outline=color)
            x, y = row[name + "_head_xy"]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=1)
            x, y = row[name + "_ground_xy"]
            draw.line((x - 3, y, x + 3, y), fill=color)
            draw.line((x, y - 3, x, y + 3), fill=color)
        if row["ball_xy"] is not None:
            x, y = row["ball_xy"]
            ball_color = "#ffe56a" if row["ball_status"] == "clear" else "#ffa739"
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=ball_color, width=1)
        draw.text((crop[0] + 4, crop[1] + 3), f"F{index:02d}  ball: {row['ball_status']}", fill="white", stroke_width=1)
        contact.paste(canvas.crop(crop), (j % 3 * width, j // 3 * height))
    contact.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    frames = read_source()
    result = annotations()
    for row in result["frames"]:
        for name in ("wearer", "teammate"):
            point = row[name + "_head_xy"]
            box = row[name + "_box_xyxy"]
            assert box[0] < point[0] < box[2] and box[1] < point[1] < box[3]
    arguments.output.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=arguments.output, delete=False) as staged:
        staged.write(json.dumps(result, indent=2) + "\n")
    Path(staged.name).replace(arguments.output / "pair03-exo-anchors.json")
    save_review(frames, result["frames"], arguments.output / "pair03-exo-anchors-review.jpg")


if __name__ == "__main__":
    main()
