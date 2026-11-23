#!/usr/bin/env python3
"""Run Week 08 image processing, fixed-ROI segmentation and manual-frame display."""
import argparse
import csv
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FONT = ImageFont.truetype('/System/Library/Fonts/Monaco.ttf', 17)


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv.imwrite(str(path), image)


def rgb(image):
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def labeled_row(images, labels, width=500, height=290):
    result = Image.new('RGB', (width * len(images), height + 32), (18, 24, 31))
    draw = ImageDraw.Draw(result)
    for i, (im, label) in enumerate(zip(images, labels)):
        scaled = ImageOps.contain(rgb(im), (width, height))
        result.paste(scaled, (i * width + (width - scaled.width) // 2,
                              32 + (height - scaled.height) // 2))
        draw.text((i * width + 8, 7), label, font=FONT, fill='white')
    return result


def scores(pred, ref):
    p, g = pred > 0, ref > 0
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    return dict(tp=tp, fp=fp, fn=fn, iou=tp / max(tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                predicted_pixels=int(p.sum()), reference_pixels=int(g.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='records/course_validation_20261019/segmentation_config_and_annotations.json')
    parser.add_argument('--output', default='output/week08/results/image_processing')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    frames = ROOT / 'output/week08/frames/scene04'
    x1, y1, x2, y2 = config['roi_xyxy']
    results = []
    montage_rows = []
    for n in config['evaluation_frames']:
        image = cv.imread(str(frames / 'ego' / f'{n:04d}.jpg'))
        assert image.shape == (540, 960, 3)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        raw = np.zeros(image.shape[:2], dtype=np.uint8)
        for rule in config['hsv_rules']:
            raw |= cv.inRange(hsv, np.array(rule['lower']), np.array(rule['upper']))
        roi_mask = np.zeros_like(raw)
        roi_mask[y1:y2, x1:x2] = 255
        raw &= roi_mask
        clean = cv.morphologyEx(raw, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        clean = cv.morphologyEx(clean, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv.findContours(clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            hull = cv.convexHull(contour)
            area = cv.contourArea(hull)
            peri = cv.arcLength(hull, True)
            circularity = 4 * np.pi * area / max(peri * peri, 1)
            _, _, w, h = cv.boundingRect(hull)
            if (config['component_area'][0] <= area <= config['component_area'][1]
                    and circularity >= config['minimum_circularity']
                    and config['aspect_ratio'][0] <= w / h <= config['aspect_ratio'][1]):
                candidates.append((area * circularity ** 2, hull))
        pred = np.zeros_like(raw)
        if candidates:
            cv.fillPoly(pred, [max(candidates, key=lambda x: x[0])[1]], 255)
        reference = np.zeros_like(raw)
        poly = np.array(config['reference_polygons'][str(n)], dtype=np.int32)
        cv.fillPoly(reference, [poly], 255)
        before, after = scores(raw, reference), scores(pred, reference)
        result = dict(frame=n, source_pts=n / 25, raw_hsv=before, refined=after,
                      reference_kind='manual_coarse_polygon', candidates=len(candidates))
        results.append(result)
        folder = output / f'frame_{n:04d}'
        for name, data in [('input.png', image), ('hsv_candidates.png', raw),
                           ('refined_mask.png', pred), ('manual_reference.png', reference)]:
            write_image(folder / name, data)
        annotated = image.copy()
        cv.rectangle(annotated, (x1, y1), (x2, y2), (255, 150, 30), 2)
        cv.polylines(annotated, [poly], True, (0, 255, 255), 2)
        overlay = image.copy()
        selected = pred > 0
        overlay[selected] = (.5 * overlay[selected] + .5 * np.array([30, 70, 255])).astype(np.uint8)
        cv.polylines(overlay, [poly], True, (0, 255, 255), 2)
        write_image(folder / 'overlay.png', overlay)
        if n in [10, 170, 280]:
            montage_rows.append(labeled_row(
                [annotated[y1:y2, x1:x2], raw[y1:y2, x1:x2], overlay[y1:y2, x1:x2]],
                [f'F{n}: manual coarse outline', 'HSV candidates', f'Refined IoU={after["iou"]:.3f}']))
        if n == 130:
            write_image(output / 'failure_outside_roi.png', annotated)
    montage = Image.new('RGB', (1500, len(montage_rows) * 322), 'white')
    for i, row in enumerate(montage_rows):
        montage.paste(row, (0, i * 322))
    montage.save(output / 'segmentation_overview.png')
    pilot = cv.imread(str(frames / 'ego' / '0000.jpg'))
    gray = cv.cvtColor(pilot, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv.Canny(blurred, 80, 160)
    for name, data in [('pilot_rgb.png', pilot), ('gray.png', gray),
                       ('gaussian_blur.png', blurred), ('canny_edges.png', edges)]:
        write_image(output / name, data)
    crop = np.s_[250:520, 300:780]
    labeled_row([pilot[crop], gray[crop], blurred[crop], edges[crop]],
                ['RGB input', 'Grayscale', 'Gaussian 5x5, sigma=1.2', 'Canny 80/160'],
                width=400, height=225).save(output / 'processing_overview.png')
    summary = dict(opencv_version=cv.__version__, numpy_version=np.__version__,
                   working_resolution=[960, 540], parameters=config,
                   evaluation_sample_count=len(results),
                   mean_iou_raw=float(np.mean([r['raw_hsv']['iou'] for r in results])),
                   mean_iou_refined=float(np.mean([r['refined']['iou'] for r in results])),
                   mean_iou_roi_contains_ball=float(np.mean([r['refined']['iou'] for r in results if r['frame'] != 130])),
                   covered_frames=sum(r['refined']['iou'] > .5 for r in results),
                   coverage_threshold_iou=.5,
                   processing=dict(gray_mean=float(gray.mean()), gray_std=float(gray.std()),
                                   gaussian_std=float(blurred.std()),
                                   edge_pixels=int((edges > 0).sum()),
                                   edge_fraction=float((edges > 0).mean())), frames=results,
                   limitations=['Single-scene, six purposefully chosen frames with one approximate annotator.',
                                'Manual polygons are coarse references, not a fine pixel ground-truth benchmark.',
                                'F130 is an intentional out-of-ROI challenge and is included in the all-frame mean.',
                                'No claim of trained semantic segmentation or automatic touch detection.'])
    (output / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'seconds', 'raw_iou', 'refined_iou', 'precision', 'recall'])
        for row in results:
            writer.writerow([row['frame'], row['source_pts'], row['raw_hsv']['iou'],
                             row['refined']['iou'], row['refined']['precision'], row['refined']['recall']])
    contact_dir = output.parent / 'keyframes'
    contact_dir.mkdir(exist_ok=True)
    rows = []
    for event in config['manual_contact_events']:
        n = event['frame']
        exo = cv.imread(str(frames / 'exo' / f'{n:04d}.jpg'))
        ego = cv.imread(str(frames / 'ego' / f'{n:04d}.jpg'))
        write_image(contact_dir / f'exo_{n:04d}.png', exo)
        write_image(contact_dir / f'ego_{n:04d}.png', ego)
        rows.append(labeled_row([exo[130:380, 320:740], ego[200:540, 250:750]],
                                [f'EXO F{n}, t={n/25:.2f}s', f'EGO F{n}, t={n/25:.2f}s'],
                                width=500, height=340))
    for i in range(0, len(rows), 2):
        pair = Image.new('RGB', (1000, 744), 'white')
        for j, row in enumerate(rows[i:i + 2]):
            pair.paste(row, (0, j * 372))
        pair.save(contact_dir / f'manual_contacts_{i//2 + 1}.png')
    (contact_dir / 'manual_annotations.json').write_text(json.dumps(
        {'annotation_method': config['event_annotation_note'], 'events': config['manual_contact_events']},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['evaluation_sample_count', 'mean_iou_raw',
                                            'mean_iou_refined', 'mean_iou_roi_contains_ball',
                                            'covered_frames']}, indent=2))


if __name__ == '__main__':
    main()
