"""Build an isolated Pair04 retry with EXO-only inverse-depth calibration."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from check_overlap_geometry import MARKER, assess, cv2, unpack
from stage_geometry import prepare


MIDDLE = 'pair04_s0052'
LAST = 'pair04_s0076'
EXPECTED_COEFFICIENTS = (1.4749212901340099, -0.08032681510204931)


def check(result):
    heldout = result['heldout_frames']
    if not result['passed_provisional_gate'] or not all(
        frame['median_relative_error'] <= .20 and frame['p90_relative_error'] <= .40
        for frame in heldout
    ):
        raise ValueError('Original aggregate or per-frame overlap gate failed')


def export(source, target):
    depth = [source[name] / 'depth/exo.zip' for name in (MIDDLE, LAST)]
    mask = [source[name] / 'mask/exo.zip' for name in (MIDDLE, LAST)]
    with zipfile.ZipFile(depth[0]) as old_depth, zipfile.ZipFile(depth[1]) as new_depth, \
         zipfile.ZipFile(mask[0]) as old_mask, zipfile.ZipFile(mask[1]) as new_mask:
        design, response = [], []
        for i in range(0, 25, 2):
            old = unpack(old_depth, f'{i+24:05}.exr', True)[::4, ::4]
            new = unpack(new_depth, f'{i:05}.exr', True)[::4, ::4]
            a = unpack(old_mask, f'{i+24:05}.png', False)[::4, ::4]
            b = unpack(new_mask, f'{i:05}.png', False)[::4, ::4]
            valid = (a == 0) & (b == 0) & (old > .5) & (old < 80) & (new > .5) & (new < 80)
            if valid.sum() < 1000:
                raise ValueError('Too few fit pixels')
            design.append(1/new[valid])
            response.append(1/old[valid])
        values = np.concatenate(design)
        slope, offset = np.linalg.lstsq(
            np.column_stack((values, np.ones_like(values))), np.concatenate(response), rcond=None
        )[0]
        if not np.allclose([slope, offset], EXPECTED_COEFFICIENTS, rtol=1e-5, atol=1e-6):
            raise ValueError(f'Remote overlap differs from the reviewed fit: {slope}, {offset}')
        fitted_low, fitted_high = float(1/values.max()), float(1/values.min())
        least, greatest = float('inf'), 0.0
        replacement = b'Y' + MARKER[1:]
        with zipfile.ZipFile(target, 'x') as output:
            for i in range(49):
                name = f'{i:05}.exr'
                raw = unpack(new_depth, name, True)
                least = min(least, float(raw.min()))
                greatest = max(greatest, float(raw.max()))
                if least < .98*fitted_low or greatest > 1.02*fitted_high:
                    raise ValueError('Depth exceeds the reviewed near-fit support')
                inverse = slope/raw + offset
                if not np.isfinite(inverse).all() or (inverse <= 0).any():
                    raise ValueError('Non-positive or non-finite inverse depth')
                value = np.asarray(1/inverse, dtype=np.float32)
                ok, encoded = cv2.imencode('.exr', value, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                content = encoded.tobytes()
                if not ok or content.count(replacement) != 1:
                    raise ValueError('Expected single-channel half EXR')
                output.writestr(name, content.replace(replacement, MARKER, 1))
        return {'fit': [float(slope), float(offset)], 'fit_frame_step': 2, 'fit_source_frames': [76, 100],
                'fit_depth_bounds': [fitted_low, fitted_high], 'native_depth_bounds': [least, greatest]}


def foreground(middle, last):
    result = []
    paths = [middle/'depth/exo.zip', last/'depth/exo.zip', middle/'mask/exo.zip', last/'mask/exo.zip']
    with zipfile.ZipFile(paths[0]) as a, zipfile.ZipFile(paths[1]) as b, \
         zipfile.ZipFile(paths[2]) as ma, zipfile.ZipFile(paths[3]) as mb:
        for i in range(1, 25, 2):
            old = unpack(a, f'{i+24:05}.exr', True)
            new = unpack(b, f'{i:05}.exr', True)
            valid = (unpack(ma, f'{i+24:05}.png', False) > 0) & (unpack(mb, f'{i:05}.png', False) > 0)
            if valid.sum() < 1000:
                raise ValueError('Too few shared foreground pixels')
            error = np.abs(new[valid]/old[valid]-1)
            med, p90 = float(np.median(error)), float(np.percentile(error, 90))
            if med > .05 or p90 > .15:
                raise ValueError('Independent foreground depth check rejected calibration')
            result.append({'source_frame':76+i,'matched_foreground_pixels':int(valid.sum()),'median':med,'p90':p90})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Original fixed-K run, or verified diagnostic archive extraction')
    parser.add_argument('--inputs', type=Path, required=True,
                        help='Original run with EXO windows, head track and legacy frame')
    parser.add_argument('--legacy', type=Path, required=True,
                        help='Frozen Pair04 initial ViPE reference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir()
    (out/'legacy').mkdir()
    (out/'legacy/Pair04').symlink_to(args.legacy.resolve(), target_is_directory=True)
    source = {name:args.source/'windows'/name/'geometry-fixed'/name for name in (MIDDLE,LAST)}
    geometries = {}
    for name in (MIDDLE, LAST):
        window = out/'windows'/name
        geometry = window/'geometry-fixed'/name
        original = source[name]
        window.mkdir(parents=True)
        for relative in ('exo.mp4', 'exo_head_anchors.json', 'meta-candidate.json'):
            shutil.copy2(args.inputs/'windows'/name/relative, window/relative)
        if name == MIDDLE:
            shutil.copytree(original, geometry)
        else:
            shutil.copytree(original, geometry, ignore=lambda folder, names: ['exo.zip'] if Path(folder) == original/'depth' else [])
        geometries[name] = geometry
    middle, last = geometries[MIDDLE], geometries[LAST]
    a = assess(args.legacy, middle, 24, 52)
    check(a)
    receipt = export(source, last/'depth/exo.zip')
    b = assess(middle, last, 52, 76)
    check(b)
    receipt['heldout_foreground'] = foreground(middle, last)
    receipt['derived_depth_sha256'] = hashlib.sha256((last/'depth/exo.zip').read_bytes()).hexdigest()
    receipt['native_depth_sha256'] = hashlib.sha256((source[LAST]/'depth/exo.zip').read_bytes()).hexdigest()
    receipt['geometry_units'] = 'Last-window depth transformed into the native middle-window gauge'
    receipt['overlap_median'] = b['heldout_median_relative_error']
    for name, verdict, scale in ((MIDDLE,a,a['scale_to_old']), (LAST,b,a['scale_to_old']*b['scale_to_old'])):
        window = out/'windows'/name
        report = prepare(geometries[name], window/'meta-candidate.json', window/'exo_head_anchors.json',
                         scale, window/'head-retry-validation')
        receipt[name] = {k:report[k] for k in ('head_supported_frames','head_relative_difference_median','head_relative_difference_p90')}
        (window/'overlap-fixed.json').write_text(json.dumps(verdict, indent=2, allow_nan=False)+'\n')
    (out/'pair04-inverse-depth-receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('PAIR04_INVERSE_DEPTH_VERIFIED',json.dumps({k:v for k,v in receipt.items() if k!='heldout_foreground'}))


if __name__ == '__main__':
    main()
