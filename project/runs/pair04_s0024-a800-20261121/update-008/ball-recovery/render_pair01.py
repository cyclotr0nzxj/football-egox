"""Use the same EXO wearer-exterior filtering as the selected Pair01 CPU preview."""

import argparse
import json
from pathlib import Path
import runpy
import sys
from zipfile import ZipFile

import numpy as np


TAKE = 'pair01_s0288'
MASK_SHA = '49f74ed9934cea6d335e41b1ddc81f0ecb70e1c692c3c17ec564a598a331695f'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bundle', required=True, type=Path)
    options, args = parser.parse_known_args()
    bundle = options.bundle.resolve(strict=True)
    prior = runpy.run_path(str(bundle / 'prior/render_pair05.py'))
    scope = prior['main'].__globals__
    plan = json.loads((bundle / 'plan.json').read_text())
    artifacts = next(row['geometry_sha256'] for row in plan if row['name'] == TAKE)

    def load_pair01_exclusions(geometry, exclusion):
        if geometry.name != TAKE:
            raise ValueError(f'Expected sealed {TAKE} source geometry.')
        for name, digest in artifacts.items():
            prior['read_verified'](geometry / name, digest)
        masks, retained = [], []
        with ZipFile(exclusion) as exclude, ZipFile(geometry / 'mask/exo.zip') as labels:
            names = prior['FRAME_NAMES']
            if any(tuple(sorted(z.namelist())) != names for z in (exclude, labels)):
                raise ValueError('Expected exactly 49 paired wearer masks and EXO labels.')
            for name in names:
                png = prior['read_mask'](exclude, name)
                instances = prior['read_mask'](labels, name)
                removed = png == 255
                keep = (instances == 2) & ~removed
                if (not removed.any() or np.any((png != 0) & ~removed)
                        or not (instances[removed] == 2).all() or not keep[::2, ::2].any()):
                    raise ValueError(f'Mask leaves the approved source wearer or removes his lower body: {name}')
                masks.append(removed)
                retained.append(int(keep.sum()))
        return masks, retained

    scope.update(TAKE=TAKE, EXCLUSION_SHA256=MASK_SHA, load_exclusions=load_pair01_exclusions)
    sys.argv = [sys.argv[0], *args]
    prior['main']()


if __name__ == '__main__':
    main()
