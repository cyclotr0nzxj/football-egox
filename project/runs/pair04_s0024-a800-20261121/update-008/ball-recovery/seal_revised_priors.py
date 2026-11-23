"""Publish the checked Pair05 and Pair01 priors in a fresh inference stage."""

import json
from pathlib import Path
import runpy
import shutil


def main():
    root = Path.cwd()
    stage = root / 'inference-update'
    archive = root / 'inference-update.initial-priors'
    draft = root / 'inference-update.resealing'
    if archive.exists() or draft.exists() or (root / 'update-egox-output').exists():
        raise FileExistsError('Inspect the existing update stage before resealing.')
    helper = runpy.run_path(str(root / 'update-bundle/stage_update_008.py'))
    helper['verify_bundle'](root / 'update-bundle')
    for line in (stage / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ')
        if name not in {'meta.json', 'validated-inputs.json'}:
            raise ValueError('Unexpected manifest entry.')
        helper['require_digest'](stage / name, sha)
    receipts = json.loads((stage / 'validated-inputs.json').read_text())
    if tuple(row['name'] for row in receipts) != helper['TAKES']:
        raise ValueError('Unexpected batch order.')
    env = root.parent / 'renderer-env/bin'
    replacements = {
        'pair05_s0024': root / 'ball-recovery/render/pair05_s0024/ego_Prior.mp4',
        'pair01_s0288': root / 'ball-recovery/pair01/pair01_s0288/ego_Prior.mp4',
    }
    for prior in replacements.values():
        helper['verify_prior'](prior, env / 'ffprobe', env / 'ffmpeg')
    shutil.copytree(stage, draft, symlinks=True)
    for row in receipts:
        take = row['name']
        if take in replacements:
            prior = replacements[take]
            link = draft / 'example/self_capture/ego_priors' / take
            if not link.is_symlink():
                raise ValueError(f'Expected an isolated prior directory link: {link}')
            link.unlink()
            link.symlink_to(prior.parent, target_is_directory=True)
            row['prior_sha256'] = helper['digest'](prior)
    helper['json_write'](draft / 'validated-inputs.json', receipts)
    (draft / 'SHA256SUMS').write_text(''.join(
        f'{helper["digest"](draft / name)}  {name}\n' for name in ('meta.json', 'validated-inputs.json')))
    stage.rename(archive)
    try:
        draft.rename(stage)
    except OSError:
        archive.rename(stage)
        raise
    print('UPDATED_PRIORS_RESEALED', flush=True)


if __name__ == '__main__':
    main()
