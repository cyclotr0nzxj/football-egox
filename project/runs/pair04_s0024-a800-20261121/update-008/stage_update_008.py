"""Verify and render a separate EXO-only update stage on the existing 008 instance."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess

import sys


TAKES = ('pair04_s0024', 'pair05_s0024', 'pair01_s0288')
BASE = Path('/root/autodl-tmp/football_egox/pair04-20261121')
RENDERER_SHA = '177721241c123019e7f1932639c59efb510e314514f561ef9d54ce122f6a1ee8'
INFER_SHA = '99199fac22858821d356f0ec3b8577495290dd579a8d80368de1fca68d559bed'
OLD_RECEIPT_SHA = '1cd414157e61e3d29157a12ef6baf9f32dff6689e68d80d52d8d295a1be642f3'
KEYS = {'exo_path', 'ego_prior_path', 'prompt', 'camera_intrinsics', 'camera_extrinsics',
        'ego_intrinsics', 'ego_extrinsics'}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def require_digest(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Changed staged/cache file: {path}')


def json_read(path):
    return json.loads(path.read_text())


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_bundle(bundle):
    hashes = json_read(bundle / 'bundle-hashes.json')
    expected = {'plan.json', 'stage_update_008.py', 'run_updated_egox_008.sh', 'prior/render_pair05.py',
                'pair05-wearer-upper-body.zip', *(f'poses/{take}.json' for take in TAKES)}
    if set(hashes) != expected:
        raise ValueError('Unexpected update-bundle contents.')
    for name, sha in hashes.items():
        require_digest(bundle / name, sha)
    plan = json_read(bundle / 'plan.json')
    if tuple(item['name'] for item in plan) != TAKES:
        raise ValueError('Update order does not match the approved three takes.')
    for item in plan:
        take = item['name']
        row, = json_read(bundle / 'poses' / f'{take}.json')['test_datasets']
        if set(row) != KEYS or len(row['ego_extrinsics']) != 49:
            raise ValueError(f'Unexpected metadata content for {take}.')
        if (row['exo_path'], row['ego_prior_path']) != (
            f'./example/self_capture/videos/{take}/exo.mp4',
            f'./example/self_capture/ego_priors/{take}/ego_Prior.mp4',
        ):
            raise ValueError(f'Paths for {take} differ from the official EXO-only format.')
    print('UPDATE_BUNDLE_VERIFIED', *TAKES, flush=True)
    return plan


def invoke(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def check_machine(root, renderer):
    if subprocess.check_output(['hostname'], text=True).strip() != 'autodl-container-cb894da57f-45e726a1':
        raise ValueError('Expected the original authorized 008 container.')
    invoke(['findmnt', '--mountpoint', str(Path('/root/autodl-fs').resolve())], stdout=subprocess.DEVNULL)
    invoke(['flock', '-n', str(root / '.egox-inference.lock'), 'true'])
    selected = subprocess.check_output([
        'nvidia-smi', '-i', '0', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits',
    ], text=True).strip().split(',')
    gpu, capacity, used = [part.strip() for part in selected]
    if (gpu, capacity) != ('NVIDIA A800 80GB PCIe', '81920') or int(used) > 500:
        raise ValueError('Update requires the idle A800 80GB PCIe from 008.')
    if 'ALL_MODELS_VERIFIED' not in (root / 'download-shared.log').read_text().splitlines():
        raise ValueError('Original weight verification receipt missing.')
    assets = json_read(root / 'records/model_verification.json')
    if len(assets) != 35 or any(not row['official_hash_verified'] for row in assets):
        raise ValueError('Expected 35 verified and indexed model assets.')
    for asset in assets:
        if (Path('/root/autodl-fs/football_egox') / asset['path']).stat().st_size != asset['bytes']:
            raise ValueError(f'Model asset size differs from receipt: {asset["path"]}')
    require_digest(renderer, RENDERER_SHA)
    invoke([str(BASE / 'renderer-env/bin/python'), '-B', '-c',
            'import torch, pytorch3d._C; x=torch.eye(4,device="cuda"); assert x.sum().item()==4'], timeout=30)


def validate_cache(root, plan):
    old = root / 'inference-four'
    require_digest(old / 'validated-inputs.json', OLD_RECEIPT_SHA)
    require_digest(old / 'infer.py', INFER_SHA)
    frozen = {row['name']: row for row in json_read(old / 'validated-inputs.json')}
    for item in plan:
        take = item['name']
        receipt = frozen[take]
        if item['source_exo_sha256'] != receipt['exo_sha256']:
            raise ValueError(f'Different source EXO for {take}.')
        require_digest(old / f'example/self_capture/videos/{take}/exo.mp4', receipt['exo_sha256'])
        depths = sorted((old / f'example/self_capture/depth_maps/{take}').glob('*.npy'))
        if len(depths) != 49 or len(receipt['depth_sha256']) != 49:
            raise ValueError(f'Incomplete original depth cache for {take}.')
        for path, sha in zip(depths, receipt['depth_sha256'], strict=True):
            require_digest(path, sha)
        geom = BASE / item['geometry_cache']
        for name, sha in item['geometry_sha256'].items():
            require_digest(geom / name, sha)
    print('UPDATE_CACHE_VERIFIED', flush=True)
    return frozen


def verify_prior(path, ffprobe, ffmpeg):
    data = json.loads(subprocess.check_output([
        str(ffprobe), '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_streams',
        '-show_entries', 'stream=width,height,avg_frame_rate,time_base:frame=pts', '-of', 'json', str(path),
    ]))
    stream, = data['streams']
    time_base = Fraction(stream['time_base'])
    if (stream['width'], stream['height'], Fraction(stream['avg_frame_rate'])) != (448, 448, 30):
        raise ValueError(f'Unexpected new prior size/frame rate: {path}')
    if [int(frame['pts']) * time_base for frame in data['frames']] != [Fraction(i, 30) for i in range(49)]:
        raise ValueError(f'Incorrect prior presentation timing: {path}')
    invoke([str(ffmpeg), '-v', 'error', '-nostdin', '-xerror', '-i', str(path), '-f', 'null', '-'])


def render_update(bundle, plan):
    root = BASE / 'egox-shared-008'
    old = root / 'inference-four'
    output = root / 'update-priors-008'
    stage = root / 'inference-update'
    temporary = root / 'inference-update.preparing'
    renderer_root = BASE / 'Renderer-sparse'
    renderer = renderer_root / 'scripts/render_vipe_pointcloud.py'
    env = BASE / 'renderer-env/bin'
    check_machine(root, renderer)
    frozen = validate_cache(root, plan)
    if stage.exists() or temporary.exists() or output.exists():
        raise FileExistsError('An update stage or prior directory already exists; inspect it before retrying.')
    output.mkdir()
    temporary.mkdir()
    settings = dict(os.environ, PATH=f'{env}:{os.environ["PATH"]}', PYTHONPATH=str(renderer_root),
                    CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                    PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', IMAGEIO_FFMPEG_EXE=str(env / 'ffmpeg'))
    for name in ('infer.py', 'core'):
        (temporary / name).symlink_to((old / name).resolve())
    entries, receipts = [], []
    for item in plan:
        take = item['name']
        meta = bundle / 'poses' / f'{take}.json'
        target = output / take
        geo = BASE / item['geometry_cache']
        runner = [str(renderer)]
        if take == 'pair05_s0024':
            runner = [str(bundle / 'prior/render_pair05.py'), '--renderer', str(renderer),
                      '--exclusion-zip', str(bundle / 'pair05-wearer-upper-body.zip')]
        with (output / f'{take}.log').open('x') as log:
            invoke([str(env / 'python'), '-B', *runner, '--input_dir', str(geo), '--out_dir', str(target),
                    '--meta_json_path', str(meta), '--point_size', '5.0', '--start_frame', '0', '--end_frame', '48',
                    '--fish_eye_rendering', '--use_mean_bg'], env=settings, timeout=360,
                   stdout=log, stderr=subprocess.STDOUT)
        prior = target / take / 'ego_Prior.mp4'
        verify_prior(prior, env / 'ffprobe', env / 'ffmpeg')
        invoke([str(env / 'ffmpeg'), '-v', 'error', '-nostdin', '-n', '-i', str(prior),
                '-vf', "select='eq(n,0)+eq(n,24)+eq(n,48)',tile=3x1", '-frames:v', '1',
                '-update', '1', str(output / f'{take}-preview.png')])
        for parent, origin in (('videos', old / f'example/self_capture/videos/{take}'),
                               ('depth_maps', old / f'example/self_capture/depth_maps/{take}'),
                               ('ego_priors', prior.parent)):
            link = temporary / f'example/self_capture/{parent}/{take}'
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(origin.resolve(), target_is_directory=True)
        entries.extend(json_read(meta)['test_datasets'])
        receipts.append(dict(frozen[take], prior_sha256=digest(prior)))
        print('UPDATED_PRIOR_VERIFIED', take, flush=True)
    json_write(temporary / 'meta.json', {'test_datasets': entries})
    json_write(temporary / 'validated-inputs.json', receipts)
    (temporary / 'SHA256SUMS').write_text(''.join(
        f'{digest(temporary / name)}  {name}\n' for name in ('meta.json', 'validated-inputs.json')))
    temporary.rename(stage)
    print('UPDATE_STAGE_READY', stage, 'Review the three prior contact sheets before EgoX.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify-bundle', 'render'))
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    plan = verify_bundle(bundle)
    if args.mode == 'render':
        render_update(bundle, plan)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'UPDATE_PREPARATION_FAILED {error}', file=sys.stderr)
        raise SystemExit(1) from error
