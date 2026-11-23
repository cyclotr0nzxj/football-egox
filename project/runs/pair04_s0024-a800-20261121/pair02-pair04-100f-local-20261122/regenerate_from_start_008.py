"""Stage six EXO-only windows and generate every frame of both new drafts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from continue_008 import BASE, BIN, PYTHON, ROOT, STOCK, WEIGHTS, adapter_ok, call, digest, env
from verify_egox_outputs_008 import verify_take


GEOMETRY = BASE / 'pair02-pair04-100f-geometry-prep'
CALIBRATED = BASE / 'pair04-inverse-depth-20261122'
WINDOWS = {
    'Pair02': ('pair02_s0000', 'pair02_s0028', 'pair02_s0052'),
    'Pair04': ('pair04_s0024', 'pair04_s0052', 'pair04_s0076'),
}
STOCK_PINS = {
    'meta.json': 'bf649f277c5987098a9815e17cb7cc8bf1a3fde1632d78af7d0451aaa343f88c',
    'validated-inputs.json': '1cd414157e61e3d29157a12ef6baf9f32dff6689e68d80d52d8d295a1be642f3',
}
META_KEYS = {'exo_path', 'ego_prior_path', 'prompt', 'camera_intrinsics',
             'camera_extrinsics', 'ego_intrinsics', 'ego_extrinsics'}
SEED = 915101
LAYOUT = Path('example/self_capture')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_window(stage, name, row, receipt):
    exo = LAYOUT / 'videos' / name / 'exo.mp4'
    prior = LAYOUT / 'ego_priors' / name / 'ego_Prior.mp4'
    depth = LAYOUT / 'depth_maps' / name
    if (set(row) != META_KEYS or row['exo_path'] != './' + exo.as_posix()
            or row['ego_prior_path'] != './' + prior.as_posix()
            or len(row['ego_extrinsics']) != 49):
        raise ValueError(f'{name}: unexpected EXO-only metadata')
    files = sorted((stage / depth).glob('*.npy'))
    if (receipt['name'] != name or receipt['frames'] != 49 or len(files) != 49
            or any(int(''.join(filter(str.isdigit, item.stem))) != index
                   for index, item in enumerate(files))):
        raise ValueError(f'{name}: unexpected 49-frame depth order')
    if (digest(stage / exo) != receipt['exo_sha256']
            or digest(stage / prior) != receipt['prior_sha256']
            or [digest(item) for item in files] != receipt['depth_sha256']):
        raise ValueError(f'{name}: verified input checksum changed')
    return exo, prior, depth


def inputs(require_all):
    for name, expected in STOCK_PINS.items():
        if digest(STOCK / name) != expected:
            raise ValueError(f'Pinned stock file changed: {name}')
    stock_rows = {Path(row['exo_path']).parent.name: row
                  for row in read_json(STOCK / 'meta.json')['test_datasets']}
    stock_receipts = {row['name']: row for row in read_json(STOCK / 'validated-inputs.json')}
    verified, pending = [], []
    for pair, names in WINDOWS.items():
        for index, name in enumerate(names):
            if index == 0:
                stage, row, receipt = STOCK, stock_rows[name], stock_receipts[name]
            else:
                window = (GEOMETRY if pair == 'Pair02' else CALIBRATED) / 'windows' / name
                stage = window / 'inference-input'
                if pair == 'Pair04' and not (window / 'render-receipt.json').is_file():
                    overlap = read_json(window / 'overlap-fixed.json')
                    if not overlap.get('passed_provisional_gate'):
                        raise ValueError(f'{name}: calibrated geometry has not passed')
                    pending.append(name)
                    continue
                receipt = read_json(window / 'render-receipt.json')
                meta = stage / 'meta.json'
                if (receipt['status'] != 'technical_pass_semantic_review_pending'
                        or digest(meta) != receipt['meta_sha256']):
                    raise ValueError(f'{name}: render metadata changed')
                row, = read_json(meta)['test_datasets']
            paths = verify_window(stage, name, row, receipt)
            if row['prompt'] != stock_rows[names[0]]['prompt']:
                raise ValueError(f'{name}: prompt differs across the source range')
            verified.append((name, stage, row, receipt, paths))
    print(json.dumps({'staged_and_hash_verified': [row[0] for row in verified],
                      'calibrated_render_pending': pending}), flush=True)
    if pending and require_all:
        raise ValueError('Both Pair04 priors must be rendered first')
    return verified, pending


def generate(run, verified, environment):
    stage = run / 'inference-input'
    stage.mkdir()
    for item in ('infer.py', 'core'):
        (stage / item).symlink_to((STOCK / item).resolve(strict=True))
    for name, source, row, receipt, paths in verified:
        for relative in paths:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to((source / relative).resolve(strict=True))
        verify_window(stage, name, row, receipt)
    write_json(stage / 'meta.json', {'test_datasets': [entry[2] for entry in verified]})
    write_json(run / 'fresh-inputs.json', {'seed': SEED, 'windows': [entry[3] for entry in verified]})
    raw = run / 'raw'
    raw.mkdir()
    for directory in ('cache', 'tmp'):
        (run / directory).mkdir()
    inference_env = dict(environment, OMP_NUM_THREADS='8', PYTHONHASHSEED=str(SEED),
                         HF_HOME=str(run / 'cache'), TMPDIR=str(run / 'tmp'),
                         EGOX_AUDIT_PATH=str(run / 'runtime_audit.json'))
    command = ['timeout', '--signal=TERM', '--kill-after=10s', '10800s',
               PYTHON, '-B', ROOT / 'scripts/instrumented_infer.py',
               '--meta_data_file', './meta.json',
               '--model_path', WEIGHTS / 'Wan2.1-I2V-14B-480P-Diffusers',
               '--lora_path', WEIGHTS / 'EgoX/pytorch_lora_weights.safetensors',
               '--lora_rank', '256', '--out', raw, '--seed', str(SEED), '--use_GGA',
               '--cos_sim_scaling_factor', '3.0', '--in_the_wild', '--idx', '-1']
    print('FRESH_BATCH_STARTED', raw, flush=True)
    call(command, env=inference_env, cwd=stage, log=run / 'inference.log', timeout=10830)
    if not adapter_ok(run / 'runtime_audit.json'):
        raise ValueError('Official adapter runtime audit failed')


def verify_and_assemble(run, environment):
    raw = run / 'raw'
    names = {name for windows in WINDOWS.values() for name in windows}
    if {path.stem for path in raw.glob('*.mp4')} != names:
        raise ValueError('Fresh batch has an unexpected set of output windows')
    for pair, windows in WINDOWS.items():
        for name in windows:
            output = run / 'verified' / name
            output.mkdir(parents=True)
            receipt = verify_take(name, raw / f'{name}.mp4', output, True)
            write_json(output / 'receipt.json', receipt)
        first, middle, last = (raw / f'{name}.mp4' for name in windows)
        call([PYTHON, '-B', GEOMETRY / 'assemble_extension.py',
              '--first', first, '--middle', middle, '--last', last,
              '--output', run / 'drafts' / pair,
              '--source-start', '0' if pair == 'Pair02' else '24'],
             env=environment, log=run / f'{pair}-assembly.log', timeout=240)
        print('FRESH_DRAFT_TECHNICALLY_VERIFIED', pair, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'run'))
    parser.add_argument('--run-name', default='pair02-pair04-from-start-100f')
    args = parser.parse_args()
    if not socket.gethostname().endswith('66fc463f'):
        raise ValueError('This runner is for AutoDL 008 ending 66fc463f')
    if args.mode == 'check':
        inputs(require_all=False)
        return
    if Path(args.run_name).name != args.run_name or not args.run_name.startswith('pair02-pair04-from-start-'):
        raise ValueError('Expected a new Pair02/Pair04 from-start run name')
    run = BASE / args.run_name
    if run.exists() or run.is_symlink():
        raise FileExistsError(f'{run} already exists')
    environment = env()
    os.environ['PATH'] = environment['PATH']
    verified, _ = inputs(require_all=True)
    if shutil.disk_usage(BASE).free < 4 * 2**30:
        raise ValueError('Insufficient disk space for full generation and collection')
    with (ROOT / '.egox-inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run.mkdir()
        try:
            call([PYTHON, '-I', '-B', GEOMETRY / 'check_runtime_008.py', ROOT, STOCK, WEIGHTS,
                  BIN / 'ffmpeg', BIN / 'ffprobe'], env=environment,
                 log=run / 'runtime-preflight.log', timeout=180)
            generate(run, verified, environment)
            verify_and_assemble(run, environment)
        finally:
            collector = call([PYTHON, '-B', GEOMETRY / 'collect_008.py', '--run', run,
                              '--output', run.with_suffix('.zip')], env=environment, timeout=180)
            print(collector.stdout.decode(), end='', flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr.decode(errors='replace')[-1800:], file=sys.stderr)
        print(f'FRESH_BATCH_STOPPED command returned {exc.returncode}', file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'FRESH_BATCH_STOPPED {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
