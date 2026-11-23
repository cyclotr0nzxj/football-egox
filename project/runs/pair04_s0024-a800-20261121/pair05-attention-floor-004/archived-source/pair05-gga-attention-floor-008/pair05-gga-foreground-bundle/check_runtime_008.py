"""Verify the pinned EgoX runtime and previously verified cache on AutoDL 008."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


GIB = 2**30
INFER_SHA = '99199fac22858821d356f0ec3b8577495290dd579a8d80368de1fca68d559bed'
CORE_TREE_SHA = '2a6167d580551b0a6a7deb4765c959ec6c490fe0d0d8b3f3b38eeaac57e4bde8'
AUDIT_SHA = '37055dcc0a44785e3e4e8846de596db1a0256af4004204f40a26a5a6941ac063'
MODEL_TREE_SHA = 'cb1c8333d1686384ba011857069aa471181ba62828d851a32573b84d8f4c7116'


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_code(stage, audit):
    if sha256_file(stage / 'infer.py') != INFER_SHA or sha256_file(audit) != AUDIT_SHA:
        raise ValueError('Expected official EgoX infer.py and its original observational wrapper.')
    core = sorted((stage / 'core').rglob('*.py'))
    files = ''.join(f'{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n' for path in core)
    if len(core) != 30 or hashlib.sha256(files.encode()).hexdigest() != CORE_TREE_SHA:
        raise ValueError('EgoX core differs from the pinned 43c095b552a8 snapshot.')


def verify_model_receipts(root, weights):
    if 'ALL_MODELS_VERIFIED' not in (root / 'download-shared.log').read_text().splitlines():
        raise ValueError('The completed 008 weights verification log is required.')
    rows = json.loads((root / 'records/model_verification.json').read_text())
    index = ''.join(f'{row["sha256"]}  {row["bytes"]}  {row["path"]}\n'
                    for row in sorted(rows, key=lambda row: row['path']))
    if len(rows) != 35 or hashlib.sha256(index.encode()).hexdigest() != MODEL_TREE_SHA:
        raise ValueError('The model receipt differs from the previously verified 35-file cache.')
    for row in rows:
        path = weights / Path(row['path']).relative_to('models')
        if row['official_hash_verified'] is not True or not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError(f'Missing or changed cached model file: {path}.')
        if row['bytes'] <= 1024 * 1024 and sha256_file(path) != row['sha256']:
            raise ValueError(f'Cached model configuration checksum mismatch: {path}.')


def available_memory():
    cgroup = Path('/sys/fs/cgroup')
    if (cgroup / 'memory.max').is_file():
        limit, used = cgroup / 'memory.max', cgroup / 'memory.current'
        file_cache_key = 'inactive_file'
    else:
        cgroup = cgroup / 'memory'
        limit, used = cgroup / 'memory.limit_in_bytes', cgroup / 'memory.usage_in_bytes'
        file_cache_key = 'total_inactive_file'
    numbers = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
    limit_value = limit.read_text().strip()
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    host_available = int(meminfo['MemAvailable'].split()[0]) * 1024
    if limit_value == 'max':
        return host_available
    cgroup_available = int(limit_value) - int(used.read_text()) + int(numbers[file_cache_key])
    return min(cgroup_available, host_available)


def verify_gpu_and_imports(stage, weights):
    import cv2
    import numpy
    import torch
    from safetensors import safe_open

    if sys.version_info[:2] != (3, 10):
        raise ValueError('Expected the existing Python 3.10 EgoX environment.')
    pinned = {'diffusers': '0.34.0', 'transformers': '4.49.0', 'accelerate': '1.5.2', 'peft': '0.17.1'}
    for package, expected in pinned.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Expected {package} {expected} in the verified EgoX environment.')
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4' or not torch.cuda.is_available():
        raise ValueError('Expected CUDA-enabled PyTorch 2.5.1+cu124.')
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != 'NVIDIA A800 80GB PCIe' or gpu.total_memory < 79 * GIB:
        raise ValueError(f'Unexpected Torch CUDA device: {gpu.name}.')
    matrix = torch.eye(8, device='cuda', dtype=torch.bfloat16)
    if not torch.equal(matrix @ matrix, matrix):
        raise ValueError('CUDA BF16 smoke test failed.')
    torch.cuda.synchronize()

    sys.path.insert(0, str(stage))
    from core.finetune.datasets.utils import iproj_disp
    from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline
    from core.inference.wan import generate_video

    if not all(callable(item) for item in (iproj_disp, WanWidthConcatImageToVideoPipeline, generate_video)):
        raise ValueError('EgoX inference imports are incomplete.')
    adapter = weights / 'EgoX/pytorch_lora_weights.safetensors'
    with safe_open(adapter, framework='pt', device='cpu') as tensors:
        keys = tensors.keys()
        if not keys or not bool(torch.isfinite(tensors.get_slice(keys[0])[0:1].float()).all()):
            raise ValueError('The cached official adapter is unreadable.')
    return {'python': sys.version.split()[0], 'opencv': cv2.__version__, 'numpy': numpy.__version__,
            'packages': sorted((item.metadata['Name'], item.version) for item in importlib.metadata.distributions())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('stage', type=Path)
    parser.add_argument('weights', type=Path)
    parser.add_argument('ffmpeg', type=Path)
    parser.add_argument('ffprobe', type=Path)
    args = parser.parse_args()

    free = shutil.disk_usage(args.root).free
    ram = available_memory()
    if free < 2 * GIB or ram < 110 * GIB:
        raise ValueError(f'Require at least 2 GiB local scratch and 110 GiB available RAM. Observed {free}, {ram} bytes.')
    verify_code(args.stage, args.root / 'scripts/instrumented_infer.py')
    verify_model_receipts(args.root, args.weights)
    versions = verify_gpu_and_imports(args.stage, args.weights)
    versions.update(ffmpeg=sha256_file(args.ffmpeg), ffprobe=sha256_file(args.ffprobe))
    signature = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    print(f'EGOX_RUNTIME_RESOURCES free_disk_bytes={free} available_ram_bytes={ram}', flush=True)
    print(f'EGOX_RUNTIME_FINGERPRINT {signature}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f'EGOX_RUNTIME_CHECK_FAILED {error}') from error
