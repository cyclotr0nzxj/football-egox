"""Resume and verify official weights on the default disk or AutoDL shared storage.

EGOX_MODEL_ROOT selects a model directory within /root/autodl-fs.
EGOX_RUN_ROOT selects the local root containing records/download_plan.json.
EGOX_LOCAL_SCRATCH selects a staging directory within /root/autodl-tmp.
EGOX_PIPELINE_LOCAL_SCRATCH=1 stages the next file while publishing the current one.
EGOX_RANGE_WORKERS sets the parallel range requests per file (default 16).
EGOX_RANGE_MAX_TIME limits each range request in seconds (default 100).
EGOX_RANGE_SPEED_LIMIT and EGOX_RANGE_SPEED_TIME set the low-speed cutoff
in bytes per second and seconds, respectively (defaults 131072 and 25).
"""
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time

LOCAL_DISK = pathlib.Path('/root/autodl-tmp')
DEFAULT_ROOT = LOCAL_DISK / 'football_egox'
SHARED_MOUNT = pathlib.Path('/root/autodl-fs')
OFFICIAL_MODELS_SHA256 = '2db988fbf78013e11e12d82e19b97d7ed39e8dcfb5abcfc04ac67c60463330f2'
FILE_WORKERS = 4
SEGMENT_THRESHOLD = 100 * 2**20
RANGE_BLOCK_SIZE = 256 * 2**20


def check_mounted_subdirectory(path, mount_root, variable):
    if path == mount_root or not path.is_relative_to(mount_root):
        raise RuntimeError(f'{variable} must be a real subdirectory of {mount_root}.')
    # AutoDL exposes the actual FUSE mount via this advertised path on some hosts.
    mounted_root = mount_root.resolve()
    canonical_root = mounted_root / path.relative_to(mount_root)
    if '..' in path.parts or canonical_root.resolve() != canonical_root:
        raise RuntimeError(f'{variable} crosses a symlink: {path}.')
    result = subprocess.run(['findmnt', '--list', '--json', '--output', 'TARGET'],
                            check=True, capture_output=True, text=True)
    mounts = {pathlib.Path(entry['target']) for entry in json.loads(result.stdout)['filesystems']}
    if mounted_root not in mounts or not mounted_root.is_dir():
        raise RuntimeError(f'{variable} storage is not mounted at {mount_root}.')
    for mount in mounts:
        if (mount != mounted_root and mount.is_relative_to(mounted_root)
                and (mount.is_relative_to(canonical_root) or canonical_root.is_relative_to(mount))):
            raise RuntimeError(f'A nested mount intersects {variable}: {mount}.')
    return canonical_root


def check_shared_mount(model_root):
    return check_mounted_subdirectory(model_root, SHARED_MOUNT, 'EGOX_MODEL_ROOT')


def file_size(path):
    if path.is_symlink():
        raise RuntimeError(f'Downloader file is a symlink: {path}')
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f'Downloader path is not a regular file: {path}')
    return path.stat().st_size


def check_capacity(jobs, model_root, shared):
    missing = 0
    range_scratch = []
    for _, item, relative in jobs:
        path = model_root / relative
        if shared and path.resolve() != path:
            raise RuntimeError(f'Model path crosses a symlink: {path}')
        size = item['size']
        installed = file_size(path)
        if installed is not None:
            if installed != size:
                raise RuntimeError(f'Size mismatch: {path}')
            continue
        partial = path.with_name(path.name + '.partial')
        partial_bytes = file_size(partial) or 0
        if partial_bytes > size:
            raise RuntimeError(f'Oversized partial file: {partial}')
        missing += size - partial_bytes
        if size > SEGMENT_THRESHOLD and partial_bytes < size:
            # Completed range blocks remain on disk until assembly is finished.
            # Existing range blocks already count against free space and need no credit here.
            range_scratch.append(size)
    scratch = sum(sorted(range_scratch, reverse=True)[:FILE_WORKERS])
    reserve = scratch if shared else max(scratch, 30 * 2**30)
    free = shutil.disk_usage(model_root).free
    if free <= missing + reserve:
        raise RuntimeError(f'Insufficient space for models on {model_root}: '
                           f'{free} bytes free; need more than {missing + reserve} bytes '
                           f'({reserve} bytes reserved).')
    if shared:
        print(f'MODEL_STORAGE_OK free={free} missing={missing} scratch={scratch}', flush=True)


def publication_path(path):
    return path.with_name(path.name + '.egox-publish.partial')


def check_publication_capacity(jobs, model_root):
    missing = 0
    for _, item, relative in jobs:
        path = model_root / relative
        if path.resolve() != path:
            raise RuntimeError(f'Model path crosses a symlink: {path}')
        size = item['size']
        installed = file_size(path)
        if installed is not None:
            if installed != size:
                raise RuntimeError(f'Size mismatch: {path}')
            continue
        existing = file_size(publication_path(path)) or 0
        missing += size - min(size, existing)
    free = shutil.disk_usage(model_root).free
    if missing and free <= missing:
        raise RuntimeError(f'Insufficient space to publish models on {model_root}: '
                           f'{free} bytes free; need more than {missing} bytes.')
    print(f'MODEL_STORAGE_OK free={free} missing={missing} scratch=0', flush=True)


def positive_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        raise ValueError(f'{name} must be a positive integer.') from None
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def matches_content_range(headers, start, end, size):
    # Curl dumps every header block, including proxy and redirect responses.
    blocks = headers.replace(b'\r\n', b'\n').split(b'\n\n')
    for block in reversed(blocks):
        status, _, fields = block.partition(b'\n')
        if status.startswith(b'HTTP/'):
            break
    else:
        return False
    if status.split()[1:2] != [b'206']:
        return False
    values = []
    for field in fields.splitlines():
        name, _, value = field.partition(b':')
        if name.lower() == b'content-range':
            values.append(value.strip().lower())
    return values == [f'bytes {start}-{end}/{size}'.encode('ascii')]


def range_state(partial, size):
    actual = file_size(partial) or 0
    chunks = partial.with_name(partial.name + '.chunks')
    if chunks.is_symlink():
        raise RuntimeError(f'Range directory is a symlink: {chunks}')
    state = chunks / 'state.json'
    if file_size(state) is None:
        return chunks, actual, actual, False
    previous = json.loads(state.read_text())
    prefix = previous['prefix']
    if previous['size'] != size or not 0 <= prefix <= actual <= size:
        raise RuntimeError(f'Unexpected partial prefix state: {state}')
    return chunks, actual, prefix, True


def range_bounds(prefix, size):
    return [(start, min(size - 1, start + RANGE_BLOCK_SIZE - 1))
            for start in range(prefix, size, RANGE_BLOCK_SIZE)]


def check_staging_capacity(staging_root, staged, size, previous_publish=None):
    if file_size(staged) is not None:
        return
    partial = staged.with_name(staged.name + '.partial')
    actual = file_size(partial) or 0
    if actual > size:
        raise RuntimeError(f'Oversized partial file: {partial}')
    needed = size - actual
    if size > SEGMENT_THRESHOLD and actual < size:
        chunks, _, prefix, _ = range_state(partial, size)
        for start, end in range_bounds(prefix, size):
            block = chunks / f'{start}-{end}'
            saved = file_size(block) or 0
            if saved > end - start + 1:
                raise RuntimeError(f'Oversized range block: {block}')
            needed += end - start + 1 - saved
    free = shutil.disk_usage(staging_root).free
    if needed and free <= needed and previous_publish is not None:
        print(f'LOCAL_STAGING_WAIT {staged.name} free={free} required={needed}', flush=True)
        previous_publish.result()
        free = shutil.disk_usage(staging_root).free
    if needed and free <= needed:
        raise RuntimeError(f'Insufficient local staging space for {staged}: '
                           f'{free} bytes free; need more than {needed} bytes.')
    print(f'LOCAL_STAGING_OK {staged.name} free={free} required={needed}', flush=True)


def segmented_download(url, partial, size):
    """Keep existing prefix; download byte ranges, then append in order."""
    workers = positive_env('EGOX_RANGE_WORKERS', 16)
    max_time = positive_env('EGOX_RANGE_MAX_TIME', 100)
    speed_limit = positive_env('EGOX_RANGE_SPEED_LIMIT', 131072)
    speed_time = positive_env('EGOX_RANGE_SPEED_TIME', 25)
    progress_threshold = speed_limit * min(speed_time, max_time)
    chunks, actual, prefix, has_state = range_state(partial, size)
    if actual == size:
        return
    chunks.mkdir(exist_ok=True)
    # Keep the original prefix boundary stable across interrupted resumptions.
    state = chunks / 'state.json'
    if actual > prefix:
        # Assembly copied from still-retained chunks; recover without network loss.
        with partial.open('r+b') as output:
            output.truncate(prefix)
    if not has_state:
        state.write_text(json.dumps({'prefix': prefix, 'size': size}))
    ranges = range_bounds(prefix, size)

    def part(bounds):
        start, end = bounds
        target = chunks / f'{start}-{end}'
        tail = target.with_name(target.name + '.tail')
        block_size = end - start + 1
        have = file_size(target) or 0
        if have > block_size:
            raise RuntimeError(f'Oversized range block: {target.name}')
        if have == block_size:
            return target

        # Limit consecutive stalls while allowing steady progress across timeouts.
        stalls = 0
        while stalls < 7:
            tail.unlink(missing_ok=True)
            request_start = start + have
            try:
                with tempfile.NamedTemporaryFile() as headers:
                    response = subprocess.run(
                        ['curl', '-4', '-fLsS', '--connect-timeout', '20',
                         '--max-time', str(max_time), '--speed-time', str(speed_time),
                         '--speed-limit', str(speed_limit), '--max-filesize', str(block_size - have),
                         '--range', f'{request_start}-{end}', '-D', headers.name,
                         '-o', str(tail), '-w', '%{http_code}', url],
                        check=False, capture_output=True, text=True,
                    )
                    http = response.stdout.strip()
                    matched = http == '206' and matches_content_range(
                        headers.read(), request_start, end, size
                    )
                received = file_size(tail) or 0
                saved = 0
                if matched and 0 < received <= block_size - have:
                    with target.open('ab') as output, tail.open('rb') as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                    have += received
                    saved = received
            finally:
                tail.unlink(missing_ok=True)
            if have == block_size:
                return target
            stalls = 0 if saved >= progress_threshold else stalls + 1
            print('RANGE_RETRY', target.name, 'curl', response.returncode, 'http', http,
                  'range_match', matched, 'saved', have, 'stalls', stalls, flush=True)
            if stalls:
                time.sleep(3)
        raise RuntimeError(f'Range retries exhausted: {target.name}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(part, ranges))
    with partial.open('ab') as dest:
        for part_path in parts:
            with part_path.open('rb') as source:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
    if file_size(partial) != size:
        raise RuntimeError(f'Unexpected assembled size: {partial}')


def verify_file(path, item):
    if file_size(path) != item['size']:
        raise RuntimeError(f'Size mismatch: {path}')
    sha = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            sha.update(block)
    expected = item.get('lfs', {}).get('sha256')
    if expected:
        if sha.hexdigest() != expected:
            raise RuntimeError(f'Hash mismatch: {path}')
    else:
        content = path.read_bytes()
        git_sha = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        if git_sha != item['blobId']:
            raise RuntimeError(f'Git blob mismatch: {path}')
    return sha.hexdigest()


def fetch_file(job, model_root, ms_files):
    model, item, relative = job
    path = model_root / relative
    partial = path.with_name(path.name + '.partial')
    if file_size(path) is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://hf-mirror.com/{model['repo']}/resolve/{model['revision']}/{item['rfilename']}?download=true"
        mirror=ms_files.get(item['rfilename']) if model['repo'].startswith('Wan-AI/') else None
        if os.environ.get('EGOX_USE_MODELSCOPE')=='1' and mirror and item.get('lfs') and mirror['Sha256']==item['lfs']['sha256'] and mirror['Size']==item['size']:
            url=f"https://www.modelscope.cn/models/{model['repo']}/resolve/{mirror['Revision']}/{item['rfilename']}"
        if file_size(partial) != item['size']:
            print('START', item['rfilename'], item['size'], flush=True)
            if item['size'] > SEGMENT_THRESHOLD:
                segmented_download(url,partial,item['size'])
            else:
                subprocess.run(['curl', '-fLsS', '--connect-timeout', '20', '--retry', '6', '--retry-delay', '5', '--speed-time', '90', '--speed-limit', '1024', '-C', '-', '-o', str(partial), url], check=True)
        sha = verify_file(partial, item)
        partial.rename(path)
    else:
        sha = verify_file(path, item)
    leftover = partial.with_name(partial.name + '.chunks')
    if leftover.exists() and not leftover.is_symlink():
        shutil.rmtree(leftover)
    return sha


def result_for(job, sha, started):
    _, item, relative = job
    result = {'path': (pathlib.Path('models') / relative).as_posix(), 'bytes': item['size'],
              'sha256': sha, 'official_hash_verified': True, 'seconds': round(time.time()-started, 2)}
    print('VERIFIED', item['rfilename'], result['seconds'], flush=True)
    return result


def fetch(job, model_root, ms_files):
    started = time.time()
    return result_for(job, fetch_file(job, model_root, ms_files), started)


def publish_file(staged, path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = publication_path(path)
    existing = file_size(temporary)
    if existing == item['size']:
        try:
            sha = verify_file(temporary, item)
        except RuntimeError:
            # A previous writer may have stopped after allocating the expected size.
            pass
        else:
            with temporary.open('r+b') as output:
                os.fsync(output.fileno())
            temporary.rename(path)
            return sha

    print('PUBLISH', item['rfilename'], item['size'], flush=True)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as output, staged.open('rb') as source:
        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    sha = verify_file(temporary, item)
    temporary.rename(path)
    return sha


def stage_file(job, staging_root, ms_files, previous_publish=None):
    _, item, relative = job
    staged = staging_root / relative
    if staged.resolve() != staged:
        raise RuntimeError(f'Local model path crosses a symlink: {staged}')
    check_staging_capacity(staging_root, staged, item['size'], previous_publish)
    fetch_file(job, staging_root, ms_files)
    return staged


def publish_staged(job, staged, path, started):
    _, item, _ = job
    sha = publish_file(staged, path, item)
    staged.unlink()
    return result_for(job, sha, started)


def fetch_staged(job, model_root, ms_files, staging_root):
    _, item, relative = job
    started = time.time()
    path = model_root / relative
    if file_size(path) is not None:
        return result_for(job, verify_file(path, item), started)
    staged = stage_file(job, staging_root, ms_files)
    return publish_staged(job, staged, path, started)


def fetch_staged_pipeline(jobs, model_root, ms_files, staging_root):
    results = []
    previous = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as publisher:
        for job in jobs:
            _, item, relative = job
            started = time.time()
            path = model_root / relative
            installed = file_size(path) is not None
            if previous is not None and (previous.done() or installed):
                results.append(previous.result())
                previous = None
            if installed:
                results.append(result_for(job, verify_file(path, item), started))
                continue

            staged = stage_file(job, staging_root, ms_files, previous)
            # Submit only when the previous publisher has verified and released its file.
            if previous is not None:
                results.append(previous.result())
            previous = publisher.submit(publish_staged, job, staged, path, started)
        if previous is not None:
            results.append(previous.result())
    return results


def main():
    run_root = pathlib.Path(os.environ.get('EGOX_RUN_ROOT', DEFAULT_ROOT))
    records = run_root / 'records'
    shared = 'EGOX_MODEL_ROOT' in os.environ
    model_root = pathlib.Path(os.environ.get('EGOX_MODEL_ROOT', run_root / 'models'))
    pipeline = os.environ.get('EGOX_PIPELINE_LOCAL_SCRATCH', '0')
    if pipeline not in ('0', '1'):
        raise RuntimeError('EGOX_PIPELINE_LOCAL_SCRATCH must be 0 or 1.')
    if pipeline == '1' and 'EGOX_LOCAL_SCRATCH' not in os.environ:
        raise RuntimeError('EGOX_PIPELINE_LOCAL_SCRATCH requires EGOX_LOCAL_SCRATCH.')
    if shared:
        model_root = check_shared_mount(model_root)
    staging_root = None
    if 'EGOX_LOCAL_SCRATCH' in os.environ:
        if not shared:
            raise RuntimeError('EGOX_LOCAL_SCRATCH requires EGOX_MODEL_ROOT.')
        staging_root = check_mounted_subdirectory(
            pathlib.Path(os.environ['EGOX_LOCAL_SCRATCH']), LOCAL_DISK, 'EGOX_LOCAL_SCRATCH'
        )
        if (staging_root.is_relative_to(model_root) or model_root.is_relative_to(staging_root)
                or LOCAL_DISK.stat().st_dev == SHARED_MOUNT.stat().st_dev):
            raise RuntimeError('EGOX_LOCAL_SCRATCH must use a separate local disk.')
    plan = json.loads((records / 'download_plan.json').read_text())
    if shared:
        models = json.dumps(plan['models'], sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(models).hexdigest() != OFFICIAL_MODELS_SHA256:
            raise RuntimeError('Shared-storage download plan differs from the fixed official manifest.')
    jobs = []
    for model in plan['models']:
        for item in model['files']:
            filename = pathlib.PurePosixPath(item['rfilename'])
            if filename.is_absolute() or '..' in filename.parts or not filename.parts:
                raise RuntimeError(f'Invalid download manifest filename: {filename}')
            relative = pathlib.Path(model['repo'].split('/')[-1]) / filename
            jobs.append((model, item, relative))
    ms_path = records / 'modelscope_files.json'
    ms_files = {f['Path']: f for f in json.loads(ms_path.read_text())['Data']['Files']} if ms_path.exists() else {}

    model_root.mkdir(parents=True, exist_ok=True)
    lock_roots = [model_root]
    if staging_root is not None:
        staging_root.mkdir(parents=True, exist_ok=True)
        lock_roots.append(staging_root)
    with contextlib.ExitStack() as locks:
        for root in lock_roots:
            lock_path = root / '.download_models.lock'
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            lock = locks.enter_context(os.fdopen(lock_fd, 'r+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if staging_root is None:
            check_capacity(jobs, model_root, shared)
            with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
                results = list(pool.map(lambda job: fetch(job, model_root, ms_files), jobs))
        else:
            check_publication_capacity(jobs, model_root)
            if pipeline == '1':
                results = fetch_staged_pipeline(jobs, model_root, ms_files, staging_root)
            else:
                results = [fetch_staged(job, model_root, ms_files, staging_root) for job in jobs]
        with tempfile.NamedTemporaryFile(mode='w', dir=records, delete=False) as temporary:
            json.dump(results, temporary, indent=2)
        pathlib.Path(temporary.name).replace(records / 'model_verification.json')
    print('ALL_MODELS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
