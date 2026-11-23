#!/usr/bin/env python3
"""Relay the pinned public EgoX LoRA into 008's SSD staging directory."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import threading
import time

from relay_public_weights import REMOTE_PYTHON, REMOTE_ROOT, remaining


REPO = "DAVIAN-Robotics/EgoX"
REVISION = "28b19ef9f079bf73c84b2347fe6701e8ab7f6634"
FILENAME = "pytorch_lora_weights.safetensors"
SIZE = 3355522016
SHA256 = "1c61d1871d379358a0e98f3acd9bfffcd1c4300b9096231327a7da5f1683e9f6"
CHUNK_SIZE = 16 * 1024**2
SSD_ROOT = f"{REMOTE_ROOT}/egox-shared-008/weight-staging"
TARGET = "root@example.invalid"
SOURCES = {"official": "huggingface.co", "mirror": "hf-mirror.com"}


# Run on the SSD host. No inputs besides the fixed public manifest and range bytes.
REMOTE_CODE = '''
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

root, name, revision, size, expected_sha, chunk_size, operation, *args = sys.argv[1:]
root = Path(root)
size, chunk_size = int(size), int(chunk_size)
target = root / "EgoX" / name
staging = target.parent / (".relay-008-" + revision)
work = staging / "weight.incomplete"
starts = range(0, size, chunk_size)


def chunk_length(start):
    return min(chunk_size, size - start)


def check_native_staging():
    for suffix in (".partial", ".partial.chunks"):
        if os.path.lexists(target.with_name(name + suffix)):
            raise RuntimeError("native LoRA staging exists; coordinate with download_models.py")


def open_regular(path, flags=os.O_RDONLY):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeError("relay path is not a regular file")
    return fd


def verify(path):
    with os.fdopen(open_regular(path), "rb") as source:
        if os.fstat(source.fileno()).st_size != size:
            raise RuntimeError("LoRA file size mismatch")
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError("LoRA file SHA-256 mismatch")


def marker_path(start):
    return staging / f"done-{start:012d}"


def marker(start):
    try:
        with os.fdopen(open_regular(marker_path(start)), "rb") as source:
            data = source.read(66)
    except FileNotFoundError:
        return None
    if not re.fullmatch(b"[0-9a-f]{64}\\n", data):
        raise RuntimeError("invalid LoRA completion marker")
    return data[:-1].decode("ascii")


@contextmanager
def staging_lock(exclusive):
    fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield fd
    finally:
        os.close(fd)


def empty_inventory():
    return {"complete": False, "chunks": [], "free_bytes": free_bytes()}


def free_bytes():
    statvfs = os.statvfs(root)
    return statvfs.f_bavail * statvfs.f_frsize


def prepare_work():
    if not os.path.lexists(work):
        for start in starts:
            marker_path(start).unlink(missing_ok=True)
        if free_bytes() < size + 16 * 1024**2:
            raise RuntimeError("SSD has less than the LoRA size plus 16 MiB free")
        with os.fdopen(open_regular(work, os.O_RDWR | os.O_CREAT | os.O_EXCL), "wb") as output:
            os.ftruncate(output.fileno(), size)
    with os.fdopen(open_regular(work, os.O_RDWR), "wb") as output:
        fd = output.fileno()
        info = os.fstat(fd)
        # A stopped initialization can leave an empty inode before ftruncate.
        if not info.st_size and not any(marker(start) for start in starts):
            os.ftruncate(fd, size)
            info = os.fstat(fd)
        if info.st_size != size or info.st_nlink != 1:
            raise RuntimeError("unexpected LoRA work file size or link count")
        missing_bytes = max(0, size - info.st_blocks * 512)
        if missing_bytes:
            if free_bytes() < missing_bytes + 16 * 1024**2:
                raise RuntimeError("SSD has insufficient free space to reserve remaining LoRA bytes")
            os.posix_fallocate(fd, 0, size)
        os.fsync(fd)


def inventory(prepare):
    if os.path.lexists(target):
        verify(target)
        if prepare and os.path.lexists(staging):
            with staging_lock(exclusive=True) as directory:
                if os.path.lexists(work) and os.path.lexists(target) and work.samefile(target):
                    work.unlink()
                    os.fsync(directory)
        return {"complete": True}
    check_native_staging()
    if prepare:
        target.parent.mkdir(exist_ok=True)
        staging.mkdir(mode=0o700, exist_ok=True)
    if not os.path.lexists(staging):
        return empty_inventory()
    with staging_lock(exclusive=True) as directory:
        if os.path.lexists(target):
            verify(target)
            return {"complete": True}
        check_native_staging()
        if prepare:
            prepare_work()
            os.fsync(directory)
        if not os.path.lexists(work):
            return empty_inventory()
        with os.fdopen(open_regular(work), "rb") as source:
            length = os.fstat(source.fileno()).st_size
            if length not in (0, size):
                raise RuntimeError("unexpected LoRA work file length")
        return {"complete": False, "chunks": [start for start in starts if marker(start)],
                "free_bytes": free_bytes()}


def store(start, sha):
    if start not in starts or not re.fullmatch("[0-9a-f]{64}", sha):
        raise RuntimeError("invalid LoRA write coordinates or chunk digest")
    length = chunk_length(start)
    data = sys.stdin.buffer.read(length + 1)
    if len(data) != length or hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError("SSH chunk bytes or SHA-256 mismatch")
    with staging_lock(exclusive=False) as directory:
        check_native_staging()
        with os.fdopen(open_regular(work, os.O_RDWR), "wb") as output:
            fd = output.fileno()
            fcntl.lockf(fd, fcntl.LOCK_EX, length, start)
            previous = marker(start)
            if previous is not None:
                if previous != sha:
                    raise RuntimeError("range disagrees with existing completion marker")
                return {"start": start, "bytes": length}
            info = os.fstat(fd)
            if info.st_size != size or info.st_nlink != 1 or info.st_blocks * 512 < size:
                raise RuntimeError("LoRA work file is not reserved")
            remaining_data = memoryview(data)
            offset = start
            while remaining_data:
                count = os.pwrite(fd, remaining_data, offset)
                if count <= 0:
                    raise RuntimeError("short LoRA work-file write")
                offset += count
                remaining_data = remaining_data[count:]
            os.fdatasync(fd)
            with tempfile.NamedTemporaryFile(dir=staging, prefix=".marker-", delete=False) as marker_file:
                temporary = Path(marker_file.name)
                try:
                    marker_file.write((sha + "\\n").encode("ascii"))
                    marker_file.flush()
                    os.fsync(marker_file.fileno())
                    os.replace(temporary, marker_path(start))
                    os.fsync(directory)
                finally:
                    temporary.unlink(missing_ok=True)
    return {"start": start, "bytes": length}


def publish():
    with staging_lock(exclusive=True) as directory:
        if not os.path.lexists(target):
            check_native_staging()
            if any(marker(start) is None for start in starts):
                raise RuntimeError("LoRA range coverage is incomplete")
            verify(work)
            check_native_staging()
            try:
                os.link(work, target, follow_symlinks=False)
            except FileExistsError:
                verify(target)
            parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        else:
            verify(target)
        work.unlink(missing_ok=True)
        os.fsync(directory)
    return {"bytes": size, "sha256": expected_sha}


try:
    if not root.is_dir() or root.resolve() != root:
        raise RuntimeError("008 SSD staging root is missing or crosses a symlink")
    if target.parent.resolve() != target.parent:
        raise RuntimeError("LoRA staging folder crosses a symlink")
    if operation in ("inventory", "prepare"):
        receipt = inventory(operation == "prepare")
    elif operation == "store":
        receipt = store(int(args[0]), args[1])
    elif operation == "publish":
        receipt = publish()
    else:
        raise RuntimeError("unexpected LoRA relay operation")
    print(json.dumps(receipt))
except (OSError, RuntimeError) as error:
    print(f"Remote filesystem error (errno={error.errno})" if isinstance(error, OSError) else error,
          file=sys.stderr)
    sys.exit(1)
'''


def check_manifest():
    plan = Path(__file__).resolve().parents[2] / "records/download_plan.json"
    models = [model for model in json.loads(plan.read_text())["models"] if model["repo"] == REPO]
    if len(models) != 1:
        raise RuntimeError("EgoX LoRA manifest entry is missing or ambiguous")
    model = models[0]
    files = [(item["rfilename"], item["size"], item["lfs"]["size"], item["lfs"]["sha256"])
             for item in model["files"]]
    if (model["revision"], model["bytes"], files) != (REVISION, SIZE, [(FILENAME, SIZE, SIZE, SHA256)]):
        raise RuntimeError("EgoX LoRA manifest does not match the pinned revision, bytes and SHA-256")


def ssh_options(socket):
    return ["/usr/bin/ssh", "-F", "/dev/null", "-S", socket, "-p", "41248", "-T",
            "-o", "BatchMode=yes", "-o", "ProxyCommand=false"]


def remote(socket, deadline, operation, *args, data=b""):
    command = shlex.join([REMOTE_PYTHON, "-I", "-c", REMOTE_CODE, SSD_ROOT, FILENAME,
                          REVISION, str(SIZE), SHA256, str(CHUNK_SIZE), operation,
                          *(str(arg) for arg in args)])
    try:
        result = subprocess.run([*ssh_options(socket), TARGET, command], input=data, capture_output=True,
                                timeout=min(180, remaining(deadline)), check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{operation}: SSH timed out") from None
    if result.returncode:
        lines = result.stderr.decode(errors="replace").strip().splitlines()
        detail = lines[-1] if lines and result.returncode != 255 else f"SSH exit {result.returncode}"
        raise RuntimeError(f"{operation}: {detail}")
    try:
        return json.loads(result.stdout)
    except ValueError:
        raise RuntimeError(f"{operation}: invalid SSH receipt") from None


def valid_headers(headers, start, end):
    blocks = headers.replace(b"\r\n", b"\n").split(b"\n\n")
    for block in reversed(blocks):
        status, _, fields = block.partition(b"\n")
        if status.startswith(b"HTTP/"):
            break
    else:
        return False
    values = {}
    for field in fields.splitlines():
        name, _, value = field.partition(b":")
        values.setdefault(name.lower(), []).append(value.strip().lower())
    return (status.split()[1:2] == [b"206"]
            and values.get(b"content-range") == [f"bytes {start}-{end}/{SIZE}".encode("ascii")]
            and values.get(b"content-encoding", []) in ([], [b"identity"]))


def fetch_range(source, start, end, deadline):
    count = end - start + 1
    url = f"https://{SOURCES[source]}/{REPO}/resolve/{REVISION}/{FILENAME}?download=true"
    with tempfile.NamedTemporaryFile(prefix="egox-lora-headers-") as headers:
        timeout = max(0.001, min(120, remaining(deadline)))
        command = ["/usr/bin/curl", "-q", "--fail", "--silent", "--location", "--max-redirs", "5",
                   "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "12",
                   "--max-time", f"{timeout:.3f}", "--speed-time", "20", "--speed-limit", "32768",
                   "--header", "Accept-Encoding: identity", "--range", f"{start}-{end}",
                   "--max-filesize", str(count), "--dump-header", headers.name, "--output", "-", url]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as curl:
            data = curl.stdout.read(count + 1)
            if len(data) > count:
                curl.kill()
            code = curl.wait()
        if code or len(data) != count or not valid_headers(headers.read(), start, end):
            raise RuntimeError(f"{source}: invalid or failed HTTP range response (curl exit {code})")
    return data


def transfer_chunk(args, deadline, stop, start):
    if stop.is_set():
        return
    end = min(start + CHUNK_SIZE, SIZE) - 1
    sources = (args.source,)
    if args.source == "auto":
        sources = ("official", "mirror") if (start // CHUNK_SIZE) % 2 == 0 else ("mirror", "official")
    failures = []
    data = None
    for attempt in range(3):
        for source in sources:
            if stop.is_set():
                return
            try:
                data = fetch_range(source, start, end, deadline)
                break
            except RuntimeError as error:
                failures.append(str(error))
        if data is not None:
            break
        if attempt < 2:
            delay = min(2 ** (attempt + 1) + random.uniform(0, 0.5), remaining(deadline))
            if stop.wait(delay):
                return
    if data is None:
        raise RuntimeError(f"range {start}-{end}: {', '.join(failures)}")
    if stop.is_set():
        return
    digest = hashlib.sha256(data).hexdigest()
    receipt = remote(args.socket, deadline, "store", start, digest, data=data)
    if receipt != {"start": start, "bytes": len(data)}:
        raise RuntimeError(f"range {start}-{end}: unexpected SSH receipt")
    return len(data), source


def run(args):
    deadline = time.monotonic() + args.deadline_seconds
    check_manifest()
    control = subprocess.run([*ssh_options(args.socket), "-O", "check", TARGET], capture_output=True,
                             timeout=min(5, remaining(deadline)), check=False)
    if control.returncode:
        raise RuntimeError("008 SSH control master is unavailable")
    status = remote(args.socket, deadline, "inventory" if args.inventory_only else "prepare")
    if not isinstance(status, dict):
        raise RuntimeError("invalid SSD inventory receipt")
    if status == {"complete": True}:
        print(f"EGOX_LORA_VERIFIED_EXISTING bytes={SIZE} sha256={SHA256}", flush=True)
        return
    starts = range(0, SIZE, CHUNK_SIZE)
    chunks = status.get("chunks", [])
    if (status.get("complete") is not False or not isinstance(chunks, list)
            or any(type(start) is not int or start not in starts for start in chunks)
            or len(chunks) != len(set(chunks)) or type(status.get("free_bytes")) is not int):
        raise RuntimeError("invalid SSD inventory receipt")
    completed = set(chunks)
    verified_bytes = sum(min(CHUNK_SIZE, SIZE - start) for start in completed)
    print(f"EGOX_LORA_RESUME written={verified_bytes}/{SIZE} ssd_free={status['free_bytes']}", flush=True)
    if args.inventory_only:
        return
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(transfer_chunk, args, deadline, stop, start)
                   for start in starts if start not in completed]
        try:
            for future in as_completed(futures):
                count, source = future.result()
                verified_bytes += count
                print(f"EGOX_LORA_PROGRESS written={verified_bytes}/{SIZE} source={source}", flush=True)
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    receipt = remote(args.socket, deadline, "publish")
    if receipt != {"bytes": SIZE, "sha256": SHA256}:
        raise RuntimeError("unexpected publication receipt")
    print(f"EGOX_LORA_PUBLISHED bytes={SIZE} sha256={SHA256}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="existing authenticated SSH control socket for 008")
    parser.add_argument("--source", choices=("auto", *SOURCES), default="auto")
    parser.add_argument("--workers", type=int, choices=range(3, 7), default=3)
    parser.add_argument("--deadline-seconds", type=int, default=2400)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 3600:
        parser.error("--deadline-seconds must be between 1 and 3600")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"LoRA relay stopped. {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("LoRA relay interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
