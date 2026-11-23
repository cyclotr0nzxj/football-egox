#!/usr/bin/env python3
"""Resume the two pinned public weight files over the existing A800 SSH master."""

import argparse
import concurrent.futures
import http.client
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


MIB = 1024 * 1024
CHUNK_SIZE = 16 * MIB
REMOTE_ROOT = "/root/autodl-tmp/football_egox/pair04-20261121"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_SOCKET = "/tmp/pair04-a800.dWQKiW/ctl"


@dataclass(frozen=True)
class Model:
    key: str
    repository: str
    revision: str
    filename: str
    size: int
    sha256: str
    destination: str

    @property
    def url(self):
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{self.filename}?download=true"
        )


MODELS = (
    Model(
        "moge",
        "Ruicheng/moge-2-vitl-normal",
        "b135031bae30b5ac2ae141a0e68717795ce38340",
        "model.pt",
        1323815904,
        "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01",
        "caches/hf/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/"
        "b135031bae30b5ac2ae141a0e68717795ce38340/model.pt",
    ),
    Model(
        "vda",
        "depth-anything/Video-Depth-Anything-Large",
        "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
        "video_depth_anything_vitl.pth",
        1538392012,
        "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
        "caches/torch/hub/checkpoints/video_depth_anything_vitl.pth",
    ),
)


# The remote process only receives fixed model metadata, chunk coordinates and
# validated HTTP response bytes. Its stdout is JSON for inventory and receipts.
REMOTE_CODE = '''
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

root, chunk_size, key, size, expected_hash, destination, operation, *coords = sys.argv[1:]
root = Path(root)
chunk_size = int(chunk_size)
size = int(size)
target = root / destination
staging = root / "caches/range-recovery"


def chunk_path(start):
    return staging / f"{key}-{start:012d}"


def chunk_length(start):
    return min(size - start, chunk_size)


def existing_length(start):
    path = chunk_path(start)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(info.st_mode) or info.st_size > chunk_length(start):
        raise RuntimeError("invalid staging chunk")
    return info.st_size


def valid_target():
    if not os.path.lexists(target):
        return False
    info = target.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise RuntimeError("existing destination has wrong type or size")
    digest = hashlib.sha256()
    with target.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError("existing destination has wrong SHA-256")
    return True


def inventory():
    if valid_target():
        return {"complete": True, "chunks": {}}
    if not staging.is_dir():
        raise RuntimeError("range-recovery staging directory is missing")
    return {"complete": False, "chunks": {
        start: existing_length(start) for start in range(0, size, chunk_size)
    }}


def append():
    start, previous_size, count = map(int, coords)
    if (start < 0 or start >= size or start % chunk_size or previous_size < 0
            or count <= 0 or previous_size + count > chunk_length(start)):
        raise RuntimeError("invalid append coordinates")
    data = sys.stdin.buffer.read(count + 1)
    if len(data) != count:
        raise RuntimeError("append input length mismatch")
    flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
    if previous_size == 0:
        flags |= os.O_CREAT
    with os.fdopen(os.open(chunk_path(start), flags, 0o600), "wb") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        info = os.fstat(output.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != previous_size:
            raise RuntimeError("staging chunk changed before append")
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
        new_size = os.fstat(output.fileno()).st_size
        if new_size != previous_size + count:
            raise RuntimeError("staging chunk changed during append")
    return {"chunk": start, "size": new_size}


def assemble():
    if valid_target():
        return {"bytes": size, "sha256": expected_hash}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        digest = hashlib.sha256()
        written = 0
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.relay-", delete=False
        ) as output:
            temp = Path(output.name)
            for start in range(0, size, chunk_size):
                if existing_length(start) != chunk_length(start):
                    raise RuntimeError("staging coverage is incomplete")
                flags = os.O_RDONLY | os.O_NOFOLLOW
                with os.fdopen(os.open(chunk_path(start), flags), "rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        written += len(block)
            if written != size or digest.hexdigest() != expected_hash:
                raise RuntimeError("assembled size or SHA-256 mismatch")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temp, target)
        except FileExistsError:
            if not valid_target():
                raise RuntimeError("destination appeared during publication")
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return {"bytes": size, "sha256": expected_hash}


try:
    result = {"inventory": inventory, "append": append, "assemble": assemble}[operation]()
    print(json.dumps(result, separators=(",", ":")))
except RuntimeError as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
except OSError as error:
    print(f"remote filesystem error (errno={error.errno})", file=sys.stderr)
    sys.exit(1)
'''


def ssh_options(socket):
    return [
        "/usr/bin/ssh", "-F", "/dev/null", "-S", socket, "-p", "45845", "-T",
        "-o", "BatchMode=yes", "-o", "ProxyCommand=false",
    ]


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise RuntimeError("relay deadline reached")
    return seconds


def remote(socket, deadline, model, operation, *coords, data=b""):
    command = shlex.join([
        REMOTE_PYTHON, "-I", "-c", REMOTE_CODE, REMOTE_ROOT, str(CHUNK_SIZE),
        model.key, str(model.size), model.sha256, model.destination, operation,
        *(str(value) for value in coords),
    ])
    timeout_cap = 180 if operation != "append" else 45
    try:
        result = subprocess.run(
            [*ssh_options(socket), "root@example.invalid", command],
            input=data, capture_output=True, timeout=min(timeout_cap, remaining(deadline)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{model.key} {operation}: SSH timed out") from None
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = message[-1] if message else f"SSH exit {result.returncode}"
        raise RuntimeError(f"{model.key} {operation}: {detail}")
    try:
        return json.loads(result.stdout)
    except ValueError:
        raise RuntimeError(f"{model.key} {operation}: invalid receipt") from None


def fetch_range(model, start, end, deadline):
    request = urllib.request.Request(model.url, headers={
        "Range": f"bytes={start}-{end}",
        "Accept-Encoding": "identity",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=min(18, remaining(deadline))) as response:
                expected_range = f"bytes {start}-{end}/{model.size}"
                encodings = response.headers.get_all("Content-Encoding", [])
                if (response.status != 206
                        or response.headers.get_all("Content-Range", []) != [expected_range]
                        or any(encoding.lower() != "identity" for encoding in encodings)):
                    raise RuntimeError(f"unexpected HTTP range response (status={response.status})")
                count = end - start + 1
                output = bytearray()
                while len(output) <= count:
                    remaining(deadline)
                    block = response.read1(min(256 * 1024, count + 1 - len(output)))
                    if not block:
                        break
                    output.extend(block)
                if len(output) != count:
                    raise RuntimeError("HTTP response body length mismatch")
                return output
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            if status < 500 or attempt == 2:
                raise RuntimeError(f"HTTP {status}") from None
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise RuntimeError("HTTP transport error after three attempts") from None
        time.sleep(min(2 ** attempt, remaining(deadline)))


def relay_chunk(socket, deadline, stop, request_size, model, chunk_start, have):
    chunk_size = min(CHUNK_SIZE, model.size - chunk_start)
    while have < chunk_size and not stop.is_set():
        count = min(request_size, chunk_size - have)
        start = chunk_start + have
        try:
            data = fetch_range(model, start, start + count - 1, deadline)
            if stop.is_set():
                return
            receipt = remote(socket, deadline, model, "append", chunk_start, have, count, data=data)
            if receipt != {"chunk": chunk_start, "size": have + count}:
                raise RuntimeError("unexpected append receipt")
        except RuntimeError as error:
            raise RuntimeError(f"{model.key} bytes {start}-{start + count - 1}: {error}") from None
        have += count
        print(f"{model.key} chunk {chunk_start}: {have}/{chunk_size} bytes", flush=True)


def run(args):
    deadline = time.monotonic() + args.deadline_seconds
    control = subprocess.run(
        [*ssh_options(args.socket), "-O", "check", "root@example.invalid"],
        capture_output=True, timeout=5, check=False,
    )
    if control.returncode:
        raise RuntimeError("A800 SSH control master is unavailable")

    jobs = []
    pending_models = []
    for model in MODELS:
        inventory = remote(args.socket, deadline, model, "inventory")
        if inventory["complete"]:
            print(f"{model.key}: verified existing {model.size} bytes, SHA-256 {model.sha256}", flush=True)
            continue
        pending_models.append(model)
        missing = 0
        for chunk_start in range(0, model.size, CHUNK_SIZE):
            have = inventory["chunks"][str(chunk_start)]
            need = min(CHUNK_SIZE, model.size - chunk_start) - have
            if need > 0:
                missing += need
                jobs.append((model, chunk_start, have))
        print(f"{model.key}: {missing} missing bytes", flush=True)

    if args.inventory_only:
        return

    stop = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(relay_chunk, args.socket, deadline, stop, args.request_mib * MIB, *job)
            for job in jobs
        ]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise

    for model in pending_models:
        receipt = remote(args.socket, deadline, model, "assemble")
        if receipt != {"bytes": model.size, "sha256": model.sha256}:
            raise RuntimeError(f"{model.key}: unexpected publication receipt")
        print(f"{model.key}: published {model.size} bytes, SHA-256 {model.sha256}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--request-mib", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 1800:
        parser.error("--deadline-seconds must be between 1 and 1800")
    try:
        run(args)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Relay stopped. {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Relay interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
