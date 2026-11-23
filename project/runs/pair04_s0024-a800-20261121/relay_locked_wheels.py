#!/usr/bin/env python3
"""Stage missing Renderer lock artifacts from PyPI through the A800 SSH master."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

from relay_public_weights import DEFAULT_SOCKET, REMOTE_ROOT, remaining, ssh_options


LOCK_SHA256 = "60f344464f17478ca3ef86717899bc3820f2a778edb424a792112f1a1d5a7199"
TARGET = "root@example.invalid"
REMOTE_PYTHON = f"{REMOTE_ROOT}/renderer-env/bin/python"
BLOCK = 256 * 1024
FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


REMOTE_CODE = '''
from email.parser import BytesHeaderParser
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import sysconfig
import tempfile

root, expected_lock_hash, operation, *args = sys.argv[1:]
root = Path(root)
wheelhouse = root / "locked-wheelhouse"


def inventory():
    lock = (root / "Renderer-sparse/envs/requirements.txt").read_bytes()
    if hashlib.sha256(lock).hexdigest() != expected_lock_hash:
        raise RuntimeError("Renderer lock hash mismatch")
    glibc = os.confstr("CS_GNU_LIBC_VERSION")
    if (sys.implementation.name, sys.version_info[:2], platform.system(),
            platform.machine(), sysconfig.get_config_var("SOABI")) != (
                "cpython", (3, 10), "Linux", "x86_64", "cpython-310-x86_64-linux-gnu"):
        raise RuntimeError("unexpected Python or Linux ABI")
    match = re.fullmatch(r"glibc (2)\\.(\\d+)", glibc or "")
    if not match:
        raise RuntimeError("GNU libc version unavailable")
    packages = set()
    for pattern in ("uv-cache/archive-v0/*/*.dist-info/METADATA",
                    "renderer-env/lib/python3.10/site-packages/*.dist-info/METADATA"):
        for path in root.glob(pattern):
            with path.open("rb") as stream:
                metadata = BytesHeaderParser().parse(stream)
            if metadata["Name"] and metadata["Version"]:
                packages.add((metadata["Name"], metadata["Version"]))
    return {
        "lock": lock.decode("utf-8"), "packages": sorted(packages),
        "python": platform.python_version(), "glibc": tuple(map(int, match.groups())),
    }


def destination(filename, size, sha):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]*", filename):
        raise RuntimeError("invalid artifact filename")
    if size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError("invalid artifact metadata")
    if os.path.lexists(wheelhouse) and not stat.S_ISDIR(wheelhouse.lstat().st_mode):
        raise RuntimeError("invalid wheelhouse directory")
    return wheelhouse / filename


def verified(target, size, expected_sha):
    if not os.path.lexists(target):
        return False
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise RuntimeError("existing artifact size or type mismatch")
    with os.fdopen(os.open(target, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError("existing artifact SHA-256 mismatch")
    return True


def check():
    entries = json.load(sys.stdin)
    if not isinstance(entries, list) or len(entries) > 150:
        raise RuntimeError("invalid inventory request")
    result = {}
    for filename, size, sha in entries:
        try:
            path = destination(filename, size, sha)
            result[filename] = "verified" if verified(path, size, sha) else "missing"
        except RuntimeError as error:
            result[filename] = str(error)
    return result


def stage():
    filename, size_text, expected_sha = args
    size = int(size_text)
    target = destination(filename, size, expected_sha)
    wheelhouse.mkdir(exist_ok=True)
    temp = None
    try:
        digest = hashlib.sha256()
        remaining_bytes = size
        with tempfile.NamedTemporaryFile(dir=wheelhouse, prefix=".relay-", delete=False) as output:
            temp = Path(output.name)
            while remaining_bytes:
                block = sys.stdin.buffer.read(min(1024 * 1024, remaining_bytes))
                if not block:
                    raise RuntimeError("relay artifact ended early")
                digest.update(block)
                output.write(block)
                remaining_bytes -= len(block)
            if sys.stdin.buffer.read(1):
                raise RuntimeError("relay artifact exceeded expected size")
            if digest.hexdigest() != expected_sha:
                raise RuntimeError("relay artifact SHA-256 mismatch")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temp, target)
        except FileExistsError:
            if not verified(target, size, expected_sha):
                raise RuntimeError("destination changed during publication")
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return {"bytes": size, "sha256": expected_sha}


try:
    print(json.dumps({"inventory": inventory, "check": check, "stage": stage}[operation]()))
except RuntimeError as error:
    print(error, file=sys.stderr)
    sys.exit(1)
except OSError as error:
    print(f"remote filesystem error (errno={error.errno})", file=sys.stderr)
    sys.exit(1)
'''


@dataclass(frozen=True)
class Artifact:
    name: str
    version: Version
    filename: str
    url: str
    size: int
    sha: str
    kind: str

    def verification(self):
        return self.filename, self.size, self.sha


def remote_command(socket, operation, *args):
    command = shlex.join([
        REMOTE_PYTHON, "-I", "-c", REMOTE_CODE, REMOTE_ROOT, LOCK_SHA256,
        operation, *(str(arg) for arg in args),
    ])
    return [*ssh_options(socket), TARGET, command]


def receipt(code, stdout, stderr):
    if code:
        lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = lines[-1] if lines else f"SSH exit {code}"
        raise RuntimeError(detail)
    try:
        return json.loads(stdout)
    except ValueError:
        raise RuntimeError("invalid SSH receipt") from None


def remote_read(socket, deadline, operation, data=b""):
    result = subprocess.run(remote_command(socket, operation), input=data, capture_output=True,
                            timeout=min(90, remaining(deadline)), check=False)
    return receipt(result.returncode, result.stdout, result.stderr)


def parse_lock(content):
    if hashlib.sha256(content.encode()).hexdigest() != LOCK_SHA256:
        raise RuntimeError("Renderer lock receipt hash mismatch")
    result = {}
    for line in content.splitlines():
        line = line.partition("#")[0].strip()
        if not line or line in ("--index-url https://pypi.org/simple",
                                "--extra-index-url https://download.pytorch.org/whl/cu128"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        specs = list(requirement.specifier)
        if (requirement.extras or requirement.marker or requirement.url or len(specs) != 1
                or specs[0].operator != "==" or "*" in specs[0].version or name in result):
            raise RuntimeError("unsupported Renderer lock entry")
        result[name] = Version(specs[0].version)
    if len(result) != 117:
        raise RuntimeError(f"unexpected Renderer lock count: {len(result)}")
    return result


def linux_tags(glibc):
    major, minor = glibc
    if major != 2 or minor < 5:
        raise RuntimeError("unexpected GNU libc version")
    legacy = {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}
    platforms = []
    for version in range(minor, 4, -1):
        platforms.append(f"manylinux_2_{version}_x86_64")
        if version in legacy:
            platforms.append(f"{legacy[version]}_x86_64")
    tags = [*cpython_tags((3, 10), ["cp310"], platforms),
            *compatible_tags((3, 10), interpreter="cp310", platforms=platforms)]
    return {tag: priority for priority, tag in enumerate(tags)}


def official_url(url, hostname):
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and parsed.netloc == hostname
            and not parsed.query and not parsed.fragment)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def open_pypi(url, hostname, deadline):
    if not official_url(url, hostname):
        raise RuntimeError("unapproved PyPI origin")
    request = Request(url, headers={"Accept-Encoding": "identity"})
    try:
        response = build_opener(NoRedirect()).open(request, timeout=min(18, remaining(deadline)))
    except HTTPError as error:
        code = error.code
        error.close()
        raise RuntimeError(f"{hostname} HTTP {code}") from None
    except URLError:
        raise RuntimeError(f"{hostname} connection failed") from None
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    if response.status != 200 or encoding != "identity":
        response.close()
        raise RuntimeError(f"{hostname} returned unexpected status or encoding")
    return response


def resolve(name, version, tags, python_version, deadline):
    url = f"https://pypi.org/pypi/{quote(name, safe='')}/{quote(str(version), safe='')}/json"
    with open_pypi(url, "pypi.org", deadline) as response:
        content = response.read(4 * 1024 * 1024 + 1)
    if len(content) > 4 * 1024 * 1024:
        raise RuntimeError("PyPI version metadata exceeds size limit")
    metadata = json.loads(content)
    if (canonicalize_name(metadata["info"]["name"]) != name
            or Version(metadata["info"]["version"]) != version):
        raise RuntimeError("PyPI release does not match the lock")

    candidates = []
    for entry in metadata["urls"]:
        filename = entry["filename"]
        kind = entry["packagetype"]
        if not FILENAME.fullmatch(filename) or kind not in ("bdist_wheel", "sdist"):
            continue
        if entry.get("requires_python") and not SpecifierSet(entry["requires_python"]).contains(python_version):
            continue
        if kind == "bdist_wheel":
            package, artifact_version, _, wheel_tags = parse_wheel_filename(filename)
            rank = min((tags[tag] for tag in wheel_tags if tag in tags), default=None)
            if rank is None:
                continue
        else:
            package, artifact_version = parse_sdist_filename(filename)
            rank = len(tags)
        if package != name or artifact_version != version:
            continue
        source = entry["url"]
        size = entry["size"]
        sha = entry["digests"]["sha256"]
        if (not official_url(source, "files.pythonhosted.org")
                or unquote(urlsplit(source).path.rsplit("/", 1)[-1]) != filename
                or not SHA256.fullmatch(sha) or not isinstance(size, int) or size <= 0):
            raise RuntimeError("invalid official artifact metadata")
        artifact = Artifact(name, version, filename, source, size, sha, kind)
        # A pinned release may have a yanked file. Prefer non-yanked files within its type.
        candidates.append(((kind == "sdist", bool(entry.get("yanked")), rank, filename), artifact))
    if not candidates:
        raise RuntimeError("no compatible wheel or source archive for exact lock version")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def send_block(fd, block, deadline):
    unsent = memoryview(block)
    while unsent:
        writable = select.select([], [fd], [], min(0.5, remaining(deadline)))[1]
        if writable:
            try:
                unsent = unsent[os.write(fd, unsent):]
            except BlockingIOError:
                pass


def stage(artifact, socket, deadline):
    with open_pypi(artifact.url, "files.pythonhosted.org", deadline) as response:
        length = response.headers.get("Content-Length")
        if length is not None and length != str(artifact.size):
            raise RuntimeError("HTTP size disagrees with PyPI metadata")
        command = remote_command(socket, "stage", *artifact.verification())
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            try:
                fd = process.stdin.fileno()
                os.set_blocking(fd, False)
                digest = hashlib.sha256()
                count = 0
                while count < artifact.size:
                    remaining(deadline)
                    block = response.read1(min(BLOCK, artifact.size - count))
                    if not block:
                        break
                    digest.update(block)
                    count += len(block)
                    send_block(fd, block, deadline)
                if count != artifact.size or response.read1(1) or digest.hexdigest() != artifact.sha:
                    raise RuntimeError("local artifact size or SHA-256 mismatch")
                process.stdin.close()
                process.stdin = None
                stdout, stderr = process.communicate(timeout=min(25, remaining(deadline)))
                expected = {"bytes": artifact.size, "sha256": artifact.sha}
                if receipt(process.returncode, stdout, stderr) != expected:
                    raise RuntimeError("unexpected remote artifact receipt")
            except BaseException:
                if process.poll() is None:
                    process.kill()
                raise


def run(args):
    deadline = time.monotonic() + args.deadline_seconds
    control = subprocess.run([*ssh_options(args.socket), "-O", "check", TARGET],
                             capture_output=True, timeout=5, check=False)
    if control.returncode:
        raise RuntimeError("A800 SSH control master is unavailable")
    inventory = remote_read(args.socket, deadline, "inventory")
    lock = parse_lock(inventory["lock"])
    cached = {(canonicalize_name(name), Version(version)) for name, version in inventory["packages"]}
    missing = {name: version for name, version in lock.items() if (name, version) not in cached}
    tags = linux_tags(inventory["glibc"])
    glibc = ".".join(map(str, inventory["glibc"]))
    print(f"Lock SHA-256 {LOCK_SHA256}; CPython {inventory['python']} Linux x86_64 glibc {glibc}", flush=True)
    print(f"{len(lock) - len(missing)}/{len(lock)} exact versions in cloud metadata; {len(missing)} need wheelhouse checks", flush=True)
    if args.inventory_only or not missing:
        return True

    artifacts = []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metadata = {pool.submit(resolve, name, version, tags, inventory["python"], deadline): name
                    for name, version in missing.items()}
        for future in as_completed(metadata):
            try:
                artifacts.append(future.result())
            except (RuntimeError, OSError, ValueError) as error:
                errors += 1
                print(f"{metadata[future]}: metadata failed: {error}", file=sys.stderr, flush=True)

        artifacts.sort(key=lambda artifact: artifact.name)
        payload = json.dumps([artifact.verification() for artifact in artifacts]).encode()
        statuses = remote_read(args.socket, deadline, "check", payload)
        downloads = []
        for artifact in artifacts:
            status = statuses[artifact.filename]
            if artifact.kind == "sdist":
                print(f"Offline build required: {artifact.filename}", flush=True)
            if status == "verified":
                print(f"Verified existing: {artifact.filename}", flush=True)
            elif status == "missing":
                downloads.append(artifact)
            else:
                errors += 1
                print(f"{artifact.filename}: {status}", file=sys.stderr, flush=True)
        print(f"{len(downloads)} files to relay, {sum(artifact.size for artifact in downloads)} bytes", flush=True)
        if args.plan_only:
            for artifact in downloads:
                print(f"Pending: {artifact.filename} {artifact.sha}", flush=True)
            return not errors

        uploads = {pool.submit(stage, artifact, args.socket, deadline): artifact for artifact in downloads}
        for future in as_completed(uploads):
            artifact = uploads[future]
            try:
                future.result()
                print(f"Staged and verified: {artifact.filename} {artifact.size} {artifact.sha}", flush=True)
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
                errors += 1
                print(f"{artifact.filename}: relay failed: {error}", file=sys.stderr, flush=True)
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inventory-only", action="store_true")
    modes.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 2700:
        parser.error("--deadline-seconds must be between 1 and 2700")
    try:
        return 0 if run(args) else 1
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Relay stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Relay interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
