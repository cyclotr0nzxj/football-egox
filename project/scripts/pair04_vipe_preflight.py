"""Read-only check and command planner for the Pair04 geometry environment."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


RENDERER_COMMIT = "17cdef688885c3ac6a4c371a2d620391cc28dab4"
EXO_PATH = Path("example/self_capture/videos/pair04_s0024/exo.mp4")


def git_stdout(renderer, *args):
    if shutil.which("git") and renderer.is_dir():
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(renderer), *args],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    renderer = args.renderer.resolve()
    exo = args.input_root.absolute() / EXO_PATH
    out_dir = args.run_dir.absolute()
    validator = Path(__file__).with_name("pair04_cpu_dry_run.py")
    cpu = subprocess.run(
        [sys.executable, "-B", str(validator), "--input-root", str(args.input_root)],
        capture_output=True, text=True,
    )
    root = git_stdout(renderer, "rev-parse", "--show-toplevel")
    revision = git_stdout(renderer, "rev-parse", "HEAD") if root == str(renderer) else None
    vipe_spec = importlib.util.find_spec("vipe")
    module_at_pin = bool(
        vipe_spec and vipe_spec.origin
        and Path(vipe_spec.origin).resolve().is_relative_to(renderer / "vipe")
    )
    vipe = shutil.which("vipe")
    nvidia = shutil.which("nvidia-smi")
    gpu_visible = bool(nvidia and subprocess.run(
        [nvidia, "-L"], capture_output=True, text=True,
    ).returncode == 0)
    checks = {
        "exo_cpu_dry_run": cpu.returncode == 0,
        "new_run_dir": not os.path.lexists(out_dir) and out_dir.parent.is_dir(),
        "renderer_commit": revision == RENDERER_COMMIT,
        "renderer_tracked_files_clean": revision == RENDERER_COMMIT and
        git_stdout(renderer, "status", "--porcelain", "--untracked-files=no") == "",
        "python_3_10": sys.version_info[:2] == (3, 10),
        "torch_and_torchvision": version("torch") == "2.7.0+cu128" and
        version("torchvision") == "0.22.0+cu128",
        "pytorch3d_0_7_9": version("pytorch3d") == "0.7.9",
        "vipe_cli_from_active_env": bool(vipe and
                                       Path(vipe).absolute().parent == Path(sys.executable).absolute().parent),
        "vipe_import_from_pinned_checkout": module_at_pin,
        "nvidia_smi_visible": gpu_visible,
    }
    command = [
        vipe or "vipe", "infer", str(exo), "--output", str(out_dir / "vipe_results"),
        "--start_frame", "0", "--end_frame", "48",
        "--assume_fixed_camera_pose", "--pipeline", "lyra",
    ]
    ready = all(checks.values())
    print(json.dumps({
        "status": "environment_checks_passed" if ready else "blocked",
        "checks": checks,
        "remaining_manual_checks": ["CUDA import and runtime compatibility", "MoGe pin", "ViPE checkpoint cache"],
        "vipe_command_after_all_checks": shlex.join(command),
        "geometry_started": False,
    }, indent=2))
    if not ready:
        print(cpu.stderr, file=sys.stderr, end="")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
