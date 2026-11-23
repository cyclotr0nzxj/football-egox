"""Exercise the extracted launcher with local files and explicit GPU/inference stand-ins."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from prepare_foreground_bundle import (
    HERE, LAUNCHER, PACKAGE, PROJECT, RUN, TAKE, sha256, stage_local_media,
)


SHIM = """
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

args = sys.argv[1:]
script_index = next(i for i, arg in enumerate(args) if arg.endswith('.py'))
script = Path(args[script_index])
params = args[script_index + 1:]
fixture = Path(os.environ['PAIR05_TEST_FIXTURE'])
mode = os.environ['PAIR05_TEST_MODE']
if script.name == 'check_runtime_008.py':
    runpy.run_path(str(script))['verify_code'](Path(params[1]), Path(params[0]) / 'scripts/instrumented_infer.py')
    print('EGOX_RUNTIME_FINGERPRINT ' + 'a' * 64)
elif script.name == 'check_foreground_cuda.py':
    assert Path.cwd() == Path(params[params.index('--stage') + 1])
    assert Path(params[params.index('--plan') + 1]).is_file()
    print('PAIR05_GGA_CUDA_OK {"local_control_flow_fixture":true}')
elif script.name == 'instrumented_infer.py':
    out = Path(params[params.index('--out') + 1])
    def option(name):
        return params[params.index(name) + 1]
    assert '--use_GGA' in params and '--do_kv_cache' not in params
    assert option('--seed') == '846514' and option('--lora_rank') == '256'
    assert option('--cos_sim_scaling_factor') == '3.0' and '--in_the_wild' in params
    assert option('--idx') == '-1' and option('--meta_data_file') == './meta.json'
    plan = Path(option('--foreground_gga_plan'))
    assert plan.is_absolute() and plan == Path.cwd() / 'plan/pair05-gga-routes.npz'
    assert Path.cwd() == out / 'work' and len(json.loads(Path('meta.json').read_text())['test_datasets']) == 1
    assert os.environ['HF_HUB_OFFLINE'] == '1' and os.environ['EGOX_AUDIT_PATH'] == str(out / 'runtime_audit.json')
    with (fixture / 'inference-count.txt').open('a') as log:
        log.write('called\\n')
    shutil.copyfile(fixture / 'pair05_s0024.mp4', out / 'pair05_s0024.mp4')
    shutil.copyfile(fixture / 'runtime_audit.json', out / 'runtime_audit.json')
    print('Pair05 foreground GGA attention calls: ' + ('7' if mode == 'positive' else '0'))
else:
    os.execv(sys.executable, [sys.executable] + args)
"""


def write_executable(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o700)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Fixture launcher does not have the expected anchor: {old}")
    return text.replace(old, new, 1)


def seal(package):
    paths = sorted(path for path in package.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (package / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(package).as_posix()}\n" for path in paths))


def verify(bundle):
    with tempfile.TemporaryDirectory(prefix="pair05-launch-fixture-") as directory:
        work = Path(directory).resolve()
        with ZipFile(bundle) as source:
            if not all(name.startswith(f"{PACKAGE}/") and ".." not in Path(name).parts
                       for name in source.namelist()):
                raise ValueError("Unexpected path in the local archive fixture.")
            source.extractall(work)
        package = work / PACKAGE
        launcher = package / LAUNCHER
        base = work / "simulated-008"
        history = RUN / "pair05-aria-008/output-008-20261122"
        shutil.copyfile(history / "raw" / f"{TAKE}.mp4", work / f"{TAKE}.mp4")
        shutil.copyfile(history / "logs/runtime_audit.json", work / "runtime_audit.json")
        fixtures = work / "bin"
        for name, body in {
            "hostname": "printf 'autodl-container-cb894da57f-45e726a1\\n'",
            "findmnt": "exit 0", "flock": "exit 0",
            "timeout": 'shift 3\nexec "$@"',
            "date": "printf '2000-01-01T00:00:00+0000\\n'",
            "nvidia-smi": """case "$3" in
    --query-gpu=name*) printf 'NVIDIA A800 80GB PCIe, 81920, 0\\n' ;;
    --query-gpu=driver*) printf 'local-fixture\\n' ;;
    --query-compute-apps=pid) exit 0 ;;
    *) exit 4 ;;
esac""",
            "sha256sum": f'if [ "${{2:-}}" = "--strict" ]; then set -- "$1" "$3"; fi\n'
                         f'exec {shlex.quote(shutil.which("sha256sum"))} "$@"',
        }.items():
            write_executable(fixtures / name, f"#!/bin/sh\n{body}\n")
        run_root = base / "egox-shared-008"
        stock = run_root / "inference-update"
        stock.mkdir(parents=True)
        (run_root / ".egox-inference.lock").touch()
        (run_root / "scripts").mkdir()
        wrapper = run_root / "scripts/instrumented_infer.py"
        shutil.copyfile(PROJECT / "scripts/instrumented_infer.py", wrapper)
        write_executable(run_root / "egox-env/bin/python", f"#!{sys.executable}\n{SHIM}")
        (base / "renderer-env/bin").mkdir(parents=True)
        for name in ("ffmpeg", "ffprobe"):
            (base / "renderer-env/bin" / name).symlink_to(shutil.which(name))
        receipt, = json.loads((package / "validated-inputs.json").read_bytes())
        media = stage_local_media(work, receipt["depth_sha256"], take=TAKE)
        (stock / "example").symlink_to(media / "example", target_is_directory=True)
        official = PROJECT / "vendor/EgoX"
        (stock / "core").symlink_to(official / "core", target_is_directory=True)
        (stock / "infer.py").symlink_to(official / "infer.py")
        output = run_root / "pair05-gga-foreground-results"
        contents = replace_once(launcher.read_text(), "base=/root/autodl-tmp/football_egox/pair04-20261121",
                                f"base={shlex.quote(str(base))}")
        contents = replace_once(contents, "weights=/root/autodl-fs/football_egox/models",
                                f"weights={shlex.quote(str(work / 'stand-in-for-cached-models'))}")
        contents = replace_once(contents, "/root/autodl-fs)", f"{shlex.quote(str(work))})")
        launcher.write_text(contents)
        seal(package)

        def invoke(action, *, mode="positive", expected):
            env = dict(os.environ, PATH=str(fixtures) + os.pathsep + os.environ["PATH"],
                       PAIR05_TEST_FIXTURE=str(work), PAIR05_TEST_MODE=mode)
            result = subprocess.run(["bash", str(launcher), action], env=env, capture_output=True,
                                    text=True, timeout=120)
            combined = result.stdout + result.stderr
            if (result.returncode == 0) != expected[0] or expected[1] not in combined:
                raise ValueError(f"Unexpected launcher fixture result {result.returncode}:\n{combined}")
            return combined

        invoke("preflight", expected=(True, "PAIR05_FOREGROUND_PREFLIGHT_OK"))
        if output.exists():
            raise ValueError("Preflight unexpectedly created an inference output directory.")
        diagnostic = invoke("run", expected=(True, "PAIR05_FOREGROUND_COMPLETE"))
        attempt, = output.glob("attempt-*")
        result_path = attempt / "verified-result.json"
        result = json.loads(result_path.read_bytes())
        if "PAIR05_FOREGROUND_STARTED" not in diagnostic or result["foreground_gga"]["attention_calls"] != 7:
            raise ValueError("The fixture did not write its synthetic positive hook count.")
        invoke("run", expected=(True, "PAIR05_FOREGROUND_COMPLETE"))
        result_path.unlink()
        invoke("run", expected=(True, "PAIR05_FOREGROUND_COMPLETE"))
        if (work / "inference-count.txt").read_text().splitlines() != ["called"]:
            raise ValueError("Rechecking or recovering a receipt reran the inference fixture.")
        if json.loads(result_path.read_bytes()) != result:
            raise ValueError("Recovered receipt differs from the previously verified attempt.")

        output.rename(work / "completed-fixture")
        invoke("run", mode="zero", expected=(False, "exactly one positive foreground GGA attention count"))
        attempt, = output.glob("attempt-*")
        if (attempt / "verified-result.json").exists():
            raise ValueError("A receipt was published for zero foreground attention calls.")
        invoke("run", expected=(False, "exactly one positive foreground GGA attention count"))
        if (work / "inference-count.txt").read_text().splitlines() != ["called", "called"]:
            raise ValueError("The unsuccessful attempt was silently rerun.")

        depth = media / f"example/self_capture/depth_maps/{TAKE}/00000.npy"
        depth_original = depth.read_bytes()
        depth.write_bytes(depth_original[:-1])
        invoke("preflight", expected=(False, "Frozen SHA-256 mismatch"))
        depth.write_bytes(depth_original)
        wrapper_original = wrapper.read_bytes()
        wrapper.write_bytes(wrapper_original + b"\n")
        invoke("preflight", expected=(False, "Expected official EgoX infer.py and its original observational wrapper"))
        wrapper.write_bytes(wrapper_original)
        plan = package / "plan/pair05-gga-routes.npz"
        plan.write_bytes(plan.read_bytes() + b"\n")
        seal(package)
        invoke("preflight", expected=(False, "Unexpected staged SHA-256: plan/pair05-gga-routes.npz"))
        return {"bundle_sha256": sha256(bundle), "stock_integrity_check": "real",
                "media_decode_and_sha256": "real", "gpu_checks": "stand-ins",
                "model_inference": "stand-in using historical video/audit fixture",
                "passed": ["preflight_without_output", "scoped_launch_and_output_verification",
                           "existing_receipt_recheck", "interrupted_receipt_recovery", "zero_hook_no_receipt_or_retry",
                           "mutated_depth_refused", "mutated_official_wrapper_refused", "resealed_plan_refused"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=HERE / f"{PACKAGE}.zip")
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle.resolve(strict=True)), indent=2))
