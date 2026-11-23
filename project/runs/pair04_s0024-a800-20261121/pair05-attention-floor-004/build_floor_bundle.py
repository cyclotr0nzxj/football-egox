"""Prepare an isolated Pair05 floor candidate from the checked 004 launch archive."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from verify_floor_diagnostics import EXPECTED_CALLS, TARGETS


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair01-gga-ab"))
from prepare_bundle import PROJECT, sha256, stage_local_media  # noqa: E402


TAKE = "pair05_s0024"
SOURCE = RUN / "clone-004-20261122/pair05-device-fix/pair05-gga-foreground-bundle-004-retry1.zip"
SOURCE_SHA = "61f6b4cae58a6e2c06a0b5fcc6d9e91b7d09b4a66aaf16d4e27d0497945e17ad"
BUNDLE = "pair05-gga-foreground-bundle"
LAUNCHER = "run_pair05_gga_foreground_008.sh"
ROUTE = "staged/pair05_gga_routes.py"
CUDA_CHECK = "check_foreground_cuda.py"
STAGE_CHECK = "stage_foreground_008.py"
RECEIPT_CHECK = "verify_foreground_008.py"
METRICS_CHECK = "verify_floor_diagnostics.py"
PLAN = "plan/pair05-gga-routes.npz"
DEFAULT_TORCH = Path("/usr/local/Caskroom/miniforge/base/envs/dino_wm_local/bin/python")
LEGACY_ROUTE_SHA = "21ad9180713726ab1469d63db27b464731b8e76fea8f4f99a28be2e6e068d19a"
INSTANCE_HOST = {"004": "d293479255-5411ceb1", "008": "cb894da57f-45e726a1"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one adaptation point for {old!r}.")
    return source.replace(old, new)


def baseline():
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("The actual 004 source archive differs from the frozen hash.")
    with ZipFile(SOURCE) as archive:
        names = archive.namelist()
        if (archive.testzip() is not None or len(names) != len(set(names))
                or not all(name.startswith(f"{BUNDLE}/") and ".." not in PurePosixPath(name).parts
                           for name in names)):
            raise ValueError("Unexpected 004 archive contents.")
        members = {name.removeprefix(f"{BUNDLE}/"): archive.read(name) for name in names}
    manifest = members.pop("SHA256SUMS")
    if manifest != checksum_manifest(members):
        raise ValueError("The frozen 004 archive checksum manifest failed.")
    return members


def checksum_manifest(members):
    return "".join(f"{digest(contents)}  {relative}\n" for relative, contents in sorted(members.items())).encode()


def candidate(instance):
    original = baseline()
    new_route = (HERE / ROUTE).read_bytes()
    if new_route == original[ROUTE] or b"PAIR05_FLOOR_SAMPLE " not in new_route:
        raise ValueError("No instrumented replacement attention code found.")
    out_name = f"pair05-gga-foreground-results-{instance}-floor25"
    launcher = replace_once(original[LAUNCHER].decode(),
                            "pair05-gga-foreground-results-004-retry1", out_name)
    if instance != "004":
        launcher = replace_once(launcher, INSTANCE_HOST["004"], INSTANCE_HOST[instance])
        launcher = replace_once(launcher, "authorized AutoDL 004", f"authorized AutoDL {instance}")
    launcher = replace_once(
        launcher, "[[ $cuda == 'PAIR05_GGA_CUDA_OK '* ]] || fail 'Missing foreground CUDA preflight evidence.'",
        "[[ ${cuda##*$'\\n'} == 'PAIR05_GGA_FLOOR_CUDA_OK '* ]] || fail 'Missing floor CUDA preflight evidence.'",
    )
    cuda = replace_once(
        original[CUDA_CHECK].decode(), "import torch\n",
        "import torch\n    sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
        "    from verify_floor_diagnostics import FLOOR, verify_samples\n",
    )
    cuda = replace_once(cuda, "import argparse\n", "import argparse\nimport contextlib\nimport io\n")
    cuda = replace_once(
        cuda, "        out = routes.apply(ego_output, ego_query, key, value, base, exo.cpu())",
        """        telemetry = io.StringIO()
        with contextlib.redirect_stdout(telemetry):
            out = routes.apply(ego_output, ego_query, key, value, base, exo.cpu())
        measurement = verify_samples(telemetry.getvalue().encode(), expected_calls=(1,))""",
    )
    cuda = replace_once(cuda, "        untouched = torch.ones", """        measured = selected[..., 1].float()
        retained = selected[..., 2].float()
        if ((measured - FLOOR).abs().max().item() > .005
                or (retained - (1 - FLOOR)).abs().max().item() > .006):
            raise ValueError("Pair05 CUDA could not achieve the selected attention-mass floor.")
        untouched = torch.ones""")
    cuda = replace_once(cuda, 'print("PAIR05_GGA_CUDA_OK "',
                        'print("PAIR05_GGA_FLOOR_CUDA_OK "')
    cuda = replace_once(
        cuda, '"allocated_peak_bytes": torch.cuda.max_memory_allocated(device),',
        '''"allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                                              "observed_selected_floor_min": measured.min().item(),
                                              "observed_selected_floor_max": measured.max().item(),
                                              "reported_branch_minimum": measurement["samples"][0]["minimum_after"],''',
    )
    receipt = replace_once(original[RECEIPT_CHECK].decode(), "from stage_foreground_008 import FROZEN",
                           "from stage_foreground_008 import FROZEN\nfrom verify_floor_diagnostics import EXPECTED_CALLS, verify_samples")
    receipt = replace_once(
        receipt, '    return {"attention_calls": int(markers[0][1]), "plan_sha256": FROZEN[PLAN],',
        '''    count = int(markers[0][1])
    if count != EXPECTED_CALLS[-1]:
        raise ValueError(f"Expected {EXPECTED_CALLS[-1]} foreground attention calls, received {count}.")
    diagnostics = verify_samples(payload)
    return {"attention_calls": count, "plan_sha256": FROZEN[PLAN], "floor_diagnostics": diagnostics,''',
    )
    stage = replace_once(original[STAGE_CHECK].decode(), LEGACY_ROUTE_SHA, digest(new_route))
    changed = {
        LAUNCHER: launcher.encode(), CUDA_CHECK: cuda.encode(), RECEIPT_CHECK: receipt.encode(),
        STAGE_CHECK: stage.encode(), ROUTE: new_route, METRICS_CHECK: (HERE / METRICS_CHECK).read_bytes(),
    }
    if any(original.get(name) == contents for name, contents in changed.items()):
        raise ValueError("An expected floor candidate adaptation was not made.")
    return original | changed, changed, out_name


def checked(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=120)
    return result.stdout.strip()


def sample_log():
    group = {"before_min": .001, "before_mean": .01, "before_max": .1,
             "after_min": .25, "after_mean": .25, "after_max": .25,
             "exo_output_rms": .05, "routed_output_rms": .0125}
    payload = b"".join(("PAIR05_FLOOR_SAMPLE " + json.dumps({
        "call": count, "floor": .25, "heads": 40, "branches": 53,
        "minimum_after": .25, "maximum_before": .1, "heads_branches_raised": 40 * 53,
        "by_target": {key: group for key in sorted(TARGETS)},
    }) + "\n").encode() for count in EXPECTED_CALLS)
    return payload + b"Pair05 foreground GGA attention calls: 4000\n"


def verify_local(work, package, torch_python):
    official = PROJECT / "vendor/EgoX"
    verify_code = runpy.run_path(str(package / "check_runtime_008.py"))["verify_code"]
    verify_code(official, PROJECT / "scripts/instrumented_infer.py")
    receipt, = json.loads((package / "validated-inputs.json").read_bytes())
    media = stage_local_media(work, receipt["depth_sha256"], take=TAKE)
    inputs = json.loads(checked([sys.executable, "-I", "-B", str(package / "check_inputs.py"),
                                "--media-root", str(media)]))
    projection = json.loads(checked([sys.executable, "-B", str(package / "prepare_infer.py"),
                                    "--official-infer", str(official / "infer.py"),
                                    "--meta", str(package / "meta.json"), "--check-only"]))
    stage = work / "stage"
    stage_check = checked([sys.executable, "-B", str(package / STAGE_CHECK), "--official", str(official),
                           "--media-root", str(media), "--stage", str(stage)])
    structure = checked([str(torch_python), "-I", "-B", str(package / CUDA_CHECK), "--check-structure",
                         "--stage", str(stage), "--plan", str(stage / PLAN), "--meta", str(stage / "meta.json")],
                        cwd=stage)
    checked(["bash", "-n", str(package / LAUNCHER)])

    log = work / "attn-fixture.log"
    test_runner = """import sys
from pathlib import Path
from verify_foreground_008 import verify_log
try:
    result = verify_log(Path(sys.argv[1]))
except ValueError:
    if sys.argv[2] == "pass":
        raise
else:
    assert sys.argv[2] == "pass" and result["attention_calls"] == 4000
"""
    valid = sample_log()
    for payload, target in ((valid, "pass"), (valid.split(b"\n", 1)[1], "reject"),
                            (valid.replace(b'4000\n', b'3999\n'), "reject"),
                            (valid.replace(b'"minimum_after": 0.25', b'"minimum_after": 0.01'), "reject")):
        log.write_bytes(payload)
        checked([sys.executable, "-B", "-c", test_runner, str(log), target], cwd=package)
    return {"inputs": inputs, "projection": projection, "stage": stage_check, "structure": structure,
            "telemetry_fixtures": ["valid", "missing_sample_refused", "wrong_count_refused", "low_mass_refused"]}


def build(output, torch_python, instance):
    if output.exists():
        raise FileExistsError(output)
    members, changed, out_name = candidate(instance)
    members["SHA256SUMS"] = checksum_manifest(members)
    with tempfile.TemporaryDirectory(prefix="pair05-attention-floor-") as directory:
        work = Path(directory)
        package = work / BUNDLE
        for name, payload in members.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            if name.endswith(".py"):
                compile(payload, name, "exec")
        local = verify_local(work, package, torch_python)
        with ZipFile(output, mode="x") as archive:
            for name, payload in sorted(members.items()):
                archive.writestr(ZipInfo(f"{BUNDLE}/{name}"), payload, compress_type=ZIP_DEFLATED)
        with ZipFile(output) as archive:
            restored = {name.removeprefix(f"{BUNDLE}/"): archive.read(name) for name in archive.namelist()}
            if archive.testzip() is not None or restored != members:
                raise ValueError("The new archive did not round-trip exactly.")
    return {"filename": output.name, "sha256": sha256(output), "size_bytes": output.stat().st_size,
            "source_sha256": SOURCE_SHA, "member_count": len(members),
            "manifest_sha256": digest(members["SHA256SUMS"]),
            "adaptations": {name: digest(payload) for name, payload in changed.items()},
            "remote_output_directory": out_name, "inference_timeout_seconds": 3600,
            "local": local, "real_cuda_preflight_executed": False, "inference_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=tuple(INSTANCE_HOST), default="004")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-python", type=Path, default=DEFAULT_TORCH)
    args = parser.parse_args()
    output = args.output or HERE / f"pair05-gga-attention-floor-{args.instance}.zip"
    print(json.dumps(build(output, args.torch_python, args.instance), indent=2))


if __name__ == "__main__":
    main()
