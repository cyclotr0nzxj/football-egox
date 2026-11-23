"""Adapt the executed 004 foreground bundle into one B candidate for the new 008."""

import argparse
import ast
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "pair05-attention-floor-004"))
from build_floor_bundle import BUNDLE, LAUNCHER, baseline, checksum_manifest, digest, replace_once


INF = "staged/infer.py"
CORE = "staged/core/finetune/models/wan_i2v/custom_transformer.py"
PIPE = "staged/core/finetune/models/wan_i2v/sft_trainer.py"
STAGER = "stage_foreground_008.py"
LAUNCH = "run_pair05_clip_b.sh"


def patch_infer(code):
    code = replace_once(code, '    pipe.to("cuda")', '''    if foreground_routes is None or take_names != ["pair05_s0024"]:
        raise ValueError("This candidate requires the single frozen Pair05 foreground row.")
    from pair05_clip_gate import ImageGate, attach
    image_gate = ImageGate()
    attach(pipe, image_gate)

    pipe.to("cuda")''')
    return replace_once(
        code, "    if foreground_routes is not None:\n        if foreground_routes.attention_calls == 0:",
        "    image_gate.finish(foreground_routes.attention_calls)\n"
        "    if foreground_routes is not None:\n        if foreground_routes.attention_calls == 0:",
    )


def patch_arm_package(original):
    changed = {
        INF: patch_infer(original[INF].decode()),
        CORE: replace_once(
            original[CORE].decode(),
            "        if hidden_states_img is not None:\n            hidden_states = hidden_states + hidden_states_img",
            """        if hidden_states_img is not None:
            image_gate = getattr(self, "_pair05_clip_gate", None)
            hidden_states = (hidden_states + hidden_states_img if image_gate is None
                             else image_gate.combine(hidden_states, hidden_states_img))""",
        ),
        PIPE: replace_once(
            original[PIPE].decode(),
            "        # 6. Denoising loop",
            """        image_gate = getattr(self, "_pair05_clip_gate", None)
        if image_gate is not None:
            image_gate.record_initial(latents=latents, condition=condition, image=image_embeds,
                                      prompt=prompt_embeds, negative_prompt=negative_prompt_embeds,
                                      timesteps=timesteps, geometry=point_vecs_per_frame, rays=cam_rays)

        # 6. Denoising loop""",
        ),
    }
    gate = (HERE / "pair05_clip_gate.py").read_bytes()
    stage = original[STAGER].decode()
    for path in (INF, CORE):
        stage = replace_once(stage, digest(original[path]), digest(changed[path].encode()))
    stage = replace_once(stage, "PATCHED = {", f"""PATCHED = {{
    Path("{PIPE.removeprefix('staged/')}"): "{digest(changed[PIPE].encode())}",
    Path("pair05_clip_gate.py"): "{digest(gate)}",""")
    launch = replace_once(original[LAUNCHER].decode(),
                          "pair05-gga-foreground-results-004-retry1", "pair05-clip-b-results-008-66fc463f")
    launch = replace_once(launch, "d293479255-5411ceb1", "cb894da57f-66fc463f")
    launch = replace_once(launch, "authorized AutoDL 004", "authorized new AutoDL 008")
    launch = replace_once(launch, "run_timeout=3600", "run_timeout=3000")
    launch = replace_once(launch, "verify_foreground_008.py", "verify_clip_b.py")
    launch = replace_once(launch, 'printf \'%s\\n\' "$cuda"', '''printf '%s\\n' "$cuda"
clip_cuda=$(cd -- "$stage" && timeout --signal=TERM --kill-after=5s 60s "$python" -B "$bundle/clip_gate_smoke.py" --device cuda)
[[ $clip_cuda == 'PAIR05_CLIP_PROBE_OK cuda' ]] || fail 'Image gate CUDA smoke test failed.'
printf '%s\\n' "$clip_cuda"''')
    launch = replace_once(launch, 'sha256sum infer.py aria_gga.py pair05_gga_routes.py',
                          'sha256sum infer.py aria_gga.py pair05_gga_routes.py pair05_clip_gate.py')
    launch = replace_once(launch, '"$projection" "$cuda" "$preflight_marker"',
                          '"$projection" "$cuda" "$clip_cuda" "$preflight_marker"')
    launch = launch.replace("PAIR05_FOREGROUND_PREFLIGHT_OK", "PAIR05_CLIP_B_PREFLIGHT_OK")
    launch = launch.replace("PAIR05_FOREGROUND_STARTED", "PAIR05_CLIP_B_STARTED")
    launch = launch.replace("PAIR05_FOREGROUND_COMPLETE", "PAIR05_CLIP_B_COMPLETE")
    result = {name: data for name, data in original.items() if name != LAUNCHER}
    result.update({name: contents.encode() for name, contents in changed.items()})
    result.update({STAGER: stage.encode(), LAUNCH: launch.encode(),
                   "staged/pair05_clip_gate.py": gate, "pair05_clip_gate.py": gate,
                   "verify_clip_b.py": (HERE / "verify_clip_b.py").read_bytes(),
                   "clip_gate_smoke.py": (HERE / "clip_gate_smoke.py").read_bytes()})
    for name, contents in result.items():
        if name.endswith(".py"):
            ast.parse(contents, filename=name, feature_version=(3, 10))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    members = patch_arm_package(baseline())
    members["SHA256SUMS"] = checksum_manifest(members)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for name, contents in sorted(members.items()):
            archive.writestr(f"{BUNDLE}/{name}", contents)
    print(json.dumps({"sha256": digest(args.output.read_bytes()), "members": len(members)}))
