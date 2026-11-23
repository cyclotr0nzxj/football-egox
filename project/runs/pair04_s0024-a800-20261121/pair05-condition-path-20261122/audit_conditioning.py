"""Replay the executed B pipeline on CPU with instrumented synthetic tensors."""

import argparse
import ast
from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
B = RUN / "pair05-clip-b-20261122"
SUFFIX = "core/finetune/models/wan_i2v/sft_trainer.py"
PLAN = "plan/pair05-gga-routes.npz"


def package_sources():
    hashes = {path: digest for digest, path in (line.split(maxsplit=1) for line in
              (B / "result-008-66fc463f/executed-code-and-meta.sha256").read_text().splitlines())}
    with ZipFile(B / "pair05-clip-b-008.zip") as archive:
        def verified(package_name, relative):
            payload = archive.read("pair05-gga-foreground-bundle/" + package_name)
            assert sha256(payload).hexdigest() == hashes[relative], relative
            return payload

        pipeline = verified("staged/" + SUFFIX, SUFFIX)
        for path in ("infer.py", "pair05_clip_gate.py",
                     "core/finetune/models/wan_i2v/custom_transformer.py"):
            verified("staged/" + path, path)
        return ast.parse(pipeline), verified(PLAN, PLAN), hashes[SUFFIX]


def execute(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SUFFIX, "exec"), scope)


def assigned(node):
    if isinstance(node, ast.Assign):
        target, = node.targets
        return ast.unparse(target)
    return None


def extract(source):
    cls, = [node for node in source.body if isinstance(node, ast.ClassDef)
            and node.name == "WanWidthConcatImageToVideoPipeline"]
    prepare, = [n for n in cls.body if getattr(n, "name", None) == "prepare_latents"]
    pipeline, = [n for n in cls.body if getattr(n, "name", None) == "__call__"]
    prepare.decorator_list = []
    namespace = {name: globals()[name] for name in ("torch", "List", "Optional", "Tuple", "Union")}
    execute([prepare], namespace)
    loop, = [node for node in ast.walk(pipeline) if isinstance(node, ast.For)
             and ast.unparse(node.target) == "(i, t)"]
    step = [n for n in loop.body if assigned(n) in ("latent_model_input", "timestep", "noise_pred", "latents")
            or (isinstance(n, ast.If) and ast.unparse(n.test) == "self.do_classifier_free_guidance")
            or (assigned(n) or "").startswith("noise_pred[")]
    assert len(step) == 6
    final, = [n for n in pipeline.body if isinstance(n, ast.If)
              and ast.unparse(n.test) == "not output_type == 'latent'"]
    return namespace["prepare_latents"], step, final


def check_pipeline(source):
    prepare, step, final = extract(source)
    initial = torch.full((1, 16, 13, 56, 154), 11, dtype=torch.bfloat16)
    initial[..., 98:] = -3
    prior = torch.full((1, 16, 13, 56, 56), 7, dtype=torch.bfloat16)
    edited = prior.clone()
    edited[:, :, (3, 6), 22:26, 24:28] = 19
    frames = torch.zeros((1, 3, 49, 448, 448), dtype=torch.uint8)
    calls, decodes = [], []
    geometry, rays = object(), object()

    def transformer(**args):
        calls.append(args)
        return (torch.ones_like(initial),)

    def decode(latent, return_dict):
        assert return_dict is False
        width = latent.shape[-1]
        decodes.append(width)
        return (torch.full((1, 3, 49, 448, width * 8), 1 if width == 98 else 2, dtype=torch.uint8),)

    vae = SimpleNamespace(dtype=torch.bfloat16, config=SimpleNamespace(
        z_dim=16, latents_mean=[0.] * 16, latents_std=[1.] * 16), decode=decode)
    # This solver stub exercises the update mask. Its values are synthetic.
    scheduler = SimpleNamespace(step=lambda v, t, old, return_dict: (old - v.float() / 2,))
    pipe = SimpleNamespace(vae_scale_factor_temporal=4, vae_scale_factor_spatial=8,
                           _shared_latent_exo_randn=initial, vae=vae, transformer=transformer,
                           scheduler=scheduler, do_classifier_free_guidance=True,
                           video_processor=SimpleNamespace(postprocess_video=lambda v, output_type: v))
    for encoded in (prior, edited):
        pipe.encode_video = lambda rgb: encoded
        latents, condition, (left, right) = prepare(
            pipe, frames, 1, 16, 448, 784, 448, 49, torch.float32, torch.device("cpu"))
        assert (left, right) == (98, 56)
        assert list(condition.shape) == [1, 20, 13, 56, 154]
        assert torch.equal(latents, initial.float())
        assert torch.equal(condition[:, 4:, ..., 98:], encoded.float())
        assert torch.equal(condition[:, 4:, ..., :98], initial[..., :98].float())
        assert not condition[:, :4, ..., 98:].any()
        assert condition[:, :4, 0, :, :98].all()
        assert not condition[:, :4, 1:, :, :98].any()
        env = dict(self=pipe, torch=torch, latents=latents, condition=condition,
                   transformer_dtype=torch.bfloat16, t=torch.tensor(900.), i=0,
                   prompt_embeds="positive", negative_prompt_embeds="negative", image_embeds="image",
                   attention_kwargs=None, attention_GGA=None, attention_mask_GGA=None,
                   point_vecs_per_frame=geometry, cam_rays=rays, cos_sim_scaling_factor=3.0,
                   do_kv_cache=False, do_custom_test=False, save_attn_dir_name=None,
                   ego_latent_width=right, exo_latent_width=left, guidance_scale=5.0, output_type="np")
        execute(step, env)
        assert torch.equal(env["latents"][..., :98], initial[..., :98].float())
        assert torch.all(env["latents"][..., 98:] == -3.5)

    delta = torch.zeros_like(calls[0]["hidden_states"])
    delta[:, 20:, ..., 98:] = edited - prior
    for index, label in enumerate(("positive", "negative")):
        baseline, changed = calls[index], calls[index + 2]
        assert baseline["encoder_hidden_states"] == changed["encoder_hidden_states"] == label
        assert changed["point_vecs_per_frame"] is geometry and changed["cam_rays"] is rays
        assert torch.equal(changed["hidden_states"] - baseline["hidden_states"], delta)
    assert torch.equal(calls[0]["hidden_states"], calls[1]["hidden_states"])
    execute([final], env)
    assert decodes == [98, 56]
    assert torch.all(env["video"][..., :784] == 1) and torch.all(env["video"][..., 784:] == 2)
    return {"scope": "executed B tensor plumbing with mocked encoder, transformer, solver and decoder",
            "transformer_input_shape": list(delta.shape), "ego_prior_channels_inclusive": [20, 35],
            "ego_uses_columns_inclusive": [98, 153], "test_nonzero_elements": int(torch.count_nonzero(delta)),
            "identical_placement_in_both_cfg_branches": True, "decoder_widths": decodes}


def count_routes(payload, projection_path):
    with np.load(io.BytesIO(payload), allow_pickle=False) as plan:
        edges, samples, frames = (plan[name] for name in ("edges", "samples", "frames"))
    admitted = {(int(q), int(label)) for q, label, _, _ in edges}
    rows = {}
    with np.load(projection_path, allow_pickle=False) as projection:
        for frame in map(int, frames):
            labels = projection[f"F{frame:02}_target_label"]
            rows[str(frame)] = {}
            for kind, label in (("teammate", 1), ("ball", 2), ("disc_2", 4), ("disc_4", 6)):
                y, x = np.nonzero(labels == label)
                queries, counts = np.unique(((frame // 4) * 28 + y // 16) * 28 + x // 16,
                                            return_counts=True)
                chosen = (samples[:, 0] == frame) & (samples[:, 3] == label)
                rows[str(frame)][kind] = {
                    "geometry_label_pixels": len(x), "blocks": len(queries),
                    "blocks_with_same_identity_route": sum((int(q), label) in admitted for q in queries),
                    "route_samples": int(chosen.sum()),
                    "max_label_pixels_in_any_256pixel_block": int(counts.max()) if len(queries) else 0,
                }
    return {"routed_queries": len(np.unique(edges[:, 0])), "total_ego_queries": 13 * 28 * 28, "frames": rows}


def check_clean_prediction(source):
    training, = [node for node in source.body if getattr(node, "name", None) == "WanI2VSftTrainer"]
    flow_targets = [node for node in ast.walk(training)
                    if assigned(node) in ("noisy_latents", "target")
                    and isinstance(node.value, ast.BinOp)]
    assert len(flow_targets) == 2
    sigma = torch.tensor([.15, .5, .85], dtype=torch.float32)
    clean = torch.tensor([1., -1., 2.], dtype=torch.float32)
    noise = torch.tensor([-2., 3., 0.], dtype=torch.float32)
    env = dict(sigmas=sigma, latent=clean, noise=noise)
    execute(flow_targets, env)
    restored = env["noisy_latents"] - sigma * env["target"]
    assert torch.allclose(restored, clean)
    return {"scope": "training flow target checked with synthetic samples",
            "equation": "predicted_clean = noisy_ego - sigma * predicted_velocity",
            "sigmas_stressed": [.15, .5, .85],
            "max_absolute_error": float((restored - clean).abs().max())}


def run():
    source, plan, code_hash = package_sources()
    payload = (HERE / "wan-transformer-config.json").read_bytes()
    expected, = [row["sha256"] for row in json.loads(
        (RUN / "multiclip-evidence/model-verification-008.json").read_text())
        if row["path"].endswith("/transformer/config.json")]
    assert sha256(payload).hexdigest() == expected
    config = json.loads(payload)
    assert config["patch_size"] == [1, 2, 2] and config["in_channels"] == 36 and config["out_channels"] == 16
    assert 98 % config["patch_size"][2] == 0
    projection = RUN / "pair05-foreground-match-20261122/results/foreground-routing.npz"
    return {"executed_pipeline_sha256": code_hash, "patch_size": config["patch_size"],
            "exo_boundary_aligned_to_patch": True, "tensor_test": check_pipeline(source),
            "clean_prediction_math": check_clean_prediction(source),
            "gga_route_geometry": count_routes(plan, projection),
            "projection_file_sha256": sha256(projection.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    with arguments.output.open("x", encoding="utf-8") as file:
        json.dump(run(), file, indent=2)
        file.write("\n")
