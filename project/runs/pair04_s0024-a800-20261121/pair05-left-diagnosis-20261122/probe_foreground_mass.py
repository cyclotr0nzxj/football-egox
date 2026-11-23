"""Reproduce the Pair05 route-mask CPU null test with the exact executed sources.

Requires Python, numpy and PyTorch. No video frames or model weights are loaded.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F


def test_group_floor(bias, exo):
    """Verify a potential intervention on one attention head with synthetic Q/K."""
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(1, 1, bias.shape[1], 4, generator=generator)
    k = torch.randn(1, 1, bias.shape[2], 4, generator=generator)
    logits = (q @ k.transpose(-2, -1) / 2)[0, 0] + bias[0]
    allowed_exo = torch.isfinite(bias[0]) & exo
    # Make one branch already favor EXO to test that the floor leaves it alone.
    bias = bias.clone()
    bias[0, 0, allowed_exo[0]] += 12
    logits[0, allowed_exo[0]] += 12
    exo_log_z = torch.logsumexp(logits.masked_fill(~exo, -torch.inf), dim=-1)
    ego_log_z = torch.logsumexp(logits.masked_fill(exo, -torch.inf), dim=-1)
    current = torch.sigmoid(exo_log_z - ego_log_z)
    floor = torch.tensor(0.25)
    offset = (torch.logit(floor) - (exo_log_z - ego_log_z)).clamp_min(0)
    adjusted = torch.where(allowed_exo, bias[0] + offset[:, None], bias[0])
    marker_values = exo.float().view(1, 1, -1, 1)
    mass = F.scaled_dot_product_attention(q, k, marker_values, attn_mask=adjusted).flatten()
    torch.testing.assert_close(mass, torch.maximum(floor, current), rtol=1e-5, atol=1e-6)
    assert torch.equal(torch.isfinite(adjusted), torch.isfinite(bias[0]))
    assert torch.equal(adjusted[:, ~exo], bias[0, :, ~exo])
    assert current[0] > floor and offset[0] == 0 and (offset[1:] > 0).all()
    return {
        "mode": "synthetic_nonuniform_qk_only",
        "example_target_minimum_matching_exo_mass": float(floor),
        "branches_raised": int((offset > 0).sum()),
        "branches_unchanged": int((offset == 0).sum()),
        "max_abs_error_vs_expected_mass": float((mass - torch.maximum(floor, current)).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovered_run", type=Path, help="Directory of the recovered GPU evidence")
    args = parser.parse_args()
    torch.set_num_threads(2)

    executed = args.recovered_run / "pair05"
    archive_path = (
        args.recovered_run / "pair05-device-fix" /
        "pair05-gga-foreground-bundle-004-retry1.zip"
    )
    receipt = {path: sha for sha, path in (
        line.split(maxsplit=1) for line in
        (executed / "executed-code-and-meta.sha256").read_text().splitlines()
    )}
    verified_files = {
        "pair05_gga_routes.py": True,
        "plan/pair05-gga-routes.npz": False,
        "core/finetune/models/wan_i2v/custom_transformer.py": True,
        "core/finetune/models/wan_i2v/sft_trainer.py": True,
        "infer.py": True,
        "meta.json": False,
    }
    with ZipFile(archive_path) as archive:
        archive_files = {}
        for name, staged in verified_files.items():
            path = "pair05-gga-foreground-bundle/" + ("staged/" if staged else "") + name
            data = archive.read(path)
            if hashlib.sha256(data).hexdigest() != receipt[name]:
                raise ValueError(f"Executed receipt hash mismatch for {name}")
            archive_files[name] = data

    route_namespace = {"__name__": "pair05_cpu_route_probe"}
    exec(compile(archive_files["pair05_gga_routes.py"], "pair05_gga_routes.py", "exec"), route_namespace)
    with np.load(io.BytesIO(archive_files["plan/pair05-gga-routes.npz"]), allow_pickle=False) as plan:
        edges = plan["edges"]
        launches = (executed / "launch.log").read_text()
        if "--cos_sim_scaling_factor 3.0" not in launches:
            raise ValueError("The equal-logit probe expects the executed geometry scale to be 3.0.")
        route = route_namespace["ForegroundRoutes"](edges, plan["cosine"], plan["samples"], 3.0)
        total = route_namespace["ALL_TOKENS"]
        query_count = route_namespace["EGO_TOKENS"]
        keys_per_query_slot = route_namespace["ROWS"] * route_namespace["ROWS"]
        exo = torch.arange(total) % route_namespace["TOTAL_COLS"] < route_namespace["EXO"]
        # The executed transformer zeros EGO direction vectors, so native
        # EGO bias is log(1 + epsilon). EXO columns are replaced by selected_bias.
        base_values = torch.full((1, 1, total), np.log(1 + 1e-6), dtype=torch.float32)
        base = base_values.expand(1, query_count, total)
        _, bias = route.selected_bias(base, exo)
        assert int(torch.isfinite(bias[..., exo]).sum()) == len(edges) == 295
        assert torch.isfinite(bias[..., ~exo]).all()

        # Equal learned Q·K scores. A dummy V=1 for EXO and V=0 for EGO
        # turns SDPA output into the probability mass on the permitted EXO group.
        q = torch.zeros(1, 1, bias.shape[1], 1)
        k = torch.zeros(1, 1, total, 1)
        v = exo.to(torch.float32).view(1, 1, -1, 1)
        neutral = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(1)).flatten()
        theoretical = torch.softmax(bias, dim=-1)[0, :, exo].sum(-1)
        torch.testing.assert_close(neutral, theoretical, rtol=1e-5, atol=1e-7)
        exo_only = bias.masked_fill(~exo, -torch.inf).unsqueeze(1)
        isolated = F.scaled_dot_product_attention(q, k, v, attn_mask=exo_only).flatten()
        torch.testing.assert_close(isolated, torch.ones_like(isolated), rtol=0, atol=1e-6)

        def summary(mass):
            return {name: float(value * 100) for name, value in (
                ("min", mass.min()), ("mean", mass.mean()), ("max", mass.max())
            )}

        branches_by_key_count = np.unique(edges[:, :2], axis=0, return_counts=True)[1]
        frames = {}
        for frame in plan["frames"]:
            target_slot = int(frame) // 4
            in_frame = edges[:, 0] // keys_per_query_slot == target_slot
            frame_summary = {}
            for identity in sorted(set(edges[in_frame, 1])):
                relevant = torch.as_tensor(
                    (route.branches[:, 0] // keys_per_query_slot == target_slot) &
                    (route.branches[:, 1] == identity)
                )
                frame_summary[str(identity)] = {
                    "branches": int(relevant.sum()),
                    "equal_logit_exo_mass_pct": summary(neutral[relevant]),
                }
            frames[str(frame)] = {
                "slot_index": target_slot,
                "routed_queries": len(set(edges[in_frame, 0])),
                "edges": int(in_frame.sum()),
                "per_object_id": frame_summary,
            }

        output = {
            "mode": "synthetic_equal_qk_cpu_sdpa_not_real_model_attention",
            "torch_version": torch.__version__,
            "plan_sha256": receipt["plan/pair05-gga-routes.npz"],
            "matched_executed_hashes": list(verified_files),
            "total_query_count": query_count,
            "routed_query_count": len(set(edges[:, 0])),
            "branches": int(bias.shape[1]),
            "ego_keys_per_branch": int((~exo).sum()),
            "exo_keys_per_branch_min": int(branches_by_key_count.min()),
            "exo_keys_per_branch_max": int(branches_by_key_count.max()),
            "equal_logit_exo_mass_pct": summary(neutral),
            "exo_only_control_max_abs_error_from_one": float((isolated - 1).abs().max()),
            "diagnostic_group_floor": test_group_floor(bias, exo),
            "frames": frames,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
