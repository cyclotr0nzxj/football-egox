"""Verify the staged Pair05 route on the server before loading model weights."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--check-structure", action="store_true", help="Check frozen input binding without CUDA")
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve(strict=True)))
    import torch
    from pair05_gga_routes import ALL_TOKENS, EGO_TOKENS, EXO, TAKE, TOTAL_COLS
    from pair05_gga_routes import attach_foreground_routes, load_foreground_routes

    shot = json.loads(args.meta.read_text())["test_datasets"]
    routes = load_foreground_routes(args.plan, args.meta, shot, [TAKE], True, False, 3.0)
    active = len(routes.source_blocks)
    branches = len(routes.branches)
    edges = len(routes.arrays["keys"])
    if (active, branches, edges, int(routes.source_counts.max())) != (171, 53, 295, 16):
        raise ValueError("Unexpected Pair05 source blocks, branches, edges, or shared budget.")
    if args.check_structure:
        print("PAIR05_GGA_STRUCTURE_OK blocks=171 branches=53 edges=295 budget=16", flush=True)
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 is required for the preflight.")
    from core.finetune.models.wan_i2v.custom_transformer import WanAttnProcessor2_0

    proc = WanAttnProcessor2_0()
    attach_foreground_routes(SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=proc))]), routes)
    if proc._pair05_foreground is not routes:
        raise ValueError("The staged Wan processor did not retain the frozen foreground plan.")

    started = time.monotonic()
    heads, width = 40, 128
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        exo = torch.arange(ALL_TOKENS, device=device) % TOTAL_COLS < EXO
        base = torch.zeros((1, 1, 1), dtype=torch.float32, device=device).expand(1, EGO_TOKENS, ALL_TOKENS)
        ego_query = torch.zeros((1, heads, EGO_TOKENS, width), device=device, dtype=torch.bfloat16)
        ego_output = torch.full_like(ego_query, -1)
        kv = (1, heads, ALL_TOKENS, width)
        key = torch.zeros(kv, device=device, dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        # A feature in inadmissible EXO values must disappear from every selected branch.
        value[:, :, exo, 0] = 1
        allowed = torch.as_tensor(routes.arrays["keys"], device=device).unique()
        value[:, :, allowed, 0] = 0
        value[:, :, allowed, 1] = 1
        value[:, :, ~exo, 2] = 1
        out = routes.apply(ego_output, ego_query, key, value, base, exo.cpu())
        queries = torch.as_tensor(routes.arrays["targets"], device=device)
        selected = out.index_select(2, queries)
        if (routes.attention_calls != 1 or not torch.isfinite(selected).all().item()
                or torch.count_nonzero(selected[..., 0]).item()
                or not (selected[..., 1:3] > 0).all().item()):
            raise ValueError("Pair05 CUDA attention failed its route/retained-EGO check.")
        untouched = torch.ones(EGO_TOKENS, device=device, dtype=torch.bool)
        untouched[queries] = False
        if not (out[:, :, untouched] == -1).all().item():
            raise ValueError("Pair05 routed attention modified an unrelated target.")
    torch.cuda.synchronize(device)
    print("PAIR05_GGA_CUDA_OK " + json.dumps({"heads": heads, "head_dim": width, "queries": EGO_TOKENS,
                                              "keys": ALL_TOKENS, "edges": edges, "dtype": "bfloat16",
                                              "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                                              "seconds": round(time.monotonic() - started, 3)}), flush=True)


if __name__ == "__main__":
    main()
