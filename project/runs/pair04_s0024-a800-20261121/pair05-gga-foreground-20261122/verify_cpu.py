"""Exercise the staged GGA seam, real geometry, and the sparse SDPA on CPU."""

import ast
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
META = RUN / "ball-view-cpu-review/pair05_s0024-ball-view-meta.json"
PREVIOUS = RUN / "pair05-foreground-match-20261122/results"
STAGED = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "staged"
PLAN = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE / "plan/pair05-gga-routes.npz"
sys.path.insert(0, str(STAGED))
from pair05_gga_routes import ALL_TOKENS, EGO_TOKENS, EXO, TAKE, TIMES, TOTAL_COLS
from pair05_gga_routes import attach_foreground_routes, load_foreground_routes, validate_plan


def rejects(run, detail):
    try:
        run()
    except (ValueError, RuntimeError) as error:
        assert detail in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejection containing {detail!r}.")


def check_plan_inputs(row):
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        for field, real in (
            ("exo_path", RUN / f"multiclip-prep/{TAKE}" / row["exo_path"]),
            ("ego_prior_path", RUN / f"update-008/output-008/priors/{TAKE}-final.mp4"),
        ):
            link = folder / row[field]
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(real)
        previous_dir = Path.cwd()
        os.chdir(folder)
        try:
            routes = load_foreground_routes(PLAN, META, [row], [TAKE], True, False, 3.0)
            rejects(lambda: load_foreground_routes(PLAN, META, [row], [TAKE], True, True, 3.0), "KV cache off")
            altered = folder / "altered.npz"
            altered.write_bytes(PLAN.read_bytes() + b"changed")
            rejects(lambda: load_foreground_routes(altered, META, [row], [TAKE], True, False, 3.0), "does not match")
        finally:
            os.chdir(previous_dir)
    return routes


def check_provenance(edges, cosine, pixels):
    previous = json.loads((PREVIOUS / "verification.json").read_text())["measurements"]
    negatives = {}
    with np.load(PREVIOUS / "foreground-routing.npz", allow_pickle=False) as known:
        for frame in (12, 24):
            _, target, source, identity = pixels[pixels[:, 0] == frame].T
            label = known[f"F{frame:02}_target_label"].ravel()
            index = known[f"F{frame:02}_source_index"].ravel()
            assert np.array_equal(label[target], identity) and np.array_equal(index[target], source)
            negatives[f"F{frame:02}"] = previous[f"F{frame:02}"]["rejected"]
            assert all(row["admitted"] == 0 for row in negatives[f"F{frame:02}"].values())

        # Rebuild legal edges from all accepted same-frame F12 pixels, before the 16-point cap.
        # This violates only the budget, with no mixed source blocks in F12.
        label = known["F12_target_label"].ravel()
        target = np.flatnonzero(label)
        old = np.column_stack((np.full(len(target), 12), target, known["F12_source_index"].ravel()[target], label[target]))
        unbounded = np.concatenate((old, pixels[pixels[:, 0] == 24]))
        time, dst, src, identity = unbounded.T
        dy, dx = np.divmod(dst, 448)
        sy, sx = np.divmod(src, 784)
        q = (time // 4 * 28 + dy // 16) * 28 + dx // 16
        k = (time // 4 * 28 + sy // 16) * 77 + sx // 16
        reformed, counts = np.unique(np.column_stack((q, identity, k)), axis=0, return_counts=True)
        rejects(lambda: validate_plan(np.column_stack((reformed, counts)), np.zeros(len(counts)), unbounded), "exceeds 16")
    for column, offset in ((1, 8), (2, 28 * 77 * 3), (2, 1)):
        altered = edges.copy()
        altered[0, column] += offset
        rejects(lambda: validate_plan(altered, cosine, pixels), "disagree")
    return negatives


def check_geometry(routes, row):
    k = np.array(row["camera_intrinsics"], dtype=np.float32)
    k[0, 0] *= 392 / k[0, 2]
    k[1, 1] *= 224 / k[1, 2]
    k[:2, 2] = (392, 224)
    y, x = np.mgrid[:448, :784].astype(np.float32)
    rays = np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)), axis=0)
    w2c = np.array(row["camera_extrinsics"], dtype=np.float32)
    rotation = np.linalg.inv(w2c[:, :3])
    cam_pos = -rotation @ w2c[:, 3]
    metadata = json.loads((RUN / "update-008/output-008/logs/validated-inputs.json").read_text())
    receipt, = [item for item in metadata if item["name"] == TAKE]
    depth_path = RUN / f"multiclip-evidence/{TAKE}/geometry/vipe_results/{TAKE}/depth/exo.zip"
    points = torch.empty((49, 3, 448, 784))
    # Use the existing OpenEXR environment to decode the real depth, and the existing
    # Torch environment for attention. Both stay local and need no new packages.
    archive_reader = """
import sys
from zipfile import ZipFile
import numpy as np
sys.path.insert(0, sys.argv[1])
from build_ball_previews import load_depth
with ZipFile(sys.argv[2]) as archive:
    np.savez(sys.argv[3], **{str(i): load_depth(archive, i) for i in range(0, 49, 4)})
"""
    with TemporaryDirectory() as directory:
        decoded = Path(directory) / "depths.npz"
        subprocess.run(["/usr/local/Caskroom/miniforge/base/bin/python3.12", "-B", "-c", archive_reader,
                        str(RUN / "ball-view-cpu-review"), str(depth_path), str(decoded)], check=True)
        with np.load(decoded, allow_pickle=False) as archive:
            cache = {frame: archive[str(frame)] for frame in range(0, 49, 4)}
        for frame in range(0, 49, 4):
            depth = cache[frame]
            data = BytesIO()
            np.save(data, depth, allow_pickle=False)
            assert hashlib.sha256(data.getvalue()).hexdigest() == receipt["depth_sha256"][frame]
            xyz = np.einsum("ij,jhw->ihw", rotation, rays * depth)
            points[frame] = torch.from_numpy(xyz + cam_pos[:, None, None])
    ext = torch.tensor(row["ego_extrinsics"][::4], dtype=torch.float32)
    origin = torch.linalg.solve(ext[:, :, :3], -ext[:, :, 3])
    origins = origin[:, None, None, :].expand(-1, 56, 98, -1)
    worlds = points[::4]
    native = F.interpolate(worlds, size=(56, 98), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    revised = routes.geometry_directions(points, origins)
    active = torch.zeros((13, 56, 98), dtype=torch.bool)
    for t, gy, gx in routes.source_blocks:
        active[t, gy * 2:gy * 2 + 2, gx * 2:gx * 2 + 2] = True
    max_native_error = 0.0
    max_replacement_error = 0.0
    for query_time in range(TIMES):
        stock = F.normalize(native - origin[query_time], dim=-1)
        max_native_error = max(max_native_error, (revised[query_time, ~active] - stock[~active]).abs().max().item())
        for group, (t, y, x) in enumerate(routes.source_blocks):
            select = routes.source_group == group
            raw = points[t * 4, :, routes.source_y[select], routes.source_x[select]].T
            expected = F.normalize(raw - origin[query_time], dim=-1).mean(dim=0)
            block = revised[query_time, t, 2*y:2*y+2, 2*x:2*x+2]
            max_replacement_error = max(max_replacement_error, (block - expected).abs().max().item())
    assert max_native_error < 2e-6 and max_replacement_error < 2e-6
    assert len(routes.source_blocks) == 171 and max(routes.source_counts) == 16
    return {"verified_depth_frames": 13, "replaced_source_blocks": len(routes.source_blocks),
            "active_max_points": int(max(routes.source_counts)), "native_max_points": 16,
            "native_unchanged_max_abs_error": max_native_error,
            "replacement_max_abs_error": max_replacement_error}


def check_sparse_attention(routes, edges):
    exo = torch.arange(ALL_TOKENS) % TOTAL_COLS < EXO
    base = torch.zeros((1, 1, 1)).expand(1, EGO_TOKENS, ALL_TOKENS)
    route, bias = routes.selected_bias(base, exo)
    assert torch.isfinite(bias[..., exo]).sum().item() == len(edges)
    assert torch.isfinite(bias[:, route["edge_branches"], route["keys"]]).all()
    assert torch.equal(bias[..., ~exo], base[:, route["branch_queries"], :][..., ~exo])
    mixed_key = (6 * 28 + 20) * 77 + 32
    assert torch.isneginf(bias[..., mixed_key]).all()

    q = torch.zeros((1, 1, EGO_TOKENS, 8))
    q[..., 0] = 1.0
    k = torch.zeros((1, 1, ALL_TOKENS, 8))
    v = k.clone()
    # All unknown or wrong routes carry a dominant EXO feature. The mask must suppress it.
    k[0, 0, exo, 0], v[0, 0, exo, 0] = 20.0, 1.0
    v[0, 0, ~exo, 7] = 1.0
    for _, obj, key, _ in edges:
        v[0, 0, key, 0], v[0, 0, key, obj] = 0.0, 1.0
    branch_scores = F.scaled_dot_product_attention(q.index_select(2, route["branch_queries"]), k, v, attn_mask=bias[:, None])[0, 0]
    for score, (_, obj) in zip(branch_scores, routes.branches):
        assert score[obj] > 0 and score[7] > 0
        assert torch.nonzero(score[:7]).ravel().tolist() == [obj]
    seed = torch.full_like(q, -12.0)
    output = routes.apply(seed.clone(), q, k, v, base, exo)[0, 0]
    mixed = route["targets"][route["branches_per_target"] == 2]
    assert len(mixed) == 2 and (output[mixed, 1] > 0).all() and (output[mixed, 2] > 0).all()
    assert torch.equal(output[mixed, 0], torch.zeros(len(mixed)))
    untouched = torch.ones(EGO_TOKENS, dtype=torch.bool)
    untouched[route["targets"]] = False
    assert torch.equal(output[untouched], seed[0, 0, untouched]) and routes.attention_calls == 1
    # A correct route cannot remove unlabelled context already encoded in an admitted value.
    first_query = q[:, :, route["branch_queries"][:1]]
    contaminated = v.clone()
    contaminated[0, 0, int(edges[0, 2]), 0] = 1
    direct_content_leak = F.scaled_dot_product_attention(first_query, k, contaminated, attn_mask=bias[:, None, :1])[0, 0, 0, 0]
    contaminated = v.clone()
    contaminated[0, 0, int(torch.where(~exo)[0][0]), 0] = 1
    ego_context_leak = F.scaled_dot_product_attention(first_query, k, contaminated, attn_mask=bias[:, None, :1])[0, 0, 0, 0]
    assert direct_content_leak > 0 and ego_context_leak > 0
    return {"finite_exo_edges": len(edges), "isolated_identity_branches": len(routes.branches),
            "mixed_ego_queries": len(mixed), "ego_rows_left_to_native_attention": int(untouched.sum()),
            "synthetic_extra_feature_in_admitted_exo_value_can_pass": True,
            "synthetic_extra_feature_in_preserved_ego_value_can_pass": True}


def check_staged_seam():
    source = STAGED / "core/finetune/models/wan_i2v/custom_transformer.py"
    parsed = ast.parse(source.read_text())
    node, = [item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == "WanAttnProcessor2_0"]
    namespace = dict(torch=torch, F=F, Attention=object, Optional=Optional, Dict=Dict)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    processor = namespace[node.name]()
    identity = torch.nn.Identity()
    attn = SimpleNamespace(processor=processor, heads=1, add_k_proj=None, norm_q=None, norm_k=None,
                           to_k=identity, to_q=identity, to_v=identity, to_out=(identity, identity))
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    args = dict(attention_mask_GGA=torch.tensor([[[1], [1], [0], [0]]]), cos_sim=torch.zeros((1, 4, 4)))
    baseline, _ = processor(attn, hidden, **args)

    def witness(ego_output, ego_query, key, value, base_bias, mask):
        assert torch.equal(ego_query, hidden[:, None, 2:, :]) and key.shape == value.shape == (1, 1, 4, 2)
        assert base_bias.shape == (1, 2, 4) and mask.tolist() == [True, True, False, False]
        return ego_output + 10

    attach_foreground_routes(SimpleNamespace(blocks=[SimpleNamespace(attn1=attn)]), SimpleNamespace(apply=witness))
    patched, _ = processor(attn, hidden, **args)
    assert torch.equal(patched[:, :2], baseline[:, :2])
    torch.testing.assert_close(patched[:, 2:], baseline[:, 2:] + 10)
    rejects(lambda: processor(attn, hidden), "requires GGA")
    rejects(lambda: processor(attn, hidden, do_kv_cache=True, **args), "requires GGA")
    return {"staged_source_method_cpu_witness": True, "missing_gga_rejected": True, "kv_cache_rejected": True}


if __name__ == "__main__":
    torch.set_num_threads(3)
    metadata, = json.loads(META.read_text())["test_datasets"]
    routes = check_plan_inputs(metadata)
    with np.load(PLAN, allow_pickle=False) as doc:
        edges, cosine, pixels = doc["edges"], doc["cosine"], doc["samples"]
    print(json.dumps({"torch": torch.__version__, "device": "cpu", "frozen_inputs_verified": True,
                      "actual_negative_matches": check_provenance(edges, cosine, pixels),
                      "joint_geometry_budget": check_geometry(routes, metadata),
                      "sparse_sdpa": check_sparse_attention(routes, edges),
                      "processor": check_staged_seam()}, indent=2))
