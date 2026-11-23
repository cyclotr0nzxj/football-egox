"""Apply vetted Pair05 EXO routes inside Wan's GGA self-attention."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROWS, EXO, EGO, TIMES = 28, 49, 28, 13
TOTAL_COLS = EXO + EGO
EGO_TOKENS = TIMES * ROWS * EGO
ALL_TOKENS = TIMES * ROWS * TOTAL_COLS
TAKE = "pair05_s0024"
FRAMES = (12, 24)
META_SHA256 = "c2422ed6f6ecb0c150b8be612bc7c927cc39f5ec70cdd5a0593d8811d8d5846c"
EXO_SHA256 = "902b187de0af168bdbc6ccf3ec4a8720d43298a65906b77e5e7d7e7e7e3f9fd7"
PRIOR_SHA256 = "c6ee04f0be3d48bcfc7564576fedfa137d2ad63f7f46ccdae711980b8713eeaa"
PLAN_SHA256 = "3cd25a102a6ecf4006add800b293dcf40b82b296392ff652aeea308fcd0f501e"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(edges, cosine, pixels):
    if (edges.ndim != 2 or edges.shape[1] != 4 or not len(edges) or edges.dtype.kind not in "iu"
            or pixels.ndim != 2 or pixels.shape[1] != 4 or not len(pixels) or pixels.dtype.kind not in "iu"
            or cosine.shape != (len(edges),) or not np.isfinite(cosine).all() or (np.abs(cosine) > 1).any()):
        raise ValueError("Malformed Pair05 GGA routes.")
    frame, dst, src, identity = pixels.astype(np.int64).T
    if (set(frame) != set(FRAMES) or (dst < 0).any() or (dst >= 448 * 448).any()
            or (src < 0).any() or (src >= 448 * 784).any() or not set(identity).issubset({1, 2, 4, 6})
            or len(np.unique(pixels[:, (0, 1)], axis=0)) != len(pixels)
            or len(np.unique(pixels[:, (0, 2)], axis=0)) != len(pixels)):
        raise ValueError("Invalid or duplicated reviewed foreground sample.")
    dy, dx = np.divmod(dst, 448)
    sy, sx = np.divmod(src, 784)
    time = frame // 4
    query = (time * ROWS + dy // 16) * EGO + dx // 16
    key = (time * ROWS + sy // 16) * TOTAL_COLS + sx // 16
    expected, counts = np.unique(np.column_stack((query, identity, key)), axis=0, return_counts=True)
    if not np.array_equal(np.column_stack((expected, counts)), edges):
        raise ValueError("Some attention edges disagree with the selected time, identity or pixels.")
    if np.unique(key, return_counts=True)[1].max() > 16:
        raise ValueError("The combined per-source-block sample budget exceeds 16.")
    if len(np.unique(np.column_stack((key, identity)), axis=0)) != len(np.unique(key)):
        raise ValueError("A source K/V token was assigned two different foreground identities.")


class ForegroundRoutes:
    def __init__(self, edges, cosine, pixels, scale):
        validate_plan(edges, cosine, pixels)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid GGA cosine scale.")
        branches, edge_branch = np.unique(edges[:, :2], axis=0, return_inverse=True)
        targets, merge, branches_per_target = np.unique(branches[:, 0], return_inverse=True, return_counts=True)
        c = np.where(cosine > 0, cosine * scale, cosine) if scale > 1 else cosine * scale
        self.arrays = {
            "branch_queries": branches[:, 0].astype(np.int64),
            "edge_branches": edge_branch,
            "keys": edges[:, 2].astype(np.int64),
            "targets": targets.astype(np.int64),
            "merge": merge,
            "branches_per_target": branches_per_target,
            "edge_bias": np.log(1 + c + 1e-6),
        }
        self.branches = branches
        frames, _, source_pixels, _ = pixels.astype(np.int64).T
        self.source_time = frames // 4
        self.source_y, self.source_x = np.divmod(source_pixels, EXO * 16)
        locations = np.column_stack((self.source_time, self.source_y // 16, self.source_x // 16))
        self.source_blocks, self.source_group, self.source_counts = np.unique(
            locations, axis=0, return_inverse=True, return_counts=True,
        )
        self._device = {}
        self.attention_calls = 0

    def geometry_directions(self, point_map, cam_origins):
        """Use one source sample set per block for both native and routed geometry."""
        if point_map.shape != (49, 3, ROWS * 16, EXO * 16) or cam_origins.shape != (TIMES, ROWS * 2, EXO * 2, 3):
            raise ValueError("Unexpected world point map or camera origins for Pair05.")
        world = point_map[::4].permute(0, 2, 3, 1)
        device = world.device
        time, row, col = torch.as_tensor(self.source_blocks, device=device).T
        sample_group = torch.as_tensor(self.source_group, device=device)
        sample_counts = torch.as_tensor(self.source_counts, device=device)[:, None]
        times, py, px = torch.as_tensor(np.column_stack((self.source_time, self.source_y, self.source_x)), device=device).T
        raw = world[times, py, px]

        use_fixed = torch.ones((TIMES, ROWS * 2, EXO * 2), dtype=torch.bool, device=device)
        subcells = [(2 * row + dy, 2 * col + dx) for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))]
        for y, x in subcells:
            use_fixed[time, y, x] = False
        t, gy, gx = torch.where(use_fixed)
        py, px = gy * 8 + 3, gx * 8 + 3
        # Exactly the native align_corners=False 1/8 samples in untouched blocks.
        upper = (world[t, py, px] + world[t, py, px + 1]) * 0.5
        lower = (world[t, py + 1, px] + world[t, py + 1, px + 1]) * 0.5
        fixed = (upper + lower) * 0.5

        result = world.new_empty((TIMES, TIMES, ROWS * 2, EXO * 2, 3))
        for query_time, origin in enumerate(cam_origins[:, 0, 0]):
            result[query_time, t, gy, gx] = F.normalize(fixed - origin, dim=-1)
            dirs = F.normalize(raw - origin, dim=-1)
            pooled = dirs.new_zeros((len(time), 3)).index_add_(0, sample_group, dirs) / sample_counts
            for y, x in subcells:
                result[query_time, time, y, x] = pooled
        return result

    def selected_bias(self, base_bias, exo_mask):
        if base_bias.ndim != 3 or base_bias.shape[1:] != (EGO_TOKENS, ALL_TOKENS) or exo_mask.shape != (ALL_TOKENS,):
            raise ValueError("Pair05 foreground routing requires the pinned GGA token layout.")
        device = base_bias.device
        exo_mask = exo_mask.to(device=device)
        if device not in self._device:
            expected_mask = torch.arange(ALL_TOKENS, device=device) % TOTAL_COLS < EXO
            if not torch.equal(exo_mask, expected_mask):
                raise ValueError("Pair05 EXO/EGO split does not match the prepared route layout.")
            self._device[device] = {name: torch.as_tensor(value, device=device) for name, value in self.arrays.items()}
        route = self._device[device]
        # Preserve the original EGO keys. Replace native EXO geometry for these rows entirely.
        bias = base_bias.index_select(1, route["branch_queries"])
        bias[..., exo_mask] = -torch.inf
        bias[:, route["edge_branches"], route["keys"]] = route["edge_bias"].to(bias.dtype)
        return route, bias

    def apply(self, ego_output, ego_query, key, value, base_bias, exo_mask):
        route, bias = self.selected_bias(base_bias, exo_mask)
        result = F.scaled_dot_product_attention(
            ego_query.index_select(2, route["branch_queries"]),
            key, value, attn_mask=bias.unsqueeze(1).to(ego_query.dtype),
            dropout_p=0.0, is_causal=False,
        )
        # Mixed query blocks get one separate attention row per identity, then equal means.
        output = torch.zeros((*result.shape[:2], len(route["targets"]), result.shape[-1]),
                             dtype=torch.float32, device=result.device)
        output.index_add_(2, route["merge"], result.float())
        output /= route["branches_per_target"][None, None, :, None]
        ego_output.index_copy_(2, route["targets"], output.to(ego_output.dtype))
        self.attention_calls += 1
        return ego_output


def load_foreground_routes(plan, metadata_path, metadata_rows, matching, use_gga, kv_cache, scale):
    if not use_gga or kv_cache or matching != [TAKE] or len(metadata_rows) != 1:
        raise ValueError("The foreground plan requires the single Pair05 take, GGA, and KV cache off.")
    row = metadata_rows[0]
    if (_hash(plan) != PLAN_SHA256 or _hash(metadata_path) != META_SHA256 or _hash(row["exo_path"]) != EXO_SHA256
            or _hash(row["ego_prior_path"]) != PRIOR_SHA256):
        raise ValueError("Foreground plan does not match the verified Pair05 camera, EXO and prior.")
    with np.load(plan, allow_pickle=False) as doc:
        if (tuple(doc["frames"]) != FRAMES or doc["metadata_sha256"].item() != META_SHA256
                or doc["exo_sha256"].item() != EXO_SHA256 or doc["prior_sha256"].item() != PRIOR_SHA256):
            raise ValueError("Pair05 plan was built from a different input set.")
        return ForegroundRoutes(doc["edges"], doc["cosine"], doc["samples"], scale)


def attach_foreground_routes(transformer, routes):
    processors = [block.attn1.processor for block in transformer.blocks]
    if not processors or len(processors) != len({id(processor) for processor in processors}):
        raise ValueError("Expected a distinct GGA self-attention processor in every Wan block.")
    for processor in processors:
        if processor.__class__.__name__ != "WanAttnProcessor2_0" or not hasattr(processor, "_pair05_foreground"):
            raise ValueError("The staged Wan GGA processor hook is unavailable.")
        processor._pair05_foreground = routes
