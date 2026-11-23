"""Validate the Pair05 floor at the production route boundary using CPU tensors."""

from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "staged"))
from pair05_gga_routes import ALL_TOKENS, EGO_TOKENS, EXO, FOREGROUND_MASS_FLOOR, TOTAL_COLS
from pair05_gga_routes import ForegroundRoutes


def state():
    origin = RUN / "clone-004-20261122/pair05-device-fix/pair05-gga-foreground-bundle-004-retry1.zip"
    with ZipFile(origin) as archive:
        plan = archive.read("pair05-gga-foreground-bundle/plan/pair05-gga-routes.npz")
    with np.load(BytesIO(plan), allow_pickle=False) as data:
        routes = ForegroundRoutes(data["edges"], data["cosine"], data["samples"], 3.0)
    return routes, hashlib.sha256(plan).hexdigest()


def check_marker(routes, exo, base):
    heads, width = 40, 4
    query = torch.zeros(1, heads, EGO_TOKENS, width, dtype=torch.bfloat16)
    key = torch.zeros(1, heads, ALL_TOKENS, width, dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    value[:, :, exo, 0] = 1
    allowed = torch.as_tensor(routes.arrays["keys"]).unique()
    value[:, :, allowed, 0] = 0
    value[:, :, allowed, 1] = 1
    value[:, :, ~exo, 2] = 1
    output = torch.full_like(query, -1)
    capture = StringIO()
    with redirect_stdout(capture):
        routes.apply(output, query, key, value, base, exo)
    log, = capture.getvalue().splitlines()
    assert log.startswith("PAIR05_FLOOR_SAMPLE ")
    telemetry = json.loads(log.removeprefix("PAIR05_FLOOR_SAMPLE "))
    targets = torch.as_tensor(routes.arrays["targets"])
    selected = output.index_select(2, targets)
    torch.testing.assert_close(selected[..., 0].float(), torch.zeros_like(selected[..., 0]).float())
    torch.testing.assert_close(selected[..., 1].float(), torch.full_like(selected[..., 1], 0.25).float())
    torch.testing.assert_close(selected[..., 2].float(), torch.full_like(selected[..., 2], 0.75).float())
    keep = torch.ones(EGO_TOKENS, dtype=torch.bool)
    keep[targets] = False
    assert (output[:, :, keep] == -1).all()
    assert telemetry["heads_branches_raised"] == 40 * 53
    assert telemetry["minimum_after"] >= 0.249999
    assert set(telemetry["by_target"]) == {"F12:1", "F12:2", "F12:6", "F24:1", "F24:2", "F24:4", "F24:6"}
    return telemetry


def compare_reference(routes, exo, base):
    torch.manual_seed(311)
    query = 0.5 * torch.randn(1, 2, EGO_TOKENS, 4)
    key = 0.5 * torch.randn(1, 2, ALL_TOKENS, 4)
    value = torch.randn_like(key)
    first_query = int(routes.arrays["branch_queries"][0])
    first_exo_key = int(routes.arrays["sparse_keys"][0, 0])
    query[0, 0, first_query, 0] = 12
    key[0, 0, first_exo_key, 0] = 12
    route, bias = routes.selected_bias(base, exo)
    q = query.index_select(2, route["branch_queries"])
    baseline = torch.nn.functional.scaled_dot_product_attention(q, key, value, attn_mask=bias.unsqueeze(1))
    adjusted = routes.apply_mass_floor(baseline, q, key, value, bias, route, sample=False)

    logits = (q.double() @ key.double().transpose(-1, -2)) * 0.5 + bias.unsqueeze(1).double()
    before = logits.softmax(dim=-1)[..., exo].sum(dim=-1)
    minimum = torch.maximum(before, torch.full_like(before, FOREGROUND_MASS_FLOOR))
    exo_probability = logits[..., exo].softmax(dim=-1)
    ego_probability = logits[..., ~exo].softmax(dim=-1)
    expected = ((exo_probability @ value[..., exo, :].double()) * minimum.unsqueeze(-1)
                + (ego_probability @ value[..., ~exo, :].double()) * (1 - minimum).unsqueeze(-1))
    torch.testing.assert_close(adjusted.double(), expected, rtol=1e-5, atol=2e-6)
    no_change = before >= FOREGROUND_MASS_FLOOR
    assert no_change.any() and (~no_change).any()
    torch.testing.assert_close(adjusted[no_change], baseline[no_change], rtol=0, atol=0)

    output = torch.full_like(query, -1)
    routes.apply(output, query, key, value, base, exo)
    for index, target in enumerate(route["targets"]):
        reference = expected[:, :, route["merge"] == index, :].mean(dim=2)
        torch.testing.assert_close(output[:, :, target].double(), reference, rtol=1e-5, atol=2e-6)
    return {
        "float64_reference_max_abs_error": float((expected - adjusted).abs().max()),
        "rows_raised": int((~no_change).sum()), "rows_unchanged": int(no_change.sum()),
        "all_mixed_and_unmixed_targets_match": True,
    }


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    routes, plan_digest = state()
    exo = torch.arange(ALL_TOKENS) % TOTAL_COLS < EXO
    base = torch.zeros(1, 1, 1).expand(1, EGO_TOKENS, ALL_TOKENS)
    with torch.inference_mode():
        marker = check_marker(routes, exo, base)
        reference = compare_reference(routes, exo, base)
    print(json.dumps({
        "scope": "local CPU with synthetic real-score Q/K and exact production route layout",
        "torch": torch.__version__, "plan_sha256": plan_digest,
        "source_sha256": hashlib.sha256((HERE.parent / "staged/pair05_gga_routes.py").read_bytes()).hexdigest(),
        "bf16_marker_40_heads_exact": True, "bf16_marker": marker, "reference_check": reference,
        "seconds": round(time.monotonic() - started, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
