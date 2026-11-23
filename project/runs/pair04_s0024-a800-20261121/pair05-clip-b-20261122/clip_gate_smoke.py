"""Check the BF16 image gate before loading any large weights."""

import argparse
import contextlib
import io

import torch

from pair05_clip_gate import CALLS, ImageGate, SAMPLED_CALLS


def run(device):
    gate = ImageGate()
    text = torch.full((1, 3, 4), 2, dtype=torch.bfloat16, device=device)
    image = torch.full_like(text, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        gate.record_initial(latents=text)
        for _ in range(CALLS):
            assert gate.combine(text, image) is text
        gate.finish(CALLS)
    assert tuple(gate.samples) == SAMPLED_CALLS and all(value == [3.0, 0.0] for value in gate.samples.values())
    print(f"PAIR05_CLIP_PROBE_OK {device}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run(parser.parse_args().device)
