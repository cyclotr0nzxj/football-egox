"""Drop only the first-frame image cross-attention contribution for one take."""

import hashlib
import json


CALLS = 4000  # 40 blocks x 50 steps x positive/unconditional CFG
SAMPLED_CALLS = (1, 2, 2000, 2001, 3999, 4000)


def tensor_digest(tensor):
    import torch

    cpu = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256(str((tuple(cpu.shape), str(cpu.dtype))).encode())
    digest.update(memoryview(cpu.view(torch.uint8).numpy()))
    return digest.hexdigest()


class ImageGate:
    def __init__(self):
        self.calls = 0
        self.initial = None
        self.samples = {}

    def record_initial(self, **tensors):
        if self.initial is not None or self.calls:
            raise RuntimeError("Initial conditions must be recorded once before denoising.")
        self.initial = {name: tensor_digest(value) for name, value in tensors.items()}
        print("PAIR05_CLIP_INITIAL " + json.dumps(self.initial), flush=True)

    def combine(self, text, image):
        if self.initial is None:
            raise RuntimeError("Image attention ran before initial condition verification.")
        self.calls += 1
        if self.calls in SAMPLED_CALLS:
            magnitude = image[:, ::256, ::64].float().abs().mean().item()
            self.samples[self.calls] = [magnitude, 0.0]
        return text

    def finish(self, foreground_calls):
        if self.calls != CALLS or foreground_calls != CALLS or tuple(self.samples) != SAMPLED_CALLS:
            raise RuntimeError("Incomplete Pair05 image or geometry attention trace.")
        print("PAIR05_CLIP_FINISH " + json.dumps({"image_calls": self.calls,
                                               "foreground_calls": foreground_calls, "samples": self.samples}),
              flush=True)


def attach(pipe, gate):
    processors = [block.attn2.processor for block in pipe.transformer.blocks]
    if len(processors) != 40 or len({id(processor) for processor in processors}) != len(processors):
        raise RuntimeError("Expected 40 independent image cross-attention processors.")
    pipe._pair05_clip_gate = gate
    for processor in processors:
        processor._pair05_clip_gate = gate
