"""Run the official MiDaS v2.1 Small ONNX export on the frozen Pair03 EXO."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
EXO = HERE.parent / "pair05-left-diagnosis-20261122/pair03-exo/exo.mp4"
EXO_SHA = "512a57b111ec1420e21def2814bdd17105d9a611f7e4e5da40f623c2e92a94b9"
MODEL_SHA = "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58"
MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
FRAMES, HEIGHT, WIDTH = 49, 448, 784


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_exo():
    if sha256(EXO) != EXO_SHA:
        raise ValueError("EXO checksum differs from the frozen Pair03 working video.")
    pixels = subprocess.check_output([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(EXO), "-map", "0:v:0",
        "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ])
    if len(pixels) != FRAMES * HEIGHT * WIDTH * 3:
        raise ValueError("Unexpected decoded Pair03 EXO length.")
    return np.frombuffer(pixels, np.uint8).reshape(FRAMES, HEIGHT, WIDTH, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if sha256(args.model) != MODEL_SHA:
        raise ValueError("MiDaS release checksum mismatch.")

    cv2.setNumThreads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    source, = session.get_inputs()
    target, = session.get_outputs()
    if source.shape != [1, 3, 256, 256] or source.type != "tensor(float)":
        raise ValueError("Unexpected tensor layout for the official MiDaS Small model.")

    start = time.monotonic()
    frames = read_exo()
    predictions = np.empty((FRAMES, HEIGHT, WIDTH), dtype=np.float32)
    for frame, rgb in enumerate(frames):
        original = rgb.astype(np.float32) / 255
        sample = cv2.resize(original, (256, 256), interpolation=cv2.INTER_CUBIC)
        inverse, = session.run([target.name], {source.name: np.ascontiguousarray(sample.transpose(2, 0, 1)[None])})
        predictions[frame] = cv2.resize(inverse.reshape(256, 256), (WIDTH, HEIGHT),
                                         interpolation=cv2.INTER_CUBIC)
        if frame in {0, 12, 24, 36, 48}:
            print(f"MiDaS CPU F{frame:02d}/{FRAMES - 1:02d}", flush=True)
    if not np.isfinite(predictions).all() or (np.ptp(predictions, axis=(1, 2)) < 1).any():
        raise ValueError("Non-finite or degenerate inverse-depth predictions.")

    data = args.output / "midas_inverse_depth.npz"
    np.savez_compressed(data, inverse_depth=predictions)
    metadata = {
        "model": "MiDaS v2.1 Small ONNX",
        "model_url": MODEL_URL,
        "model_sha256": MODEL_SHA,
        "exo_sha256": EXO_SHA,
        "inverse_depth_sha256": sha256(data),
        "depth_semantics": "relative inverse depth; scale and offset unknown before calibration",
        "shape": list(predictions.shape),
        "frame_times": "i/25",
        "provider": session.get_providers(),
        "onnxruntime": ort.__version__,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "per_frame_range": np.column_stack((predictions.min(axis=(1, 2)),
                                              predictions.max(axis=(1, 2)))).tolist(),
    }
    (args.output / "depth-receipt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata['shape']} relative inverse depth in {metadata['elapsed_seconds']} seconds.", flush=True)


if __name__ == "__main__":
    main()
