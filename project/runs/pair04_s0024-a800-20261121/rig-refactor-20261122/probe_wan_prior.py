"""Check the prior or run an isolated, offline Wan VAE round-trip."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from verified_pixels import TARGET, check_reference, file_sha256, review_frames


KEY_FRAMES = (0, 12, 24, 36, 48)


def preview_stats(frames):
    y, x = np.mgrid[:TARGET, :TARGET] - TARGET // 2
    upper = (y < 0) & (x * x + y * y <= (TARGET // 2)**2)
    return [float((frame.max(axis=-1)[upper] <= 8).mean()) for frame in frames]


def roundtrip(frames, model, seed):
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This probe expects the previously verified CUDA BF16 runtime.")
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(str(model), local_files_only=True, torch_dtype=dtype)
    vae = vae.to("cuda").eval()
    processor = VideoProcessor(vae_scale_factor=8)
    data = processor.preprocess(list(frames.values()), height=TARGET, width=TARGET)
    if tuple(data.shape) != (49, 3, TARGET, TARGET):
        raise ValueError("VAE preprocessing differs from the frozen inference layout.")
    video = data.permute(1, 0, 2, 3).unsqueeze(0).to("cuda", dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        sampled = vae.encode(video).latent_dist.sample(generator=generator)
        if tuple(sampled.shape[2:]) != (13, 56, 56):
            raise ValueError("Unexpected Wan temporal or spatial reduction.")
        mean = sampled.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        sigma = sampled.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        condition = (sampled - mean) * (1 / sigma)
        decoded = vae.decode(condition * sigma + mean, return_dict=False)[0]
    if decoded.shape != video.shape or not torch.isfinite(decoded).all().item():
        raise ValueError("Decoded prior is not the expected finite video.")
    pixels = (((decoded[0].float().cpu().permute(1, 2, 3, 0) + 1) * 127.5)
              .clamp(0, 255).round().to(torch.uint8).numpy())
    return pixels, list(condition.shape)


def save_comparison(source, decoded, output):
    sheet = Image.new("RGB", (2 * TARGET, (TARGET + 30) * len(KEY_FRAMES)), "#101621")
    draw = ImageDraw.Draw(sheet)
    for row, frame in enumerate(KEY_FRAMES):
        y = row * (TARGET + 30)
        draw.text((8, y + 8), f"F{frame:02} | input prior", fill="white")
        draw.text((TARGET + 8, y + 8), "Wan VAE reconstruction", fill="white")
        sheet.paste(source[frame], (0, y + 30))
        sheet.paste(Image.fromarray(decoded[frame]), (TARGET, y + 30))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--fps", type=int, choices=(25, 30), required=True)
    parser.add_argument("--vae-dir", type=Path, help="Existing official local vae directory")
    parser.add_argument("--vae-weights-sha256", help="Expected SHA from the existing model receipt")
    parser.add_argument("--seed", type=int, default=846514)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.vae_dir is None) != (args.vae_weights_sha256 is None):
        parser.error("Provide both the local VAE directory and its frozen weight SHA for a round-trip.")
    if file_sha256(args.prior) != args.sha256:
        raise ValueError("Prior SHA differs from its frozen input.")
    check_reference(args.prior, TARGET, args.fps)
    images = review_frames(args.prior, TARGET, range(49))
    source = np.stack([np.asarray(frame) for frame in images.values()])
    result = {
        "prior_sha256": args.sha256,
        "mode": "input_pixels_only",
        "frames": 49,
        "fps": args.fps,
        "causal_direct_input_blocks": [[0]] + [list(range(i - 3, i + 1)) for i in range(4, 49, 4)],
        "prior_upper_circle_near_black_rgb_fraction": preview_stats(source),
    }
    os_path = args.output
    os_path.mkdir(parents=True, exist_ok=False)
    if args.vae_dir:
        weights = args.vae_dir / "diffusion_pytorch_model.safetensors"
        if file_sha256(weights) != args.vae_weights_sha256:
            raise ValueError("Wan VAE weights do not match the frozen cache.")
        decoded, latent_shape = roundtrip(images, args.vae_dir.resolve(), args.seed)
        save_comparison(images, decoded, os_path / "wan-vae-roundtrip.png")
        result.update({
            "mode": "isolated_wan_vae_roundtrip",
            "seed": args.seed,
            "vae_weight_sha256": args.vae_weights_sha256,
            "vae_config_sha256": file_sha256(args.vae_dir / "config.json"),
            "latent_shape": latent_shape,
            "rgb_mae": np.abs(decoded.astype(np.int16) - source).mean(axis=(1, 2, 3)).tolist(),
            "manual_object_visibility_review": "pending",
        })
    (os_path / "verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
