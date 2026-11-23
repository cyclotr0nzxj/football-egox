import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor

run, vae_directory, render_bin = (Path(value) for value in sys.argv[1:])
expected_prior = 'c6ee04f0be3d48bcfc7564576fedfa137d2ad63f7f46ccdae711980b8713eeaa'
expected_weights = 'd6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793'
expected_config = 'f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527'

def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

candidates = [run / relative for relative in (
    'ball-recovery/render/pair05_s0024/ego_Prior.mp4',
    'inference-update/example/self_capture/ego_priors/pair05_s0024/ego_Prior.mp4',
    'update-priors-008/pair05_s0024/pair05_s0024/ego_Prior.mp4',
)]
prior = next((path for path in candidates if path.is_file() and digest(path) == expected_prior), None)
assert prior is not None, 'Frozen Pair05 prior not found in recorded paths'
assert digest(vae_directory / 'config.json') == expected_config
assert digest(vae_directory / 'diffusion_pytorch_model.safetensors') == expected_weights
print('Frozen prior, config and weights match', flush=True)
ffmpeg = str(render_bin / 'ffmpeg')
ffprobe = str(render_bin / 'ffprobe')

def validate_frames(path):
    probe = json.loads(subprocess.check_output([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
        '-show_entries', 'stream=width,height,avg_frame_rate,time_base:frame=pts,width,height',
        '-of', 'json', str(path),
    ]))
    stream, = probe['streams']
    assert (stream['width'], stream['height'], Fraction(stream['avg_frame_rate'])) == (448, 448, 30)
    assert len(probe['frames']) == 49
    for index, frame in enumerate(probe['frames']):
        assert (frame['width'], frame['height']) == (448, 448)
        assert Fraction(stream['time_base']) * frame['pts'] == Fraction(index, 30)

validate_frames(prior)
raw = subprocess.check_output([ffmpeg, '-nostdin', '-v', 'error', '-i', str(prior), '-map', '0:v:0',
                               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
source = np.frombuffer(raw, np.uint8).reshape(49, 448, 448, 3)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.ones(1, device='cuda', dtype=torch.bfloat16).sum().item() == 1
start = time.monotonic()
vae = AutoencoderKLWan.from_pretrained(str(vae_directory), local_files_only=True,
                                       torch_dtype=torch.bfloat16).to('cuda').eval()
processor = VideoProcessor(vae_scale_factor=8)
input_video = processor.preprocess([Image.fromarray(x) for x in source], height=448, width=448)
assert tuple(input_video.shape) == (49, 3, 448, 448)
input_video = input_video.permute(1, 0, 2, 3).unsqueeze(0).to('cuda', dtype=vae.dtype)
seed = 846514
generator = torch.Generator(device='cuda').manual_seed(seed)
print('VAE encode starting', flush=True)
with torch.inference_mode():
    sample = vae.encode(input_video).latent_dist.sample(generator=generator)
    assert tuple(sample.shape) == (1, 16, 13, 56, 56) and torch.isfinite(sample).all().item()
    mean = sample.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    sigma = sample.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    condition = (sample - mean) * (1 / sigma)
    print('VAE decode starting', flush=True)
    decoded = vae.decode(condition * sigma + mean, return_dict=False)[0]
    assert decoded.shape == input_video.shape and torch.isfinite(decoded).all().item()
    pixels = (((decoded[0].float().cpu().permute(1, 2, 3, 0) + 1) * 127.5)
              .clamp(0, 255).round().to(torch.uint8).numpy())
elapsed = time.monotonic() - start
for indices, name in (((0, 12, 24, 36, 48), 'keyframes.png'),
                      ((8, 12, 16, 20, 24, 28), 'context.png')):
    sheet = Image.new('RGB', (896, (448 + 24) * len(indices)), '#101621')
    draw = ImageDraw.Draw(sheet)
    for row, frame in enumerate(indices):
        y = row * 472
        draw.text((8, y + 5), f'F{frame:02} | frozen geometry prior', fill='white')
        draw.text((456, y + 5), 'Wan VAE round-trip', fill='white')
        sheet.paste(Image.fromarray(source[frame]), (0, y + 24))
        sheet.paste(Image.fromarray(pixels[frame]), (448, y + 24))
    sheet.save(name)
output = Path('pair05-vae.mp4')
subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '448x448', '-framerate', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-preset', 'fast', '-crf', '12', '-pix_fmt', 'yuv420p', '-video_track_timescale', '30000',
                str(output)], input=pixels.tobytes(), check=True)
validate_frames(output)
record = {'status': 'VAE_ROUND_TRIP_COMPLETE', 'take': 'pair05_s0024', 'prior_sha256': expected_prior,
          'vae_weights_sha256': expected_weights, 'vae_config_sha256': expected_config,
          'frames': 49, 'fps': 30, 'frame_pts_exact': True, 'seed': seed, 'latent_shape': list(condition.shape),
          'dtype': str(vae.dtype), 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
          'seconds_load_encode_decode': round(elapsed, 3), 'max_allocated_bytes': torch.cuda.max_memory_allocated(),
          'rgb_mae_by_frame': np.abs(pixels.astype(np.int16) - source).mean(axis=(1, 2, 3)).tolist(),
          'products': {name: digest(Path(name)) for name in ('keyframes.png', 'context.png', 'pair05-vae.mp4', 'probe.py')}}
Path('verification.json').write_text(json.dumps(record, indent=2) + chr(10))
print('PAIR05_VAE_COMPLETE', flush=True)
