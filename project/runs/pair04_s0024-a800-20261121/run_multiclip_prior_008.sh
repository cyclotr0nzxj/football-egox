#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/root/autodl-tmp/football_egox/pair04-20261121
root="$base/multiclip-008"
take=${1:?clip identifier required}
[[ $take =~ ^pair(01|02|05)_s[0-9]{4}$ ]]
clip="$root/$take"
take_geometry="$clip/geometry/vipe_results/$take"
exo="$clip/example/self_capture/videos/$take/exo.mp4"
normal_input="$clip/egox-input"
renderer="$base/Renderer-sparse"
python="$base/renderer-env/bin/python"

export PATH="$base/renderer-env/bin:$PATH" PYTHONPATH="$renderer"
export CUDA_VISIBLE_DEVICES=0 IMAGEIO_FFMPEG_EXE="$base/renderer-env/bin/ffmpeg"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
grep -Fq "VIPE_MULTICLIP_VERIFIED $take " "$clip/geometry-launch.log"
test ! -e "$normal_input"
timeout --signal=TERM --kill-after=10s 120s "$python" -B "$root/build_multiclip_virtual_view.py" \
    "$take_geometry" "$exo" "$clip/exo_head_anchors.json" "$clip/viewpoint"
printf 'VIRTUAL_VIEW_VERIFIED %s\n' "$take"
timeout --signal=TERM --kill-after=10s 300s "$python" -B "$renderer/scripts/render_vipe_pointcloud.py" \
    --input_dir "$take_geometry" --out_dir "$clip/prior" --meta_json_path "$clip/viewpoint/meta.json" \
    --point_size 5.0 --start_frame 0 --end_frame 48 --fish_eye_rendering --use_mean_bg >"$clip/render.log" 2>&1
prior="$clip/prior/$take/ego_Prior.mp4"
ffprobe -v error -show_frames -show_streams -select_streams v:0 \
    -show_entries frame=pts,width,height:stream=width,height,avg_frame_rate,time_base -of json "$prior" > "$clip/prior-probe.json"
ffmpeg -nostdin -y -v error -i "$prior" -vf "select='eq(n,0)+eq(n,24)+eq(n,48)',tile=3x1" -frames:v 1 "$clip/prior-preview.png"
mkdir -p "$normal_input/example/self_capture"/{videos,ego_priors,depth_maps}/"$take"
cp -- "$exo" "$normal_input/example/self_capture/videos/$take/"
cp -- "$prior" "$normal_input/example/self_capture/ego_priors/$take/"
cp -- "$clip/viewpoint/meta.json" "$normal_input/"
"$python" -B "$renderer/scripts/convert_depth_zip_to_npy.py" --depth_path "$take_geometry/depth" \
    --egox_depthmaps_path "$clip/depth-conversion" > "$clip/depth-conversion.log" 2>&1
mv -- "$clip/depth-conversion/exo/"*.npy "$normal_input/example/self_capture/depth_maps/$take/"
"$python" -B - "$clip" "$take" <<'PY'
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from vipe.utils.io import read_depth_artifacts

clip, take = Path(sys.argv[1]), sys.argv[2]
doc = json.loads((clip / 'prior-probe.json').read_text())
stream, = doc['streams']
frames = doc['frames']
assert (stream['width'], stream['height'], Fraction(stream['avg_frame_rate']), len(frames)) == (448, 448, 30, 49)
assert all((row['width'], row['height'], int(row['pts']) * Fraction(stream['time_base'])) == (448, 448, Fraction(i, 30))
           for i, row in enumerate(frames))
artifacts = clip / 'geometry/vipe_results' / take
staged = clip / 'egox-input/example/self_capture/depth_maps' / take
arrays = sorted(staged.glob('*.npy'))
assert len(arrays) == 49
for (i, raw), name in zip(read_depth_artifacts(artifacts / 'depth/exo.zip'), arrays, strict=True):
    depth = np.load(name, allow_pickle=False)
    assert int(''.join(c for c in name.stem if c.isdigit())) == i and depth.dtype == np.float32
    assert depth.shape == (448, 784) and np.array_equal(raw.numpy(), depth) and np.isfinite(depth).all() and (depth > 0).all()
PY
printf 'MULTICLIP_PRIOR_AND_DEPTH_VERIFIED %s\n' "$take"
