# Dynamic Harris / SIFT evidence

Pair04: all 311 visible frames at25fps, 12.44s. The saved 960x540 H.264 exo.mp4 and ego.mp4 working copies are the actual numerical inputs; detection does not use unsaved original-video frames. Nano AAC audio is preserved in the four-panel output.

Left: R8/EXO; right: Nano/EGO; top: Harris; bottom: SIFT. Every panel shows its method, view, frame index, time and retained-center count. Frames/views are independently detected with the static experiment's exact parameters: seed42, one OpenCV thread, Harris2/3/0.04 with1% positive maximum and3x3 NMS; SIFT500/3/0.04/10/1.6; duplicate centers removed, minimum5px spacing, maximum500 positions,16px boundary margin. There is no tracking, point trail, cross-view correspondence or inter-frame R. Grass texture, motion blur and compression effects are retained. Count differences from old static JPEGs are expected and not corrected.

counts.csv has1244 frame-view-method rows. manifest.json records parameters, versions, input SHA-256 and timeline mappings; audit.json records full decoding,311 timestamps in all three videos and unchanged source hashes. Historical raw EXO indices are references only because the raw backup is absent. Five JPEG previews are display illustrations, not detector inputs. Timing is descriptive, not a speed benchmark.

Portable reproduction after installing the repository dependencies and FFmpeg; start from the repository root and choose an empty destination:

```sh
python scripts/build_feature_video.py --exo data/video_pair04/exo.mp4 --ego data/video_pair04/ego.mp4 --output dynamic_reproduced
```

The original-project preparation command is `python scripts/build_feature_video.py --prepare-inputs --output output/week08_harris_sift/dynamic`; it reads the existing trim manifest, excludes hidden negative-PTS preroll, and saves working copies before detection. It is not required for portable reproduction.
