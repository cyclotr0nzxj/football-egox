"""Validate a 49-frame locked-camera EXO clip before the ViPE stage on 008."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def frame_check(video):
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-show_streams', '-show_frames', '-show_entries',
        'stream=codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,time_base:frame=pts,width,height',
        '-of', 'json', str(video),
    ], capture_output=True, check=True, text=True, timeout=45)
    info = json.loads(probe.stdout)
    stream, = info['streams']
    frames = info['frames']
    if (probe.stderr or stream['codec_type'] != 'video' or stream['codec_name'] != 'h264' or
            stream['pix_fmt'] != 'yuv420p' or len(frames) != 49 or Fraction(stream['avg_frame_rate']) != 25):
        raise ValueError('Expected only one H.264 stream with 49 frames at 25 fps.')
    for i, row in enumerate(frames):
        if (row['width'], row['height'], int(row['pts']) * Fraction(stream['time_base'])) != (784, 448, Fraction(i, 25)):
            raise ValueError(f'Unexpected dimensions or PTS at frame {i}.')


def camera_check(video):
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    if not ok:
        raise ValueError('Cannot decode EXO.')
    reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(reference)
    mask[14:124, 20:764] = 255
    mask[135:425, 20:135] = 255
    mask[135:425, 650:764] = 255
    mask[369:425, 135:650] = 255
    points = cv2.goodFeaturesToTrack(reference, 1700, 0.005, 4, mask=mask)
    if points is None:
        raise ValueError('No background features.')
    displacements, angles, counts = [0.0], [0.0], [len(points)]
    center = np.array([392, 224, 1])
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        forward, mask_a, _ = cv2.calcOpticalFlowPyrLK(reference, gray, points, None)
        back, mask_b, _ = cv2.calcOpticalFlowPyrLK(gray, reference, forward, None)
        good = (mask_a[:, 0] == 1) & (mask_b[:, 0] == 1) & (np.linalg.norm((back - points)[:, 0], axis=1) < 0.5)
        matrix, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC,
                                                     ransacReprojThreshold=0.8)
        if matrix is None or int(inliers.sum()) < 120:
            raise ValueError('Insufficient fixed-scene support for camera motion estimate.')
        counts.append(int(inliers.sum()))
        displacements.append(float(np.linalg.norm(matrix @ center - center[:2])))
        angles.append(float(abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))))
    cap.release()
    if len(counts) != 49 or max(displacements) > 6 or max(angles) > 0.5:
        raise ValueError(f'Observed motion exceeds fixed-camera working limit: {len(counts)} frames; '
                         f'{max(displacements):.2f}px, {max(angles):.3f} degrees.')
    return {'max_center_displacement_px': max(displacements), 'max_rotation_degrees': max(angles),
            'min_inlier_count': min(counts), 'frames': 49, 'method': 'bidirectional background LK plus RANSAC'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('sha256')
    args = parser.parse_args()
    if not args.video.is_file() or args.video.is_symlink() or args.video.name != 'exo.mp4':
        raise ValueError('Expected a regular EXO input clip.')
    if hashlib.sha256(args.video.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('EXO video digest mismatch.')
    frame_check(args.video)
    print(json.dumps({'status': 'exo_and_background_verified', 'sha256': args.sha256,
                      'fps': 25, 'size': [784, 448], 'timing': 'i/25',
                      'camera': camera_check(args.video)}, indent=2))


if __name__ == '__main__':
    main()
