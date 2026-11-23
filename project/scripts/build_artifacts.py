"""Build exact-index triptych and native-grid descriptive metrics from a real run."""
import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

TAKE='cmu_soccer06_6_877_925'

def decode(path):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    assert frames, f'Cannot decode {path}'
    return np.stack(frames),fps

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/root/autodl-tmp/football_egox')
    parser.add_argument('--alignment-review',required=True,help='JSON recording a completed native image-plane review')
    args=parser.parse_args();root=pathlib.Path(args.root).resolve()
    attempts=json.loads((root/'records/attempts.json').read_text())
    successes=[a for a in attempts if a['status']=='success']
    assert len(successes)==1,'Exactly one real successful run required'
    run=successes[0];out=pathlib.Path(run['output_directory'])
    raw=out/f'{TAKE}.mp4'
    assert sha(raw)==run['output_sha256'],'Run output changed'
    review=json.loads(pathlib.Path(args.alignment_review).read_text())
    assert review['sample']==TAKE
    assert review['generated_sha256']==sha(raw),'Review must refer to this exact output'
    source=root/'EgoX/example/egoexo4D/videos'/TAKE
    exo,exo_fps=decode(source/'exo.mp4');gt,gt_fps=decode(source/'ego_GT.mp4');combined,gen_fps=decode(raw)
    assert combined.shape==(49,448,1232,3),combined.shape
    assert exo.shape==(50,448,784,3),exo.shape
    assert gt.shape==(50,448,448,3),gt.shape
    assert (exo_fps,gt_fps,gen_fps)==(20,20,30)
    generated=combined[:,: ,784:1232,:].copy()
    exo=exo[:49];gt=gt[:49]
    dest=root/'artifacts';dest.mkdir(exist_ok=True)
    (dest/'frames').mkdir(exist_ok=True)
    (dest/'generated_frames').mkdir(exist_ok=True)
    (dest/'source').mkdir(exist_ok=True)
    for name in ['exo.mp4','ego_GT.mp4']:shutil.copy2(source/name,dest/'source'/name)
    shutil.copy2(raw,dest/'official_output.mp4')
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    video=subprocess.Popen([ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1680x528','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'comparison.mp4')],stdin=subprocess.PIPE)
    rows=[]
    pixel_allowed=bool(review.get('native_target_grid_verified')) and bool(review.get('frame_index_mapping_verified'))
    for i,(x,g,y) in enumerate(zip(exo,gt,generated)):
        frame=np.concatenate([x,g,y],axis=1)
        Image.fromarray(frame).save(dest/'frames'/f'{i:03d}.png')
        Image.fromarray(y).save(dest/'generated_frames'/f'{i:03d}.png')
        labeled=Image.new('RGB',(1680,528),(16,23,19));labeled.paste(Image.fromarray(frame),(0,40));draw=ImageDraw.Draw(labeled)
        for xx,label in [(14,'EXO INPUT'),(798,'EGO GT / REFERENCE'),(1246,'EGOX GENERATED / SEED 42')]:draw.text((xx,10),label,font=font,fill=(235,245,235))
        draw.text((14,496),f'Frame {i:02d}/48 | source {i/20:.3f}s (20 fps) | playback {i/30:.3f}s (30 fps) | index matched, no image warping',font=font,fill=(210,230,210))
        video.stdin.write(np.asarray(labeled).tobytes())
        row={'frame':i,'source_pts_seconds':i/20,'generated_pts_seconds':i/30}
        if pixel_allowed:
            mse=float(np.mean((g.astype(np.float64)-y.astype(np.float64))**2))
            row.update(mse_rgb=mse,psnr_rgb_db=10*np.log10(255**2/mse) if mse>0 else None,
                ssim_rgb=float(structural_similarity(g,y,data_range=255,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
        if i:
            row.update(gt_adjacent_mae=float(np.abs(gt[i].astype(np.float32)-gt[i-1]).mean()),generated_adjacent_mae=float(np.abs(generated[i].astype(np.float32)-generated[i-1]).mean()))
        rows.append(row)
    video.stdin.close();assert video.wait()==0,'Comparison video encoding failed'
    subprocess.run([ffmpeg,'-y','-v','error','-framerate','30','-i',str(dest/'generated_frames/%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'generated_ego.mp4')],check=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (dest/'metrics_per_frame.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    metrics={'pixel_metrics_computed':pixel_allowed,'alignment_review':review,'frames':49,
       'comparison_domain':'Decoded official GT RGB uint8 versus rightmost 448 columns of unmodified official output; full 448x448 including black fisheye border',
       'transformations':'No scaling, rotation, translation, warping, masking, colour correction, temporal shift or best-match search. Select source frames 0..48 per official truncation; extract generated branch x=784..1231.',
       'decoder':f'OpenCV {cv2.__version__} BGR to RGB; metrics before extra delivery encode',
       'aggregation':'Arithmetic mean across 49 per-frame PSNR and SSIM values; PSNR uses 8-bit RGB MSE. SSIM Gaussian sigma=1.5, population covariance, per-channel average.',
       'scope':'Descriptive error in supplied benchmark raster; not a calibration validation or reproduction of paper aggregate results.',
       'lpips':'Not computed; PSNR and SSIM suffice for this basic evaluation. No additional metric model downloaded.',
       'temporal_note':'Adjacent-frame MAE includes camera/object motion and must not be called a flicker metric.'}
    if pixel_allowed:metrics.update(mean_psnr_rgb_db=float(np.mean([r['psnr_rgb_db'] for r in rows])),mean_ssim_rgb=float(np.mean([r['ssim_rgb'] for r in rows])))
    metrics.update(mean_gt_adjacent_mae=float(np.mean([r['gt_adjacent_mae'] for r in rows[1:]])),mean_generated_adjacent_mae=float(np.mean([r['generated_adjacent_mae'] for r in rows[1:]])))
    (dest/'evaluation.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
    check,check_fps=decode(dest/'comparison.mp4');assert check.shape==(49,528,1680,3) and check_fps==30
    manifest={'sample':TAKE,'gpu_inference_completed':True,'fps':30,'source_fps':20,'frame_count':49,'width':1680,'height':448,
       'frames':[f'artifacts/frames/{i:03d}.png' for i in range(49)],
       'run_summary':[f'A800 80GB · 第 {run["attempt"]} 次启动成功 · 推理耗时 {run["elapsed_seconds"]/60:.1f} 分钟','seed 42 · 49 帧 · 50 步 · guidance 5.0 · GGA 3.0','官方 Wan 14B + EgoX LoRA；没有训练、微调、量化或减帧。'],
       'alignment_summary':['源视频 50 帧 / 20fps；遵循官方代码选用前 49 帧。生成和对照回放均为 30fps，按帧编号配对。','生成画面取官方输出右侧 448×448 分支；未做旋转、缩放、掩膜或对齐搜索。']+review.get('summary_zh',[]),
       'observations':['真实结果已生成，逐帧观察记录待写入。'],
       'downloads':[{'label':'下载三路对照视频','path':'artifacts/comparison.mp4'},{'label':'下载生成第一人称','path':'artifacts/generated_ego.mp4'},{'label':'下载官方原始输出','path':'artifacts/official_output.mp4'},{'label':'逐帧指标 CSV','path':'artifacts/metrics_per_frame.csv'},{'label':'评估与对齐记录','path':'artifacts/evaluation.json'}]}
    if pixel_allowed:manifest['alignment_summary'].append(f'原生交付平面：PSNR {metrics["mean_psnr_rgb_db"]:.3f} dB；SSIM {metrics["mean_ssim_rgb"]:.4f}。包括鱼眼黑边，仅代表这个官方样例。')
    else:manifest['alignment_summary'].append('像素平面对齐未充分验证，暂不计算 PSNR / SSIM。')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (dest/'file_hashes.json').write_text(json.dumps({str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='file_hashes.json'},indent=2))
    print(json.dumps(metrics,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
