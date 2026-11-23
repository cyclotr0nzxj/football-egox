"""Audited launcher: one official seed, at most three total starts, no auto retries."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path('/root/autodl-tmp/football_egox')
TAKE = 'cmu_soccer06_6_877_925'

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reason',required=True,help='Initial baseline or log-supported repair reason')
    args=parser.parse_args()
    lock=(ROOT/'records/inference.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempts_path=ROOT/'records/attempts.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    assert len(attempts)<3, 'Authorized three starts exhausted'
    assert not any(a.get('status')=='success' for a in attempts),'Already succeeded: reuse output; no new sampling'
    assert not any(a.get('status')=='running' for a in attempts),'Resolve existing running attempt before retry'
    preflight=json.loads((ROOT/'records/preflight.json').read_text())
    assert preflight['cpu_imports_passed'] and preflight['cuda_smoke']=='passed','Preflight not passed'
    verified=json.loads((ROOT/'records/model_verification.json').read_text())
    expected=sum(len(model['files']) for model in json.loads((ROOT/'records/download_plan.json').read_text())['models'])
    assert len(verified)==expected and all(x['official_hash_verified'] for x in verified)
    assert (ROOT/'records/sample_verification.json').exists()
    number=len(attempts)+1
    out=ROOT/'runs'/f'attempt_{number:02d}'
    out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(ROOT/'scripts/instrumented_infer.py'),
         '--meta_data_file','./example/egoexo4D/football_meta.json',
         '--model_path',str(ROOT/'models/Wan2.1-I2V-14B-480P-Diffusers'),
         '--lora_path',str(ROOT/'models/EgoX/pytorch_lora_weights.safetensors'),
         '--lora_rank','256','--out',str(out),'--seed','42','--use_GGA','--cos_sim_scaling_factor','3.0']
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
               PYTHONUNBUFFERED='1',HF_HOME=str(ROOT/'cache/huggingface'),TMPDIR=str(ROOT/'tmp'),
               EGOX_AUDIT_PATH=str(out/'runtime_audit.json'),OMP_NUM_THREADS='8')
    entry={'attempt':number,'reason':args.reason,'started_utc':utc(),'status':'running','command':cmd,
           'cwd':str(ROOT/'EgoX'),'seed':42,'official_commit':'43c095b552a8f09d3a81c0882491f52c5cf4e953',
           'settings':{'frames':49,'steps':50,'guidance_scale':5.0,'fps':30,'ego_size':[448,448],'exo_size':[784,448],'GGA':True,'cos_sim_scaling_factor':3.0,'kv_cache':False,'quantized':False},
           'gt_used_for_inference':False,'output_directory':str(out)}
    attempts.append(entry)
    attempts_path.write_text(json.dumps(attempts,indent=2))
    (out/'config.json').write_text(json.dumps(entry,indent=2))
    stopped=threading.Event()
    def monitor():
        with (out/'resources.jsonl').open('w') as log:
            while not stopped.is_set():
                stats={'utc':utc()}
                try:
                    stats['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
                    stats['cgroup_memory_bytes']=int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text())
                except Exception as exc:stats['error']=str(exc)
                log.write(json.dumps(stats)+'\n');log.flush();stopped.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    started=time.monotonic()
    with (out/'inference.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT/'EgoX',env=env,stdout=log,stderr=subprocess.STDOUT)
    stopped.set();thread.join()
    entry.update(ended_utc=utc(),elapsed_seconds=time.monotonic()-started,exit_code=result.returncode)
    output=out/f'{TAKE}.mp4'
    output_valid=False
    if result.returncode==0 and output.is_file() and output.stat().st_size>0:
        import cv2
        cap=cv2.VideoCapture(str(output));fps=cap.get(cv2.CAP_PROP_FPS);count=0;shapes=set()
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1;shapes.add(frame.shape)
        cap.release()
        entry['output_probe']={'frames':count,'fps':fps,'shapes':[list(shape) for shape in shapes]}
        output_valid=count==49 and fps==30 and shapes=={(448,1232,3)}
    entry['status']='success' if output_valid else 'failed'
    if entry['status']=='success':entry['output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    attempts_path.write_text(json.dumps(attempts,indent=2))
    (out/'config.json').write_text(json.dumps(entry,indent=2))
    print(json.dumps(entry,indent=2),flush=True)
    sys.exit(0 if entry['status']=='success' else 1)

if __name__=='__main__':main()
