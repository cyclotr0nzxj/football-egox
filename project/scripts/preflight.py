"""CPU integrity/import checks and a separately recorded tiny CUDA smoke check."""
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys

ROOT=pathlib.Path('/root/autodl-tmp/football_egox')
os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(ROOT/'EgoX'))

def main():
    import numpy as np
    import torch
    import torchvision
    from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline
    from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA
    from core.finetune.datasets.utils import iproj_disp
    hashes=json.loads((ROOT/'records/official_code_hashes.json').read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((ROOT/'EgoX'/name).read_bytes()).hexdigest()==expected,f'Official code changed: {name}'
    meta=json.loads((ROOT/'EgoX/example/egoexo4D/football_meta.json').read_text())['test_datasets']
    assert len(meta)==1 and not any('gt' in key.lower() for key in meta[0])
    data=ROOT/'EgoX/example/egoexo4D/depth_maps/cmu_soccer06_6_877_925'
    paths=sorted(data.glob('*.npy'));assert len(paths)==49
    for path in paths:
        array=np.load(path);assert array.shape==(448,796) and np.isfinite(array).all()
    assert np.asarray(meta[0]['ego_extrinsics']).shape==(49,3,4)
    check={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_imports_passed':True,
        'official_code_files_verified':len(hashes),'sample_structure_passed':True,'python':sys.version,
        'versions':{p:importlib.metadata.version(p) for p in ['torch','torchvision','diffusers','transformers','accelerate','peft','numpy','decord','scikit-image']},
        'cuda_smoke':'not_started','diffusion_inference_started':False}
    (ROOT/'records/preflight.json').write_text(json.dumps(check,indent=2))
    operation={'utc':check['utc'],'type':'cuda_smoke','description':'128x128 BF16 ones matrix product; no model, no video generation','status':'started'}
    log=ROOT/'records/gpu_operations.jsonl'
    with log.open('a') as stream:stream.write(json.dumps(operation)+'\n')
    assert torch.cuda.is_available()
    tensor=torch.ones((128,128),device='cuda',dtype=torch.bfloat16)
    value=(tensor@tensor).float().sum().item()
    assert value==2097152.0
    torch.cuda.synchronize()
    check.update(cuda_smoke='passed',cuda_result=value,cuda_runtime=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0))
    operation['status']='passed'
    with log.open('a') as stream:stream.write(json.dumps(operation)+'\n')
    (ROOT/'records/preflight.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
