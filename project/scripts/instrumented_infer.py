"""Execute unmodified official infer.py, adding observational adapter/resource logs."""
import functools
import json
import os
import pathlib
import resource
import runpy
import sys
import time

sys.path.insert(0,str(pathlib.Path.cwd()))
import torch
from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline

audit={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'events':[],'gt_policy':'GT path removed from inference metadata; no GT video used'}
for method in ['load_lora_weights','fuse_lora']:
    original=getattr(WanWidthConcatImageToVideoPipeline,method)
    def wrap(fn,name):
        @functools.wraps(fn)
        def observed(self,*args,**kwargs):
            result=fn(self,*args,**kwargs)
            event={'operation':name,'active_adapters':self.get_active_adapters(),'peft_config':{key:{'r':value.r,'lora_alpha':value.lora_alpha} for key,value in getattr(self.transformer,'peft_config',{}).items()},'kwargs':kwargs}
            event['component_dtypes']={key:str(getattr(self,key).dtype) for key in ['transformer','vae','text_encoder','image_encoder']}
            assert event['active_adapters'] and event['peft_config'],'Official adapter did not activate'
            assert all(config['r']==256 for config in event['peft_config'].values()),'Unexpected official LoRA rank'
            audit['events'].append(event)
            print('ADAPTER_AUDIT',json.dumps(event),flush=True)
            return result
        return observed
    setattr(WanWidthConcatImageToVideoPipeline,method,wrap(original,method))

def deny_gt(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).lower()
        if name.endswith('ego_gt.mp4'):
            raise RuntimeError('GT read forbidden during inference')
sys.addaudithook(deny_gt)
started=time.monotonic()
try:
    runpy.run_path('infer.py',run_name='__main__')
finally:
    audit.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if torch.cuda.is_initialized():
        audit.update(max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    pathlib.Path(os.environ['EGOX_AUDIT_PATH']).write_text(json.dumps(audit,indent=2))
