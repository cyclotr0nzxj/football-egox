"""Record the exact official conditions passed to the single baseline."""
import hashlib
import json
from pathlib import Path

root=Path('/root/autodl-tmp/football_egox')
meta_path=root/'EgoX/example/egoexo4D/football_meta.json'
meta=json.loads(meta_path.read_text())['test_datasets']
assert len(meta)==1 and 'ego_gt_path' not in meta[0]
item=meta[0]
file_hash=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result={'sample':'cmu_soccer06_6_877_925','inference_metadata_sha256':file_hash(meta_path),
    'provided_conditions':item,'source':'Pinned official EgoX sample; prompt, geometry, trajectory and prior preserved; only GT path excluded from inference metadata.',
    'camera_trajectory_count':len(item['ego_extrinsics']),
    'automatic_head_estimation':False,'prior_recomputed':False,'gt_used':False,
    'video_sha256':{k:file_hash(root/'EgoX'/item[k]) for k in ('exo_path','ego_prior_path')}}
(root/'records/input_conditions.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print('Recorded exact official conditions and input hashes')
