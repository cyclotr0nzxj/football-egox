"""Retrieve only one official sample and verify Git blob IDs from the pinned tree."""
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root',default='/root/autodl-tmp/football_egox')
ROOT = pathlib.Path(parser.parse_args().root).resolve()
REV = '43c095b552a8f09d3a81c0882491f52c5cf4e953'
rows = (ROOT/'records/official_sample_tree.txt').read_text().splitlines()

def fetch(row):
    info, name = row.split('\t')
    expected = info.split()[2]
    path = ROOT/'EgoX'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        subprocess.run(['curl','-fLsS','--retry','5','--connect-timeout','15','--max-time','180',f'https://raw.githubusercontent.com/DAVIAN-Robotics/EgoX/{REV}/{name}','-o',str(path)],check=True)
    content = path.read_bytes()
    sha = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
    assert sha == expected, f'Git blob mismatch: {name}'
    print('VERIFIED',name,len(content),flush=True)
    return {'path':name,'bytes':len(content),'git_blob':sha,'sha256':hashlib.sha256(content).hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    verified = list(pool.map(fetch,rows))
(ROOT/'records/sample_verification.json').write_text(json.dumps(verified,indent=2))
meta = json.loads((ROOT/'EgoX/example/egoexo4D/meta.json').read_text())
selected = [m for m in meta['test_datasets'] if '/cmu_soccer06_6_877_925/' in m['exo_path']]
assert len(selected)==1
# The official metadata has an ego_gt_path used for display; infer.py never reads it.
# Remove that unused path from the actual inference manifest as defense in depth.
selected[0].pop('ego_gt_path',None)
assert not any('gt' in k.lower() or k=='ego_path' for k in selected[0]), 'Unexpected GT field'
(ROOT/'EgoX/example/egoexo4D/football_meta.json').write_text(json.dumps({'test_datasets':selected},indent=2))
print('FOOTBALL_SAMPLE_READY',flush=True)
