"""Prepare a smaller launch bundle, reusing four hash-verified remote legacy files."""
from pathlib import Path
import hashlib,json,re,zipfile
P=Path(__file__).resolve().parent;base='/root/autodl-tmp/football_egox/pair04-20261121/'
reuse={'legacy/Pair02/depth/exo.zip':base+'multiclip-008/pair02_s0000/geometry/vipe_results/pair02_s0000/depth/exo.zip','legacy/Pair04/depth/exo.zip':base+'first_vipe/vipe_results/pair04_s0024/depth/exo.zip','legacy/Pair02/prior.mp4':base+'multiclip-008/pair02_s0000/prior/pair02_s0000/ego_Prior.mp4','legacy/Pair04/prior.mp4':base+'prior-008-upright/pair04_s0024/ego_Prior.mp4'}
with zipfile.ZipFile(P/'exo-geometry-prep.zip') as source:
 checks=json.loads(source.read('SHA256.json'));remote={k:{'path':v,'sha256':checks[k]} for k,v in reuse.items()}; payload={n:source.read(n) for n in checks if n not in reuse}
 payload['legacy-reuse.json']=(json.dumps(remote,indent=2)+'\n').encode(); manifests={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}
 with zipfile.ZipFile(P/'exo-geometry-lite.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for n,data in payload.items():z.writestr(n,data)
  z.writestr('SHA256.json',json.dumps(manifests,indent=2)+'\n')
with zipfile.ZipFile(P/'exo-geometry-lite.zip') as z:
 assert z.testzip() is None
 for n,sha in manifests.items():assert hashlib.sha256(z.read(n)).hexdigest()==sha
sha=hashlib.sha256((P/'exo-geometry-lite.zip').read_bytes()).hexdigest();s=(P/'prepare_vipe_after_boot.sh').read_text();s=re.sub(r'^expected=[a-f0-9]{64}$','expected='+sha,s,flags=re.M)
needle="PY\n# Check the complete pinned EgoX environment"; insert="""PY
# Restore only the four explicitly named legacy assets after verifying their hashes.
\"$python\" -I - \"$run\" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=json.loads((root/'legacy-reuse.json').read_text())
expected={'legacy/Pair02/depth/exo.zip','legacy/Pair04/depth/exo.zip','legacy/Pair02/prior.mp4','legacy/Pair04/prior.mp4'}
if set(rows)!=expected: raise ValueError('Unexpected reused asset set')
for name,row in rows.items():
 source=Path(row['path']); h=hashlib.sha256()
 with source.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
 if h.hexdigest()!=row['sha256']: raise ValueError('Remote legacy asset hash mismatch: '+name)
 dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as src,dest.open('xb') as out: shutil.copyfileobj(src,out)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
 if h.hexdigest()!=row['sha256']: raise ValueError('Copied asset differs: '+name)
print('REMOTE_LEGACY_ASSETS_VERIFIED')
PY
# Check the complete pinned EgoX environment""";assert needle in s;s=s.replace(needle,insert);(P/'prepare_vipe_lite.sh').write_text(s)
s=(P/'run_all_after_boot.sh').read_text().replace('prepare_vipe_after_boot.sh','prepare_vipe_lite.sh');(P/'run_all_lite.sh').write_text(s)
for x in ('prepare_vipe_lite.sh','run_all_lite.sh'):(P/x).chmod(0o755)
receipt={'full_sha256':hashlib.sha256((P/'exo-geometry-prep.zip').read_bytes()).hexdigest(),'lite_sha256':sha,'lite_bytes':(P/'exo-geometry-lite.zip').stat().st_size,'remote_reuse':remote};(P/'lite-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
