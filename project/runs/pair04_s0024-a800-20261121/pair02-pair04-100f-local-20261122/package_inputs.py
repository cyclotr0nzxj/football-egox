"""Package EXO windows, corrected annotations and old reference geometry only."""
from pathlib import Path
import hashlib,json,zipfile
P=Path(__file__).resolve().parent; R=P.parent
manifest=json.loads((P/'windows/manifest.json').read_text()); files={}
for row in manifest['windows']:
 name=row['name']; files[f'windows/{name}/exo.mp4']=R/row['exo']; files[f'windows/{name}/exo_head_anchors.json']=P/f'windows/{name}/exo_head_anchors.json'; files[f'windows/{name}/meta-candidate.json']=P/f'windows/{name}/meta-candidate.json'
for pair,geo,view in [
 ('Pair02',R/'multiclip-evidence/pair02_s0000/geometry/vipe_results/pair02_s0000',R/'multiclip-evidence/pair02_s0000/viewpoint'),
 ('Pair04',R/'geometry-008/pair04_s0024',R/'geometry-008/viewpoint-008-upright')]:
 for suffix in ['depth/exo.zip','mask/exo.zip','intrinsics/exo.npz','pose/exo.npz']:
  files[f'legacy/{pair}/{suffix}']=geo/suffix
 for suffix in ['trajectory-check.json','meta.json']:
  files[f'legacy/{pair}/viewpoint/{suffix}']=view/suffix
 prior=(R/'multiclip-evidence/pair02_s0000/prior/pair02_s0000/ego_Prior.mp4' if pair=='Pair02' else R/'geometry-008/prior-008-upright/pair04_s0024/ego_Prior.mp4')
 files[f'legacy/{pair}/prior.mp4']=prior
for pair in ('Pair02','Pair04'):
 files[f'candidates/{pair}-trajectory.json']=P/f'{pair}-trajectory-candidate.json'
 files[f'candidates/{pair}-foot-depth.json']=P/f'{pair}-foot-depth.json'
files['manifest.json']=P/'windows/manifest.json'
for name in ('check_overlap_geometry.py','stage_geometry.py','continue_008.py','assemble_extension.py','verify_egox_outputs_008.py','check_runtime_008.py','collect_008.py'):
 files[name]=P/name
for name,path in files.items():
 if not path.is_file(): raise FileNotFoundError(path)
checks={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
out=P/'exo-geometry-prep.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for name,path in files.items():z.write(path,name)
 z.writestr('SHA256.json',json.dumps(checks,indent=2)+'\n')
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for name,digest in checks.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
receipt={'archive':out.name,'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'files':len(files),'decoded_archive_and_member_hashes_ok':True,'contents':'EXO windows, provisional 2D/3D head candidates, old geometry/viewpoint, overlap consistency script. No Nano, new ViPE geometry, newly rendered prior, or new inference result. Pipeline is locally simulated; actual GPU execution remains unverified.'}
(P/'package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))
