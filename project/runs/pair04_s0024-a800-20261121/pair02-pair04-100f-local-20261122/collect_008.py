"""Collect actual available EXO-only outputs; never delete remote evidence."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.run.resolve(strict=True)
 if a.output.exists():raise FileExistsError('Evidence archive exists; choose a new name')
 files=[]
 for f in root.rglob('*'):
  if not f.is_file() or f.is_symlink():continue
  rel=f.relative_to(root); parts=rel.parts
  if parts[0]=='legacy' or parts[0]=='candidates' or parts[0]=='inference-input':continue
  if any(x in parts for x in ('cache','tmp','depth-conversion')):continue
  if 'geometry' in parts:
   relstr=rel.as_posix()
   if not any(relstr.endswith('/'+x) for x in ('depth/exo.zip','mask/exo.zip','mask/exo.txt','intrinsics/exo.npz','pose/exo.npz','rgb/exo.mp4')) and f.name!='vipe.log':continue
  if f.suffix not in ('.json','.log','.txt','.mp4','.npz','.zip','.png','.jpg') and f.name!='runtime-preflight.log':continue
  files.append(f)
 files.sort(); index={f.relative_to(root).as_posix():{'sha256':digest(f),'bytes':f.stat().st_size} for f in files}
 with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  for f in files:z.write(f,f.relative_to(root).as_posix())
  z.writestr('collection-manifest.json',json.dumps({'status':'actual available evidence; generation and visual acceptance may be incomplete','files':index},indent=2)+'\n')
 with zipfile.ZipFile(a.output) as z:
  if z.testzip() is not None:raise ValueError('Archive CRC check failed')
  for name,row in index.items():
   h=hashlib.sha256()
   with z.open(name) as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
   if h.hexdigest()!=row['sha256']:raise ValueError(f'Archive differs: {name}')
 print(json.dumps({'archive':str(a.output),'sha256':digest(a.output),'files':len(index),'bytes':a.output.stat().st_size,'collection_verified':True,'cloud_shutdown':'not performed'}))
if __name__=='__main__':main()
