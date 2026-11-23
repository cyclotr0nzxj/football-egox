"""Verify and unpack an evidence archive retrieved from the explicitly authorized 008."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def hashfile(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);p.add_argument('--sha256',required=True);a=p.parse_args()
 if len(a.sha256)!=64 or hashfile(a.archive)!=a.sha256:raise ValueError('Archive SHA-256 differs from remote receipt')
 if a.output.exists():raise FileExistsError('Output exists; choose a new directory')
 with zipfile.ZipFile(a.archive) as z:
  if z.testzip() is not None:raise ValueError('Archive CRC failure')
  manifest=json.loads(z.read('collection-manifest.json'));files=manifest['files']
  if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(files)|{'collection-manifest.json'}:raise ValueError('Unexpected or duplicate archive members')
  for name,row in files.items():
   rel=Path(name)
   if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive member')
   info=z.getinfo(name)
   if info.file_size!=row['bytes']:raise ValueError('Member length differs from manifest')
   h=hashlib.sha256()
   with z.open(name) as source:
    for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
   if h.hexdigest()!=row['sha256']:raise ValueError(f'Member SHA-256 differs: {name}')
  a.output.mkdir(parents=True)
  for name in files:
   target=a.output/Path(name);target.parent.mkdir(parents=True,exist_ok=True)
   with z.open(name) as source,target.open('xb') as dest:
    for block in iter(lambda:source.read(1024*1024),b''):dest.write(block)
  (a.output/'collection-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  (a.output/'retrieval-receipt.json').write_text(json.dumps({'archive_sha256':a.sha256,'members':len(files),'status':'archive_and_each_member_verified','semantic_acceptance':'pending'},indent=2)+'\n')
 print(json.dumps({'members':len(files),'output':str(a.output),'status':'archive_and_each_member_verified'}))
if __name__=='__main__':main()
