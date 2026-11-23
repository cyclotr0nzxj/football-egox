from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil
root=Path('.')
now=datetime.now(ZoneInfo('Asia/Shanghai'))
label='足球第一视角_旧报告_'+now.strftime('%Y%m%d_%H%M%S')
try:
 recycle=Path.home()/'.Trash'/label
 recycle.mkdir(parents=True,exist_ok=False)
except PermissionError:
 recycle=root.parent/'.项目清理回收'/label
 recycle.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
latest=root/'output/week08_harris_sift/第8周_Harris与SIFT实验报告.pdf'
packaged=root/'submission/week08_harris_sift/report/第8周_Harris与SIFT实验报告.pdf'
zipfile=root/'dist/第8周_Harris与SIFT_提交包.zip'
before={'report_sha256':sha(latest),'zip_sha256':sha(zipfile)}
assert sha(packaged)==before['report_sha256']
# Old generated deliverables only. Experiment inputs/results and footage stay put.
retire=[
 'dist/week08_harris_sift',
 'output/week08_harris_sift/history',
 'output/week08/第8周实验报告.pdf',
 'output/week08/实验报告.md',
 'output/week08/report_data.json',
 'tmp/week08_paper_revision',
 'tmp/week08_report_revision',
]
rows=[]
for rel in retire:
 p=root/rel
 if not p.exists():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(root)
 dest=recycle/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 count=0;size=0
 files=[q for q in p.rglob('*') if q.is_file()] if p.is_dir() else [p]
 for q in files:count+=1;size+=q.stat().st_size
 pdfs=[{'path':str(q.relative_to(root)),'sha256':sha(q)} for q in files if q.suffix.lower()=='.pdf']
 shutil.move(str(p),str(dest))
 rows.append({'old_project_path':rel,'recoverable_path':str(dest),'file_count':count,'bytes':size,'pdfs':pdfs})
# Equal PDF snapshots can share an inode while preserving both workflow paths.
linked=False
if not os.path.samefile(latest,packaged):
 temp=packaged.with_name(packaged.name+'.cleanup-hardlink')
 assert not temp.exists()
 os.link(latest,temp);os.replace(temp,packaged);linked=True
assert sha(latest)==sha(packaged)==before['report_sha256']
assert sha(zipfile)==before['zip_sha256']
metadata=[]
for base in ['output','dist','submission','scripts']:
 for p in (root/base).rglob('.DS_Store'):
  if p.is_file():p.unlink();metadata.append(str(p.relative_to(root)))
record={'status':'passed','created':now.isoformat(),'scope':'Remove obsolete generated report versions and stale distribution extraction; keep latest report, repository, ZIP, footage and reproducible evidence','recovery_root':str(recycle),'retired':rows,'finder_metadata_removed':metadata,'latest_report':str(latest.relative_to(root)),'submission_pdf_same_inode_as_output':os.path.samefile(latest,packaged),'duplicate_pdf_replaced_with_hardlink':linked,'latest_report_sha256':before['report_sha256'],'latest_zip_sha256':before['zip_sha256'],'latest_contents_changed':False}
(root/'records/project_cleanup_20261021/manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','retired_items':len(rows),'retired_files':sum(x['file_count'] for x in rows),'latest_report_unchanged':True,'latest_zip_unchanged':True,'report_pdf_same_inode':True,'recovery_root':str(recycle)},ensure_ascii=False))
