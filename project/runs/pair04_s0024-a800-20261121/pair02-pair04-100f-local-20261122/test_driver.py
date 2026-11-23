"""Exercise driver control flow using fake commands, without cloud or GPU."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;out=[]
for fail_pair in (None,'Pair02'):
 with tempfile.TemporaryDirectory() as t:
  root=Path(t);base=root/'base';scripts=root/'scripts';scripts.mkdir();(base/'renderer-env/bin').mkdir(parents=True);(base/'renderer-env/bin/python').symlink_to(Path(sys.executable));run=base/'pair02-pair04-100f-geometry-prep'
  driver=(P/'run_all_after_boot.sh').read_text().replace('base=/root/autodl-tmp/football_egox/pair04-20261121',f'base={base}');(scripts/'run.sh').write_text(driver)
  prep=f'''#!/usr/bin/env bash
set -e
mkdir -p "{run}/windows/pair02_s0028" "{run}/windows/pair02_s0052" "{run}/windows/pair04_s0052" "{run}/windows/pair04_s0076"
printf 'EGOX_RUNTIME_FINGERPRINT fake\\n' > "{run}/runtime-preflight.log"
for name in pair02_s0028 pair02_s0052 pair04_s0052 pair04_s0076; do printf '{{}}\\n' > "{run}/windows/$name/overlap.json"; done
cp "{scripts}/fake_continue.py" "{run}/continue_008.py"
cp "{scripts}/fake_collect.py" "{run}/collect_008.py"
''';(scripts/'prepare_vipe_after_boot.sh').write_text(prep)
  cont=f'''import sys
from pathlib import Path
pair=sys.argv[sys.argv.index('--pair')+1]
Path({str(root/'calls')!r}).open('a').write(pair+'\\n')
if pair=={fail_pair!r}:raise SystemExit(7)
''';(scripts/'fake_continue.py').write_text(cont)
  (scripts/'fake_collect.py').write_text("import sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--output')+1]).write_text('simulated')\nprint('MOCK_COLLECTED')\n")
  result=subprocess.run(['bash',str(scripts/'run.sh'),'dummy'],capture_output=True,text=True);calls=(root/'calls').read_text().splitlines();assert calls==['Pair02','Pair04'];assert (result.returncode==0)==(fail_pair is None);assert 'EVIDENCE_ARCHIVE' in result.stdout
  second=subprocess.run(['bash',str(scripts/'run.sh'),'dummy'],capture_output=True,text=True);assert second.returncode!=0 and 'existing run' in second.stderr
  out.append({'forced_failure':fail_pair,'both_pairs_attempted':True,'evidence_collection_attempted':True,'existing_run_rejected':True,'returncode':result.returncode})
(P/'driver-selftest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
