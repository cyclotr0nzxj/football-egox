"""Exercise the retry scheduler on fake processes and legacy evidence, without a GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ('pair02_s0028', 'pair02_s0052', 'pair04_s0052', 'pair04_s0076')

def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)

def case(failing=None, frame_outlier=None):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = root / 'base'
        run = base / 'pair02-pair04-100f-geometry-prep'
        binary = base / 'renderer-env/bin'
        binary.mkdir(parents=True)
        executable(binary / 'python', '#!/bin/sh\nexec '+sys.executable+' "$@"\n')
        envbin = base / 'egox-shared-008/egox-env/bin'
        envbin.mkdir(parents=True)
        executable(envbin / 'python', '#!/bin/sh\nexec '+sys.executable+' "$@"\n')
        (run / 'check_runtime_008.py').parent.mkdir(parents=True)
        (run / 'check_runtime_008.py').write_text("print('MOCK_RUNTIME_OK')\n")
        (run / 'continue_008.py').write_text('''import json,sys
from pathlib import Path
run=Path(sys.argv[sys.argv.index('--run')+1]);pair=sys.argv[sys.argv.index('--pair')+1]
name={'Pair02':'pair02_s0028','Pair04':'pair04_s0052'}[pair]
win=run/'windows'/name
geom=win/'geometry'/name
report=json.loads((win/'overlap.json').read_text())
assert report['passed_provisional_gate'] and geom.parent.name=='geometry-fixed'
with (run/'calls').open('a') as f:f.write(pair+'\\n')
''')
        (run / 'collect_008.py').write_text('''import sys
from pathlib import Path
Path(sys.argv[sys.argv.index('--output')+1]).write_text('mock archive')
print('MOCK_EVIDENCE_COLLECTED')
''')
        (run / 'check_overlap_geometry.py').write_text('''import json,os,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output')+1]);name=out.parent.name
passed=name!=os.environ.get('MOCK_FAIL_WINDOW')
rows=[{'median_relative_error':0.01,'p90_relative_error':0.02} for _ in range(12)]
if name==os.environ.get('MOCK_FRAME_OUTLIER'):
    rows[-1]={'median_relative_error':0.10,'p90_relative_error':0.41}
out.write_text(json.dumps({'passed_provisional_gate':passed,'scale_to_old':1.0,'heldout_frames':rows}))
raise SystemExit(0 if passed else 2)
''')
        for pair in ('Pair02', 'Pair04'):
            k = run / 'legacy' / pair / 'intrinsics'
            k.mkdir(parents=True)
            np.savez(k / 'exo.npz', data=np.tile([700., 700., 392., 224.], (49, 1)))
        for name in NAMES:
            win = run / 'windows' / name
            win.mkdir(parents=True)
            (win / 'exo.mp4').write_text('mock')
            (win / 'overlap.json').write_text('historical evidence')
        executable(binary / 'vipe', '''#!/usr/bin/env python3
import json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output')+1]); name=out.parent.name
(out/name).mkdir(parents=True)
k=json.loads(sys.argv[sys.argv.index('--use_exo_intrinsic_gt')+1]); assert k[0][0]==700.0 and k[2][2]==1
''')
        executable(binary / 'nvidia-smi', '#!/bin/sh\necho "NVIDIA A800 80GB PCIe, 81920, 0"\n')
        executable(binary / 'flock', '#!/bin/sh\nexit 0\n')
        executable(binary / 'timeout', '#!/bin/sh\nshift 3\nexec "$@"\n')
        executable(binary / 'df', '#!/bin/sh\necho "Filesystem 1024-blocks Used Available Capacity Mounted"\necho "mock 100000000 1000 99999000 1% /"\n')
        reused = {}
        for name in ('legacy/Pair02/depth/exo.zip', 'legacy/Pair04/depth/exo.zip',
                     'legacy/Pair02/prior.mp4', 'legacy/Pair04/prior.mp4'):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            reused[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (run / 'legacy-reuse.json').write_text(json.dumps(reused))
        listed = [f for f in run.rglob('*') if f.is_file() and f.name != 'overlap.json'
                  and f.relative_to(run).as_posix() not in reused]
        checks = {f.relative_to(run).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in listed}
        archive = base/'egox-shared-008/pair05-vae-66fc463f.FlNTz4tU/exo-geometry-ready.zip'
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive,'w') as z:
            for f in listed: z.write(f, f.relative_to(run).as_posix())
            z.writestr('SHA256.json', json.dumps(checks))
        mock_archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
        driver = (HERE / 'run_fixed_k_008.sh').read_text().replace('base=/root/autodl-tmp/football_egox/pair04-20261121', f'base={base}')
        driver = driver.replace('477153e0d753ba961e3324197db198e78d32165dfdd07c0e16050184aca64743', mock_archive_hash)
        script = root / 'driver.sh'
        script.write_text(driver)
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'], MOCK_FAIL_WINDOW=failing or '',
                   MOCK_FRAME_OUTLIER=frame_outlier or '')
        if failing is None and frame_outlier is None:
            checked = run / 'windows' / NAMES[0] / 'exo.mp4'
            original = checked.read_bytes()
            checked.write_bytes(b'corrupted mock input')
            rejected = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=env)
            assert rejected.returncode != 0 and 'FIXED_K_BLOCKED input integrity' in rejected.stderr
            assert not (run / 'fixed-k-attempt').exists()
            checked.write_bytes(original)
        result = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=env)
        expected = ['Pair04'] if failing or frame_outlier else ['Pair02', 'Pair04']
        assert (run / 'calls').exists(), (result.returncode, result.stdout, result.stderr)
        calls = (run / 'calls').read_text().splitlines()
        assert calls == expected, (calls, result.stdout, result.stderr)
        assert (result.returncode == 0) == (failing is None and frame_outlier is None), (result.returncode, result.stderr)
        if frame_outlier:
            path = run / 'windows' / frame_outlier / 'overlap-frame-guard.json'
            assert json.loads(path.read_text())['passed'] is False
            assert 'FIXED_K_FRAME_GUARD_FAILED' in result.stderr
        assert all((run/'windows'/name/'overlap.json').read_text() == 'historical evidence' for name in NAMES)
        assert 'MOCK_EVIDENCE_COLLECTED' in result.stdout
        second = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=env)
        assert second.returncode != 0 and 'existing or missing run' in second.stderr
        return {'failed_window':failing, 'frame_outlier_window':frame_outlier, 'pairs_continued':calls, 'legacy_evidence_unchanged':True,
                'evidence_collected':True, 'duplicate_rejected':True, 'corrupted_input_rejected': True if failing is None and frame_outlier is None else None, 'returncode':result.returncode}

results = [case(), case(failing='pair02_s0028'), case(frame_outlier='pair02_s0028')]
(HERE/'fixed-k-driver-selftest.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
