"""Verify source mapping and EXO/anchor mismatches cannot reach the renderer."""
from pathlib import Path
import json,tempfile
import continue_008 as app
P=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as t:
 run=Path(t);w=run/'windows/pair02_s0028';w.mkdir(parents=True);(w/'exo.mp4').write_bytes(b'fake-exo');(w/'exo_head_anchors.json').write_text(json.dumps({'name':'pair02_s0028','video_sha256':'0'*64}));(w/'overlap.json').write_text(json.dumps({'passed_provisional_gate':True,'scale_to_old':1,'old_start':1,'new_start':28}))
 app.call=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Renderer reached'))
 try:app.render_pair(run,'Pair02',{})
 except ValueError as e:assert 'source frame mapping' in str(e)
 else:raise AssertionError('Misaligned window accepted')
 (w/'overlap.json').write_text(json.dumps({'passed_provisional_gate':True,'scale_to_old':1,'old_start':0,'new_start':28}))
 try:app.render_pair(run,'Pair02',{})
 except ValueError as e:assert 'anchors do not match' in str(e)
 else:raise AssertionError('Wrong anchor video accepted')
 result={'wrong_source_window_rejected_before_renderer':True,'wrong_exo_anchor_hash_rejected_before_renderer':True}
 (P/'rejection-selftest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
