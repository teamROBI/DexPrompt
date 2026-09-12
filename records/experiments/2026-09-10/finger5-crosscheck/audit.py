"""Read saved inputs and end states; no optimisation or policy execution."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path.cwd() / 'scripts'))
from perturb_initialisation import environment  # noqa: E402
from report_missing_digit_gap import load_end_state  # noqa: E402

root = Path('results/2026-09-10/finger5-flexion')
variants = json.loads((root / 'variants.json').read_text())
results = json.loads((root / 'results.json').read_text())
result_index = {(r['sha256'], r['variant']): r for r in results}
assert len(result_index) == 40
rows = []
for candidate in variants['candidates']:
    source = Path(candidate['source_init'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == candidate['source_init_sha256']
    original = np.load(source, allow_pickle=True).item()
    env = environment(str(original['scene_path']))
    q0 = np.asarray(original['grasp_qpos'])[0]
    env.reset_qpos(q0)
    mujoco.mj_forward(env._model, env._data)
    names = list(original['hand_cbody'])
    reference = np.asarray(env.transform_cpn_w2b(names, original['hand_cpn_w']))
    for variant in candidate['variants']:
        row = {'sha256': candidate['source_init_sha256'], 'control': candidate['control'],
               'variant': variant['variant']}
        result = result_index[(row['sha256'], row['variant'])]
        if variant.get('not_run'):
            assert result['not_run'] and variant['path'] is None
            row['not_run'] = variant['not_run']
            rows.append(row)
            continue
        path = Path(variant['path'])
        saved = np.load(path, allow_pickle=True).item()
        q1 = np.asarray(saved['grasp_qpos'])[0]
        changed = np.flatnonzero(q1 != q0).tolist()
        assert changed == ([] if variant['variant'] == 'original' else [24])
        if variant['variant'] == 'original':
            assert path.read_bytes() == source.read_bytes()
        env.reset_qpos(q1)
        mujoco.mj_forward(env._model, env._data)
        actual = np.asarray(env.transform_cpn_w2b(names, saved['hand_cpn_w']))
        delta = float(np.abs(actual-reference).max())
        assert delta <= 1e-6
        state = load_end_state(Path('out/finger5/end-state'), str(path))
        touching = set(str(x) for x in state['contact_bodies'])
        satisfied = [bool(set(group) & touching) for group in saved['required_cbody']]
        assert sum(satisfied) == result['groups_satisfied']
        assert bool(state['passed']) == result['passed_every_filter']
        assert result['passed_every_filter'] == result['refined_file_produced']
        row.update(target_drift=delta, changed_qpos_indices=changed,
                   satisfied=sum(satisfied), required=len(satisfied),
                   passed=bool(state['passed']), filter_status=dict(state['filter_status']))
        rows.append(row)
counts = {c: {'planned':sum(r['control']==c for r in rows),
              'run':sum(r['control']==c and not r.get('not_run') for r in rows),
              'passed':sum(r['control']==c and r.get('passed',False) for r in rows)}
          for c in ('positive', 'negative')}
gaps = json.loads((root/'gaps.json').read_text())
d5 = [r for r in gaps if r['control']=='negative' and r['digit']==5]
assert len(d5) == 28 and not any(r['held'] for r in d5)
output = dict(counts=counts, max_target_drift=max(r.get('target_drift',0) for r in rows),
              negative_digit5_min_gap_mm=min(r['gap_mm'] for r in d5),
              filters=dict(Counter(json.dumps(r['filter_status'],sort_keys=True)
                                   for r in rows if not r.get('not_run'))), rows=rows)
Path('results/2026-09-10/finger5-crosscheck/audit.json').write_text(json.dumps(output,indent=2)+'\n')
print({k:v for k,v in output.items() if k!='rows'})
