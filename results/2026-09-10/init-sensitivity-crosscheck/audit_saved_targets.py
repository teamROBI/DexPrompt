import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / 'scripts'))
from perturb_initialisation import environment

plan = json.loads(Path('results/2026-09-10/init-sensitivity/variants.json').read_text())
rows = []
for c in plan['candidates']:
    original = np.load(c['source_init'], allow_pickle=True).item()
    env = environment(str(original['scene_path']))
    env.reset_qpos(np.asarray(original['grasp_qpos'])[0])
    import mujoco
    mujoco.mj_forward(env._model, env._data)
    bodies = list(original['hand_cbody'])
    reference = np.asarray(env.transform_cpn_w2b(bodies, original['hand_cpn_w']))
    for v in c['variants']:
        saved = np.load(v['path'], allow_pickle=True).item()
        env.reset_qpos(np.asarray(saved['grasp_qpos'])[0])
        mujoco.mj_forward(env._model, env._data)
        actual = np.asarray(env.transform_cpn_w2b(bodies, saved['hand_cpn_w']))
        rows.append({'source': c['source_init'], 'variant': v['variant'],
                     'path': v['path'], 'sha256': hashlib.sha256(Path(v['path']).read_bytes()).hexdigest(),
                     'saved_target_max_abs_delta': float(np.max(np.abs(actual-reference))),
                     'original_qpos_equal': bool(np.array_equal(saved['grasp_qpos'],original['grasp_qpos'])),
                     'stored_world_targets_equal': bool(np.array_equal(saved['hand_cpn_w'],original['hand_cpn_w']))})
print('records', len(rows), 'actual_max_drift', max(r['saved_target_max_abs_delta'] for r in rows))
print('original_variants', [r for r in rows if r['variant']=='original'])
Path('results/2026-09-10/init-sensitivity-crosscheck/saved-targets.json').write_text(json.dumps(rows,indent=2)+'\n')
