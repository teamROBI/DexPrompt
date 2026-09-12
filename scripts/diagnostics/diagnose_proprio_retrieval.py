"""Non-visual nearest-demo control, strictly a known-task diagnostic.

Policy input is current measured joint position only. No time index, target object
pose, forces, or teacher phase is used for retrieval. This is not a VLA result.
"""
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
from collections import deque
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from replay_hx5_student import restore_recorder, rollout
    from icra.dataset import load_episode, load_manifest

    p=argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=load_manifest(args.data,require_train=False)
    if len(manifest['episodes'])!=1:
        raise ValueError('single known task only')
    entry=manifest['episodes'][0]
    demo=load_episode(args.data,entry)
    run=json.loads(args.run.read_text())
    scale=np.asarray(run['normalizer']['state_scale'])
    execute=run['config']['execute_steps']
    states=np.asarray(demo['right_state'],dtype=float)
    labels=np.asarray(demo['action'],dtype=float)
    r,initial,ticks=restore_recorder(args.data,manifest,entry,render=False)
    indices=[]

    class Retrieval:
        saturated_ticks=0

        def reset(self,seed):
            self.pending=deque()

        def command_recorder(self,recorder,*unused,**kwargs):
            if not self.pending:
                current=recorder.measured()
                distance=np.mean(((states-current)/scale)**2,axis=1)
                index=int(np.argmin(distance))
                indices.append({'policy_tick':self.ticks,'nearest_index':index,
                                'normalized_state_rmse':float(np.sqrt(distance[index]))})
                for offset in range(execute):
                    self.pending.append(labels[min(index+offset,len(labels)-1)].copy())
            action=self.pending.popleft()
            low,high=recorder.model.actuator_ctrlrange[recorder.actuators].T
            self.saturated_ticks+=int(np.any((action<low)|(action>high)))
            action=np.clip(action,low,high)
            recorder.command(action)
            self.ticks+=1
            return {},action

    actor=Retrieval()
    actor.ticks=0
    outcome,trace=rollout(r,actor,entry,initial,ticks)
    report={'completed':True,'training_started':False,
            'scope':'nonvisual nearest-neighbour imitation on the same training task; not VLA',
            'execute_steps':execute,'policy_input':'measured 27-joint position',
            'outcome':outcome,'retrievals':indices}
    (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.out/'trace.npz',**trace)
    print(json.dumps({'outcome':outcome,'queries':len(indices)}))


if __name__=='__main__':
    main()
