"""Headless control diagnostic on the pinned archived model; never a dataset."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/yspark/projects/ICRA')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from replay_hx5_student import restore_recorder
from icra.arm import Arm
from icra.execution import TeacherExecutor, ExecutionConfig, ContactMeter

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--config', type=Path)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
root = ROOT/'out/teacher-preparation-v4'
manifest = json.loads((root/'manifest.json').read_text())
entry = manifest['rejected'][0]
r, initial, _ = restore_recorder(root, manifest, entry, render=False)
r.arm = Arm(r.model, r.data)
row = json.loads((ROOT/'results/2026-09-08/fixed-student/selection.json').read_text())['candidates'][0]
config = ExecutionConfig(**(json.loads(args.config.read_text()) if args.config else {}))

class Probe(TeacherExecutor):
    def tick(self, phase, action, *, record):
        before = self.r.measured().copy()
        try:
            result = super().tick(phase, action, record=False)
        except RuntimeError as error:
            if phase == 'feedback':
                self.trials.append({'failure': str(error)})
            raise
        if phase == 'feedback':
            self.trials.append({'passed': True})
        if record:
            self.trace.append({'action': action.copy(), 'right_state': before,
                               'phase': phase, **result})
        return result

    def guarded(self, proposed, **kwargs):
        self.trials = []
        try:
            action = super().guarded(proposed, **kwargs)
        except RuntimeError:
            self.guards.append({'trials': self.trials, 'stopped': True})
            raise
        self.guards.append({'trials': self.trials, 'arm': self.guard_fractions[-1],
                            'fingers': self.guard_finger_fractions[-1],
                            'depths': self.meter.read(control=True)['digit_penetration_max'].tolist()})
        return action

e = Probe(r, row, config=config)
e.trace, e.trials, e.guards = [], [], []
e.initial, e.initial_clear = initial, True
e.target = np.array(entry['provenance']['execution']['original_target']['world_root'])
e.meter = ContactMeter(r, e.body, row['required'])
e.initial_signals = e.meter.read()
e.run(((0, 1),))
arrays = {k: np.stack([x[k] for x in e.trace]) for k in e.trace[0]}
np.savez_compressed(args.out/'trace.npz', **arrays)
summary = {'scope': 'headless fixed-model feedback diagnostic; no training images or dataset',
           'status': e.status, 'ticks': len(e.trace),
           'max_rise': float(arrays['target_height_min'].max()-e.initial_signals['target_height_min']),
           'depth': float(arrays['robot_object_penetration_max'].max()),
           'force': float(arrays['digit_force_max'].max()),
           'max_groups': int(arrays['contact_groups'].sum(axis=1).max()),
           'hold_ticks': int(np.sum(arrays['phase']=='hold'))}
(args.out/'summary.json').write_text(json.dumps(summary, indent=2))
(args.out/'guards.json').write_text(json.dumps(e.guards, indent=2))
print(json.dumps(summary), flush=True)
