"""Freeze a three-object diagnostic before training; copy no rendered images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from stage1_dp3.environment import Outcome, SIGNALS

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def guard_from_record(row, arrays):
    height = row['initial_object_poses'][row['object_id']]['position_m'][2]
    outcome = Outcome(row['quality']['limits'], height)
    for i in range(len(arrays['action'])):
        outcome.update({k: arrays[k][i] for k in SIGNALS})
    return outcome.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    contract = ROOT / 'records/experiments/2026-09-12/stage1-bc-data-contract/episodes.jsonl'
    records = [json.loads(x) for x in contract.read_text().splitlines()]
    selected, decisions = [], []
    for obj in ('AlphabetSoup', 'Peaches', 'Raisins'):
        candidates = [r for r in records if r['pool'] == 'default' and r['object_id'] == obj
                      and r['placement'] == 'centre' and r['episode_yaw_deg'] == 0]

        def margin(r):
            q = r['quality']
            return min(1 - q['max_robot_object_penetration'] / q['limits']['max_penetration'],
                       1 - q['max_digit_force'] / q['limits']['max_digit_force'])

        for row in sorted(candidates, key=lambda r: (-margin(r), r['episode_sha256'])):
            with np.load(row['episode'], allow_pickle=False) as z:
                arrays = {k: z[k] for k in (*SIGNALS, 'right_state', 'action', 'timestamp', 'control_dt', 'phase')}
            guard = guard_from_record(row, arrays)
            decisions.append(dict(object=obj, episode=row['episode_sha256'], margin=margin(row), guard=guard))
            if not guard['success']:
                continue
            name = obj + '-' + row['episode_sha256'][:12]
            files = {}
            for kind in ('model', 'initial_state'):
                source = Path(row[kind])
                expected = row[kind + '_sha256']
                if sha(source) != expected:
                    raise ValueError(f'{kind} hash mismatch')
                relative = f'files/{expected}{source.suffix}'
                dest = out / relative
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    dest.symlink_to(source)
                files[kind] = relative
                files[kind + '_sha256'] = expected
            np.savez_compressed(out / (name + '.npz'), **arrays)
            selected.append(dict(name=name, object_id=obj, candidate_id=row['candidate_id'],
                                 original_episode_sha256=row['episode_sha256'],
                                 arrays=name + '.npz', arrays_sha256=sha(out / (name + '.npz')),
                                 required_contact_groups=row['dexonomy_target']['required_contact_groups'],
                                 executed_target={k:row['executed_target'][k] for k in
                                                  ('world_root', 'finger_closure_target')},
                                 quality_limits=row['quality']['limits'], baseline_guard=guard,
                                 frames=len(arrays['action']), **files))
            break
        else:
            raise ValueError(f'no baseline guard-compliant demonstration for {obj}')
    dump(out / 'episodes.json', selected)
    dump(out / 'selection.json', dict(rule='one central episode per object, highest existing quality margin '
                                         'among guard-compliant records; ties by episode SHA',
                                     source_sha256=sha(contract), decisions=decisions))
    config = dict(seed=42, updates=6000, batch_size=64, learning_rate=1e-4,
                  weight_decay=1e-6, warmup=500, horizon=16, n_obs_steps=2, n_action_steps=8,
                  down_dims=[256, 512, 1024], encoder_output_dim=64,
                  diffusion_step_embed_dim=128, diffusion_steps=100, inference_steps=10,
                  prediction_type='sample', point_count=640, state_dim=125,
                  evaluation_seeds=[1000, 1001, 1002], device='cuda:0',
                  numerical_replay_tolerance=dict(state=1e-6, force=1e-5, geometry=1e-8),
                  task='goal-conditioned known-training-condition diagnostic; no held-out generalization',
                  observation='512 metric target mesh points at CURRENT geom pose + 128 table points; '
                              'joint position/velocity, previous final actuator command, current wrist '
                              'pose, fixed requested grasp root and closure targets',
                  action='27 final actuator commands; no additional gravity bias or teacher feedback',
                  upstream_commit=subprocess.check_output(['git', '-C', str(ROOT/'third_party/dp3'),
                                                           'rev-parse', 'HEAD'], text=True).strip())
    dump(out / 'plan.json', config)
    code = out / 'code'
    (code / 'scripts').mkdir(parents=True)
    shutil.copytree(ROOT/'scripts/stage1_dp3', code/'scripts/stage1_dp3',
                    ignore=shutil.ignore_patterns('__pycache__'))
    from icra.paths import snapshot_library
    snapshot_library(code/'src/icra')
    vendor = code/'vendor/diffusion_policy_3d'
    shutil.copytree(ROOT/'third_party/dp3/3D-Diffusion-Policy/diffusion_policy_3d', vendor,
                    ignore=shutil.ignore_patterns('__pycache__'))
    changes = []
    for name, line in [('policy/dp3.py', 'import pytorch3d.ops as torch3d_ops\n'),
                       ('model/vision/pointnet_extractor.py', 'import torchvision\n')]:
        file = vendor/name
        before = sha(file)
        content = file.read_text()
        token = 'torch3d_ops' if 'dp3.py' in name else 'torchvision'
        if content.count(token) != 1 or content.count(line) != 1:
            raise ValueError('upstream unused-import assumption changed')
        file.write_text(content.replace(line, ''))
        changes.append(dict(file=name, before=before, after=sha(file), removed_unused_import=line.strip()))
    license_file = ROOT/'third_party/dp3/LICENSE'
    if license_file.exists():
        shutil.copy2(license_file, code/'vendor/DP3_LICENSE')
    dump(out/'upstream-adaptation.json', changes)
    files = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out/'hashes.json', files)
    print(json.dumps(dict(episodes=len(selected), frames=sum(r['frames'] for r in selected),
                          objects=[r['object_id'] for r in selected], files=len(files))), flush=True)


if __name__ == '__main__':
    main()
