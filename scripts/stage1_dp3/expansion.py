"""Frozen placement holdout experiment, reusing hash-verified archived replay states."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from stage1_dp3.environment import Archive, Outcome, SIGNALS, transform_points
from stage1_dp3.experiment import check_files, evaluate, load
from stage1_dp3.package import ROOT, dump, sha

REMOTE_SOURCE = Path('/data1/yspark/projects/dexprompt/stage1-goal-bc-v1')
HELDOUT = {'x_plus20mm', 'y_plus20mm', 'yaw_plus15'}
OBJECTS = ('AlphabetSoup', 'Peaches', 'Raisins')


def split_rows(rows):
    train = [r for r in rows if r['placement'] not in HELDOUT]
    heldout = [r for r in rows if r['placement'] in HELDOUT]
    if not train or not heldout:
        raise ValueError('empty split')
    if {r['original_episode_sha256'] for r in train} & {r['original_episode_sha256'] for r in heldout}:
        raise ValueError('episode leakage')
    # Same shape may occur on both sides: the claim is placement, not shape/candidate generalization.
    if {tuple(r['pose_group']) for r in train} & {tuple(r['pose_group']) for r in heldout}:
        raise ValueError('same physical placement appears across splits')
    diagnostic = []
    for obj in OBJECTS:
        central = sorted([r for r in train if r['object_id'] == obj and r['placement'] == 'centre'],
                         key=lambda r: (r['candidate_id'], r['original_episode_sha256']))
        chosen = {}
        for r in central:
            chosen.setdefault(r['candidate_id'], r)
        if len(chosen) < 2:
            raise ValueError('need two central candidate controls per object')
        diagnostic.extend(list(chosen.values())[:2])
    return train, heldout, diagnostic


def build(out, replay_report):
    out.mkdir(parents=True, exist_ok=False)
    source = Path('/data1/yspark/sim/icra/stage1-goal-bc-v1/package')
    rows = [json.loads(x) for x in (source/'episodes.jsonl').read_text().splitlines()]
    contract = {r['episode_sha256']:r for r in map(json.loads,
        (ROOT/'records/experiments/2026-09-12/stage1-bc-data-contract/episodes.jsonl').read_text().splitlines())}
    reports = {r['name']:r for r in json.loads(replay_report.read_text())}
    kept, decisions = [], []
    for r in rows:
        rp = reports[r['name']]
        if not rp['right_state_identical'] or not rp['signals_identical']:
            raise ValueError('source replay did not match')
        if sha(source/r['arrays']) != r['arrays_sha256'] or rp['source_arrays_sha256'] != r['arrays_sha256']:
            raise ValueError('source arrays checksum mismatch')
        with np.load(source/r['arrays'], allow_pickle=False) as z:
            recorded = {k:z[k] for k in SIGNALS}
            height = r['initial_object_poses'][r['object_id']]['position_m'][2]
            judge = Outcome(r['quality_limits'], height)
            for i in range(r['frames']):
                judge.update({k:recorded[k][i] for k in SIGNALS})
            result = judge.summary()
        if result['success'] != rp['judge']['success']:
            raise ValueError('DP3 and augmented-data eligibility disagree')
        decisions.append(dict(name=r['name'], admitted=result['success'], outcome=result))
        if not result['success']:
            continue
        original = contract[r['episode_sha256']]
        pose = original['initial_object_poses'][r['object_id']]
        w,x,y,z = pose['quaternion_wxyz']
        yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        pose_group = [r['object_id'], *np.round(np.array(pose['position_m'][:2])*1000).astype(int).tolist(),
                      int(round(yaw))]
        kept.append(dict(name=r['name'], object_id=r['object_id'], candidate_id=r['candidate_id'],
            original_episode_sha256=r['episode_sha256'], placement=r['placement'], pose_group=pose_group,
            frames=r['frames'], required_contact_groups=r['required_contact_groups'],
            quality_limits=r['quality_limits'], baseline_guard=result,
            executed_target={k:r['executed_target'][k] for k in ('world_root','finger_closure_target')},
            **{k:str(REMOTE_SOURCE/'package'/r[k]) for k in ('model','initial_state','arrays')},
            **{k:r[k] for k in ('model_sha256','initial_state_sha256','arrays_sha256')},
            augment=str(REMOTE_SOURCE/'runs/replay-all'/rp['augment']), augment_sha256=rp['augment_sha256']))
    train, heldout, diagnostic = split_rows(kept)
    old = ROOT/'out/stage1-dp3-package-v1'
    config = json.loads((old/'plan.json').read_text())
    config.update(updates=20000, task='known shapes, archived but DP3-untrained placement holdout',
                  train_names=[r['name'] for r in train], heldout_names=[r['name'] for r in heldout],
                  diagnostic_names=[r['name'] for r in diagnostic],
                  heldout_placement_labels=sorted(HELDOUT),
                  heldout_gate='at least one success in the 18 prespecified training-condition rollouts',
                  evaluation_horizon='archived episode length; no early success termination',
                  normalization_scope='training rows only',
                  replay_source_report_sha256=sha(replay_report),
                  source_manifest_sha256=sha(source/'MANIFEST.sha256'),
                  old_feature_run='/data1/yspark/projects/dexprompt/stage1-dp3-v1/run',
                  stage_budgets={'prepare':1800,'controls':1800,'train':7200,
                                 'diagnostic':1800,'heldout':3600}, total_budget_seconds=4*3600)
    dump(out/'episodes.json', kept)
    dump(out/'plan.json', config)
    dump(out/'selection.json', dict(frozen_at=time.time(), decisions=decisions,
        train=len(train), heldout=len(heldout), diagnostic=len(diagnostic),
        train_objects=dict(Counter(r['object_id'] for r in train)),
        heldout_objects=dict(Counter(r['object_id'] for r in heldout)),
        limitation='All source episodes were selected teacher successes and historically evaluated. '
                   'The heldout split is not a fresh prospective placement or new-shape test. '
                   'Candidates and shapes overlap by design; no heldout actions/statistics enter training.'))
    shutil.copy2(replay_report, out/'source-replay-report.json')
    shutil.copytree(old/'code/vendor', out/'code/vendor')
    shutil.copytree(old/'code/src', out/'code/src')
    shutil.copytree(ROOT/'scripts/stage1_dp3', out/'code/scripts/stage1_dp3',
                    ignore=shutil.ignore_patterns('__pycache__'))
    dump(out/'hashes.json', {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(json.loads((out/'selection.json').read_text()) | {'decisions':'see file'}), flush=True)


def rotations(quats):
    q = np.asarray(quats, dtype=np.float64)
    if not np.allclose(np.linalg.norm(q,axis=-1),1,rtol=0,atol=1e-6):
        raise ValueError('invalid object quaternion')
    w,x,y,z = np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)


def augmented_features(sim, aug, actions):
    """The target is a rigid mesh: move fixed body-frame samples with each pre-action pose."""
    d = sim.data
    world = transform_points(sim.local_points, d.geom_xpos[sim.geom], d.geom_xmat[sim.geom])
    local = (world-d.xpos[sim.body]) @ d.xmat[sim.body].reshape(3,3)
    target = np.einsum('tij,nj->tni', rotations(aug['object_quat']), local)+aug['object_pos'][:,None,:]
    table = transform_points(sim.table_points,d.geom_xpos[sim.table],d.geom_xmat[sim.table])
    point_cloud = np.concatenate([target,np.broadcast_to(table,(len(actions),128,3))],axis=1)
    previous = np.concatenate([sim.previous[None],actions[:-1]])
    state = np.concatenate([aug['q'],aug['qd'],previous,aug['hand_pos'],aug['hand_mat'],
                            np.broadcast_to(sim.goal,(len(actions),32))],axis=1)
    return dict(point_cloud=point_cloud.astype(np.float32),agent_pos=state.astype(np.float32),
                action=actions.astype(np.float32))


def prepare(package, out):
    config, rows = load(package)
    check_files(package)
    (out/'features').mkdir(exist_ok=False)
    checked, reports = {}, []
    old_rows = json.loads((Path(config['old_feature_run']).parent/'package/episodes.json').read_text())
    old = {r['original_episode_sha256']:r for r in old_rows}
    for r in rows:
        if (out/'STOP').exists():
            raise RuntimeError('STOP requested')
        for k in ('model','initial_state','arrays','augment'):
            path = r[k]
            if path not in checked:
                checked[path] = sha(path)
            if checked[path] != r[k+'_sha256']:
                raise ValueError(f'input checksum mismatch: {path}')
        with np.load(r['augment'],allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        with np.load(r['arrays'],allow_pickle=False) as z:
            actions = z['action'].copy()
            if not np.array_equal(a['q'].astype(np.float32),z['right_state']):
                raise ValueError('augmentation time alignment mismatch')
            if not np.allclose(a['time'],z['timestamp'],rtol=0,atol=1e-10):
                raise ValueError('augmentation timestamp mismatch')
        sim = Archive(package,r)
        features = augmented_features(sim,a,actions)
        if any(not np.isfinite(v).all() for v in features.values()):
            raise ValueError('nonfinite features')
        # Compare causal feature assembly with simulator observation at archived reset.
        gaps = {k:float(np.max(np.abs(features[k][0]-v))) for k,v in sim.observation().items()}
        if max(gaps.values()) > 2e-6:
            raise ValueError(f'initial feature assembly mismatch: {gaps}')
        # Previous diagnostic saved observations measured during actual physics at EVERY tick.
        old_gaps = None
        if r['original_episode_sha256'] in old:
            name = old[r['original_episode_sha256']]['name']
            with np.load(Path(config['old_feature_run'])/'features'/f'{name}.npz') as z:
                old_gaps = {k:float(np.max(np.abs(v-z[k].astype(np.float32)))) for k,v in features.items()}
            if max(old_gaps.values()) > 2e-6:
                raise ValueError(f'all-tick direct observation mismatch: {old_gaps}')
        report = dict(name=r['name'], reset_gaps=gaps, prior_direct_replay_gaps=old_gaps)
        np.savez_compressed(out/'features'/f"{r['name']}.npz",**features)
        reports.append(report)
        dump(out/'feature-check.json',dict(completed=len(reports)==len(rows),episodes=reports,
                                          checked_input_hashes=checked))
        print(json.dumps(report),flush=True)
        del sim
    if sum(r['prior_direct_replay_gaps'] is not None for r in reports) != 3:
        raise ValueError('missing three independent all-tick reference comparisons')


def controls(package, out):
    config, rows = load(package)
    wanted = set(config['diagnostic_names']+config['heldout_names'])
    results = []
    for r in rows:
        if (out/'STOP').exists():
            raise RuntimeError('STOP requested')
        if r['name'] not in wanted:
            continue
        sim = Archive(package,r)
        judge = Outcome(r['quality_limits'],sim.height)
        with np.load(r['arrays'],allow_pickle=False) as z:
            actions = z['action'].copy()
            signals = {k:z[k] for k in SIGNALS}
            q = z['right_state'].copy()
        gaps = {k:0. for k in SIGNALS}
        qgap = 0.
        start = time.monotonic()
        for i,command in enumerate(actions):
            qgap=max(qgap,float(np.max(np.abs(sim.data.qpos[sim.addresses].astype(np.float32)-q[i]))))
            s=sim.step(command)
            judge.update(s)
            for k in SIGNALS:
                gaps[k]=max(gaps[k],float(np.max(np.abs(np.asarray(s[k],float)-signals[k][i]))))
        result=dict(name=r['name'],object=r['object_id'],split='heldout' if r['name'] in config['heldout_names'] else 'train',
                    seconds=time.monotonic()-start,state_max_abs=qgap,signals_max_abs=gaps,**judge.summary())
        results.append(result)
        dump(out/'controls.json',dict(completed=len(results)==len(wanted),episodes=results))
        print(json.dumps(result),flush=True)
        if qgap != 0 or any(gaps.values()) or not result['success']:
            raise ValueError('positive control mismatch; do not train')
        del sim


def summarize(package,out):
    config,rows=load(package)
    sections=[]
    for name in ('diagnostic','heldout'):
        file=out/f'{name}.json'
        if not file.exists():
            sections.append(f'- {name}: 미실행. 학습 조건 완주 0이면 확장하지 않는 사전 규칙.'); continue
        data=json.loads(file.read_text())
        sections.append(f"- {name}: {data['successes']}/{data['attempts']} 완주")
        for obj in OBJECTS:
            e=[r for r in data['episodes'] if r['object']==obj]
            sections.append(f"  - {obj}: {sum(r['success'] for r in e)}/{len(e)}")
    text='# DP3 다배치 학습·미학습 배치 평가\n\n'+'\n'.join(sections)+'\n\n'
    text+=('학습 자료와 정규화에서 평가 배치를 제외했다. 알려진 형상·후보의 과거 teacher 성공 배치만을 '
           '비교한 결과이며, 새 물체 일반화나 자동 목표 선정 결과가 아니다. 원본 명령 재생은 기준선이며 '
           '후보 생성·경로 탐색 비용을 포함하지 않아, 이 비교로 전체 시범 생성 효율의 우위를 주장하지 않는다.\n')
    (out/'FINAL.md').write_text(text)


def run(package,out):
    out.mkdir(parents=True,exist_ok=False)
    config,_=load(package)
    check_files(package)
    stages=[]
    deadline=time.time()+config['total_budget_seconds']
    dump(out/'frozen-input.json',dict(plan_sha256=sha(package/'plan.json'),manifest_sha256=sha(package/'hashes.json')))
    try:
        for name in ('prepare','controls','train','diagnostic','heldout'):
            if time.time() >= deadline:
                raise RuntimeError('total deadline exhausted before stage start')
            if (out/'STOP').exists():
                raise RuntimeError('STOP requested')
            if name=='heldout' and json.loads((out/'diagnostic.json').read_text())['successes']==0:
                stages.append(dict(stage=name,skipped=True,reason='zero training-condition completions'))
                dump(out/'stages.json',stages); break
            module='stage1_dp3.experiment' if name=='train' else 'stage1_dp3.expansion'
            cmd=[sys.executable,'-m',module,name,'--package',str(package),'--out',str(out)]
            entry=dict(stage=name,started_at=time.time(),timeout=False)
            stages.append(entry)
            dump(out/'status.json',dict(stage=name,running=True,deadline=deadline))
            with (out/f'{name}.log').open('x') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                entry['pid']=child.pid; dump(out/'stages.json',stages)
                try:
                    code=child.wait(timeout=max(1,min(config['stage_budgets'][name],deadline-time.time())))
                except subprocess.TimeoutExpired:
                    entry['timeout']=True
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait()
                    code=child.returncode
            entry.update(exit_code=code,finished_at=time.time()); dump(out/'stages.json',stages)
            if code!=0 or entry['timeout']:
                raise RuntimeError(f'{name}: exit {code}, timeout {entry["timeout"]}')
        summarize(package,out)
        dump(out/'status.json',dict(stage='completed',running=False,finished_at=time.time()))
    except Exception as e:
        dump(out/'status.json',dict(stage='failed',running=False,error=str(e),finished_at=time.time()))
        (out/'FINAL.md').write_text('# DP3 다배치 실험\n\n중단: '+str(e)+'\n인프라 오류는 파지 실패로 세지 않는다.\n')
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['build','prepare','controls','diagnostic','heldout','run'])
    p.add_argument('--package',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replay-report',type=Path)
    a=p.parse_args()
    if a.stage=='build':
        build(a.out,a.replay_report)
    elif a.stage in ('diagnostic','heldout'):
        config,_=load(a.package)
        evaluate(a.package,a.out,config[a.stage+'_names'],tag=a.stage)
    else:
        globals()[a.stage](a.package,a.out)


if __name__=='__main__':
    main()
