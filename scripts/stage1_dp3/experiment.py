"""Run frozen DP3 replay, learning and closed-loop evaluation in separate processes."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from stage1_dp3.environment import Archive, Outcome, SIGNALS
from stage1_dp3.package import dump, sha


def sequence_indices(tick, length, horizon=16, history=2):
    start = tick - history + 1
    obs = np.clip(np.arange(start, tick + 1), 0, length - 1)
    actions = np.clip(np.arange(start, start + horizon), 0, length - 1)
    return obs, actions


def load(package):
    return json.loads((package/'plan.json').read_text()), json.loads((package/'episodes.json').read_text())


def check_files(package):
    for name, expected in json.loads((package/'hashes.json').read_text()).items():
        if sha(package/name) != expected:
            raise ValueError(f'package checksum mismatch: {name}')


def replay(package, out):
    config, rows = load(package)
    reports = []
    passed = True
    (out/'features').mkdir(exist_ok=True)
    for row in rows:
        sim = Archive(package, row)
        judge = Outcome(row['quality_limits'], sim.height)
        features, points, observations = [], [], []
        gaps = {k:0. for k in SIGNALS}
        state_gap = 0.
        with np.load(package/row['arrays'], allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        started = time.monotonic()
        for i, command in enumerate(arrays['action']):
            obs = sim.observation()
            features.append(obs['agent_pos'])
            points.append(obs['point_cloud'])
            observations.append(sim.data.xpos[sim.body].copy())
            state_gap = max(state_gap, float(np.max(np.abs(
                sim.data.qpos[sim.addresses].astype(np.float32) - arrays['right_state'][i]))))
            signals = sim.step(command)
            judge.update(signals)
            for key in SIGNALS:
                gaps[key] = max(gaps[key], float(np.max(np.abs(
                    np.asarray(signals[key], float) - np.asarray(arrays[key][i], float)))))
        tolerance = config['numerical_replay_tolerance']
        numerical = state_gap <= tolerance['state'] and all(
            gap <= (0 if key == 'contact_groups' else tolerance['force'] if 'force' in key
                    else tolerance['geometry']) for key, gap in gaps.items())
        guarded = judge.summary()
        ok = numerical and guarded['success'] == row['baseline_guard']['success'] and guarded['success']
        passed &= ok
        report = dict(name=row['name'], passed=ok, replay_numerically_matches=numerical,
                      all_bit_identical=state_gap == 0 and all(v == 0 for v in gaps.values()),
                      state_max_abs=state_gap, signals_max_abs=gaps, phase_free_outcome=guarded,
                      original_teacher_accepted=True, seconds=time.monotonic()-started)
        reports.append(report)
        # Save observations before their action, with full precision actuator labels.
        np.savez_compressed(out/'features'/f"{row['name']}.npz",
                            agent_pos=np.stack(features), point_cloud=np.stack(points),
                            action=arrays['action'], object_position=np.stack(observations))
        dump(out/'replay.json', dict(passed=passed, completed=len(reports)==len(rows), episodes=reports))
        print(json.dumps(report), flush=True)
        del sim
    if not passed:
        raise RuntimeError('replay or positive-control gate failed; learning must not proceed')


def policy(config):
    import torch
    from diffusers import DDIMScheduler
    from omegaconf import OmegaConf
    from diffusion_policy_3d.policy.dp3 import DP3
    return DP3(
        shape_meta=OmegaConf.create({'action':{'shape':[27]}, 'obs':{'point_cloud':{'shape':[640,3]},
                                                 'agent_pos':{'shape':[125]}}}),
        noise_scheduler=DDIMScheduler(num_train_timesteps=config['diffusion_steps'],
                                      beta_start=.0001, beta_end=.02, beta_schedule='squaredcos_cap_v2',
                                      clip_sample=True, set_alpha_to_one=True, steps_offset=0,
                                      prediction_type=config['prediction_type']),
        horizon=config['horizon'], n_action_steps=config['n_action_steps'],
        n_obs_steps=config['n_obs_steps'], num_inference_steps=config['inference_steps'],
        diffusion_step_embed_dim=config['diffusion_step_embed_dim'],
        down_dims=config['down_dims'], kernel_size=5, n_groups=8, condition_type='film',
        encoder_output_dim=config['encoder_output_dim'], use_pc_color=False,
        pointcloud_encoder_cfg=OmegaConf.create(dict(in_channels=3,
            out_channels=config['encoder_output_dim'], use_layernorm=True, final_norm='layernorm')),
    ).to(config['device'])


def batch(data, rng, config, size):
    # Episode-balanced sampling prevents the longest demonstration dominating this diagnostic.
    observations = {'point_cloud':[], 'agent_pos':[]}
    actions = []
    for _ in range(size):
        ep = data[int(rng.integers(len(data)))]
        tick = int(rng.integers(len(ep['action'])))
        oi, ai = sequence_indices(tick, len(ep['action']), config['horizon'], config['n_obs_steps'])
        for key in observations:
            observations[key].append(ep[key][oi])
        actions.append(ep['action'][ai])
    import torch
    return dict(obs={k:torch.as_tensor(np.stack(v), device=config['device'], dtype=torch.float32)
                     for k,v in observations.items()},
                action=torch.as_tensor(np.stack(actions), device=config['device'], dtype=torch.float32))


def finite(value):
    import torch
    if not torch.isfinite(value).all():
        raise FloatingPointError('nonfinite tensor')


def check(package, out):
    """Official model forward/loss/sampling check; deliberately no optimizer or backward."""
    import importlib.metadata
    import torch
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
    torch.set_num_threads(1)
    config, _ = load(package)
    torch.manual_seed(config['seed'])
    check_files(package)
    normalizer = LinearNormalizer()
    normalizer.fit({k:np.stack([-np.ones(n), np.ones(n)]).astype(np.float32)
                    for k,n in [('point_cloud',3), ('agent_pos',125), ('action',27)]})
    model = policy(config)
    model.set_normalizer(normalizer)
    model.to(config['device']).eval()
    probe = dict(obs={'point_cloud':torch.zeros((2,2,640,3), device=config['device']),
                     'agent_pos':torch.zeros((2,2,125), device=config['device'])},
                 action=torch.zeros((2,16,27), device=config['device']))
    with torch.no_grad():
        loss, _ = model.compute_loss(probe)
        predicted = model.predict_action(probe['obs'])['action']
    finite(loss)
    finite(predicted)
    if tuple(predicted.shape) != (2,8,27):
        raise ValueError('official model output shape mismatch')
    dump(out, dict(passed=True, loss=float(loss), output_shape=list(predicted.shape),
                   backward_calls=0, optimizer_steps=0,
                   versions={k:importlib.metadata.version(k) for k in
                             ('torch','numpy','mujoco','diffusers','zarr','einops','omegaconf')},
                   parameters=sum(p.numel() for p in model.parameters())))


def save_policy(path, model, config, step):
    import torch
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), config=config, updates=step), tmp)
    tmp.replace(path)


def training_rows(config, rows):
    if 'train_names' not in config:
        return rows
    names = set(config['train_names'])
    if not names or names.intersection(config.get('heldout_names', [])):
        raise ValueError('empty or leaking training split')
    selected = [row for row in rows if row['name'] in names]
    if len(selected) != len(names):
        raise ValueError('training manifest incomplete')
    return selected


def train(package, out):
    import torch
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
    from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
    config, rows = load(package)
    rows = training_rows(config, rows)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    data = []
    for row in rows:
        with np.load(out/'features'/f"{row['name']}.npz", allow_pickle=False) as z:
            data.append({k:z[k].astype(np.float32) for k in ('point_cloud', 'agent_pos', 'action')})
    normalizer = LinearNormalizer()
    # Only the selected training rows contribute examples AND normalization statistics.
    normalizer.fit({k:np.concatenate([ep[k] for ep in data]) for k in data[0]}, mode='limits')
    model = policy(config)
    model.set_normalizer(normalizer)
    model.to(config['device'])
    ema_model = copy.deepcopy(model)
    ema = EMAModel(ema_model, power=.75)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['learning_rate'], weight_decay=config['weight_decay'])
    first = batch(data, rng, config, 2)
    model.eval()
    with torch.no_grad():
        predicted = model.predict_action(first['obs'])['action']
        finite(predicted)
        normalized = model.normalizer['action'].normalize(first['action'])
        restored = model.normalizer['action'].unnormalize(normalized)
        if not torch.allclose(restored, first['action'], atol=2e-6, rtol=1e-6):
            raise ValueError('action normalization roundtrip failed')
    dump(out/'preflight.json', dict(passed=True, action_shape=list(predicted.shape),
                                   parameters=sum(p.numel() for p in model.parameters()),
                                   training_started=False))
    model.train()
    start = time.monotonic()
    with (out/'progress.tsv').open('x') as log:
        log.write('update\tloss\tgrad_norm\tlearning_rate\tupdates_per_second\n')
        for step in range(1, config['updates']+1):
            if (out/'STOP').exists():
                raise RuntimeError('STOP requested')
            factor = min(step/config['warmup'], 1.)
            if step > config['warmup']:
                factor *= .5*(1+math.cos(math.pi*(step-config['warmup'])/(config['updates']-config['warmup'])))
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate']*factor
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.compute_loss(batch(data, rng, config, config['batch_size']))
            finite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            ema.step(model)
            if step == 1:
                dump(out/'training.json', dict(training_started=True, started_at=time.time(),
                                               config=config))
            if step == 1 or step % 50 == 0:
                log.write(f'{step}\t{loss.item():.9g}\t{float(norm):.9g}\t{group["lr"]:.9g}\t'
                          f'{step/(time.monotonic()-start):.3f}\n')
                log.flush()
                dump(out/'train-status.json', dict(update=step, loss=loss.item(),
                                                   elapsed_seconds=time.monotonic()-start))
                print(f'update={step} loss={loss.item():.6f}', flush=True)
            if step % 2000 == 0:
                save_policy(out/'latest.pt', ema_model, config, step)
        save_policy(out/'final.pt', ema_model, config, config['updates'])
    dump(out/'training-result.json', dict(completed=True, updates=config['updates'],
                                          final_loss=float(loss), seconds=time.monotonic()-start,
                                          checkpoint='final.pt', checkpoint_sha256=sha(out/'final.pt')))


def evaluate(package, out, row_names=None, tag='evaluation'):
    import torch
    torch.set_num_threads(1)
    config, rows = load(package)
    if row_names is not None:
        names = set(row_names)
        rows = [row for row in rows if row['name'] in names]
        if len(rows) != len(names):
            raise ValueError('evaluation manifest incomplete')
    saved = torch.load(out/'final.pt', map_location=config['device'], weights_only=False)
    if saved['updates'] != config['updates'] or saved['config'] != config:
        raise ValueError('not the prespecified final checkpoint')
    model = policy(config)
    model.load_state_dict(saved['model'])
    model.to(config['device']).eval()
    results = []
    for row in rows:
        for seed in config['evaluation_seeds']:
            if (out/'STOP').exists():
                raise RuntimeError('STOP requested')
            torch.manual_seed(seed)
            np.random.seed(seed)
            sim = Archive(package, row)
            judge = Outcome(row['quality_limits'], sim.height)
            history = deque([sim.observation() for _ in range(config['n_obs_steps'])],
                            maxlen=config['n_obs_steps'])
            queue = deque()
            actions, measured, timestamps, signals_saved = [], [], [], []
            clipped = 0
            started = time.monotonic()
            for tick in range(row['frames']):
                current = sim.observation()
                if tick:
                    history.append(current)
                if not queue:
                    obs = {k:torch.as_tensor(np.stack([x[k] for x in history])[None],
                                             device=config['device']) for k in current}
                    with torch.no_grad():
                        prediction = model.predict_action(obs)['action'][0]
                        finite(prediction)
                    queue.extend(prediction.cpu().numpy().astype(np.float64))
                command = queue.popleft()
                bounded = np.clip(command, sim.low, sim.high)
                clipped += int(np.any(bounded != command))
                measured.append(sim.data.qpos[sim.addresses].copy())
                timestamps.append(float(sim.data.time))
                actions.append(bounded)
                signals = sim.step(bounded)
                signals_saved.append(signals)
                judge.update(signals)
                if judge.failure:
                    break
            result = dict(name=row['name'], object=row['object_id'], seed=seed,
                          saturated_ticks=clipped, seconds=time.monotonic()-started, **judge.summary())
            results.append(result)
            trajectory = out/f"{tag}-rollout-{row['name']}-{seed}.npz"
            np.savez_compressed(trajectory,
                                right_state=np.stack(measured), action=np.stack(actions),
                                timestamp=np.array(timestamps),
                                **{k:np.stack([s[k] for s in signals_saved]) for k in SIGNALS})
            result.update(trajectory=trajectory.name, trajectory_sha256=sha(trajectory),
                          model_sha256=row['model_sha256'], initial_state_sha256=row['initial_state_sha256'],
                          candidate_id=row['candidate_id'], student_ready=False)
            dump(out/f'{tag}.json', dict(completed=len(results)==len(rows)*len(config['evaluation_seeds']),
                                             successes=sum(r['success'] for r in results),
                                             attempts=len(results), episodes=results))
            print(json.dumps(result), flush=True)
            del sim


def run(package, out):
    out.mkdir(parents=True, exist_ok=False)
    check_files(package)
    dump(out/'frozen-input.json', dict(plan_sha256=sha(package/'plan.json'),
                                      manifest_sha256=sha(package/'hashes.json')))
    stages = []
    start = time.time()
    deadline = start + 4*3600
    dump(out/'status.json', dict(stage='starting', running=True, deadline=deadline,
                                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        for name, budget in [('replay', 2400), ('train', 9000), ('evaluate', 2400)]:
            if (out/'STOP').exists():
                raise RuntimeError('STOP requested')
            cmd = [sys.executable, '-m', 'stage1_dp3.experiment', name,
                   '--package', str(package), '--out', str(out)]
            entry = dict(stage=name, command=cmd, started_at=time.time(), timeout=False)
            stages.append(entry)
            dump(out/'status.json', dict(stage=name, running=True, deadline=deadline))
            with (out/f'{name}.log').open('x') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = child.pid
                dump(out/'stages.json', stages)
                try:
                    code = child.wait(timeout=max(1,min(budget,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    entry['timeout'] = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = child.returncode
                entry.update(exit_code=code, finished_at=time.time())
                dump(out/'stages.json', stages)
            if code != 0 or entry['timeout']:
                raise RuntimeError(f'{name} failed, exit={code}, timeout={entry["timeout"]}')
        result = json.loads((out/'evaluation.json').read_text())
        dump(out/'status.json', dict(stage='completed', running=False, finished_at=time.time(),
                                     successes=result['successes'], attempts=result['attempts']))
        (out/'FINAL.md').write_text(
            '# 목표 조건부 DP3 최소 실험\n\n'
            f"학습 조건 3개, 평가 {result['attempts']}회 중 완주 {result['successes']}회.\n\n"
            '정확한 현재 점군·관절 상태·Dexonomy 목표에서 팔·손 행동을 생성했다. '
            '새 물체 일반화, 목표 자동 선정, VLA 성능 개선의 결과가 아니다. '
            '공식 DP3 구조를 HX5 관측·행동과 작은 네트워크 폭에 맞춘 비교다.\n')
    except Exception as error:
        dump(out/'status.json', dict(stage='failed', running=False, error=str(error), finished_at=time.time()))
        (out/'FINAL.md').write_text('# 목표 조건부 DP3 최소 실험\n\n중단: '+str(error)+'\n'
                                  '인프라·재생·학습 오류를 파지 실패로 세지 않는다.\n')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['run', 'replay', 'train', 'evaluate', 'check'])
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.package.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
