"""Bounded direct-regression comparison; run in a dedicated lecun snapshot/tmux."""
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', default='2')
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--baseline-run', type=Path)
    p.add_argument('--wait-for', type=Path, help='Run only if this job finishes with zero successes')
    p.add_argument('--state-residual', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = Path('/home/yspark/projects/DexPrompt')
    baseline = args.baseline_run or Path('/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-v1/run.json')
    if args.job.exists() or args.out.exists():
        raise FileExistsError('fresh job and training output required')
    args.job.mkdir(parents=True)
    os.nice(10)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, MUJOCO_GL='egl',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      PYTHONUNBUFFERED='1')
    if args.wait_for:
        write(args.job/'status.json', {'stage':'waiting_for_direct_result','running':True})
        while not (args.wait_for/'result.json').exists():
            status_path = args.wait_for/'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if (time.time() >= args.deadline or (args.job/'STOP').exists()
                    or status.get('stage') == 'failed'):
                write(args.job/'status.json', {'stage':'blocked','running':False,
                      'reason':'dependency failed, STOP or deadline','training_started':False})
                return
            time.sleep(15)
        prior = json.loads((args.wait_for/'result.json').read_text())
        if prior.get('completed') is not True or prior.get('compliant_successes') != 0:
            write(args.job/'status.json', {'stage':'skipped','running':False,
                  'reason':'direct result not a completed zero-success evaluation','training_started':False})
            return
    source = json.loads(baseline.read_text())
    config = json.loads(json.dumps(source['config']))
    config['prediction_type'] = 'direct'
    if args.state_residual:
        config['action_parameterization'] = 'state_residual'
    config_path = args.job / 'config.json'
    write(config_path, config)
    data, cache = base/'out/student-single-bottle-v1', base/'out/student-single-bottle-v1-cache'
    for path, key in [(data/'manifest.json', 'dataset_sha256'), (cache/'manifest.json', 'cache_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source[key]:
            raise ValueError(f'baseline data/cache mismatch: {path}')
    write(args.job/'plan.json', {
        'started_at':time.time(), 'deadline':args.deadline, 'baseline':str(baseline),
        'changed_config':{k:[source['config'].get(k),config.get(k)]
                          for k in set(source['config'])|set(config)
                          if source['config'].get(k)!=config.get(k)},
        'dataset_sha256':source['dataset_sha256'],
        'cache_sha256':source['cache_sha256'], 'epochs':13, 'optimizer_steps':6331,
        'final_checkpoint':'epoch_0012.pt', 'evaluation_seeds':[1000,1001,1002],
        'scope':'known training task; deterministic direct regression, not independent repeats',
        'root':str(root), 'gpu':args.gpu, 'training_authorization':'2026-09-12 seven-hour user task',
    })
    stages = []

    def stage(name, command, timeout):
        seconds = min(timeout, args.deadline-time.time())
        if seconds <= 0 or (args.job/'STOP').exists():
            raise RuntimeError('deadline or STOP reached')
        write(args.job/'status.json', {'stage':name,'running':True,'updated_at':time.time()})
        entry = {'stage':name, 'command':list(map(str,command)), 'started_at':time.time()}
        stages.append(entry)
        write(args.job/'stages.json',stages)
        with (args.job/f'{name}.log').open('x') as log:
            process = subprocess.Popen(list(map(str,command)), cwd=root, env=os.environ,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            entry['pid']=process.pid
            write(args.job/'stages.json',stages)
            try:
                code=process.wait(timeout=seconds)
                entry['timed_out']=False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                entry['timed_out']=True
                code=process.returncode
            entry.update(exit_code=code,finished_at=time.time())
            write(args.job/'stages.json',stages)
        if code or entry['timed_out']:
            raise RuntimeError(f'{name} failed: {code}, timeout={entry["timed_out"]}')

    trainer=[sys.executable,'scripts/student/train_hx5_student.py']
    common=['--config',config_path,'--data',data,'--cache',cache,'--allow-train-only-diagnostic']
    evaluator=[sys.executable,'scripts/student/evaluate_student_suite.py','--data',data]
    try:
        free=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'],text=True).strip())
        if free<22000:
            raise RuntimeError('GPU no longer has 22000 MiB free')
        stage('preflight',[*trainer,'preflight',*common,'--out',args.job/'preflight'],900)
        if not json.loads((args.job/'preflight/preflight.json').read_text())['training_ready']:
            raise RuntimeError('preflight not ready')
        stage('controls',[*evaluator,'--out',args.job/'controls','--check-controls'],600)
        if not json.loads((args.job/'controls/summary.json').read_text())['checks_passed']:
            raise RuntimeError('recorded action control failed')
        stage('train',[*trainer,'train',*common,'--out',args.out,'--device','cuda:0',
                       '--cuda-memory-fraction','0.90'],10800)
        stage('evaluate',[*evaluator,'--out',args.job/'evaluation','--checkpoint',args.out/'epoch_0012.pt',
                          '--device','cuda:0','--repeats','3','--seed','1000','--train-task-diagnostic'],1800)
        result=json.loads((args.job/'evaluation/summary.json').read_text())
        write(args.job/'result.json',{k:result.get(k) for k in
                                     ['completed','attempts','compliant_successes','sustained_lifts']})
        write(args.job/'status.json',{'stage':'completed','running':False,'updated_at':time.time()})
    except Exception as e:
        write(args.job/'status.json',{'stage':'failed','running':False,'error':repr(e),'updated_at':time.time()})
        raise


if __name__=='__main__':
    main()
