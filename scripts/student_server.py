"""Verify transferred student assets and launch one-GPU evaluation in tmux."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from package_student_transfer import digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'out/student-transfer/manifest.json'
sys.path.insert(0, str(ROOT / 'src'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def assets():
    if not MANIFEST.is_file():
        raise FileNotFoundError('unpack the artifact archive into the clone first')
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported artifact manifest')
    seen = set()
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if (not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file()
                or path.is_symlink() or entry['path'] in seen):
            raise ValueError(f'missing, duplicate, or nonlocal artifact: {entry["path"]}')
        seen.add(entry['path'])
        if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError(f'artifact hash mismatch: {entry["path"]}')
    if manifest.get('checkpoint') not in seen:
        raise ValueError('checkpoint must be a verified artifact')
    if str(Path(manifest.get('dataset', '')) / 'manifest.json') not in seen:
        raise ValueError('dataset manifest must be a verified artifact')
    return manifest


def verify(load_model=False):
    import mujoco
    import torch

    from icra.dataset import validate_dataset

    if sys.version_info[:2] != (3, 10):
        raise ValueError('the tested runtime requires Python 3.10')
    if mujoco.__version__ != '3.3.3':
        raise ValueError('archived MJB requires mujoco==3.3.3')
    if torch.__version__.split('+')[0] != '2.6.0':
        raise ValueError('the tested checkpoint runtime requires torch==2.6.0')
    manifest = assets()
    validate_dataset(ROOT / manifest['dataset'])
    checkpoint = ROOT / manifest['checkpoint']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('optimizer_steps') != 6255 or payload.get('epoch') != 2:
        raise ValueError('expected the preselected final 3-epoch checkpoint')
    run = payload['run']
    changed = [name for name, sha in run['source_sha256'].items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != sha]
    if changed:
        raise ValueError(f'code differs from the trained checkpoint: {changed}')
    result = {'passed': True, 'artifact_files': len(manifest['files']),
              'dataset': manifest['dataset'], 'checkpoint': manifest['checkpoint'],
              'optimizer_steps': payload['optimizer_steps'], 'model_loaded': False,
              'torch': str(torch.__version__), 'mujoco': mujoco.__version__,
              'training_started': False, 'cuda_available': torch.cuda.is_available()}
    del payload
    if load_model:
        from icra.student_inference import load_checkpoint
        model, _ = load_checkpoint(checkpoint, device='cpu')
        result['model_loaded'] = True
        del model
    print(json.dumps(result, indent=2), flush=True)
    return manifest


def start(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.session):
        raise ValueError('use letters, digits, underscores or hyphens for the tmux session')
    if args.out.exists():
        raise FileExistsError('use a new --out directory')
    egl_device = args.gpu if args.egl_device is None else args.egl_device
    if egl_device < 0:
        raise ValueError('EGL device index must be nonnegative')
    if subprocess.run(['tmux', 'has-session', '-t', args.session],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise ValueError('tmux session already exists; inspect it rather than starting twice')
    gpu_text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    devices = {int(v[0]): v for v in
               ([s.strip() for s in line.split(',')] for line in gpu_text.splitlines())}
    if args.gpu not in devices:
        raise ValueError('requested physical GPU index not found')
    _, uuid, free, util = devices[args.gpu]
    if int(free) < 12000 or int(util) > 20:
        raise ValueError('GPU busy or less than 12GB free; choose an available GPU and retry later')
    manifest = verify()
    args.out.mkdir(parents=True)
    command = [sys.executable, str(ROOT / 'scripts/evaluate_student_suite.py'),
               '--data', str(ROOT / manifest['dataset']), '--checkpoint',
               str(ROOT / manifest['checkpoint']), '--out', str(args.out.resolve() / 'results'),
               '--device', 'cuda:0', '--train-task-diagnostic', '--seed', '1000',
               '--repeats', str(args.repeats)]
    if args.smoke:
        data = json.loads((ROOT / manifest['dataset'] / 'manifest.json').read_text())
        command += ['--episode', data['episodes'][0]['episode_id']]
    spec = {'command': command, 'gpu_index': args.gpu, 'gpu_uuid': uuid,
            'egl_device_index': egl_device,
            'session': args.session, 'created_at': datetime.now(timezone.utc).isoformat(),
            'smoke': args.smoke, 'repeats': args.repeats,
            'environment': {'CUDA_VISIBLE_DEVICES': uuid, 'MUJOCO_GL': 'egl',
                            'MUJOCO_EGL_DEVICE_ID': str(egl_device),
                            'PYOPENGL_PLATFORM': 'egl',
                            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                            'MALLOC_ARENA_MAX': '2', 'PYTHONUNBUFFERED': '1'}}
    job = args.out.resolve() / 'job.json'
    write(job, spec)
    invocation = shlex.join([sys.executable, str(Path(__file__).resolve()), '_run', str(job)])
    subprocess.run(['tmux', 'new-session', '-d', '-s', args.session, invocation], check=True)
    print(f'tmux attach -t {args.session}\nLogs: {args.out.resolve() / "evaluation.log"}')


def run(job):
    spec = json.loads(job.read_text())
    env = dict(os.environ, **spec['environment'])
    status = {'state': 'running', 'training_started': False,
              'started_at': datetime.now(timezone.utc).isoformat()}
    write(job.parent / 'status.json', status)
    with (job.parent / 'evaluation.log').open('w') as log:
        try:
            # Setup failures must stop the job before any policy rollout is counted.
            probe = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '_gpu-check'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if probe.returncode:
                raise RuntimeError('CUDA/EGL preflight failed; see log above')
            result = subprocess.run(['nice', '-n', '10', 'ionice', '-c', '3',
                                     *spec['command']], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
            code = result.returncode
        except KeyboardInterrupt:
            code = 130
        except Exception as error:
            log.write(f'{type(error).__name__}: {error}\n')
            code = 1
    status.update(state='completed' if code == 0 else 'interrupted' if code == 130 else 'error',
                  exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    write(job.parent / 'status.json', status)
    return code


def gpu_check():
    import mujoco
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('expected exactly one visible CUDA GPU')
    torch.zeros(1, device='cuda:0').add_(1)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
    if pixels.shape != (64, 64, 3):
        raise RuntimeError('unexpected EGL render shape')
    print(json.dumps({'gpu_preflight': 'passed', 'gpu': torch.cuda.get_device_name(0),
                      'egl_device_index': os.environ.get('MUJOCO_EGL_DEVICE_ID')}), flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('verify')
    check.add_argument('--load-model', action='store_true')
    launch = commands.add_parser('start')
    launch.add_argument('--gpu', type=int, default=0)
    launch.add_argument('--egl-device', type=int,
                        help='EGL enumeration index, defaults to the physical GPU index')
    launch.add_argument('--out', type=Path, required=True)
    launch.add_argument('--session', default='dexprompt-eval')
    launch.add_argument('--smoke', action='store_true')
    launch.add_argument('--repeats', type=int, default=3)
    worker = commands.add_parser('_run')
    worker.add_argument('job', type=Path)
    commands.add_parser('_gpu-check')
    args = parser.parse_args()
    if args.command == 'verify':
        verify(args.load_model)
    elif args.command == 'start':
        if args.repeats < 1:
            parser.error('repeats must be positive')
        start(args)
    elif args.command == '_gpu-check':
        gpu_check()
    else:
        return run(args.job)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
