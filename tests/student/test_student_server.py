import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import package_student_transfer as pack  # noqa: E402
import student_server as server  # noqa: E402


def fixture_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'MANIFEST', tmp_path / 'transfer.json')
    entries = []
    for name in ['dataset/manifest.json', 'checkpoint.pt']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
        entries.append({'path': name, 'size': 8, 'sha256': pack.digest(path)})
    manifest = {'schema_version': 1, 'dataset': 'dataset',
                'checkpoint': 'checkpoint.pt', 'files': entries}
    server.MANIFEST.write_text(json.dumps(manifest))
    return manifest


def test_assets_relocate_and_detect_same_size_corruption(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch)
    assert server.assets() == manifest
    (tmp_path / 'checkpoint.pt').write_bytes(b'modified')
    with pytest.raises(ValueError, match='hash mismatch'):
        server.assets()


@pytest.mark.parametrize('change', ['duplicate', 'outside', 'unverified_checkpoint'])
def test_manifest_rejects_invalid_paths(tmp_path, monkeypatch, change):
    manifest = fixture_manifest(tmp_path, monkeypatch)
    if change == 'duplicate':
        manifest['files'].append(manifest['files'][0])
    elif change == 'outside':
        manifest['files'][0]['path'] = '../outside'
    else:
        manifest['checkpoint'] = '/unverified/checkpoint.pt'
    server.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        server.assets()


def test_packer_refuses_dangling_nested_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(pack, 'ROOT', tmp_path)
    data = tmp_path / 'out/student-bottle-can-v1'
    data.mkdir(parents=True)
    (data / 'external').symlink_to('/missing/source')
    with pytest.raises(ValueError, match='self-contained'):
        list(pack.sources(tmp_path / 'checkpoint.pt'))


def test_busy_gpu_does_not_verify_or_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(server.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(server.subprocess, 'check_output',
                        lambda *a, **k: '0, GPU-test, 24000, 99\n')
    monkeypatch.setattr(server, 'verify', lambda: pytest.fail('busy GPU must stop early'))
    args = SimpleNamespace(session='test', out=tmp_path / 'job', gpu=0, egl_device=None)
    with pytest.raises(ValueError, match='GPU busy'):
        server.start(args)
    assert not args.out.exists()


def test_failed_gpu_preflight_never_runs_policy(tmp_path, monkeypatch):
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'environment': {}, 'command': ['must-not-run']}))
    calls = []

    def fail_probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(server.subprocess, 'run', fail_probe)
    assert server.run(job) == 1
    assert len(calls) == 1 and calls[0][-1] == '_gpu-check'
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'error'


def test_start_records_single_gpu_fixed_evaluation_and_quoted_job(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'verify', lambda: {'dataset': 'data', 'checkpoint': 'final.pt'})
    monkeypatch.setattr(server.subprocess, 'check_output',
                        lambda *a, **k: '2, GPU-selected, 23000, 0\n')
    calls = []

    def record(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[1] == 'has-session' else 0)

    monkeypatch.setattr(server.subprocess, 'run', record)
    args = SimpleNamespace(session='new-job', out=tmp_path / 'job with spaces', gpu=2,
                           egl_device=1, smoke=False, repeats=3)
    server.start(args)
    job = json.loads((args.out / 'job.json').read_text())
    assert job['environment']['CUDA_VISIBLE_DEVICES'] == 'GPU-selected'
    assert job['environment']['MUJOCO_EGL_DEVICE_ID'] == '1'
    assert job['command'][job['command'].index('--device') + 1] == 'cuda:0'
    assert job['command'][job['command'].index('--seed') + 1] == '1000'
    assert '--train-task-diagnostic' in job['command']
    import shlex
    assert shlex.split(calls[-1][-1])[-1] == str(args.out / 'job.json')
