"""Package evaluation assets separately from Git; preserve original file bytes."""
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'out/student-transfer/manifest.json'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def sources(checkpoint):
    mappings = [
        (ROOT / 'out/student-bottle-can-v1', 'out/student-bottle-can-v1'),
        (ROOT / 'out/student-assets/dinov2', 'out/student-assets/dinov2'),
        (ROOT / 'out/student-assets/dinov2_vitb14_pretrain.pth',
         'out/student-assets/dinov2_vitb14_pretrain.pth'),
        (ROOT / 'out/student-assets/dinov2_vitl14_pretrain.pth',
         'out/student-assets/dinov2_vitl14_pretrain.pth'),
        (checkpoint, 'out/student-checkpoints/bottle-can-bc-v1/epoch_0002.pt'),
    ]
    for source, target in mappings:
        if not source.exists():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob('*')) if source.is_dir() else [source]:
            relative = path.relative_to(source) if source.is_dir() else Path()
            if any(p in ('.git', '__pycache__', '.pytest_cache') for p in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f'asset tree must be self-contained: {path}')
            if path.is_file():
                yield path, str(Path(target) / relative) if source.is_dir() else target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix + '.partial').exists():
        raise FileExistsError('use a fresh archive destination')
    entries = list(sources(args.checkpoint))
    manifest = {'schema_version': 1, 'purpose': 'bottle-can trained-task evaluation',
                'dataset': 'out/student-bottle-can-v1',
                'checkpoint': 'out/student-checkpoints/bottle-can-bc-v1/epoch_0002.pt',
                'expected_python': '3.10', 'expected_mujoco': '3.3.3', 'files': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.partial')
    with tarfile.open(temporary, 'w') as archive:
        for path, name in entries:
            before = path.stat()
            sha = digest(path)
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as stream:
                archive.addfile(info, stream)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'asset changed during packaging: {name}')
            manifest['files'].append({'path': name, 'size': before.st_size, 'sha256': sha})
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    temporary.replace(args.out)
    sha = digest(args.out)
    args.out.with_suffix(args.out.suffix + '.sha256').write_text(f'{sha}  {args.out.name}\n')
    print(json.dumps({'archive': str(args.out), 'bytes': args.out.stat().st_size,
                      'sha256': sha, 'files': len(entries)}, indent=2))


if __name__ == '__main__':
    main()
