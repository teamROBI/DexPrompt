"""Project paths shared by relocated command-line scripts.

Runtime-frozen root scripts retain their old imports. Adding the categorized
directories also keeps sibling module names usable during the staged migration.
This module imports no simulator, model, or training code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('assets', 'data', 'diagnostics', 'experiments', 'ops', 'rl', 'student', 'teacher')


def script_path(name):
    relative = name if name.startswith('scripts/') else 'scripts/' + name
    mapping = json.loads((ROOT / 'scripts/legacy_paths.json').read_text())
    path = ROOT / mapping.get(relative, relative)
    if (not path.resolve().is_relative_to(ROOT / 'scripts')
            or path.suffix != '.py' or not path.is_file()):
        raise ValueError(f'not a project Python script: {name}')
    return path


def setup():
    paths = [ROOT, ROOT / 'src', ROOT / 'scripts']
    paths.extend(ROOT / 'scripts' / group for group in GROUPS)
    for path in paths:
        value = str(path)
        if value not in sys.path:
            sys.path.append(value)
    return ROOT
