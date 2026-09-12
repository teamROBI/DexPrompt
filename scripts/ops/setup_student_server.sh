#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
student_python="${STUDENT_PYTHON:-python3.10}"
"$student_python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for the tested runtime"'
if [[ ! -d .venv-student ]]; then
  "$student_python" -m venv --without-pip .venv-student
fi
.venv-student/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing .venv-student must also use Python 3.10"'
if ! .venv-student/bin/python -m pip --version >/dev/null 2>&1; then
  # Ubuntu may provide venv without ensurepip; bootstrap only this local environment.
  .venv-student/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlopen

destination = Path('.venv-student/get-pip.py')
with urlopen('https://bootstrap.pypa.io/get-pip.py', timeout=60) as response:
    destination.write_bytes(response.read())
PY
  .venv-student/bin/python .venv-student/get-pip.py
fi
.venv-student/bin/python -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
.venv-student/bin/python -m pip install -r requirements/student-server.txt
.venv-student/bin/python -m pip install --no-deps -e .
.venv-student/bin/python -m pip check
printf '%s\n' 'Installed. Next: unpack the artifact archive and run scripts/ops/student_server.py verify.'
