"""Transfer helper fixes the remote code and endpoint names."""

import ast
from pathlib import Path


def test_transfer_installs_only_diagnostic_helpers():
    path = Path(__file__).resolve().parents[2] / "scripts/ops/transfer_hope_bundle_lecun.py"
    tree = ast.parse(path.read_text())
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert "scripts/diagnostics/audit_visual_intake.py" in strings
    assert "scripts/ops/check_hope_bundle_lecun.py" in strings
    assert "scripts/ops/verify_tree_transfer.py" in strings
    assert "train" not in strings
