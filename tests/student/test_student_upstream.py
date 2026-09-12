"""Guard the optional upstream boundary without importing extra packages."""
import hashlib

import pytest

torch = pytest.importorskip("torch")
from icra.student_upstream import official_type, optimizer_groups


def test_modified_upstream_source_is_rejected_before_import(tmp_path):
    names = ["controller/model/diffusion/transformer_for_action_diffusion.py",
             "controller/model/common/module_attr_mixin.py"]
    hashes = {}
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# sealed source\n")
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / names[1]).write_text("raise RuntimeError('must not import changed source')\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        official_type({"upstream_source": str(tmp_path), "upstream_sha256": hashes})


def test_missing_upstream_dependency_hash_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="both imported files"):
        official_type({"upstream_source": str(tmp_path), "upstream_sha256": {}})


def test_optimizer_groups_cover_trainable_parameters_once_and_preserve_frozen():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    model[0].bias.requires_grad_(False)
    groups = optimizer_groups(model, 0.01)
    assigned = [p for group in groups for p in group['params']]
    assert len(assigned) == len({id(p) for p in assigned}) == 3
    assert {id(p) for p in assigned} == {id(p) for p in model.parameters() if p.requires_grad}
    assert groups[0]['weight_decay'] == 0.01 and groups[1]['weight_decay'] == 0
    assert all(p.ndim >= 2 for p in groups[0]['params'])
    assert all(p.ndim < 2 for p in groups[1]['params'])
