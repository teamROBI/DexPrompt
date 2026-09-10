"""The probe's bookkeeping, which twice said something untrue.

Both faults produced confident, wrong numbers rather than errors: a record keyed
without its experiment let one run's end state overwrite another's, so an
accepted grasp was measured against a different candidate's pose; and a reader
that fell back to a name glob picked whichever file matched first. Neither is
about Dexonomy -- they are about this diagnosis being able to say which pose it
measured.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_backend = __import__("os").environ.get("MUJOCO_GL")
gaps = importlib.import_module("report_missing_digit_gap")
residuals = importlib.import_module("report_contact_residuals")
if _backend is None:
    __import__("os").environ.pop("MUJOCO_GL", None)
else:
    __import__("os").environ["MUJOCO_GL"] = _backend

A = ("third_party/dexonomy/output/run_a_hx5_d20/init_data/T/obj/floating/"
     "scale008/0_96_grasp.npy")
B = ("third_party/dexonomy/output/run_b_hx5_d20/init_data/T/obj/floating/"
     "scale008/0_96_grasp.npy")


@pytest.mark.parametrize("module", [gaps, residuals])
def test_two_runs_of_one_candidate_get_different_keys(module):
    """Same object, scale and file name in two experiments must not collide."""
    assert module.key_for(A) != module.key_for(B)
    assert "run_a_hx5_d20" in module.key_for(A)


@pytest.mark.parametrize("module", [gaps, residuals])
def test_the_key_still_names_the_candidate(module):
    key = module.key_for(A)
    for part in ("obj", "floating", "scale008", "0_96_grasp"):
        assert part in key


@pytest.mark.parametrize("module", [gaps, residuals])
def test_a_missing_record_is_refused_not_guessed(module, tmp_path):
    """A near-miss must raise, not silently measure another candidate's pose."""
    np.savez(tmp_path / f"{module.key_for(B)}.npz", hand_qpos=np.zeros(3))
    with pytest.raises(FileNotFoundError):
        module.load_end_state(tmp_path, A)


@pytest.mark.parametrize("module", [gaps, residuals])
def test_the_right_record_is_found(module, tmp_path):
    np.savez(tmp_path / f"{module.key_for(A)}.npz", hand_qpos=np.arange(3))
    np.savez(tmp_path / f"{module.key_for(B)}.npz", hand_qpos=np.zeros(3))
    assert list(module.load_end_state(tmp_path, A)["hand_qpos"]) == [0, 1, 2]


def test_links_map_to_digits_four_at_a_time():
    assert [gaps.digit_of(f"finger_r_link{n}") for n in (1, 4, 5, 8, 17, 20)] \
        == [1, 1, 2, 2, 5, 5]
