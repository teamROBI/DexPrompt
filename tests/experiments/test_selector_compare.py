"""Selector comparison plan: fixed draw, fixed split sizes, near-duplicates kept together."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_selector_compare as compare  # noqa: E402


def rows(n, base=0.0):
    return [{"source_sha256": f"{i:04d}", "grasp_qpos": [base + i] * 27,
             "squeeze_qpos": [base + i] * 27} for i in range(n)]


def test_split_sizes_and_disjoint():
    r = rows(32)
    label, evaluation = compare.split(r, compare.near_duplicate_groups(r))
    assert len(label) == 16 and len(evaluation) == 16
    assert not {x["source_sha256"] for x in label} & {x["source_sha256"] for x in evaluation}


def test_near_duplicates_stay_on_one_side():
    r = rows(32)
    # candidate 20 practically equals candidate 3 (a label candidate)
    r[20]["grasp_qpos"] = list(np.asarray(r[3]["grasp_qpos"]) + 1e-5)
    r[20]["squeeze_qpos"] = list(np.asarray(r[3]["squeeze_qpos"]) + 1e-5)
    groups = compare.near_duplicate_groups(r)
    assert groups[20] == groups[3]
    label, evaluation = compare.split(r, groups)
    sides = {x["source_sha256"]: s for s, part in (("l", label), ("e", evaluation)) for x in part}
    assert sides["0020"] == sides["0003"] == "l"


def test_the_draw_is_fixed_by_the_seed():
    a = np.random.default_rng(compare.SEED).permutation(179)[:32]
    b = np.random.default_rng(compare.SEED).permutation(179)[:32]
    assert (a == b).all()


def test_budget_adds_up():
    assert sum(compare.CAPS.values()) == compare.TOTAL == 102
