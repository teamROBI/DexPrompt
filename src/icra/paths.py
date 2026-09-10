"""Every path this project uses, resolved from the repository root.

Scripts here reach into three places outside their own tree -- a GraspGenX
checkout, a Dexonomy checkout, and a few gigabytes of checkpoints and generated
grasps that have no business on the repo's disk. Those were spelled out as
absolute paths under one particular home directory, which makes the scripts
unrunnable anywhere else and the results awkward to reproduce.

They are reached from the repository root instead:

    modules/graspgenx     -> ../../GraspGenX    (symlink: sibling checkout)
    third_party/dexonomy  -> ../../Dexonomy
    third_party/dexlearn, graspvla, dexgraspvla (cloned, read-only references)
    data                  -> wherever the bulk storage is  (machine-specific)

The split between the two directories is whether we change the code.

``third_party`` is taken as given. Dexonomy is there because we only ever *add*
to it -- a hand's assets under ``assets/hand/`` and one config file -- and never
touch its source.

``modules`` is for checkouts we patch. GraspGenX is the only one: it carries the
joint-limit fix, without which its playback drives the hand's joints hundreds of
degrees past their range. The patch itself stays in ``patches/`` so the checkout
can be re-cloned and the change re-applied rather than quietly becoming part of
it.

Both are symlinks rather than clones because each carries a multi-gigabyte
virtual environment with mutually incompatible pins -- GraspGenX needs torch
<2.7 on cu124, Dexonomy pins mujoco 3.3.3, this project's own venv is a third.

``data`` cannot be a relative link -- it points at whichever disk has room -- so
it is gitignored, and :func:`check` reports it missing rather than failing
obscurely several steps later.

Import ``paths`` and use the constants; do not rejoin strings from ``__file__``
in individual scripts, which is how the absolute paths accumulated the first
time.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root: this file is ``<root>/src/icra/paths.py``.
ROOT = Path(__file__).resolve().parents[2]

# -- external checkouts, via third_party/ ----------------------------------

THIRD_PARTY = ROOT / "third_party"
#: Checkouts we patch; see the module docstring for why this is not third_party.
MODULES = ROOT / "modules"

GRASPGENX = MODULES / "graspgenx"
DEXONOMY = THIRD_PARTY / "dexonomy"

#: Read-only references. Nothing here is run from this repo; they are checked out
#: so the code being followed is at hand and pinned.
DEXLEARN = THIRD_PARTY / "dexlearn"
GRASPVLA = THIRD_PARTY / "graspvla"
DEXGRASPVLA = THIRD_PARTY / "dexgraspvla"

#: Interpreters. Each checkout carries its own environment: GraspGenX needs
#: torch <2.7 on cu124, Dexonomy pins mujoco 3.3.3, and this project's own venv
#: is a third. Nothing is gained by trying to merge them.
GRASPGENX_PYTHON = GRASPGENX / ".venv/bin/python"
DEXONOMY_PYTHON = DEXONOMY / ".venv/bin/python"
DEXONOMY_RUN = DEXONOMY / ".venv/bin/dexrun"

#: Dexonomy's per-hand asset directory, written by onboard_dexonomy_hand.py.
DEXONOMY_HAND = DEXONOMY / "assets/hand/hx5_d20"

# -- models, in the repo ----------------------------------------------------

MODEL = ROOT / "model"
#: Vendored upstream URDFs, kept beside the generated ones for a readable diff.
HAND_URDF_SOURCE = MODEL / "hx5_d20_right.urdf.orig"
ROBOT_URDF_SOURCE = MODEL / "ffw_sh5_follower.urdf.orig"

GENERATED = MODEL / "graspgen"
HAND_URDF = GENERATED / "hx5_d20_right/gripper.urdf"
ROBOT_URDF = GENERATED / "ffw_sh5/ffw_sh5.urdf"
#: Scene layout saved by scene_setup_gui.py.
SCENE = GENERATED / "ffw_sh5/scene.yaml"

MESHES = MODEL / "assets"

#: Hand poses, one per grasp type, found by ``scripts/find_grasp_poses.py``.
#:
#: This replaces GraspGenX's gripper config as the source of the hand's poses.
#: That config holds exactly two -- ``open`` and ``close`` -- which cannot
#: describe a hand that wraps a bottle one way and pinches a ball another, and
#: any joint it leaves at zero in both can never be reached by interpolating
#: between them. On this hand that silently pinned the knuckles, whose 0..2.0 rad
#: range is the largest it has. The poses here are searched for against the
#: objects each grasp type is meant to handle.
GRASP_POSES = GENERATED / "hx5_d20_right/grasp_poses.json"

# -- generated bulk, via data/ ---------------------------------------------

DATA = ROOT / "data"
#: Object meshes plus the convex decomposition and scene config Dexonomy needs.
OBJECTS = DATA / "objects"
#: Dexonomy's successful grasps, which is what DexLearn trains on.
GRASPS = DATA / "grasps"
#: Rendered DexGraspVLA episodes.
EPISODES = DATA / "episodes"
#: Scratch for individual pipeline runs.
RUNS = DATA / "runs"

#: Small outputs worth looking at directly; gitignored but on the repo disk.
OUT = ROOT / "out"


def check(*required: Path) -> list[str]:
    """Report which of ``required`` is missing, as human-readable lines.

    Returns an empty list when everything resolves. Callers should print the
    lines and stop, rather than proceeding to fail somewhere less obvious -- a
    missing ``data`` symlink otherwise surfaces as a confusing error inside
    whichever subprocess happened to need it first.
    """
    problems = []
    for path in required:
        if path.exists():
            continue
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        hint = ""
        if not DATA.exists() and DATA in path.parents:
            hint = "  (create the data symlink: ln -s /path/with/room data)"
        elif THIRD_PARTY in path.parents or MODULES in path.parents:
            hint = "  (expects GraspGenX and Dexonomy checked out beside this repo)"
        problems.append(f"missing: {shown}{hint}")
    return problems
