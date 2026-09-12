"""Onboard the HX5-D20 right hand into Dexonomy.

Dexonomy optimises a hand's joint angles per object, in MuJoCo, for a
contact-rich penetration-free grasp. This writes the files it needs for a new
hand, and one template per grasp type.

Where the hand comes from
-------------------------
Two sources, and which is right is not yet settled.

``--from urdf`` (the default) compiles our exported URDF: convex hulls and a palm
decomposed by CoACD. Grasps refined against it hold when Dexonomy shakes them --
236 of 295 on three objects -- and *lose most of their contacts* when executed
against ROBOTIS's model, which is the geometry the episodes actually use. The
same grasp that makes five to seven contacts here makes one there, and one
contact holds nothing: measured, 1.1 N against an object weighing 1.05 N.

``--from robotis`` extracts the hand from ROBOTIS's whole-robot model, which is
the geometry that will execute the grasp and so ought to be the one refined
against -- and having grasps refined in one hand and run in another is what made
the earlier ones worthless, five to seven contacts where they were scored and one
where they ran.

The skeleton was long suspected of breaking this build, because it is generated
from the URDF and spans 39x183x222 mm against the collision meshes' 65x208x236.
It is not the cause. Every one of its twenty bones lands within 0.00 mm of the
corresponding body in ROBOTIS's model, palm included: the two models' link frames
agree exactly, and the palm's five bones are re-keyed from the URDF's name to
``hx5_r_base`` here for exactly that reason. The size difference is the point of
a skeleton -- ``op=init`` collides bone *centre-lines* against the object mesh,
so a pose it accepts routinely has the hand's solid parts overlapping the object,
in this hand and in every hand Dexonomy ships. That is what refinement is for,
and it is not evidence of anything being wrong.

The poses come from ``scripts/assets/retarget_templates.py``, which carries the taxonomy
over from a hand somebody has already posed -- the Shadow's thirty-six human
annotations. Before that they came from ``scripts/assets/find_grasp_poses.py``, which
searches the hand's own joint space against objects each type is meant to handle;
that is automatic but produces poses no hand makes, with all four fingers at
identical flexion, and it is kept only as the fallback for a hand with no
annotated counterpart to retarget from. They used to come
from GraspGenX's onboarding wizard, which records one ``open`` and one ``close``
-- a pair that cannot describe a hand that wraps a bottle one way and pinches a
ball another, and that silently pins any joint it leaves at zero in both. On
this hand that was the knuckles, whose 0..2.0 rad range is the largest it has,
so no pose reachable from that pair ever folded a finger at its base.

The contact annotation, and why it is derived rather than picked
----------------------------------------------------------------
``convert_template.py`` fills ``contact`` with an empty dict when the annotation
omits it, so a template *file* builds from a ``qpos`` alone. That is not enough
to be useful: ``gen_grasp.py``'s refinement loop is

    curr_diff = sim_env.apply_contact_forces(hand_cbody, hand_cpn_b, obj_cpn_w)

and ``apply_contact_forces`` iterates over the annotated bodies, pulling each
annotated hand point toward its matched point on the object. With no contacts
the loop body never runs and the hand does not move. The ``contact`` **filter**
is a config switch; the ``contact`` **annotation** is the drive.

What the tutorial has a human do in CloudCompare is pick, on each finger pad, the
point that faces the object and read off its normal. That is derivable from
geometry we already have. :func:`derive_contacts` sets the hand to the annotated
close pose and, for each pad, takes the surface point nearest the **centroid of
the five fingertips** -- which is where the object sits when the hand is closed
-- with that point's outward surface normal, written in the handframe the
annotation format expects.

Using the fingertip centroid rather than the GraspGenX sweep volume's centre is
deliberate: it is intrinsic to the hand, so it needs none of the base-rotation
and world-joint composition that the sweep box's frame depends on, and that
composition is exactly where this project has repeatedly gone wrong.

``op=tmpl`` re-checks every derived point against the collision meshes and warns
if one is far from its body's surface, so the derivation is verified by the tool
that consumes it.

What is derived here
--------------------
* ``right.xml`` -- MuJoCo compiles the URDF directly; this injects the
  ``<mujoco><compiler meshdir>`` block URDF needs for its mesh paths to resolve,
  then adds the position actuators Dexonomy drives the hand with. ``kp=5`` and
  ``forcerange=[-3, 3]`` are the tutorial's recommendation for its 0.1 kg test
  object; ``kp=1`` is called out there as a cause of abnormal squeeze qpos.
* ``body_group.yaml`` -- one group per finger, plus the palm. Dexonomy uses these
  to relax its contact requirement from "every annotated body touches" to "at
  least one body per group touches", so the grouping has to follow the real
  finger chains. Walked from the URDF tree rather than parsed out of the joint
  names, for the same reason the MJCF loader in ``src/icra/robots/hand.py`` does it that
  way: reading structure off a naming convention is how a ring joint ends up
  labelled a thumb joint.
* ``skeleton.yaml`` -- a line segment per link, from the link's own origin to its
  child joint's origin. That is the bone, and it is exactly what the URDF already
  encodes, so there is nothing to eyeball.
* ``raw_anno/<name>.yaml`` -- the qpos, in MJCF joint order.
* the hand config yaml, written into the Dexonomy checkout.

Run with Dexonomy's interpreter::

    cd /path/to/Dexonomy
    .venv/bin/python /path/to/ICRA/scripts/assets/onboard_dexonomy_hand.py
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

HAND_NAME = "hx5_d20"

#: Position gain for the finger servos. The tutorial recommends 5 for its 0.1 kg
#: test object and that is what this used, but this hand's links are far heavier
#: than the Shadow's and 5 gives them no authority: the fingers cannot reproduce
#: the contact forces a grasp was refined to hold, so a grasp that survives
#: ``op=eval`` when the hand is *placed* at its refined pose fails when the hand
#: has to arrive there under servo control. Swept over the 918 refined grasps of
#: the three scene objects, counting how many survive the shake test:
#:
#:     kp    5 ->  34     kp   20 -> 731
#:     kp   50 -> 756     kp  100 -> 736
#:
#: For a long time this looked like the approach knocking the object aside. It
#: is not: traced segment by segment, the object does not move at all on the way
#: in. It was always the grip.
ACTUATOR_KP = 50.0
ACTUATOR_FORCERANGE = (-3.0, 3.0)

#: CoACD concavity threshold for the palm. Same value ``scripts/assets/decompose_palm.py``
#: settles on for the MJCF model. See :func:`decompose_palm`.
PALM_COACD_THRESHOLD = 0.03
#: Cap on convex pieces, matching ``decompose_palm.py``. Uncapped, this palm
#: decomposes into 371 parts -- a faithful fit, but every one is a collision geom
#: that MuJoCo pairs against twenty finger links on every step.
PALM_MAX_PARTS = 32
PALM_MESH = "hx5_d20_base_unit"
#: Toggle for the experiment below; see :func:`decompose_palm`.
#:
#: On: only the thumb and pinky knuckles sit inside a palm piece. Off: the single
#: convex hull of the palm swallows all five knuckles, at every pose. Measured on
#: this hand, so this stays on.
DECOMPOSE_PALM = True

#: How far apart in the kinematic tree two links may be and still have their
#: contact excluded. Parent/child (distance 1) MuJoCo already filters. Distance 2
#: is a knuckle against the palm it is mounted on, or two neighbouring knuckles --
#: pairs that overlap because a convex collision hull is bigger than the part it
#: stands for, not because the hand is in a bad pose. Every hand Dexonomy ships
#: lists exactly these by hand; see :func:`derive_excludes`.
EXCLUDE_TREE_DISTANCE = 2

#: The palm, as the URDF names it -- used to walk the finger chains.
PALM_LINK = "hx5_d20_right_base"

#: The same link, as ROBOTIS's model names it. Dexonomy looks its ``body_group``
#: and ``skeleton`` entries up by MJCF body name, and ``get_body_id`` raises on
#: anything else.
PALM_BODY = "hx5_r_base"

#: The links that actually touch an object: the distal and tip of each finger.
#: The palm is not among them -- MuJoCo folds it into the ``world`` body when it
#: compiles the URDF, because ``world_joint`` is fixed, so it has no body name of
#: its own to annotate against.
CONTACT_LINKS = tuple(
    f"finger_r_link{i}" for chain in range(5) for i in (4 * chain + 3, 4 * chain + 4)
)

#: The five fingertip links, whose centroid stands in for where a grasped object
#: sits when the hand is closed.
TIP_LINKS = tuple(f"finger_r_link{4 * chain + 4}" for chain in range(5))

#: Which links belong to each digit, in the order find_grasp_poses.py names them.
#: Only the two outermost of each chain are annotated -- those are the pads.
DIGIT_LINKS = {
    name: (f"finger_r_link{4 * chain + 3}", f"finger_r_link{4 * chain + 4}")
    for chain, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=paths.HAND_URDF,
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=paths.GRASP_POSES,
        help="Grasp-type poses, one template per entry. retarget_templates.py "
             "writes these now, carrying the taxonomy over from a hand a person "
             "already posed; find_grasp_poses.py searched for them instead and is "
             "kept as the fallback for a hand with no annotated counterpart.",
    )
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    parser.add_argument(
        "--from", dest="source", choices=["urdf", "robotis"], default="urdf",
        help="Where the hand's geometry comes from; see the module docstring.",
    )
    return parser.parse_args(argv)


def urdf_tree(urdf: Path) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[float]]]:
    """``(child link -> parent link)``, ``(parent -> children)``, joint origins."""
    root = ET.parse(urdf).getroot()
    parent_of: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    origin_of: dict[str, list[float]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        parent_of[child] = parent
        children.setdefault(parent, []).append(child)
        origin = joint.find("origin")
        xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
        origin_of[child] = [float(v) for v in xyz.split()]
    return parent_of, children, origin_of


def finger_groups(urdf: Path) -> list[list[str]]:
    """One list of link names per finger, walked from the palm outward."""
    _, children, _ = urdf_tree(urdf)
    groups = []
    for start in children.get(PALM_LINK, []):
        chain, node = [], start
        while node is not None:
            chain.append(node)
            kids = [c for c in children.get(node, []) if not c.startswith("finger_end")]
            node = kids[0] if kids else None
        groups.append(chain)
    return groups


def skeleton(urdf: Path) -> dict[str, list[list[float]]]:
    """A bone per link: origin to the origin of its child joint."""
    _, children, origin_of = urdf_tree(urdf)
    bones: dict[str, list[list[float]]] = {}
    for link, kids in children.items():
        if link == "world":
            continue
        for kid in kids:
            end = origin_of[kid]
            if max(abs(v) for v in end) < 1e-6:
                continue
            bones.setdefault(link, []).append([0.0, 0.0, 0.0, *end])
    return bones


def decompose_palm(mesh_dir: Path, root: ET.Element) -> int:
    """Replace the palm's collision geometry with convex parts.

    MuJoCo's narrow phase only collides convex shapes, so a ``<geom type="mesh">``
    is silently stood in for by that mesh's convex hull. For this palm the hull is
    475.9 cm3 against a real volume of 172.3 cm3 -- it fills in the hollow the
    fingers are meant to curl into, so they collide with geometry the hand does
    not have. ``scripts/assets/decompose_palm.py`` documents the measurement.

    Dexonomy rejects any grasp with hand-hand penetration, and on the first run
    every one of twenty refined grasps failed on exactly this: five overlapping
    body pairs at the close pose, up to 7.4 mm, **all of them palm-against-finger
    and none finger-against-finger**. A union of convex pieces represents the
    cavity, so the fingers can close into it.

    The original mesh geom is kept and made visual-only rather than deleted --
    nothing is wrong with how the palm looks.
    """
    import coacd
    import trimesh

    palm = trimesh.load(mesh_dir / f"{PALM_MESH}.stl", force="mesh")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(palm.vertices), np.asarray(palm.faces)),
        threshold=PALM_COACD_THRESHOLD,
        max_convex_hull=PALM_MAX_PARTS,
    )

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    for index, (verts, faces) in enumerate(parts):
        name = f"palm_col_{index}"
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
            mesh_dir / f"{name}.obj"
        )
        ET.SubElement(asset, "mesh", {"name": name, "file": f"{name}.obj"})

    for body in worldbody.iter("body"):
        pass  # the palm's geom sits on the worldbody itself, not a child body
    for geom in worldbody.findall("geom"):
        if geom.get("mesh") != PALM_MESH:
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "2")
        for index in range(len(parts)):
            ET.SubElement(
                worldbody,
                "geom",
                {"type": "mesh", "mesh": f"palm_col_{index}", "group": "3"},
            )
        break
    return len(parts)


def derive_excludes(mjcf: Path) -> list[tuple[str, str]]:
    """Body pairs that overlap in the hand's rest pose, to be excluded from contact.

    A hand with every joint at zero cannot really be inside itself, so anything
    reported there is the collision model rather than the pose: MuJoCo collides
    convex geometry, and the convex hull of a knuckle housing reaches into the
    palm that houses it. Left in, these show up as a constant self-penetration
    and Dexonomy's ``hh_thre = 0.0`` rejects every grasp before it starts.

    Every hand Dexonomy ships carries the same list written out by hand -- the
    Shadow Hand excludes ``rh_palm``/``rh_ffproximal`` and six more, Allegro five,
    the G1 four. This derives them instead, and restricts the result to pairs
    within :data:`EXCLUDE_TREE_DISTANCE` of each other so that a genuine
    collision between distant links, such as a thumb tip folding into the palm,
    is still reported.

    The model has to be built the way ``MujocoEnv._add_hand`` builds it: attached
    as a body *and* given a freejoint. Loaded directly the palm is the world body
    and welded to it, and MuJoCo's parent filter does not apply to the world weld
    group -- so the palm collides with its own fingers whatever is excluded, and
    the numbers bear no relation to what Dexonomy sees.
    """
    import mujoco

    spec = mujoco.MjSpec()
    child = mujoco.MjSpec.from_file(str(mjcf))
    for mesh in child.meshes:
        mesh.file = str(mjcf.parent / child.meshdir / mesh.file)
    child.meshdir = ""
    # Prefixed because the hand's own root link is called "world", which would
    # otherwise clash with the parent worldbody; stripped again below so the
    # pairs name bodies as the MJCF itself does.
    prefix = "hand-"
    spec.worldbody.add_frame().attach_body(
        child.body(PALM_BODY) or child.worldbody, prefix, ""
    ).add_freejoint(name="fj")

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0  # the freejoint's quaternion
    mujoco.mj_forward(model, data)

    def ancestors(body: int) -> list[int]:
        chain = [body]
        while body:
            body = model.body_parentid[body]
            chain.append(body)
        return chain

    def tree_distance(a: int, b: int) -> int:
        up_a, up_b = ancestors(a), ancestors(b)
        for steps, node in enumerate(up_a):
            if node in up_b:
                return steps + up_b.index(node)
        return len(up_a) + len(up_b)

    def name(body: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body).removeprefix(
            prefix
        )

    pairs = set()
    for contact in data.contact[: data.ncon]:
        if contact.dist >= 0:
            continue
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if tree_distance(a, b) <= EXCLUDE_TREE_DISTANCE:
            pairs.add(tuple(sorted((name(a), name(b)))))
    return sorted(pairs)


def _finish_mjcf(dest: Path, names: list[str]) -> list[str]:
    """Derive the contact exclusions and fold them into the file just written."""
    excludes = derive_excludes(dest)
    if excludes:
        root = ET.parse(dest).getroot()
        contact = ET.SubElement(root, "contact")
        for first, second in excludes:
            ET.SubElement(contact, "exclude", {"body1": first, "body2": second})
        ET.indent(root, space="  ")
        dest.write_text(ET.tostring(root, encoding="unicode"))
        print(f"  excluded {len(excludes)} structurally overlapping pairs:")
        for first, second in excludes:
            print(f"    {first} <-> {second}")
    return names


def build_from_robotis(dest: Path) -> list[str]:
    """Lift the right hand out of ROBOTIS's whole-robot model."""
    import mujoco

    from icra.scene import ROBOT_MJCF

    source = mujoco.MjSpec.from_file(str(ROBOT_MJCF))
    hand = source.body(PALM_BODY)
    if hand is None:
        raise KeyError(f"{PALM_BODY!r} is not a body in {ROBOT_MJCF}")

    spec = mujoco.MjSpec()
    spec.meshdir = str((Path(ROBOT_MJCF).parent / source.meshdir).resolve())
    root = spec.worldbody.add_frame().attach_body(hand, "", "")
    # At the origin, facing along the file's own axes. It arrives carrying the
    # pose it had on the robot -- 78 mm down and turned -- and everything
    # downstream assumes the palm frame *is* the file's frame: derive_contacts
    # writes its annotation in world coordinates because Dexonomy reads that as
    # handframe, and calibrate_gripper measures the tool centre the same way.
    root.pos = [0.0, 0.0, 0.0]
    root.quat = [1.0, 0.0, 0.0, 0.0]

    # ROBOTIS gives the palm a single mesh geom for collision, which MuJoCo
    # collides as its convex hull -- 2.76 times the palm's real volume, filling
    # the cavity the fingers close into. The URDF build has always cut it up
    # (:func:`decompose_palm`); this build was refining grasps against a hand
    # whose fingers could not close, which is the likeliest reason it produced
    # nothing usable.
    from icra.scene import open_palm

    open_palm(spec, Path(spec.meshdir))

    names = [joint.name for joint in spec.joints]
    # The geometry is ROBOTIS's; the servos are not. Dexonomy refines by applying
    # contact forces and stepping, so the hand has to be pushed into shape.
    low, high = ACTUATOR_FORCERANGE
    by_name = {actuator.name: actuator for actuator in spec.actuators}
    for name in names:
        actuator = by_name.get(name) or spec.add_actuator(
            name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        actuator.gainprm = [ACTUATOR_KP] + [0.0] * 9
        actuator.biasprm = [0.0, -ACTUATOR_KP] + [0.0] * 8
        actuator.forcerange = [low, high]
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # MjSpec's XML carries a <default> with an empty class name that MuJoCo will
    # not read back, so it is unwrapped on the way out.
    spec.compile()
    root_xml = ET.fromstring(spec.to_xml())
    for parent in root_xml.iter():
        for child in list(parent):
            if child.tag == "default" and not child.get("class"):
                for grandchild in list(child):
                    parent.append(grandchild)
                parent.remove(child)
    for compiler in root_xml.findall("compiler"):
        compiler.set("meshdir", spec.meshdir)
    ET.indent(root_xml, space="  ")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(ET.tostring(root_xml, encoding="unicode"))
    return _finish_mjcf(dest, names)


def build_from_urdf(urdf: Path, dest: Path) -> list[str]:
    """Compile our exported URDF, with position actuators and a decomposed palm."""
    import mujoco

    # MuJoCo drops the directory part of a URDF mesh path and looks beside the
    # URDF, so the meshdir has to be declared in a <mujoco> block inside it.
    text = urdf.read_text()
    if "<mujoco>" not in text:
        insert = ('  <mujoco>\n    <compiler meshdir="meshes" '
                  'balanceinertia="true"/>\n  </mujoco>\n')
        first_child = text.index(">", text.index("<robot")) + 1
        text = text[:first_child] + "\n" + insert + text[first_child:]
    staged = urdf.parent / "_dexonomy_staging.urdf"
    staged.write_text(text)
    try:
        model = mujoco.MjModel.from_xml_path(str(staged))
        mujoco.mj_saveLastXML(str(dest), model)
    finally:
        staged.unlink(missing_ok=True)

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    root = ET.parse(dest).getroot()
    for compiler in root.findall("compiler"):
        compiler.set("meshdir", "meshes")
    actuator = ET.SubElement(root, "actuator")
    low, high = ACTUATOR_FORCERANGE
    for name in names:
        ET.SubElement(actuator, "position", {
            "name": name, "joint": name, "kp": str(ACTUATOR_KP),
            "forcerange": f"{low} {high}",
        })
    parts = decompose_palm(dest.parent / "meshes", root) if DECOMPOSE_PALM else 0
    ET.indent(root, space="  ")
    dest.write_text(ET.tostring(root, encoding="unicode"))
    if parts:
        print(f"  palm decomposed into {parts} convex parts")
    return _finish_mjcf(dest, names)


def build_mjcf(urdf: Path, dest: Path, source: str = "urdf") -> list[str]:
    """The standalone hand MJCF, from whichever source is asked for."""
    if source == "robotis":
        return build_from_robotis(dest)
    return build_from_urdf(urdf, dest)


def derive_contacts(
    mjcf: Path, qpos: dict[str, float], digits: list[str] | None = None
) -> dict[str, list[float]]:
    """One contact point + outward normal per pad, in that pad's body frame.

    Returns ``{body name: [x, y, z, nx, ny, nz]}``, the format Dexonomy's
    ``raw_anno`` expects. Reads the MJCF rather than the URDF so the points land
    on exactly the meshes ``op=tmpl`` will check them against. ``qpos`` is the
    grasp type's own pose, so each type gets its own contact annotation.

    ``digits`` restricts which of them are annotated, and has to: the annotated
    bodies become the template's ``required_cbody``, and ``op=grasp`` rejects any
    grasp that does not have *all* of them touching. Annotating all five fingers
    on a two-finger pinch demands a five-finger grasp of it and nothing passes.
    """
    import mujoco
    import trimesh

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        data.qpos[model.jnt_qposadr[index]] = qpos[name]
    mujoco.mj_forward(model, data)

    def body_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    wanted = (
        CONTACT_LINKS
        if digits is None
        else tuple(link for digit in digits for link in DIGIT_LINKS[digit])
    )
    # The object sits where the participating fingertips converge, so a two-finger
    # pinch aims its contacts at its own two tips rather than at the centre of a
    # fist it never makes.
    tips = (
        TIP_LINKS if digits is None else tuple(DIGIT_LINKS[digit][1] for digit in digits)
    )
    centre = np.mean([data.xpos[body_id(n)] for n in tips], axis=0)

    contacts: dict[str, list[float]] = {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name not in wanted:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = np.asarray(model.mesh_vert[start : start + count], dtype=float)

        # Work entirely in the body frame -- that is the frame the annotation is
        # written in, and round-tripping through world coordinates is what put
        # the first attempt 111 mm off its own mesh. The geom carries its own
        # pose inside the body, so lift the mesh vertices through that.
        geom_rot = np.zeros(9)
        mujoco.mju_quat2Mat(geom_rot, model.geom_quat[geom])
        geom_rot = geom_rot.reshape(3, 3)
        verts_b = verts @ geom_rot.T + model.geom_pos[geom]
        # Dexonomy checks the point against the body's **collision** mesh, and
        # MuJoCo's collision geometry is the convex hull (``mujoco_env`` takes
        # ``tm.convex_hull``). Picking a raw vertex can land in a concavity that
        # the hull bridges over -- one pad came out 2.5 mm outside its own hull,
        # against a 2 mm tolerance. Hull vertices are a subset of the mesh's, so
        # choosing among them puts the point exactly on the surface checked.
        hull = trimesh.Trimesh(vertices=verts_b, process=False).convex_hull
        verts_b = hull.vertices

        body_rot = data.xmat[body].reshape(3, 3)
        centre_b = body_rot.T @ (centre - data.xpos[body])

        nearest = int(np.argmin(np.linalg.norm(verts_b - centre_b, axis=1)))
        point_b = verts_b[nearest]

        # The hull's own outward normal at that vertex. Taking it as the
        # direction from the mesh centroid to the vertex instead -- which is
        # what this did -- lands a median 38 degrees off, because these links
        # are long and thin and the vertex nearest the grasp centre sits near
        # the tip, so that direction runs *along* the finger rather than out of
        # the pad. op=init matches these normals to the object's surface and
        # its filter wants them within 32 degrees, so a 38 degree annotation
        # error on its own puts every grasp of this hand outside the threshold:
        # measured 0 of 192 passing, and 0 of 10 by Dexonomy's own unseeded
        # route. trimesh orients a convex hull outward, so no sign check is
        # needed here.
        outward = np.array(hull.vertex_normals[nearest], dtype=float)
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            continue
        # ``convert_template.py`` treats an inline 6-vector as **handframe** --
        # world coordinates at the annotated pose -- and only resolves body-frame
        # values when they come through ``anno_keypoint.yaml`` by index. The
        # search above runs in the body frame because that is where the mesh
        # lives; the output has to be lifted back out to world.
        point_w = body_rot @ point_b + data.xpos[body]
        normal_w = body_rot @ (outward / norm)
        contacts[name] = [round(float(v), 6) for v in (*point_w, *normal_w)]
    return contacts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hand_dir = args.dexonomy / "assets" / "hand" / HAND_NAME
    (hand_dir / "raw_anno").mkdir(parents=True, exist_ok=True)

    shutil.copytree(args.urdf.parent / "meshes", hand_dir / "meshes", dirs_exist_ok=True)
    joints = build_mjcf(args.urdf, hand_dir / "right.xml", args.source)

    groups = finger_groups(args.urdf)
    if len(groups) != 5:
        raise ValueError(
            f"walked {len(groups)} chains off {PALM_LINK}, expected 5. The hand's "
            f"structure is not what this assumes; inspect it rather than relabelling."
        )
    palm = PALM_BODY if args.source == "robotis" else "world"
    body_group = [*groups, [palm]]
    (hand_dir / "body_group.yaml").write_text(
        "body_group: [\n" + "".join(f"    [{','.join(g)}],\n" for g in body_group) + "]\n"
    )

    bones = skeleton(args.urdf)
    bones = {(palm if k == PALM_LINK else k): v for k, v in bones.items()}
    lines = []
    for link, segs in bones.items():
        rows = ",\n  ".join("[" + ", ".join(f"{v:.5f}" for v in s) + "]" for s in segs)
        lines.append(f"{link}: [\n  {rows},\n]")
    (hand_dir / "skeleton.yaml").write_text("\n".join(lines) + "\n")

    poses = json.loads(args.poses.read_text())
    written = []
    for template_name, entry in sorted(poses.items()):
        pose = entry["qpos"]
        qpos = [float(pose[name]) for name in joints]
        contacts = derive_contacts(hand_dir / "right.xml", pose, entry["digits"])
        contact_block = "".join(
            f"    {body}: [{', '.join(f'{v:.6f}' for v in cpn)}]\n"
            for body, cpn in sorted(contacts.items())
        )
        (hand_dir / "raw_anno" / f"{template_name}.yaml").write_text(
            f"# {entry['doc']}\n"
            "#\n"
            "# qpos in MJCF joint order, searched for by scripts/assets/find_grasp_poses.py\n"
            f"# against this type's own objects ({entry['placements_held']} placements\n"
            f"# held, {len(entry['digits'])} digits: {', '.join(entry['digits'])}).\n"
            "# Only those digits are annotated: the annotated bodies become the\n"
            "# template's required contacts, and op=grasp demands all of them.\n"
            "# The contacts below are derived, not picked by hand -- see\n"
            "# derive_contacts() in scripts/assets/onboard_dexonomy_hand.py.\n"
            "qpos: [" + ", ".join(f"{v:.4f}" for v in qpos) + "]\n\n"
            "contact:\n" + contact_block
        )
        written.append((template_name, len(contacts)))

    cfg_path = args.dexonomy / "dexonomy" / "config" / "hand" / f"{HAND_NAME}.yaml"
    cfg_path.write_text(
        f"xml_path: ${{asset_root}}/hand/{HAND_NAME}/right.xml\n"
        "color: [0.55, 0.60, 0.70, 1.0]\n"
        "arm_hand:\n"
    )

    print(f"wrote {hand_dir}")
    print(f"  right.xml        {len(joints)} joints, position actuators kp={ACTUATOR_KP}")
    print(f"  body_group.yaml  {len(groups)} fingers + palm")
    print(f"  skeleton.yaml    {sum(len(v) for v in bones.values())} bones over {len(bones)} links")
    for template_name, count in written:
        print(f"  raw_anno/{template_name}.yaml  ({count} derived contacts)")
    print(f"  {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
