# Local patches for the optional GraspGenX route

These files reproduce local changes in the sibling NVlabs/GraspGenX checkout,
reached through modules/graspgenx. GraspGenX proposals are an optional candidate
source; the main research direction uses Dexonomy as teacher and synthetic
demonstrations to train a visual student. Current gripper calibration/search
also uses GraspGenX's asset layout.

Record the upstream commit and patch status with an experiment. Results below
are historical local diagnostics, not current pipeline or learned-policy scores.

## graspgenx-joint-limit-clamp.patch

Apply the patch inside the matching GraspGenX checkout after checking whether it
is already present:

~~~bash
git apply --check /path/to/ICRA/patches/graspgenx-joint-limit-clamp.patch
git apply /path/to/ICRA/patches/graspgenx-joint-limit-clamp.patch
~~~

The tested velocity controller commanded closing velocity throughout the phase
without stopping at the joint's close target. In the local HX5-D20 test, limit
constraints did not prevent large overshoot: twelve joints ended 6–7 radians
outside their URDF ranges. The recorded worst overrun was 391 degrees, with
about -1.17 N m limit force against +1.35 N m actuator force in one diagnostic.

The joint ranges were present in the imported model. The patch adds position
feedback to stop commanded closing velocity at the target. The local recorded
worst overrun fell to 0.9 degrees. That is a result for this controller and test,
not a universal joint-limit guarantee. Recheck limits and control timestep on the
actual model. Contact equilibrium alone is not a reliable stop condition for
fingers closing in free space.

## ur10e_hx5_d20_right.yaml

This is a historical end-to-end configuration for a UR10e with the HX5-D20,
intended for GraspGenX's end2end/robots directory. It is a stand-in platform,
not evidence about the FFW-SH5 right arm and not the paper's main embodiment.

grasp_to_tool_transform combines the gripper wizard's base rotation with the
URDF world-joint translation. Verify this convention against the pinned checkout
and actual exported frames before using it for a new hand or arm.

coacd_link_keywords restricts convex decomposition to the palm. A local attempt
to decompose every link produced 7034 shapes including an invalid zero-volume
sliver; palm-only decomposition produced 64 shapes in that configuration.
These counts describe that GraspGenX export, not scene.py's collision model.
gripper_body_keywords explicitly includes hx5_d20_right_base because generic
finger keywords do not select the palm.
