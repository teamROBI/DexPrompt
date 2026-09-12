#!/usr/bin/env bash
# Synthesise grasps by letting Dexonomy's template library evolve.
#
# This is how the current grasp set was made: 5085 refined grasps of which 4495
# survive the shake test, over twelve types and sixty objects, in about four
# hours on 32 cores.
#
# Three settings carry all of it, and each was wrong here for a long time:
#
#   op=grasp_admm, not op=grasp.  op=grasp only drags the template's annotated
#   hand points toward fixed points on the object, which cannot help when a
#   retargeted template's contacts do not match the object -- and on this hand
#   they do not. ADMM optimises the contact points as well. On identical
#   initializations: 6 grasps of 500 against 63, and 0 surviving the shake test
#   against 7.
#
#   Floating scene configs, no table.  op=init's plane_thre rejects any pose
#   whose hand skeleton comes within 20 mm of a plane, which an enveloping grasp
#   of anything on a table does: 0 grasps from 1800 with the table, 49 from 360
#   without. The table belongs where the grasp is executed.
#
#   ACTUATOR_KP = 50 in the hand this refines against.  At the tutorial's 5 the
#   fingers cannot reproduce the contact forces a grasp was refined to hold and
#   almost nothing survives op=eval: 34 survivors against 756 over the same 918
#   grasps.
#
# The loop itself is Dexonomy's own design and was the fourth thing missing.
# op=eval promotes every surviving grasp into new_tmpl and op=init reads its
# templates from init_tmpl *and* new_tmpl, so the library bootstraps out of this
# hand's own successes rather than staying with the retargeted Shadow poses.
# Every earlier run used op.epoch=1 once and never came back; survivors per round
# went 93, 352, 422, 345, 242, 183 as it evolved, and the library saturates at
# Dexonomy's tmpl_max_evo of two generations.
#
# init_data and the stage logs are cleared between rounds so nothing is skipped;
# grasp_data, succ_grasp and new_tmpl accumulate, and new_tmpl entries are
# symlinks into grasp_data, so that directory has to stay.
#
#   scripts/assets/build_object_set.py 100      # once, to pin the objects
#   ROUNDS=6 NCFG=60 scripts/teacher/evolve_grasps.sh
set -u
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/../../third_party/dexonomy"

EXP=${EXP:-evo}
ROUNDS=${ROUNDS:-6}
NCFG=${NCFG:-60}
WORKERS=${WORKERS:-24}
SAVE="output/${EXP}_hx5_d20"
OBJECTS="$(pwd)/assets/object/fixed/**/floating/*.npy"
TEMPLATES=${TEMPLATES:-"10_Power_Disk 11_Power_Sphere 1_Large_Diameter
6_Prismatic_4_Finger 3_Medium_Wrap 2_Small_Diameter 14_Tripod 9_Palmar_Pinch
8_Prismatic_2_Finger 33_Inferior_Pincer 22_Parallel_Extension 26_Sphere_4_Finger"}

# A stage that fails must say so. Sent to /dev/null and never checked, a failed
# op=init leaves the previous round's files in place and the round reports a
# poor yield instead of an error -- which is how a broken run reads as a result.
stage () {  # stage <name> <args...>
  local name="$1"; shift
  if ! .venv/bin/dexrun "op=$name" hand=hx5_d20 exp_name="$EXP" "$@" \
       > "$SAVE/last_$name.log" 2>&1; then
    echo "op=$name failed; see $SAVE/last_$name.log" >&2
    tail -5 "$SAVE/last_$name.log" >&2
    return 1
  fi
}

mkdir -p "$SAVE/new_tmpl"
printf 'round\tinits\tgrasps\tsurvived\tlibrary\ttime\n'
for R in $(seq 1 "$ROUNDS"); do
  # init_data and the stage logs go, so nothing is skipped on the next round.
  # grasp_data, succ_grasp and new_tmpl stay -- new_tmpl holds symlinks into
  # grasp_data, so removing it would break the library the loop exists to grow.
  rm -rf "$SAVE/init_data" "$SAVE/log"
  for T in $TEMPLATES; do
    stage init tmpl_name="$T" op.epoch=1 op.object.n_cfg="$NCFG" \
      "op.object.cfg_path=$OBJECTS" skip_done=False || exit 1
  done
  stage grasp_admm n_worker="$WORKERS" skip_done=False || exit 1
  stage eval       n_worker="$WORKERS" skip_done=False || exit 1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$R" \
    "$(find "$SAVE/init_data"  -name '*.npy' 2>/dev/null | wc -l)" \
    "$(find "$SAVE/grasp_data" -name '*.npy' 2>/dev/null | wc -l)" \
    "$(find "$SAVE/succ_grasp" -name '*.npy' 2>/dev/null | wc -l)" \
    "$(find "$SAVE/new_tmpl"   -name '*.npy' 2>/dev/null | wc -l)" \
    "$(date +%H:%M)"
done
