<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# Contact physics and numerical sensitivity audit

No training was run. The saved geometry_repeat0 commands from contact-feedback-v3
were replayed at the same 50 Hz control frequency on the unchanged scene. These
are open-loop trajectory diagnostics, not reruns of the feedback controller.

| Numerical profile | Peak depth | Peak digit normal force | Final lift | Force-supported ticks |
|---|---:|---:|---:|---:|
| Historical forward measurement, 2 ms | 7.482 mm | 60.711 N | 15.163 cm | 141 |
| Preserve measurement warmstart, 2 ms | 7.482 mm | 60.711 N | 15.163 cm | 141 |
| Preserve warmstart, 1 ms | 6.181 mm | 58.380 N | 15.091 cm | 0 |
| Preserve warmstart, 2 ms, 300 iterations / tolerance 1e-12 | 7.487 mm | 60.707 N | 15.163 cm | 141 |

All four trajectories violate the 2 mm / 30 N diagnostic quality limits. At 1 ms,
the maximum simultaneous contact count is four. The successful contact pattern
is therefore sensitive to timestep in this open-loop test; neither numerical
convergence nor clean demonstrations are established. Tightening the solver
does not remove excessive penetration. The archived commands encode the original
2 ms controller's decisions, so this is not a test of adapting feedback at 1 ms.

## Confirmed code errors

- collect_episodes.main shadowed the imported paths module with a later loop
  variable, causing UnboundLocalError before path validation on every CLI run.
  The variable was renamed and CLI path validation now has a regression test.
- mj_forward mutates qacc_warmstart. Restoring integration state and immediately
  calling raw mj_forward did not restore the original integration state. A real
  scene check measured a maximum warmstart change of 36.658 after one step.
  New simulation helpers preserve full integration fields around observations.
- Recorder resets restored only qpos, qvel and ctrl. Previous simulation time,
  applied forces, activation and solver state could leak into the next episode.
  Resets now clear MjData before applying the intended initial pose and command.
- Dynamic IK and diagnostic probes restored only qpos/qvel. They now restore full
  integration state in try/finally.
- Arm.reach used arm_r_link6 as the shoulder and a sampled radius to reject
  targets. It now uses the first arm joint and a conservative serial-chain bound
  without changing caller state. The current robot's bound is 0.854132 m.
- Object masks ignored child bodies and segmentation object types; they now
  include descendant geoms and exclude matching IDs belonging to other types.

The warmstart fix changes this pinned replay's peak depth by only 0.00003 mm;
it is a reproducibility error, not an explanation of the 7.5 mm penetration.

## Physical assumptions retained and disclosed

Compiled finger servos have kp=500, kv=30 and a +/-3.09 Nm actuator limit, with
an additional joint damping coefficient of 1.0. The 30 N digit threshold is a
chosen diagnostic bound, not a hardware-certified limit. Reported digit pressure
is the sum of nonnegative contact-frame normal forces; it is neither the net
vector force nor actuator torque. [MuJoCo force extraction](https://mujoco.readthedocs.io/en/3.3.3/APIreference/APIfunctions.html#mj-contactforce)

Contact solref is [0.02, 1] and solimp is [0.9, 0.95, 0.001, 0.5, 2]. Soft contact
can have finite penetration under load. Equal-priority mixing also means the
object's intended anti-spin setting, condim=4 and torsional friction=0.05, affects
finger-object contacts, not only the table. It dominates the finger value 0.005.
No contact defaults were changed during this audit. [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/3.3.3/modeling.html#contact-parameters)

The torsional coefficient has units of length and represents support from a
finite contact patch; 0.05 must not be interpreted as dimensionless sliding
friction. [MuJoCo contact dimensions](https://mujoco.readthedocs.io/en/3.3.3/computation/index.html#contact)

The pinned object's compiled mass is 0.095288429969 kg despite the nominal
OBJECT_MASS=0.1. Density is based on metadata volume while mass is integrated
over collision pieces. The historical mass remains unchanged for this comparison;
new experiments must record compiled mass rather than assert exact 0.1 kg.

## Reproduction

Run with a fresh directory and MuJoCo 3.3.3:

```bash
OPENBLAS_NUM_THREADS=1 third_party/dexonomy/.venv/bin/python scripts/audit_contact_physics.py \
  --source out/contact-feedback-v3 --trajectory geometry_repeat0 \
  --out out/my-contact-physics-audit
```

Full per-profile summaries, compiled physics and source hashes from this run:
out/contact-physics-audit-v1/{manifest,results}.json. The script builds without a
renderer and performs no parameter fitting or training. Scene/robot assets and
the original saved action bundle are required.
