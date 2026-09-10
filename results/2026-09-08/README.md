# Historical stage ladder, 2026-09-08

These raw TSVs predate the current ladder audit. Keep them unchanged. They are
hand-only replay/shake diagnostics, not arm or visual-policy success rates.

| Rung | Intended change | Criterion |
|---|---|---|
| 0 | Floating object, teacher hand/servos | Dexonomy shake test |
| 1 | Execution servo parameters on the same hand geometry | Same shake test |
| 2 | Add a plane derived from the resting pose | Same shake test, with external table support/obstruction |
| 3 | Full arm/scene execution | Unimplemented; the current script rejects this request |

Rungs 0–2 help localize differences within this replay setup. They do not prove
a single general cause of arm failure. A full-scene lift changes geometry,
physical settings, trajectory, and criterion as well as adding the arm.

## Synthesis candidates

[ladder_synthesis.tsv](ladder_synthesis.tsv) contains 240 candidates sampled
across 12 types: 240 passed rung 0, 234 passed rung 1, and 68 passed rung 2.
The original report included one candidate on
core_mug_10f6e09036350e92b3f21f1137c3c347 whose table setup was flagged invalid.
That row passed rung 1 and failed rung 2 in the TSV. Consequently, claiming it
was excluded while reporting 68/234 was inconsistent: excluding that flagged row
would give 68/233, not 68/234.

Even 68/233 is not a newly validated table-feasibility estimate. The old support
check could count hand-plane penetration as invalid object support, and the TSV
encoded invalid/skipped trials as zero. The old flag cannot be conclusively
reinterpreted without rerunning the corrected geometric check. Retain the raw
counts as historical program output and rerun with explicit valid/invalid status.

## Search candidates

[ladder_search.tsv](ladder_search.tsv) records 1/240 at rung 0. Synthesis
candidates had already been selected by this shake criterion; search candidates
had been selected by a different lift procedure. Thus 240/240 versus 1/240 is
not a neutral comparison of candidate sources. Use a common downstream execution
protocol and disclose source selection when comparing them.

## Current reporting

The corrected script writes NA for unattempted or invalid rungs and a status
column for invalid support. Each printed denominator counts valid attempted
trials. Object support checks include collision meshes on descendant bodies,
exclude visual planes, and ignore hand-table contact. Execution-hand XML is
created in temporary storage. See [audit verification](audit/README.md) for the
bounded check; it does not replace this historical population experiment.

## Fixed executor and student follow-up

See the [fixed student experiment](fixed-student/README.md) for a fresh original-teacher
close/lift reproduction, synthetic data export, recurrent visual BC training, and
closed-loop evaluation. It uses an explicit open-hand grasp-position reset and
reports lift separately from the teacher contact groups it fails to preserve.

See also [contact and full-approach validation](contact-approach/README.md): a
screened candidate-specific direction reproduces home-to-lift, while the required
contact groups remain unpreserved.


[Contact-feedback closure](contact-feedback/README.md) subsequently reproduces
five-group lifts but exposes transient penetration. Gentler and explicitly
constrained failures are retained alongside successful diagnostic runs.

The [contact physics audit](contact-physics/README.md) subsequently found
numerical sensitivity and full-state restoration errors. The
[training preparation implementation](training-preparation/README.md) provides
fresh dynamic collection, schema-3 quality gates, independent splits and the
27D student entrypoints. No training was run in that preparation task.

The grip offset is now taken from what the render recorded rather than
reconstructed: [the fix](grip-offset/README.md), [what it is worth at the filter
stage](grip-offset-compare/README.md) (88 vs 27 passing variants of 600; 6 vs 3
of 10 candidates), and [the stages after it](grip-offset-pipeline/README.md),
where the successful Pincer run reproduces bit for bit. Those refinement counts
were then withdrawn: the evolved template library had been copied as dangling
symlinks. [Candidate provenance](grasp-provenance/README.md) pins the template
choice, carries prediction identity through the filters, and records op=eval
direction by direction.

Scaled to all 93 predictions with three alignment seeds
([grip-offset-93](grip-offset-93/README.md)), the small-sample direction
reverses: 7 of 93 predictions complete an execution with the corrected offset
against 14 of 93 without it, and 3 against 5 also hold six wrench directions.
The offset's measured advantage stays at the filter stage (57 vs 40 predictions
surviving); refinement is reached by 24 against 23, so the difference appears in
selection and execution. Holding six directions in simulation is not a proof of
force closure.

The approach environment ([approach-rl](approach-rl/README.md)) learns the arm's
approach and hands the grasp to the existing executor. Verified without training:
the recorded approach reaches handover at 2.65 mm and 0.15 degrees and still
completes, five negative controls all fail, and the approach and grasp budgets
are separated (12 s and 19 s against the recorded run's 18.5 s). Splitting
close_and_lift and approach_targets out of the executor left its episode sha256
unchanged.
