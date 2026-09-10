# Manuscript scope, evidence, and submission preparation

## Research scope

The draft concerns transfer of Dexonomy grasp/contact/type knowledge through
synthetic demonstrations into a DexGraspVLA-style visual policy. The main task is
requested-object/part grasping in mild clutter on the FFW-SH5 / HX5-D20.
RL is an optional executor component, not a mandatory contribution.

ICRA is the preferred venue fit. The submission cycle and year are undecided.
Do not optimize the research around a presumed deadline. ICLR would require a
broader learning claim and appropriate evidence, not just different formatting.

## Evidence rules

* Treat the manuscript as a working draft. Separate implemented components,
  proposed mechanisms, hypotheses, and measured results.
* Do not write a result in past tense, fill a performance table, or assert a
  contribution is established until reproducible experiments support it.
* Old execution rates and the static-contact pose-tolerance claim are withdrawn;
  see [the evidence audit](../results/2026-09-07/README.md).
* A stored candidate count is not lift success; demonstrations are not learned
  policy results; simulation results are not hardware transfer.
* Claim no newly collected real-robot grasp demonstrations, not no human
  information. Existing human-annotated templates and pretrained models remain.
* Distinguish object selection from part/contact constraints and part grasping
  from demonstrated functional use.
* “Any object,” “any hand,” guaranteed closed-loop recovery, and zero-shot
  sim-to-real are unestablished ambitions unless evaluated explicitly.
* Cite primary work accurately. ClutterDexGrasp and DemoFunGrasp support the
  direction and constrain novelty; neither can be omitted to invent an empty
  synthetic/target-conditioned/closed-loop niche.
* Use matched student/data/compute controls, geometry-level splits before
  synthesis, explicit success criteria, failed-attempt denominators, seeds, and
  uncertainty. Keep historical raw logs unchanged.

## Writing and format

Keep one central argument: what teacher knowledge is preserved, how it reaches
the student, and what measured benefit it provides. Mark unresolved method
choices and experiment thresholds with TODOs rather than selecting favorable
values after evaluation. Keep limitations explicit.

The existing IEEE conference template is a working layout. Page limits,
reference allowances, appendices/supplements, anonymity requirements, and video
specifications depend on the selected venue and cycle. Check the official call
when a submission is chosen; no fixed page or video limit is asserted here.

Keep the draft's author block empty while preparing an anonymous version.
For an actual anonymous submission, also inspect acknowledgments, repository
links, supplementary files, and PDF metadata under that call's requirements.

## Submission checklist, after a venue/cycle is selected

1. Record the official call and author-instruction URLs and check the then-current
   format, anonymity, supplementary, and video requirements.
2. Build and inspect the PDF, including figures, citations, overflow, and metadata.
3. Verify every result against its archived run and every claim against the
   final implementation and experiments.
4. Prepare the required submission-form declarations. The project's settled
   workflow is to handle AI disclosure in the submission form; do not insert an
   unsolicited manuscript disclosure. Any explicit requirement of the selected
   venue must be checked against that workflow at submission time.
