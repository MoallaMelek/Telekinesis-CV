# Precision follow-up review — 2026-10-04

Objective: address confusing targeting and inaccurate outlines, including dim light and shadows;
add normal-use hand skeletons. Preserve CPU EdgeSAM and the local webcam project.

Writer/verifier: Codex desktop. Consultant: separate read-only Codex agent; no edits or test
execution by the consultant. CLI unavailability is recorded in polish-review-response.md.

Reviewed: segmentation.py, lighting.py, selection.py, scene.py, main.py, hand_tracker.py,
hud.py, regression tests, and README. Target component ownership, small genuine holes,
weak-mask rejection and explicit alternatives replace indiscriminate cleanup/automatic cycling.
Lighting preparation affects inference only. The focused decoder reuses encoded features and
accepts only spatially consistent, more stable masks with bounded confidence/overlap changes.

Findings resolved: auxiliary components capped at half the target area; focused refinement
cannot lose over .10 score or increase hand/person overlap over .02. Tests cover close equal
neighbors, incorrect targets, weaker confidence and hand overlap. Reviewer confirmed no
remaining concrete blocker; recommendation approve. Confidence high in targeted correctness,
moderate in generalization. Model replacement was considered unnecessary without broader
representative evidence; darkness and clutter remain limitations.

Verification: 93 tests passed including real-model dark/noisy and shadow cases; synthetic
preview and public sample evaluation rendered; compilation, pip check, diff check passed.
Six-second GUI camera smoke detected hands, processed 124 frames and shut down cleanly.
No webcam images saved/uploaded. Synthetic mask overlap is not a broad accuracy benchmark.
Physical interaction feel remains a human verification step.
