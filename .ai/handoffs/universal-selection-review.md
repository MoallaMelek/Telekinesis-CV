# Systemic selection review — 2026-10-05

Objective: arbitrary object categories, no mug/note-specific fixes; correct real-use failures
and a native Python memory-read crash reported by the human. Parent Codex desktop writes
and verifies; separate Codex consultant is read-only. CLI unavailability is recorded earlier.

Diagnosis: a local-only diagnostic of user-provided screenshots reproduced strong masks
rejected by coarse selfie foreground and imagined forearm hulls. Click-before-dwell was
ignored, mouse motion stole hand priority, and users could not express whole-object intent
when one point meant a part. Screenshots contain rendered HUD/skeletons, so this is a
policy diagnostic, not a raw-camera accuracy benchmark. No user image was exported/uploaded.
Native access violation reproduced when reading the borrowed selfie numpy_view after the
native owner was released; ascontiguousarray did not guarantee a copy. scene.py now copies.

Changes reviewed: scene.py native ownership, segmentation.py soft hints and all-positive
component retention, main.py pixel ownership/input/editor/confirmation transitions,
selection.py prompt versioning and pinned corrections, precision.py undo/box/point state,
tests, CI and docs. Existing CPU EdgeSAM and local architecture retained. No class lists,
network inference, private image assets, or class-specific thresholds were added.

Review blockers resolved: cleanup preserves all deliberately included components; corrected
objects skip later automatic silhouette refinement so exclusions cannot silently disappear.
The consultant confirmed both fixes, no remaining concrete blocker, approve with tests.

Verification: 107 tests passed including false foreground, immediate click/retry, input
precedence, native memory ownership, stale correction results, disconnected components,
preserved manual corrections, arbitrary synthetic shapes and real-model box/point isolation.
Real-model precision_demo verifies frozen image encoding reuse, correction, Enter and lift.
Existing full gesture/physics/refinement pipeline passes; compile/diff checks pass.

Risks: soft hints can admit skin/background; visible previews require confirmation and offer
correction. Anatomical cores cannot resolve depth perfectly. No guarantee of perfect masks
for every scene or hidden/near-black detail. Heavier models would not fix these interaction
and ownership bugs and were deferred without hardware/quality evidence. Confidence high
in targeted code correctness; live quality remains dependent on scene/prompt ambiguity.
