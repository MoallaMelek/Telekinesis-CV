2026-09-27: Implement Phase 3 only per latest brief. Reuse camera and hand code; remove crystal rendering from active main. Keep prior reusable physics dormant. Use CPU SlimSAM ONNX based on measured hardware and 4.5–5.4s image encoding with ~100–160ms cached decoding. Capture a hand-free scene containing target objects, not a background-removal plate. Delay static-mask confirmation until hand clears enough of target. Track reference/prompt versions and reject obsolete work. No video recording/upload and no GitHub push for this local experimental milestone.

2026-09-27 (Claude Code) — Redesign after failed usability test:
- Model: EdgeSAM-3x ONNX replaces SlimSAM (measured encoder 0.3-1.1 s vs 3.3-5.1 s; decoder 80-340 ms). S-Lab non-commercial licence (fine for this learning project).
- Selection from the LIVE frame (hand-free reference + manual recapture removed): prompt ahead of fingertip, hand landmarks as negative prompts, hand-overlap filter; cached encoding reused while fresh.
- Root causes fixed: pinch motion retargeted/cancelled prompts (preview now latched while inside outline and frozen once the pinch starts); score-only ranking chose parts (now largest steady plausible candidate; stability rejects merges; border-touching surfaces rejected); slow encode.
- Tracking: translation-only masked NCC template tracking; holds pose when occluded/lost; never re-discovers.
- Reconstruction: clean plate > scene memory > push-pull fill (+ FSR in background, B toggles).
- Held objects follow the palm centre; throw velocity from samples before release; 0.25 s dropout grace.
- Adaptive hand count (1 while pointing, 2 while holding): MediaPipe palm detection runs every frame when fewer hands than num_hands are visible.
- Removed interaction.py/physics.py (crystal era, preserved in git history f0ce5e8).

2026-10-04 (Codex desktop) — Reliability and product polish:
- Preserve EdgeSAM CPU inference, adaptive hand tracking, screen-space physics,
  translation-only physical tracking, and clean-plate/memory/inpainting reconstruction.
- Repair Python wheel input and rebase held transforms; hard-filter rejected outlines
  in both cycling and refinement; bound extracted objects consistently at six.
- Keep snapshot + prompt identity explicit. Esc/X cancel pending intent; E reloads failed
  models and clears stale selection. Model failures surface in the console and HUD.
- Unify four checksum-verified assets with atomic unique temporary files, size limits,
  socket/total timeouts and cancellation; add camera-free model preparation.
- Correct late-refinement translation and centroid changes without moving sprite pixels;
  merge two hand regions independently instead of dilating the first hand twice.
- Add compact width-aware HUD, active/hidden object feedback, synthetic preview, setup
  and troubleshooting guidance, model attribution, and explicit source-license status.
- Add offline and real-model CI. Pace asynchronous integration waits to wall time so
  slower hosted inference does not expire valid results.
- No browser/cloud deployment: this remains a local desktop application. No license
  assigned to owner-authored source; commercial model use remains an owner decision.

2026-10-04 (Codex desktop) — Human targeting/lighting follow-up:
- Show anatomical hand skeletons in ordinary use; guide point, pinch, move contextually.
- Keep previews steady; explicit M/right-click alternatives prevent unexpected changes.
- Own masks by the target component; constrain fragments, preserve real holes/thin handles,
  and reject weak masks instead of locking broad guesses.
- Enhance dim inference frames only; original camera/object appearance is unchanged.
- Reuse encoded features for a focused decoder pass; require agreement, bounded size,
  confidence and hand/body overlap, plus increased stability. Keep coarse alternatives.
- Retain model and architecture. Synthetic dark/noisy bottle overlap rose .887 to .958
  with lighting preparation; shadowed bottle .959 to .992 with focused decoding. These
  examples do not establish general-world accuracy.

2026-10-05 (Codex desktop) — General selection and native ownership correction:
- Human webcam testing invalidated the prior hard hand/body/quality veto strategy.
  Treat selfie foreground and anatomical hand cores as soft evidence; show uncertain
  masks for deliberate confirmation. No object-class special cases.
- Intentional clicks submit immediately; movement alone cannot steal a real hand's
  control. Re-click retries failed targets without needing to move outside a small object.
- S freezes raw pixels for box/include/exclude correction with cached encoding and stale
  prompt rejection. Enter pins the preview until confirmation; Backspace undoes edits.
- Preserve every included component and skip automatic silhouette refinement on corrected
  objects. Selected pixel ownership overrides false selfie foreground downstream.
- Copy MediaPipe's borrowed selfie buffers explicitly before native results are released.
  A local diagnostic reproduced the memory access violation before this fix.
- Preserve EdgeSAM CPU inference and avoid costly new models/cloud/class-based detectors.
  General prompting supplies missing intent; it cannot guarantee perfect segmentation.
