# Telekinesis polish consultation — 2026-10-04

Objective: improve the existing local webcam product, preserving its identity and architecture.
Writer and verifier: Codex desktop. Consultant: fresh read-only Codex CLI session.
Constraints: no webcam recording/uploads, no secrets, no unrelated rewrites; preserve Git history.
Relevant files: main.py, selection.py, segmentation.py, manipulation.py, scene.py, hand_tracker.py,
test_core.py, test_selection.py, test_pipeline.py, tools/, README.md and requirements.txt.
Inspection: all active source/tests/docs reviewed; clean main at a604556; remote HEAD matches.
Confirmed: cv2.getMouseWheelDelta is absent in Python OpenCV 4.13; wheel callback crashes.
Risks found: rank_candidates can have an empty score/stability intersection; cycling exposes
rejected masks; E does not reload a failed startup model; worker errors are not printed; stale
refinement resets the physical tracker to old coordinates; hidden originals can be selected
again; large translations can break shifted(); HUD instructions overflow at 640 pixels.
Plan: targeted fixes and regression tests, unified bounded checksum-verified model downloads,
readable HUD, bounded fixture demos, CI plus setup/troubleshooting/licensing docs.
Tests so far: dependency versions checked; isolated environment installation in progress.
Uncertainties: real-camera access and model network availability; candidate fallback policy;
refinement transform preservation and retry correctness.
Response wanted: critique with actionable risks, recommendation, alternatives, confidence.
Do not edit files or delegate. Review only. Do not read environment or credential files.
