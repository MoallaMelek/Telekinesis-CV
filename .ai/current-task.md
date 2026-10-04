2026-10-05 — Systemic selection failure and native-memory crash follow-up.
Writer/orchestrator and verifier: Codex desktop. Consultant: separate read-only Codex agent.
Previous precision changes are committed at 82dbae7; all hosted checks passed, but
human testing exposed false hand/body vetoes and a fragmented-object selection failure.
Follow-up removes semantic vetoes, fixes click-first selection/input precedence, adds a
frozen multi-point/box correction workflow, preserves corrected silhouettes, and owns
MediaPipe native buffers. Foreground handling respects selected-object pixel ownership.
Status: 107 tests passed; real-model correction demo, syntax and diff checks passed.
Native crash reproduced locally and no longer reproduced after the buffer-copy fix.
Read-only review approved after multi-positive cleanup and correction-preservation fixes.
Commit/push and hosted CI verification follow. See handoffs/universal-selection-review.md.
No webcam recording or upload. Original checkout and historical sources preserved.
