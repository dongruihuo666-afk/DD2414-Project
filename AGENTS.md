# DD2414 project instructions for coding agents

This repository contains a course project, not only the original Simple-BEV
implementation. Read `PROJECT_HANDOFF.md` before changing the model, data
pipeline, experiments, or project documentation.

## Working agreement

- Keep repository content, code comments, commit messages, and GitHub discussion
  in English. The human-facing conversation may use the user's language.
- Preserve the official Simple-BEV path and its checkpoint compatibility unless
  the task explicitly calls for changing it. Keep experimental paths opt-in.
- Distinguish supervised labels, frozen DINOv2 teacher features, radar-derived
  targets, and any trained student. Do not call a small-sample loss decrease a
  validation or generalization result.
- For every repository-changing task, update `PROJECT_HANDOFF.md` in the same
  commit with the changed files, rationale, reproduction command, actual test
  result, limitations, and next step. Revise stale status statements in other
  docs when necessary.
- Check `git status` before staging. Do not stage unrelated user/teammate files,
  datasets, checkpoints, cached DINO features, credentials, or logs.
- Run checks proportionate to the change, then commit and push the scoped work
  to the current collaboration branch when authorized and connectivity allows.
  If push is blocked, record the local commit and explain the blocker. Never
  merge a PR or force-push solely because of this instruction.
- Read-only questions do not authorize edits or GitHub writes. For those turns,
  use the existing handoff document and leave the repository unchanged.

The repository-local `Stop` hook is a reminder for unfinished local changes,
not a replacement for this review process or an automatic Git push.
