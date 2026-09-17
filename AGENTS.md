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
- Treat project-scoped commits and pushes to the current feature branch as
  authorized by the project owner. After proportionate checks, commit and push
  each completed repository-changing task to that branch without requesting
  another conversational confirmation. Verify the remote branch received the
  commit. If the environment requires a tool approval, follow that boundary;
  if a push is blocked, report the local commit and blocker.
- Do not develop directly on `main`, force-push, or merge a PR as part of the
  per-task push routine. Merge a feature branch into `main` only after the
  bounded feature and its checks are complete and the team has reviewed it.
- Read-only questions do not authorize edits or GitHub writes. For those turns,
  use the existing handoff document and leave the repository unchanged.

The repository-local `Stop` hook reminds the agent about unfinished local
changes or commits that are ahead of the feature branch's upstream. It never
pushes on its own and does not replace reviewing the exact files being sent.
