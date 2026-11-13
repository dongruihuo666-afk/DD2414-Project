# DD2414 integration handoff

## Current source and recovery

- Integration branch: `codex/team-integration-20261113`, isolated worktree
  `.cache/worktrees/team-integration-20261113`. Original workstation checkout,
  staged project PDF, environments and cluster execution roots were preserved.
- Workstation snapshot commit `32d9f35ef3f98bb84023d4e547996768da5baa71`,
  remote branch `codex/pre-integration-snapshot-20261113T144801Z`, annotated tag
  `pre-integration-refactor-20261113T144801Z`. The independent `-depth-study`
  tag preserves `0f91e222d658439d253dde4babf9a353056eecd3`. Both were verified
  remotely before any extraction. [Full backup record](docs/integration/BACKUP.md).
- Depth-branch model/DDP source and workstation-only official-study results were
  reconciled with a merge preserving both histories. No source conflicts were
  discarded. `main` was not merged or updated.

## What changed and why

1. Shared teacher/target, model wrapper, loss, evaluation, sampling, checkpoint
   and provenance implementations moved to `src/dd2414`. Old helper paths alias
   these modules; experiments retain their CLI/rendering duties. Formal modules
   and recommended trainers no longer import tiny/demo/smoke utilities or other
   trainers for generic functions. Original Simple-BEV source stays untouched.
2. Zero-copy named batch views and a caller-controlled feature API define the
   [team interface](docs/INTERFACES.md). The legacy loader's tuple/order remains.
   New package provenance extends the existing implementation-hash bucket;
   checkpoint schemas/state keys and strict resume checks are unchanged.
3. `pyproject.toml` installs explicit namespaces. `dd2414 train` uses the
   unchanged argument configuration and legacy driver; `dd2414 evaluate` reads
   saved logits/GT/valid arrays and computes official binary counts/IoU.
   Import/help has no training, download or cluster side effect.
4. Documentation/history, regression tests, experiment/visualization programs,
   portable deployment and explicit publication now have separate indexes.
   Historical fixed-source controllers are retained and labeled. Training
   queues record a publication command but never execute a Git publisher.
5. Repository-local handoff hook is advisory by default; explicit
   `DD2414_ENFORCE_HANDOFF=1` enables blocking for a task. Global user settings
   and unrelated staged files are untouched. Team rules remain in `AGENTS.md`.

## Reproduce and verify

```bash
python -m pip install --no-deps --no-build-isolation -e .
dd2414 train --help
dd2414 evaluate --help
python tools/quality/check_source.py
python tools/testing/run_cpu_regressions.py
```

Actual evidence: pre-extraction 103 CPU tests passed; the final expanded suite
passed 194 CPU tests. Ten fixed CPU input groups match the preserved source exactly, including gradients,
depth projection, voxel preparation, wrapper keys and post-resume optimizer
state. Actual official RGB checkpoint strict loads pass for the full model
(677 state items) and encoder (568 tensors). Editable and wheel
build/install/import/help succeeded in temporary environments. Commands,
checks, path mappings and actual limits are in [validation](docs/integration/VALIDATION.md).

## Limits and next action

This is engineering cleanup, not a new training method or a reported new IoU.
No full-data training, CUDA inference or remote Slurm action was used to verify
the cleanup. Existing source-pinned runs continue from their original checkout;
cross-refactor exact resume is deliberately rejected where source hashes differ.
Never bypass strict/hash/config checks or reset state to make resume appear
successful. Use [compatibility](docs/integration/COMPATIBILITY.md).

Current protocols remain [official full-data](FULL_DATA_TEST_PROTOCOL.md) and
[random metric-depth](CLUSTER_RANDOM_DEPTH_50EPOCH_PLAN.md). Committed results
and job ledgers are historical evidence, not a live cluster audit. The next
team step is review of this feature branch and an opt-in asset/data/CUDA
integration check in a separate run directory; no merge to main is authorized.

All prior detailed handoff entries are preserved in
[the indexed archive](docs/archive/pre-integration/PROJECT_HANDOFF.md).
