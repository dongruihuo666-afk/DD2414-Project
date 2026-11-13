# DD2414 branch handoff

## Scope

Branch `experiment/hybrid-teacher-10sweep-cluster` is an independent experiment. Independent historical full hybrid-teacher experiment, its own model/readout and RPL environment protocol.
`main` and all other experiments keep their own histories. No experiment merge,
new handoff branch, force push, data deletion or training interruption occurred.
The prior codex cleanup branches/worktrees were retired after verified backup.

## Changes and reproduction

- Extracted this branch's own shared model/teacher/loss/data/state helpers into
  `src/dd2414`; legacy aliases/CLIs retained. Named batch views and branch-local
  feature API do not change arithmetic or model state keys. Strict resume keeps
  optimizer/RNG/sampler/counters/source/hash checks; old source resumes via tag.
- Consolidated root records into focused `docs/` and this short handoff.
  Untracked 65 generated artifact files and 9
  old documents/reports from the active index, retaining their bytes locally and
  in source snapshot history. Added coherent ignore rules without artifact
  whitelists. Necessary qualitative tokens moved to configuration when used.
- Added explicit CPU tests/CI, install metadata, pure help, deployment planner
  and separate publication. No ordinary training Git writes or global changes.

```bash
python -m pip install --no-deps --no-build-isolation -e .
dd2414 downstream --help
python tools/testing/run_cpu.py
python tools/quality/check_source.py
```

## Evidence, limits and next step

Local-tooling follow-up (2026-11-14): removed the two tracked .codex hook files
and AGENTS.md from the index, retaining their local bytes. Added assistant/IDE
ignore rules; .github/workflows/cpu.yml remains shared project CI. Verification:
`git ls-files .codex AGENTS.md` is empty; `git check-ignore` matches the reviewed
local-tooling paths, local SHA-256 values are unchanged, documentation links
and diff checks pass. No runtime code changed, so training tests were not rerun.
Next: keep personal tooling local and review shared project configuration.

Pre/post extraction: 44 existing CPU tests pass. Fixed-input comparisons
against `33f2d86d70286c514cede6539f991825dde726ee` match exactly in 9 groups,
including gradients, calibrated inputs/voxels, wrapper state, sampler and
post-resume optimizer behavior.
Final CPU suite: 53 tests, with one inapplicable qualitative-renderer check skipped. All applicable checks passed.
Editable/wheel build, installation, independent import/help and actual package
source matching passed in this branch's temporary environment. The actual
official RGB checkpoint loaded strictly into the full model (677 state items)
and encoder (568 tensors). All 37 pinned model/data/manifest files are byte-identical.
Syntax, documentation links and Git whitespace checks pass. No CUDA/data
inference or hosted-CI pass is claimed. Optional old report generators may
require ignored original artifacts, retrievable by tag. Next: review this
experiment independently; no new full training is required by cleanup.

Recovery: `repository-snapshot-20261113T185518Z/experiment/hybrid-teacher-10sweep-cluster`. Full local bundle/inventory under
`.cache/repository-snapshots/20261113T185518Z/` on the workstation. See
[branches](docs/BRANCHES.md), [interfaces](docs/INTERFACES.md),
[development](docs/DEVELOPMENT.md), [experiment](docs/EXPERIMENT.md),
[deployment](docs/DEPLOYMENT.md) and [recovery/artifact policy](docs/REPRODUCIBILITY.md).
