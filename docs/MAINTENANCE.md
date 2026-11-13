# Maintenance and collaboration

Read `PROJECT_HANDOFF.md`, `AGENTS.md` and the current protocol before changing
a model/data/experiment path. Work on an isolated feature branch/worktree when
another run uses a fixed checkout. Review local staged/untracked files before
fetch/merge or publication; do not overwrite local work from a remote branch.

The package boundary owns shared implementations. Historical scripts can
re-export/alias them for compatibility, but new production code must not import
helpers from a tiny/demo/smoke program or a training entrypoint. Use named
batch fields at integration boundaries; keep the legacy loader and model state
names stable unless a separately reviewed research change requires otherwise.

The source audit in `tools/quality/check_source.py` parses Python without heavy
imports and enforces the formal-package dependency boundary. The explicit CPU
suite is described in [tests](../tests/README.md); data/CUDA experiments are
opt-in. Before changing a mathematical helper, run fixed-input original/new
comparisons for outputs/gradients and applicable checkpoint state. A shape-only
test is not enough for geometry, velocity or resume semantics.

Manifests define sample identities/order and have recorded hashes; do not
regenerate them as a side effect of installation, tests or model config changes.
Current full-data learning uses official 700 scenes / 28,130 frames; validation
uses official 150 scenes / 6,019 frames. Public test annotations are absent;
these results are validation, not a newly untouched blind test. Frozen official
checkpoint provenance is supervised and distinct from teacher pseudo-targets.
No downstream GT labels should enter representation pretraining objectives.

New checkpoint producers record actual package source in the existing
implementation digest field. Old source exact resume remains old source exact
resume, including optimizer, scaler, per-rank sampler, RNG and counters. Do not
translate hashes, downgrade strict loading or quietly reset state. A new phase
may intentionally initialize model weights under its explicit existing protocol;
that is not exact resume. [Compatibility](integration/COMPATIBILITY.md).

Record scoped changes, reproduction, actual tests, limits and next action in
the short handoff. Place detailed dated work logs in the research archive.
Routine feature-branch commits/pushes follow owner authorization; no main merge
or force-push is implied. Ordinary training cannot publish Git state. Explicit
publication and deployment procedures are in [deployment](DEPLOYMENT.md).

The repository's `.codex` hook is read-only/advisory by default and never stages
or pushes. Optional enforcement is task-local via `DD2414_ENFORCE_HANDOFF=1`.
Unrelated teammate changes and read-only user questions are not authorization
to commit. This cleanup does not alter global agent, SSH, mamba or system config.
