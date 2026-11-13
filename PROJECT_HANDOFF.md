# Project handoff

Branch: `experiment/workstation-radar-baseline`

This branch contains the radar-scaling experiments and the completed ten-sweep full-data study initialized from the official RGB encoder.
Experiment settings are in [EXPERIMENT.md](docs/EXPERIMENT.md); source and asset
recovery details are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Branch-name adaptation — 2026-11-14

- Renamed `dd2414-mini-baseline` to `experiment/workstation-radar-baseline`
  to describe the full-data workstation experiments. Updated active documentation,
  deployment examples, launch guards and the explicit publication allowlist.
- The pre-change commit is preserved by
  `pre-workstation-branch-rename-20261114-c23dee8`. Original snapshot tag names,
  benchmark provenance and the recorded source boundary keep their historical names.
- Local checkouts can follow the migration commands in [BRANCHES.md](docs/BRANCHES.md).
  Existing run directories and pinned training sources are unchanged.
- Validation passed: four tooling regressions, documentation links, Python
  syntax/import boundaries, four shell syntax checks, and mocked publication,
  deployment and queue branch guards. All 38 pinned Git blobs match the recorded
  hashes and have no working-tree changes. No training or CUDA inference was run.

## Earlier changes — 2026-11-14

- Shortened repetitive setup/recovery explanations and rewrote the command tools
  with normal spacing, descriptive names and explicit control flow.
- The training CLI restores cwd, argv and import paths when its legacy driver
  returns or raises. The Markdown checker can now be imported without exiting.
- Added regressions for process-state restoration and link-checker imports.
  Numerical model code, checkpoint checks and data manifests are unchanged.

## Validation

```bash
python tools/testing/run_cpu.py
python tools/quality/check_source.py
python tools/quality/check_links.py
```

This review passed 86 CPU checks.
The new cases cover normal return, exceptions and SystemExit during dispatch,
plus import-safe link checking. Wheel build/install, imports, command help and
installed/source hash matching passed. Syntax, local links and diff checks pass.
Earlier extraction checks matched fixed-input losses, gradients and restored
state against the original source. Full-data CUDA inference and hosted CI were
not run for this review.

## Follow-up

The training entrypoints still combine setup, optimization, validation and
checkpoint logic in long functions. Splitting those phases is a useful later
refactor, with numerical and exact-resume checks around each change.

Keep experiment branches separate. Generated artifacts and local assistant/IDE
files remain ignored.
