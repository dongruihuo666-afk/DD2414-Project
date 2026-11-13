# Project handoff

Branch: `experiment/hybrid-teacher-10sweep-cluster`

This branch implements the original hybrid-teacher experiment and its RPL cluster setup.
Experiment settings are in [EXPERIMENT.md](docs/EXPERIMENT.md); source and asset
recovery details are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Latest changes — 2026-11-14

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

This review ran 57 CPU checks: 56 passed and one renderer check was skipped.
The new cases cover normal return, exceptions and SystemExit during dispatch,
plus import-safe link checking. Wheel build/install, imports, command help and
installed/source hash matching passed. Syntax, local links and diff checks pass.
Earlier extraction checks matched fixed-input losses, gradients and restored
state against the original source. Full-data CUDA inference and hosted CI were
not run for this review.

## Follow-up

The historical hybrid readout calls a mean-per-frame IoU an official score;
official Simple-BEV uses aggregate intersection/union. The experiment document
now explains the discrepancy. Preserve old results; correct the metric and
re-evaluate in a separately reviewed change.

Keep experiment branches separate. Generated artifacts and local assistant/IDE
files remain ignored.
