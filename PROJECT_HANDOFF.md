# Project handoff

Branch: `experiment/rpl-depth-10sweep-velocity-study`

This branch contains the ten-sweep depth and velocity studies, including the 50-epoch experiment that trains ResNet101 from scratch.
Experiment settings are in [EXPERIMENT.md](docs/EXPERIMENT.md); source and asset
recovery details are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Verified experiment results — 2026-11-16

- Both random-student 50-epoch pretrains and six GT readouts completed, with
  6,019-sample final validation in every run. Final Vehicle IoU (DINO only /
  DINO + metric depth): camera probe **0.0076% / 4.87%**, fused probe **7.43% /
  11.95%**, five-epoch full segmentation **50.89% / 45.94%**. Depth improves the
  frozen probes but lowers full segmentation by 4.95 pp. These are single-seed
  validation results, not hidden-test scores or isolated proof of radar benefit.
- [Results report](docs/CLUSTER_DEPTH_RESULTS_20261116.md) consolidates setup,
  final/best epochs, confusion metrics, paired scene diagnostics, limitations
  and exact backup-reading reproduction command. [Results index](results/README.md)
  links it. Only reviewed English documentation is committed; generated JSON,
  figures, PDF and fetched mask panels remain outside tracked publication files.
- Actual checks: all eight final summary/config SHA-256 values match the
  verified backup manifest; six sets of 6,019 per-frame records reproduce final
  TP/FP/FN/TN and cover 150 scenes; final IoU/Recall and common initialization/
  budget checked. All eight summaries are complete; original trainer/controller/
  helper worktrees remain clean. No training/inference or model modification
  was performed. This is result-data verification, not a new full CUDA test.
  All four primary metrics were recomputed from final counts; documentation
  links and diff checks pass. The generated eight-page PDF was text-checked,
  and the result/scene figures visually reviewed.
- Jobs 1251176-1251183 and backup/cleanup 1251184 succeeded. Backup contains
  543 verified files; authorized eowyn dataset removal completed on 15 November.
  Original results/checkpoints remain in shared durable storage and local run
  output. The report records its exact path; this audit did not rehash every
  large checkpoint. Current account queue is empty.
- Next: inspect the camera probe's extreme underprediction and feature/logit
  scaling before changing architecture; investigate the depth arm's worse
  end-to-end recall. Add matched no-DINO controls and seeds before causal claims.

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

This review passed 196 CPU checks.
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
