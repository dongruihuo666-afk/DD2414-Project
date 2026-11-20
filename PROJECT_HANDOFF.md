# Project handoff

Branch: `experiment/rpl-depth-10sweep-velocity-study`

This branch contains the ten-sweep depth and velocity studies, including the 50-epoch experiment that trains ResNet101 from scratch.
Experiment settings are in [EXPERIMENT.md](docs/EXPERIMENT.md); source and asset
recovery details are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Published direct checkpoint results — 2026-11-20

- Both requested untouched-checkpoint evaluations completed on 16 November;
  verified/publication update is 20 November. Raw Vehicle IoU is **1.9571%**
  for DINO only and **1.6342%** with metric-depth supervision. Each covers all
  6,019 validation samples / 150 scenes, with ZERO added training updates and
  no probe. Both decoders match the same original random initialization; before/
  after model and BN hashes are equal. These are not the separately GT-adapted
  50.89% / 45.94% scores. Frozen probes remain historical auxiliary records.
- Changed files: the [results report](docs/CLUSTER_DEPTH_RESULTS_20261116.md),
  [curated exact metric record](results/pretrained_direct_evaluation_20261116.json),
  this handoff and the results index. The report leads with the requested direct
  metrics and records extreme foreground overprediction and the untrained decoder
  limitation. Generated PDFs, per-frame data, logs and weights remain ignored;
  no other experiment branch or official path is changed.
- Actual validation: freshly rehashed both checkpoint files against archived
  SHA-256; checked both successful final summaries, 6,019 unique tokens and
  150 scenes; summed per-frame TP/FP/FN/TN and recomputed IoU/Precision/Recall/F1;
  verified model-state equality and common decoder identity. Published JSON,
  all eight reported percentages and evidence hashes match local summaries;
  documentation links and diff checks pass. No training or new inference was
  run for publication. Reproduce the inference with
  `tools/evaluate_pretrained_checkpoint.py` and the command in
  [EXPERIMENT.md](docs/EXPERIMENT.md), or inspect the published fractional metrics
  and evidence hashes without rerunning. Raw local files remain under
  `.cache/pretrained_direct_eval_20261116` in this worktree.
- Limitations/next: the archived segmentation decoder never received vehicle GT;
  same metric definitions do not imply matched supervised capability. Explain
  that boundary before designing further training. No new experiment is queued.

## Direct checkpoint evaluation protocol — 2026-11-16

- Owner requests Simple-BEV-aligned Vehicle IoU/Precision/Recall/F1 as primary
  reporting, without further frozen-probe training. Historical probes remain
  preserved; do not restart them or silently equate their results with direct
  inference. The new request evaluates BOTH untouched fifty-epoch pretraining
  checkpoints with no additional GT training, optimizer or fitted readout.
- `tools/evaluate_pretrained_checkpoint.py` uses the original immutable
  `a31f9039f9fe` model/data/metric code, verified archive hashes, strict tensor
  loading and full completion/source/config checks. The archived decoder must
  match its random initialization, and full-model parameter/BN hashes must
  remain unchanged after inference. This decoder received no segmentation GT
  during pretraining; any raw-checkpoint score must carry that qualification.
- [Experiment protocol](docs/EXPERIMENT.md) contains the exact CLI template.
  Runtime: all 6,019 validation frames, 448x800, ten sweeps with velocity,
  FP32/TF32 off, batch four, two loader workers, original valid-cell 0.5 decision.
  Different inference precision from the historical BF16 downstream runs is
  explicit; no threshold calibration or model modification is performed.
- Actual CPU validation: three new integrity/protocol tests passed; CLI help,
  compilation and source-boundary checks passed. One actual CUDA batch-four
  pilot passed in 1.83 s, peak tensor allocation 4.964 GiB; it reports NO subset
  IoU. Generated runtime/results stay ignored under
  `.cache/pretrained_direct_eval_20261116`. Both archived checkpoints were copied
  read-only to this workstation and their file hashes checked before loading.
- Full validation was run to completion in persistent USER service
  `dd2414-raw-pretrain-iou-20261116`, on the idle workstation GPU, sequentially
  for control and depth. `pipeline.json`, per-arm `status.json`, console logs
  and `summary.json` are under that ignored runtime directory. Both stages are
  complete with successful exits; the full results and actual checks are now
  published above. Do not duplicate the evaluator. A runtime pilot or partial
  validation is not a final result.

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
