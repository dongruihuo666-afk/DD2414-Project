# DD2414 project technical handoff

This is the short, continuously updated entry point for teammates and coding
agents. `README.md` introduces the repository; `PROJECT_PROGRESS.md` contains
the broader experiment report; this file records the current implementation,
reproduction method, evidence, limitations, and handoff decisions.

## Current state

### 2026-11-14 — Independent branch cleanup: shared code stage

- This branch remains the independent experiment `experiment/rpl-depth-10sweep-velocity-study`. No merges,
  cherry-picks from other experiments, branch renames, or main updates occurred.
  Each extracted function comes from THIS branch's preserved `0f91e222d658439d253dde4babf9a353056eecd3`.
- The rejected standalone handoff/snapshot branches were retired after verified
  remote tags and an all-refs bundle. All original experiments remain separate.
  Recovery tag: `repository-snapshot-20261113T185518Z/experiment/rpl-depth-10sweep-velocity-study`.
- Shared teacher/target/model/loss/state helpers now live in `src/dd2414`;
  old helper modules remain aliases and experiment programs keep their CLIs.
  Named batch views do not copy, reorder or convert inputs. Installation uses
  explicit namespaces; new CLI help is stdlib-only. Model/manifest/parameter
  semantics remain branch-local. New config records bind actual package bytes;
  strict historical resume requires the original source, with no state reset.
- Reproduction: `PYTHONPATH=src:.:scripts:cluster_rpl:cluster_experiment
  CUDA_VISIBLE_DEVICES='' python tools/testing/run_cpu.py` (runner added in the
  documentation stage); parity fixture: `tests/parity/probe.py --implementation
  legacy|package --source-root /path/to/the/corresponding/source`.
- Actual baseline before/after extraction: 183 CPU tests passed on each
  side, including this branch's existing regression assertions. Gloo/worker IPC
  checks need local socket/shared-memory permissions. Fixed-input parity now matches exactly in 10 groups, including gradients,
  calibrated voxel inputs and optimizer resume; portable docs are next. No data/CUDA training or cluster mutation.


### 2026-11-12 — Repaired continuations are training; downstream graph replaced

- User explicitly authorized repair and resumption. Helper source is pinned at
  `4a4fbdca3459`; actual CPU admission **1251175 COMPLETED** in 11 seconds.
  BOTH patched `--check-only` commands succeeded on eowyn, identifying the exact
  hongen-owned original parents and their `FAILED|75:0` exits. Both real saved
  checkpoints passed model/optimizer finiteness, config/implementation hashes,
  fifty-epoch full-data budget and FIVE rank-specific sampler-state checks.
  Source cursors are control **199546 / 35.4685 epochs**, depth **165092 /
  29.3445 epochs**. The admission records file/config/all-resume-state SHA-256;
  no checkpoint state was rewritten, reset or migrated by this repair.
- Actual replacement continuations **1251176 (DINO-only)** and **1251180
  (DINO + metric depth)** are RUNNING on eowyn, each with five GPUs, twenty CPUs,
  128-GiB host RAM and one 47-hour allocation. These replace launch-failed jobs
  1250704/1250705; they do not add another fifty epochs. The original admitted
  controller `aec677657cd0` and model/trainer `a31f9039f9fe` remain unchanged.
  Latest recorded verification reached **199783 / 165290 updates** (35.5107 /
  29.3797 epochs), with finite losses/gradients and contiguous counters. BOTH
  native periodic checkpoints have been freshly written after resumption.
  Sampled host RAM is 81.47 / 77.70 GiB against 128 GiB, with zero OOM kills.
- Exactly SEVEN old own PENDING jobs were owner/name/reason-checked and
  cancelled: 1250696-1250698 / 1250700-1250703. Their failed dependencies could
  never run. Fresh 10-epoch camera/fused probes and 5-epoch full segmentation
  are **1251177/1251178/1251179** after successful 1251176, and
  **1251181/1251182/1251183** after successful 1251180. Guarded final verified
  backup/cleanup **1251184** requires all SIX GT readouts. No other user's job,
  workstation run, arwen storage or official Simple-BEV path was changed.
- Reproduction and exact submission argv are recorded in the shared
  `.cache/continuation_repair_20261112.json` ledger and
  [actual repair evidence](artifacts/rpl_continuation_repair_20261112.json).
  Original submission ledger now points to this active repair record. Preserve
  the immutable helper/controller/trainer roots and same settings file
  `.cache/random_depth_workers0_resume_20261109.json`; do not duplicate jobs.
- Tests: **11 continuation CPU regression tests passed**, compilation/diff
  checks passed, actual patched node accounting and full checkpoint admission
  passed, and real five-GPU optimizer updates are observed for BOTH groups.
  Preliminary CPU audit 1251174 was replaced because its checker incorrectly
  compared rank-specific sampler identities; the corrected audit compares
  shared cursor/config fields and verifies each rank separately. This was an
  audit-script correction, not a checkpoint or model change; history is kept.
- Limitations/next: neither study has finished 50 epochs, and no new final IoU
  exists yet. Continue the remaining original budget, verify periodic saves and
  full final validation, then collect all six GT results before backup/cleanup.
  Readouts still require SUCCESSFUL full pretraining; timeouts/errors/partial
  results cannot release them or delete data. Library paths are scoped to the
  accounting subprocess only; nothing was installed or reconfigured globally.

### 2026-11-12 — Repair continuation accounting without changing training

- The owner authorized correcting the launcher and continuing both studies.
  Parent pretrains 1250695 / 1250699 gracefully saved and exited `FAILED|75:0`
  at the 47-hour boundary on 11 November. Their summary/checkpoint files retain
  control update **199546** (35.4685 epochs) and depth **165092** (29.3445 epochs),
  with 35 / 29 completed 6,019-sample validations. Continuations 1250704 /
  1250705 failed BEFORE launching the trainer: eowyn's `sacct` loader could not
  resolve `libslurmdb.so.32`. Six GT readouts are `DependencyNeverSatisfied`;
  no final IoU or completed 50-epoch result exists yet.
- Actual CPU-only eowyn checks confirmed that the library already exists in
  `/usr/local/lib`. Adding `/usr/local/lib:/usr/local/lib/slurm` to only the
  `sacct` child process resolves both parent states successfully. No package,
  system configuration, shell startup file or cluster service was changed.
- `cluster_rpl/continue_recovered_depth.py` now pins `/usr/local/bin/sacct`,
  scopes library paths to that query subprocess, preserves existing paths and
  bounds each query by ten seconds. It verifies the exact numeric parent job
  AND current Unix owner, refuses ambiguous rows, and retains the original
  partial-exit-75/complete-full-validation policy. No failure is silently
  converted into a resumable training state. `--check-only` exercises real
  accounting plus the committed protocol on a CPU allocation without execing
  the trainer. The admitted controller `aec677657cd0`, trainer `a31f9039f9fe`,
  checkpoints, optimizer/LR, RNG/BN/sampler and scientific config stay unchanged.
- Tests: `PYTHONPATH=cluster_rpl CUDA_VISIBLE_DEVICES=''
  /home/students2026/miniconda3/envs/bev/bin/python -m unittest
  test_continue_recovered_depth`: **11 tests passed**, including subprocess-only
  path isolation, exact owner/job query, bounded accounting wait and all previous
  OOM/cancellation/budget/final-validation refusals. Compilation and diff checks
  pass. The actual diagnostic proves the library-path remedy; the pinned patched
  CLI and saved checkpoints still require admission before GPU launch.
- Next: push/pin the repaired helper, run BOTH `--check-only` commands and CPU
  checkpoint integrity checks on eowyn, then submit replacement five-GPU
  continuations from the saved cursors. Replace only the seven verified own
  blocked readout/cleanup jobs and record their new IDs. Keep the same 50 TOTAL
  epochs, 448x800 inputs, ten sweeps with velocity, two treatments and 10/10/5
  downstream budgets. Preserve all failure evidence and workstation work.

### 2026-11-09 — Actual recovery and automatic continuation graph verified

- [Actual recovery evidence](artifacts/rpl_worker_free_recovery_20261109.json)
  records checkpoint/state hashes, full-data CPU audit, both real CUDA/RAM
  admissions, exact submission argv, measured telemetry and current queue.
  Latest verified control/depth status was **3850 / 4250 updates**, approximately
  **0.684 / 0.755 of the first epoch**; both are RUNNING, not completed results.
- Bounded continuations **1250704 / 1250705** were actually submitted from
  immutable helper source `012b653fe8b5`. They depend on `afterany` current
  pretrains 1250695 / 1250699. The helper permits only completed/full-validation
  no-op or a committed `FAILED|75:0` partial resume, and delegates to unchanged
  admitted controller `aec677657cd0` / trainer `a31f9039f9fe`. Both immutable
  Git revisions and clean worktrees were independently checked after submission.
- All SIX own pending readout dependencies were actually changed and checked:
  1250696/1250697/1250698 now require successful 1250704; 1250700/1250701/1250702
  require successful 1250705. Final backup/cleanup 1250703 still requires ALL
  six readouts. Current queue is TWO running pretrains plus NINE dependent jobs
  (two continuations, six GT evaluations, one guarded backup/cleanup). It is
  not eleven concurrent training jobs. No other user's job was changed.
- Tests: **40 CPU tests passed**, including real multi-process Gloo checkpoint
  resume and worker equivalence; six added continuation checks cover exit 75,
  cancellation/OOM refusal, full final-validation coverage and unchanged epoch
  budget. Compilation, JSON parsing, exact queue dependency assertions and
  diff checks pass. The first sandbox-only run's restricted sockets/shared
  memory were resolved by running the SAME CPU command outside the sandbox.
- Reproduction/settings and immutable execution paths are recorded in the
  evidence artifact and preceding handoff entry. No new model or source edit
  is needed to let the queue continue; do not manually duplicate these jobs.
  Next: monitor epoch validation/host RAM and collect the two frozen probes
  plus full segmentation metrics per group. A second partial allocation or
  other failure blocks evaluations for inspection; no infinite retry, extra
  epoch or early data deletion is enabled. The existing 47-hour segments give
  at most 94 allocation hours/group, not a guarantee of finishing in that time.

### 2026-11-09 — Both worker-free groups admitted and running; bounded continuation prepared

- Actual 128-GiB recovery admission **1250694 COMPLETED** in 7m06s. Each
  treatment made 100 updates and resumed for one more update in a fresh process.
  Control retains the preceding trial's committed progress (2351 -> 2452);
  depth advances 3000 -> 3101. All learned/optimizer/RNG/BN/sampler state is
  preserved, with only worker count and fresh output path changed. Measured
  admission host peaks were 78.066 / 83.547 GiB, below the actual 128-GiB limits.
- Formal control **1250695** and metric-depth **1250699** are RUNNING on eowyn,
  each with five GPUs, twenty CPUs and 128 GiB host RAM. The two experiments
  still use random ResNet101, RGB 448x800, ten sweeps with velocity, global
  batch five and FIFTY TOTAL epochs including inherited updates. The trainer
  stays at immutable `a31f9039f9fe`; admitted controller is `aec677657cd0`.
  No scientific/model change or workstation experiment change was made.
- Six downstream jobs **1250696-1250698 / 1250700-1250702** are waiting for
  completed pretraining, using one GPU/four CPUs/32-GiB host RAM each. Final
  backup/cleanup **1250703** waits for all six full-GT readouts. Only seven own
  stale failed-parent jobs 1250638-1250640 / 1250642-1250645 were cancelled.
  The completed arwen retirement and all previous evidence remain preserved.
- A real 30-second simultaneous training window advanced control 2559 -> 2599
  and depth 3189 -> 3222, approximately 6.67 / 5.50 samples/s per FIVE-GPU group.
  Anonymous RAM was 67.49 / 67.53 GiB; total cgroup RAM 78.41 / 74.64 GiB also
  included 10.60 / 6.83 GiB file cache. Control GPU activity averaged 50.27-70.33%
  per card in this window; no corresponding depth activity samples were
  captured in that window. These are short measurements, not a full-run ETA,
  FLOP efficiency, completed epoch, segmentation quality or final IoU claim.
- `cluster_rpl/continue_recovered_depth.py` prepares ONE additional 47-hour
  allocation per group. It only resumes a parent with committed `partial`
  summary AND scheduler `FAILED|75:0`; OOM, cancellation and arbitrary errors
  fail instead. A parent already complete at 50 epochs plus full 6,019-frame
  validation makes continuation a no-op. The helper delegates to the exact
  admitted controller, so no checkpoint/source hash is disguised. Readouts
  will depend on successful continuation; at most 94 allocation hours/group,
  with no repeated-error retry or added epochs. This paragraph describes the
  reviewed helper; the actual continuation submission is recorded separately.
- Reproduction: the existing shared settings are
  `.cache/random_depth_workers0_resume_20261109.json`; run the helper only in
  the matching eowyn allocation with `--config SETTINGS --treatment dino-only`
  (or `dino-metric-depth-bev`) `--previous-job PARENT_ID`. CPU checks use
  `PYTHONPATH=scripts:cluster_rpl CUDA_VISIBLE_DEVICES=''` and the local bev
  Python with `-m unittest test_continue_recovered_depth
  test_recovery_checkpoint test_distributed_training test_random_depth_study
  test_random_depth_migration test_training_checkpoint test_metric_memory`.
  **40 CPU tests passed** (including actual two-process Gloo resume and worker
  equivalence), compilation and diff checks passed. The first sandbox run
  could not open Gloo/shared-memory resources; the identical CPU command passed
  outside that restriction. Actual continuation IDs follow queue submission.

### 2026-11-09 — Measured worker-free RAM floor; retry admission with 128-GiB jobs

- Actual worker-free preparation/full CPU audit **1250692 COMPLETED** in 91 s:
  both real checkpoints were copied with non-config resume-state hash equality,
  unchanged trainer/model digests and the complete original dataset admission.
  New root is
  `/local_storage/users/hongen/dd2414_runs/random_resnet101_metric_bev_50epochs_workers0_resume_20261109`;
  settings are shared `.cache/random_depth_workers0_resume_20261109.json`.
- Actual five-GPU test **1250693** successfully ran control updates 2251–2350
  and a fresh-process update 2351. It then deliberately FAILED the headroom
  gate (`Insufficient verified host-memory headroom`), rather than releasing
  formal training. Sampled host RAM was already 73.7–75.6 GiB against an 80-GiB
  limit; reducing workers alone does not make five full SDK replicas small.
  No new CUDA/model error is established by this guarded failure. The new
  2351 checkpoint/logs are preserved; depth remains at inherited update 3000.
- Recovery admission now reads each actual saved starting cursor, retains ALL
  already committed pilot progress, checks contiguous logs from the original
  migrated cursor and runs another 100+1 fresh-process segment. The gate records
  original parent AND actual pilot-start counters, validates both real partial
  exits and actual RAM headroom. It never rewinds to discard the first trial.
  This changes the recovery controller only; the original immutable trainer,
  model, data/teacher hashes and scientific settings remain unchanged.
- Next allocations request **128 GiB host RAM per five-GPU group**, with the
  same twenty CPUs, local/global batch 1/5, worker count zero and dataset/budget.
  This is a Slurm job resource request, not a system/cluster configuration
  change. The current 80-GiB trial has ended; no training jobs are silently
  duplicated. Both treatments must pass the same new admission before formal
  work. Existing CPU data audit remains valid for the unchanged old trainer and
  exact public runtime settings; the new controller gets its own pinned source.
- Tests: the previous recovery/distributed/protocol/checkpoint/projection
  command now passes **34 CPU tests**, including a real serialization roundtrip,
  deterministic 0/2-worker samples, progress-preserving admission retry, and
  refusal of inadequate measured RAM headroom. This is not a 128-GiB CUDA
  admission claim. Retired arwen paths and old failed checkpoints remain intact
  in their verified backups; only our eowyn continuation is being repaired.

### 2026-11-09 — Prepare exact-state worker-free recovery of both experiments

- Owner directs attention to our experiments rather than investigating the
  cluster outage. The active repair targets our confirmed 80-GiB host-memory
  failures only. `cluster_rpl/recovery_checkpoint.py` prepares FRESH checkpoint
  copies with exactly TWO config changes: `num_workers: 2 -> 0` and the new
  output directory. Old failed outputs/checkpoints stay untouched. It validates
  all original model/data/training implementation digests, schema, fifty-epoch
  full-data budget and five-rank cursors, then hashes ALL non-config state before
  and after real serialization: model, optimizer, scalers, progress, sampler,
  per-rank RNG and BN. An external `runtime_migration.json` records old/new file
  and config hashes plus the allowed changes; config hashes are recomputed for
  those honest runtime settings, never rewritten to hide model/source changes.
- `cluster_rpl/recover_random_depth.py` controls preparation, real recovery
  admission, pretraining and GT readouts. The TRAINER remains the original
  clean immutable `a31f9039f9fe` snapshot, so no model/optimizer/loss code is
  changed. Both groups use ZERO additional DataLoader processes per rank,
  preventing the previous ten extra worker processes per five-GPU experiment.
  This tests the metadata-replication hypothesis; long-run host-RAM stability
  still requires measurement. Seeded per-sample CPU augmentation is unchanged.
- Admission starts at saved updates 2250/3000, runs another 100 updates, then
  resumes in a fresh process for update 2351/3101. It requires finite learned
  state/gradients, exact contiguous new logs, preserved five-rank progress and
  measured host peak below 90% of the actual Slurm cgroup limit. The controller
  samples its OWN cgroup RAM and allocated GPU activity/VRAM every two seconds;
  a steady training window yields samples/s. Both treatment reports must pass
  the new recovery gate. Original fresh-initialization update-three admission
  reports are NOT fabricated or reused for this different continuation test.
- The recovery uses a new root, preserving uncheckpointed old JSONL/TensorBoard
  tails. No old logged-but-unsaved update is counted as resumed progress.
  Scientific settings stay random student, RGB 448x800, ten sweeps with velocity,
  the same two DINO objectives/optional metric depth, global batch five,
  optimizer/LR state and FIFTY TOTAL epochs including completed sample visits.
  Frozen camera/fused probes and supervised segmentation retain 10/10/5 epochs
  and full GT validation. No batch-size increase is bundled into this repair.
- Tests: `PYTHONPATH=scripts:cluster_rpl CUDA_VISIBLE_DEVICES=''
  MPLCONFIGDIR=/tmp/dd2414-recovery-tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
  /home/students2026/miniconda3/envs/bev/bin/python -m unittest
  test_recovery_checkpoint test_distributed_training test_random_depth_study
  test_random_depth_migration test_training_checkpoint test_metric_memory`:
  **33 CPU tests passed**, including actual checkpoint roundtrip, corruption/
  state/hash rejection, old-source command mapping, both-treatment gate and
  real zero-versus-two-worker deterministic sample equivalence. Compilation,
  CLI help and diff checks pass. This is not actual CUDA/RAM admission yet.
- Next: push/pin the new CONTROLLER, prepare and audit the new eowyn outputs
  with zero GPUs, run both real five-GPU recovery segments under RAM telemetry,
  then replace only the seven failed-dependency own jobs with admitted formal
  continuations, GT readouts and guarded final cleanup. Keep arwen retired and
  the original trainer snapshot untouched. No system setting, external user's
  job or global cluster diagnostic is part of this task.

### 2026-11-09 — Access restored, complete arwen retirement verified, host-RAM failures identified

- User reported the cluster repaired; hongen login and commands now work.
  The clean visible feature clone was fast-forwarded to `f26ae51`; execution of
  the reviewed retirement helper was pinned to NEW immutable worktree
  `.cache/deployments/f26ae519bfa9` under the shared project. Training source
  `.cache/deployments/a31f9039f9fe` was NOT modified. Main workstation training
  and the unrelated staged PDF remain untouched.
- The previously authorized COMPLETE arwen retirement has ACTUALLY finished.
  CPU-only inspection **1250688 COMPLETED** found 180 owner-root entries,
  3,911,261,426 logical file bytes (~3.64 GiB), and seven own old TorchElastic
  directories under `/tmp`. No own arwen job was running or queued. Home used
  14,661 MiB of its 100-GiB quota before backup. The remaining owner root was
  old assets, runs and the empty dataset parent; the full dataset was already
  absent, as expected from the earlier successful source retirement.
- CPU-only execution **1250689 COMPLETED** in 44 seconds. It used the tested
  `retire()` implementation to back up, hash-verify and delete the exact
  `/local_storage/users/hongen` on ARWEN plus the seven explicitly inventoried,
  own `/tmp/torchelastic_*` paths. Guards also checked own process working
  directories/open FDs and own Slurm jobs before each deletion. No outside
  symlink target, other user's data, shared Home, or eowyn path was deleted.
  Backup root:
  `/Midgard/home/hongen/projects/DD2414-Project/fullsize_baseline/runs/arwen_retirement/job-1250689`.
  Original owner-root files are under `owner-root/files`, with the SHA-256 entry
  manifest beside them; temporary directories are archived under `temporary`.
- Independent CPU-only verification **1250690 COMPLETED**: all EIGHT source
  paths are absent and every archived file/link/directory entry matches its
  saved manifest after a fresh SHA-256 pass. Common locations `/local_storage`,
  `/local_storage/users`, `/tmp`, `/var/tmp`, `/dev/shm` had no remaining top-level
  entries owned by hongen in the execution report. This verifies known personal
  storage and common temporary locations, not an unrestricted whole-system scan.
  The root deletion's observed local free-space increase was 3,911,823,360 bytes,
  plus seven 20,480-byte temporary-directory deltas. Shared durable backups are
  retained, NOT Git-tracked; no new dataset migration was performed.
- Training diagnosis is now CURRENT: **1250637 and 1250641 are OUT_OF_MEMORY**,
  after 22m47s and 35m21s respectively on 7 November. Both stderr files contain
  Slurm `oom-kill` events in the batch cgroup; this is HOST memory exhaustion
  against each job's 80-GiB RAM request, not evidence of a new CUDA allocator
  OOM. Slurm recorded MaxRSS 81,509,140 / 81,456,884 KiB. GPU allocation peaks
  in the saved telemetry remained about 4.86 / 4.99 GiB. Control stderr also
  reports failed name resolution for `slurmctl1.rpl`; whether the separate
  login/controller outage was caused by our jobs remains UNRESOLVED.
- CPU-only eowyn checkpoint audit **1250691 COMPLETED**: dataset metadata remains
  present; both latest checkpoints have finite model/optimizer tensors and all
  five acknowledged rank cursors. Control can resume at update **2250**, 11,250
  samples (~0.400 epoch); depth treatment at **3000**, 15,000 samples (~0.533
  epoch). Last LOGGED updates 2425/3075 are newer than the saved checkpoints and
  must not be reported as durable resume positions. Neither has finished one
  epoch/full validation. Their old `status.json` still says `running` because
  rank zero was killed; it is STALE and scheduler accounting takes precedence.
- No GPU training job was restarted. Six readouts 1250638–1250640 /
  1250642–1250644 now show `DependencyNeverSatisfied`; final eowyn backup/cleanup
  1250645 remains pending on those readouts, and cannot delete the training
  dataset. No real-time training GPU-utilization benchmark is meaningful while
  the jobs are stopped. The prior logged five-GPU training-only throughput was
  ~9.94 / 7.65 samples/s; this is historical cumulative telemetry, not a new
  steady-window utilization measurement or proof that a larger batch is faster.
- Reproduction/evidence: `artifacts/rpl_recovery_arwen_retirement_20261109.json`
  contains preflight, actual deletion argv, resource limits, complete retirement
  and independent verification reports, checkpoint counters, memory/error
  accounting, quota and final queue snapshot. Ten `test_retire_arwen` temporary-
  filesystem tests passed again before execution. All four recovery jobs used
  ZERO GPUs and one CPU each; arwen jobs used 2 GiB RAM, checkpoint inspection
  4 GiB. They completed and released their allocations.
- Next: investigate total HOST/cgroup memory and per-rank DataLoader workers
  before any larger-batch or long GPU restart. Multiprocess nuScenes metadata
  replication is a candidate, not yet an established root cause. Preserve the
  two intact checkpoints and original source/config; changing worker settings
  requires an explicitly recorded, tested restart compatibility path because
  strict checkpoint config hashes include them. Then repair the failed Slurm
  dependencies and reconcile/archive uncheckpointed JSONL/TensorBoard tails
  before resuming, to avoid duplicate rows after the saved update boundary.
  Measure sustained RAM/VRAM/throughput during resumed work.
  Administrator outage cause/new resource rules have been requested but are
  not yet provided. Full-data architecture, fifty-epoch budget and objectives
  remain unchanged. The previous blocked-retirement entries below are history.

### 2026-11-07 — Full arwen retirement authorized; prepared, execution blocked by SSH

- Owner clarified that the earlier dataset-only retirement is insufficient:
  future training will not use arwen. Authorization now covers ALL own files
  under arwen's `/local_storage/users/hongen`, including old assets, caches,
  results, checkpoints and empty directories. Shared Home/project/environment
  and the IDENTICALLY NAMED path on eowyn are separate and must be preserved.
  Valuable remaining files will first be backed up off arwen; this removes the
  node-local copies while retaining the previously requested checkpoint history.
- `cluster_rpl/retire_arwen.py` is a CPU-only retirement entry point pinned to
  account hongen, physical hostname arwen, one Slurm node and the exact owner
  root; no arbitrary deletion-path option is exposed. It inventories ownership,
  mounts, files/directories and symlinks without following links. Any other own
  queued/running arwen job blocks it. Default mode only inspects. `--execute`
  copies the ENTIRE remainder to the shared project's already-ignored
  `fullsize_baseline/runs/arwen_retirement/job-JOBID/files`, checks every file's
  SHA-256/link/entry list against the original, rechecks source stability and
  idle state, then removes the owner root and verifies it is absent. Backup
  manifest/retirement evidence stay on shared storage. A surprising remaining
  full dataset blocks copying it into the 100-GiB Home quota. Source is not
  deleted if backup/copy/quota/verification fails. A deletion-time filesystem
  error can leave partial removal and must be reported as incomplete.
- Actual tests: `PYTHONPATH=cluster_rpl
  /home/students2026/miniconda3/envs/bev/bin/python -m unittest
  test_retire_arwen` passed TEN temporary-filesystem checks: actual verified
  backup/deletion, outside symlink-target preservation, inspection-only mode,
  refusal to overwrite backups, corruption and source-change failures, active
  job guards, owner/symlink/special-file rejection, unexpected dataset and wrong
  node handling. This is NOT evidence of actual arwen deletion.
- Actual connection probe used the hongen SSH config/key and bounded timeout;
  it again ended `Timeout, server moria.csc.kth.se not responding`. Earlier
  debug reached public-key authentication but stalled opening a remote session;
  owner also reports their existing terminal is stuck. Therefore NO new arwen
  retirement job has been submitted and the remaining ~2.8-GiB runs/~960-MiB
  assets have NOT been freshly inventoried or removed. Those sizes and the
  earlier dataset deletion are historical verified observations. Current DDP
  progress/utilization cannot be reverified during this access outage.
- Once access recovers, fast-forward the visible feature clone, verify its
  clean branch/source and inspect own arwen jobs/quota. Submit the following
  one-CPU, zero-GPU job FROM MORIA (normal partition/time restrictions apply):

  ```bash
  cd /Midgard/home/hongen/projects/DD2414-Project
  sbatch --partition=pdebug --job-name=dd2414-retire-arwen-owner \
    --nodes=1 --ntasks=1 --nodelist=arwen --cpus-per-task=1 --mem=2G \
    --time=00:30:00 --chdir=/Midgard/home/hongen/projects/DD2414-Project \
    --output=/Midgard/home/hongen/projects/DD2414-Project/.cache/retire-arwen-%j.out \
    --error=/Midgard/home/hongen/projects/DD2414-Project/.cache/retire-arwen-%j.err \
    --wrap='/Midgard/home/hongen/miniforge3/envs/dd2414/bin/python /Midgard/home/hongen/projects/DD2414-Project/cluster_rpl/retire_arwen.py --execute'
  ```

- Next: execute that authorized retirement only after live scope/idle/quota
  checks, retain the resulting job/backup/hash evidence, verify owner-root
  absence, and inspect any project remnants outside this known root rather
  than claiming the whole node was searched already. Do not repeat dataset
  migration or affect eowyn's two training jobs/queued evaluations. The running
  training snapshot stays at `a31f9039f9fe`; this helper/docs addition does not
  modify its model, objective, checkpoint hashes or experiment budget.

### 2026-11-07 — Repaired five-GPU DDP admitted; both formal jobs now updating

- Memory repair commit `a31f9039f9fe3e265a503a1ffdedc03b71e361a0` was pushed,
  remote ref verified, and the clean hongen feature clone fast-forwarded. Actual
  jobs use NEW immutable snapshot `.cache/deployments/a31f9039f9fe` beneath
  `/Midgard/home/hongen/projects/DD2414-Project`; later evidence/docs commits
  must not change this pinned execution source. The initial sync helper hit
  old Git's unsupported `branch --show-current` before any mutation; using
  `rev-parse --abbrev-ref HEAD` fixed that orchestration compatibility check.
- Exactly the nine verified PENDING own jobs 1250545–1250553 were cancelled
  and replaced. Old dataset migration/audit/source cleanup were NOT rerun.
  Existing eowyn data, the failed pilot/source and all old checkpoints remain.
  New settings are `.cache/random_depth_ddp_streamed_20261107.json`; fresh run
  root is `/local_storage/users/hongen/dd2414_runs/random_resnet101_metric_bev_50epochs_ddp_streamed_20261107`;
  durable backup has the same name under the shared clone's
  `fullsize_baseline/runs/`. No old checkpoint is resumed with changed hashes.
- NEW CPU audit **1250635 COMPLETED**: all 2,631,083 referenced sensor files
  are present/nonempty/safe, exact official 28,130/6,019 train/val membership
  and zero scene/sample/log overlap, matching model hashes and mamba packages.
- NEW real-model pilot **1250636 COMPLETED**: BOTH treatments ran optimizer
  updates 1/2, saved, exited partial intentionally, and completed update 3 in
  fresh torchrun processes. Full five-rank optimizer/RNG/sampler state passed;
  517 floating state tensors changed in each resumed update, losses/gradients
  finite, and both students started with the SAME SHA-256
  `077315f67731a31e7d1389926d83e2bfdc220d9c4a3a0cf89740a042ab958587`.
  Across all five ranks, peak PyTorch allocated memory was 4.800 GiB maximum
  for DINO-only and 4.919 GiB for DINO + metric depth; maximum reserved memory
  5.330 / 5.461 GiB. Each RTX 3080 exposes 9.775 GiB. These are actual pilot
  measurements, not predicted savings, convergence or full-validation scores.
- Both formal jobs started AUTOMATICALLY after shared admission:

  | Experiment/phase | Job | Actual observed state |
  | --- | --- | --- |
  | DINO-only, 50 epochs, five-GPU DDP | 1250637 | RUNNING; update 56 verified |
  | DINO + metric depth, 50 epochs, five-GPU DDP | 1250641 | RUNNING; update 44 verified |
  | Control camera/fused probes and segmentation | 1250638 / 1250639 / 1250640 | PENDING; after 1250637 |
  | Depth camera/fused probes and segmentation | 1250642 / 1250643 / 1250644 | PENDING; after 1250641 |
  | Verified durable backup / eowyn data retirement | 1250645 | PENDING; after ALL six readouts plus result/backup guards |

- A short CPU-only step inside existing job 1250637 read both node-local run
  files: actual persisted configs are RGB **448x800**, ten sweeps, fifty epochs
  and the same streamed memory recipe. At the sampled updates, control loss /
  gradient norm were 0.7820 / 0.4575; depth loss / gradient norm 1.0407 / 0.5271,
  with depth CE 4.3638. Different total objectives MUST NOT be compared as model
  quality. Constructor logs print legacy 112x192 BEFORE `load_datasets` sets
  each actual dataset instance to 448x800; that banner is not the run config.
- Evidence: `artifacts/rpl_ddp_streamed_submission_20261107.json` records exact
  submitted argv/dependencies and retired IDs;
  `artifacts/rpl_ddp_streamed_admission_20261107.json` records full CPU audit,
  both real pilot/checkpoint hashes, per-rank memory and scheduler snapshot.
  All 77 CPU regressions, compilation, shell syntax and diff checks passed.
- Monitor from moria: `squeue -u hongen`; tail `dino_pretrain-1250637.out` or
  `depth_pretrain-1250641.out` under the NEW execution snapshot's `.cache`.
  The two jobs use five GPUs, twenty CPUs and 80 GiB RAM EACH; exact per-epoch
  sample coverage, equal global batch five and original objectives stay fixed.
  Next: monitor sustained finite updates/first full validation and estimate
  duration from steady throughput. The 47-hour graceful-stop/partial-exit-75
  limit remains; explicit same-config continuation/dependency repair is needed
  if fifty epochs do not fit. No finish-time or semantic improvement guarantee.
  Target data cannot be deleted before all eight completed full-validation
  results plus verified durable backup. Workstation queue/data/PDF are unchanged.

### 2026-11-07 — Repair DDP projection memory without changing the study

- Owner chose to continue DDP. Live checks confirmed migration 1250541 and
  full target audit 1250542 COMPLETED; source retirement 1250543 COMPLETED and
  only arwen's authorized dataset was removed. Eowyn data remain available.
  Five-GPU pilot 1250544 FAILED during the second backward: CUDA requested
  another 938 MiB on a 9.77-GiB RTX 3080. One control update was logged; the
  depth treatment and BOTH formal pretrains did not start. Nine dependent jobs
  1250545–1250553 remain pending and cannot recover from that failed dependency.
  The older submission entries below are historical, not current run status.
- `metric_depth_bev.py` now projects one camera at a time and immediately
  accumulates the SAME per-channel nonzero-masked mean, including its EPS and
  zero-cell gradient convention. Stateless projection alone uses non-reentrant
  activation checkpointing; the six-view ResNet forward and BatchNorm updates
  are untouched. `nets/segnet.py` exposes an opt-in reducer; ordinary Vox objects
  retain the original official projection path and state-dict keys. No image,
  depth-bin, BEV-grid, sweep, loss, learning-rate or epoch reduction is applied.
- `metric_bev_objective.py` runs frozen metric teacher inference before student
  activations become live; teacher targets and loss location stay unchanged.
  The strict run config records the memory recipe. Collective checkpoint/status
  telemetry now records peak allocated/reserved/total memory for EVERY rank in
  this profile; the pilot report retains it. Old strict checkpoints/source
  admissions are not silently reused with changed implementation hashes.
- Reproduce from this branch: `PYTHONPATH=scripts:cluster_rpl
  CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/dd2414-metric-tests OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
  /home/students2026/miniconda3/envs/bev/bin/python -m unittest test_metric_memory
  test_metric_depth_bev test_distributed_training test_depth_objective
  test_depth_continuation test_random_depth_pilot test_random_depth_study
  test_random_depth_migration test_depth_pilot test_depth_study
  test_baseline_followup test_multisweep_teacher test_bevcar_voxel_adapter`.
  Actual CPU result: 77 tests passed, including output/geometry/input/head
  gradient equivalence, zero features, frozen inference, local two-rank DDP and
  checkpoint/resume checks. The six-view fixture saves less than HALF the
  reference autograd tensor bytes and recomputes individual views in backward;
  this is NOT a CUDA peak-memory or throughput result.
- Next: push this scoped fix, make a NEW immutable deployment and fresh matched
  output/backup roots, rerun full CPU admission on the EXISTING eowyn dataset,
  then both real five-GPU/resume pilots before releasing formal jobs. Replace
  only the nine blocked own-account jobs; preserve the failed run/logs and old
  snapshot. Do NOT remigrate/redelete arwen data. Target cleanup stays gated by
  all eight completed full-validation results and verified durable backup.
  Workstation training, original dataset and staged user PDF are unchanged.

### 2026-11-07 — Latest cluster code synchronized; eowyn pipeline actually submitted

- User-authorized push and cluster `git pull --ff-only` succeeded. Reviewed
  implementation is `e97dace40f3d6270d33cecfc5285d32e3f15c649`; visible cluster
  clone stays on `experiment/rpl-depth-10sweep-velocity-study`. Queued jobs use
  immutable execution worktree
  `/Midgard/home/hongen/projects/DD2414-Project/.cache/deployments/e97dace40f3d`.
  Subsequent handoff-only commits may update the visible clone without changing
  this execution snapshot or its admission/source digests. DINO cache is linked;
  the distinct metric depth weight is installed with the pinned SHA-256. The
  workstation feature checkout remains separate from its unchanged main queue
  and unrelated staged PDF.
- Actual target GPU compatibility job **1250540 COMPLETED** on eowyn, with
  TWO RTX 3080s (9.77 GiB each), torch 2.5.1+cu121/CUDA runtime 12.1. Native
  BF16, NCCL reduction, finite SGD gradients and DDP parameter synchronization
  passed on both ranks; GPUs released. This is synthetic compatibility, NOT
  the new full-model admission, convergence or quality evidence.
- Actual config: visible cluster `.cache/random_depth_50epochs_20261107.json`;
  eowyn data `/local_storage/users/hongen/datasets/nuscenes`, runs
  `/local_storage/users/hongen/dd2414_runs/random_resnet101_metric_bev_50epochs_20261107`,
  shared result backup under the visible clone's
  `fullsize_baseline/runs/random_resnet101_metric_bev_50epochs_20261107`.
  Shared pinned BEVCar cache was cloned successfully from the old arwen asset.
- ALL thirteen jobs were accepted; current observed state is migration RUNNING
  and twelve downstream jobs PENDING / Dependency. No new training epoch or
  full-model update is verified. Pending GRES requests do NOT allocate GPUs.
  Exact jobs/dependencies/resources:

  | Phase | Job | Node/resources | Requires success |
  | --- | --- | --- | --- |
  | Dataset stream | 1250541 | arwen + eowyn; 2 CPUs/4 GiB EACH; 6 h; 0 GPUs | None |
  | Target full asset/sensor/split audit | 1250542 | eowyn; 2 CPUs/16 GiB; 30 min | 1250541 |
  | Retire old arwen dataset | 1250543 | arwen; 1 CPU/2 GiB; 30 min | 1250542 + target evidence guard |
  | Both new model/resume pilots | 1250544 | eowyn; 5 GPUs/20 CPUs/80 GiB; 40 min | 1250542 |
  | DINO-only 50 epochs | 1250545 | eowyn; 5 GPUs/20 CPUs/80 GiB; 47 h | 1250544 + both-pilot guard |
  | DINO-only camera/fused probes, segmentation | 1250546 / 1250547 / 1250548 | eowyn; EACH 1 GPU/4 CPUs/16 GiB; 47 h | 1250545 |
  | DINO + projected metric depth 50 epochs | 1250549 | eowyn; 5 GPUs/20 CPUs/80 GiB; 47 h | 1250544 + both-pilot guard |
  | Depth camera/fused probes, segmentation | 1250550 / 1250551 / 1250552 | eowyn; EACH 1 GPU/4 CPUs/16 GiB; 47 h | 1250549 |
  | Verify durable backup, retire eowyn dataset | 1250553 | eowyn; 1 CPU/2 GiB; 2 h | ALL six readouts + all eight result/backup guards |

- The two pretraining jobs are independent after shared admission; neither
  waits for the other's 50 epochs. Each group has identical random student,
  full data, seed/budget/global batch; the only added objective is projected
  metric teacher supervision in the depth treatment. Frozen probes are ten
  epochs each, supervised decoder adaptation five, with full 6,019-frame
  validation and standard GT Vehicle IoU. Their launch wrapper returns 75 on
  partial completion rather than releasing success dependencies incorrectly.
- Migration has genuinely entered `STREAMING_DATASET_ARWEN_TO_EOWYN`, with
  tar checkpoints 500000 and 1000000 observed (roughly 10.24 GB archive output,
  not a full-data completion claim). Temporary destination is
  `nuscenes.incoming-1250541`; source has NOT been deleted. No dataset passes
  through Home or moria. Arwen appears in the NEW graph only for CPU copying
  and verified source retirement, never for the obsolete GPU experiments.
- The account-wide inspection is complete: cancelled old IDs 1250363/1250364
  remain CANCELLED; prior 1250539 audit is COMPLETED and has released its two
  CPUs. No old queued/held/orphan training tasks remain. New pending jobs above
  are intentional success dependencies. Other accounts' jobs were not changed.
- Actual new CPU validation: 21 focused node/path, transfer-admission, cleanup
  backup/result, partial-readout, pilot and launch tests pass; compilation and
  diff checks pass. Earlier model implementation had 131 passing CPU checks.
  Real five-GPU full-model/resume admission is still pending, not substituted
  by these CPU or synthetic CUDA checks. Raw audit/submission ledgers remain
  ignored; the scoped permanent record with exact submitted argv is
  `artifacts/rpl_random_depth_submission_20261107.json`.
- Monitor from moria: `squeue -u hongen`; transfer stderr is execution-root
  `.cache/stage-1250541.err`, stdout `.cache/stage-1250541.out`. Audit/pilot
  logs use `audit-1250542` / `pilot-1250544`; formal logs use
  `dino_pretrain-1250545` / `depth_pretrain-1250549`, all `.out`/`.err` under
  that execution cache. Submission ledger is the visible clone's
  `.cache/random_depth_submission_20261107.json`. Do NOT resubmit duplicate
  writers or modify the immutable snapshot while jobs reference it.
- Limits/next: let transfer/audit/pilots advance through Slurm, inspect actual
  finite model updates before claiming training health, then read full GT
  results. A partial/failing phase blocks dependent training/cleanup safely.
  A formal 47-hour signal exit saves state and returns 75; if 50 epochs do not
  fit, explicit same-config continuation and dependency repair are required.
  No unbounded automatic requeue, finish-time guarantee or semantic gain claim.
  Target cleanup cannot delete data before all required results and verified
  shared backups exist. Workstation data and old/new checkpoints are retained.

### 2026-11-07 — Full account audit, eowyn staging and explicitly bounded cleanup

- Owner asked to cancel ALL old arwen queues and comprehensively inspect
  cluster changes/waiting work. Actual audit: hongen queue empty; old IDs
  1250363/1250364 are CANCELLED, 1250362 remains historical FAILED. No cron,
  at, user systemd timers/jobs, tmux/screen sessions or orphan training workers
  were found. Login processes are normal SSH/VS Code and this audit itself.
  One clean project clone, three historical local branches, no extra worktrees
  and only mamba env `dd2414` were present; `dd2414-hybrid` is absent. Home uses
  about 14.2/100 GiB. This inventories current own-account state and retained
  Slurm history, not an impossible reconstruction of previously deleted files.
- CPU inspections 1250538 (eowyn import/space) and 1250539 (two-node file/process
  inventory) COMPLETED with zero GPUs. The latter is ONE Slurm allocation with
  `.extern` and `.0` steps, not three training jobs; duration 2m28s, two CPUs
  total. Arwen: approximately 397 GiB dataset, 2.8 GiB old runs, 960 MiB model
  assets, root owned by hongen; no lingering training processes. Eowyn: no own
  data directory, about 1,912 GiB free, shared torch 2.5.1+cu121 imports. Other
  users' GPU jobs were only read, never modified. Raw own-account/node audit
  reports are in the cluster clone's ignored `.cache/cluster_*_20261107.json`.
- Owner approved migrating to eowyn and explicitly selected deletion of the
  arwen DATASET only after the destination is verified, then deletion of the
  eowyn dataset only after experiments finish. Workstation data/checkpoints
  are retained. The earlier sync helper failed on moria's system Python 3.6
  BEFORE any mutation; all further login orchestration uses existing Miniforge
  Python 3.12.10. No system Python/environment change is needed.
- NEW `stage_random_depth.py` streams the complete dataset inside a CPU-only
  two-node Slurm allocation, writes to a fresh temporary target, checks
  extraction and atomically promotes it. No dataset staging into 100-GiB Home,
  deletion of source, overwrite of existing target, or GPU idle reservation.
  Pinned BEVCar source is cloned from arwen into the shared cache for eowyn.
  `random_depth_study.py` accepts staged Ampere nodes arwen/eowyn/galadriel;
  default example selects eowyn. Model/loss/sampling remains unchanged.
- `audit_random_depth.py` records actual node/job and mirrors passed admission
  into the frozen shared execution cache. `cleanup_random_depth.py` can retire
  the exact hongen source only after matching target transfer/admission; target
  retirement waits for ALL eight full-data pretraining/GT results and a
  byte-hash-verified shared run backup. Owner/path/hostname/active-job guards
  preserve unrelated data and checkpoints. No deletion has occurred yet.
- `random_depth_readout.py` runs the existing planned frozen camera/fused 1x1
  probes and full decoder GT adaptation, with partial exit 75 and safe signal
  forwarding. This enables an automatic end-of-experiments backup/cleanup
  dependency, rather than prematurely deleting data after pretraining alone.
  Every readout still requires a completed matching 50-epoch checkpoint.
- Reproduce CPU checks with the previous interpreter/env and `python -m
  unittest test_random_depth_migration test_random_depth_pilot
  test_random_depth_study test_depth_pilot test_depth_study`; also compile new
  launch/migration/cleanup modules and run `git diff --check`. All 21 targeted
  CPU tests and static checks pass. Next: push/sync, pin execution snapshot and submit migration,
  audit, both-pilot admission, both formal pretrains, GT readouts and cleanup
  with afterok dependencies. Failed/partial phases must preserve data and block
  downstream work. No GPU admission, transferred target or new epoch is claimed
  by this preparatory entry; actual IDs/status follow in a deployment entry.

### 2026-11-07 — Prepare automatic cluster admission and formal submission

- Owner requested pushing the latest code, synchronizing the cluster and
  submitting the NEW experiment. Also explicitly authorized clearing the old
  experiment queue. Read-only SSH confirmed the clean hongen clone at c0e352f,
  correct account/data node, 14.2 GiB of the 100 GiB home quota in use, all ten
  arwen GPUs allocated, and only old pending jobs 1250363/1250364 in this account.
  Their exact commands, owner and pending states were checked before cancelling
  those two IDs; no other account/job was touched. Transfer of the distinct
  pinned metric teacher weight uses an ignored temporary file, verified before
  atomic rename. This does not replace the earlier relative teacher asset.
- Added `cluster_rpl/random_depth_pilot.py` and its CPU contracts. A single
  bounded five-GPU allocation runs BOTH real batch wrappers through update two
  and fresh-process resume to update three. It verifies 50-epoch random student
  provenance, full-data config, all five sampler/RNG states, finite model and
  optimizer tensors, changed weights, contiguous metrics and identical initial
  students. Combined success is written ONLY after both pass. It never submits
  jobs, silently falls back, or calls this a quality/throughput result.
- `random_depth_study.py --require-pilot` now gates formal execution on both
  matching real-model reports, exact code commit/config/world size. Planned
  scheduler graph: 2-CPU audit -> 5-GPU pilot -> two INDEPENDENT 5-GPU/20-CPU/
  80-GiB/47-hour formal jobs, both resuming their three-update checkpoint.
  Failures block the long jobs through `afterok`, without manual intervention.
  Fifty epochs remain the total budget, not fifty extra epochs after pilots.
- Execution will be pinned to an immutable detached worktree under the cluster
  clone's `.cache/deployments/COMMIT`; the visible feature checkout can then
  receive status documentation without invalidating audit/source guards or
  changing queued code. Only the existing DINO cache is linked into the
  snapshot; student data/output stay arwen-local. No workstation model/data/
  queue or staged user PDF changes. The updated 50-epoch plan explains this.
- Reproduce checks: `PYTHONPATH=scripts:cluster_rpl CUDA_VISIBLE_DEVICES=''
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /home/students2026/miniconda3/envs/bev/bin/python -m unittest
  test_random_depth_pilot test_random_depth_study test_depth_pilot
  test_depth_study`. Actual result: all 15 targeted CPU tests pass; static
  compilation/diff checks pass. The previous 131 CPU regressions remain model
  implementation evidence; this patch does not change model/loss/sampler code.
- Limitations/next: GPUs are currently occupied. Deployment/audit/pilot/formal
  job IDs and observed states will be recorded after actual submission. This
  preparatory entry is not a claim of CUDA admission, a started epoch or a
  semantic result. Formal 47-hour partial exits still require same-config
  continuation; no indefinite resubmission loop is introduced.

### 2026-11-06 — Random ResNet101, two matched metric-depth studies, 50 epochs

- Owner approved implementing the geometric depth-probability design and a
  50-full-epoch first budget, with configurable multi-GPU training. Work is
  isolated to `experiment/rpl-depth-10sweep-velocity-study` in its worktree.
  The workstation branch/queue and its unrelated staged `DD2414_Project_Plan.pdf`
  are untouched. Original `nets/segnet.py`, official defaults and checkpoints
  are unchanged; this is an opt-in model/profile.
- NEW scientific matrix supersedes the three warm-start continuations below:
  `dino-only` versus `dino-metric-depth-bev`. Both students are entirely random
  (seed 125), RGB 448x800, 10 radar sweeps, velocity PROVIDED, and identical
  80-bin probability-weighted camera lifting. Both use frozen DINO image and
  radar-supported fused-BEV terms (0.2/0.8). Only the second adds frozen outdoor
  metric Depth Anything supervision, CE weight 0.05, AFTER calibrated camera
  BEV-volume projection and BEFORE radar fusion. No velocity-hidden group,
  official RGB/previous student initialization, radar-depth loss or velocity
  reconstruction auxiliary is used. The last two exclusions implement the
  owner's DINO-only control; velocity still remains an input feature.
- `scripts/metric_depth_bev.py`, `metric_bev_objective.py`, and
  `metric_bev_profile.py` implement this profile. The single 1x1 depth head
  predicts 80 probabilities from 1 to 80 m, and remains at inference. Teacher
  metric depths become soft targets (2 m width), compared against student log
  probabilities sampled onto the same camera geometry. Depth-axis bin centers
  honor `align_corners=False`; existing image XY sampling is retained. A 20 m
  target increases mass near 20 m and softmax suppresses alternate distances
  along that ray. No extra semantic decoder or student transformer is added.
- Frozen teacher is the genuine `metric_depth` VKITTI Small implementation,
  not the earlier relative teacher. Downloaded ignored metric weights, verified
  SHA-256 `9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`,
  strictly loaded and ran TWO synthetic CPU images (84x126): finite [2,1,84,126]
  output, eval mode, zero trainable teacher parameters. This does not measure
  real-scene depth accuracy. Relative teacher weights remain separate.
- Extended `scripts/train_depth_ddp.py` and `train_multisweep_teacher.py` behind
  explicit `--training-profile random-metric-bev`. New profile requires 50
  epochs, full train/validation, BF16, random provenance and matched AdamW
  rates 2e-4, one-epoch warmup/cosine, clipping 5. Supports 2/5/10 local ranks,
  without padding/dropping, per-rank batch 1. Default candidate is FIVE GPUs
  per model (20 CPUs/80 GiB); two separate five-GPU experiments may overlap
  when resources permit. Four/eight ranks deliberately fail on indivisible
  28,130 samples. Same-world exact resume preserves optimizer/sampler/RNG/BN.
- Every full epoch validates ALL 6,019 frames and writes telemetry/TensorBoard;
  retains validated epochs 5/10/20/30/40/50 plus periodic/signal latest saves.
  Milestone retention also handles resuming into pending epoch-end validation.
  Fifty epochs mean 1,406,500 training visits per model; this is a budget, not
  proof of convergence. Downstream strictly rebuilds/loads the new head and
  requires completed pretraining AND final full validation. Frozen camera/fused
  1x1 probes and supervised Simple-BEV decoder adaptation remain SINGLE GPU;
  primary score is GT Vehicle IoU. DINO/depth losses are auxiliary diagnostics.
- New `cluster_rpl/random_depth_study.py`, example config, batch wrapper, and
  `audit_random_depth.py` provide stdlib previews, explicit allocated execution
  and a fresh exact-commit/config CPU admission gate. They never submit jobs
  automatically and do not require an old official RGB/parent checkpoint.
  The audit checks official scene/sample/log membership, pinned source/teacher
  content and all metadata sensor references. Absolute arwen-local paths and
  resource checks are explicit. No system/mamba environment mutation occurs.
- Fixed `cluster_rpl/depth_ddp_job.sh` repository resolution using explicit root
  or `SLURM_SUBMIT_DIR`, instead of the Slurm spool script location. NEW batch
  wrapper uses the same correction. Read-only SSH confirmed old job 1250362
  failed exit 2 in one second on that exact path error; 1250363 is blocked by
  `DependencyNeverSatisfied`, 1250364 by `Dependency`. Those jobs/spool copies
  were NOT modified, cancelled or resubmitted. The earlier resource-waiting
  entry below is historical. Current arwen inspection: ten of ten GPUs
  allocated, CPUAlloc 44/48; no new CUDA pilot or formal run was started.
- Actual checks: new calibrated geometry/20 m counterexample/gradient tests,
  real TWO-process Gloo gradients for BOTH new forward paths (two optimizer
  updates each), legacy objective equivalence and fresh-process resume tests,
  launch/asset/matrix/downstream regressions pass. Both batch wrappers execute
  from relocated temporary spool copies (including paths with spaces), with
  wrong roots rejected. Shell syntax, Python compilation, plan preview and
  `git diff --check` pass. The full combined unit command below passes **131
  tests**; see `artifacts/random_metric_bev_implementation_20261106.json` for
  construction/teacher/split/old-job evidence.
- Full real student CPU CONSTRUCTION check: both states hash identically to
  `077315f67731a31e7d1389926d83e2bfdc220d9c4a3a0cf89740a042ab958587`;
  42,767,929 parameters, 38,936,880 pretraining trainable, [80,128,1,1] depth
  head; decoder/motion head frozen. Checkpoint/hub loading was forbidden in
  that check. Inherited Segnet allocates its coordinate grid on CUDA, so only
  that allocation was redirected to CPU for this construction check; no full
  model forward/CUDA memory/throughput claim follows from it.
- Live local official metadata check passes: 700/150 scenes, 28,130/6,019
  samples, 50/18 logs and ZERO train/validation scene/sample/log overlaps;
  manifest hash `dc39006dffbf036d5ea7728dad9348b822c78cd47cd02c226b638271c3420c5d`.
  Existing official validation is not renamed a blind test. Depth pseudo
  targets can have metric/domain bias; they are not dense GT or occupancy.
- Reproduction (from this worktree, no CUDA):
  `PYTHONPATH=scripts:cluster_rpl CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /home/students2026/miniconda3/envs/bev/bin/python -m unittest
  test_metric_depth_bev test_random_depth_study test_distributed_training
  test_depth_objective test_depth_study test_depth_assets test_depth_pilot
  test_cluster test_multisweep_teacher test_depth_continuation
  test_official_full_data test_training_checkpoint test_fullsize_training
  test_full_data_protocol test_radar_scaling_data test_baseline_followup
  test_radar_velocity_rotation test_radar_evaluation test_trainval_configuration`.
  Plan preview: `python3 cluster_rpl/random_depth_study.py --config
  cluster_rpl/random_depth_study.example.json` (no allocations/model imports).
- Detailed deployment, asset transfer/hash, CPU audit, bounded two-update
  pilot, new-process resume and 47-hour formal/resume commands are in
  `CLUSTER_RANDOM_DEPTH_50EPOCH_PLAN.md`; legacy plan/README now route there.
  Next: deploy the committed branch, transfer the metric weight, refresh CPU
  admission, then run BOTH finite-update/resume CUDA pilots with the actual
  batch wrapper and candidate world size. Lock the same world size for the
  formal pair only after memory/throughput evidence. No implicit requeue or
  automatic OOM/config changes; a partial signal exit 75 needs another
  same-config allocation. No finish-time or semantic gain is claimed yet.

### 2026-11-06 — Formal holds released; first treatment awaits resources

- Owner explicitly requested removing all three holds and starting the
  experiments. After checking the exact own job IDs/commands/configuration,
  unchanged clean deployed implementation (`c0e352f`) and successful formal
  audit `1250361`, ordinary Slurm `scontrol release` succeeded for `1250364`,
  `1250363` and `1250362`, in that order. No job was resubmitted, cancelled,
  reprioritized, moved to another account/node, or given different resources.
  This records the owner's execution request and actual scheduler operation,
  not a new assertion about external administrator approval.
- Verified post-release state: `1250362` no depth is `PENDING / Resources`
  with no remaining dependency; `1250363` radar depth is `PENDING / Dependency`
  on `afterok:1250362`; `1250364` radar-plus-Depth-Anything is `PENDING /
  Dependency` on `afterok:1250363`. Priority is nonzero (3893 at inspection),
  and NONE is `JobHeldUser`. Every job retains two GPUs, eight CPUs, 32 GiB,
  arwen pinning, the 47-hour cap and original signal/checkpoint configuration.
- Resource evidence: `scontrol show node arwen` reports ten configured GPUs
  and ten allocated GPUs, with no free pair. The first job therefore has not
  started, has runtime zero/no allocated node and no verified formal optimizer
  update. This is scheduler resource waiting, NOT a new manual pause or model
  failure. Other nodes' free GPUs do not make arwen's local dataset shared;
  no data migration or arbitrary node change was attempted.
- Persistence: Slurm can start the first job automatically when scheduling
  allows; successful completion then enables the next treatment. All holds
  were removed, so no further user-hold release is needed for this chain.
  No interactive session/agent must remain connected for scheduling/execution.
  A failure/partial exit still blocks dependent jobs and needs inspection;
  this is not a promise of automatic fault recovery or a full downstream queue.
- Monitor on moria: `squeue -j 1250362,1250363,1250364 -o
  "%.18i %.45j %.10T %.10M %R"`. First-treatment logs, created only once the
  job actually starts, are project `.cache/depth-none-formal-1250362.out` and
  `.err`; run telemetry is node-local formal root `depth_none/pretrain/` with
  `status.json`, `metrics.jsonl` and TensorBoard. Do not call missing pre-start
  logs a training crash or execute node-local compute on moria.
- Changed only this handoff and `CLUSTER_DEPTH_RUN_PLAN.md` to replace stale
  current-held status with the observed release/resource-waiting status;
  earlier dated records remain history. Actual checks: all three post-release
  scheduler records/dependencies verified; `git diff --check` passes and
  model/launch code is unchanged. No new CPU/CUDA regression is required for
  this documentation-only record. Workstation queue and staged PDF untouched.
- Next: let the existing scheduler requests obtain resources. On RUNNING,
  verify a real finite optimizer update and logs before reporting training
  progress/throughput. No exact start/finish time is guaranteed; no formal
  epoch/validation or new semantic result has completed at this checkpoint.

### 2026-11-06 — Deduplicate assistant-written GPU restriction commentary

- Owner request: remove the assistant's single-GPU safety-review commentary.
  Scope is documentation cleanup, not disabling platform/tool review or
  overriding external cluster rules. Neither `AGENTS.md` nor the repository's
  Stop hook contains a GPU-count limit; those files/settings are unchanged.
- Changed `CLUSTER_DEPTH_RUN_PLAN.md` and `cluster_rpl/README.md`: removed
  repeated assistant-written single-GPU/account-policy explanations and the
  duplicate historical review narrative from the operational guide. The
  detailed original permission/rejection records remain UNCHANGED below in
  this handoff, and application/release status has this single entry point.
  This reduces duplicate instructions without falsifying execution history or
  asserting any new permission. Site manuals and platform controls are not
  edited. No memory files or credentials are changed.
- Kept technical GPU profiles explicit: depth continuation uses two DDP
  processes/GPUs; inherited deployment and downstream readouts use one GPU as
  implementation configurations, not a repository-wide safety cap. No model,
  sampler, resource request, admission guard, checkpoint validation or loss is
  changed. The plan also removes stale statements that the parent is not yet
  transferred/formal applications are not yet submitted, and points to the
  actual formal configuration, refreshed audit and dependency chain above.
- Actual checks: `git diff --check` passes; only these three Markdown files
  change; training code, `AGENTS.md`, hooks and existing historical handoff
  contents are unchanged. No CPU/CUDA regression is needed for this doc-only
  cleanup. Reproduce review with `git diff --name-only` and `git diff --
  CLUSTER_DEPTH_RUN_PLAN.md cluster_rpl/README.md PROJECT_HANDOFF.md`.
- Limitations/next: removing repeated prose does not establish external
  authorization or release submitted jobs. No cluster connection, Slurm
  submission/release/cancellation or workstation training change occurred.
  The previously recorded held jobs are not altered by this edit; address
  launch/release separately under the current execution permissions. The
  removed duplicate text remains recoverable from Git history.

### 2026-11-06 — Formal two-GPU requests submitted HELD, not started

- Owner request: submit the formal application after clarifying that earlier
  jobs were bounded pilots only. Real Slurm applications are now submitted,
  NOT merely `--test-only`. They were deliberately submitted `--hold`, as
  announced before execution: longer two-GPU site-permission coverage remains
  unconfirmed. No hold was released and no formal GPU allocation/training has
  started. Scheduler acceptance must not be called administrator approval.
- New ignored configuration, on both workstation and cluster:
  `.cache/depth_formal_20261106.json`. It preserves every scientific setting
  from `cluster_rpl/depth_study.example.json`, changing ONLY `run_root` to
  `/local_storage/users/hongen/dd2414_runs/depth_continuation_formal_20261106`.
  The three fresh five-epoch continuations load the SAME completed parent;
  previous pilot outputs/checkpoints are preserved, not resumed or overwritten.
- Refreshed zero-GPU CPU audit `1250361` completed `0:0` in 1m49s (audit work
  98.7 seconds), validating the formal config's full metadata/sensor references
  and parent/teacher assets. Report: formal root `depth_assets_audit.json`;
  logs: project `.cache/depth-formal-audit-1250361.{out,err}`. The preceding
  scheduler-only check printed hypothetical `1250360`; that is NOT a real job.
- Actual held job inventory, verified with `squeue` and `scontrol show job`:
  `1250362` no depth, initially `afterok:1250361` (CPU audit);
  `1250363` radar depth, `afterok:1250362`;
  `1250364` radar depth + Depth Anything V2, `afterok:1250363`.
  All are `PENDING`, reason `JobHeldUser`, priority zero, runtime zero and no
  allocated node. Each requests arwen, ONE node/task, TWO GPUs, eight CPUs,
  32 GiB, a 47-hour maximum and `--signal=B:USR1@300`. After a future explicit
  release, success dependencies prevent concurrent treatments and stop the
  chain after failure/partial exit. No downstream readout jobs were submitted.
- Submission/reproduction: use `sbatch --hold --dependency=afterok:PRIOR_JOB`
  with the resource/signal settings above and the ABSOLUTE script path
  `/Midgard/home/hongen/projects/DD2414-Project/cluster_rpl/depth_ddp_job.sh`,
  followed by the absolute ignored config path and `none`, `radar` or
  `radar-dav2`. The first relative-script submission failed to open the file
  and issued no job ID; switching to the verified absolute path resolved it.
  `--chdir` controls execution cwd, not where sbatch finds its input script.
- Checks: all three formal commands/configurations passed local stdlib
  validation; shell syntax and `git diff --check` pass. Actual scheduler
  records confirm resources, dependency chain and holds. These are submission
  checks, NOT new model tests; the previously recorded 92 CPU tests and three
  real CUDA pilots remain the training implementation evidence. Only this
  handoff and the depth plan change in this outcome; no model/data code,
  environments, other users' jobs or workstation queue/PDF are altered.
- Next/boundaries: report these real IDs transparently. Confirm longer two-GPU
  site authorization before any `scontrol release`; no automatic release,
  retries or cancellation are part of this task. Keep research code/config
  stable and recheck admission before launch. Slurm's default `Requeue=1` is
  recorded, but the launcher does not automatically add `--resume` after a
  restart: do not claim unattended fault recovery. A partial/failed dependency
  requires inspection and a separately authorized resume application.

### 2026-11-06 — Three real two-GPU depth admission pilots passed

- Actual cluster results: zero-GPU full-data audit `1250328` completed `0:0`
  in 1m46s; two-GPU no-depth `1250329`, radar-depth `1250330` and radar-plus-
  Depth-Anything `1250331` each completed `0:0` in 2m18s / 2m17s / 2m18s.
  Each used arwen, two GPUs, eight CPUs, 32 GiB and a ten-minute hard cap,
  sequentially, never overlapping own GPU allocations. Each ran two real
  full-model updates, saved, launched NEW torchrun processes and completed
  update three. Both rank-local cursors advanced 2 -> 3, global samples 4 -> 6;
  finite checkpoint tensors, optimizer state, metrics exactly 1,2,3, and actual
  changed model state passed verification. All allocations were released.
- Evidence: compact `artifacts/rpl_depth_admission_20261106.json` records exact
  commits/configuration, job IDs, hash/coverage evidence, memory/loss diagnostics
  and limitations. Full reports/checkpoints remain under arwen's personal
  `dd2414_runs/depth_continuation_10sweep_velocity_present_ddp`; stdout/stderr
  are project `.cache/depth-{none,radar,dav2}-pilot-JOBID.{out,err}`. Tested code
  is `0e628d7`. No five-epoch phase, full validation or downstream readout has
  completed, and no semantic-performance result is claimed.
- Observed rank-zero PyTorch peaks across initial/resumed steps: 6.83 GiB
  no-depth, 7.80 GiB radar-depth, 7.89 GiB radar-plus-Depth-Anything. These are
  NOT combined GPU memory or a guarantee that all later samples fit. Depth
  Anything's PRE-clipping gradient norms were 557, 768, 305; finite but much
  larger than other treatments. Existing global clipping at five remained
  unchanged. Monitor clipping/relative-depth behavior in longer training;
  three changing-input losses do not diagnose divergence or convergence.
- Telemetry fix: a pilot exposed constructor-time optimizer group dictionaries
  becoming stale after `optimizer.load_state_dict`. The DDP status logger now
  reads `optimizer.param_groups`, just like the existing single-GPU trainer.
  CPU inspection of ALL three saved optimizer states confirmed the REAL
  update-three learning rates were camera `1.0076791808873722e-6`, student
  `2.0153583617747444e-5`, not the stale base rates printed in old logs.
  Optimizer/schedule/update math is unchanged. Added a regression using actual
  optimizer restore and the real telemetry expression. The fix is not yet a
  new full-model CUDA run; the artifact distinguishes code versions.
- Checks: 92 CPU tests pass with the existing combined suite plus
  `cluster_rpl.test_depth_assets` / `cluster_rpl.test_depth_pilot`; compilation,
  shell syntax and diff checks pass. Changed files in this outcome are the
  DDP telemetry line/test, the compact artifact and its exact `.gitignore`
  exception, this handoff and the depth plan/README. The workstation source/
  service/staged user PDF remain untouched.
- Retrieval limitation: a binary rsync download through `srun --unbuffered`
  failed with protocol/buffer-overflow code 22; no local report files were
  produced, and the originals remain intact. Text JSON retrieval through a
  CPU-only step succeeded. Do not infer the previously successful upload route
  is reliable for binary downloads, or modify sensor files to address this.
- Next/boundary: the owner authorized deployment/short trials; formal long
  two-GPU permission coverage was asked for and is still unconfirmed. No
  formal long training was submitted. Confirm that site coverage first.
  The telemetry fix changes strict implementation digests: retain old pilot
  directories as evidence, do NOT relax resume checks, and choose a NEW formal
  output root with fresh five-epoch phases from the SAME approved complete
  parent for all treatments. Refresh CPU admission for that config and run
  matched formal continuations only after authorization. Downstream stays
  single-GPU, with Vehicle BEV IoU primary. Hidden-velocity depths stay deferred.

### 2026-11-06 — Owner authorizes depth deployment and admission checks

- The owner lifted the previous local-only restriction. Initial read-only
  cluster inspection found the account queue empty and the old project clone
  clean at `f9f333f`. The singular-path clone
  `/Midgard/home/hongen/projects/DD2414-Project` was switched to the retained
  depth branch at `39ed99e`, preserving existing remote fetch mappings. The
  abandoned velocity-hidden multi-GPU branch is not restored or deployed.
- Transferred the completed immutable parent (483,346,682 bytes) to arwen's
  personal assets directory and the Depth Anything V2 Small weight (99,218,434
  bytes) plus exact Python source/license to the shared project cache. All
  transfers finished with exit zero; remote hash verification is a required
  next gate, not inferred from transfer success. A CPU-only allocated check
  found the expected dataset directories, original RGB/BEVCar assets, about
  2.9 TiB node-local free disk, and `pip check` passed in the existing mamba
  environment. No environment replacement or workstation queue change.
- New files: `cluster_rpl/check_depth_assets.py` audits full sensor references,
  official split/manifest membership, pinned parent completion/provenance,
  original encoder/BEVCar and both frozen teachers, environment and model/data
  source hashes. It refuses login-node/out-of-allocation execution and writes
  the compact admission report under the personal study output root.
  `test_depth_assets.py` checks the allocation and exact-coverage guards.
  `assets.json` now pins the actual Depth Anything weight/Python-tree hash and
  source commit; runtime weights/source remain ignored and outside Git.
- Reproduce CPU admission (only in an allocated arwen CPU job, zero GPUs):
  `python cluster_rpl/check_depth_assets.py --config .cache/depth_study.json`.
  CPU regression: `PYTHONPATH=scripts:cluster_rpl CUDA_VISIBLE_DEVICES=''
  python -m unittest cluster_rpl.test_depth_assets cluster_rpl.test_depth_study
  cluster_rpl.test_cluster scripts.test_depth_continuation -q`.
- Actual local checks: 39 CPU tests passed; Python compilation and
  `git diff --check` passed. Full remote sensor/hash audit and actual two-GPU
  finite-update/resume pilots are still pending at this preparation checkpoint.
  Prior diagnostic permission is not silently broadened to long formal jobs.
- Next: deploy this committed audit, run zero-GPU admission, then bounded
  real-model update-two/new-process-update-three tests for each treatment.
  Do not launch formal long-running two-GPU training until resource permission
  coverage is confirmed and all three full-model gates pass. No claim of
  semantic results, CUDA memory feasibility or training completion is made.

Deployment follow-up: CPU audit job `1250328` on arwen reported `passed` in
96.0 seconds. It verified all 2,631,083 metadata-referenced sensor files present
and nonempty, official full split membership/disjointness, original RGB,
BEVCar, DINO and Depth Anything hashes, and the complete 727-tensor parent.
The detailed node-local report is `RUN_ROOT/depth_assets_audit.json`; logs are
project `.cache/depth-assets-audit-1250328.{out,err}`. This is file-presence and
provenance evidence, not a bytewise dataset audit or full-model CUDA evidence.

Added `cluster_rpl/depth_pilot.py` and `test_depth_pilot.py`: within ONE approved
bounded two-GPU allocation it runs one treatment to update two, preserves that
checkpoint, restarts torchrun from it, and requires update three with both rank
states/finite tensors/changed model state and metrics exactly 1,2,3. It requires
the passed data audit and a fresh run directory; it neither submits nor starts
formal training. Reproduce: `python cluster_rpl/depth_pilot.py --config
.cache/depth_study.json --treatment none` inside the selected allocation; repeat
for the other two modes. Actual local checks: 41 CPU tests passed including
the two pilot-verifier tests; compilation and diff checks pass. The depth plan
documents both admission commands. Actual GPU pilot results are still pending.

### 2026-11-06 — Withdraw velocity-hidden multi-GPU branch

- Owner decision: abandon `experiment/rpl-velocity-hidden-multigpu` and resume
  work exclusively on `experiment/rpl-depth-10sweep-velocity-study`. This does
  not cancel the workstation's existing velocity-hidden experiment/queue.
- Admission: the abandoned branch pointed to `612a428`, identical to the depth
  branch, with no unique commits or upstream. Exact GitHub ref inspection found
  no remote velocity-hidden branch. Its isolated worktree contained only two
  modified tracked files (`scripts/train_depth_ddp.py` and
  `scripts/train_multisweep_teacher.py`) and one untracked file
  (`scripts/pretraining_budget.py`), with no ignored runtime assets. No training
  process using that worktree was found during inspection.
- Recovery before removal: saved those three complete working files under
  `/home/students2026/dd2414/.cache/branch_backups/velocity-hidden-multigpu-20261106/uncommitted-files.tar.gz`.
  All three extracted streams matched the original SHA-256 hashes. Saved and
  verified incremental `branch.bundle` beside the archive; it retains the
  abandoned ref at `612a428` and requires `157857c`, which remains in history.
  These ignored local recovery files are not pushed and contain no credentials.
- Completed: removed exactly the abandoned worktree and deleted its local
  branch. The withdrawn multi-GPU velocity profile and timing-budget changes
  were NOT applied to the retained depth branch. Its existing two-GPU depth
  implementation remains unchanged at the code content from `612a428`.
  Only this handoff entry changes on the retained branch. The main checkout's
  staged user PDF, other branches, dataset, checkpoints and running queue are
  untouched. No SSH, cluster update, job submission or cancellation occurred.
- Verify: `git branch --list experiment/rpl-velocity-hidden-multigpu`,
  `git worktree list`, and `git ls-remote --heads origin
  refs/heads/experiment/rpl-velocity-hidden-multigpu` must show no abandoned
  branch/worktree. `git diff 612a428 -- scripts cluster_rpl` must be empty.
  Actual checks: bundle verification and archive hash checks passed; exact
  local branch/worktree absence and unchanged depth implementation were checked;
  `git diff --check` passed. No model tests were rerun for this metadata-only
  withdrawal; the retained implementation's previous 87 CPU tests remain the
  recorded evidence, not a new CUDA result.
- Limitations/next: discarded edits are recoverable only from the local archive,
  not GitHub. If recovery is explicitly requested, inspect the archive with
  `tar -tzf` and extract into a NEW temporary directory, never over the active
  depth worktree. Continue the existing depth plan and its pending admission
  gates; the earlier no-cluster-deployment restriction remains in effect until
  the owner explicitly authorizes deployment.

### 2026-11-06 — Local-only two-GPU depth-continuation implementation

- Scope: implemented on `experiment/rpl-depth-10sweep-velocity-study` in its
  isolated worktree. The owner explicitly requested code preparation only:
  NO SSH, cluster checkout update, asset transfer, Slurm submission or CUDA
  training was performed in this update. The workstation checkout/queue and
  unrelated staged `DD2414_Project_Plan.pdf` remain unchanged by this work.
- Changed training files: `scripts/train_depth_ddp.py` implements the opt-in
  single-node/two-rank NCCL loop. `scripts/train_multisweep_teacher.py` adds
  explicit `--distributed` dispatch, refuses accidental duplicate non-DDP
  torchrun workers, and rejects incomplete DDP parents for downstream use.
  Official Simple-BEV execution/model tensors and existing single-GPU loss,
  optimizer, sampler and checkpoint format are not changed. The experimental
  single-GPU config now records `distributed=false`; this branch is not a
  claim of cross-code-version exact resume for historical runs.
- `scripts/depth_pretraining_objective.py` executes all trainable projection
  heads INSIDE DDP forward, with frozen per-rank DINO / optional Depth Anything
  teachers. Its losses and student gradients match the actual old single-GPU
  closure under controlled CPU inputs for all three depth treatments.
  `scripts/distributed_training.py` adds disjoint no-padding acknowledged
  sampling, prefetch-independent sample RNG, collective finite/stop/error
  handling, uneven validation shards and atomic schema-two checkpoints.
  Rank-local training RNG starts at seed-plus-rank. Each rank's RNG, sampler,
  scaler and BN buffers are preserved; model and
  optimizer updates are synchronized, and only rank zero writes outputs.
  Resume strictly rejects changed world size/configuration/implementation.
  Completed checkpoint model keys stay unprefixed for single-GPU readouts.
- Experiment recipe: the approved complete velocity-present parent
  `ae9daf1124a03ee5a47f5fac68886334bcdd12dcbe5684c8a64167982c632520`
  initializes all three treatments strictly. Each adds five full epochs with
  fresh optimizer/sampler/schedule. RGB is 448x800, both teachers 448x798,
  radar ten sweeps, BF16, seed 125, full official 28,130/6,019 split. Local
  batch one / global batch TWO gives 14,065 updates per epoch / 70,325 per
  continuation. Losses are averaged per-sample rank objectives, not global
  valid-cell normalization. LR stays camera 1e-5 / student 2e-4. BatchNorm is
  rank-local with rank-zero buffer broadcast, NOT SyncBatchNorm. This is a
  matched new recipe across depth modes, not an identical trajectory to old
  global-batch-one runs. Each readout branches independently from its own
  completed continuation, not another downstream checkpoint.
- New deployment preparation: `cluster_rpl/depth_study.py` and
  `depth_study.example.json` preview three two-GPU continuation commands and
  twelve ONE-GPU downstream commands. No automatic submit/queue/requeue/SSH.
  `depth_ddp_job.sh` launches exactly one explicitly chosen treatment only
  inside Slurm. The wrapper's pre-time-limit signal writes a checkpoint flag,
  avoiding torchrun's terminate-workers handler; workers stop collectively.
  The flag is cleared only before child startup, avoiding a lost early signal.
  An output lock prevents duplicate wrapper launches. Partial completion exits
  75, never dependency-success. `CLUSTER_DEPTH_RUN_PLAN.md` and the RPL README
  document the new commands, resource/scientific differences and pending gates.
- Actual verification: **87 CPU tests pass**, including 74 inherited tests,
  eight DDP tests, one three-treatment loss/gradient equivalence test and four
  matrix/launcher tests. Local Gloo processes verify gradient averaging,
  multiple heads/unused parameters, parameter agreement, exact fresh-process
  CPU stochastic resume (including AdamW/BN/RNG/sampler), complete train/val
  coverage, changed-config rejection and incomplete-parent rejection. Preview
  arguments for all fifteen tasks parse correctly. Python compilation,
  shell syntax and `git diff --check` pass. The first sandboxed multiprocessing
  attempt could not open loopback/shared memory; the CPU-only suite then passed
  outside those restrictions. A find-unused graph-traversal warning in the toy
  all-heads-active case is nonfatal; real dynamic VoxelNet use remains untested.
- Reproduce tests from THIS branch's worktree (no data/GPU/cluster required):

  ```bash
  env PYTHONPATH=scripts:cluster_rpl CUDA_VISIBLE_DEVICES='' \
    MPLCONFIGDIR=/tmp/dd2414-ddp-cpu-test PYTHONWARNINGS=ignore \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/students2026/miniconda3/envs/bev/bin/python -m unittest \
    scripts.test_depth_continuation scripts.test_official_full_data \
    scripts.test_full_data_protocol scripts.test_fullsize_training \
    scripts.test_training_checkpoint scripts.test_multisweep_teacher \
    scripts.test_distributed_training scripts.test_depth_objective \
    cluster_rpl.test_cluster cluster_rpl.test_depth_study -q
  python cluster_rpl/depth_study.py --config cluster_rpl/depth_study.example.json
  bash -n cluster_rpl/depth_ddp_job.sh
  ```

- Limitations/next: CPU/Gloo tests and the earlier tiny CUDA probe do NOT
  establish full-model CUDA correctness, GPU deterministic resume, memory fit
  or speedup. Two RTX 3080 memories are not pooled; each rank holds its own
  teacher/student, and high-resolution Depth Anything may still OOM. No
  performance/resolution fallback is automatic. Confirm longer two-GPU site
  authorization, deploy only after the owner lifts this turn's no-deployment
  restriction, transfer/hash-check parent/teacher assets, extend the admission
  audit, and run bounded update-two/new-process-update-three pilots in EACH
  depth mode before formal five-epoch continuation. Automatic dispatch is not
  implemented; downstream remains single-GPU. Hidden-velocity depth branches
  remain deferred until their matching completed parent is available.

### Earlier checkpoint — authorized diagnostic, before the implementation above

2026-11-06 **temporary two-GPU diagnostic passed after permission confirmation**:
the owner explicitly confirmed administrator permission for the bounded test,
resolving the earlier missing-authorization condition for that test only. Real
Slurm job `1250320` allocated two RTX 3080 GPUs, four CPUs and 8 GiB on arwen,
with a three-minute limit. It completed in 16 seconds (`COMPLETED`, `0:0`),
printed `TWO_GPU_PROBE_PASSED`, and released both GPUs; own `squeue` was empty.
Both ranks had distinct GPU UUIDs and 9.64 GiB per card, finite native-BF16
compute, a correct NCCL all-reduce (1+2=3), and matching post-update parameter
sums in a tiny DDP example. This is actual GPU/communication evidence, not
merely scheduler admission, but is NOT full-model DDP or a speedup benchmark.

Changed files: `cluster_rpl/two_gpu_probe.py` retains the runtime diagnostic's
logic with an additional Slurm guard; `cluster_rpl/test_cluster.py` tests that
guard without GPUs; the README and depth plan distinguish the completed
temporary test from unfinished formal deployment. Reproduction: use the
two-process torch.distributed command in `CLUSTER_DEPTH_RUN_PLAN.md` only in
another specifically authorized, three-minute two-GPU allocation. Actual
runtime log: cluster project `.cache/rpl_two_gpu_probe-1250320.log`.

Checks: seven continuation CPU tests, 41 inherited training tests and 26
deployment CPU tests pass (74 total); Python compilation and `git diff --check`
pass. The GPU-tested cache copy and tracked copy differ only in descriptive
header and the tracked copy's extra allocation guard; the tracked version has
not itself consumed another GPU allocation. Next: adapt and validate the
research trainer for genuine DDP if formal two-GPU execution is authorized;
the temporary permission is not broadened to ongoing training. The depth
matrix/deployment gates below remain unfinished. Workstation code, queue and
staged PDF remain untouched. No formal cluster training is running.

Earlier 2026-11-06 rejected attempt (superseded by the authorized result above):
the owner requested a temporary real
two-card check. A synthetic CUDA/NCCL/tiny-DDP script was prepared and uploaded
only under ignored project `.cache/`. A proposed arwen allocation (two GPUs,
four CPUs, 8 GiB, three minutes) was rejected by automatic execution risk review
BEFORE the SSH/Slurm submission ran, citing the supplied site single-GPU rule
and lack of an administrator exception. No real job ID/allocation or GPU test
result was obtained in that attempt. Do not bypass a rejection through an alternate execution
path. The earlier `sbatch --test-only` result proves scheduling feasibility
only; current deployment remains partial. See `CLUSTER_DEPTH_RUN_PLAN.md`.

Changed files for this diagnostic outcome: this handoff and the depth plan;
no tracked training code changed. Reproduce the non-allocating evidence using
the documented `sbatch --test-only` command. The real test could not execute;
`git diff --check` passed. Next: establish site authorization for a real two-GPU
test, or retain single-GPU execution; existing workstation training is untouched.

2026-11-06 **isolated depth-study preparation only**: branch
`experiment/rpl-depth-10sweep-velocity-study`, based on `f9f333f`, is in a new
worktree. The owner approved using the completed velocity-present five-epoch
double-DINO model as a common parent for no-depth, radar-depth and radar-plus-
Depth-Anything-V2 continuation branches. All three add five equal-budget full
epochs, with new optimizer/sampler/schedule; the no-depth branch must also
continue training. This is not a comparison against the old 51.09% score with
less pretraining exposure. Hidden-velocity depth branches are deferred until
their completed parent is available. See `CLUSTER_DEPTH_RUN_PLAN.md`.

Changed files: `scripts/source_provenance.py` adds fail-closed parent validation;
`scripts/train_multisweep_teacher.py` adds opt-in
`--pretrain-initial-checkpoint` plus required `--pretrain-initial-sha256`, loads
all model tensors strictly, and records the fresh-phase initialization and
total pretraining exposure; `scripts/multisweep_teacher.py` and its caller add
explicit relative-depth teacher resolution while retaining the old default.
`scripts/test_depth_continuation.py` tests complete-parent requirements, matched
velocity/data/settings, relocation with preserved RGB identity, strict loading
for all three depth modes, and teacher resolution/no-gradient behavior.
`cluster_rpl/README.md` distinguishes the inherited runner from this unfinished
depth deployment. Official Simple-BEV defaults and the running workstation
checkout/service remain untouched; the staged user PDF is not included.

Reproduction from this worktree:

```bash
PYTHONPATH=scripts python -m unittest scripts.test_depth_continuation -v
PYTHONPATH=scripts python -m unittest scripts.test_official_full_data scripts.test_full_data_protocol scripts.test_fullsize_training scripts.test_training_checkpoint scripts.test_multisweep_teacher
PYTHONPATH=scripts python -m unittest discover -s cluster_rpl -p test_cluster.py
```

Actual tests: seven new CPU tests, 41 inherited training tests and 25 inherited
deployment tests passed (73 total); `git diff --check` passed. These are not
real GPU continuation/depth/DDP checks. Parent checkpoint SHA-256 was measured
as `ae9daf1124a03ee5a47f5fac68886334bcdd12dcbe5684c8a64167982c632520`.
The locally cached Depth Anything V2 Small weight was measured as
`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`,
with source commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.

Resource decision: after the single-GPU plan, the owner requested a real
two-GPU attempt and asked to ignore the manual's master-account single-GPU
rule. A non-allocating `sbatch --test-only` two-GPU check on arwen was accepted;
`squeue -u hongen` was empty before and after. The reported hypothetical job
number is not a submitted training job. The association query displayed
`defaultusr` / `normal` with blank MaxTRES/GrpTRES, not proof of administrator
permission. At that preparation stage, no administrator exception had been
confirmed and no actual allocation was attempted. The later authorized
temporary diagnostic is recorded at the top of this handoff.

Limitations/next: this commit is tested initialization groundwork, NOT a
complete cluster depth deployment. The three-treatment matrix, full-data audit
extensions, parent/teacher transfer, actual continuation/resume pilots and
persistent automatic dispatch remain unfinished. The cluster clone remains on
the earlier branch; no formal depth training is started. Clarify whether the
site permits two GPUs for this account, or retain single-GPU execution, before
locking the execution path. If permitted, add genuine DDP and verify effective
batch/loss normalization, data coverage, per-rank state and interruption safety;
do not silently treat single-card checkpoints as exact multi-card resumes.

2026-11-05 RPL deployment continuation: the owner has now authorized routine
cluster queries, project sync, personal environment/data preparation and the
approved single-GPU workflow without per-command prompts. Elevated privilege,
interference with existing jobs, other users' files or shared settings still
requires approval. The owner separately approved removal of exactly
`dd2414-hybrid`; preserve its package manifest and validate the replacement
before removing it. That removal is now complete (details below).

The cluster clone was verified at `07c0718ab8d0c261b00910978f7575a804e39b39`
on `experiment/rpl-full-data-double-distill`, with a clean worktree. Moria's
older Git lacks `switch`; this single-branch clone also needed a narrowly added
`remote.origin.fetch` rule for the new branch before `checkout --track` worked.
The existing old-branch fetch rule was preserved. These are repository-local
changes, not shared cluster configuration changes.

Read-only CPU allocations on arwen found about 3.4 TiB local free space. The
owner belongs to the `datasets` group, and personal `datasets/nuscenes`,
`dd2414_assets` and `dd2414_runs` directories were successfully created under
`/local_storage/users/hongen`. The source dataset measures 419,984,351,322 bytes
(about 391.14 GiB). A GPU query in a CPU-only allocation correctly exposed no
devices. A later real one-GPU check, job `1250200`, completed in 12 seconds:
RTX 3080, 9.64 GiB visible memory, capability 8.6, native BF16, finite convolution
forward/backward under torch 2.5.1+cu121. The allocation was released on exit.
This synthetic compatibility check is not the full-model training/resume pilot.
The workstation pretraining records about 6.67 GiB peak allocation, not a
guarantee that the complete RPL workload fits.

CPU-only mamba setup job `1250185` failed after 35 seconds, before a new
environment was registered: conda Pillow 9.4/libxcb conflicted with modern
libgl, and scipy 1.15.3 was not offered in that solve. The environment YAML is
revised to let mamba manage Python 3.10.14, pip and user-space libgl/libglib,
while pinned official CUDA 12.1 torch/torchvision and scientific wheels install
inside that environment. No version pin or model setting is loosened. The
base environment is untouched. Re-read submission manual attachment
`4299901b-39d6-4b12-884e-6169875b6752` in full: one GPU total for the master
account, explicit limits <=48 hours, salloc <=6 hours, no compute on moria,
no idle GPU placeholders, node-local data binding, no blanket cancellation.

Replacement setup job `1250190` completed in 8m48s. CPU validation job `1250197`
completed in 2m47s: `pip check`, actual training imports, frozen DINO ViT-S/14
forward (1x4x384 patch tokens), 25 deployment tests and 41 inherited CPU tests
all passed. PyYAML 6.0.3 was installed for the YAML contract test and is now
explicitly pinned in `cluster_rpl/environment.yml`; the regression contract
checks this dependency. Missing xFormers and matplotlib deprecation warnings
were nonfatal. Reproduction: use the two CPU-suite commands below, run
`python -m pip check`, and load the cached DINO teacher on CPU. Full output is
in `.cache/rpl_environment_check-1250197.log`; installation logs remain in
`.cache/rpl_setup-1250185.log` and `.cache/rpl_setup-1250190.log` on the cluster.

Before deleting the authorized old environment, own-job/process inspection
found only our CPU upload and no old-environment interpreter. Cleanup job
`1250199` completed in 31 seconds using `mamba env remove --prefix
/Midgard/home/hongen/miniforge3/envs/dd2414-hybrid --yes`; path absence and
the remaining base/new `dd2414` environment list were verified. Old manifests
are preserved in `.cache/env_backups/dd2414-hybrid.yml` and
`dd2414-hybrid-pip.txt`; the new validated environment is exported beside them
as `dd2414-validated.yml` and `dd2414-validated-pip.txt`. These are reconstruction
records, not an instant rollback of the removed environment. Cleanup log:
`.cache/rpl_environment_cleanup-1250199.log`. No other environment was removed.

CPU-only Slurm rsync dry-run works with
`srun --unbuffered`; the default-buffered route stalled its bounded test.
An actual 88,283,115-byte DINO weight transfer also passed the pinned SHA-256
check at about 12 MB/s. Full data upload is now started as the workstation user
service `dd2414-rpl-nuscenes-upload.service`, writing only arwen's personal
nuScenes directory. The remote rsync step requests one CPU, 4 GiB, zero GPUs,
47 hours, and releases its allocation on exit; no idle holder is used. Upload
logs are `.cache/rpl_nuscenes_upload.log` in this isolated worktree. Rsync has
partial-file recovery, a 50 MiB/s ceiling and three bounded service restart
attempts per hour; there is no `--delete`. The initial buffered dry-run job
`1250186` was naturally stopped by its five-minute time limit, not by a blanket
cancel. The receiver is job `1250189`; a restart may change that ID. Slurm is
17.11.10; `--overlap` is unsupported, so modern monitoring flags must not be
assumed. At the inspected snapshot, payload transfer reached about 12.58 GiB
of 391.14 GiB (3.22%), with zero service restarts. Rsync's displayed percentage
uses a changing scanned-file denominator and is not whole-dataset progress.
The official RGB checkpoint, pinned BEVCar source and DINO weight/Python tree
have separately arrived and passed their pinned hashes. Full dataset transfer,
the complete sensor-file audit, actual three-update/new-process-resume pilot
and formal training have not yet occurred. Next: verify this scoped commit on
GitHub, fast-forward the idle cluster clone, keep the persistent upload running,
then audit all data and run the real pilot. Refresh workstation task ownership
before any long formal run; do not stop or reorder its unchanged queue.

2026-11-05 **RPL deployment branch only**: the owner approved isolated cluster
preparation on `experiment/rpl-full-data-double-distill`, based on `3f8a328`.
All edits are in a separate worktree; the workstation baseline checkout,
systemd queue, protocol and staged PDF are unchanged. The owner initially
required separate approval for every cluster command; the continuation entry
above records the subsequent relaxation and current deployment evidence.
The earlier approved-session inspection found a clean cluster clone at
`/Midgard/home/hongen/projects/DD2414-Project`, still on the historical
`experiment/hybrid-teacher-10sweep-cluster` branch (`4a086d0`), mamba 2.1.1,
5.5 GiB of the 100 GiB home quota used, and no account jobs. These are snapshots,
not a claim of current node availability or completed cluster deployment.

Local source inventory at this preparation step contains three completed tasks:
the official-encoder initial camera probe, velocity-present five-epoch double
distillation, and its ten-epoch camera-BEV frozen probe. The velocity-present
fused-BEV probe is active, around eight of ten epochs in the inspected snapshot.
This supersedes the old "not yet training" statement below for that study.
Seven tasks are not started in that snapshot. RPL preparation prioritizes the
missing velocity-hidden group, respecting the pretraining dependency before
segmentation, object motion and the two probes. Completion snapshots must be
refreshed before actual launch; there is no cross-host reservation mechanism.

Changed files and rationale:

- `cluster_rpl/environment.yml`, `setup_environment.sh`: isolated mamba
  environment, pinned Python 3.10/torch 2.5.1/torchvision 0.20.1/numpy 1.23.5/
  Pillow 9.4/nuScenes 1.1.11; CUDA 12.1 is an explicit deployment difference
  from the workstation CUDA 12.4 build. Setup prints by default and refuses
  to overwrite an existing environment. OpenCV's package name matches the
  pinned SDK requirement (no duplicate full/headless cv2 providers); a user-
  environment libgl dependency avoids a system-level graphics install.
- `cluster_rpl/control.py`, `job.sh`, `config.example.json`: opt-in,
  dependency-aware reverse priority, explicit one-task/one-GPU Slurm submission,
  chosen node binding, no physical-GPU-0/22 GiB idle gate, no systemd/GitHub
  publisher, no automatic SSH/queue/requeue/cancel. An asset audit and a real
  finite-update/new-process-resume pilot are mandatory before formal admission.
  Early signals are forwarded only to the owned child. Partial work does not
  unlock dependent tasks. Output locking prevents concurrent writers per run.
- `cluster_rpl/check_assets.py`, `assets.json`: pinned RGB/DINO/BEVCar checks,
  official full split membership and every referenced sensor file's presence/
  nonzero size. The audit requires an allocated CPU context on the data node.
- `scripts/source_provenance.py`, `train_multisweep_teacher.py`: explicit
  downstream-only `--allow-source-relocation` accepts identical official RGB
  content at a new path, preserving hash/tensor-count/scope and strict model
  loading. Exact same-run optimizer/sampler/RNG resume validation is unchanged;
  old implementation signatures are not rewritten or bypassed.
- `scripts/fullsize_training.py`: opt-in allocated-GPU UUID telemetry supplied
  by the RPL runner; unset retains the existing default behavior.
- `cluster_rpl/test_cluster.py`, `README.md`: CPU regression contracts and
  step-by-step approval-aware deployment, data transfer, environment, audit,
  checkpoint/resume, monitoring and backup instructions.

Reproduction (from this branch, using the existing workstation environment):

```bash
PYTHONPATH=scripts python -m unittest discover -s cluster_rpl -p test_cluster.py -v
PYTHONPATH=scripts python -m unittest scripts.test_official_full_data scripts.test_full_data_protocol scripts.test_fullsize_training scripts.test_training_checkpoint scripts.test_multisweep_teacher -v
bash -n cluster_rpl/job.sh cluster_rpl/setup_environment.sh
python cluster_rpl/control.py plan
bash cluster_rpl/setup_environment.sh
```

Initial preparation tests: 24 new CPU tests and 41 inherited CPU tests passed
(65 total); the continuation above adds a 25th deployment contract (66 total).
Shell syntax, Python compilation, plan/setup previews and `git diff --check`
passed. A real source-run completion export validated the three completed
configurations and marked the active fused probe separately; its index is
runtime-only under `.cache/`, not committed. Tests involving pilot behavior
mock CUDA/child training: they are not real GPU updates on RPL.

Initial preparation limitations (superseded where verified above): the
environment solver/install, data node choice/free disk,
approved upload route, data/checkpoint transfer, allocated assets audit, actual
CUDA memory feasibility, pilot and formal cluster training have not run.
The cluster checkout has not yet switched to this branch. File existence checks
are not byte checksums of the full dataset. Native BF16 requires a suitable GPU;
if FP16 is necessary, lock and disclose the changed comparison conditions.
One-GPU account policy remains applicable despite an earlier two-GPU dry run.
There is no DDP or automatic multi-card memory pooling. The unchanged local
queue can later reach a task assigned to RPL; coordinate ownership before
launch instead of promising a distributed lock or silently stopping local work.

Current next steps and approval boundaries are in the continuation entry above
and `cluster_rpl/README.md`; the earlier per-command prompt rule is superseded.

2026-11-04 branch housekeeping: the owner withdrew the isolated
`experiment/official-rgb-double-distill-cluster` deployment. Its local branch
and clean worktree have been removed after saving a verified recovery bundle.
The matching GitHub branch has now also been deleted after the owner requested
an authentication retry; SSH succeeded and remote absence was verified. The
active local eleven-task plan, code, checkpoints and queue are unchanged.
See the cleanup entry below for the preserved recovery bundle.

2026-11-03 priority update: subset generation and numeric-scale/prefix training
selection are retired. The active manifest now contains only the full 28,130
train / 6,019 validation records, with hash
`dc39006dffbf036d5ea7728dad9348b822c78cd47cd02c226b638271c3420c5d`.
The owner has now approved the unchanged official 700-scene train / 150-scene
validation split. No extra holdout is created, and scores are called validation
results, not blind-test results. The 448x800, ten-sweep, official-RGB-encoder
warm-start study is deployed as `dd2414-official-full-data.service` and is
**waiting for the occupied GPU, not yet training**. The other compute process
(PID 3344413, about 20,204 MiB) is not interrupted. The persistent queue first
requires real finite-update/new-process-resume checks, then runs eleven tasks.
See `FULL_DATA_TEST_PROTOCOL.md`, `configs/full_data_official_protocol.json`,
and live `fullsize_baseline/runs/official_rgb_full_10sweep_20261103/pipeline.json`.
The admission audit is ready; it does not certify a passed CUDA pilot.
The historical progress statements below do not imply this new study is complete.

2026-10-05 trainval status: the Ubuntu host has full labeled `v1.0-trainval`;
mini has been removed. The official split is 700 train / 150 val scenes and
28,130 / 6,019 samples; all 2,631,083 referenced sensor files were present and
nonempty. Official BEVCar is pinned at commit `29cacda3` under `external/BEVCar`;
project probes import only its random-initialized `nets/voxelnet.py`, not the
full BEVCar model or checkpoint.

The exact prior BEVCar held-out protocol was reproduced with trainval: head
64-train / 16-val samples, 512 steps, three seeds and one sweep. Motion-only and
joint zero-velocity penalties were `+0.0768` and `+0.0804`; the head subset
covered only 2/1 train/val scenes and removing all radar improved motion loss,
so it is weak generalization evidence. A separate uniform 64/16-scene control
gave `+0.7980` and `+0.8602`, with correct radar outperforming no radar. See
`TRAINVAL_RUN_PLAN.md` and the 2026-10-05 work-log entry. These are bounded
held-out feature-loss probes, not all-frame training or downstream accuracy.
The scaling trainer provides all-frame streaming self-supervision. The
combined Hybrid Teacher implementation and RPL deployment live separately on
`experiment/hybrid-teacher-10sweep-cluster`; that branch is outside the current
baseline work scope. The baseline meeting follow-up is specified in
`MEETING_FOLLOWUP_DAY_PLAN.md`. The owner's expanded follow-up now supersedes
its pending order: see `MULTISWEEP_TEACHER_DAY_PLAN.md` for local double-DINO
pretraining, controlled five/ten-sweep inputs, actual depth-aware lifting,
frozen/unfrozen downstream comparisons and independent GT-object motion
readouts. The replacement queue has no idle, elapsed-time or 24-hour kill rule.
The old timed service was explicitly checkpoint-paused for engineering tests;
its already-started historical finetuning remains resumable in the new queue.

The owner has adopted a staged full-data radar scaling study. The frozen order
is streaming data, checkpoint/resume, wrong-scene evaluation, a throughput
pilot, then single-seed `64 -> 256 -> 1,024 -> 4,096 -> 28,130` training-scale
results evaluated on the same complete 6,019-frame validation split. Each
completed milestone must be committed and pushed before the next begins. See
`FULL_DATA_RADAR_SCALING_PLAN.md`; its dependence curve precedes multi-sweep or
hybrid-teacher expansion.

P1 is complete: the seed-125 manifest deterministically nests
64/256/1,024/4,096/full train subsets, covers 64/256/700/700/700 scenes, and
keeps all 700 train scenes disjoint from all 150 validation scenes. A real
512-batch one-sweep traversal retained zero CUDA allocation at every 64-batch
checkpoint. P2 is also complete: atomic periodic checkpoints restore model,
optimizer, AMP scaler, sampler position and all RNG states; a CUDA interrupted
resume matched uninterrupted training exactly. P3 is complete: the full val
wrong-radar map is a deterministic 6,019-token cross-scene bijection, and all
four input modes emit finite incremental per-frame records. P4 is complete: a
128-update online-DINO joint pilot stopped at update 64, resumed in a new
process at the exact sampler position and finished with bounded 3.574 GiB peak
allocation. Measured throughput locks the curve at 30,000 updates per scale
without a teacher-feature cache. P5 is complete: the 64-sample run performed
30,000 effective updates and evaluated all 6,019 validation frames in four
radar modes. Its semantic penalties are small, while its motion penalties are
large and positive; these establish radar dependence for the diagnostic motion
head, not downstream vehicle-segmentation utility. P6 is also complete under
the unchanged protocol: the 256-sample point lowers matched validation losses
and increases semantic and motion radar-intervention penalties relative to P5.
P7 is complete as well: at 1,024 samples, matched losses fall again and empty /
wrong-scene penalties strengthen substantially. P8 is now complete: the 4,096
sample point continues the monotonic increase in semantic and motion
wrong-scene penalties, with positive semantic penalties in all 150 validation
scenes. P9 is complete as well: the full 28,130-sample run covered every train
token, exercised a real process restart at update 1,000 and evaluated all 6,019
validation frames. Its semantic/motion wrong-scene penalties remain strong at
`+0.02470`/`+1.56690`, but are slightly below P8's `+0.02854`/`+1.59196`.
The single-seed curve therefore strengthens through 4,096 samples and then
plateaus or slightly recedes under the fixed 30,000-update budget; it is not a
monotonic full-range trend or a downstream utility claim. P10 is complete: a
paired 150-scene bootstrap places persistent semantic wrong-scene dependence
at 256 samples, while all motion penalties are positive from 64 samples. The
semantic full-minus-4,096 wrong-scene change is `-0.00383` with 95% interval
`[-0.00460, -0.00309]`; the motion change is `-0.02506` with interval
`[-0.06251, +0.01415]`. This supports saturation or a small late decline, not
indefinite growth. See `RADAR_SCALING_ANALYSIS.md`. P11 downstream IoU remains
the later utility boundary after the newly requested exposure control.

On 2026-10-08 the owner approved a separate epoch-matched follow-up without
changing the P9 architecture: independently train the deterministic 4,096
subset and all 28,130 samples for eight epochs, using identical optimized
training parameters. The new opt-in path under `fullsize_baseline/` supports
true multi-sample batches, epoch-native warmup/cosine AdamW training,
prefetch-safe exact resume, and live JSON/TensorBoard/PNG monitoring. Real
engineering pilots selected batch 5 with eight workers. The longer 256-update
pilot processed 1,280 unique samples at 12.27 samples/s, with 16.721 GiB peak
PyTorch allocation, 20,560/24,564 MiB total process memory at the final status
sample, finite losses/gradients, and zero AMP overflow retries. A synchronized
profile attributed only 8.3% of a stable step to DINO, so the roughly 49 GiB
teacher cache was rejected. The formal 4,096-sample run completed all eight
epochs and four full validations in 10,225.6 seconds with 6,560 finite updates
and zero AMP retries. From epoch 1 to 8, matched semantic/motion validation
loss improved from `0.33595/0.91730` to `0.30503/0.28907`; at epoch 8 the
empty-radar penalties were `+0.03868/+2.29595` and wrong-scene penalties were
`+0.02051/+2.12402`. The full-split run started as the persistent user service
`dd2414-fullsize-full.service` at 2026-10-08 20:37 CEST.

The teacher-requested velocity control was implemented as an unattended,
restart-aware sequence. It completed one backbone with raw radar velocity and
a matched backbone with the two BEVCar velocity input channels zeroed. At
epochs 1 and 8 it froze each backbone and trained an identical single-layer
vehicle-segmentation probe, reporting IoU, precision, recall, and F1 on all
6,019 validation frames. This directly tests explicit radar-velocity utility
under the existing frozen-camera architecture; it does not add camera SSL,
dual-level distillation, a hybrid teacher, soft guidance, or an end-to-end
downstream head.

That unattended control is complete. Both configurations finished eight
epochs, 45,008 updates and full validation at epochs 1/3/5/8. At epoch 8, the
raw-velocity model reached motion/semantic losses `0.0881/0.2739`; the model
trained with velocity channels zeroed reached `1.5684/0.2794`. The frozen
one-layer vehicle probes reached `4.962%` versus `4.848%` Vehicle IoU, only a
`+0.114` percentage-point raw-velocity gain. Precision remained about `5%`
with roughly 69% recall, exposing widespread false positives. Reproducible
meeting figures, including direct prediction-mask/GT/error-map comparisons,
are documented in `FULLSIZE_MEETING_VISUALS.md`.

On 2026-10-09 the owner explicitly advanced the legacy multi-sweep density
follow-up ahead of the previously planned dual-level-distillation study. The
epoch trainer, launcher, progress renderer, batch-equivalence audit and frozen
probe now support strict 1/5/10-sweep configurations in isolated run
directories. A real five-sweep two-sample audit observed 2,549/2,154 raw radar
points, 1,228/769 occupied voxels (below the 3,500 limit), and about 0.32 s of
history. A batch-5 BF16 smoke ran 15 finite updates with a real restart at
update 10, zero overflow retries and 16.723 GiB peak PyTorch allocation. This
is still the legacy frozen-random-camera architecture and is only a radar
density follow-up, not the requested final dual-level-distillation model. The
formal eight-epoch run subsequently completed all 45,008 updates and four
6,019-frame validations, as verified from its local summary on 2026-10-16.
Epoch-8 matched semantic/motion losses are `0.269462/0.107164`. Training took
8.86 hours and four-mode validations another 2.80 hours. No five-sweep frozen
probe summary was found, so the lower semantic loss does not yet establish
a segmentation improvement.

The results below this update are historical mini experiments unless an entry
explicitly says trainval.

The official Simple-BEV camera-only and camera-plus-radar pipelines run on
nuScenes v1.0-mini. A separate, opt-in radar point encoder has been integrated
with the camera BEV feature. Frozen DINOv2 targets have driven a four-sample
label-free student experiment using the **legacy radar path**. The new radar
encoder has passed camera-fusion integration tests but has **not** been trained
against the DINO target in that fused model. A separate adapter and official
BEVCar encoder passed a one-frame CUDA smoke test. Both radar encoders have
now been optimized **standalone** on the same four cached frozen-DINOv2
targets. A standalone 12-frame, two-scene validation check with three seeds
reduced target loss, but a cross-scene radar-swap control found little or no
radar-alignment effect. Further no-radar and measurement ablations found that
the target loss does not require radar measurements in either branch and does
not require radar at all for the BEVCar branch. The encoders are now compared under a matched supervised camera-fusion
budget; the larger BEVCar encoder shows a clearer radar-dependence gap
(see the 2026-09-28 work log entry). A separate **supervised**
Simple-BEV + BEVCar mini demo now shows a modest radar-dependent IoU gain on
adaptation-held-out frames using box-derived labels; it does not solve the
self-supervised objective. A switchable **image-level dense semantic
distillation** (`--image-distill` in `semantic_distill_smoke.py`) now distills
the pre-projection image features against the frozen DINOv2 patch features
alongside the existing radar-guided fusion term; on a single-sample smoke the
image term converges cleanly and no longer drags the fusion loss (see the
2026-09-28 work log entry).
An opt-in motion head now exists in the motion-target probe scripts (2026-09-29); no full-dataset self-supervised evaluation exists.
The three-seed 64-train / 16-held-out dual-teacher probe now completes with the
vectorized target builder. It improves held-out fusion loss over its 4-sample
predecessor, but every no-radar fusion penalty remains small (0.0086--0.0172),
so it still does not establish strong radar dependence (see the 2026-09-30 work
log entry).
The matched 1/5/10-sweep 64-sample comparison found the best held-out fusion
loss at 10 sweeps with the radar-weighted dual teacher, but only a small
no-radar penalty (+0.0196); multi-sweep is a useful input-density control, not
proof of strong radar dependence (see the 2026-09-30 work log entry).

The collaboration branch is `dd2414-mini-baseline`. Each completed,
project-scoped change is pushed there so teammates can follow the work. Check
the GitHub PR state before assuming that its commits are on `main`. Merge only
after the bounded feature and its checks are complete and the team has reviewed
the PR; the routine per-task push does not merge anything.

## Data and pretrained weights

| Component | Exact choice and provenance | Local role |
| --- | --- | --- |
| nuScenes (current host) | Official `v1.0-trainval` metadata and ten blob archives; standard layout under `${HOME}/datasets/nuscenes` | 850 scenes / 34,149 samples; 28,130 train and 6,019 val; file audit passed; bounded BEVCar VoxelNet trainval probes completed |
| nuScenes (historical results) | Official `v1.0-mini` archive | 10 scenes / 404 samples; 323 train and 81 val; removed from this host after trainval replacement |
| Simple-BEV | Official upstream [aharley/simple_bev](https://github.com/aharley/simple_bev) base commit `be46f0ef71960c233341852f3d9bc3677558ab6d` | Official camera and camera+radar checkpoints for baseline inference and supervised sanity checks |
| DINOv2 teacher | Meta's pretrained [`dinov2_vits14`](https://github.com/facebookresearch/dinov2#pretrained-models) ViT-S/14 backbone, **without registers or a task head**; weight file [`dinov2_vits14_pretrain.pth`](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | Extracts 384-channel patch features; `eval()` and `requires_grad_(False)` keep every teacher parameter frozen |
| New radar point encoder | `nets/radar_encoder.py`, randomly initialized in the integration smoke test | Converts seven numeric radar fields to a `(B,64,200,200)` BEV feature |

The DINO code loads from a local PyTorch Hub cache if present, otherwise from
`facebookresearch/dinov2`. **We did not fine-tune DINOv2 on nuScenes.** The
adaptation is geometric: camera calibration and measured radar range place
frozen image features into soft BEV target regions. The trainable student is
Simple-BEV plus a semantic projection head. In the four-sample experiment, the
student starts without a supervised BEV checkpoint, but its original radar
input path is enabled. Distinguish this from the later *new* radar encoder.

Datasets, pretrained weights, official checkpoints, generated student
checkpoints, logs, and raw teacher caches are local dependencies excluded from
Git; teammates must obtain them separately. See `SETUP_LOCAL.md`.

## Implemented pipeline and evidence

```text
six cameras -> calibrated camera BEV -------------------------+
                                                          fusion -> shared BEV
radar points -> new 7-field encoder -> 64-channel radar BEV --+

separate label-free experiment:
six cameras -> frozen DINOv2 patch features --+
radar range + calibration --------------------+-> soft BEV targets
legacy-radar Simple-BEV student ----------------> cosine feature loss
```

| Check | Reproduce | Observed result | Interpretation |
| --- | --- | --- | --- |
| Official camera/radar demo | `./scripts/run_meeting_demo.sh` | One mini frame displayed with both predictions and radar positions | Visual coordinate sanity check, not a benchmark |
| Official fixed subset | `./scripts/run_mini_subset_eval.sh` | Mean IoU 0.121 camera vs. 0.291 camera+radar on 10 fixed validation samples; radar higher on 10/10 | Small-subset official-checkpoint comparison; checkpoints were separately trained |
| Supervised loop | `./scripts/run_baseline_overfit.sh` | Four training-sample IoU 0.273 -> 0.803 | Uses human-box-derived labels; overfit check only |
| DINO target geometry | `./scripts/run_dinov2_bev_demo.sh` | Frozen teacher output `(6,384,14,24)`; radar-anchored local BEV targets | No box-derived target/loss |
| Label-free student | `./scripts/run_dinov2_mini4.sh` | Four-sample mean DINO cosine loss 1.023 -> 0.196 over 60 updates | Trains the legacy-radar student, **not** the new radar encoder; memorization only |
| New radar encoder | `./scripts/run_radar_point_encoder_test.sh` | 403 returns -> 125 quality/ROI points -> `(1,64,200,200)` | Point transform, pooling, empty input, order invariance, and gradient checks pass |
| New fusion integration | `./scripts/run_radar_fusion_test.sh` | Fused `(1,128,200,200)`; gradient norm 0.031452; empty radar and checkpoint reload pass; peak 3.440 GiB | Random-weight engineering check; no accuracy claim |
| BEVCar voxel input | `bash scripts/run_bevcar_voxel_adapter_test.sh` | 403 returns -> 251 in range -> 243 voxels; optional quality filter gives 125 points/123 voxels; zero BEV-cell mismatches | Upstream seventh feature verified as valid-point mask; pretrained-checkpoint compatibility not established |
| Official BEVCar radar encoder | `bash scripts/run_bevcar_encoder_smoke.sh --device cuda` with external BEVCar checkout | 403 returns -> 251 in range -> 243 voxels -> `(1,128,200,200)`; finite nonzero gradients; peak allocated CUDA 0.729 GiB | Isolated random-weight encoder only; no camera fusion, training, accuracy, or checkpoint compatibility claim |
| Radar-only frozen-DINOv2 comparison | `bash scripts/run_radar_dino_tiny.sh --samples 4 --steps 60` with cached targets and external BEVCar | Same-frame cosine loss: light 0.986 -> 0.890, BEVCar 1.002 -> 0.376; finite encoder gradients | Four training frames only; different capacity/filtering; not held-out accuracy or camera-radar fusion |
| Radar-only mini validation | `bash scripts/run_radar_dino_tiny.sh --samples 4 --steps 60 --heldout-samples 12 --seed-list 125,126,127` | 12 unseen frames, 2 scenes, 3 seeds: loss light 0.979 -> 0.903, BEVCar 0.995 -> 0.493; all 36 seed-frame losses fell | Cross-scene radar swap: light 0.909, BEVCar 0.490, versus aligned 0.903/0.493; no robust BEVCar radar-alignment evidence |
| Fixed-model radar ablation | Add `--diagnose-radar` to the validation command | Correct/no-radar loss light 0.903/0.911; BEVCar 0.493/0.489; zeroing or mixing radar numeric values barely changes either | Current DINO target loss does not prove use of radar measurements; BEVCar needs no radar to retain its low loss |
| Supervised BEVCar fusion demo | `bash scripts/run_bevcar_supervised_mini.sh --steps 80 --seed 125` and repeat seeds 126/127, then `python scripts/summarize_bevcar_supervised.py` | Mean 12-frame IoU across 3 seeds: correct radar 0.180, empty 0.166, wrong scene 0.171; SVFE/CML gradients nonzero | Human-box-derived labels; 4-frame adaptation, camera/decoder pretrained on broader nuScenes; not self-supervised or matched-budget benchmark |

Selected visual results are under `artifacts/`. `PROJECT_PROGRESS.md` explains
each plot and its caveats. The official legacy radar checkpoint still loads
after the new opt-in path was added.

## Reproduction notes

On the current Ubuntu host use `TRAINVAL_RUN_PLAN.md` and
`bash scripts/run_trainval.sh dual-teacher` (preview only). The following mini
commands are historical and need a separate mini installation.

1. Follow `SETUP_LOCAL.md` to install the `simplebev` environment, obtain
   nuScenes mini, and obtain the official Simple-BEV checkpoints. Shell scripts
   accept `CONDA_ROOT`, `CONDA_ENV`, `NUSCENES_ROOT`, and `OUTPUT_DIR` overrides.
2. Start with `./scripts/check_environment.sh` and
   `./scripts/smoke_test_mini.sh --data-only`.
3. Run the bounded scripts in the table above for the specific pipeline under
   investigation. They are not a single end-to-end benchmark command.
4. Check `git status`, branch/PR state, and test outputs before reporting a
   result. Small-sample training curves must not be described as validation.

## Agent handoff workflow

`AGENTS.md` gives repository-wide instructions to Codex. The project-local
`.codex/hooks.json` uses a read-only `Stop` hook to notice uncommitted local
changes **or committed work ahead of the upstream feature branch** and ask for
one final handoff/push review. On each machine, trust the
project's `.codex/` configuration and review the hook with Codex `/hooks`;
untrusted hooks are skipped. The hook never performs a Git write or contacts
GitHub. It remains quiet when the worktree is clean and there are no unpushed
commits. It does not turn a read-only question into permission to edit or push.

After a material code or experiment change, the responsible agent should:

1. update this file's `Current state` and append one dated work-log entry;
2. list the exact changed paths, test command and observed result, limitations,
   and next action;
3. inspect `git status`, stage only in-scope files, commit with an English
   message, push the collaboration branch as the default completion step, and
   verify the remote branch head; and
4. give teammates the commit/PR link and any blocked or omitted step.

The agent should do this **before** declaring the task complete; a lifecycle
hook is a fallback reminder, not a guarantee that interrupted or offline work
can be published.

Routine pushes to the feature branch are already authorized by the project
owner; another conversational confirmation is unnecessary. Codex may still
need to follow an environment-level network or filesystem approval. This
workflow does not authorize automatic `main` merges, force-pushes, or uploading
unrelated local work.

## Next bounded task

The raw-velocity and velocity-channels-zeroed eight-epoch runs and their frozen
one-layer vehicle probes are complete. Treat them as the legacy frozen-camera
velocity control, not the requested final dual-level-distillation model. Audit
probe convergence and frozen-backbone integrity before starting another full
run; the completed meeting figures already document the absolute IoU, raw-
velocity difference, radar interventions, and false-positive failure mode.

The subsequent primary experiment is a matched one-sweep comparison of the
camera-only configuration, camera plus radar without velocity, and camera plus
radar with velocity. Its shared objective must use DINO at both the trainable
camera encoder and the fused BEV, with the BEV term restricted to radar-
supported regions. Keep any motion-head objective as a separately reported
ablation. Only after all three configurations complete pretraining, frozen
linear probing, and qualitative evaluation should radar history change. Then
run a strict 1/5/10-sweep follow-up, beginning with a 5-sweep smoke and
advancing to 10 sweeps only after resource and alignment checks. Incomplete
long runs are acceptable, but must retain resumable checkpoints and must not
be reported as completed results.

Keep the pinned official BEVCar VoxelNet and the project voxel adapter
unchanged across matched radar configurations. Treat the earlier supervised
BEVCar result as a small positive radar-use control, not a solution to DINOv2
insensitivity.
Investigate why the self-supervised target loss ignores radar removal and
measurements; test output-map sensitivity and add an objective/control in
which matched radar beats empty and mismatched radar. Any architecture claim
requires matched data/update budgets and a larger genuinely unseen split.
Resolve capacity, feature/filter, frame, and memory differences.
Do not use BEVCar's supervised checkpoint for a label-free claim. The motion
head now exists only in the opt-in motion-target probe scripts (2026-09-29).

## Work log and update template

Append a dated entry here for each material repository-changing task. Keep the
entry concise and factual: changed paths, reason, exact command(s), observed
result, limitation, next action, commit/PR link or push blocker. Update the
`Current state` section when a milestone changes. Do not duplicate entire
chat transcripts, secrets, or unreviewed generated data.

### 2026-10-16 — Repair the legacy-probe watchdog compatibility

- Changed `scripts/run_baseline_followup_day.py`,
  `scripts/test_baseline_followup.py`, and added
  `scripts/resume_baseline_followup_after_current.sh`. A live status check found
  the five-sweep frozen probe terminated after 1,801.9 seconds at
  28,000/28,130 train samples. Its flushed logs and status showed finite,
  advancing training, not a stalled CUDA job.
- Cause: the legacy probe does not emit `updated_unix_time`. The queue used
  the attempt start time as its only fallback and incorrectly triggered the
  1,800-second stale-progress watchdog. Heartbeats now also accept the mtime
  of the atomic status file and flushed console log, with attempt start as a
  floor so old files cannot count as new progress. A resumed queue explicitly
  resets its display state to running.
- Recovery: preserve the active raw/zero finetuning sequence. A separate
  persistent waiting service will run the corrected queue after the current
  service exits. It reuses completed summaries and the same absolute study
  deadline, then retries the five-sweep probe and remaining optional work.
  The terminated legacy probe had not saved an epoch checkpoint and therefore
  needs a fresh probe epoch; no backbone pretraining is repeated.
- Tests: `PYTHONPATH=scripts MPLCONFIGDIR=/tmp/dd2414-mpl
  $HOME/miniconda3/envs/bev/bin/python -m unittest scripts.test_baseline_followup
  scripts.test_fullsize_training scripts.test_fullsize_linear_probe
  scripts.test_training_checkpoint` passed all 24 tests, including legacy
  status-without-timestamp and stale-JSON/recent-log regression cases. Python
  compilation, shell syntax for the waiting wrapper, and `git diff --check`
  passed. No model, objective, data, epoch, batch, or downstream resume hash
  was changed.
- Verified results so far: both official references evaluated all 6,019 frames:
  RGB Vehicle IoU `47.247%`; RGB plus five-sweep metaradar `55.495%`.
  Raw-velocity full finetuning remains active with finite losses, batch five,
  about 8.65 samples/s and observed 100% GPU utilization. Its own first
  full-validation score is still pending.
- Recovery service verified: `dd2414-baseline-followup-recovery.service` is
  waiting (PID 2245100); the primary PID remains 2231565 and its GPU child was
  not interrupted. The corrected probe retry will not run until the primary
  sequence finishes. Review the eventual
  complete/partial status and full-validation scores, not the old training-only
  probe status.

### 2026-10-16 — Execute the unattended baseline meeting follow-up

- Authorization: the owner approved proceeding through the day without
  intermediate confirmations. All work is on `dd2414-mini-baseline`; the
  cluster branch remains unchanged.
- Changes: added `scripts/baseline_followup.py`,
  `train_baseline_downstream.py`, `eval_pretrained_baseline.py`,
  `run_baseline_followup_day.py`, `start_baseline_followup_day.sh`,
  `report_baseline_followup.py`, `monitor_baseline_followup.py`,
  `publish_baseline_followup.py`, `audit_baseline_followup_targets.py`, and
  `test_baseline_followup.py`; updated `MEETING_FOLLOWUP_DAY_PLAN.md`.
  All historical entrypoints and the cluster implementation are preserved.
- Behavior: strict historical checkpoint/source loading; complete manifest
  token-order checks; full student supervised decoder finetuning with the
  original loss components; frozen/unfrozen identical-head modes; sample-
  acknowledged periodic checkpoints containing model/buffers, optimizer,
  scaler, sampler and RNG. Resume rejects changed implementation or settings.
  Metrics use official global valid-cell IoU with per-frame counts, fixed-token
  GT/prediction/error panels, TensorBoard and live status. The queue locks the
  common epoch budget from a 200-update throughput pilot, runs the two official
  references, missing five-sweep probe, and matched raw/zero finetuning pair,
  then optional same-head/depth/annotation diagnostics if time permits.
- Persistence: local user service, per-stage retries, absolute 24-hour
  deadline, stale-progress checks, checkpointed stop, isolated logs and atomic
  stage summaries. Each stage generates compact reports and attempts a scoped
  baseline commit/push with a handoff update; Git failures are recorded without
  cancelling training. Explicit-path commits exclude the staged project PDF.
- Real host checks: outside the tool sandbox, CUDA sees the idle RTX 3090 Ti
  with 24,564 MiB. Both official archives downloaded successfully and strict
  four-frame CUDA inference passed at 448x800, including five-sweep metaradar.
  RGB/radar checkpoint SHA-256 values are
  `ccaaf31b7590b73c55d67de0e70c1b99f705bedc2f773cb7124fd0344892749e` /
  `a08475daf65f50e8df84a69a8372c626191bba4437133413b069f5aed4f03bad`.
  An eight-train/four-val finetuning smoke made finite real updates, stopped
  at update two, resumed at update three in another process, and completed
  update four and validation. Batch two peaked near 7.41 GiB allocated.
- Numerical limitation: the interrupted and independent continuous CUDA
  runs had identical sampler positions but were not bitwise equivalent;
  maximum floating parameter difference was about `2.79e-4`, and smoke IoU
  differed by about `9.90e-5`. CUDA camera projection backward uses
  `grid_sample`, which does not guarantee deterministic backward. Exact
  checkpoint-state restoration must not be described as bitwise trajectory
  equivalence. The CPU optimizer/sampler/RNG resume test is exact.
- Reproduction: `PYTHONPATH=scripts MPLCONFIGDIR=/tmp/dd2414-mpl
  $HOME/miniconda3/envs/bev/bin/python -m unittest scripts.test_baseline_followup
  scripts.test_fullsize_training scripts.test_fullsize_linear_probe
  scripts.test_training_checkpoint`; `python -m py_compile` on the added
  Python files; `bash -n scripts/start_baseline_followup_day.sh`;
  `bash scripts/start_baseline_followup_day.sh` on the actual CUDA host.
- Checks: 22 CPU tests, shell syntax, Python compilation, and whitespace
  checks pass. Real bounded CUDA checks above passed; they are engineering
  checks, not full-validation research results. The previous plan's historical
  224x400 input claim was corrected: these checkpoint/probe loaders use
  112x192; official reference inference uses 448x800.
- Remaining limitations: one seed, unequal official/custom resolution and
  pretraining history; two finetuning recipes are not a pure test-time radar
  ablation. Depth outputs are relative illustrations, not metric validation;
  object-motion work is annotation coverage only, not a trained readout.
  Actual full-data progress/results are recorded in
  `fullsize_baseline/runs/meeting_followup_20261016/pipeline.json` and its
  child summaries, not inferred from this implementation entry.
- Launch: `dd2414-baseline-followup-day.service` is active on the actual host.
  The batch-five pilot reached 25 finite updates at about 7.5 samples/s,
  100% observed GPU utilization, 17.65 GiB peak allocation and approximately
  19.3 GiB total process memory. The terminal monitor also reads the pilot's
  live status while the parent queue is waiting for its result.
- Next: verify the complete 200-update pilot and first production validation,
  then allow sequential execution and review the final complete/partial
  status, full-val metrics, and prediction/GT masks.

### 2026-10-16 — Meeting-driven 24-hour baseline follow-up plan

- Change: added `MEETING_FOLLOWUP_DAY_PLAN.md` and refreshed this handoff's
  current baseline status. Switched back to `dd2414-mini-baseline` at the
  owner's request; the Hybrid Teacher cluster branch remains unchanged.
- Rationale: prioritize the teacher's official pretrained segmentation
  reference and unfrozen downstream evaluation before adding depth training.
  Complete the missing five-sweep frozen probe instead of repeating finished
  pretraining. Distinguish radar-cell velocity reconstruction from independent
  GT-object motion evaluation, and decoder capacity from a pure freeze effect.
- Evidence: inspected the supplied edited meeting transcript and supplement,
  baseline loaders/evaluators/trainers, official model download scripts,
  upstream Simple-BEV and Depth Anything V2 documentation, and local summaries.
  Five-sweep pretraining is complete at eight epochs; its matched semantic
  loss is `0.269462`, but no matching frozen probe summary was found. The
  official evaluator accumulates global intersections/unions; mean per-frame
  IoU must not be called the official primary aggregation.
- Reproduce planning checks: `git diff --check`; read
  `fullsize_baseline/runs/full28130_seed125/summary.json`,
  `fullsize_baseline/runs/full28130_zero_velocity_seed125/summary.json`,
  `fullsize_baseline/runs/full28130_sweeps5_seed125/summary.json`, and
  `fullsize_baseline/probes/velocity_full_epoch008/summary.json`; inspect
  `eval_nuscenes.py` lines computing `intersection`/`union` and the full-split
  accumulation. Verify target branch with `git branch --show-current`.
- Actual checks: full/zero/five-sweep summaries all report `complete`, eight
  epochs, and 6,019 validation frames; measured historical times were used for
  the plan. Documentation whitespace and referenced-path checks passed.
  `git fetch origin dd2414-mini-baseline` found no upstream divergence.
- Limitations: this task only writes the work plan. No finetuning runner,
  supervised finetuning, depth loss, object-motion readout, GPU smoke, or new
  score was implemented or run. This execution environment cannot communicate
  with an NVIDIA driver. Official pretrained weights were not found under
  the expected checkpoint tree. The 24-hour slots and proposed three-epoch
  finetuning schedule require a real throughput pilot; fallback must keep
  paired epoch budgets equal.
- Next: implement the opt-in baseline downstream/queue programs, pass the
  actual GPU/resume gates, and execute the ordered schedule. Preserve the
  unrelated staged `DD2414_Project_Plan.pdf`; commit only this plan/handoff.

### 2026-10-09 — Five-sweep full-data support and real CUDA gates

- Change: generalized the epoch trainer, launcher, status renderer,
  batch-equivalence audit and frozen linear probe from the historical one-sweep
  input to explicit 1/5/10-sweep configurations. Run names include their sweep
  count, so the formal five-sweep output is isolated at
  `fullsize_baseline/runs/full28130_sweeps5_seed125`. The model, losses,
  optimizer, seed, split, eight-epoch budget and pinned VoxelNet are unchanged.
- Reason: the owner requested starting the multi-sweep experiment now. This
  intentionally advances the legacy density control ahead of the planned
  dual-level study; it must not be presented as the final architecture.
- Reproduce static checks: `git diff --check`; `$HOME/miniconda3/envs/bev/bin/python
  -m py_compile scripts/train_fullsize_baseline.py
  scripts/train_fullsize_linear_probe.py scripts/render_fullsize_progress.py
  scripts/check_fullsize_batch_equivalence.py`; `bash -n
  scripts/run_fullsize_baseline.sh`; and `PYTHONPATH=scripts
  $HOME/miniconda3/envs/bev/bin/python -m unittest
  scripts.test_fullsize_training scripts.test_fullsize_linear_probe`.
- Reproduce real-data gate: `$HOME/miniconda3/envs/bev/bin/python
  scripts/check_fullsize_batch_equivalence.py --data-root
  $HOME/datasets/nuscenes --bevcar-source external/BEVCar --manifest
  configs/radar_scaling_manifest_seed125.json --output
  fullsize_baseline/smoke/multisweep5_batch_equivalence.json --nsweeps 5`.
  Then run `NSWEEPS=5 RUN_DIR=$PWD/fullsize_baseline/smoke/multisweep5_10steps
  bash scripts/run_fullsize_baseline.sh --execute --stop-after-steps 10
  --checkpoint-every-steps 5 --print-every-steps 1 --validation-epochs 8` and
  repeat with `--resume --stop-after-steps 15`.
- Actual result: all 10 CPU tests, compilation, shell syntax and diff checks
  passed. Two real five-sweep frames contained 2,549/2,154 raw points,
  1,228/769 occupied voxels and maximum time lags 0.325/0.320 s. Batched versus
  per-sample semantic/motion outputs stayed inside the existing tolerance. The
  trainer completed 15 finite updates and resumed exactly from update 10;
  batch 5 peaked at 16.723 GiB, reported zero BF16 overflow retries, and
  reached about 4.8 samples/s after startup.
- Limitations: the audit covers two frames and the optimization smoke 75 sample
  exposures; the adapter still fails closed if any later sample exceeds 3,500
  occupied voxels. Five sweeps add older transformed points but no temporal
  attention. The camera encoder remains random and frozen, so this experiment
  measures radar-density effects within the legacy baseline only.
- Commit and launch: tested support was pushed as `721615f`. The formal run
  first completed 100 updates in
  `fullsize_baseline/runs/full28130_sweeps5_seed125` to establish a durable
  checkpoint, then `dd2414-fullsize-5sweep.service` resumed it with
  `Restart=on-failure`. The first post-resume status reached update 150,
  750 sample exposures, loss/semantic/motion `0.7836/0.4632/0.6407`, about
  9.03 samples/s, 63% sampled GPU utilization and 20,606/24,564 MiB process
  memory. Monitor with `watch -n 10 $HOME/miniconda3/envs/bev/bin/python
  scripts/render_fullsize_progress.py --run-dir
  fullsize_baseline/runs/full28130_sweeps5_seed125 --no-plot`, or inspect
  `systemctl --user status dd2414-fullsize-5sweep.service` and
  `journalctl --user -u dd2414-fullsize-5sweep.service -f`.
- Next: let the five-sweep run complete its scheduled full validations at
  epochs 1/3/5/8, then run the identical frozen linear probe and compare with
  the completed one-sweep raw-velocity checkpoint. Test ten sweeps only after
  the five-sweep result is reviewed.

### 2026-10-09 — Final full-data results and meeting visualizations

- Experiment completion: the persistent controller completed both eight-epoch
  full-data configurations, all epoch 1/3/5/8 validations, both epoch-1 and
  epoch-8 frozen probes, and the comparison reports. Each backbone processed
  225,040 sample exposures in 45,008 optimizer updates with zero AMP overflow
  retries and about 16.72 GiB peak PyTorch allocation.
- Result: at epoch 8, raw velocity reduced held-out motion loss from `1.5684`
  to `0.0881` and semantic loss from `0.2794` to `0.2739`. The downstream
  Vehicle IoU changed only from `4.848%` to `4.962%` (`+0.114` percentage
  points). Raw-velocity precision/recall/F1 were `5.074%/69.185%/9.455%`;
  false-positive cells outnumbered true-positive cells by about `18.7x`.
- Change: added `scripts/render_fullsize_meeting_results.py`,
  `FULLSIZE_MEETING_VISUALS.md`, five reviewed PNG figures, and a JSON
  provenance record. The direct mask figure compares GT, both hard prediction
  masks, and TP/FP/FN maps on four deterministic scene-diverse validation
  samples. Selection uses only scene position and GT non-emptiness, never model
  performance. The remaining figures cover validation curves, radar input
  interventions, linear-probe metrics/failure counts, and a 16:9 overview.
  Active planning and run documentation now uses full configuration names
  instead of the ambiguous `V+`/`V0` and `A`/`B`/`C` shorthand.
- Reproduce: `MPLCONFIGDIR=/tmp/dd2414-matplotlib PYTHONPATH=scripts
  $HOME/miniconda3/envs/bev/bin/python
  scripts/render_fullsize_meeting_results.py`.
- Verification: Python compilation and `git diff --check` passed. The real CUDA
  render ended with `FULLSIZE_MEETING_VISUALS_OK`; all five PNGs passed Pillow
  decoding and were visually inspected for labels, orientation, values, and
  error colors. The metadata contains four distinct validation scenes and the
  exact sample/scene tokens, GT-cell counts, radar-return counts and per-sample
  IoUs.
- Limitations: this is one seed, one radar sweep, raw rather than compensated
  velocity, and binary vehicle IoU rather than multi-class mIoU. The probe ran
  for one epoch. The camera ResNet is random-initialized and frozen. The four
  displayed frames illustrate the aggregate failure but do not replace metrics
  over all 6,019 validation frames.
- Next: use these figures as the controlled legacy-baseline report. Audit probe
  convergence and frozen-weight integrity, then implement the one-sweep
  trainable-camera dual-level comparison of camera-only, radar without
  velocity, and radar with velocity before the planned 5/10-sweep follow-up.

### 2026-10-09 — Low-IoU diagnosis and future-plan slide

- Change: added `scripts/render_fullsize_future_plan_slide.py` and the 16:9
  meeting slide `artifacts/fullsize_low_iou_future_plan.png`, with usage notes
  in `FULLSIZE_MEETING_VISUALS.md`. The slide separates the measured metric
  failure (false positives dominate the IoU denominator) from plausible but
  not yet causally proven contributors.
- Rationale: the meeting package needed one concise page that explains why the
  completed frozen-probe Vehicle IoU is low and turns each limitation into an
  ordered, controlled future experiment.
- Reproduce: `MPLCONFIGDIR=/tmp/dd2414-matplotlib
  $HOME/miniconda3/envs/bev/bin/python
  scripts/render_fullsize_future_plan_slide.py`.
- Verification: the renderer reads the epoch-8 comparison JSON rather than
  copying headline metrics by hand. Python compilation, a real render, PNG
  decoding, visual inspection and `git diff --check` passed.
- Limitations: only the `4.96%` IoU, `5.07%` precision, `69.2%` recall and
  `18.7x` false-positive/true-positive ratio are direct aggregate findings.
  The frozen random camera encoder, objective/readout mismatch, limited probe
  audit and one-sweep/one-seed scope are hypotheses to test, not established
  causal attributions.
- Next: first audit probe convergence and the precision-recall/threshold curve;
  then run the requested trainable-camera dual-level comparison before changing
  radar density to 5 or 10 sweeps.

### 2026-10-09 — Post-main 5/10-sweep follow-up added to the plan

- Change: updated `SELF_SUPERVISED_EXTENSION_PLAN.md` and this handoff to place
  a matched multi-sweep experiment after the one-sweep dual-level comparison
  of camera-only, radar without velocity, and radar with velocity. The order is
  1-sweep reference,
  5-sweep smoke and formal run, then 10-sweep smoke and formal run if time and
  resources permit. The camera-only checkpoint is reusable because it has no
  radar-sweep input; the preferred controlled matrix pairs radar-no-velocity
  and radar-with-velocity at each sweep count.
- Rationale: the supervisor also requested testing denser temporal radar, but
  introducing it before the main three-configuration result would confound
  radar modality, velocity, and point-density effects.
- Reproduction: no experiment command is claimed yet. Before a formal run,
  execute loader/coordinate/velocity/voxel-capacity, memory, forward/backward,
  and resume smokes at the selected sweep count, then use the same split,
  epochs, batch/effective batch, seed, losses, probe, and validation samples as
  the locked one-sweep reference.
- Verification: documentation-only change; `git diff --check` passed. No
  training process or current one-sweep configuration was changed.
- Limitations: past 64-sample 1/5/10-sweep probes are engineering evidence,
  not a substitute for the future full-data dual-level comparison. More
  sweeps add older points but no temporal camera model, and may exceed the
  current 3,500-voxel cap or require a smaller batch. An unfinished run is only
  a progress artifact, not a result.
- Next: complete and validate the one-sweep dual-level camera-only, radar-
  without-velocity, and radar-with-velocity study, and start the 5-sweep smoke
  only after those deliverables are reviewed.

### 2026-10-09 — Unattended raw-velocity versus velocity-zeroed control

- Change: added a locked `full|zero` radar-velocity input mode to the epoch
  trainer and all its validation interventions. `zero` clones the BEVCar voxel
  features and zeros only channels 4:6, preserving positions, RCS, masks,
  shapes, model parameter count, supervised targets, losses, and every other
  formal training setting. The legacy `full` run configuration stays byte-for-
  byte compatible with its active checkpoint. Added an epoch-boundary clean
  stop so the velocity-zeroed run can expose an epoch-1 checkpoint before
  exact resume.
- Change: added `scripts/train_fullsize_linear_probe.py`, which freezes the
  complete checkpoint backbone and trains exactly one `1x1` convolution on
  the box-derived vehicle mask. It records balanced BCE plus validation vehicle
  IoU, precision, recall, F1, and confusion counts. Added a comparison writer,
  a compact progress renderer, CPU tests, and a persistent restart-aware driver
  that runs both required smokes before formal velocity-zeroed/probe work.
- Reproduce implementation checks: `PYTHONPATH=scripts
  $HOME/miniconda3/envs/bev/bin/python -m unittest
  scripts.test_radar_evaluation scripts.test_fullsize_training
  scripts.test_fullsize_linear_probe`; `PYTHONPATH=scripts
  $HOME/miniconda3/envs/bev/bin/python -m py_compile
  scripts/radar_evaluation.py scripts/train_radar_scaling.py
  scripts/train_fullsize_baseline.py scripts/train_fullsize_linear_probe.py
  scripts/summarize_velocity_probe.py scripts/run_teacher_overnight.py
  scripts/render_teacher_overnight_progress.py`; and `bash -n
  scripts/run_fullsize_baseline.sh`.
- Actual result: all 17 CPU tests passed, Python compilation passed, the shell
  launcher syntax passed, and a non-executing velocity-zeroed preview selected
  `fullsize_baseline/runs/full28130_zero_velocity_seed125` with the expected
  zero-mode argument. A real batch inspection confirmed vehicle/valid labels
  have shape `(B,1,1,200,200)`, matching the probe after removal of the time
  dimension. The CUDA smokes deliberately remain pipeline gates because the
  active raw-velocity process currently occupies the GPU.
- Launch: commit `9fd53c4` was pushed to `origin/dd2414-mini-baseline`, then
  `dd2414-teacher-overnight.service` started at 2026-10-09 00:28 CEST. Its
  first recorded stage is `waiting_for_full_velocity`; both that controller
  and `dd2414-fullsize-full.service` were verified active. User lingering is
  enabled (`Linger=yes`), so closing terminals or ending an agent session does
  not stop the user services. At launch the raw-velocity run was at epoch
  4.808/8 with 86% GPU utilization and no failure status.
- Monitor: `watch -n 10 $HOME/miniconda3/envs/bev/bin/python
  scripts/render_teacher_overnight_progress.py`; service state is available via
  `systemctl --user status dd2414-teacher-overnight.service` and logs via
  `journalctl --user -u dd2414-teacher-overnight.service -f`.
- Limitations: this is one seed and one radar sweep. The probe measures linear
  separability of a frozen representation rather than end-to-end fine-tuned
  segmentation. Both backbones retain the historical frozen random camera
  encoder, so the result isolates velocity utility in this controlled baseline
  rather than establishing final camera-radar model quality. Runtime and
  completion depend on the host, dataset, and GPU remaining available; the
  pipeline fails closed and records the failing stage instead of silently
  changing parameters.
- Next: allow the persistent pipeline to complete, review the epoch-1 and
  epoch-8 raw-velocity minus velocity-zeroed IoU/F1 reports, then decide with
  the teacher whether a trained camera encoder or additional seeds are the
  next controlled change.

### 2026-10-08 — Epoch-matched trainer and live monitoring

- Change: added the isolated `fullsize_baseline/` output convention and an
  opt-in epoch-native trainer for the unchanged P9 one-sweep model. The target,
  motion and loss paths now support real batches; AdamW uses quarter-epoch
  warmup and cosine decay; an acknowledgement-based sampler saves only samples
  whose optimizer update completed, so worker prefetch remains exactly
  resumable. Added atomic `status.json`, batch metrics, TensorBoard events, an
  auto-refreshing PNG/terminal monitor, automatic console retention, CPU tests
  and a real two-sample batch-equivalence checker. `VizData` now constructs its
  temporary centroid on CPU, which is geometrically equivalent and permits
  DataLoader workers.
- Reason: the fixed 30,000-update study exposed the 4,096 subset about 7.32
  times but full train only 1.07 times. The follow-up compares both scales at
  eight complete passes while using the 3090 Ti more effectively and giving a
  human operator loss, ETA, throughput, VRAM and GPU telemetry throughout.
- Reproduce: run `MPLCONFIGDIR=.cache/matplotlib
  /home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_fullsize_training.py`; preview with `bash
  scripts/run_fullsize_baseline.sh`; check batching with
  `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_fullsize_batch_equivalence.py --data-root
  /home/students2026/datasets/nuscenes --bevcar-source external/BEVCar
  --manifest configs/radar_scaling_manifest_seed125.json --output
  fullsize_baseline/runs/batch_equivalence.json`. Monitoring commands are in
  `fullsize_baseline/README.md`.
- Verification: six CPU tests, Python compilation, launcher previews, shell
  syntax and `git diff --check` passed. A real batch-2 BF16 run stopped and
  resumed from step 2 to step 4 with finite losses and no overflow; a separate
  four-worker run restored its acknowledged sample cursor and continued from
  step 5 to step 10. Against two historical batch-one forwards, batched
  target/confidence maximum absolute
  differences were `0.000244/0`, motion target/coverage differences were zero,
  and FP32 semantic/motion prediction mean differences were
  `0.001630/0.001422`. Online-DINO pilots measured batch 4/workers 4 at 9.08
  samples/s and 13.432 GiB peak allocation, batch 5/workers 8 at 9.85 samples/s
  and 16.721 GiB, and batch 6/workers 8 at 10.01 samples/s and 20.010 GiB. The
  batch-6 process consumed 23,449/24,564 MiB in `nvidia-smi`, so its small
  throughput gain did not justify the low memory headroom; batch 5 used
  20,638 MiB. BF16 and FP16 batch-4 throughput were effectively equal; BF16 had
  no overflow.
- Limitations: pilots cover at most 40 optimizer steps and do not establish
  eight-epoch numerical stability. Online DINO and unused supervised-label
  construction still limit throughput; the planned compact teacher cache is
  not implemented. Validation remains the historical batch-one four-mode path.
  No 4,096/full scientific result was produced.
- Next: run a longer batch-5/workers-8 stability pilot, implement and verify
  the compact DINO cache if its measured end-to-end benefit justifies the path,
  freeze one shared configuration, then run 4,096 and full independently for
  eight epochs
  with complete validation at epochs 1/3/5/8.

### 2026-10-08 — Full-size stability pilot and configuration freeze

- Change: completed a 256-update full-split engineering pilot with batch 5,
  eight workers and BF16. Added the opt-in `--profile-timing` diagnostic and
  separated frozen-DINO extraction from target projection so their costs can
  be measured independently without changing the default path.
- Reason: freeze the formal 4,096/full comparison from measured stability and
  determine whether the proposed compact teacher cache would materially reduce
  end-to-end time before accepting its storage and consistency costs.
- Reproduce: run `FULLSIZE_BATCH_SIZE=5 FULLSIZE_NUM_WORKERS=8
  RUN_DIR=fullsize_baseline/runs/pilot_batch5_workers8_256 bash
  scripts/run_fullsize_baseline.sh --execute --stop-after-steps 256
  --print-every-steps 25 --checkpoint-every-steps 1000
  --validation-epochs '' --val-samples 4`. Add `--profile-timing` with a fresh
  run directory for the synchronized 50-update phase profile.
- Verification: all 256 metric rows were finite and covered exactly 1,280
  unique subset positions and dataset indices. The run reached 12.27 samples/s,
  16.721 GiB peak PyTorch allocation and zero overflow retries; stable
  25-update windows fell from mean loss 1.992 to 0.957. Stable phase time was
  8.3% DINO, 6.8% target projection, 14.0% input/motion target, and 70.9%
  optimization. Seven CPU tests and Python compilation passed after the timing
  refactor.
- Decision and limitations: keep online DINO. Eliminating its measured 0.0322
  seconds per batch gives only a 14.05 samples/s theoretical ceiling before
  cache I/O, which does not justify an approximately 49 GiB FP16 cache. This
  pilot covers only 0.046 epoch and is not evidence of final convergence or
  validation quality. The shared formal configuration is now frozen at batch
  5, eight workers, BF16, AdamW, one sweep, eight epochs and validation at
  epochs 1/3/5/8.
- Next: launch the independent 4,096-sample run, complete its scheduled full
  validations, then launch the all-28,130-sample run under the same frozen
  configuration. Do not enable `--profile-timing` in either formal run.

### 2026-10-08 — Formal 4,096-sample run launched

- Action: launched `SCALE=4096 bash scripts/run_fullsize_baseline.sh --execute`
  as the persistent user service `dd2414-fullsize-4096.service`. Runtime output
  is isolated under `fullsize_baseline/runs/scale4096_seed125/` and remains
  ignored by Git.
- Initial health check: at update 50/6,560 the process reported 10.62 samples/s,
  finite loss 1.7161, 85--95% sampled GPU utilization, 20,557/24,564 MiB VRAM,
  256 W power and no launch error. This is a startup health check, not an
  experiment result.
- Limitations and next: training and the four complete 6,019-frame validations
  are still running. Do not interpret the partial loss or start the full-split
  run concurrently. After 4,096 completes, verify its checkpoint, metrics,
  validation summaries and exact sample exposure, commit the reviewed summary,
  and only then launch the full-split run.

### 2026-10-08 — Formal 4,096-sample control completed

- Result: the run ended with `FULLSIZE_TRAIN_OK` after eight epochs, 6,560
  optimizer updates and exactly 32,768 sample exposures. All metric rows were
  finite, AMP overflow retries were zero, and epoch 1/3/5/8 checkpoints plus
  complete 6,019-frame validation records were written.
- Runtime: total wall time was 10,225.6 seconds (2 h 50 min), including four
  validations of approximately 1,947--1,956 seconds each. Runtime artifacts
  remain under the git-ignored
  `fullsize_baseline/runs/scale4096_seed125/` directory.
- Validation trend: matched semantic loss improved monotonically
  `0.33595 -> 0.32169 -> 0.31200 -> 0.30503`; matched motion loss improved
  `0.91730 -> 0.63530 -> 0.42343 -> 0.28907`. At epoch 8, semantic penalties
  for zero velocity / empty / wrong scene were
  `+0.00476 / +0.03868 / +0.02051`; motion penalties were
  `+2.01498 / +2.29595 / +2.12402`.
- Interpretation and limitation: the control did not collapse, and dependence
  on matched radar strengthened across training, especially for the
  radar-derived motion objective. Motion is not an independent human-labelled
  velocity ground truth, and this remains a distillation/motion evaluation
  rather than downstream vehicle-segmentation IoU.
- Next: launch the independent 28,130-sample run with the identical frozen
  configuration, then compare its epoch-matched validation curve against this
  control.

### 2026-10-08 — Formal full-split run launched

- Action: launched `SCALE=full bash scripts/run_fullsize_baseline.sh --execute`
  as the persistent user service `dd2414-fullsize-full.service`, using the same
  batch 5, eight-worker, BF16, AdamW, one-sweep and eight-epoch configuration
  as the completed control.
- Initial health check: at update 100/45,008 the process had consumed 500 of
  225,040 planned sample exposures, reported finite loss `1.2278`, reached
  11.68 samples/s, and used approximately 20.5/24.6 GiB VRAM at 84--99% sampled
  GPU utilization. The run directory is
  `fullsize_baseline/runs/full28130_seed125/`.
- Limitation and next: this is only a launch check, not a result. Let the run
  complete validation at epochs 1/3/5/8, verify exact exposures and finite
  metrics, then perform the epoch-matched 4,096-versus-full analysis.

### 2026-10-07 — P10 radar dependence curve analysis

- Change: added `scripts/analyze_radar_scaling_curve.py` and its CPU tests,
  generated the machine-readable
  `artifacts/trainval/radar_scaling_curve_seed125.json`, rendered
  `artifacts/trainval/radar_scaling_curve_seed125.png`, and documented the
  result in `RADAR_SCALING_ANALYSIS.md`. Updated the plan, README and broader
  progress report to make P10 and its downstream boundary visible.
- Reason: raw validation means do not show whether penalties are stable across
  scenes or whether a data-scale onset is more than a single noisy point. The
  analysis resamples the 150 validation scenes rather than treating 6,019
  temporally adjacent frames as independent.
- Reproduce: `MPLCONFIGDIR=.cache/matplotlib python
  scripts/analyze_radar_scaling_curve.py --bootstrap-iterations 10000
  --bootstrap-seed 20261007`.
- Verification: four CPU analysis tests passed; the integration run reproduced
  every stored aggregate from per-scene paired penalties within `1e-9`, checked
  identical scenes/configuration across all five inputs, completed 10,000
  deterministic bootstrap resamples and emitted the JSON, Markdown and PNG.
  The three pre-existing scaling/evaluation test suites also remain passing.
- Result: semantic zero-velocity and empty-radar intervals are positive from 64
  samples, while the semantic wrong-scene interval first becomes and remains
  positive at 256 (`+0.00450`, 95% CI `[+0.00319, +0.00582]`). All motion
  intervals are positive from 64. Semantic and motion wrong-scene penalties
  peak at 4,096. Full-minus-4,096 is `-0.00383`
  `[-0.00460, -0.00309]` for semantic and `-0.02506`
  `[-0.06251, +0.01415]` for motion. Matched losses improve from
  0.46199/1.58663 at 64 to 0.30538/0.79639 at full scale.
- Interpretation/limitations: scene-level persistence supports a semantic
  alignment onset followed by saturation, and the absolute diagnostic losses
  do not collapse. All training points still use seed 125, so the bootstrap
  does not establish training-seed repeatability. The random frozen camera
  encoder and diagnostic targets do not establish downstream vehicle utility.
- Next: P11, use a matched camera-only/fused labeled protocol and complete-val
  vehicle IoU before changing sweeps, teachers or fusion objectives.

### 2026-10-07 — P9 full-data radar scaling result

- Change: completed the unchanged seed-125, one-sweep, full 28,130-sample
  scale at 30,000 effective optimizer updates and evaluated all 6,019
  validation frames under the four locked radar modes. Whitelisted the compact
  result `artifacts/trainval/radar_scaling_full_seed125.json`; checkpoint and
  raw JSONL outputs remain ignored.
- Reason: P9 is the final fixed-budget data-scale point and tests whether the
  radar-dependence increase through 4,096 samples persists when every official
  training token is included.
- Reproduce: use the P5 command with `--scale full` and replace each
  `scale64_seed125` output stem with `full_seed125`. Run the first process with
  `--stop-after-update 1000`, then repeat the otherwise identical command with
  `--resume` and without `--stop-after-update` to complete training and
  validation.
- Verification: the first process stopped cleanly at update 1,000 and the
  second restored epoch zero and sampler position 1,000. The completed run made
  30,000 consecutive effective updates, covered all 28,130 unique training
  indices, recovered 20 AMP overflows, sustained 10.225 updates/s and peaked at
  3.574 GiB CUDA allocation. Independent checks found exactly 30,000 train and
  6,019 finite validation records, 150 scene summaries, four modes per frame,
  no same-scene wrong-radar pair and no partial record.
- Result: full-validation matched semantic loss was 0.30538 versus 0.30966
  zero-velocity, 0.34502 empty and 0.33008 wrong-scene, giving penalties
  +0.00428/+0.03964/+0.02470. Matched motion loss was 0.79639 versus
  2.33211/2.40259/2.36329, giving penalties +1.53571/+1.60619/+1.56690.
  Moving-frame penalties were +1.73423/+1.81382/+1.76945 and cell-weighted
  penalties were +1.80552/+1.90730/+1.87137. Semantic wrong-scene penalties
  were positive in 150/150 scenes and motion penalties in 144/150 scenes.
- Interpretation/limitations: radar dependence remains strong at full scale,
  but semantic and motion wrong-scene penalties are slightly below P8's
  +0.02854/+1.59196. The five-point curve rises through 4,096 samples and then
  plateaus or slightly recedes, rather than increasing monotonically to full
  scale. Full scale exposes each token only about once under the fixed 30,000
  updates, whereas smaller subsets repeat tokens more often. This remains one
  seed with diagnostic losses and a random frozen camera encoder, not evidence
  of downstream vehicle-segmentation IoU.
- Next: P10, produce the paired per-scene uncertainty analysis and compact
  dependence-curve report before starting downstream IoU work.

### 2026-10-06 — Adopted full-data radar scaling plan

- Change: added `FULL_DATA_RADAR_SCALING_PLAN.md`, linked it from `README.md`,
  and updated the current state and next bounded task in this handoff. The plan
  separates streaming, exact resume and wrong-scene evaluation into independently
  testable milestones, followed by committed seed-125 scale results at 64, 256,
  1,024, 4,096 and all 28,130 training frames. Every scale uses the same full
  6,019-frame validation split and one radar sweep.
- Reason: the mini comparison changed both sample and update counts and did not
  identify a data threshold. A nested trainval curve with matched, zero-velocity,
  empty and wrong-scene controls can test whether radar dependence increases with
  data diversity before multi-sweep or hybrid-teacher changes add confounders.
- Verification: `git diff --check` passed. File/link checks confirmed the P0--P11
  table, the adopted scale sequence, full-validation count, four radar modes and
  per-milestone commit/push rule. No loader, model or training run was part of
  this documentation-only milestone.
- Limitations: the 30,000-update budget is provisional until the P4 throughput
  pilot. Self-supervised penalties demonstrate dependence, not downstream utility;
  P11 requires a matched camera-only comparison and complete-val vehicle IoU.
- Next: P1, deterministic nested manifests and streaming batches. Commit and push
  each later milestone separately before continuing.

### 2026-10-06 — P1 deterministic manifests and streaming data

- Change: added `scripts/radar_scaling_data.py` with stable SHA-256 scene/sample
  ordering, prefix-nested scale manifests, validation and a non-materializing
  `Subset`/`DataLoader` path. Added CPU tests and the real-data checker in
  `scripts/test_radar_scaling_data.py` and
  `scripts/check_radar_scaling_stream.py`. Generated the committed seed-125
  manifest at `configs/radar_scaling_manifest_seed125.json` and the compact
  acceptance summary at `artifacts/trainval/radar_scaling_stream_check.json`;
  whitelisted only that summary in `.gitignore`.
- Reason: the full-data study needs identical, reproducible nested subsets and
  must fetch batches on demand rather than converting a loader to a Python list
  and retaining all tensors on GPU.
- Reproduce: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_radar_scaling_data.py`; then run
  `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_radar_scaling_stream.py --data-root
  /home/students2026/datasets/nuscenes --manifest-out
  configs/radar_scaling_manifest_seed125.json --summary-out
  artifacts/trainval/radar_scaling_stream_check.json --check-batches 512
  --nsweeps 1` with GPU access.
- Verification: four CPU unit tests passed, Python compilation and
  `git diff --check` passed. The real manifest contains 28,130 train samples in
  700 scenes and 6,019 validation samples in 150 disjoint scenes; scale scene
  counts are 64/256/700/700/700. Streaming 512 real batches read 221,173 radar
  points in 50.923 seconds (10.054 batch/s). CUDA allocated/reserved memory was
  0 MiB at baseline, every 64 batches and completion, so no previous batch was
  retained by this data-only path. Manifest SHA-256 is
  `8905db7995784bbee4d641c9756a8648ebf5510a891fa633fe65c1eab8dcb0fc`.
- Limitations: this milestone did not load DINOv2, build dense BEV targets,
  train a model, or test optimizer state. At the time, `VizData` initialized
  temporary CUDA state and therefore used `num_workers=0`; the 2026-10-08
  epoch-based path removed that construction-time restriction. It still
  constructs supervised labels that self-supervised pretraining does not need.
  Throughput here is a data-only measurement, not the P4 end-to-end timing.
- Next: P2, periodic atomic checkpoints and exact interrupted resume.

### 2026-10-06 — P2 periodic checkpoints and exact resume

- Change: added `scripts/training_checkpoint.py` with atomic `latest` writes,
  fixed-update periodic/forced-save policy, a stateful epoch-shuffle sampler,
  run-configuration validation and complete Python/NumPy/Torch CPU/CUDA RNG
  capture. It saves model, optimizer, GradScaler, sampler and progress state.
  Added CPU failure/round-trip tests in `scripts/test_training_checkpoint.py`,
  the CUDA equivalence check in `scripts/check_training_resume.py`, and the
  compact result `artifacts/trainval/training_resume_check.json`.
- Reason: the 30,000-update scale runs must survive preemption without silently
  changing sample order, dropout/random augmentation, optimizer moments or AMP
  scaling. Atomic replacement prevents a partially written `latest` file from
  being treated as valid.
- Reproduce: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_training_checkpoint.py`; then run
  `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_training_resume.py --checkpoint
  artifacts/trainval/checkpoint_resume_smoke/latest.pt --summary-out
  artifacts/trainval/training_resume_check.json --total-updates 8
  --interrupt-after 4 --seed 125` with GPU access.
- Verification: four CPU tests and Python compilation passed. On the RTX 3090
  Ti, uninterrupted eight-update AMP training and a four-update checkpoint plus
  process-state rebuild/resume reached identical sampler state (epoch 1,
  position 3); maximum parameter and per-step loss absolute differences were
  both `0.0`, tighter than the declared `1e-7` tolerance. The first CUDA run
  exposed that `map_location=cuda` moved serialized CPU RNG bytes to CUDA; the
  loader now explicitly returns all generator-state bytes to CPU, and the
  rerun passed with `TRAINING_RESUME_OK`.
- Limitations: the equivalence test uses a small CUDA network rather than the
  full BEV fusion model. This historical sampler requires `num_workers=0`;
  the separate 2026-10-08 epoch trainer adds the required acknowledgement of
  consumed rather than merely prefetched indices. The P4 end-to-end pilot must
  call this utility from the real streaming trainer and exercise a real
  checkpoint interruption.
- Next: P3, deterministic wrong-scene radar evaluation alongside matched,
  zero-velocity and empty controls.

### 2026-10-06 — P3 deterministic wrong-scene evaluation

- Change: added `scripts/radar_evaluation.py` with locked matched,
  zero-velocity, truly empty and wrong-scene VoxelNet inputs. The wrong-scene
  map groups stable-hash-ordered samples by scene and rotates by the largest
  scene size, producing a deterministic source-token bijection with no
  same-scene pairs. Added an incremental JSONL writer, CPU tests, and
  `scripts/check_wrong_scene_evaluation.py`. Committed candidates include the
  compact two-frame smoke summary and records under `artifacts/trainval/`.
- Reason: empty-radar sensitivity alone cannot show that the model uses the
  radar aligned to the current camera frame. A bijective wrong-scene control
  preserves the validation radar-token distribution while breaking alignment,
  and streaming JSONL avoids retaining 6,019 frame results in memory.
- Reproduce: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_radar_evaluation.py`; then set the repository DINO cache and run
  `TORCH_HOME=/home/students2026/dd2414/.cache/torch
  /home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_wrong_scene_evaluation.py --data-root
  /home/students2026/datasets/nuscenes --bevcar-source
  /home/students2026/dd2414/external/BEVCar --manifest
  configs/radar_scaling_manifest_seed125.json --output
  artifacts/trainval/wrong_scene_evaluation_check.jsonl --summary-out
  artifacts/trainval/wrong_scene_evaluation_check.json --samples 2 --seed 125
  --nsweeps 1` with GPU access.
- Verification: five CPU tests and Python compilation passed. The real checker
  verified a 6,019-source full-val bijection and zero same-scene pairs. Two
  scene-spread real frames produced matched/zero-velocity/empty/wrong-scene
  records with finite semantic and motion losses; mean motion loss was
  `1.4486/1.4486/1.4523/1.4526`. The first mapping algorithm (a cyclic shift of
  a globally hashed order) failed a valid uneven-scene unit case and was
  replaced by the guaranteed grouped-block construction. Initial random-model
  FP16 smoke produced NaNs; non-finite JSON is now rejected and the engineering
  smoke uses FP32, matching its path-validation purpose.
- Limitations: the smoke model is random and untrained, so its loss differences
  are not radar-effect evidence. Full 6,019-frame metric evaluation, AMP on a
  trained fused model, scene-level summaries and confidence intervals belong to
  P4/P5 and later milestones. Interrupted JSONL files remain as `.partial` for
  diagnosis; record-level resume is not yet implemented.
- Next: P4, integrate the real streaming joint trainer, exercise resume, and
  measure end-to-end throughput/memory before locking the scale-run budget.

### 2026-10-07 — P8 4,096-sample radar scaling result

- Change: completed the unchanged seed-125, one-sweep, 4,096-sample scale at
  30,000 effective optimizer updates and evaluated all 6,019 validation frames
  under the four locked radar modes. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale4096_seed125.json`; checkpoint and raw
  JSONL outputs remain ignored.
- Reason: P8 tests whether the growing radar-dependence curve persists after
  another fourfold increase in nested training-set size under the same compute
  budget.
- Reproduce: use the P5 command with `--scale 4096` and replace each
  `scale64_seed125` output stem with `scale4096_seed125`; keep all other locked
  arguments unchanged.
- Verification: 30,000 consecutive effective updates covered exactly 4,096
  unique training indices and seven completed sampler epochs. The run recovered
  20 AMP overflows, sustained 10.269 updates/s, peaked at 3.574 GiB CUDA
  allocation and had mean joint/semantic/motion training losses
  0.5524/0.3100/0.4847. Independent checks found exactly 30,000 train and 6,019
  finite validation records, 150 scene summaries, no same-scene wrong-radar
  pair and no partial record.
- Result: full-validation matched semantic loss was 0.30256 versus
  0.30866 zero-velocity, 0.34537 empty and 0.33110 wrong-scene, giving
  penalties +0.00609/+0.04281/+0.02854. Matched motion loss was 0.79707 versus
  2.35184/2.42179/2.38903, giving penalties +1.55477/+1.62472/+1.59196.
  Moving-frame penalties were +1.75575/+1.83475/+1.79775 and cell-weighted
  penalties were +1.82575/+1.93068/+1.89969. Semantic wrong-scene penalties
  were positive in 150/150 scenes and motion penalties in 142/150 scenes.
- Interpretation/limitations: P5 through P8 wrong-scene penalties rise
  monotonically from +0.00017 to +0.00450 to +0.02093 to +0.02854 for semantic
  loss, and from +0.77715 to +1.10714 to +1.41765 to +1.59196 for motion loss.
  This strongly supports increasing scene-aligned radar dependence with data
  diversity. It remains one seed and a diagnostic loss result with a random
  frozen camera encoder, not proof of downstream IoU improvement.
- Next: P9, run the complete 28,130-sample training split under the unchanged
  protocol and verify that every training token is seen.

### 2026-10-07 — P7 1,024-sample radar scaling result

- Change: completed the unchanged seed-125, one-sweep, 1,024-sample scale at
  30,000 effective optimizer updates and evaluated all 6,019 validation frames
  under the four locked radar modes. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale1024_seed125.json`; checkpoint and raw
  JSONL outputs remain ignored.
- Reason: P7 tests whether the increasing radar-dependence trend at 64 and 256
  samples persists when the nested training subset grows by another factor of
  four without adding optimizer updates.
- Reproduce: use the P5 command with `--scale 1024` and replace each
  `scale64_seed125` output stem with `scale1024_seed125`; keep all other locked
  arguments unchanged.
- Verification: 30,000 consecutive effective updates covered exactly 1,024
  unique training indices and 29 completed sampler epochs. The run recovered
  21 AMP overflows, sustained 10.262 updates/s, peaked at 3.574 GiB CUDA
  allocation and had mean joint/semantic/motion training losses
  0.4761/0.3056/0.3410. Although the interactive terminal handle expired, the
  process completed both training and validation atomically. Independent
  checks found exactly 30,000 train and 6,019 finite validation records, 150
  scene summaries, no same-scene wrong-radar pair and no partial record.
- Result: full-validation matched semantic loss was 0.31452 versus
  0.31841 zero-velocity, 0.34782 empty and 0.33544 wrong-scene, giving
  penalties +0.00389/+0.03330/+0.02093. Matched motion loss was 0.98694 versus
  2.27702/2.42932/2.40459, giving penalties +1.29008/+1.44237/+1.41765.
  Moving-frame penalties were +1.45684/+1.62883/+1.60091 and cell-weighted
  penalties were +1.54969/+1.73035/+1.70491. Semantic wrong-scene penalties
  were positive in 149/150 scenes and motion penalties in 146/150 scenes.
- Interpretation/limitations: P5 to P6 to P7 wrong-scene penalties increase
  from +0.00017 to +0.00450 to +0.02093 for semantic loss, and from +0.77715
  to +1.10714 to +1.41765 for motion loss, while matched losses decrease.
  This is strong evidence that the diagnostic learns greater scene-aligned
  radar dependence with more diverse data. It remains one seed, a random
  frozen camera path, a feature/motion objective rather than downstream IoU,
  and does not yet establish that radar improves a matched camera-only model.
- Next: P8, repeat the unchanged protocol with the nested 4,096-sample subset.

### 2026-10-06 — P6 256-sample radar scaling result

- Change: completed the unchanged seed-125, one-sweep, 256-sample scale at
  30,000 effective optimizer updates and evaluated all 6,019 validation frames
  under the four locked radar modes. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale256_seed125.json`; the checkpoint and
  raw train/validation JSONL files remain ignored.
- Reason: P6 is the second fixed-budget point and tests whether the dependence
  seen at 64 samples strengthens when the nested subset contains four times as
  many samples and scenes.
- Reproduce: use the P5 command with `--scale 256` and replace each
  `scale64_seed125` output stem with `scale256_seed125`; all other arguments,
  including 30,000 updates, seed 125, one sweep and 6,019 validation samples,
  are unchanged.
- Verification: the run completed 30,000 consecutive effective updates over
  exactly 256 unique training indices, with 19 recovered AMP-overflow retries,
  10.295 updates/s, 3.574 GiB peak CUDA allocation and mean
  joint/semantic/motion training losses 0.3282/0.2517/0.1531. Independent
  checks found exactly 30,000 train and 6,019 finite validation records, 150
  scene summaries, no same-scene wrong-radar pair and no partial record. The
  same test suites and compilation used by P5 remained passing before the run;
  result integrity assertions passed after it.
- Result: full-validation matched semantic loss was 0.38045 versus
  0.38740 zero-velocity, 0.39019 empty and 0.38495 wrong-scene, giving
  penalties +0.00694/+0.00974/+0.00450. Matched motion loss was 1.25646 versus
  2.36205/2.37656/2.36359, giving penalties +1.10560/+1.12010/+1.10714.
  Moving-frame penalties were +1.24852/+1.26490/+1.25025 and cell-weighted
  penalties were +1.37098/+1.38008/+1.36908. Semantic wrong-scene penalties
  were positive in 111/150 scenes and motion penalties in 136/150 scenes.
- Interpretation/limitations: from P5 to P6, matched semantic and motion losses
  decreased while wrong-scene penalties increased from +0.00017 to +0.00450
  and +0.77715 to +1.10714. This is consistent with greater radar dependence
  at 256 samples, but two single-seed points cannot establish a reproducible
  onset. The random frozen camera path and lack of downstream IoU remain the
  same limitations as P5.
- Next: P7, repeat the unchanged protocol with the nested 1,024-sample subset.

### 2026-10-06 — P5 64-sample radar scaling result

- Change: completed the locked seed-125, one-sweep, 64-sample scale at 30,000
  effective optimizer updates and evaluated all 6,019 validation frames under
  matched, zero-velocity, empty and deterministic wrong-scene radar. Added
  per-scene metric aggregation, paired penalties, configurable progress output,
  and recoverable AMP gradient-overflow retries to
  `scripts/train_radar_scaling.py`. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale64_seed125.json`; raw JSONL logs and
  the 188 MB checkpoint remain ignored.
- Reason: P5 is the first fixed-budget point in the adopted data-scaling curve.
  Full-validation paired interventions are needed to distinguish semantic and
  motion dependence on radar before changing the training sample count.
- Reproduce: run `TORCH_HOME=.cache/torch MPLCONFIGDIR=.cache/matplotlib
  python scripts/train_radar_scaling.py --data-root
  /home/students2026/datasets/nuscenes --bevcar-source external/BEVCar
  --manifest configs/radar_scaling_manifest_seed125.json --scale 64
  --max-updates 30000 --checkpoint-every 1000 --checkpoint
  checkpoints/radar_scaling/scale64_seed125/latest.pt --train-log
  artifacts/trainval/radar_scaling_scale64_seed125_train.jsonl
  --validation-records
  artifacts/trainval/radar_scaling_scale64_seed125_val.jsonl --summary-out
  artifacts/trainval/radar_scaling_scale64_seed125.json --val-samples 6019
  --print-every 100 --validation-print-every 500 --seed 125 --nsweeps 1`.
  Add `--resume` with otherwise identical arguments after an interruption.
- Verification: the initial process saved update 7,000 and then exposed an AMP
  gradient overflow at update 7,217. The resumed process reproduced that exact
  event, reduced the scaler from 16,384 to 8,192, retried the same batch, and
  completed. The final run made 30,000 effective updates over exactly 64 unique
  training indices, with 18 recovered overflow retries, 10.267 updates/s,
  3.574 GiB peak CUDA allocation and mean joint/semantic/motion training losses
  0.1794/0.1510/0.0568. Independent checks found 30,000 consecutive train-log
  records, 6,019 finite validation records, 150 scene summaries, no same-scene
  wrong-radar pair and no leftover partial record. The three relevant CPU test
  suites passed 14/14 and Python compilation passed.
- Result: full-validation matched semantic loss was 0.46199 versus 0.46504
  zero-velocity, 0.46413 empty and 0.46216 wrong-scene, giving penalties
  +0.00304/+0.00214/+0.00017. Matched motion loss was 1.58663 versus
  2.29385/2.37895/2.36377, giving much larger penalties
  +0.70722/+0.79232/+0.77715. Moving-frame penalties were
  +0.79864/+0.89474/+0.87761 and cell-weighted penalties were
  +0.88404/+0.98410/+0.96961. Motion penalties were positive in 142/150 scenes
  for empty and wrong-scene radar; semantic wrong-scene penalties were positive
  in only 74/150 scenes.
- Limitations: this is a dependence result for feature and motion losses, not a
  vehicle-segmentation IoU result or proof that fusion improves over a matched
  camera-only baseline. The camera encoder remains frozen at random
  initialization, the radar encoder starts randomly, only one seed and one
  sweep are used, and the 64 samples repeat for 468 epochs. P10 must compute
  scene-level confidence intervals before making an onset claim.
- Next: P6, repeat the unchanged 30,000-update/full-validation protocol with
  the nested 256-sample subset.

### 2026-10-06 — P4 end-to-end streaming and resume pilot

- Change: added `scripts/train_radar_scaling.py`, an opt-in batch-streaming
  trainer that builds frozen-DINO radar-anchored targets online, optimizes the
  current BEVCar semantic+motion joint model under AMP, calls the P2 periodic
  checkpoint utility, truncates stale post-checkpoint log records on resume,
  and performs streaming four-mode validation with all/moving/cell-weighted
  motion summaries. Added `IndexedSubset`/prefix helpers to
  `scripts/radar_scaling_data.py`, shared scene-spread validation positions,
  and the compact result `artifacts/trainval/radar_scaling_pilot128.json`.
- Reason: P1--P3 tested the components independently. Scale runs require proof
  that the real DINO target, VoxelNet fusion, optimizer/AMP state, sampler and
  validation path remain bounded and coherent across an actual process restart,
  plus a measured compute budget before starting the 64-sample result.
- Reproduce: first run the trainer with `--scale 128 --max-updates 128
  --checkpoint-every 64 --stop-after-update 64 --val-samples 4` and paths
  `checkpoints/radar_scaling/pilot128/latest.pt`,
  `artifacts/trainval/radar_scaling_pilot128_train.jsonl`,
  `artifacts/trainval/radar_scaling_pilot128_val.jsonl`, and
  `artifacts/trainval/radar_scaling_pilot128.json`; rerun the identical command
  without `--stop-after-update` and with `--resume`. Both use seed 125, one
  sweep, semantic weight 1.0 and motion weight 0.5. Set `TORCH_HOME` and
  `MPLCONFIGDIR` to the repository caches.
- Verification: CPU manifest/evaluation/checkpoint suites and compilation
  passed. A two-update end-to-end precursor passed joint backward, a 188 MB
  checkpoint and trained-model AMP validation. The formal run stopped at update
  64 with sampler position 64, then a new process printed
  `resumed update=64 epoch=0 position=64` and completed all 128 unique training
  positions. It saw moving targets in 113/128 updates and 738 covered cells;
  mean joint/semantic/motion losses were 1.5688/0.5384/2.0608. Online-DINO
  throughput was 9.608 updates/s, estimated 30,000-update pure training time
  0.867 h, peak allocation 3.574 GiB, and the final checkpoint was 187,965,586
  bytes. Four scene-spread AMP validation frames (3 moving, 18 cells) ran at
  3.063 frames/s with finite four-mode metrics. Matched semantic/motion loss was
  0.3828/1.4902 versus empty 0.4875/1.6268 and wrong-scene 0.4557/1.6909.
- Decision/limitations: lock 30,000 updates and online teacher extraction for
  P5--P9; do not allocate the optional ~49 GiB DINO patch cache. Checkpoint
  interval becomes 1,000 for scale runs. The pilot's four-frame losses are an
  engineering signal only, not radar-effect evidence. The current diagnostic
  still freezes a randomly initialized camera encoder, and full-val downstream
  utility remains P11. Initialization/devkit loading and full 6,019-frame
  validation add wall time beyond the 0.867 h pure-training estimate.
- Next: P5, the committed 64-sample / 30,000-update / full-val seed-125 result.

### 2026-09-28 — Matched lightweight-vs-BEVCar supervised comparison

- Change: extended `scripts/bevcar_supervised_mini.py` with an opt-in
  `--encoder {bevcar,light}` flag so the lightweight 7-field point encoder
  runs under the same supervised mini protocol as the BEVCar meeting demo.
  The BEVCar-only gradient checks and the success sentinel are now conditional
  on that flag. Whitelisted `artifacts/light_supervised_*.json` and `.png` in
  `.gitignore`. Updated `RADAR_ENCODER_NOTES.md` and this handoff. Official
  Simple-BEV and lightweight encoder paths are unchanged.
- Reason: the supervisor asked whether a stronger encoder makes radar matter
  more. The existing meeting demo ran only BEVCar and was explicitly not a
  matched-budget comparison; this adds the lightweight branch under an
  identical 4-frame adaptation / 12-frame two-scene validation / three-seed /
  80-update protocol.
- Verification: `bash scripts/run_bevcar_supervised_mini.sh --encoder light
  --steps 80 --seed <125|126|127>` passed for all three seeds, writing
  `artifacts/light_supervised_mini*.json`. Mean 12-frame validation IoU was
  correct radar 0.188, empty radar 0.188, wrong-scene radar 0.188 (light),
  versus the committed BEVCar 0.180 / 0.166 / 0.171. Correct radar beat empty
  in 19/36 light cases (29/36 for BEVCar) and wrong radar in 22/36 (30/36).
  Mean correct-empty probability difference was 0.000054 (light) versus
  0.003379 (BEVCar). Peak allocated CUDA memory 3.618 GiB; about 9.3 s per
  seed. `python -m py_compile` and `git diff --check` passed.
- Limitations: the two branches use different radar input features and
  filtering (lightweight keeps a quality mask and seven numeric fields; the
  BEVCar adapter keeps all in-range returns with seven channels), so encoder
  size is confounded with preprocessing. BEVCar's larger radar-dependence gap
  coexists with a slightly lower absolute correct IoU (0.180 vs 0.188) in this
  4-frame budget. Two mini scenes, three seeds, one radar sweep, box-derived
  labels; no self-supervised or motion objective.
- Next: use this supervised gap as the encoder-change answer and keep the
  self-supervised objective/control redesign as the separate open task.
  Continue review in
  [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1).

### 2026-09-18 — Supervised Simple-BEV + BEVCar meeting demo

- Change: added `nets/bevcar_radar_bridge.py` to import official external
  VoxelNet and project its output into the existing Simple-BEV experimental
  fusion interface. Added `scripts/bevcar_supervised_mini.py`, its shell
  launcher, `scripts/summarize_bevcar_supervised.py`, three-seed JSON metrics,
  a representative prediction/radar/label panel, and a three-seed summary
  figure; updated `.gitignore`, `README.md`, `RADAR_ENCODER_NOTES.md`, and this
  handoff. The official legacy and lightweight paths remain unchanged.
- Reason: separate the earlier self-supervised objective failure from the
  question of whether BEVCar VoxelNet can help under Simple-BEV's original
  human-box-derived labels. Transfer official camera-only encoder/decoder and
  image-fusion weights, freeze camera/decoder, train new radar/fusion layers.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_bevcar_supervised_mini.sh --steps 80 --train-samples 4
  --val-samples 12 --seed <125|126|127>` passed for all three seeds;
  `python scripts/summarize_bevcar_supervised.py` produced the summary. Mean
  12-frame validation IoU was correct radar `0.180`, same fused model with
  empty radar `0.166`, wrong-scene radar `0.171`, and unadapted camera-only
  context `0.121`. Correct radar beat empty in 29/36 seed-frame cases and
  wrong radar in 30/36. SVFE and CML gradient norms were positive in each
  run; peak allocated CUDA memory was 3.623 GiB. A one-step smoke test passed.
- Limitations: supervised labels are derived from human 3D boxes; DINOv2 is
  absent. Only four mini frames were used for adaptation; the 12 validation
  frames are scene-disjoint from that adaptation, but the official camera
  checkpoint was pretrained on broader nuScenes data. The unadapted camera
  model is not a matched-budget comparison. IoU gains are modest and not
  universal across frames. One radar sweep and a custom adapter differ from
  released BEVCar preprocessing; no BEVCar supervised weights were used.
- Next: keep this as the supervised meeting baseline and fix the separate
  self-supervised objective/controls before claiming label-free radar learning.
  Continue review in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  do not merge `main` as a routine push step.

### 2026-09-18 — Fixed-model radar-input ablation

- Change: extended `scripts/compare_radar_dino_tiny.py` with optional
  `--diagnose-radar` evaluation for empty radar, positions only, permuted
  numeric measurements, and the existing cross-scene swap. Added
  `artifacts/radar_dino_ablation.json` and `.png`; updated `.gitignore`,
  `README.md`, `RADAR_ENCODER_NOTES.md`, and this handoff. Training models,
  teacher targets, and official Simple-BEV paths were not altered.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_radar_dino_tiny.sh --samples 4 --steps 60 --heldout-samples 12
  --seed-list 125,126,127 --diagnose-radar` passed. Across 36 seed-frame
  evaluations, mean correct/no-radar loss was light `0.903/0.911` and BEVCar
  `0.493/0.489`; positions-only `0.903/0.495`; permuted-measurements
  `0.903/0.493`. The preliminary two-frame smoke test also passed.
- Additional checks: synthetic CPU inputs confirmed positions/masks remain
  unchanged in the intended ablations, numeric values are permuted rather
  than dropped, and empty BEVCar input has zero occupied voxels. Python
  compile and `git diff --check` passed.
- Interpretation: the lightweight branch has a small radar-geometry effect,
  but numeric measurements barely affect this target loss. BEVCar's low
  target loss does not require any radar under these interventions. This is
  evidence about the objective/metric, not proof that the architecture is
  incapable of radar perception or that its outputs are literally identical.
- Limitation: two mini validation scenes and three seeds, with unequal model
  capacity/filtering. Frozen-DINOv2 cosine loss is not detection accuracy;
  output-map sensitivity and a radar-dependent objective remain untested.
- Next: inspect output-map changes and test an objective/control requiring
  correctly paired radar to outperform empty/mismatched radar. Continue
  review in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  do not merge `main` based on loss reduction alone.

### 2026-09-18 — Two-scene, three-seed held-out radar diagnostic

- Change: extended `scripts/compare_radar_dino_tiny.py` with frozen-DINOv2
  target generation on 12 uniformly spaced mini validation frames, three
  initialization seeds, scene/token separation checks, and a cross-scene
  radar-swap control. Added `artifacts/radar_dino_heldout_comparison.json`
  and `.png`; updated `scripts/dinov2_bev_demo.py` with an optional validation
  loader selection, `.gitignore`, `README.md`, `RADAR_ENCODER_NOTES.md`, and
  this handoff. Existing Simple-BEV and lightweight paths remain unchanged.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_radar_dino_tiny.sh --samples 4 --steps 60 --heldout-samples 12
  --seed-list 125,126,127` passed. Six frames per validation scene; all 36
  seed-frame losses fell for both branches. Mean loss light `0.979 -> 0.903`,
  BEVCar `0.995 -> 0.493`. Final seed-mean ranges were `0.864-0.925` and
  `0.475-0.515`. The paired cross-scene radar swap yielded `0.909` (light)
  and `0.490` (BEVCar), versus aligned `0.903` and `0.493`.
- Limitation: BEVCar's validation loss reduction survives unseen scenes but
  does not show positive dependence on frame-aligned radar under this control.
  The light branch shows only a small alignment effect. Frozen-teacher cosine
  loss is not downstream segmentation accuracy, and only two validation
  scenes were available. Capacity/filtering are still unequal; no complete
  camera-radar fusion or motion model was tested.
- Next: strengthen radar-dependence tests before choosing an encoder or
  claiming radar semantic learning. Continue review in
  [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  no automatic `main` merge.

### 2026-09-18 — Four-frame radar-only DINOv2 target comparison

- Change: added `scripts/compare_radar_dino_tiny.py` and its shell entry,
  `artifacts/radar_dino_tiny_comparison.json` and `.png`; updated `.gitignore`,
  `README.md`, `RADAR_ENCODER_NOTES.md`, and this handoff. Existing model paths
  remain untouched.
- Reason: test whether the lightweight and official BEVCar radar encoders can
  each optimize the same cached frozen-DINOv2 BEV targets on four mini frames,
  using 60 updates per branch and no human box-derived loss.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_radar_dino_tiny.sh --samples 4 --steps 60` passed. Same-frame
  mean cosine loss was light `0.985960 -> 0.889696` and BEVCar
  `1.001846 -> 0.375588`; encoder gradients were finite/nonzero. Peak allocated
  CUDA was 1.070/1.009 GiB respectively. Calibration matched the target cache.
- Limitations: radar-only and all four frames were training samples. BEVCar
  has 495,328 trainable parameters versus 35,712 in the light student, and
  they differ in filtering/features; the loss is not held-out accuracy.
  No camera fusion, supervised checkpoint, or motion model was involved.
- Next: held-out mini data, then camera-radar integration under a fixed
  protocol. Continue review in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  do not merge `main` merely because this optimization check passed.

### 2026-09-18 — Official BEVCar radar encoder isolated smoke test

- Change: corrected the seven-field BEVCar adapter after inspecting official
  source commit `29cacda3bc5416d47428c1d0f017527acad34f90` (seventh field
  is a valid-point mask, not time lag); added an opt-in external-source encoder
  smoke script and documentation. Changed `.gitignore`,
  `nets/bevcar_voxel_adapter.py`, `scripts/test_bevcar_voxel_adapter.py`,
  `scripts/test_bevcar_encoder_smoke.py`,
  `scripts/run_bevcar_encoder_smoke.sh`,
  `artifacts/bevcar_voxel_adapter_audit.png`, `README.md`,
  `RADAR_ENCODER_NOTES.md`, and this handoff. The lightweight path is unchanged.
- Verification: `bash scripts/run_bevcar_voxel_adapter_test.sh` and
  `BEVCAR_SOURCE_DIR=<official-checkout> bash scripts/run_bevcar_encoder_smoke.sh
  --device cpu|cuda` on mini token `cd9964f8c3d34383b16e9c2997de1ed0`:
  403 returns, 251 in range, 243 voxels, zero BEV-cell mismatches, output
  `(1,128,200,200)`, finite nonzero SVFE/CML gradients. Isolated CUDA peak
  PyTorch allocation was 0.729 GiB. See the radar notes for exact setup.
- Limitations: one frame, one radar sweep, random weights; no camera fusion,
  trained accuracy, or label-free objective. Our camera-frame velocity correction,
  deterministic point selection and dynamic voxel count differ from BEVCar's
  released preprocessing. Official code remains an external dependency.
- Next: compare both radar branches under identical frozen-DINOv2 targets,
  examples, and update budgets; avoid motion head for now.
- Collaboration: publish on `dd2414-mini-baseline` and continue review in
  [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1), without
  merging `main` as part of this smoke test.

### 2026-09-18 — BEVCar-shaped voxel input geometry audit

- Change: added `nets/bevcar_voxel_adapter.py`,
  `scripts/test_bevcar_voxel_adapter.py`,
  `scripts/run_bevcar_voxel_adapter_test.sh`, and
  `artifacts/bevcar_voxel_adapter_audit.png`; updated `.gitignore`, `README.md`,
  `RADAR_ENCODER_NOTES.md`, and this handoff. The existing lightweight radar
  encoder and model paths were not changed.
- Reason: establish a shared camera-reference BEV coordinate grid and the
  BEVCar VoxelNet input tensor shapes before considering network transplant.
- Verification: `bash scripts/run_bevcar_voxel_adapter_test.sh` passed on the
  first fixed mini training frame (token `cd9964f8c3d34383b16e9c2997de1ed0`):
  403 returns, 251 in-range, 125 quality-filtered, 123 3D voxels, no point
  truncation, zero BEV-cell mismatches; synthetic collision, empty/batched,
  overflow, and axis checks passed. `git diff --check` passed.
- Limitation: this is CPU input/coordinate validation, not BEVCar encoder
  inference, backward, GPU memory, training, or proof of pretrained-checkpoint
  compatibility. Its explicit seventh feature and deterministic overflow
  policy need upstream preprocessing comparison. The current environment
  denied NVIDIA access, so the GPU-only `VizData` loader was bypassed using
  the same first nuScenes key frame and direct calibrated-sensor transforms.
- Next: confirm upstream feature order and point sampling, then add a separate
  random-weight BEVCar encoder smoke test without replacing the light baseline.
- Collaboration: pushed through the `dd2414-mini-baseline` feature branch and
  tracked in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  this entry does not imply a merge into `main`.

### 2026-09-18 — Push-on-feature-branch workflow

- Change: clarified that each completed project change is committed and pushed
  to the active feature branch without a new conversational permission request;
  the `Stop` reminder now also detects commits ahead of the upstream branch.
- Reason: keep teammates' clones and coding agents synchronized while reserving
  `main` for reviewed, completed features.
- Verification: hook JSON, Python compile, simulated dirty/continued events,
  and a real ahead-of-upstream check before the push of this documentation
  change.
- Limitation: the hook itself never pushes and cannot bypass Codex environment
  approvals or guarantee publication while offline.
- Next: finish and review the radar+DINO experiment before deciding when to
  merge its feature branch into `main`.

### 2026-09-18 — Technical handoff and Codex workflow

- Change: documented dataset/DINO checkpoint provenance, frozen-teacher versus
  trainable-student roles, reproducible experiments, and current radar-fusion
  status; added repository-level agent instructions and a read-only `Stop` hook
  that reminds Codex to review unfinished local changes.
- Reason: let teammates and their coding agents distinguish completed results
  from the next experiment and find the exact reproduction commands.
- Verification: `python3 -m json.tool .codex/hooks.json`, Python compile,
  simulated dirty/continued `Stop` events, Markdown link inspection, and
  `git diff --check` passed. The simulated dirty event requested one handoff
  continuation; a second event with `stop_hook_active=true` returned `{}`.
- Limitation: hooks must be trusted in each Codex installation and cannot
  safely auto-push arbitrary local changes; read-only chats should not mutate
  the repository.
- Next: run the new radar+DINO tiny-subset experiment after team review.

### 2026-09-28 — Camera-side dense semantic distillation switch (opt-in ablation)

- Change: added a switchable camera-side DINOv2 semantic distillation term so the
  dual-level distillation from the meeting can be compared against the existing
  fusion-only term. `nets/segnet.py` now returns the pure camera BEV
  (`camera_bev`, B, feat2d_dim*Y, Z, X) alongside the fused `feat_bev` when
  `return_shared_bev=True`; `scripts/dinov2_bev_demo.py` additionally caches
  `dense_semantic_target` (the naive-rays `teacher_bev`) and `camera_coverage`
  (camera visibility) alongside the existing radar-anchored `soft_targets`;
  `scripts/semantic_distill_smoke.py` gains `--camera-distill`,
  `--camera-weight`, and a decoupled `--train-encoder` flag plus a small
  `camera_head` that maps the camera BEV to the DINOv2 feature space.
- Reason: the supervisor's dual-level distillation keeps DINOv2 semantics from
  being washed out by radar-guided sparsity. The camera-side term distills the
  DENSE (un-radar-guided) DINO features into the camera BEV before fusion,
  while the existing fusion term keeps the radar-anchored (sparse) targets.
  The only differences between the two terms are the student representation
  (camera-only vs fused) and the target mask (camera visibility vs radar soft
  region).
- Verification: `python3 -m py_compile` on the three changed files and
  `git diff --check` passed. Four 20-step single-sample smoke configs
  (cosine-to-DINO loss, NOT a validation result) gave fusion losses:
  baseline (frozen, camera off) 0.2039; +camera-distill (frozen) 0.2039;
  +train-encoder only (trainable, camera off) 0.1730; +train-encoder
  +camera-distill (trainable) 0.1767. Peak CUDA ~4.2 GiB.
- Limitation: the camera term reaches the representation only when the encoder
  is unfrozen, and the fusion-loss gain is driven by encoder trainability
  (0.1730) rather than the dense camera term (0.1767, slightly worse) —
  consistent with the naive-rays depth ambiguity of `teacher_bev`. Single mini
  sample, 20 updates; these are training losses, not held-out results.
- Next: replace the dense target with a depth-disambiguated one (e.g. lift the
  camera BEV to the radar-anchored depth) or evaluate on a held-out split
  before concluding on the camera term.

### 2026-09-28 — Image-level dense DINOv2 distillation (replaces the camera-BEV term)

- Change: moved the second distillation term from the camera BEV to the
  pre-projection image features, where DINOv2 supervision needs no depth and no
  radar. `nets/segnet.py` now also returns `feat_camXs_` (B*S, feat2d_dim,
  Hf, Wf) from `return_shared_bev=True`; `scripts/semantic_distill_smoke.py`
  drops `--camera-distill`/`--camera-weight` and adds `--image-distill` /
  `--image-weight`, with an `image_head` (feat2d_dim -> 384) that maps the image
  features to the frozen DINOv2 patch features already cached as `features` in
  `dinov2_teacher_features.npz`. No demo change was needed and no interpolation
  is required: the res101 image features (112x192 input, stride 8) and the
  DINOv2 vits14 patch grid (196x336 input, patch 14) are both 14x24.
- Reason: distilling on the BEV is a dead end because "dense + depth-correct +
  radar-free" cannot all hold there — a depth-correct dense BEV target needs
  radar (making it radar-guided again and near-identical to the fused term) or
  is a naive-rays blur. Distilling the image features directly keeps the term
  dense, radar-free, and disjoint from the fusion term (it shapes the encoder
  output before projection/fusion), matching "keep DINO semantics from being
  diluted".
- Verification: `python3 -m py_compile` and `git diff --check` passed. Four
  20-step single-sample smokes (fusion loss, cosine-to-DINO): baseline 0.2039;
  +image-distill (frozen) 0.2039 with image loss 0.430; +train-encoder only
  0.1730; +train-encoder +image-distill 0.1726 with image loss 0.405.
- Limitation: the image term now converges cleanly (image loss ~1.0 -> ~0.41)
  and no longer drags the fusion loss (0.1726 vs 0.1730), but the fusion-loss
  improvement is still driven by encoder trainability, and the ~0.0004 image
  delta is within single-sample smoke noise. Single mini sample, 20 updates,
  training losses only. The image term's real target is semantic retention in
  the image features, which a fusion-loss readout only sees indirectly.
- Next: judge the image term on its own objective (image-feature similarity to
  DINO on held-out frames, or a frozen-feature probe) rather than on fusion
  loss, and run a larger multi-sample comparison before claiming any benefit.

### 2026-09-28 — Held-out probe of the image-level distillation term

- Change: added `scripts/eval_image_distill_heldout.py` and its
  `scripts/run_image_distill_heldout.sh` runner, which train two
  trainable-encoder variants (baseline vs image-distilled) on a few training
  samples and measure (a) held-out fusion loss and (b) held-out image-feature
  similarity on scene-disjoint validation samples. `scripts/dinov2_mini4_experiment.py`
  was fixed to unpack the now four-valued `SemanticDistillationModel.forward`
  with `prediction, *_ = model(...)`.
- Reason: the previous smoke could only read the image term indirectly through
  fusion loss. This probe measures the image term on its own objective — how
  well held-out image features retain frozen-DINOv2 semantics — which is the
  term's actual claim.
- Verification: `python3 -m py_compile` and `git diff --check` passed. Three
  seeds (125/42/7) with 4 train + 4 held-out mini samples and 60 steps each
  gave held-out fusion-loss deltas that flip sign across seeds (image_distill
  minus baseline: -0.077, +0.040, -0.020), i.e. no reliable effect on held-out
  fusion loss. The held-out image loss was stable at ~0.57 across seeds
  (random/collapsed cosine is ~1.0), so the image features do retain DINOv2
  semantics on held-out frames even though this does not move fusion loss.
- Limitation: 4 train + 4 held-out mini samples, 60 updates, three seeds. The
  fusion-loss delta is noise-level and the image-retention result is a
  small-sample probe, not a downstream benchmark.
- Next: keep the image term as the camera-side half of the advisor-endorsed
  dual-teacher (image DINO + radar-anchored BEV DINO), and turn to the
  advisor's primary open problem — making matched radar explicitly beat empty
  or mismatched radar — via a stronger radar encoder, multi-sweep input, and
  balanced loss weights.

### 2026-09-29 — Multi-sweep radar ablation (10 sweeps)

- Change: threaded an `--nsweeps` option through `scripts/dinov2_mini4_experiment.py`
  (target generation) and `scripts/compare_radar_dino_tiny.py` (comparison and
  radar ablation), with nsweeps-specific target-cache directories and output
  filenames; `run_dinov2_mini4.sh` and `run_radar_dino_tiny.sh` forward it via
  the `NSWEEPS` environment variable. Re-cloned the pinned BEVCar source
  (`https://github.com/robot-learning-freiburg/BEVCar.git` at `29cacda3`) under
  the git-ignored `external/` directory for the `bevcar` branch.
- Reason: the advisor's recommendation #2 was multi-sweep radar (10-12 sweeps);
  every prior experiment used one sweep, so this is the first direct test of
  whether denser radar input makes the self-supervised objective depend on
  radar.
- Verification: at 5 and 10 sweeps, 3 of 4 training frames grew from ~400
  points to ~2100 and ~3300-4300 respectively (the first scene frame has no
  history and stays at 403). Mean held-out cosine loss across 12 frames / 3
  seeds, over the sweep-size curve:

  | Sweeps | Branch | Correct | Other scene | Positions only | Mixed values | No radar |
  | ---: | --- | ---: | ---: | ---: | ---: | ---: |
  | 1 | Lightweight | 0.903 | 0.909 | 0.903 | 0.903 | 0.911 |
  | 1 | BEVCar | 0.493 | 0.490 | 0.495 | 0.493 | 0.489 |
  | 5 | Lightweight | 0.890 | 0.904 | 0.890 | 0.890 | 0.910 |
  | 5 | BEVCar | 0.488 | 0.485 | 0.492 | 0.488 | 0.484 |
  | 10 | Lightweight | 0.879 | 0.900 | 0.879 | 0.879 | 0.909 |
  | 10 | BEVCar | 0.486 | 0.483 | 0.491 | 0.486 | 0.480 |

  The lightweight encoder's "No radar" penalty (No radar minus Correct) grows
  monotonically with sweeps — +0.008, +0.020, +0.030 — and correct beats
  other-scene by 0.006, 0.014, 0.021. BEVCar's penalty stays near zero and
  slightly negative throughout (-0.004, -0.004, -0.006).
- Limitation: multi-sweep strengthens the small radar-dependence of the
  lightweight encoder but does not repair the strong BEVCar encoder, whose low
  loss remains consistent with a statistical/spatial shortcut. This confirms
  that the objective (not input density) is the bottleneck for label-free radar
  dependence.
- Next: attack the objective itself — loss balancing / cosine weighting and the
  dual-teacher (image DINO + radar BEV DINO) so matched radar must outperform
  empty radar, rather than only increasing radar input density.

### 2026-09-29 — Dual-teacher 0.8/0.2 joint training (image DINO + radar BEV DINO)

- Change: extended `scripts/eval_image_distill_heldout.py` to train a weighted
  dual-teacher objective `L = fusion_weight * L_fusion + image_weight * L_image`
  (camera DINOv2 on image features vs. radar-anchored BEV DINO on the fusion
  BEV), swept four variants — fusion-only baseline, 1:1, 0.8-camera/0.2-radar,
  and 0.2-camera/0.8-radar — and added a no-radar sensitivity probe to
  `evaluate` (zero the radar voxel input, re-measure held-out fusion loss).
  Emits `artifacts/dual_teacher_heldout.json` (committed) plus a metrics
  `.npz` (git-ignored).
- Reason: the advisor's recommendation #5 — the student team's own "two
  teachers, one camera-side and one BEV/radar-side, ~0.8+0.2 weights" idea
  that the advisor endorsed but had not been implemented. It is the last
  input/weight-side lever untested after the strong-encoder and multi-sweep
  experiments.
- Verification: `python3 -m py_compile` passed. 4 train + 4 held-out mini
  samples, 60 steps, seed 125, trainable res101 encoder. Held-out fusion loss
  and no-radar penalty (no-radar minus correct; positive = radar-dependent):

  | Variant | fusion_w | image_w | Held-out fusion | No-radar fusion | Radar penalty | Held-out image |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | baseline (fusion only) | 1.0 | — | 0.556 | 0.561 | +0.0053 | — |
  | image_distill (1:1) | 1.0 | 1.0 | 0.534 | 0.538 | +0.0040 | 0.575 |
  | dual 0.8 camera / 0.2 radar | 0.2 | 0.8 | 0.556 | 0.565 | +0.0089 | 0.577 |
  | dual 0.2 camera / 0.8 radar | 0.8 | 0.2 | 0.584 | 0.590 | +0.0059 | 0.593 |

  The radar penalty stays tiny (~0.004–0.009, <2% relative) under every
  weighting, so zeroing the radar input barely moves held-out fusion loss. Both
  heads train and generalize (image loss ~0.57–0.59); the 1:1 image term
  actually lowers held-out fusion loss to 0.534, the best of the sweep.
- Limitation: the dual-teacher does not make the model depend on radar
  measurements. This closes out the advisor's input/weight-side
  recommendations: the radar-anchored teacher uses only radar *position* as
  depth anchors and never radar's measurement values (RCS / velocity /
  Doppler), so no encoder strength, input density, or loss weighting can force
  the student to require radar. The bottleneck is the target itself.
- Next: replace or extend the radar-anchored target with one that requires the
  radar measurement signal — the radar Doppler/motion-preservation target
  (SELF_SUPERVISED_EXTENSION_PLAN.md section 4) — and re-run the sensitivity
  protocol on the full fusion model.

### 2026-09-29 — Radar velocity/motion target (Doppler preservation) diagnostic

- Change: added `scripts/compare_radar_velocity_tiny.py` and
  `scripts/run_radar_velocity_tiny.sh` — a training-sample overfit diagnostic
  that attaches a 2-channel velocity head to the light and BEVCar radar
  encoders and trains it to predict the ego-motion-compensated velocity
  (vx_comp, vy_comp) of moving radar returns splatted into BEV cells, then
  measures the loss change when the velocity columns are ablated
  (`geometry_only` zeros them, `permuted_measurements` shuffles them, `empty`
  zeroes all radar). Emits `artifacts/radar_velocity_tiny.json` (committed).
- Reason: SELF_SUPERVISED_EXTENSION_PLAN.md section 4 — the one lever that
  forces radar-*measurement* dependence, since every prior target used only
  radar position (depth anchors) and never RCS / velocity / Doppler. Directly
  tests whether a target that requires velocity makes the encoder require
  velocity.
- Verification: `python3 -m py_compile` passed. 4 mini samples, 100 steps, 3
  seeds. The target is restricted to moving points (speed > 1 m/s) because a
  naive all-point mean-L1 velocity target is dominated by near-zero static
  velocity (first attempt: correct ≈ before ≈ 0.22, geometry_only penalty
  ≈ +0.001). Moving returns are sparse — ~10–14 of ~400+ points per sample
  (~3%) — so 12 steps under-trains; 100 steps converges. Mean Huber loss on
  moving cells, and the no-velocity penalty (geometry_only minus correct):

  | Branch | Correct | Positions-only (zero velocity) | Penalty | Mixed values | No radar |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | BEVCar | 0.059 | 0.400 | **+0.341 (6.8×)** | 0.548 | 2.666 |
  | Lightweight | 1.429 | 1.647 | +0.218 (15%) | 1.637 | 2.649 |

  The BEVCar encoder reads velocity to near-zero loss (0.059) and zeroing the
  velocity columns raises the loss 6.8× — the first large radar-measurement
  penalty in the whole project, confirming the target is what forces
  measurement dependence. The lightweight point-encoder only partially recovers
  velocity (1.43 vs the 2.65 "predict zero" floor), because its mean/max voxel
  pooling dilutes the sparse per-point velocity.
- Limitation: training-sample overfit diagnostic, not a generalization result;
  it shows only that the encoder *can* read velocity and that ablation moves
  the loss in the favorable setting. The moving-restricted target needs ~100
  steps (vs ~12 for the dense semantic target) because motion is sparse and the
  gradient is weak.
- Next: integrate the motion target into the joint objective
  `L = λ_sem·L_DINO_BEV + λ_motion·L_Doppler` (using BEVCar or a
  velocity-preserving lightweight encoder) and re-run the full fusion
  sensitivity protocol to confirm the full model now depends on radar
  measurements.

### 2026-09-29 — Motion target on the full fusion model (metaradar) — insufficient alone

- Change: added `scripts/compare_motion_target_tiny.py` and
  `scripts/run_motion_target_tiny.sh` — attaches a 2-channel motion head to the
  full image+radar fusion model's shared BEV (`MotionDistillationModel`
  subclasses `SemanticDistillationModel`), trains it on the moving-return
  compensated velocity target with the camera encoder frozen (so velocity can
  only reach the head through the radar branch), and ablates the radar input:
  `zero_velocity` (metaradar channels 3:7 = vx, vy, vx_comp, vy_comp) and
  `no_radar`. Emits `artifacts/motion_target_tiny.json` (committed).
- Reason: the plan's Doppler/motion target applied to the real fusion model
  (not just the isolated radar encoders), to test whether the target alone
  makes the full model depend on radar velocity.
- Verification: `python3 -m py_compile` passed. 4 samples, 100 steps, 3 seeds,
  frozen res101 camera encoder, trainable bev_compressor + heads. Mean loss and
  the no-velocity penalty (zero_velocity minus correct):

  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | semantic_only | semantic | 0.313 | 0.314 | +0.001 | 0.323 |
  | motion_only | motion | 0.564 | 0.612 | +0.048 (9%) | 1.175 |
  | joint (1.0 sem / 0.5 mot) | semantic | 0.268 | 0.269 | +0.001 | 0.280 |
  | joint (1.0 sem / 0.5 mot) | motion | 0.589 | 0.642 | +0.053 | 1.225 |

  The motion head learns the sparse target (motion loss 2.85 → 0.56) but the
  no-velocity penalty stays small (+0.05, ~9% relative) while no-radar is large
  (+0.61). So the full model predicts the motion map mainly from radar
  *position* and barely reads velocity — because the metaradar input's
  `voxelize_xyz_and_feats` "last-write-wins" voxelization overwrites the sparse
  per-voxel velocity with a random point (unlike BEVCar's learned voxel encoder,
  which kept the 6.8× velocity penalty in the radar-encoder diagnostic).
- Limitation: training-sample overfit diagnostic, not generalization. It shows
  the motion target alone is insufficient on the current fusion model, not that
  a motion target can never work.
- Next: replace the fusion model's metaradar branch with a velocity-preserving
  encoder (BEVCar VoxelNet) and re-run the motion-target sensitivity probe; the
  velocity penalty should then be large, confirming the recipe is a
  motion/Doppler target *plus* a velocity-preserving radar encoder.

### 2026-09-29 — Motion target + BEVCar branch in the full fusion — camera shortcut defeats it

- Change: added an opt-in `use_bevcar_encoder` / `bevcar_encoder` /
  `bevcar_encoder_channels` path plus a `zero_camera_bev` diagnostic flag to
  `nets/segnet.py` (the BEVCar VoxelNet `(B,128,Z,X)` output is concat'd with
  `camera_bev` before `bev_compressor`, mirroring `use_radar_encoder`; all other
  radar/lidar paths unchanged). Added `scripts/compare_motion_target_bevcar_tiny.py`
  and `scripts/run_motion_target_bevcar_tiny.sh`: the same frozen-camera
  motion-target probe as the metaradar experiment, but with the fusion's radar
  branch replaced by BEVCar VoxelNet. Ablations zero the BEVCar velocity channels
  (`features[..., 4:6]` = raw_vx/raw_vz) or all radar. `--zero-camera` zeros the
  camera BEV to cut the camera shortcut. Emits
  `artifacts/motion_target_bevcar_tiny.json` and
  `..._zerocam.json` (both committed).
- Reason: finish the recipe — a velocity-preserving radar encoder (BEVCar) in the
  real fusion model, expected to restore the large velocity penalty that metaradar
  lost.
- Verification: `python3 -m py_compile` passed. 4 samples, 100 steps, 3 seeds,
  frozen res101 camera encoder. Mean loss and velocity penalty (zero_velocity
  minus correct):

  Camera ON (normal fusion):
  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | motion_only | motion | 0.444 | 0.425 | −0.019 | 1.532 |
  | joint (1.0/0.5) | motion | 0.433 | 0.435 | +0.002 | 1.501 |

  Camera ZEROED (control):
  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | motion_only | motion | 0.052 | 0.116 | **+0.064 (2.2×)** | 2.697 |
  | joint (1.0/0.5) | motion | 0.024 | 0.080 | +0.056 (2.3×) | 2.723 |

  With the camera on, the velocity penalty is still ~0 — no better than metaradar.
  The tell is `no_radar = 1.53`, *below* the predict-zero floor (~2.70, the mean
  |velocity| of moving points): the model predicts velocity better than chance even
  with radar fully zeroed, because the frozen camera BEV is a per-sample
  memorization key (4 fixed samples), so the trainable bev_compressor/head look up
  the velocity field without reading radar velocity. Zeroing the camera removes the
  shortcut: correct drops to 0.052 (matching the standalone BEVCar diagnostic's
  0.059), the velocity penalty becomes +0.064 (2.2×, clearly positive vs metaradar's
  +0.05 and the camera-on ~0), and no_radar returns to the 2.70 floor. So the
  VoxelNet → bev_compressor path *does* preserve velocity through the fusion; the
  camera shortcut was the only blocker.
- Limitation: small-sample overfit, not generalization; the camera memorization is
  an artifact of 4 fixed samples (a frozen random camera encoder is still
  sample-identifiable). Freezing the encoder is therefore not enough to isolate
  radar velocity in a 4-sample diagnostic. The penalty is 2.2× here vs 6.8× in the
  standalone radar encoder, because the bev_compressor mixes in position features
  that soften the loss when velocity alone is removed.
- Next: the recipe is now motion/Doppler target + velocity-preserving radar encoder
  + cutting the camera shortcut. To prove the full model needs radar velocity
  end-to-end, either (a) run the motion probe on a held-out split (camera cannot
  memorize unseen samples), or (b) route the motion head off a radar-only branch so
  the camera can never supply velocity, and re-measure the penalty at scale.

### 2026-09-29 — Motion target + BEVCar held-out probe — velocity does not generalize at 4 samples

- Change: added `scripts/compare_motion_target_bevcar_heldout.py` and
  `scripts/run_motion_target_bevcar_heldout.sh`. Same BEVCar fusion + motion target
  as the overfit probe, but training samples come from the `train` split and
  evaluation from the scene-disjoint `val` split, with the camera encoder kept ON
  and frozen (an unseen sample's camera BEV is a pattern the bev_compressor has
  never seen and cannot memorize). The probe's helpers are reused by importing from
  `compare_motion_target_bevcar_tiny` rather than duplicated. Emits
  `artifacts/motion_target_bevcar_heldout.json` (committed).
- Reason: option (a) from the previous entry — prove whether the full model needs
  radar velocity end-to-end once the camera cannot memorize unseen samples.
- Verification: 4 train / 4 val samples, 100 steps, 3 seeds. Held-out (val) motion
  loss and velocity penalty (zero_velocity − correct):

  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | motion_only | motion | 2.193 | 2.181 | −0.012 | 2.585 |
  | joint (1.0/0.5) | motion | 2.409 | 2.405 | −0.004 | 2.570 |

  Held-out motion `correct ≈ 2.2–2.4` is essentially the predict-zero floor (~2.70)
  and the velocity penalty is ≈0 (slightly negative = noise). So on unseen samples
  the model does not use radar velocity for the motion target: the velocity-reading
  ability does not generalize from 4 training samples. Radar *position* still helps
  a little (no_radar 2.57–2.59 vs correct 2.19–2.41), but the velocity channels add
  nothing. This is the decisive control — the earlier overfit "success"
  (camera-zeroed correct=0.052, penalty 2.2×) was memorization of the 4 training
  samples, not a learned velocity function.
- Limitation / interpretation: not a recipe failure but the expected small-sample
  result. The tiny probes establish the mechanism (BEVCar carries velocity through
  the fusion) but cannot demonstrate generalization; whether the velocity penalty
  turns positive on held-out data is exactly what needs many more samples (the full
  nuScenes run). Also flags a real scaling risk: the motion target supervises only
  ~3% of returns (~8–16 moving cells of 40 000), so the velocity signal is sparse
  and may need re-weighting at scale.
- Next: scale up the held-out probe (train ~24–64 samples) to see whether the
  velocity penalty turns positive before committing to the full supercomputer run,
  or go straight to the full nuScenes run with the motion loss re-weighted to the
  sparse moving-point coverage.

### 2026-09-29 — Held-out probe scaled to 64 samples — velocity penalty turns positive

- Change: added a `--tag` argument to
  `scripts/compare_motion_target_bevcar_heldout.py` (suffixes the output filename)
  and a matching `TAG` env passthrough in `scripts/run_motion_target_bevcar_heldout.sh`
  so scale runs write separate JSONs instead of overwriting the 4-sample one.
  `.gitignore` now un-ignores `artifacts/motion_target_bevcar_heldout_*.json`. Ran
  the same held-out probe with 64 train / 16 val samples and 512 steps (8 epochs).
  Emits `artifacts/motion_target_bevcar_heldout_train64.json` (committed).
- Reason: option b1 — test whether the velocity penalty turns positive as the
  training set grows, before committing to the full supercomputer run.
- Verification: mean held-out velocity penalty (zero_velocity − correct) vs training
  samples:

  | Train samples | motion_only penalty | joint penalty |
  | ---: | ---: | ---: |
  | 4 | −0.012 | −0.004 |
  | 64 | +0.009 | +0.022 |

  Held-out motion `correct` also falls as data grows (motion_only 2.19→1.88; joint
  2.41→1.64), so the model starts predicting velocity rather than sitting at the
  predict-zero floor (~2.70). The penalty turns positive but remains small: velocity
  dependence is emerging but weak at 64 samples. The trend confirms the
  data-scarcity reading — more samples → more radar-velocity dependence, which is
  what the full nuScenes run should deliver.
- Limitation: the val split contains many stationary scenes with zero moving points
  (`motion_loss` returns 0 there, diluting the mean), so the reported penalty is a
  mean over all val samples and the per-moving-sample penalty is larger than shown.
  Still small in absolute terms at 64 samples.
- Next: re-run at 128 samples to confirm the monotonic trend, then decide whether to
  proceed to the full nuScenes run (with the motion loss re-weighted to the sparse
  moving-point coverage).

### 2026-09-30 — 128-sample probe: fixed motion_loss crash, deferred to 20 GB GPU

- Change: fixed a crash in `motion_loss` (`scripts/compare_motion_target_bevcar_tiny.py`):
  when a sample has no moving points it returned a detached `torch.zeros(())`, so in the
  `motion_only` variant `loss` became a leaf with no grad and `.backward()` raised
  "element 0 of tensors does not require grad and does not have a grad_fn". It now
  returns `prediction.sum() * 0.0` (a true zero that keeps the graph connected). Also
  added `flush=True` to the seed-result prints in
  `scripts/compare_motion_target_bevcar_heldout.py` so long-run progress is visible.
- Reason: the 128-sample run (128 train / 32 val, 1024 steps) crashed after the first
  control variant. The 64-sample run only survived because its 64 train samples happened
  to all contain moving points; the 128-sample set adds stationary samples.
- Verification: script compiles and the fixed branch is exercised by the val set of the
  completed 64-sample run. The 128-sample re-run was started and confirmed progressing,
  but was stopped early: the laptop GPU (RTX 5060 Laptop) was clock-capped at 1065 MHz
  (34% of its 3090 MHz max, 30 W against a 115 W limit) with VRAM 97% full, giving a
  ~2.5–3 h ETA instead of ~25 min. No 128-sample result was produced.
- Next: run the 128-sample (or larger) probe on the machine room's 20 GB GPU.

### 2026-09-30 — Dual-teacher held-out probe: seed-list + tag

- Change: `scripts/eval_image_distill_heldout.py` now accepts `--seed-list`
  (comma-separated seeds, replacing the single `--seed`) and `--tag` (output
  filename suffix); per-variant metrics are aggregated across seeds before
  writing `artifacts/dual_teacher_heldout{_tag}.json`. The previous fixed output
  name made consecutive runs overwrite each other.
- Reason: run the 64-sample dual-teacher + weight-sweep probe (baseline,
  image_distill, 0.8-camera/0.2-radar, 0.2-camera/0.8-radar x 3 seeds) without
  clobbering results, and let the 20 GB machine run the same probe in parallel.
- Verification: script compiles; the `--tag train64` run is in flight on the laptop.

### 2026-09-30 — Vectorized the radar-anchored soft-target splat

- Change: `scripts/dinov2_bev_demo.py` `radar_anchored_soft_targets` no longer
  scatters each visible radar point in a Python loop (one GPU-to-CPU sync per
  point per field). The Gaussian scatter is now one vectorized pass (broadcast
  grid + boolean window mask + `einsum`).
- Reason: the loop dominated target extraction, making 64-sample feature
  extraction take ~90 min on the throttled laptop instead of minutes.
- Verification: a synthetic old-loop-vs-vectorized equivalence test matches to
  float32 precision (max abs diff ~5e-7, identical nonzero cell counts); module
  compiles.

### 2026-09-30 — 64-sample, three-seed dual-teacher held-out probe

- Change: ran the committed multi-seed/tagged dual-teacher held-out probe and
  committed `artifacts/dual_teacher_heldout_train64.json`; whitelisted that
  compact JSON result in `.gitignore`. No model, target, or official
  Simple-BEV path changed. The downloaded nuScenes mini dataset and DINOv2
  source/weights remain local dependencies and are not tracked.
- Reason: scale the four-sample dual-teacher weight sweep to 64 training and 16
  scene-disjoint held-out mini samples on the available 24 GB GPU, using the
  newly vectorized target construction and three independent seeds.
- Verification: `CONDA_ENV=bev NUSCENES_ROOT=/home/students2026/datasets/nuscenes
  TORCH_HOME=/home/students2026/dd2414/.cache/torch TRAIN_SAMPLES=64
  VAL_SAMPLES=16 STEPS=512 bash scripts/run_image_distill_heldout.sh
  --seed-list 125,42,7 --tag train64` completed with
  `DUAL_TEACHER_HELDOUT_OK`. Mean held-out fusion / no-radar fusion / penalty
  (no-radar minus correct) across 48 seed-frame evaluations were: baseline
  `0.5238 / 0.5410 / +0.0172`; image-distill `0.4910 / 0.4996 / +0.0086`;
  0.8-camera/0.2-radar `0.4760 / 0.4897 / +0.0137`; and
  0.2-camera/0.8-radar `0.4703 / 0.4847 / +0.0144`. Held-out image cosine
  losses were `0.5274`, `0.5272`, and `0.5288` for the three image-supervised
  variants, respectively. The official nuScenes devkit loaded 10 scenes, 404
  samples, and 31,206 sample-data records; the repository data-only smoke and
  Python compilation passed before the run.
- Interpretation: increasing data and updates improves the held-out fusion
  losses relative to the preceding 4-train-sample probe, with the
  radar-weighted 0.2-camera/0.8-radar setting best on that loss. But all
  no-radar penalties remain only about 2--3% of the correct loss, so this
  semantic objective still provides weak evidence of radar use and no evidence
  that it reads radar measurement values.
- Limitations: this is a 64-sample mini probe, not a full-dataset benchmark.
  It tests radar removal rather than matched-versus-mismatched radar or
  velocity/Doppler ablations; the radar-anchored DINO target still uses radar
  positions as depth anchors, not its measurement fields. The DINOv2 teacher
  is frozen and no human-box supervision is used here.
- Next: retain this result as the scaled dual-teacher baseline; prioritize the
  motion-target held-out scale-up (128 samples or larger) and/or a control in
  which matched radar must beat empty and mismatched radar before making a
  radar-learning claim.

### 2026-09-30 — Dual-teacher 64-sample multi-sweep comparison

- Change: added `--nsweeps` to `scripts/eval_image_distill_heldout.py` and
  forwarded `NSWEEPS` in `scripts/run_image_distill_heldout.sh`, so the same
  dual-teacher held-out protocol can select the radar history merged by the
  nuScenes loader. The value is stored in report metadata. Ran and committed
  the compact 5- and 10-sweep JSON reports; the code preserves the previous
  default of one sweep and all official Simple-BEV paths.
- Reason: test the sparse-radar hypothesis before combining architectural
  changes. Hold train/validation samples (64/16), updates (512), seeds
  (125/42/7), frozen DINOv2 teacher, target construction, and four loss
  configurations fixed; change only the merged radar sweeps (1, 5, or 10).
- Verification: `CONDA_ENV=bev NUSCENES_ROOT=/home/students2026/datasets/nuscenes
  TORCH_HOME=/home/students2026/dd2414/.cache/torch TRAIN_SAMPLES=64
  VAL_SAMPLES=16 STEPS=512 NSWEEPS=<5|10> bash
  scripts/run_image_distill_heldout.sh --seed-list 125,42,7
  --tag train64_nsweeps<5|10>` completed for both sweep counts with
  `DUAL_TEACHER_HELDOUT_OK`. A 4-train / 4-held-out / 1-step 5-sweep and
  10-sweep smoke also passed. Mean held-out fusion / no-radar penalty across
  48 seed-frame evaluations were:

  | Sweeps | Variant | Fusion | Penalty |
  | ---: | --- | ---: | ---: |
  | 1 | baseline | 0.5238 | +0.0172 |
  | 1 | image-distill | 0.4910 | +0.0086 |
  | 1 | 0.8 camera / 0.2 radar | 0.4760 | +0.0137 |
  | 1 | 0.2 camera / 0.8 radar | 0.4703 | +0.0144 |
  | 5 | baseline | 0.4698 | +0.0147 |
  | 5 | image-distill | 0.4613 | +0.0079 |
  | 5 | 0.8 camera / 0.2 radar | 0.4556 | +0.0180 |
  | 5 | 0.2 camera / 0.8 radar | 0.5143 | +0.0150 |
  | 10 | baseline | 0.4612 | +0.0148 |
  | 10 | image-distill | 0.5176 | +0.0052 |
  | 10 | 0.8 camera / 0.2 radar | 0.4677 | +0.0133 |
  | 10 | 0.2 camera / 0.8 radar | 0.4556 | +0.0196 |

- Interpretation: the best loss is the 10-sweep, radar-weighted
  0.2-camera/0.8-radar configuration (0.4556), marginally below the best
  5-sweep result (0.4556 before rounding). Its no-radar penalty rises from
  +0.0144 at one sweep to +0.0196 at ten sweeps, a modest favorable signal.
  The curve is not monotonic across variants: at five sweeps the radar-heavy
  configuration degrades to 0.5143, while the camera-heavy configuration is
  best. Extra radar density therefore changes optimization, but it has not
  established a universal weight preference or strong radar dependence.
- Limitations: a +0.0196 penalty is only about 4.3% of the correct loss. This
  target still uses radar positions as depth anchors and does not require RCS,
  Doppler, or velocity; it evaluates empty radar but not matched-versus-wrong
  radar. This remains a nuScenes-mini held-out feature-loss probe, not a
  downstream detection/segmentation benchmark or a BEVCar fusion result.
- Next: at 10 sweeps, sweep radar-majority dual-teacher weights (for example
  0.6, 0.7, and 0.8 radar) under this same protocol. Then add the BEVCar
  encoder, motion/soft-region controls, and hybrid-teacher components one at a
  time with matched ablations rather than combining them before attribution.

### 2026-10-05 — Trainval configuration and review, no experiment execution

- Sync: `git fetch origin` succeeded. HEAD and the feature-branch remote both
  resolved to `3745647` (0 ahead / 0 behind); no pull was needed. The earlier
  staged mini 64-sample/multi-sweep changes were already present and preserved.
- Changes: added `configs/trainval.env`, `scripts/run_trainval.sh`,
  `scripts/heldout_data.py`, `scripts/test_trainval_configuration.py` and
  `TRAINVAL_RUN_PLAN.md`. Updated `scripts/dinov2_bev_demo.py`, both held-out
  Python probes and their shell launchers, `scripts/smoke_test_mini.py`, and
  `train_nuscenes.py`. Refreshed `README.md`, `SETUP_LOCAL.md`,
  `SELF_SUPERVISED_EXTENSION_PLAN.md` and this handoff.
- Rationale: remove hard-coded mini selection from the active held-out paths,
  expose motion-probe sweeps, keep both splits on one devkit instance, avoid
  CUDA-in-forked-worker problems with worker counts of zero, spread trainval
  probe frames across scenes, record exact sample manifests, and prevent
  accidental dense all-dataset GPU caching. Historical defaults and official
  model/checkpoint interfaces remain unchanged.
- Prepared parameters: `bev`, `dset=trainval`, batch 1, workers 0, sweeps 5
  (override 1/10); bounded probes use 64 train / 16 val, 512 updates and seeds
  125/42/7. A separate supervised streaming profile uses 30,000 updates,
  accumulation 1, 224x400 images, checkpoint interval 1,000, and full-split
  validation through the separate eval mode. These are proposed budgets,
  not measured trainval results.
- Verification: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_trainval_configuration.py` passed all four CPU-only mocked
  tests (selection/bounds, cache-budget guards, manifest/disjointness, mini and
  trainval loader routing). Python compilation and shell syntax checks passed.
  `bash scripts/run_trainval.sh <data-check|smoke|dual-teacher|motion|supervised>`
  and `INIT_DIR=/placeholder/checkpoint bash scripts/run_trainval.sh eval`
  printed valid commands and returned without running a loader/model. Metadata
  inspection counted 700/150 scenes and 28,130/6,019 samples; a complete
  sample-data filename scan found 0 missing and 0 zero-byte files out of
  2,631,083. `git diff --check` passed.
- Limitations: no real loader batches, DINO extraction, model forward/backward,
  training or evaluation were run for this change, as requested. Full-file
  existence checks do not prove every sensor payload decodes correctly. BEVCar
  external source is missing locally. Current GPU-cached distillation probes
  remain bounded to 160 total frames and lack resume; a full-data streaming
  self-supervised trainer and unified hybrid model still need implementation.
- Publication blocker: `git var GIT_AUTHOR_IDENT` fails with "Author identity
  unknown" because no Git name/email is configured. Do not invent a teammate's
  identity. No new commit/push is claimed; previously staged work is retained.
- Next: user review of the plan, then the explicitly requested smoke stage;
  supply a Git author identity before publishing these changes.

### 2026-10-05 — BEVCar VoxelNet 64/16 reproduction on trainval

- Scope: cloned the official BEVCar repository at the project-pinned commit
  `29cacda3bc5416d47428c1d0f017527acad34f90` (git-ignored external dependency).
  The experiment dynamically imported only `nets/voxelnet.py`, initialized it
  randomly, and used the existing Simple-BEV fusion/motion wrapper. No full
  BEVCar architecture or BEVCar checkpoint was loaded.
- Exact reproduction: after a successful one-batch data check and one-step
  VoxelNet smoke, ran `TRAIN_SAMPLES=64 VAL_SAMPLES=16 STEPS=512
  SEED_LIST=125,42,7 NSWEEPS=1 SAMPLE_SELECTION=head
  TAG=trainval_train64_val16_nsweeps1_head bash scripts/run_trainval.sh motion
  --execute`. This preserved the latest mini protocol and changed only the
  dataset version. The selected trainval subset covered 2 train scenes and 1
  val scene; 10/16 val frames had moving target cells.
- Exact result (correct / zero-velocity / penalty / no-radar motion loss):
  semantic-only `1.0922 / 1.0920 / -0.0002 / 1.0760`; motion-only
  `1.0641 / 1.1409 / +0.0768 / 0.9097`; joint
  `1.2210 / 1.3014 / +0.0804 / 0.9473`. Moving-frame-only penalties were
  `+0.1228` and `+0.1286` for motion-only/joint.
- Scene-coverage control: repeated the same budget with
  `SAMPLE_SELECTION=uniform`, covering 64 distinct train and 16 distinct val
  scenes; 15/16 val frames had moving cells. Motion-only was
  `1.6998 / 2.4977 / +0.7980 / 2.6245`; joint was
  `1.5796 / 2.4398 / +0.8602 / 2.6140`. Moving-frame-only penalties were
  `+0.8512` and `+0.9176`.
- Interpretation: semantic-only remains velocity-insensitive. The exact head
  result has a positive but modest velocity effect and no-radar is better than
  correct, so it does not establish reliable radar use. The broad-scene control
  shows strong, seed-consistent VoxelNet velocity dependence and correct radar
  beats removal, but it changes sample selection and remains a bounded probe.
- Artifacts: committed candidates are
  `artifacts/trainval/motion_target_bevcar_heldout_trainval_train64_val16_nsweeps1_head.json`
  and `..._uniform.json`; each records exact indices, tokens and scene tokens.
- Verification/limitations: data check loaded 28,130/6,019 samples and 313
  radar points in the first train frame; one-step smoke and both three-seed runs
  ended with `MOTION_TARGET_BEVCAR_HELDOUT_OK`. No 5/10-sweep, all-frame
  streaming pretraining, checkpoint/resume test or downstream IoU was run.
- Publication blocker: no commit/push was created because repository Git author
  name/email are unset. Existing staged mini work remains preserved.
- Next: choose between a matched 1/5/10-sweep broad-scene comparison and the
  streaming/checkpoint implementation required for all 28,130 train frames.

<!-- baseline-followup-runtime:start -->

### 2026-10-16 — Unattended baseline follow-up runtime

- Stage: `finished`; queue state: `finished_with_partial_or_failed_stages`.
- Generated files: `artifacts/meeting_followup_day_results.json` and
  `artifacts/meeting_followup_day_report.md`, plus this runtime handoff block.
- Reproduce: `python scripts/run_baseline_followup_day.py --study-dir
  fullsize_baseline/runs/meeting_followup_20261016` (requires the documented local CUDA environment).
- Tests/evidence: completed stages require exit code zero and a complete
  summary; exact metrics, frame counts, configurations, and failures are in the JSON.
- Limitations: one seed; different official/legacy input resolutions;
  CUDA backward is not guaranteed bitwise deterministic when unfreezing camera projection.
- Next: the old timed queue is superseded by `MULTISWEEP_TEACHER_DAY_PLAN.md`;
  resume its saved historical finetuning in the replacement queue, not this launcher.

<!-- baseline-followup-runtime:end -->

### 2026-10-16 — Baseline double distillation, depth lifting and vehicle readout

- Scope/rationale: the owner requested actual local camera DINO SSL plus
  radar-masked fused-BEV DINO, future training at five/ten sweeps, depth
  supervision, frozen/unfrozen downstream comparisons and independent
  vehicle-motion evaluation. The cluster branch remains at `33f2d86` and is
  neither merged nor edited; original Simple-BEV and legacy training files
  remain unchanged. The unrelated staged `DD2414_Project_Plan.pdf` is excluded.
- Changed files: new `scripts/multisweep_teacher.py`,
  `scripts/train_multisweep_teacher.py`, `scripts/run_multisweep_followup.py`,
  `scripts/start_multisweep_followup.sh`,
  `scripts/check_multisweep_teacher_cuda.py`,
  `scripts/test_multisweep_teacher.py`,
  `scripts/publish_multisweep_followup.py`, and
  `MULTISWEEP_TEACHER_DAY_PLAN.md`; update the existing monitor, old plan and
  this handoff. The new path is opt-in and uses the installed workstation
  environment, not a cluster environment replacement.
- Implementation: trainable ResNet-101, pinned BEVCar VoxelNet, dense image
  and radar-soft-masked BEV DINO losses (0.2/0.8), retained radar-motion
  reconstruction auxiliary (0.5), optional metric radar-depth loss (0.2),
  and optional frozen Depth Anything V2 relative inverse-depth shape loss
  (0.05). Depth predictions actually weight voxel lifting; no GT depth is
  supplied during inference. Both five/ten-sweep treatments keep target radar
  age <=0.1 seconds while all sweeps enter the encoder. The common input
  occupied-voxel guard is 7,000, avoiding the legacy 3,500 guard rejecting a
  dense ten-sweep frame without changing VoxelNet parameters or discarding
  points. Original legacy preprocessing is untouched.
- Evaluation: official global valid-cell Vehicle IoU, precision, recall and
  F1; identical frozen/unfrozen 1x1 head comparisons plus full official-decoder
  finetuning; fixed-token GT/prediction/error PNGs and complete per-frame
  records. A separate frozen supervised linear readout pools GT oriented
  vehicle ROIs and predicts temporal annotation `box_velocity` in camera XZ
  m/s. Matched/velocity-zeroed/empty/wrong-scene inputs use fixed GT locations
  and targets; report EPE/RMSE/speed MAE and moving/stationary groups, plus
  zero and train-mean baselines. This is not detection mAVE/NDS.
- Actual gates: 57 CPU tests pass with `PYTHONPATH=scripts .../envs/bev/bin/python
  -m unittest discover -s scripts -p 'test_*.py'`; new Python modules compile,
  launcher passes `bash -n`, and `git diff --check` passes. Ten real CUDA
  engineering cases pass at batch four: four pretrain variants, a real
  save/restart, all downstream modes, depth finetuning and frozen depth probe.
  The same 55 GT vehicles in four validation frames were evaluated in every
  motion mode. Largest observed allocated memory is 18.275 GiB. A batch-five
  depth engineering run actually OOMed; its failed output remains local.
  These 8-train/4-val smokes are not model-quality evidence. Checks live at
  `fullsize_baseline/smoke/multisweep_teacher_20261016/validated_batch4_motion_aux/checks.json`.
- Final-source throughput/resume gate: the heaviest radar+DAV2-depth,
  double-DINO, motion-auxiliary configuration processed 400 full-manifest
  frames in 100 updates at 6.632 samples/s including input wait (excluding
  checkpoint/validation overhead), peak allocated 18.276 GiB. Live GPU
  utilization reached 94–100% during computation. A fresh process restored
  update 100 and continued to update 101 / sampler cursor 404 with finite
  losses. The pilot is intentionally partial, not a full-data quality result;
  see `fullsize_baseline/smoke/multisweep_teacher_20261016/full_manifest_depth_throughput`.
  Lock all new scientific treatments to physical batch four and eight workers.
- Safe migration: stopped the CPU-only recovery watcher and explicitly
  signalled the old queue to checkpoint its existing historical finetuning.
  It saved update 6,451 / 32,255 exposures (epoch 1.1466 of three), with its
  completed epoch-one full-val IoU 23.109% preserved. This was an intentional
  checkpointed migration, not a timeout or a training crash. The replacement
  queue resumes that exact job and recovers the legacy five-sweep probe;
  the pending new one-sweep velocity-zeroed job is deferred under the owner's
  multi-sweep priority.
- Reproduce: `bash scripts/start_multisweep_followup.sh`; inspect
  `fullsize_baseline/runs/multisweep_teacher_20261016/pipeline.json`, `REPORT.md`
  and `queue.log`, or use `scripts/monitor_baseline_followup.py --study-dir`
  with that path. The service is `dd2414-multisweep-teacher.service`.
  The new trainer saves an initial checkpoint and every 250 successful
  updates, epoch boundaries and requested stop, including optimizer/scaler,
  acknowledged sampler and RNG. No idle, elapsed-time or 24-hour kill rule;
  stale status warns only. Real failures retry from durable state at most
  three times, then independent configurations continue. Git publication
  failures are nonfatal. There is no silent sample/batch/epoch reduction.
- Limits/next: this first screen uses all 28,130/6,019 frames but one
  pretraining epoch and one downstream epoch per configuration, one seed and
  112x192 cameras. Official 448x800 scores are references; the new queue also
  evaluates official weights at 112x192. Short probes can underfit, radar
  depth is not dense GT, and object locations are supplied. Depth loss and
  lifting change together. Historical single-sweep results are not a strict
  three-way sweep ablation under the new objectives. Do not claim that camera
  SSL/sparsity was the proven cause of low IoU or that short-budget training
  establishes a capability ceiling. Continue the persistent queue, review
  full-val masks and motion errors, then run longer matched epochs/seeds.

<!-- multisweep-runtime:start -->

### Multi-sweep double-distillation runtime

- Queue: `complete`; last stage: `double_dino_5sweep_depth_radar-dav2_object-motion_1epochs`.
- Changed runtime files: the two compact `artifacts/multisweep_teacher_*` reports and this block.
- Reproduce/continue: `bash scripts/start_multisweep_followup.sh` on the baseline branch.
- Evidence: a stage is complete only after a zero exit code and its complete summary;
  actual full-split metrics and missing/failed stages are listed in the JSON/Markdown.
- Limitations: one seed, short epoch budget, legacy/new objective differences,
  low image resolution, radar-derived depth anchors and oracle GT vehicle locations.
- Next: review full-val masks, paired 5/10 sweeps, frozen/unfrozen probes and object velocity errors;
  extend epoch-matched training before claiming convergence or causal root causes.

<!-- multisweep-runtime:end -->

### 2026-10-16 — Depth-first queue reprioritization

- Scope/rationale: the owner asked to pause the incomplete ten-sweep screen and
  prioritize depth evidence for the next meeting, before replacing the
  low-resolution/random-camera protocol with the formal high-resolution,
  ImageNet-pretrained ResNet-101 experiment. The ten-sweep worker stopped
  cooperatively at update 863 / 3,452 samples (12.27% of one epoch) and wrote a
  resumable 483,343,930-byte `latest.pt`; this was not a crash or deletion.
- Changed files: `scripts/run_multisweep_followup.py` accepts repeatable
  `--configuration` selections and `--skip-official` while preserving its
  original default plan; `scripts/start_multisweep_followup.sh` exposes those
  options through `CONFIGURATIONS`, `SKIP_OFFICIAL`, and `INCLUDE_LEGACY`;
  `scripts/test_multisweep_teacher.py` covers a depth-only two-configuration
  plan. No model architecture, loss, checkpoint format, or official Simple-BEV
  path changed.
- Reproduce the depth-first plan: `STUDY_DIR=$PWD/fullsize_baseline/runs/
  depth_priority_20261016 RUN_SERVICE=dd2414-depth-priority
  CONFIGURATIONS=5-radar,5-radar-dav2 SKIP_OFFICIAL=1 INCLUDE_LEGACY=0 bash
  scripts/start_multisweep_followup.sh`. This schedules, for each depth
  treatment, full-data pretraining followed by frozen 1x1 probing, official-head
  finetuning, same-head unfrozen probing, and the GT-location motion readout.
- Tests: `PYTHONPATH=scripts .../envs/bev/bin/python -m unittest
  scripts.test_multisweep_teacher` passes 15 tests; the selected plan contains
  exactly ten depth stages and no redundant official-checkpoint evaluation.
- Limitations: these priority runs intentionally retain the current 112x192,
  randomly initialized trainable ResNet-101 and one-epoch screen so they remain
  comparable to the completed no-depth five-sweep control. Sparse radar depth
  is not dense ground truth, and Depth Anything V2 contributes only relative
  inverse-depth shape. Results are meeting diagnostics, not the final fair
  Simple-BEV comparison.
- Next: after collecting the no-depth/radar-depth/radar-plus-DAV2 table and
  masks, implement a separate opt-in 448x800 protocol initialized from
  `ResNet101_Weights.DEFAULT`, rerun the double-DINO pretraining, and compare
  matched end-to-end and frozen-probe results with official Simple-BEV metrics.

### 2026-10-16 — Meeting figures from the completed five-sweep experiments

- Changed files: `scripts/render_multisweep_meeting.py`, a narrow `.gitignore`
  allowlist for `artifacts/meeting_20261016/`, this handoff, and the generated
  meeting artifacts in that directory. The CPU-only renderer reads existing
  summaries, per-frame records, saved mask panels and training metrics. It
  neither loads model checkpoints nor occupies the training GPU.
- Deliverable: twelve English 16:9 PNG slides, `meeting_results.pdf`, a contact
  sheet, English `SPEAKER_NOTES.md`, raw-fraction `metrics.csv`, and
  `evidence.json` with timestamped source-file SHA-256 fingerprints. Four fixed
  validation scenes directly compare GT, probability, prediction and error
  masks for the frozen probe and end-to-end model. Additional slides explain
  architecture, global IoU, precision/recall and missed vehicles, distributions
  across 150 scenes, independent GT-velocity readout, pretraining trends,
  preliminary depth trends, and the formal high-resolution experiment.
- Reproduce: `MPLCONFIGDIR=$PWD/.cache/matplotlib
  $HOME/miniconda3/envs/bev/bin/python scripts/render_multisweep_meeting.py`.
  The output is a static snapshot. Regenerate after depth stages complete;
  completed depth probe/finetuning summaries populate the depth comparison
  table automatically, while incomplete results remain `Pending`.
- Actual evidence checks: generation verifies all 6,019 frames for final
  segmentation metrics; recomputes IoU/precision/recall/F1 from confusion
  counts; checks unique, identical frame-token order and per-frame count sums;
  verifies shared manifest hashes and all 150 paired validation scenes. It
  successfully renders 12 slides. PDF metadata confirms twelve 16:9 pages;
  the contact sheet and a full-size mask slide were visually inspected.
- Key findings: frozen/same-head-unfrozen/end-to-end Vehicle IoU is
  `10.448% / 25.182% / 30.020%`. The end-to-end model misses `60.145%` of valid
  GT vehicle cells. Same-head supervised adaptation improves scene IoU in
  all 150 validation scenes. Velocity EPE with matched/empty/wrong radar is
  `1.323 / 2.323 / 2.044 m/s`; the benefit is concentrated in moving vehicles.
- Correction: current saved official 448x800 summaries give `47.247%` camera
  and `55.495%` camera+radar global Vehicle IoU, not the `47.56% / 55.81%`
  previously mentioned in conversation. Figures use the saved summaries.
- Limitations: one seed, random ResNet initialization, 112x192 student images,
  one pretraining and one downstream epoch; low-resolution official-checkpoint
  inference is not matched retraining. Velocity uses GT vehicle positions.
  Depth curves are explicitly marked as a training snapshot at radar anchors;
  independent dense depth GT and final depth-treatment segmentation results
  are not yet available. None of these figures establishes a DINO benefit
  over a same-architecture no-pretraining control.
- Next: regenerate for the group meeting after depth validation is available;
  retain the selected scenes and full-validation metrics, then execute the
  separately planned ImageNet-initialized 448x800 matched experiment.

### 2026-11-03 — Full-only entry points and test-provenance admission audit

- Scope: the owner requested removing subset experiments, establishing a
  non-leaking train/validation/test protocol, obtaining official test if usable,
  then automatically running the eleven-task high-resolution study. This commit
  completes subset-entry cleanup and the test feasibility audit; it does not
  claim the requested new model trainer or scheduler has been implemented.
- Changed files: the active manifest, `scripts/radar_scaling_data.py`, the stream
  check and batch-equivalence check, legacy full-data baseline/linear-probe/
  downstream/multi-sweep trainers, `scripts/run_fullsize_baseline.sh`, and the
  full-manifest tests. Numeric scale construction, prefix selection and the
  4,096 launcher branch are removed. Formal sample-count options reject bounded
  training and validation. Historical results are retained; old pinned resumes
  require their original manifest/code instead of bypassing the changed hash.
- New files: `scripts/audit_full_data_protocol.py`, its tests,
  `FULL_DATA_TEST_PROTOCOL.md`, and `artifacts/full_data_protocol_audit.json`;
  the artifact allowlist and current README/old-plan status are updated.
- Actual download: fetched 70,803,751 bytes of official test metadata to the
  ignored `.cache/nuscenes/v1.0-test_meta.tgz`. SHA-256 is
  `c5f5169875eead4874fc345235e3fc9a4b60dcad42a18386a824a1d1d58f3ed6`.
  Reading the archive directly verified 150 scenes / 6,008 samples, zero
  sample annotations, and zero instances. Sensor blobs are not downloaded:
  they cannot supply missing GT for local Vehicle BEV IoU or GT-box velocity.
- Reproduce audit: `PYTHONPATH=scripts MPLCONFIGDIR=$PWD/.cache/matplotlib
  $HOME/miniconda3/envs/bev/bin/python scripts/audit_full_data_protocol.py
  --output artifacts/full_data_protocol_audit.json --require-ready`.
  Expected exit code is 2 while no valid test protocol is selected. The eleven
  pending records are task specifications, not a running executable queue.
- Actual checks: all 65 CPU tests pass with `PYTHONPATH=scripts .../envs/bev/bin/python
  -m unittest discover -s scripts -p 'test_*.py'`; shell syntax passes;
  `SCALE=4096 bash scripts/run_fullsize_baseline.sh` correctly refuses subset
  execution; `git diff --check` passes. Full manifest membership matches official
  metadata and train/val overlap is zero for samples, scenes and logs. The
  requested RGB checkpoint exists and its SHA-256 is recorded in the audit.
- Blocker/limitations: partitioning official train cannot erase pretrained
  exposure; splitting historically evaluated val cannot create a pristine
  blind test. The audit includes a metadata-only whole-log holdout proposal,
  explicitly inactive and historically observed. No new scientific training,
  final-test scoring, high-resolution GPU pilot or scheduler launch occurred.
- Next: obtain the owner's choice of labeled test protocol and acceptable
  historical exposure. Then implement the high-resolution strict RGB encoder
  initialization, two-location probes, training-time velocity controls, binding
  of caches/checkpoints to the new approved split, and persistent eleven-task
  queue with a measured CUDA/resume gate. Do not silently change initialization
  or call an unlabeled inference set a scored test set.

### 2026-11-03 — Official train/val approval and executable high-resolution queue

- Scope/decision: the owner chose the original official split. All eleven
  experiments train on 28,130 frames / 700 scenes and validate on 6,019 frames /
  150 scenes. Sample/scene/log overlaps are zero. The inactive 75/75-scene
  holdout alternative is not selected. Test blobs are not needed for these
  validation experiments, and no test-set or validation-label gradient updates
  are introduced.
- Changed implementation: `scripts/audit_full_data_protocol.py` and its compact
  artifact admit the explicitly selected protocol; the new config freezes
  full-data budgets and checkpoint identity. `scripts/multisweep_teacher.py`
  strictly loads official `encoder.*` tensors and exposes fixed-height-mean
  camera-BEV features before fusion. `scripts/train_multisweep_teacher.py`
  supports official initialization, high image/teacher resolution, both probe
  locations, initial-camera probing, and velocity hidden throughout training
  and evaluation. `scripts/fullsize_training.py` accepts explicit DINO patch
  resolution. The new opt-in resize in `nuscenesdataset.py` matches the official
  checkpoint-era Pillow default; legacy/default preprocessing is unchanged.
- New execution/monitoring: `scripts/run_official_full_data.py`,
  `scripts/start_official_full_data.sh`, `scripts/show_official_full_data.py`,
  and `scripts/publish_official_full_data.py`. The queue waits for three idle
  GPU checks, never signals other users' jobs, performs full-manifest updates
  two and three across a process restart, then executes all eleven tasks with
  durable checkpoints, epoch validation, progress/TensorBoard, bounded retries,
  disk guard and scoped stage-result publication. Git publication failures are
  recorded without discarding training. `README.md`, the historical plan header,
  `.gitignore`, and `FULL_DATA_TEST_PROTOCOL.md` describe the new path.
- Training matrix: one ten-epoch initial frozen camera probe; two independent
  five-epoch double-DINO models (raw input velocity vs zero input velocity);
  four ten-epoch frozen pre/post-fusion probes; two five-epoch GT-object velocity
  readouts; two five-epoch supervised end-to-end segmentation adaptations.
  Camera input is 448x800, frozen DINO input 448x798 (patch size 14), all radar
  inputs use ten sweeps, depth is disabled, batch size one, BF16, eight workers.
  Warm-start encoder LR is 1e-5, other pretraining LR 2e-4, probe LR 1e-3;
  the existing 0.2/0.8 image/BEV terms and 0.5 motion auxiliary remain explicit.
- Reproduce: `bash scripts/start_official_full_data.sh`. Preview exact commands
  with `python scripts/run_official_full_data.py --study-dir
  fullsize_baseline/runs/official_rgb_full_10sweep_20261103 --plan-only`.
  Inspect with `watch -n 10 "$HOME/miniconda3/envs/bev/bin/python
  scripts/show_official_full_data.py"`, `systemctl --user status
  dd2414-official-full-data.service`, and the study's `queue.log`/`pipeline.json`.
  Safe stop: `systemctl --user stop dd2414-official-full-data.service`.
- Actual checks: 73 CPU regression tests pass via `PYTHONPATH=scripts
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  MPLCONFIGDIR=/tmp/dd2414-mpl $HOME/miniconda3/envs/bev/bin/python -m unittest
  discover -s scripts -p 'test_*.py'`. Python compilation, shell syntax and
  `git diff --check` pass. A real CPU strict load verified all 568 official
  encoder tensors and checkpoint SHA-256
  `ccaaf31b7590b73c55d67de0e70c1b99f705bedc2f773cb7124fd0344892749e`.
  The official-protocol audit exits zero. The background service was verified
  active/running with infinite runtime/stop timeout; `pipeline.json` reports
  `waiting_for_gpu`, 0/11 tasks, no training child, and about 3,578 MiB free
  while the unrelated GPU process remains active at 97-100% utilization.
- Limitations: no high-resolution CUDA update or scientific epoch has completed
  yet. CPU checks are not CUDA training evidence. The queue contains 80 training
  epochs plus validation, so no overnight finish or duration is guaranteed.
  Batch-one throughput/memory remain unmeasured until GPU admission; the CUDA
  gate stops on failure instead of silently changing the model or sample count.
  The workstation queue cannot reserve a GPU against new external jobs.
  Official RGB initialization already contains nuScenes supervised knowledge;
  new DINO pretraining is not training from a label-free initialization. The
  pre/post probes match 128-channel readout capacity but differ in feature
  compression, so they do not isolate a causal radar gain. Motion uses oracle
  GT boxes; radar reconstruction loss remains auxiliary. Earlier official exact
  reproduction covered 6,016 val frames due to drop-last; this queue scores all
  6,019. One seed and finite budgets do not establish convergence.
- Next: let the pre-existing job finish, inspect the automatic finite-update /
  resume gate and measured GPU footprint, then review full-validation IoU,
  confusion counts, masks and independent motion results as stages finish.
  Checkpoints, logs and per-object/frame records remain local; only compact
  results and code are pushed. The unrelated staged project PDF is preserved.

<!-- official-full-runtime:start -->

### Official-split full-data runtime

- Changed: compact `artifacts/official_full_data_results.json` and this runtime block.
- Queue state: `running`; last stage: `official_rgb_double_dino_10sweep_velocity_full_pretrain_camera_bev_frozen_probe`.
- Reproduce/resume: `bash scripts/start_official_full_data.sh`.
- Evidence: completed summaries require a successful worker exit and full validation;
  the JSON records exact commands, initialization, epochs and actual metrics.
- Limitations: one seed, finite epoch budgets, no unseen local test set,
  pre/post compression differs, GT boxes supplied to motion readout.
- Next: continue pending tasks or inspect the recorded failure before restarting.

<!-- official-full-runtime:end -->

### 2026-11-04 — Retire the isolated cluster deployment branch

- Request: clean up exactly `experiment/official-rgb-double-distill-cluster`,
  not the older `experiment/hybrid-teacher-10sweep-cluster` branch or the active
  `dd2414-mini-baseline` training study. No model/training-plan edits or process
  signals are part of this housekeeping task.
- Read-only admission: the target worktree was clean; its only ignored files
  were generated Python bytecode caches. No dataset, environment, checkpoint,
  local configuration or training results were present there. Local and public
  HTTPS remote inspection resolved the branch to
  `dbaada28a7d132a7769c2f22c26b51c223278d12`.
- Recovery: created and verified the approximately 27-KiB incremental bundle
  `.cache/branch_backups/official-rgb-double-distill-cluster-dbaada2.bundle`.
  It includes the cluster commit and requires base `ec95b07`, which remains
  in the active branch history. The ignored bundle is local only, not staged.
  To restore the local branch if explicitly desired, use `git fetch
  .cache/branch_backups/official-rgb-double-distill-cluster-dbaada2.bundle
  refs/heads/experiment/official-rgb-double-distill-cluster:refs/heads/experiment/official-rgb-double-distill-cluster`.
- Completed: `git worktree remove
  /home/students2026/dd2414/.cache/worktrees/official-rgb-double-distill-cluster`
  and `git branch -d experiment/official-rgb-double-distill-cluster` succeeded
  without forcing worktree removal. The source checkout and disposable bytecode
  caches were removed; only the verified recovery bundle is retained.
- Initial authentication blocker (resolved on retry): `git push origin --delete
  experiment/official-rgb-double-distill-cluster` failed with
  `Permission denied (publickey)`. The earlier SSH remote read failed the same
  way. Public HTTPS `ls-remote` confirms the remote branch is still present;
  its remote-tracking ref is deliberately not pruned or falsely reported gone.
  That attempt could not delete the branch or push the handoff record.
  No credentials/configuration were changed by the agent.
- Changed repository file: only `PROJECT_HANDOFF.md`, to record the requested
  retirement, exact targets, recovery, checks and authentication limitation.
  Checks: bundle verification passes; the local branch and extra worktree are
  absent; `git diff --check` passes. CPU/CUDA regression runs are unnecessary
  for a metadata/documentation-only cleanup. The live queue remains running
  at its velocity-present pre-fusion probe (about 5.44/10 epochs at inspection),
  and the unrelated staged `DD2414_Project_Plan.pdf` is untouched.
- Owner-requested retry: SSH remote access succeeded. The remote target still
  matched the backed-up `dbaada2` commit. `git push origin --delete
  experiment/official-rgb-double-distill-cluster` succeeded, and a subsequent
  `git ls-remote --heads origin
  refs/heads/experiment/official-rgb-double-distill-cluster` returned no ref
  with exit code zero. Both the local and remote branch are now absent.
- Reproduce verification: `git branch --list
  experiment/official-rgb-double-distill-cluster`, `git worktree list`, and the
  exact remote lookup above. This follow-up changes only this handoff document;
  `git diff --check` passes. The preserved bundle remains available for recovery.
- Next: push the retirement and resolution records on `dd2414-mini-baseline`,
  verify the remote HEAD, and continue the original training plan. Do not merge
  the retired code or modify the unrelated staged project PDF.
