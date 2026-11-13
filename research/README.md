# Research artifacts and provenance

This directory indexes retained research evidence. Large runtime products stay
outside Git; selected small metrics and meeting figures already tracked in
`artifacts/` keep their paths because report/readout code refers to them.

| Record | Meaning |
| --- | --- |
| [Pre-cleanup inventory](pre_integration/inventory.json) | Original branch/source inventory, exclusions and external versions |
| [DINO cached source](pre_integration/dinov2-cached-source.tar.gz) | Source-only archive with recorded file/tree/archive hashes; no weights |
| [Official full-data results](../artifacts/official_full_data_results.json) | Complete eleven-stage study snapshot, provenance and GT metrics |
| [Meeting packet](../artifacts/meeting_20261016/SPEAKER_NOTES.md) | Slides/figures/PDF/data and explanations in one dated directory |
| [Mask comparison evidence](../artifacts/fullsize_meeting_visuals.json) | Fixed selected frames and rendering provenance; used by historical visualization code |
| [Depth admission](../artifacts/rpl_depth_admission_20261106.json) | Bounded real-cluster admission evidence, not completed training |
| [Continuation repair](../artifacts/rpl_continuation_repair_20261112.json) | Dated repaired source/job/checkpoint ledger, not a live status claim |
| [Research documents](../docs/archive/README.md) | Preserved plans, progress and meeting reproduction history |

Do not delete an artifact just because its experiment is superseded. Avoid
committing new bulk image sequences, logs, datasets, weights or cache trees.
Store new run products under an explicit external run root; publish a reviewed
small evidence selection with commands, source/asset hashes and limitations.
Manifests remain separate from model parameters and deployment configuration.
