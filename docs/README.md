# Documentation index

## Current team interface and operation

- [Installation and recommended commands](INSTALL.md)
- [Tensor/data/coordinate contract and feature API](INTERFACES.md)
- [Architecture and research boundaries](ARCHITECTURE.md)
- [External sources and pinned assets](EXTERNAL_ASSETS.md)
- [Portable deployment and explicit publication](DEPLOYMENT.md)
- [Maintenance, review and source provenance](MAINTENANCE.md)
- [Tests and opt-in integration programs](../tests/README.md)
- [Cleanup validation and path mapping](integration/VALIDATION.md)
- [Snapshot/recovery refs](integration/BACKUP.md), [checkpoint compatibility](integration/COMPATIBILITY.md)
- [Short current handoff](../PROJECT_HANDOFF.md)

## Current experimental protocols

These are retained at their original addresses for source/hash and command
compatibility; they are not archived as obsolete notes:

- [Official pretrained RGB encoder, full-data ten-sweep study](../FULL_DATA_TEST_PROTOCOL.md)
  with [frozen protocol config](../configs/full_data_official_protocol.json).
  Its eleven-stage results are committed in
  [official_full_data_results.json](../artifacts/official_full_data_results.json).
- [Random ResNet101, metric-BEV depth, 50 total epochs](../CLUSTER_RANDOM_DEPTH_50EPOCH_PLAN.md).
  Two treatments: DINO only and DINO plus metric pseudo-depth. Both use velocity,
  ten sweeps and the same probability-based lifting. Its job ledger is a dated
  snapshot; check live execution separately before an operational action.

These protocols have different initialization, losses and budgets; do not
combine their numbers as a controlled single-factor result. A new experiment
requires its own output/config/provenance; directory cleanup grants no launch.

## Research history and reports

- [Pre-integration history](archive/README.md): plans, technical notes, progress,
  the full chronological handoff and original upstream README/citation.
- [Report and evidence archive](../research/README.md), including the
  [16 October meeting packet](../artifacts/meeting_20261016/SPEAKER_NOTES.md).
- [Historical experiment programs](../experiments/README.md).
- [Radar encoder technical notes](reference/RADAR_ENCODER_NOTES.md).

Data manifests stay in `configs/` at their recorded paths. They are not model
hyperparameters and were not regenerated or reordered during cleanup.
