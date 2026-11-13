# Reproducibility

## Source snapshots

This experiment's original revision is `0225bbef5f69ae04e0d52adbd4ee9d393cb1678e`:

```bash
git fetch origin --tags
git worktree add --detach ../DD2414-original repository-snapshot-20261113T185518Z/dd2414-mini-baseline
```

The `repository-snapshot-20261113T185518Z/*` annotated tags preserve all original
branch heads and were checked against their peeled commits on origin. The
`original-working-tree` tag also includes the staged project PDF and five small
JSON records. A verified all-refs bundle and index/patch inventory are stored
under `.cache/repository-snapshots/20261113T185518Z/` on the workstation.
Data, weights, logs and environments are stored separately from source.

## Resuming a run

Use the source revision, environment, assets and output directory recorded by
the run. Restore the complete tensor/state checkpoint, including optimizer,
scaler, per-rank sampler, RNG and training counters. Exact restore validates
configuration and source hashes. Changed source requires a new run or an
explicitly supported continuation protocol; keep strict validation enabled.
Whole-object Python model pickles are outside this checkpoint contract.

New runs record the extracted package files in the existing implementation-hash
field. Historical checkpoints still use their original source and restore path.

## Assets

Official RGB/camera+radar acquisition remains `get_rgb_model.sh` /
`get_rad_model.sh`, executed explicitly in a staging directory. BEVCar source:
`https://github.com/robot-learning-freiburg/BEVCar.git`, commit
`29cacda3bc5416d47428c1d0f017527acad34f90`; research imports only VoxelNet.
DINOv2 ViT-S/14 weight SHA-256:
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
Official RGB checkpoint SHA-256:
`ccaaf31b7590b73c55d67de0e70c1b99f705bedc2f773cb7124fd0344892749e`.
BEVCar `nets/voxelnet.py` SHA-256:
`43ed0ee30f70c328bd47885cbc68272e7f5e3625c61b325b54f713a851f3e6ac`.
Use the initialization specified by the experiment protocol.

The recorded Torch Hub DINO source has no Git revision metadata. Its actual source-only archive/inventory is
recoverable from `pre-integration-refactor-20261113T144801Z` at
`research/pre_integration/`; no weights are in that archive.
Full-data manifests have both raw-byte and canonical unsigned-JSON hashes;
record both values when checking a manifest.

## Records and outputs

Keep reusable configuration in `configs/` and reviewed summaries in `results/`.
Generated `artifacts/`, logs, weights, old meeting records and personal assistant/
IDE settings are ignored. Shared CI remains in `.github/`. Previously tracked
records were removed from the index; their local files and Git history remain.

To retrieve a report from this experiment's snapshot:

```bash
git show 'repository-snapshot-20261113T185518Z/dd2414-mini-baseline:artifacts/meeting_20261016/SPEAKER_NOTES.md' > /tmp/meeting-notes.md
```

Historical report generators may require those artifacts and the original run
outputs. Where used, active evaluation reads fixed qualitative sample tokens
from `configs/visualization/`. Source snapshots retain old Git blobs for recovery;
the active tree is smaller, while repository history is retained.
