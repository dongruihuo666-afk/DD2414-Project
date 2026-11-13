# External source and asset provenance

No external weights/data/credentials are included in the package or Git cleanup.
The reviewable pins are [configs/assets.json](../configs/assets.json), inherited
from actual study records, [cluster asset pins](../cluster_rpl/assets.json) and
[the pre-cleanup inventory](../research/pre_integration/inventory.json).
This cleanup does not silently replace a dependency with the current Git tip.

| Component | Exact source / acquisition | Validation |
| --- | --- | --- |
| Original Simple-BEV | Upstream `aharley/simple_bev`, `be46f0ef71960c233341852f3d9bc3677558ab6d`; local legacy path retained | Strict full-model tensor loading, original LICENSE/citation |
| Official RGB weights | Original `get_rgb_model.sh` Dropbox archive; run explicitly in an asset staging directory, then retain extracted checkpoint | Pinned checkpoint SHA-256 in assets; research selects all `encoder.*`, strict=True, 568 tensors |
| Official camera+radar weights | Original `get_rad_model.sh`; unchanged acquisition script | Retain original checkpoint provenance/reproduction record; do not substitute BEVCar weights |
| BEVCar VoxelNet | `robot-learning-freiburg/BEVCar`, `29cacda3bc5416d47428c1d0f017527acad34f90` | Hash `nets/voxelnet.py`; only radar encoder loaded, not complete BEVCar |
| DINOv2 ViT-S/14 | Recorded Torch Hub cached source; original cache has no Git metadata | Actual source-only tar archive + per-file inventory, weight hash; no invented Git revision |
| Depth Anything V2 | `DepthAnything/Depth-Anything-V2`, `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` | Commit, selected file/tree hashes and strict weight loading |
| Relative Depth Anything Small | Official project weight `depth_anything_v2_vits.pth` (historical relative-depth arm) | Separate pinned weight hash; not metric depth |
| Metric Depth Anything Small | Official `Depth-Anything-V2-Metric-VKITTI-Small`, `depth_anything_v2_metric_vkitti_vits.pth` | Pinned metric hash, outdoor 80 m maximum; separate source namespace |

For exact BEVCar source acquisition:

```bash
git clone https://github.com/robot-learning-freiburg/BEVCar.git /sources/BEVCar
git -C /sources/BEVCar checkout 29cacda3bc5416d47428c1d0f017527acad34f90
python tools/verify_asset.py --asset bevcar_voxelnet --path /sources/BEVCar/nets/voxelnet.py
```

Use the corresponding pinned Depth Anything commit, obtain its named official
weight separately, and verify using `tools/verify_asset.py`. The helper only
reads/hashes files. Neither it nor ordinary imports access the network.

For exact historical DINO source, unpack the retained source archive into the
expected `facebookresearch_dinov2_main` directory after verifying the archive
hash. The tar contains that directory's contents, not weights. Historical
training expects its separately verified weight in
`.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`. Set `TORCH_HOME` to the
matching checkout's `.cache/torch` when needed; existing driver provenance uses
that checkout path too. An uncached **actual** legacy training call may invoke
Torch Hub download as before; prepare the verified cache first for offline,
reproducible execution. Help/import does not call Torch Hub.

Inventory source-tree hashes and cluster Python-tree hashes use different
algorithms/selection and must not be compared as if they were the same field.
Likewise the frozen manifest has a raw-file SHA-256 (`f46868da...`) and its
existing canonical unsigned-JSON content digest (`dc39006d...`); both are
preserved, not substituted for one another.
The inventory lists individual file digests and the archive digest; checkpoint
provenance retains its own Python-tree scheme. Never weaken either validation.

External clones/source archives are subject to their original licenses and
copyright notices, which are retained with the source. Package installation
does not grant access to nuScenes: obtain it under the original data terms.
No external clone was modified, replaced or downloaded by this cleanup.
