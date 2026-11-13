# Compatibility and source provenance

The package extraction changes module locations, not model computation. The
original `nets/`, `utils/`, `nuscenesdataset.py`, `train_nuscenes.py`, official
checkpoint download scripts and Simple-BEV parameter names remain unchanged.
The opt-in streamed metric lifting inherited from the depth branch remains
opt-in; no model, objective, loss weight, depth bin, velocity transform, sweep
filter, initialization or experiment budget is changed by this cleanup.

## Imports and checkpoints

Legacy helper modules under `scripts/` alias the formal module object so existing
imports and module-level test patches continue to address the same functions.
Historical demo/tiny/smoke programs keep their CLI/rendering responsibilities
and re-export extracted implementations. Do not delete them based on filenames.
Source-checkout support is centralized in `scripts/_bootstrap.py`; installed
consumers import `dd2414` directly. Explicit external-teacher source loading is
retained rather than replaced by an uncontrolled global import.

State-dictionary keys, tensor shapes, schema versions, optimizer states, RNG,
scaler and acknowledged/distributed sampler conventions remain unchanged.
The supported serialization contract is the existing tensor/state checkpoint,
not arbitrary pickling of whole Python model instances.

**Exact historical continuation requires its recorded original source.**
Source checks correctly reject a resume whose implementation hashes changed.
Use a recovery tag/worktree or the original immutable cluster source, environment,
assets and output directory. No code translates historical hashes, disables
`strict=True`, resets optimizer/sampler/RNG, or restarts counters to claim resume.
Installing this package does not make old source/config hashes interchangeable.

New training entries include every actual formal Python implementation in the
existing `model_data_implementation_sha256` field. An installed wheel differing
from the nominated source checkout fails before recording valid provenance.
This extends content verification without inventing a checkpoint schema or
accepting weaker source checks. Existing running allocations use their old
fixed source and are unaffected by installation in a separate development path.

## Numerical verification

The pre-extraction CPU baseline passed 103 tests. After extraction, the same
103 checks pass; the loader-routing AST test now reads the extracted production
definitions and still checks both mini and trainval, sweep count and worker
routing with a fake SDK. No dataset call is made by that test.

`tests/parity/probe.py` executes original and extracted production functions in
separate CPU processes with fixed seeds/inputs. Compare its JSON outputs:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tests/parity/probe.py --implementation legacy --source-root /path/to/preserved-depth-checkout > /tmp/before.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tests/parity/probe.py --implementation package --source-root "$PWD" > /tmp/after.json
python -c 'import json; assert json.load(open("/tmp/before.json")) == json.load(open("/tmp/after.json"))'
```

Actual result: all TEN groups match exactly, including tensor hashes of losses,
gradients, toy-student wrapper state dictionaries, streamed projected depth,
radar rotation, empty-motion connected gradients, sampling/LR schedule and
post-resume optimizer/model state and calibrated input/voxel preparation. These are controlled CPU fixtures, not full
ResNet/VoxelNet CUDA inference or a scientific-quality comparison.
