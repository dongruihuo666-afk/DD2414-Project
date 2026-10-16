# Baseline meeting follow-up results

Vehicle IoU uses full-split valid-cell intersection / union.
Official references use 448x800 images; legacy downstream uses 112x192.
Different resolution, heads, supervision, and initialization prevent attribution to architecture alone.

| Experiment | Status | Vehicle IoU (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| official_camera | complete | 47.247 | 65.261 | 63.123 | 64.174 |
| Historical frozen 1x1: velocity_full_epoch008 | complete | 4.962 | 5.074 | 69.185 | 9.455 |
| Historical frozen 1x1: velocity_zero_epoch008 | complete | 4.848 | 4.955 | 69.271 | 9.248 |

Inspect each config, per-frame JSONL, and fixed-token mask PNG before drawing conclusions.
Raw/zero finetuning compares two pretraining recipes; one seed does not measure seed uncertainty.
A complete queue can contain failed stages: consult pipeline status and individual summaries.
Optional depth/object-motion outputs are diagnostics, not downstream validation improvements.
