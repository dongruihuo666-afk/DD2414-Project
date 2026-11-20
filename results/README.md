# Curated evidence only

This directory is reserved for small reviewed benchmark summaries with source,
protocol and limitations. Raw per-frame metrics, plots, PDFs, job ledgers and
checkpoints belong in ignored output storage. Original detailed evidence remains
recoverable from this branch's pre-cleanup tag. Records are dated snapshots,
not current training-status queries or newly measured results.

The [16 November cluster-depth report](../docs/CLUSTER_DEPTH_RESULTS_20261116.md)
records the completed 50-epoch random-student comparison and six downstream
evaluations, verified against the live durable backup. Generated evidence and
figures remain in ignored/local report storage.

The [direct-checkpoint metric record](pretrained_direct_evaluation_20261116.json)
was verified and published on 20 November. It records both completed, untouched
pretraining-checkpoint evaluations (1.9571% / 1.6342% Vehicle IoU), with no probe
or GT adaptation. Their archived segmentation decoder is untrained; read that
limitation and the separate supervised results in the report before comparison.
