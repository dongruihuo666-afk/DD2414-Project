# Meeting presentation notes

Snapshot: 2026-10-16 12:09 UTC.

Open `meeting_results.pdf` in presentation mode. All individual PNGs are 16:9.
Regenerate with `MPLCONFIGDIR=$PWD/.cache/matplotlib $HOME/miniconda3/envs/bev/bin/python scripts/render_multisweep_meeting.py`.
This reads local results and uses CPU plotting only; it does not interrupt training.
The output is a static snapshot; rerun after new experiment stages finish.

## 01. What we built and what we can measure

We have implemented both DINO distillation terms and a five-sweep radar encoder. All completed student results here use a random camera initialization and 112 by 192 images. We pretrain once, then branch from that same checkpoint for the downstream experiments. The frozen probe measures linear accessibility of vehicle information; end-to-end training measures supervised adaptability.

## 02. Vehicle segmentation: two evaluation protocols

The frozen probe reaches 10.45 percent IoU. With the same linear head but a trainable backbone, IoU reaches 25.18 percent; the full segmentation decoder reaches 30.02 percent. The official weights fall to 12.44 and 22.86 percent when evaluated at our low resolution. We cannot use this table to claim superiority over matched, fully trained Simple-BEV.

## 03. Prediction masks versus GT: fixed validation scene 1

This is fixed scene 1, selected before looking at these predictions. The top row uses the frozen representation and a trained linear probe; the bottom row allows full supervised adaptation. Read GT, probability, binary prediction, and errors from left to right. Blue regions expose missed vehicles; red regions show false detections. Each frame IoU is illustrative; headline IoU uses all validation cells.

## 04. Prediction masks versus GT: fixed validation scene 2

This is fixed scene 2, selected before looking at these predictions. The top row uses the frozen representation and a trained linear probe; the bottom row allows full supervised adaptation. Read GT, probability, binary prediction, and errors from left to right. Blue regions expose missed vehicles; red regions show false detections. Each frame IoU is illustrative; headline IoU uses all validation cells.

## 05. Prediction masks versus GT: fixed validation scene 3

This is fixed scene 3, selected before looking at these predictions. The top row uses the frozen representation and a trained linear probe; the bottom row allows full supervised adaptation. Read GT, probability, binary prediction, and errors from left to right. Blue regions expose missed vehicles; red regions show false detections. Each frame IoU is illustrative; headline IoU uses all validation cells.

## 06. Prediction masks versus GT: fixed validation scene 4

This is fixed scene 4, selected before looking at these predictions. The top row uses the frozen representation and a trained linear probe; the bottom row allows full supervised adaptation. Read GT, probability, binary prediction, and errors from left to right. Blue regions expose missed vehicles; red regions show false detections. Each frame IoU is illustrative; headline IoU uses all validation cells.

## 07. What limits segmentation right now?

Precision asks how often a predicted vehicle cell is correct; recall asks how many real vehicle cells we find. Recall is only 39.85 percent for the end-to-end model, so 60.15 percent of GT vehicle cells remain missed. High background accuracy would obscure these errors because vehicle cells are rare. The plots identify the error pattern, but do not prove whether initialization, resolution or training budget caused it.

## 08. Does the improvement hold across validation scenes?

The boxplots show scene-to-scene variation, rather than selecting a few favorable images. Using the same linear head, allowing the backbone to update improves IoU in 150 of 150 scenes. This demonstrates supervised adaptability. It does not isolate DINO from initialization and does not replace repeated training seeds.

## 09. Radar contributes to vehicle velocity estimation

Velocity EPE is the Euclidean error between predicted and GT XZ velocity vectors, in metres per second. For all vehicles the error is 1.323 with matched radar, compared with 2.044 for wrong-scene radar and 2.323 with empty radar. The gain is larger for moving vehicles. The zero-velocity input slightly helps stationary vehicles, so the benefit is not uniform. GT vehicle boxes are supplied and a supervised readout is trained; this does not measure detection or nuScenes NDS.

## 10. Pretraining learns the requested teacher objectives

Both requested DINO losses decrease during the full training epoch. The image term supervises the camera encoder and the BEV term supervises the fusion features under radar support. The motion reconstruction term is an auxiliary radar target. These losses are proxies, so their decline alone cannot demonstrate segmentation accuracy.

## 11. Depth experiment: current evidence and the next test

This page is a timestamped snapshot of the ongoing depth experiment. The left curve is the sparse radar depth loss, and the right curve is training-anchor depth MAE. The same radar anchors provide supervision, so these errors are not independent dense-depth ground truth. Depth predictions actively weight lifting into BEV. We will compare no depth, radar depth, and radar plus Depth Anything V2 using the frozen probe and full downstream Vehicle IoU. Because lifting and supervision change together, we cannot attribute a future IoU gain to depth loss alone.

## 12. Next: a fair comparison at the standard resolution

Using the local official-checkpoint summaries, original-resolution Vehicle IoU is 47.25 percent for camera-only and 55.50 percent for camera plus radar. These are corrected directly from saved results. Reducing only inference resolution causes a large drop, establishing resolution sensitivity of these checkpoints. For our formal run we will use ImageNet-pretrained ResNet101, 448 by 800 inputs and ten radar sweeps. We also need a same-architecture control without DINO, matched supervised budgets and multiple seeds before crediting double distillation.

## Provenance and checks

`evidence.json` records exact source paths and SHA-256 fingerprints. `metrics.csv` stores raw fractions.
Generation verifies complete 6,019-frame results, confusion-count formulas, unique paired tokens, per-frame count totals, manifest identity, and the same 150 validation scenes.
Original-resolution references are 47.247% / 55.495% in the current local summaries; earlier conversational values 47.56% / 55.81% were not used.
