# CNR temporal activity experiment v1

## Purpose and scientific scope

This experiment asks a narrow question: does a small temporal convolutional classifier improve **observed activity classification** over hand-engineered acceleration features when a whole cow is held out? It does not train an autonomous bovine policy. The source contains no water/forage state, physiological need, location, resource choice, social context, or action consequences, so it cannot identify those decisions.

The source is the Kaggle [Cow nose ring data set](https://www.kaggle.com/datasets/fandaoerji/cow-nose-ring-data-set), whose Kaggle metadata declares CC0-1.0. The downloaded CSV is pinned by SHA-256 in `training/train.py`. The original paper says seven healthy cows from one farm were recorded over eleven days, with a nose-mounted accelerometer and video-synchronized activity labels; its reported ontology is feeding, rumination, and other, not five balanced independent postures. [Feng et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11047668/)

## Frozen protocol

- Parse contiguous 10 Hz runs using the shared timestamp-aware extractor. The source CSV stores whole-second timestamps repeated across rows; order within each verified block is assumed because the file has no sub-second samples. Never join across incomplete or ambiguous blocks.
- Use non-overlapping five-second windows and require at least 80% of rows to share the window label.
- Map source labels 0→feeding, 1→rumination, and 2/3/4→other. This follows the source paper's coarse ontology, but `other` combines standing, lying, and walking and cannot support claims about posture recognition.
- Compare a training-majority baseline, frozen ExtraTrees on 18 summary features, and a small three-layer temporal CNN on raw 50×3 accelerometer samples.
- Use seven leave-one-cow-out folds. Every window from one animal remains in its single test fold; there is no model selection using outer-fold scores. The CNN has fixed 35 epochs, AdamW, inverse-frequency class weights, seed 20261022, and CUDA is mandatory for Kaggle runs.
- Report macro-F1, weighted-F1, per-class F1, confusion matrices, mean and standard deviation across held-out cows, source checksum, extraction diagnostics, model settings, and runtime GPU.

## Existing evidence and limits

The earlier five-class exploratory CNR leave-one-animal-out experiment achieved mean macro-F1 0.490 (SD 0.069) with ExtraTrees and 0.471 (SD 0.058) with a CUDA MLP. Standing had only 18 accepted windows and lying 37; their per-class F1 was near zero. These scores are descriptive, post-selection results from one cohort, not an independent validation claim. The current coarse three-class comparison changes the task definition and should not be compared directly against those five-class scores.

A 2026 cross-dataset cattle study reports that transfer varies sharply with dataset pairing and label conventions; one reverse transfer direction reached 47.5% accuracy / 0.324 weighted-F1 while the opposite direction was much stronger. It also reports dataset 1 accuracy falling from 92.7% on a stratified 80/20 split to 79.2% on leave-one-individual-out. This is a concrete warning that window-level splits can preserve individual signatures and overstate generalization. The Campo Vivo protocol keeps all windows from one cow in the same fold. [James et al. 2026](https://www.frontiersin.org/journals/animal-science/articles/10.3389/fanim.2026.1676504/full)

ActBeCalf offers a valuable next external activity test: 30 pre-weaned calves, 27.4 annotated hours, 23 ethogram behaviors, four group pens, and 25 Hz neck-collar acceleration. It is a distinct age, breed, placement, frequency, site, and observation design; its selected annotation windows do not directly estimate daily behavior budgets. [Dissanayake et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11994922/), [Zenodo record](https://zenodo.org/records/13259482)

## Execution record

The Kaggle entry point is `training/kernels/cow-activity-temporal-v1/activity_tcn_bundle.py`; it is generated from `training/train.py` and `training/activity_tcn_loso.py` by `training/build_activity_tcn_bundle.py`. The kernel uses the CC0 Kaggle dataset and requests a Tesla T4-compatible GPU. Its output is private to the Kaggle account and is not a public model or dataset release.

Kaggle kernel version 1 failed before training because the generated single-file entry point placed a second `from __future__ import annotations` after shared module code. The bundler was corrected, and version 2 completed on a Tesla T4 in about 105 s to the last metric. It verified the pinned CSV hash and extracted 18,446 windows. Aggregated results (macro-F1, mean ± SD across the seven held-out cows) were:

| Candidate | Macro-F1 | Per-class mean F1: feeding / rumination / other |
|---|---:|---:|
| Training-majority baseline | 0.223 ± 0.035 | 0 / 0 / 0.668 |
| ExtraTrees | 0.768 ± 0.084 | 0.803 / 0.675 / 0.826 |
| Temporal CNN | 0.754 ± 0.111 | 0.804 / 0.651 / 0.808 |

The CNN did not beat ExtraTrees on the across-animal macro-F1 mean and had higher fold variability. Its cow-7 fold scored 0.562 versus 0.766 for ExtraTrees. The paired CNN-minus-ExtraTrees fold difference was -0.0137 macro-F1 (SD 0.0921; t-based 95% interval [-0.0989, 0.0715], n=7 cows), which does not support a reliable advantage for either candidate. This interval is only descriptive: folds share a cohort and are not independent farm replications. Neither candidate is selected for simulator integration. Version 2 output JSON SHA-256 is `8b35b98ef960b512cf89f9b67335b84c1b3db61a495dee0b635fc3e3aef023a2`; Kaggle log SHA-256 is `1f57ff6ecb48dd9b5a5d4c062e2e659929c6067701e4a59a66a1ee21e56543b4`. Aggregated, de-identified v2 metrics are stored in `training/experiments/cow-activity-temporal-v1-v2-summary.json`.

Kaggle kernel version 3 reran the same fixed candidate settings on a Tesla T4, reproduced all v2 metrics, saved seven per-fold temporal-CNN checkpoints (one held-out-cow fold each), and directly recorded 15,260 optimizer updates. Version 4 repeated this run with tensor-only normalization metadata and an in-kernel `weights_only` reload check; all seven artifacts reloaded successfully. Its seven checkpoint hashes were verified against the output manifest after download. The recoverable output is in ignored path `training/outputs/activity-tcn-loso-v4/training/outputs/activity-tcn-loso`; checksums, per-fold step counts, evaluation and log hashes are in `training/experiments/cow-activity-temporal-v1-v4-summary.json`.

## Acceptance boundary

This experiment can establish only whether a temporal classifier recognizes three coarse observed activity labels on held-out cows from this same single-farm cohort. It cannot establish behavioral autonomy, internal needs, physiology, causality, social interaction, navigation, pasture choice, water-seeking, or farm-level generalization. Do not wire this output to the simulator's decision policy: simulated bovines do not emit nose-ring samples, and the dataset has no labeled action/context pairs.
