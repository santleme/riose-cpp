# Water-intake data access and transfer audit (2026-10-22)

## Decision

Two recent WVU water-intake papers are useful scientific references but their animal-level measurements are not currently available for Campo Vivo training. The 2026 water-intake LSTM paper reports that data may be available from the corresponding author; its supplementary DOCX is not the training dataset. Do not copy summary tables into pseudo-row data, scrape figures as labels, or treat the published metrics as a Campo Vivo validation result.

## Sources and access

| Source | Published evidence | Data access status for Campo Vivo |
|---|---|---|
| [Characterization of individual beef cattle water intake, skag054](https://academic.oup.com/jas/article/doi/10.1093/jas/skag054/8494533) | More than 130,000 individual daily WI, DMI, and BW records across reported breeds and systems; the article includes aggregate analyses. | No raw-data repository, downloadable individual records, code, or explicit data-availability statement found in the article/source audit. The article lists CC BY-NC-ND 4.0; that is the article's reuse license, not a declared license for the raw measurements. |
| [PSXI-25 conference abstract, 2024](https://academic.oup.com/jas/article/102/Supplement_3/770/7757864) | Describes 2,020 WVU animals and 149,292 daily WI observations collected through RFID-enabled systems. | Abstract only; no individual data or data supplement located. This appears to describe the same WVU instrumentation/collection program as skag054, so it is not an independent validation source. |
| [Predicting individual water intake with LSTM, skag138](https://academic.oup.com/jas/article/doi/10.1093/jas/skag138/8665117) | LSTM uses animal/feed/weight/weather time series; reported RMSE 3.85 L/d, R² 0.74, and drylot validation RMSE about 4.2 L/d. The official article has a supplementary DOCX and says data may be available by contacting the corresponding author. | Article and supplement are methodology/results, not raw cohort rows. No model checkpoint, reproducible training bundle, or verified public Kaggle/Hugging Face copy identified. Need author-provided dataset and terms before reproduction. |
| [Generalizable grazing DMI LSTM, skag133](https://academic.oup.com/jas/article/doi/10.1093/jas/skag133/8675989) | A 3-layer LSTM uses 7-day engineered sequences of intake, animal, and weather covariates; reports pooled RMSE 1.329 kg DMI/d against 1.858 for NASEM on its evaluation masks. | It is a strong architecture and validation-design reference, not an available pretrained model or Campo Vivo evidence. No underlying animal-level data or weights were found in a public repository during this audit. |

Searches of indexed Kaggle, Hugging Face, GitHub, Zenodo, Dryad, and Figshare records by title/DOI did not identify a source mirror. This is a time-bounded search result, not proof that a private or unindexed deposition does not exist. The data catalog records this status explicitly.

## Biological and validation interpretation

The 2026 intake-characterization paper reports meaningful class, breed, system, and THI differences, while BW and DMI alone have weak association with WI (R² reported as 0.06). The authors explicitly limit extrapolation: no Bos indicus genetics or warm-season grasses, and grazing observations are crossbred steers only. Its US data therefore cannot directly validate tropical Nelore water behavior even if the raw file is later shared.

The skag138 LSTM estimates a measured daily intake outcome from time series. It is not an animal policy for whether, where, or when to seek water. Its published evaluation is useful as a candidate supervised forecaster, not evidence that the model learns thirst, hydration, shade preference, or causal response to a farm intervention. The study's reported data-availability statement requires contacting the corresponding author; until access is granted and its terms are reviewed, no reproduction claim is possible.

The skag133 DMI work is similar: measured water intake is an input to DMI estimation, so deploying it to select a drinking action would reverse the prediction direction and introduce circularity. Its unseen grazing test holds out animal IDs within a Wardensville grazing cohort, while the geographically external tests are drylot. Thus, evidence is strongest for transfer to those specific held-out drylot cohorts; same-site/year grazing animal holdout is not an unseen grazing-farm validation. The methods also describe retrospective backward filling for short missing gaps. That may be acceptable for retrospective reconstruction, but an online simulator must not use future observations; re-run with causal forward-only imputation or explicit missingness before claiming deployable forecasts.

The papers are observational prediction analyses, not randomized causal evaluations of shade, water restriction, pasture rotation, or supplementation. Predictive fit cannot establish those intervention effects.

## Actionable use in Campo Vivo

1. Request data dictionary, row-level WI/DMI/BW/weather files, split definitions, preprocessing and feature code, model config/weights, and a written data/model reuse license from the corresponding authors. No outreach has been sent.
2. If access is granted, register a versioned private Kaggle Dataset and verify source checksums. Keep any redistribution or public checkpoint blocked until terms explicitly allow it.
3. Reproduce skag138 as an independent forecasting benchmark with animal-grouped and site/time holdouts. Add a strict prospective replay in which all engineered features and imputation stop at the forecast cutoff.
4. For the DMI LSTM, maintain the directional contract: measured water/weight/weather -> DMI estimate. Do not use it as behavior policy or feedback it into a water-choice model without a separately identified causal model.
5. To validate tropical Nelore physiology, acquire an independent warm-season/Bos indicus cohort or design a prospective collection. Until then, use the literature equation only inside its published domain and report extrapolation explicitly.

## Provenance notes

- skag054: [DOI landing page](https://doi.org/10.1093/jas/skag054), [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13023050/), article-level license CC BY-NC-ND 4.0.
- skae234.868: [conference abstract](https://academic.oup.com/jas/article/102/Supplement_3/770/7757864), [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC11400759/).
- skag138: [official article](https://doi.org/10.1093/jas/skag138), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42054591/); official article search record includes a supplementary DOCX and the data-availability statement.
- skag133: [official article](https://doi.org/10.1093/jas/skag133), [full text indexed through PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13186321/), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42114127/).
- This audit does not download restricted measurements or reproduce any paper's model.
