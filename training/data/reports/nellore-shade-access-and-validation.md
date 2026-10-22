# Mendeley Nellore shade dataset: access and validation status

**Status checked:** 2026-10-22. **Outcome:** no anonymous file-level access; no workbook was downloaded. There are no row-level bias, MAE, calibration results, or holdout metrics in this report.

## Access and provenance

The record is Mendeley Data DOI [10.17632/cctgpznjrc.1](https://data.mendeley.com/datasets/cctgpznjrc/1), version 1. The public landing page responds HTTP 200, identifies the shade study and states CC BY 4.0. It describes four sheets/categories: drinking behaviour, dry matter intake, liveweight, and weather. The landing page says there are 19,388 drinking events, 3,854 DMI rows, 282 liveweight records, and 7,968 records for each weather metric at 15-minute intervals.

The official API probes were made without credentials and without following redirects:

| Resource | Result | What can be concluded |
| --- | ---: | --- |
| Mendeley landing page | HTTP 200 | Public metadata and stated CC BY 4.0 license are visible. |
| Public file listing API | HTTP 401 | No anonymous file listing/schema available through this route. |
| Public ZIP API | HTTP 401 | No anonymous archive available through this route. |
| DataCite DOI metadata | HTTP 200 | Rights list includes both `embargoedAccess` and CC BY 4.0; it records an available date of 2025-08-14. |

HTTP 401 establishes that these API file routes require authorization in this run; it is not itself a 451 embargo response. The official API documentation specifies OAuth2 for the public dataset file endpoints. The contradictory landing-page/DataCite rights and file API status are preserved as unresolved metadata. No credentials were used, redirects were not followed, and no attempt was made to circumvent a restriction. Consequently, filenames, file hashes, spreadsheet schemas, units recorded in the workbook, individual-to-pen mapping, and treatment labels at row level remain unverified. The reproducible probe is [`audit_nellore_shade.py`](../audit_nellore_shade.py); it does not save raw records.

## Study design from the primary paper

The paper reports 48 steers initially, with one removed after leg injury; 47 Nellore steers remained for 83 days in four 400 m² collective pens. Shade was applied to two pens, with two pens unshaded. It reports average water intake of 35.1 L/day with shade and 37.9 L/day without shade; average DMI was 11.6 kg DM/day. The reported maximum daily air temperature averaged 29.1 °C and mean relative humidity 70.8%. The paper considered 47 individual animals as experimental units, despite treatment being assigned at the pen/group level. This creates clustered treatment data and pseudoreplication risk if individual-day rows are analyzed as independent treatment replicates. A valid treatment holdout must keep whole pens together; with two pens per condition, precision and generalization are inherently limited.

There is a source-frequency discrepancy: the Mendeley page describes 7,968 weather records per variable at 15-minute frequency, while the article's methods text says hourly weather observations. At 83 days, 7,968 matches 15-minute sampling and 1,992 matches hourly sampling. Do not aggregate daily TMAX/RH or claim their time resolution until the actual workbook is accessible.

Primary references: [PLOS ONE study](https://doi.org/10.1371/journal.pone.0331238), [Mendeley dataset record](https://data.mendeley.com/datasets/cctgpznjrc/1), [official Mendeley Data API docs](https://data.mendeley.com/api/docs/), and [DataCite DOI metadata](https://api.datacite.org/dois/10.17632/cctgpznjrc.1).

## BR-CORTE 2023 equation and what can be said now

BR-CORTE 2023 chapter 14 gives the empirical daily water-intake equation (credited to Zanetti et al. 2019):

```text
WI (L/day) = 9.449
             + 0.190 × BW^0.75 (kg^0.75)
             + 0.271 × TMAX (°C)
             − 0.259 × RH (%)
             + 0.489 × DMI (kg DM/day)
```

Relative humidity is entered on a 0–100 percentage scale, not as a 0–1 proportion. The BR-CORTE chapter reports validation on 14,774 daily observations from five other Nellore experiments: observed mean 16.1 versus estimated 16.3 L/day; regression intercept −0.3109 and slope 1.005; concordance correlation coefficient (CCC) 0.574; observed range 2.0–49.9 L/day versus predicted range 2.3–32.3 L/day. This is useful prior evidence, but not validation on the 47-steer shade trial. These are repeated daily observations, not 14,774 independent animals.

The shade trial's middle-weight summary (551 kg) gives `BW^0.75 ≈ 113.7 kg^0.75`, just above the largest training-experiment group maximum reported in BR-CORTE (113.5) and within its validation summary maximum (114.2). Its final mean of 591 kg gives about 119.8 kg^0.75, above those listed group maxima. This suggests late-trial weights may test the equation near or beyond its summarized BW domain; it is not a definitive domain check because no row-level shade BW values or original calibration rows were available.

For context only, plugging the shade paper's middle weigh-in (551 kg), mean DMI (11.6 kg/day), mean daily TMAX (29.1 °C), and mean RH (70.8%) into BR-CORTE gives about **26.28 L/day**, while the paper reports group means of 35.1 and 37.9 L/day. The differences (−8.82 and −11.62 L/day) are **not bias estimates**: 551 kg is one middle weigh-in, the predictors are pooled means, and applying a nonlinear `BW^0.75` transform to a representative mean does not equal the average of individual predictions. No individual data, pen effects, residuals, MAE, or calibration can be recovered from those two published group means.

To calculate a valid comparison when files become legitimately available, require a key for animal, date, pen and treatment; verify units; join daily water intake to contemporaneous DMI, BW and weather; and compute `BW^0.75`. Report predictions against observed daily L/day with bias, MAE, RMSE, calibration intercept/slope and CCC. Split by whole pen for treatment/generalization (only four pens total); separately, use chronological periods within pen for temporal checks. Do not random-split animal-day rows. Report both pooled and treatment-stratified metrics with pen-level uncertainty. The BR-CORTE paper's own validation found predicted maximum below observed maximum, so inspect upper-tail underprediction as well.

The equation is an empirical expectation for daily demand. It does not predict thirst state, drinking-event timing, choice to seek shade, or the causal individual response to shade. The shade dataset, if opened, can validate aggregate daily intake and visit patterns in this feedlot/TMR context, not extensive pasture behavior without additional evidence.

## Reproduction

```bash
python3 training/data/audit_nellore_shade.py
python3 -m unittest discover -s training/data -v
```

The first command probes public endpoints only and writes a sanitized status JSON next to this report. It will not follow redirects or retrieve a signed download URL. The second checks the BR-CORTE calculator against the official book's worked example; it does not claim model validation against the unavailable workbook.
