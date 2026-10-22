# Campo Vivo data ingestion

The catalog in [`datasets.json`](./datasets.json) records stable source identifiers, version, license, paper, field scope, and known limitations. No third-party records are committed. Raw downloads and generated outputs live under the root `data/` directory, which is git-ignored.

## Kaggle: Nellore electronic feed-bin events

This is the first runnable data pipeline because the public Kaggle archive can be fetched without credentials and has a documented primary paper. The paper describes 37 young Nellore males observed for 21 days in one paddock with five GrowSafe feed bins. The inspected Kaggle v2 archive has 90,211 rows and only `animal_id`, `equip_id`, `IN`, and `OUT`. These are feeder access events. They are not a general activity ethogram, pasture movement, or water intake. Event duration is safely derivable from `OUT - IN`; feed consumption is not present in this Kaggle file.

The article reports feed-consumption records in an underlying table and the CRAN `socialh` package distributes a 7-column object with `duration`, `consumption`, and `pen`. The article and package documentation do not establish that those enriched columns match the current Kaggle rows one-to-one. The ingestion intentionally excludes these fields. Reconcile the R object with this Kaggle version before using consumption or pen.

The paper reports 37 animals, but the downloaded Kaggle v2 file contains 38 distinct IDs; one ID has only one event. The audit records this mismatch. Do not silently remove that row or declare the file a 38-animal cohort.

The dataset license is CC BY-NC-SA 4.0. It is suitable for noncommercial research subject to attribution and share-alike obligations, but it is excluded from commercial redistribution by its stated terms. This repository does not redistribute the file. Obtain a separate data-use determination before using it in a commercial product or publishing model weights derived from it.

## Download and validate

Requires Python 3.10+ and the standard library only:

```bash
python3 training/data/ingest_nellore_trough.py --download
python3 training/data/ingest_nellore_trough.py --input data/raw/kaggle/nellore-trough-v2.zip
```

The second command validates the archive, emits normalized records, generates deterministic animal-group and chronological split indices, and writes an aggregate audit report. It never outputs animal IDs into the report. Events spanning midnight across a chronological partition boundary are marked `purged`; they are not assigned to train/validation/test. The split CSV is local data and stays git-ignored.

The time split is a chronology check within this 21-day cohort, not an independent farm test. The animal holdout is the primary estimate of unseen-animal transport; it still cannot establish between-farm or pasture generalization. Do not split random event rows: repeated observations from the same animals and the same 21-day management context would leak across partitions.

## Brazilian physiology/environment references

The UFMS Mendeley dataset (DOI `10.17632/3wdjvxmvsp.7`, CC BY 4.0) is a stronger public source for interval-scale weight, supplementation, weather, and satellite proxies. It has 26 Nelore in one paddock over roughly one year, with measurements every 13–28 days. It is observational and its vegetation indices are not a direct measure of forage biomass or chemistry.

The shade Mendeley dataset (DOI `10.17632/cctgpznjrc.1`, page license CC BY 4.0) reports 47 Nelore steers, 83 days, 19,388 drinking events, daily DMI, six liveweight timepoints and weather. The paper says shade was applied to two of four pens, so treatment replication is two pens per group; the paper analyzes individual animals as experimental units. For treatment effects, that creates pen-level clustering/pseudoreplication risk; animal rows are not 47 independent treatment assignments. The dataset page lists four worksheets, but the file-level workbook schema has not been verified here. DataCite metadata also carries an `embargoedAccess` rights URI, so verify current file access before building an automated Mendeley fetcher. This pipeline records the source and limitation but does not claim it has downloaded or validated those files.

## Primary references

- [Kaggle dataset and version history](https://www.kaggle.com/datasets/juliavalente/data-from-visits-to-the-trough-of-nellore-cattle)
- [Primary event/replacement methods and cohort paper](https://doi.org/10.1371/journal.pone.0271337)
- [CRAN `socialh` documentation for the enriched 7-column object](https://search.r-project.org/CRAN/refmans/socialh/html/feeding_event_data.html)
- [UFMS Mendeley record](https://data.mendeley.com/datasets/3wdjvxmvsp/7) and [Data in Brief dataset paper](https://doi.org/10.1016/j.dib.2024.110520)
- [Shade Mendeley record](https://data.mendeley.com/datasets/cctgpznjrc/1) and [PLOS ONE paper](https://doi.org/10.1371/journal.pone.0331238)
