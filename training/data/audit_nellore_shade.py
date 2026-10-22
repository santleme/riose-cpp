#!/usr/bin/env python3
"""Probe only public metadata/endpoints for the Mendeley Nellore shade dataset.

This script sends no credentials, does not follow redirects, does not download
raw data, and does not attempt to get around authentication or embargo controls.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


DATASET_ID = "cctgpznjrc"
VERSION = "1"
DOI = "10.17632/cctgpznjrc.1"
LANDING_URL = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
FILES_API_URL = (
    f"https://api.data.mendeley.com/datasets/publics/{DATASET_ID}/files?version={VERSION}"
)
ZIP_API_URL = f"https://api.data.mendeley.com/datasets/{DATASET_ID}/zip/file_downloaded?version={VERSION}"
DATACITE_URL = f"https://api.datacite.org/dois/{DOI}"
ROOT = Path(__file__).resolve().parents[2]
OBSERVED_REPORT = ROOT / "training/data/reports/nellore-shade-access-2026-10.json"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def probe(url: str, accept: str = "text/html,application/json") -> dict[str, object]:
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CampoVivoShadeAudit/1.0", "Accept": accept},
    )
    try:
        response = opener.open(request, timeout=20)
    except urllib.error.HTTPError as error:
        # Do not record or display Location: it can contain a temporary signed URL.
        return {
            "status": error.code,
            "content_type": error.headers.get("Content-Type"),
            "redirect_followed": False,
        }
    with response:
        body = response.read()
        return {
            "status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "bytes_received": len(body),
            "redirect_followed": False,
            "body": body,
        }


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.metas: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        name = values.get("name") or values.get("property")
        content = values.get("content")
        if name and content:
            self.metas[name.lower()] = html.unescape(content)

    def handle_data(self, data: str) -> None:
        pass


def br_corte_water_intake_l_day(
    body_weight_kg: float,
    max_temperature_c: float,
    relative_humidity_percent: float,
    dry_matter_intake_kg_day: float,
) -> float:
    """BR-CORTE (2023), chapter 14 empirical equation; RH is 0..100 percent."""
    if body_weight_kg <= 0 or dry_matter_intake_kg_day < 0:
        raise ValueError("body weight must be positive and DMI must be non-negative")
    if not 0 <= relative_humidity_percent <= 100:
        raise ValueError("relative humidity must be expressed in percent from 0 to 100")
    metabolic_body_weight = body_weight_kg**0.75
    return (
        9.449
        + 0.190 * metabolic_body_weight
        + 0.271 * max_temperature_c
        - 0.259 * relative_humidity_percent
        + 0.489 * dry_matter_intake_kg_day
    )


def build_report() -> dict[str, object]:
    landing = probe(LANDING_URL)
    files_api = probe(FILES_API_URL)
    zip_api = probe(ZIP_API_URL)
    datacite = probe(DATACITE_URL)

    page_meta: dict[str, str] = {}
    if landing.get("body"):
        parser = _MetaParser()
        parser.feed(landing["body"].decode("utf-8", errors="replace"))
        page_meta = parser.metas
    datacite_attrs: dict[str, object] = {}
    if datacite.get("body"):
        datacite_attrs = json.loads(datacite["body"])["data"]["attributes"]

    # A transparent plug-in to published study summaries, not a row-level validation.
    illustrative_prediction = br_corte_water_intake_l_day(
        body_weight_kg=551.0,  # study's reported middle weigh-in, not the 83-d mean
        max_temperature_c=29.1,
        relative_humidity_percent=70.8,
        dry_matter_intake_kg_day=11.6,
    )
    report = {
        "audit_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "doi": DOI,
            "version": VERSION,
            "landing_url": LANDING_URL,
            "license_on_landing_page": "CC BY 4.0 (observed on public page)",
            "primary_article_doi": "10.1371/journal.pone.0331238",
        },
        "access_check": {
            "landing_page": {key: value for key, value in landing.items() if key != "body"},
            "landing_page_title": page_meta.get("title") or page_meta.get("og:title"),
            "public_file_list_endpoint": {
                "url": FILES_API_URL,
                **{key: value for key, value in files_api.items() if key != "body"},
            },
            "public_zip_endpoint": {
                "url": ZIP_API_URL,
                **{key: value for key, value in zip_api.items() if key != "body"},
            },
            "file_access_verified": files_api.get("status") == 200 and zip_api.get("status") == 200,
            "raw_file_downloaded": False,
            "file_names_schema_hash_and_units": None,
            "interpretation": "HTTP 401 on both public API routes without credentials. This establishes no anonymous file-level access; it does not by itself prove a 451 embargo response.",
        },
        "datacite": {
            "status": datacite["status"],
            "rights_list": datacite_attrs.get("rightsList"),
            "dates": datacite_attrs.get("dates"),
            "note": "Metadata simultaneously lists an embargoedAccess rights URI and CC BY 4.0; reconcile with actual file access and depositor terms.",
        },
        "reported_study_facts_from_primary_article": {
            "animals_at_completion": 47,
            "initial_animals": 48,
            "treatment_assignment_level": "four collective pens: two shaded and two unshaded; animal groups allocated to pens",
            "duration_days": 83,
            "individual_water_intake_mean_l_day": {"shade": 35.1, "no_shade": 37.9},
            "mean_liveweights_kg_reported_at_weighings": [477, 551, 591],
            "dmi_kg_dm_day": 11.6,
            "mean_daily_max_temperature_c": 29.1,
            "mean_relative_humidity_percent": 70.8,
            "published_water_events": 19388,
            "published_dmi_records": 3854,
            "published_liveweight_records": 282,
            "weather_frequency_discrepancy": "Mendeley page describes 7,968 weather records at 15-minute intervals; article methods text says hourly observations. Verify against workbook before deriving daily TMAX/RH.",
        },
        "br_corte_2023": {
            "equation": "WI_L_day = 9.449 + 0.190 * BW_kg^0.75 + 0.271 * TMAX_C - 0.259 * RH_percent + 0.489 * DMI_kg_DM_day",
            "source_url": "https://brcorte.com.br/assets/book2023/br/14.pdf",
            "published_independent_validation": "BR-CORTE chapter reports 14,774 daily measurements in five separate Nellore experiments; prediction mean 16.3 vs observed 16.1 L/day, slope 1.005, intercept -0.3109, CCC 0.574, and predicted range 2.3–32.3 versus observed 2.0–49.9 L/day.",
            "domain_note": "For context, 551 kg corresponds to BW^0.75≈113.7 kg^0.75, slightly above the chapter's largest training-experiment group maximum (113.5) and within the largest validation-experiment group maximum (114.2); the 591 kg final mean corresponds to ≈119.8, above the reported validation group maxima. These are study-level summary ranges, not a full row-level domain test.",
            "illustrative_group_mean_plugin": {
                "inputs": {
                    "body_weight_kg": 551,
                    "body_weight_note": "middle of three reported weigh-ins; not a time-weighted 83-day mean",
                    "TMAX_C": 29.1,
                    "RH_percent": 70.8,
                    "DMI_kg_DM_day": 11.6
                },
                "predicted_l_day": illustrative_prediction,
                "published_observed_group_means_l_day": {"shade": 35.1, "no_shade": 37.9},
                "difference_predicted_minus_published_mean_l_day": {
                    "shade": illustrative_prediction - 35.1,
                    "no_shade": illustrative_prediction - 37.9
                },
                "status": "illustrative calculation only; not a validation metric. Equation is linear but BW^0.75 and covariate means do not yield an exact mean prediction, BW value is not an interval average, and daily inputs/animal rows/pen identifiers were unavailable.",
            },
            "row_level_metrics": None,
            "reason_metrics_missing": "The Mendeley file list and ZIP endpoints returned HTTP 401 without authentication; no raw animal-by-day or pen-keyed observations were available. No bias/MAE/calibration or train/test split was computed.",
        },
        "causal_and_validation_limits": [
            "Shade is assigned at pen/group level (2 pens per treatment) while the article analyzes 47 animal records as experimental units; animal-level independence can understate treatment uncertainty.",
            "A valid group split must keep entire pen(s) together. With only two pens per treatment, robust treatment generalization is not estimable from random animal splits.",
            "Only treatment means are in the article text, so the plug-in cannot compare shade-specific residuals or calibrate individual predictions.",
            "The source page's meteorological interval and article methods are inconsistent; do not infer daily maxima from the advertised time series until raw schema is inspected.",
            "Data source is feedlot TMR and four collective pens, not extensive pasture.",
        ],
    }
    return report


def main() -> int:
    report = build_report()
    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "landing_status": report["access_check"]["landing_page"].get("status"),
        "files_status": report["access_check"]["public_file_list_endpoint"].get("status"),
        "zip_status": report["access_check"]["public_zip_endpoint"].get("status"),
        "file_access_verified": report["access_check"]["file_access_verified"],
        "row_level_metrics": report["br_corte_2023"]["row_level_metrics"],
        "report": str(OBSERVED_REPORT.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["access_check"]["landing_page"].get("status") == 200 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"shade audit error: {exc}", file=sys.stderr)
        raise SystemExit(2)
