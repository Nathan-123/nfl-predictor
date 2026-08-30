"""Orchestrates fetching each dataset and caching it locally as Parquet.

Idempotent: rerunning overwrites the parquet file + manifest entry for the
datasets requested, leaving other datasets' cached files untouched.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tqdm import tqdm

from nfl_predictor.config import DATA_DIR, MANIFEST_PATH, MIN_SEASON, PARTITIONED_DATASETS
from nfl_predictor.data.fetch import DATASETS

log = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    name: str
    status: str  # "ok" | "empty" | "failed"
    rows: int = 0
    seasons_requested: list[int] = field(default_factory=list)
    seasons_clipped_from: int | None = None
    error: str | None = None


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _fetch_with_trailing_year_fallback(fetch_fn, years: list[int], name: str):
    """Call fetch_fn(years), retrying with the most recent season dropped each
    time it errors. Datasets like weekly stats/injuries have no published
    file yet for a season that hasn't started/finished, which nfl_data_py
    surfaces as an HTTP 404 for the *whole* multi-year request -- rather than
    hardcoding a cutoff, just back off a season at a time until it works.
    """
    remaining = list(years)
    dropped: list[int] = []
    last_exc: Exception | None = None
    while remaining:
        try:
            df = fetch_fn(remaining)
            if dropped:
                log.warning(
                    "%s: season(s) %s unavailable (likely not yet published), using %s-%s",
                    name,
                    sorted(dropped),
                    remaining[0],
                    remaining[-1],
                )
            return df, remaining
        except Exception as exc:  # noqa: BLE001 - trying successively smaller ranges
            last_exc = exc
            dropped.append(remaining.pop())
    raise last_exc


def run(
    start_season: int,
    end_season: int,
    dataset_names: list[str] | None = None,
) -> list[DatasetResult]:
    """Fetch and cache the requested datasets for [start_season, end_season].

    dataset_names: subset of nfl_predictor.data.fetch.DATASETS keys, or
    None for all of them.
    """
    names = dataset_names or list(DATASETS.keys())
    unknown = set(names) - set(DATASETS.keys())
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Available: {sorted(DATASETS)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    results: list[DatasetResult] = []

    for name in tqdm(names, desc="Fetching datasets"):
        fetch_fn = DATASETS[name]

        effective_start = start_season
        clipped_from = None
        min_season = MIN_SEASON.get(name)
        if min_season and start_season < min_season:
            clipped_from = start_season
            effective_start = min_season

        if effective_start > end_season:
            log.warning(
                "Skipping %s: requires season >= %s, requested range ends at %s",
                name,
                min_season,
                end_season,
            )
            results.append(
                DatasetResult(name=name, status="empty", seasons_requested=[], seasons_clipped_from=clipped_from)
            )
            continue

        years = list(range(effective_start, end_season + 1))
        try:
            df, years = _fetch_with_trailing_year_fallback(fetch_fn, years, name)
        except Exception as exc:  # noqa: BLE001 - want to keep going across datasets
            log.error("Failed to fetch %s: %s", name, exc)
            results.append(DatasetResult(name=name, status="failed", seasons_requested=years, error=str(exc)))
            continue

        rows = len(df)
        if rows == 0:
            log.warning("%s returned 0 rows for seasons %s-%s", name, effective_start, end_season)
            results.append(
                DatasetResult(
                    name=name, status="empty", rows=0, seasons_requested=years, seasons_clipped_from=clipped_from
                )
            )
            continue

        if name in PARTITIONED_DATASETS:
            out_dir = DATA_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for season, group in df.groupby("season"):
                group.to_parquet(out_dir / f"{int(season)}.parquet", index=False)
            path = out_dir
        else:
            path = DATA_DIR / f"{name}.parquet"
            df.to_parquet(path, index=False)

        manifest[name] = {
            "rows": rows,
            "seasons_requested": years,
            "seasons_clipped_from": clipped_from,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "path": str(path.relative_to(DATA_DIR.parent.parent)),
            "partitioned_by_season": name in PARTITIONED_DATASETS,
        }
        results.append(
            DatasetResult(name=name, status="ok", rows=rows, seasons_requested=years, seasons_clipped_from=clipped_from)
        )

    _save_manifest(manifest)
    return results
