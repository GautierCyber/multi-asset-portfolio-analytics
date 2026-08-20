"""Market-data acquisition pipeline for the multi-asset portfolio project.

The module downloads daily Yahoo Finance data through yfinance, validates each
asset before use, builds a common adjusted-close panel, and persists local raw
and processed snapshots together with a machine-readable quality report.

No price observation is repaired, forward-filled, backward-filled or
interpolated by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import ASSET_UNIVERSE, AssetSpec
from .validation import (
    MAX_ABS_DAILY_RETURN,
    MAX_MISSING_FRACTION,
    MIN_COMMON_HISTORY_YEARS,
    PriceFrameValidationReport,
    build_common_price_panel,
    validate_asset_frame,
    validate_common_history,
)


PROJECT_ROOT: Path = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR: Path = (
    PROJECT_ROOT
    / "data"
)

RAW_DATA_DIR: Path = (
    DATA_DIR
    / "raw"
)

PROCESSED_DATA_DIR: Path = (
    DATA_DIR
    / "processed"
)

DEFAULT_START_DATE: str = "2005-01-01"
INTERVAL: str = "1d"

YFINANCE_TIMEOUT_SECONDS: float = 30.0
YFINANCE_RETRIES: int = 2

REQUIRED_PROVIDER_COLUMNS: tuple[
    str,
    ...,
] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
)

PROVIDER_NAME: str = (
    "Yahoo Finance via yfinance"
)


class DataAcquisitionError(RuntimeError):
    """Raised when provider data cannot be acquired or normalised safely."""


@dataclass(frozen=True)
class DownloadedAssetData:
    """Validated result of one asset download."""

    frame: pd.DataFrame
    metadata: dict[str, Any]
    validation_report: (
        PriceFrameValidationReport
    )


@dataclass(frozen=True)
class DataPipelineResult:
    """Artifacts produced by one successful data-pipeline run."""

    adjusted_close: pd.DataFrame
    quality_report: dict[str, Any]
    raw_files: dict[str, Path]
    adjusted_close_file: Path
    quality_report_file: Path


def _resolve_start_date(
    value: str | date,
) -> date:
    """Resolve the inclusive start date."""
    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    try:
        return date.fromisoformat(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "start_date must use ISO format "
            "YYYY-MM-DD."
        ) from exc


def _resolve_end_date(
    value: str | date | None,
) -> date:
    """Resolve the exclusive end date."""
    if value is None:
        return (
            datetime.now(
                timezone.utc
            )
            .date()
        )

    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    try:
        return date.fromisoformat(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "end_date must use ISO format "
            "YYYY-MM-DD."
        ) from exc


def _validate_date_range(
    start_date: date,
    end_date: date,
) -> None:
    """Validate an inclusive-start / exclusive-end date interval."""
    if (
        start_date
        >= end_date
    ):
        raise ValueError(
            "start_date must be strictly "
            "earlier than the exclusive "
            "end_date."
        )


@contextmanager
def _yfinance_runtime_configuration() -> Iterator[
    None
]:
    """Temporarily configure yfinance for explicit failures and retries."""
    previous_hide_exceptions = (
        yf.config
        .debug
        .hide_exceptions
    )

    previous_retries = (
        yf.config
        .network
        .retries
    )

    yf.config.debug.hide_exceptions = False
    yf.config.network.retries = (
        YFINANCE_RETRIES
    )

    try:
        yield

    finally:
        (
            yf.config
            .debug
            .hide_exceptions
        ) = previous_hide_exceptions

        (
            yf.config
            .network
            .retries
        ) = previous_retries


def _normalise_history_index(
    frame: pd.DataFrame,
    *,
    ticker: str,
) -> pd.DataFrame:
    """Normalise daily timestamps while preserving exchange-local dates."""
    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            "Downloaded market data for "
            f"{ticker} must be a pandas "
            "DataFrame."
        )

    normalised = frame.copy()

    if not isinstance(
        normalised.index,
        pd.DatetimeIndex,
    ):
        try:
            normalised.index = (
                pd.DatetimeIndex(
                    pd.to_datetime(
                        normalised.index
                    )
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise DataAcquisitionError(
                "Could not convert the history "
                f"index for {ticker} to a "
                "DatetimeIndex."
            ) from exc

    if (
        normalised.index.tz
        is not None
    ):
        normalised.index = (
            normalised.index
            .tz_localize(
                None
            )
        )

    normalised.index = (
        normalised.index
        .normalize()
    )

    normalised.index.name = "Date"

    return normalised


def _validate_download_schema(
    frame: pd.DataFrame,
    *,
    ticker: str,
) -> None:
    """Validate the provider-specific schema before numerical checks."""
    if frame.empty:
        raise DataAcquisitionError(
            "Yahoo Finance returned no "
            f"observations for {ticker}."
        )

    if isinstance(
        frame.columns,
        pd.MultiIndex,
    ):
        raise DataAcquisitionError(
            "Yahoo Finance returned unexpected "
            "MultiIndex columns for "
            f"{ticker}."
        )

    missing_columns = sorted(
        set(
            REQUIRED_PROVIDER_COLUMNS
        ).difference(
            frame.columns
        )
    )

    if missing_columns:
        raise DataAcquisitionError(
            "Yahoo Finance history for "
            f"{ticker} is missing expected "
            f"columns: {missing_columns}."
        )


def _json_safe(
    value: Any,
) -> Any:
    """Convert common scientific-Python objects to strict JSON values."""
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
    ):
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
            np.bool_,
        ),
    ):
        return _json_safe(
            value.item()
        )

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(
            value
        ):
            return value

        return None

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    return str(
        value
    )


def _selected_history_metadata(
    ticker_object: yf.Ticker,
) -> dict[str, Any]:
    """Retrieve a stable, optional subset of public yfinance metadata."""
    try:
        metadata = (
            ticker_object
            .get_history_metadata(
                repair=False
            )
        )

    except Exception as exc:
        return {
            "metadata_available": False,
            "metadata_error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }

    if not isinstance(
        metadata,
        dict,
    ):
        return {
            "metadata_available": False,
            "metadata_error": (
                "yfinance returned history "
                "metadata in an unexpected "
                "format."
            ),
        }

    selected_keys = (
        "currency",
        "symbol",
        "exchangeName",
        "fullExchangeName",
        "instrumentType",
        "timezone",
        "exchangeTimezoneName",
        "firstTradeDate",
    )

    selected: dict[
        str,
        Any,
    ] = {
        "metadata_available": True,
    }

    for key in selected_keys:
        if key in metadata:
            selected[
                key
            ] = _json_safe(
                metadata[
                    key
                ]
            )

    return selected


def _validate_provider_identity(
    *,
    asset: AssetSpec,
    metadata: dict[str, Any],
) -> None:
    """Cross-check stable provider identifiers when metadata are available."""
    provider_symbol = (
        metadata.get(
            "symbol"
        )
    )

    if (
        provider_symbol
        is not None
        and (
            str(
                provider_symbol
            ).upper()
            != asset.ticker.upper()
        )
    ):
        raise DataAcquisitionError(
            f"Ticker mismatch for "
            f"{asset.ticker}: Yahoo Finance "
            f"reports {provider_symbol!r}."
        )

    provider_currency = (
        metadata.get(
            "currency"
        )
    )

    if (
        provider_currency
        is not None
    ):
        expected_currency = (
            asset
            .listing_currency
            .upper()
        )

        observed_currency = (
            str(
                provider_currency
            )
            .upper()
        )

        if (
            observed_currency
            != expected_currency
        ):
            raise DataAcquisitionError(
                f"Currency mismatch for "
                f"{asset.ticker}: Yahoo Finance "
                f"reports {observed_currency}, "
                "while the investment universe "
                f"specifies {expected_currency}."
            )


def download_asset_history(
    asset: AssetSpec,
    *,
    start_date: str | date = DEFAULT_START_DATE,
    end_date: str | date | None = None,
) -> DownloadedAssetData:
    """Download and validate daily history for one investment-universe asset."""
    if not isinstance(
        asset,
        AssetSpec,
    ):
        raise TypeError(
            "asset must be an AssetSpec instance."
        )

    resolved_start = (
        _resolve_start_date(
            start_date
        )
    )

    resolved_end = (
        _resolve_end_date(
            end_date
        )
    )

    _validate_date_range(
        resolved_start,
        resolved_end,
    )

    with (
        _yfinance_runtime_configuration()
    ):
        ticker_object = yf.Ticker(
            asset.ticker
        )

        try:
            frame = (
                ticker_object.history(
                    start=(
                        resolved_start
                        .isoformat()
                    ),
                    end=(
                        resolved_end
                        .isoformat()
                    ),
                    interval=INTERVAL,
                    prepost=False,
                    actions=True,
                    auto_adjust=False,
                    back_adjust=False,
                    repair=False,
                    keepna=True,
                    rounding=False,
                    timeout=(
                        YFINANCE_TIMEOUT_SECONDS
                    ),
                )
            )

        except Exception as exc:
            raise DataAcquisitionError(
                "Failed to download Yahoo "
                "Finance history for "
                f"{asset.ticker}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ) from exc

        frame = (
            _normalise_history_index(
                frame,
                ticker=asset.ticker,
            )
        )

        _validate_download_schema(
            frame,
            ticker=asset.ticker,
        )

        metadata = (
            _selected_history_metadata(
                ticker_object
            )
        )

    _validate_provider_identity(
        asset=asset,
        metadata=metadata,
    )

    validation_report = (
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )
    )

    return DownloadedAssetData(
        frame=frame,
        metadata=metadata,
        validation_report=(
            validation_report
        ),
    )


def _ensure_output_directories(
    *,
    raw_dir: Path,
    processed_dir: Path,
) -> None:
    """Create local output directories."""
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def _write_csv_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV atomically."""
    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    try:
        frame.to_csv(
            temporary_path,
            index=True,
            date_format="%Y-%m-%d",
        )

        temporary_path.replace(
            path
        )

    finally:
        if (
            temporary_path
            .exists()
        ):
            temporary_path.unlink()


def _write_json_atomic(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """Write one strict UTF-8 JSON document atomically."""
    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    finally:
        if (
            temporary_path
            .exists()
        ):
            temporary_path.unlink()


def _sha256_file(
    path: Path,
) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file_handle:
        for chunk in iter(
            lambda: (
                file_handle.read(
                    1024 * 1024
                )
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _report_path(
    path: Path,
) -> str:
    """Return a project-relative path when possible."""
    resolved_path = (
        path.resolve()
    )

    resolved_root = (
        PROJECT_ROOT.resolve()
    )

    try:
        return str(
            resolved_path
            .relative_to(
                resolved_root
            )
        )

    except ValueError:
        return str(
            resolved_path
        )


def _corporate_action_counts(
    frame: pd.DataFrame,
) -> dict[str, int]:
    """Count non-zero corporate-action observations."""
    counts: dict[
        str,
        int,
    ] = {}

    for column in (
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    ):
        if (
            column
            not in frame.columns
        ):
            continue

        values = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .fillna(
                0.0
            )
        )

        counts[
            column
        ] = int(
            (
                values
                != 0.0
            ).sum()
        )

    return counts


def _build_asset_report(
    *,
    asset: AssetSpec,
    downloaded: DownloadedAssetData,
    raw_path: Path,
) -> dict[str, Any]:
    """Build the quality-report section for one asset."""
    return {
        "asset_metadata": (
            _json_safe(
                asdict(
                    asset
                )
            )
        ),
        "provider_metadata": (
            _json_safe(
                downloaded.metadata
            )
        ),
        "validation": (
            downloaded
            .validation_report
            .to_dict()
        ),
        "corporate_actions_non_zero_rows": (
            _corporate_action_counts(
                downloaded.frame
            )
        ),
        "raw_snapshot": {
            "path": _report_path(
                raw_path
            ),
            "sha256": _sha256_file(
                raw_path
            ),
        },
    }


def run_data_pipeline(
    *,
    start_date: str | date = DEFAULT_START_DATE,
    end_date: str | date | None = None,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> DataPipelineResult:
    """Run acquisition, validation, panel construction and persistence."""
    resolved_start = (
        _resolve_start_date(
            start_date
        )
    )

    resolved_end = (
        _resolve_end_date(
            end_date
        )
    )

    _validate_date_range(
        resolved_start,
        resolved_end,
    )

    raw_dir = Path(
        raw_dir
    )

    processed_dir = Path(
        processed_dir
    )

    downloaded_by_key: dict[
        str,
        DownloadedAssetData,
    ] = {}

    frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for (
        asset_key,
        asset,
    ) in ASSET_UNIVERSE.items():
        downloaded = (
            download_asset_history(
                asset,
                start_date=(
                    resolved_start
                ),
                end_date=(
                    resolved_end
                ),
            )
        )

        downloaded_by_key[
            asset_key
        ] = downloaded

        frames[
            asset.ticker
        ] = downloaded.frame

    (
        adjusted_close,
        missing_within_active_history,
    ) = build_common_price_panel(
        frames
    )

    common_history_years = (
        validate_common_history(
            adjusted_close
        )
    )

    adjusted_close.index.name = "Date"

    _ensure_output_directories(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    raw_files: dict[
        str,
        Path,
    ] = {}

    per_asset_reports: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        asset_key,
        asset,
    ) in ASSET_UNIVERSE.items():
        downloaded = (
            downloaded_by_key[
                asset_key
            ]
        )

        raw_path = (
            raw_dir
            / f"{asset.ticker}.csv"
        )

        _write_csv_atomic(
            downloaded.frame,
            raw_path,
        )

        raw_files[
            asset.ticker
        ] = raw_path

        per_asset_reports[
            asset_key
        ] = _build_asset_report(
            asset=asset,
            downloaded=downloaded,
            raw_path=raw_path,
        )

    adjusted_close_path = (
        processed_dir
        / "adjusted_close.csv"
    )

    _write_csv_atomic(
        adjusted_close,
        adjusted_close_path,
    )

    quality_report: dict[
        str,
        Any,
    ] = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "provider": {
            "name": (
                PROVIDER_NAME
            ),
            "yfinance_version": (
                yf.__version__
            ),
        },
        "runtime": {
            "python_version": (
                platform
                .python_version()
            ),
            "numpy_version": (
                np.__version__
            ),
            "pandas_version": (
                pd.__version__
            ),
        },
        "request": {
            "start_date_inclusive": (
                resolved_start
                .isoformat()
            ),
            "end_date_exclusive": (
                resolved_end
                .isoformat()
            ),
            "interval": INTERVAL,
            "prepost": False,
            "actions": True,
            "auto_adjust": False,
            "back_adjust": False,
            "repair": False,
            "keepna": True,
            "rounding": False,
            "timeout_seconds": (
                YFINANCE_TIMEOUT_SECONDS
            ),
            "network_retries": (
                YFINANCE_RETRIES
            ),
            "hide_exceptions": False,
        },
        "validation_thresholds": {
            "max_abs_daily_return": (
                MAX_ABS_DAILY_RETURN
            ),
            "max_missing_fraction": (
                MAX_MISSING_FRACTION
            ),
            "min_common_history_years": (
                MIN_COMMON_HISTORY_YEARS
            ),
        },
        "universe": {
            "number_of_assets": len(
                ASSET_UNIVERSE
            ),
            "tickers": [
                asset.ticker
                for asset
                in ASSET_UNIVERSE.values()
            ],
        },
        "assets": (
            per_asset_reports
        ),
        "common_panel": {
            "rows": len(
                adjusted_close
            ),
            "columns": len(
                adjusted_close.columns
            ),
            "start_date": (
                adjusted_close
                .index[0]
                .date()
                .isoformat()
            ),
            "end_date": (
                adjusted_close
                .index[-1]
                .date()
                .isoformat()
            ),
            "history_years": (
                common_history_years
            ),
            "missing_adjusted_close_within_active_history": (
                missing_within_active_history
            ),
            "snapshot": {
                "path": _report_path(
                    adjusted_close_path
                ),
                "sha256": _sha256_file(
                    adjusted_close_path
                ),
            },
        },
    }

    quality_report_path = (
        processed_dir
        / "data_quality_report.json"
    )

    _write_json_atomic(
        quality_report,
        quality_report_path,
    )

    return DataPipelineResult(
        adjusted_close=(
            adjusted_close
        ),
        quality_report=(
            quality_report
        ),
        raw_files=(
            raw_files
        ),
        adjusted_close_file=(
            adjusted_close_path
        ),
        quality_report_file=(
            quality_report_path
        ),
    )


def main() -> None:
    """Run the default pipeline from the command line."""
    result = run_data_pipeline()

    panel = (
        result.adjusted_close
    )

    print(
        "Market-data pipeline "
        "completed successfully."
    )

    print(
        f"Assets: "
        f"{len(panel.columns)}"
    )

    print(
        f"Common observations: "
        f"{len(panel)}"
    )

    print(
        "Common period: "
        f"{panel.index[0].date()} "
        "to "
        f"{panel.index[-1].date()}"
    )

    print(
        "Common history: "
        f"{result.quality_report['common_panel']['history_years']:.2f} "
        "years"
    )

    print(
        "Processed prices: "
        f"{result.adjusted_close_file}"
    )

    print(
        "Quality report: "
        f"{result.quality_report_file}"
    )


if __name__ == "__main__":
    main()