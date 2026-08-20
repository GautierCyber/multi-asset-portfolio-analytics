import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from multi_asset_portfolio import data as data_module
from multi_asset_portfolio.universe import (
    ASSET_UNIVERSE,
    TICKERS,
    AssetSpec,
)
from multi_asset_portfolio.validation import (
    DataValidationError,
    validate_asset_frame,
)


def make_valid_frame(
    start: str = "2010-01-01",
    periods: int = 4000,
    *,
    timezone_name: str | None = None,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        start=start,
        periods=periods,
    )

    if (
        timezone_name
        is not None
    ):
        dates = (
            dates.tz_localize(
                timezone_name
            )
        )

    prices = pd.Series(
        [
            100.0
            * (
                1.0001 ** i
            )
            for i in range(
                periods
            )
        ],
        index=dates,
    )

    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.001,
            "Low": prices * 0.999,
            "Close": prices,
            "Adj Close": prices,
            "Volume": 1_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


def sha256_file(
    path: Path,
) -> str:
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


def make_downloaded_asset(
    asset: AssetSpec,
    *,
    periods: int = 4000,
    missing_adjusted_close_date: (
        pd.Timestamp | None
    ) = None,
) -> data_module.DownloadedAssetData:
    frame = make_valid_frame(
        periods=periods
    )

    if (
        missing_adjusted_close_date
        is not None
    ):
        if (
            missing_adjusted_close_date
            not in frame.index
        ):
            raise AssertionError(
                "Synthetic missing date "
                "is not present in the "
                "test frame."
            )

        frame.loc[
            missing_adjusted_close_date,
            "Adj Close",
        ] = float(
            "nan"
        )

    validation_report = (
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )
    )

    return (
        data_module
        .DownloadedAssetData(
            frame=frame,
            metadata={
                "metadata_available": True,
                "currency": "EUR",
                "symbol": asset.ticker,
            },
            validation_report=(
                validation_report
            ),
        )
    )


class FakeTicker:
    last_history_kwargs: dict[
        str,
        object,
    ] = {}

    last_metadata_repair: object = None

    config_during_history: (
        tuple[
            object,
            object,
        ]
        | None
    ) = None

    def __init__(
        self,
        ticker: str,
    ) -> None:
        self.ticker = ticker

    def history(
        self,
        **kwargs: object,
    ) -> pd.DataFrame:
        type(
            self
        ).last_history_kwargs = dict(
            kwargs
        )

        type(
            self
        ).config_during_history = (
            data_module
            .yf
            .config
            .debug
            .hide_exceptions,
            data_module
            .yf
            .config
            .network
            .retries,
        )

        return make_valid_frame()

    def get_history_metadata(
        self,
        repair: bool = False,
    ) -> dict[str, object]:
        type(
            self
        ).last_metadata_repair = (
            repair
        )

        return {
            "currency": "EUR",
            "symbol": self.ticker,
            "exchangeName": "GER",
            "fullExchangeName": "XETRA",
            "instrumentType": "ETF",
            "timezone": "CEST",
            "exchangeTimezoneName": (
                "Europe/Berlin"
            ),
        }


def test_download_uses_explicit_parameters_and_restores_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        FakeTicker,
    )

    previous_hide_exceptions = (
        data_module
        .yf
        .config
        .debug
        .hide_exceptions
    )

    previous_retries = (
        data_module
        .yf
        .config
        .network
        .retries
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    downloaded = (
        data_module
        .download_asset_history(
            asset,
            start_date="2010-01-01",
            end_date="2026-01-01",
        )
    )

    kwargs = (
        FakeTicker
        .last_history_kwargs
    )

    assert kwargs == {
        "start": "2010-01-01",
        "end": "2026-01-01",
        "interval": "1d",
        "prepost": False,
        "actions": True,
        "auto_adjust": False,
        "back_adjust": False,
        "repair": False,
        "keepna": True,
        "rounding": False,
        "timeout": (
            data_module
            .YFINANCE_TIMEOUT_SECONDS
        ),
    }

    assert (
        "raise_errors"
        not in kwargs
    )

    assert (
        FakeTicker
        .last_metadata_repair
        is False
    )

    assert (
        FakeTicker
        .config_during_history
        == (
            False,
            data_module
            .YFINANCE_RETRIES,
        )
    )

    assert (
        downloaded
        .validation_report
        .ticker
        == asset.ticker
    )

    assert (
        downloaded
        .metadata[
            "currency"
        ]
        == "EUR"
    )

    assert (
        data_module
        .yf
        .config
        .debug
        .hide_exceptions
        == previous_hide_exceptions
    )

    assert (
        data_module
        .yf
        .config
        .network
        .retries
        == previous_retries
    )


def test_download_wraps_provider_failure_and_restores_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTicker(
        FakeTicker
    ):
        def history(
            self,
            **kwargs: object,
        ) -> pd.DataFrame:
            raise TimeoutError(
                "synthetic timeout"
            )

    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        FailingTicker,
    )

    previous_hide_exceptions = (
        data_module
        .yf
        .config
        .debug
        .hide_exceptions
    )

    previous_retries = (
        data_module
        .yf
        .config
        .network
        .retries
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        data_module
        .DataAcquisitionError,
        match="synthetic timeout",
    ):
        (
            data_module
            .download_asset_history(
                asset,
                start_date=(
                    "2010-01-01"
                ),
                end_date=(
                    "2026-01-01"
                ),
            )
        )

    assert (
        data_module
        .yf
        .config
        .debug
        .hide_exceptions
        == previous_hide_exceptions
    )

    assert (
        data_module
        .yf
        .config
        .network
        .retries
        == previous_retries
    )


def test_incomplete_provider_schema_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteTicker(
        FakeTicker
    ):
        def history(
            self,
            **kwargs: object,
        ) -> pd.DataFrame:
            return (
                make_valid_frame()
                .drop(
                    columns=[
                        "Adj Close"
                    ]
                )
            )

    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        IncompleteTicker,
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        data_module
        .DataAcquisitionError,
        match=(
            "missing expected columns"
        ),
    ):
        (
            data_module
            .download_asset_history(
                asset,
                start_date=(
                    "2010-01-01"
                ),
                end_date=(
                    "2026-01-01"
                ),
            )
        )


def test_currency_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CurrencyMismatchTicker(
        FakeTicker
    ):
        def get_history_metadata(
            self,
            repair: bool = False,
        ) -> dict[str, object]:
            return {
                "currency": "USD",
                "symbol": self.ticker,
            }

    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        CurrencyMismatchTicker,
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        data_module
        .DataAcquisitionError,
        match="Currency mismatch",
    ):
        (
            data_module
            .download_asset_history(
                asset,
                start_date=(
                    "2010-01-01"
                ),
                end_date=(
                    "2026-01-01"
                ),
            )
        )


def test_ticker_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TickerMismatchTicker(
        FakeTicker
    ):
        def get_history_metadata(
            self,
            repair: bool = False,
        ) -> dict[str, object]:
            return {
                "currency": "EUR",
                "symbol": "WRONG.DE",
            }

    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        TickerMismatchTicker,
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        data_module
        .DataAcquisitionError,
        match="Ticker mismatch",
    ):
        (
            data_module
            .download_asset_history(
                asset,
                start_date=(
                    "2010-01-01"
                ),
                end_date=(
                    "2026-01-01"
                ),
            )
        )


def test_timezone_is_removed_without_calendar_date_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimezoneTicker(
        FakeTicker
    ):
        def history(
            self,
            **kwargs: object,
        ) -> pd.DataFrame:
            return make_valid_frame(
                timezone_name=(
                    "Europe/Berlin"
                )
            )

    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        TimezoneTicker,
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    downloaded = (
        data_module
        .download_asset_history(
            asset,
            start_date="2010-01-01",
            end_date="2026-01-01",
        )
    )

    assert (
        downloaded
        .frame
        .index
        .tz
        is None
    )

    assert (
        downloaded
        .frame
        .index[0]
        == pd.Timestamp(
            "2010-01-01"
        )
    )

    assert (
        downloaded
        .frame
        .index
        .name
        == "Date"
    )


def test_metadata_failure_does_not_discard_valid_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MetadataFailureTicker(
        FakeTicker
    ):
        def get_history_metadata(
            self,
            repair: bool = False,
        ) -> dict[str, object]:
            raise RuntimeError(
                "synthetic metadata failure"
            )

    monkeypatch.setattr(
        data_module.yf,
        "Ticker",
        MetadataFailureTicker,
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    downloaded = (
        data_module
        .download_asset_history(
            asset,
            start_date="2010-01-01",
            end_date="2026-01-01",
        )
    )

    assert (
        downloaded
        .metadata[
            "metadata_available"
        ]
        is False
    )

    assert (
        "synthetic metadata failure"
        in downloaded
        .metadata[
            "metadata_error"
        ]
    )

    assert not (
        downloaded
        .frame
        .empty
    )


def test_pipeline_writes_complete_snapshot_and_quality_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_download_asset_history(
        asset: AssetSpec,
        *,
        start_date: object,
        end_date: object,
    ) -> (
        data_module
        .DownloadedAssetData
    ):
        return (
            make_downloaded_asset(
                asset
            )
        )

    monkeypatch.setattr(
        data_module,
        "download_asset_history",
        fake_download_asset_history,
    )

    raw_dir = (
        tmp_path
        / "raw"
    )

    processed_dir = (
        tmp_path
        / "processed"
    )

    result = (
        data_module
        .run_data_pipeline(
            start_date=(
                "2005-01-01"
            ),
            end_date=(
                "2026-01-01"
            ),
            raw_dir=raw_dir,
            processed_dir=(
                processed_dir
            ),
        )
    )

    assert tuple(
        result
        .adjusted_close
        .columns
    ) == TICKERS

    assert (
        result
        .adjusted_close
        .notna()
        .all()
        .all()
    )

    assert set(
        result.raw_files
    ) == set(
        TICKERS
    )

    for raw_file in (
        result
        .raw_files
        .values()
    ):
        assert (
            raw_file.exists()
        )

        raw_snapshot = (
            pd.read_csv(
                raw_file
            )
        )

        assert set(
            data_module
            .REQUIRED_PROVIDER_COLUMNS
        ).issubset(
            raw_snapshot.columns
        )

    assert (
        result
        .adjusted_close_file
        .exists()
    )

    assert (
        result
        .quality_report_file
        .exists()
    )

    report = json.loads(
        result
        .quality_report_file
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        report[
            "universe"
        ][
            "number_of_assets"
        ]
        == len(
            ASSET_UNIVERSE
        )
    )

    assert tuple(
        report[
            "universe"
        ][
            "tickers"
        ]
    ) == TICKERS

    assert (
        report[
            "common_panel"
        ][
            "history_years"
        ]
        >= 10.0
    )

    assert (
        report[
            "request"
        ][
            "repair"
        ]
        is False
    )

    assert (
        report[
            "request"
        ][
            "auto_adjust"
        ]
        is False
    )

    assert (
        report[
            "request"
        ][
            "keepna"
        ]
        is True
    )

    assert (
        report[
            "request"
        ][
            "network_retries"
        ]
        == (
            data_module
            .YFINANCE_RETRIES
        )
    )

    assert (
        report[
            "validation_thresholds"
        ][
            "min_common_history_years"
        ]
        == (
            data_module
            .MIN_COMMON_HISTORY_YEARS
        )
    )

    assert (
        report[
            "common_panel"
        ][
            "snapshot"
        ][
            "sha256"
        ]
        == sha256_file(
            result
            .adjusted_close_file
        )
    )

    missing = (
        report[
            "common_panel"
        ][
            "missing_adjusted_close_within_active_history"
        ]
    )

    assert missing == {
        ticker: 0
        for ticker
        in TICKERS
    }

    for (
        asset_key,
        asset,
    ) in ASSET_UNIVERSE.items():
        raw_path = (
            result
            .raw_files[
                asset.ticker
            ]
        )

        assert (
            report[
                "assets"
            ][
                asset_key
            ][
                "raw_snapshot"
            ][
                "sha256"
            ]
            == sha256_file(
                raw_path
            )
        )

        assert (
            report[
                "assets"
            ][
                asset_key
            ][
                "validation"
            ][
                "zero_ohlc_count_by_column"
            ]
            == {
                "Close": 0,
                "High": 0,
                "Low": 0,
                "Open": 0,
            }
        )


def test_identical_inputs_produce_identical_snapshot_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_download_asset_history(
        asset: AssetSpec,
        *,
        start_date: object,
        end_date: object,
    ) -> (
        data_module
        .DownloadedAssetData
    ):
        return (
            make_downloaded_asset(
                asset
            )
        )

    monkeypatch.setattr(
        data_module,
        "download_asset_history",
        fake_download_asset_history,
    )

    first = (
        data_module
        .run_data_pipeline(
            start_date=(
                "2005-01-01"
            ),
            end_date=(
                "2026-01-01"
            ),
            raw_dir=(
                tmp_path
                / "run_1"
                / "raw"
            ),
            processed_dir=(
                tmp_path
                / "run_1"
                / "processed"
            ),
        )
    )

    second = (
        data_module
        .run_data_pipeline(
            start_date=(
                "2005-01-01"
            ),
            end_date=(
                "2026-01-01"
            ),
            raw_dir=(
                tmp_path
                / "run_2"
                / "raw"
            ),
            processed_dir=(
                tmp_path
                / "run_2"
                / "processed"
            ),
        )
    )

    assert sha256_file(
        first
        .adjusted_close_file
    ) == sha256_file(
        second
        .adjusted_close_file
    )

    for ticker in TICKERS:
        assert sha256_file(
            first
            .raw_files[
                ticker
            ]
        ) == sha256_file(
            second
            .raw_files[
                ticker
            ]
        )


def test_pipeline_never_forward_fills_missing_adjusted_price(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_ticker = (
        TICKERS[0]
    )

    missing_date = (
        pd.Timestamp(
            "2010-05-21"
        )
    )

    def fake_download_asset_history(
        asset: AssetSpec,
        *,
        start_date: object,
        end_date: object,
    ) -> (
        data_module
        .DownloadedAssetData
    ):
        missing_for_asset = (
            missing_date
            if (
                asset.ticker
                == target_ticker
            )
            else None
        )

        return (
            make_downloaded_asset(
                asset,
                missing_adjusted_close_date=(
                    missing_for_asset
                ),
            )
        )

    monkeypatch.setattr(
        data_module,
        "download_asset_history",
        fake_download_asset_history,
    )

    result = (
        data_module
        .run_data_pipeline(
            start_date=(
                "2005-01-01"
            ),
            end_date=(
                "2026-01-01"
            ),
            raw_dir=(
                tmp_path
                / "raw"
            ),
            processed_dir=(
                tmp_path
                / "processed"
            ),
        )
    )

    assert (
        missing_date
        not in (
            result
            .adjusted_close
            .index
        )
    )

    assert len(
        result.adjusted_close
    ) == 3999

    missing = (
        result
        .quality_report[
            "common_panel"
        ][
            "missing_adjusted_close_within_active_history"
        ]
    )

    assert (
        missing[
            target_ticker
        ]
        == 1
    )


def test_insufficient_common_history_is_rejected_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_download_asset_history(
        asset: AssetSpec,
        *,
        start_date: object,
        end_date: object,
    ) -> (
        data_module
        .DownloadedAssetData
    ):
        return (
            make_downloaded_asset(
                asset,
                periods=1000,
            )
        )

    monkeypatch.setattr(
        data_module,
        "download_asset_history",
        fake_download_asset_history,
    )

    raw_dir = (
        tmp_path
        / "raw"
    )

    processed_dir = (
        tmp_path
        / "processed"
    )

    with pytest.raises(
        DataValidationError,
        match=(
            "Common history spans"
        ),
    ):
        (
            data_module
            .run_data_pipeline(
                start_date=(
                    "2005-01-01"
                ),
                end_date=(
                    "2026-01-01"
                ),
                raw_dir=raw_dir,
                processed_dir=(
                    processed_dir
                ),
            )
        )

    assert not (
        raw_dir.exists()
    )

    assert not (
        processed_dir.exists()
    )


def test_acquisition_failure_does_not_write_partial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failing_ticker = (
        TICKERS[2]
    )

    def fake_download_asset_history(
        asset: AssetSpec,
        *,
        start_date: object,
        end_date: object,
    ) -> (
        data_module
        .DownloadedAssetData
    ):
        if (
            asset.ticker
            == failing_ticker
        ):
            raise (
                data_module
                .DataAcquisitionError(
                    "synthetic acquisition failure"
                )
            )

        return (
            make_downloaded_asset(
                asset
            )
        )

    monkeypatch.setattr(
        data_module,
        "download_asset_history",
        fake_download_asset_history,
    )

    raw_dir = (
        tmp_path
        / "raw"
    )

    processed_dir = (
        tmp_path
        / "processed"
    )

    with pytest.raises(
        data_module
        .DataAcquisitionError,
        match=(
            "synthetic acquisition failure"
        ),
    ):
        (
            data_module
            .run_data_pipeline(
                start_date=(
                    "2005-01-01"
                ),
                end_date=(
                    "2026-01-01"
                ),
                raw_dir=raw_dir,
                processed_dir=(
                    processed_dir
                ),
            )
        )

    assert not (
        raw_dir.exists()
    )

    assert not (
        processed_dir.exists()
    )