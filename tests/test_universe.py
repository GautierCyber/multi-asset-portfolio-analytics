from datetime import date

from multi_asset_portfolio.universe import ASSET_UNIVERSE, TICKERS


EXPECTED_TICKERS = (
    "EUNL.DE",
    "IQQE.DE",
    "EUNH.DE",
    "EUN5.DE",
    "4GLD.DE",
    "EXXY.DE",
)


def test_universe_contains_expected_assets() -> None:
    assert len(ASSET_UNIVERSE) == 6
    assert TICKERS == EXPECTED_TICKERS


def test_tickers_are_unique() -> None:
    assert len(TICKERS) == len(set(TICKERS))


def test_asset_identifiers_are_unique() -> None:
    isins = [
        asset.isin
        for asset in ASSET_UNIVERSE.values()
    ]

    assert len(isins) == len(set(isins))


def test_all_assets_use_xetra_eur_listings() -> None:
    assert all(
        asset.exchange == "Xetra"
        and asset.listing_currency == "EUR"
        and asset.ticker.endswith(".DE")
        for asset in ASSET_UNIVERSE.values()
    )


def test_listing_dates_are_valid_iso_dates() -> None:
    for asset in ASSET_UNIVERSE.values():
        parsed = date.fromisoformat(
            asset.xetra_listing_date
        )

        assert parsed <= date.today()


def test_gold_proxy_metadata_is_consistent() -> None:
    gold = ASSET_UNIVERSE["GOLD"]

    assert gold.ticker == "4GLD.DE"
    assert gold.isin == "DE000A0S9GB0"
    assert gold.instrument_type == "ETC"
    assert gold.asset_class == "Commodity"
    assert gold.fx_hedged is False


def test_broad_commodity_proxy_metadata_is_consistent() -> None:
    commodities = ASSET_UNIVERSE[
        "BROAD_COMMODITIES"
    ]

    assert commodities.ticker == "EXXY.DE"
    assert commodities.isin == "DE000A0H0728"
    assert commodities.instrument_type == "ETF"
    assert commodities.asset_class == "Commodity"
    assert commodities.fund_base_currency == "EUR"
    assert commodities.fx_hedged is False


def test_fx_metadata_matches_economic_exposure() -> None:
    assert (
        ASSET_UNIVERSE[
            "DEVELOPED_EQUITY"
        ].fx_hedged
        is False
    )

    assert (
        ASSET_UNIVERSE[
            "EMERGING_EQUITY"
        ].fx_hedged
        is False
    )

    assert (
        ASSET_UNIVERSE[
            "GOLD"
        ].fx_hedged
        is False
    )

    assert (
        ASSET_UNIVERSE[
            "BROAD_COMMODITIES"
        ].fx_hedged
        is False
    )

    assert (
        ASSET_UNIVERSE[
            "EURO_GOVERNMENT_BONDS"
        ].fx_hedged
        is None
    )

    assert (
        ASSET_UNIVERSE[
            "EURO_IG_CREDIT"
        ].fx_hedged
        is None
    )