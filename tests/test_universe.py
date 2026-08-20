from multi_asset_portfolio.universe import ASSET_UNIVERSE, TICKERS


def test_universe_contains_expected_number_of_assets() -> None:
    assert len(ASSET_UNIVERSE) == 6
    assert len(TICKERS) == 6


def test_tickers_are_unique() -> None:
    assert len(TICKERS) == len(set(TICKERS))


def test_all_assets_are_listed_in_eur() -> None:
    assert all(
        asset.listing_currency == "EUR"
        for asset in ASSET_UNIVERSE.values()
    )


def test_asset_identifiers_are_unique() -> None:
    isins = [asset.isin for asset in ASSET_UNIVERSE.values()]

    assert len(isins) == len(set(isins))


def test_all_tickers_are_xetra_tickers() -> None:
    assert all(
        asset.ticker.endswith(".DE")
        for asset in ASSET_UNIVERSE.values()
    )


def test_fx_hedging_metadata_is_consistent() -> None:
    for asset in ASSET_UNIVERSE.values():
        if asset.fund_base_currency == "EUR":
            assert asset.fx_hedged is None
        else:
            assert asset.fx_hedged is False