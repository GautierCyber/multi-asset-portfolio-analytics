"""Investment universe specification for the multi-asset portfolio project."""

from dataclasses import dataclass
from typing import Literal


InstrumentType = Literal["ETF", "ETC"]
AssetClass = Literal["Equity", "Fixed Income", "Commodity"]
IncomePolicy = Literal["Accumulating", "Distributing", "N/A"]


@dataclass(frozen=True)
class AssetSpec:
    """Static metadata describing one investable asset proxy."""

    name: str
    ticker: str
    isin: str
    instrument_type: InstrumentType
    asset_class: AssetClass
    exposure: str
    benchmark: str
    exchange: str
    listing_currency: str
    fund_base_currency: str
    income_policy: IncomePolicy
    fx_hedged: bool | None
    economic_currency_exposure: str
    xetra_listing_date: str


ASSET_UNIVERSE: dict[str, AssetSpec] = {
    "DEVELOPED_EQUITY": AssetSpec(
        name="iShares Core MSCI World UCITS ETF",
        ticker="EUNL.DE",
        isin="IE00B4L5Y983",
        instrument_type="ETF",
        asset_class="Equity",
        exposure="Developed Markets Equities",
        benchmark="MSCI World Index (Net)",
        exchange="Xetra",
        listing_currency="EUR",
        fund_base_currency="USD",
        income_policy="Accumulating",
        fx_hedged=False,
        economic_currency_exposure="Multi-currency developed markets",
        xetra_listing_date="2009-10-20",
    ),
    "EMERGING_EQUITY": AssetSpec(
        name="iShares MSCI EM UCITS ETF",
        ticker="IQQE.DE",
        isin="IE00B0M63177",
        instrument_type="ETF",
        asset_class="Equity",
        exposure="Emerging Markets Equities",
        benchmark="MSCI Emerging Markets Index (Net)",
        exchange="Xetra",
        listing_currency="EUR",
        fund_base_currency="USD",
        income_policy="Distributing",
        fx_hedged=False,
        economic_currency_exposure="Multi-currency emerging markets",
        xetra_listing_date="2005-11-18",
    ),
    "EURO_GOVERNMENT_BONDS": AssetSpec(
        name="iShares Core EUR Govt Bond UCITS ETF",
        ticker="EUNH.DE",
        isin="IE00B4WXJJ64",
        instrument_type="ETF",
        asset_class="Fixed Income",
        exposure="Eurozone Investment Grade Government Bonds",
        benchmark="Bloomberg Euro Treasury Bond Index",
        exchange="Xetra",
        listing_currency="EUR",
        fund_base_currency="EUR",
        income_policy="Distributing",
        fx_hedged=None,
        economic_currency_exposure="EUR",
        xetra_listing_date="2009-10-20",
    ),
    "EURO_IG_CREDIT": AssetSpec(
        name="iShares Core EUR Corp Bond UCITS ETF",
        ticker="EUN5.DE",
        isin="IE00B3F81R35",
        instrument_type="ETF",
        asset_class="Fixed Income",
        exposure="Euro Investment Grade Corporate Bonds",
        benchmark="Bloomberg Euro Corporate Index",
        exchange="Xetra",
        listing_currency="EUR",
        fund_base_currency="EUR",
        income_policy="Distributing",
        fx_hedged=None,
        economic_currency_exposure="EUR",
        xetra_listing_date="2009-05-26",
    ),
    "GOLD": AssetSpec(
        name="WisdomTree Physical Gold",
        ticker="VZLD.DE",
        isin="JE00B1VS3770",
        instrument_type="ETC",
        asset_class="Commodity",
        exposure="Physical Gold",
        benchmark="Gold Spot Price",
        exchange="Xetra",
        listing_currency="EUR",
        fund_base_currency="USD",
        income_policy="N/A",
        fx_hedged=False,
        economic_currency_exposure="Gold / USD-linked",
        xetra_listing_date="2007-05-08",
    ),
    "BROAD_COMMODITIES": AssetSpec(
        name="WisdomTree Broad Commodities",
        ticker="OD7V.DE",
        isin="GB00B15KY989",
        instrument_type="ETC",
        asset_class="Commodity",
        exposure="Diversified Commodity Futures",
        benchmark="Bloomberg Commodity Commodities 4W Total Return",
        exchange="Xetra",
        listing_currency="EUR",
        fund_base_currency="USD",
        income_policy="N/A",
        fx_hedged=False,
        economic_currency_exposure="USD-based global commodities",
        xetra_listing_date="2006-11-03",
    ),
}


TICKERS: tuple[str, ...] = tuple(
    asset.ticker for asset in ASSET_UNIVERSE.values()
)