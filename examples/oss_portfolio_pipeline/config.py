"""The agreed investment universe, strategic benchmark and dated construction settings."""
from dataclasses import dataclass

import pandas as pd


# Ticker, descriptive label, reporting sleeve. Listed real assets remain equity-like risk.
INSTRUMENTS = (
    ('SPY', 'US large caps', 'Equities'),
    ('QQQ', 'Nasdaq 100', 'Equities'),
    ('IWM', 'US small caps', 'Equities'),
    ('EFA', 'Developed ex-US/Canada', 'Equities'),
    ('VGK', 'Europe', 'Equities'),
    ('EWJ', 'Japan', 'Equities'),
    ('EWA', 'Australia', 'Equities'),
    ('EEM', 'Emerging markets', 'Equities'),
    ('AAXJ', 'Asia ex-Japan', 'Equities'),
    ('MCHI', 'China', 'Equities'),
    ('INDA', 'India', 'Equities'),
    ('QUAL', 'US quality', 'Equity styles'),
    ('MTUM', 'US momentum', 'Equity styles'),
    ('USMV', 'US minimum volatility', 'Equity styles'),
    ('VLUE', 'US value', 'Equity styles'),
    ('SHY', 'US Treasuries 1-3 years', 'Government bonds'),
    ('IEF', 'US Treasuries 7-10 years', 'Government bonds'),
    ('TLT', 'US Treasuries 20+ years', 'Government bonds'),
    ('GOVT', 'Broad US Treasuries', 'Government bonds'),
    ('IGOV', 'Developed non-US government bonds', 'Government bonds'),
    ('TIP', 'US inflation-linked Treasuries', 'Government bonds'),
    ('VCSH', 'Short investment-grade corporates', 'Credit and EM debt'),
    ('LQD', 'Investment-grade corporates', 'Credit and EM debt'),
    ('HYG', 'High-yield corporates', 'Credit and EM debt'),
    ('SHYG', 'Short high-yield corporates', 'Credit and EM debt'),
    ('EMB', 'USD emerging-market sovereign debt', 'Credit and EM debt'),
    ('EMLC', 'Local-currency emerging-market debt', 'Credit and EM debt'),
    ('VNQ', 'US listed property', 'Listed real assets'),
    ('VNQI', 'Non-US listed property', 'Listed real assets'),
    ('IGF', 'Global listed infrastructure', 'Listed real assets'),
    ('DBC', 'Diversified commodity futures', 'Commodities'),
    ('DBE', 'Energy futures', 'Commodities'),
    ('DBA', 'Agriculture futures', 'Commodities'),
    ('DBB', 'Industrial-metals futures', 'Commodities'),
    ('GLD', 'Gold', 'Commodities'),
    ('BIL', 'US Treasury bills', 'Cash'),
)
FACTOR_ONLY_TICKERS = ('URTH', 'DBP', 'UUP', 'CEW')
BENCHMARK_PERCENT = {
    'SPY': 25, 'EFA': 15, 'EEM': 10,
    'GOVT': 15, 'IGOV': 5, 'TIP': 5, 'LQD': 5, 'HYG': 3, 'EMB': 2,
    'GLD': 5, 'DBC': 5, 'BIL': 5,
}


@dataclass(frozen=True)
class PipelineConfig:
    """Dates are inclusive starts/exclusive end; vol span and warm-up count weekly returns."""

    end: str
    start: str = '2014-12-01'
    factor_start: str = '2017-12-31'
    backtest_start: str = '2019-12-31'
    vol_span: int = 156
    vol_warmup_period: int = 13

    def __post_init__(self) -> None:
        """Reject ambiguous dates and configurations without an estimation history."""
        dates = [pd.Timestamp(getattr(self, key)) for key in
                 ('start', 'factor_start', 'backtest_start', 'end')]
        if any(pd.isna(d) or d.tzinfo is not None or d != d.normalize() for d in dates):
            raise ValueError('Dates must be timezone-naive calendar dates')
        if not dates[0] < dates[1] <= dates[2] < dates[3]:
            raise ValueError('Require start < factor_start <= backtest_start < end')
        for name in ('vol_span', 'vol_warmup_period'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'{name} must be an integer of at least two weekly periods')


def all_tickers() -> tuple[str, ...]:
    """Return the 36 investable ETFs followed by the four additional factor sources."""
    return tuple(row[0] for row in INSTRUMENTS) + FACTOR_ONLY_TICKERS


def benchmark_weights() -> pd.Series:
    """Return the 100% strategic target, including explicit zeros for active alternatives."""
    return pd.Series(BENCHMARK_PERCENT, dtype=float).reindex(
        [row[0] for row in INSTRUMENTS], fill_value=0.0).div(100.0).rename('benchmark_weight')


def universe_table() -> pd.DataFrame:
    """Return labels, reporting sleeves, economic groups and benchmark targets."""
    table = pd.DataFrame(INSTRUMENTS, columns=['ticker', 'name', 'group']).set_index('ticker')
    table['economic_group'] = table['group'].replace({
        'Equity styles': 'Equities', 'Listed real assets': 'Equities'})
    table['benchmark_weight'] = benchmark_weights()
    return table
