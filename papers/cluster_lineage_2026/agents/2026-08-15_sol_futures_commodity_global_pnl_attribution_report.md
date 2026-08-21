# Futures Commodities global net-P&L attribution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The accepted standalone global Commodities long-short book was attributed contract by
contract. The frozen construction is q=20%, production 48-week-minus-4-week momentum,
203 monthly decisions, one W-WED implementation lag, and 20 bp costs over the corrected
U1 calendar window.

Following the owner's exclusion ruling, `CUA1 Comdty` (ethanol) is ineligible on every
date. The portfolio earned 10.4623 percentage points of total net P&L from 2009-09-02
through 2026-06-24. Twenty-one of 33 eligible commodity futures contributed positively
and 12 negatively. TSI iron ore, London cocoa, palladium, US cocoa, and milling wheat
were the five largest gains. ULSD, lean hogs, Brent, gasoline, and WTI were the five
largest losses.

## Ranked net contribution

Contributions are percentage points of beginning net NAV, ordered from highest to lowest.
They are cumulative, not annualised. Names reproduce the source taxonomy labels.

| rank | ticker | name | subclass | net P&L contribution |
|---:|---|---|---|---:|
| 1 | SCO1 Comdty | TSI IRON ORE | Metals | +20.05% |
| 2 | QC1 Comdty | LONDON COCOA | Agriculture | +18.45% |
| 3 | PA1 Comdty | PALLADIUM | Metals | +15.32% |
| 4 | CC1 Comdty | COCOA | Agriculture | +15.01% |
| 5 | CA1 Comdty | WHEAT MIL | Agriculture | +11.22% |
| 6 | QW1 Comdty | WHITE SUGAR | Agriculture | +10.00% |
| 7 | GC1 Comdty | GOLD | Metals | +7.89% |
| 8 | BO1 Comdty | SOYB OIL | Agriculture | +6.66% |
| 9 | CT1 Comdty | Cotton #2 | Agriculture | +6.50% |
| 10 | MODEC1 Comdty | CARBON CREDIT | Energy | +5.01% |
| 11 | FC1 Comdty | FEEDER CATTLE | Agriculture | +4.95% |
| 12 | IJ1 Comdty | RAPESEED | Agriculture | +4.74% |
| 13 | SI1 Comdty | SILVER | Metals | +4.12% |
| 14 | HG1 Comdty | COPPER | Metals | +3.53% |
| 15 | DF1 Comdty | ROBUSTA | Agriculture | +2.75% |
| 16 | SB1 Comdty | SUGAR #11 | Agriculture | +2.52% |
| 17 | RS1 Comdty | CANOLA | Agriculture | +1.97% |
| 18 | LC1 Comdty | LIVE CATTLE | Agriculture | +1.56% |
| 19 | KC1 Comdty | COFFEE C | Agriculture | +1.48% |
| 20 | S 1 Comdty | SOYBEANS | Agriculture | +1.26% |
| 21 | MWE1 Comdty | WHEAT MINNEAPOLIS | Agriculture | +0.53% |
| 22 | SM1 Comdty | SOYB MEAL | Agriculture | -1.00% |
| 23 | C 1 Comdty | CORN | Agriculture | -4.87% |
| 24 | W 1 Comdty | WHEAT CHICAGOE | Agriculture | -5.14% |
| 25 | PL1 Comdty | PLATINUM | Metals | -5.50% |
| 26 | KW1 Comdty | WHEAT KANSAS | Agriculture | -7.49% |
| 27 | NG1 Comdty | NAT GAS | Energy | -11.08% |
| 28 | QS1 Comdty | GASOIL | Energy | -11.30% |
| 29 | CL1 Comdty | WTI | Energy | -11.48% |
| 30 | XB1 Comdty | GASOLINE | Energy | -17.78% |
| 31 | CO1 Comdty | BRENT | Energy | -18.89% |
| 32 | LH1 Comdty | LEAN HOGS | Agriculture | -20.01% |
| 33 | HO1 Comdty | NYH ULSD | Energy | -20.52% |

Gross instrument P&L sums to +43.0611% of beginning NAV. Instrument transaction costs
sum to 32.5988%, leaving +10.4623% net. This cumulative cost number is not the annualised
cost-drag statistic; the portfolio's annualised cost drag is 250.43 bp.

## Attribution convention and reconciliation

For instrument *i* on date *t*, exact net currency P&L is

`units[i,t-1] * (price[i,t] - price[i,t-1]) - realised_cost[i,t]`.

The first reported NAV observation is the attribution base, so its already-incurred
initial allocation effect is not counted again. Summing the subsequent instrument P&L
and dividing by beginning NAV gives the reported percentage-point contributions.

This path is used instead of `PortfolioData.get_instruments_pnl(is_net=True)` because the
current qis accessor applies its default 260-observation rolling cost aggregation. Raw
stored realised costs are required for a one-date P&L identity.

Measured reconciliation versus tolerance:

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| contracts ranked | 33 / 33 eligible | exact | PASS |
| CUA1 eligible dates / non-zero weights | 0 / 0 | exact | PASS |
| futures eligible per date | 88 to 94 | point-in-time | PASS |
| commodities eligible per date | 31 to 33 | point-in-time | PASS |
| eligible assets lacking valid score | 0 | exact | PASS |
| maximum eligible assets with partial lookback | 3 | diagnostic | REPORTED |
| maximum daily instrument-sum vs NAV-change error | 3.653e-14 | 1e-10 | PASS |
| cumulative instrument-sum vs NAV-change error | 1.421e-14 | 1e-10 | PASS |
| P&L outside Commodities | 0.000e+00 | 1e-10 | PASS |
| ranked-total reconciliation error | 1.066e-14 | 1e-10 | PASS |
| deterministic artifacts | 5 / 5 byte-identical | 100% | PASS |

The attributed NAV moves from 100.0000 to 110.4623. Instrument net P&L of 10.4623 NAV
units therefore matches the portfolio's +10.4623% net total return exactly to floating
precision.

## Point-in-time availability

The backtest does account for the changing cross-section. There are 88 eligible futures
at the beginning of the window and 94 after all late entrants have passed the source
eligibility warmup. Commodities increase from 31 to 33 when TSI iron ore and carbon
credits enter on 2016-03-31. At q=20%, each commodity side contains seven contracts both
before and after those entries.

The six contracts entering after the first decision are WN1 (2010-04-30), MODEC1 and
SCO1 (2016-03-31), BMR1 (2018-03-31), SFR5 (2018-07-31), and SFI5 (2018-08-31).
Eligibility and valid-score counts coincide on all 203 decisions.

However, the executed momentum implementation sums up to 48 observations with
`min_count=1`; it does not require all 48 pre-skip observations. Thus late contracts use
shorter momentum histories after the eligibility warmup. This affects 32 of 203 total
decision dates, with at most three partial-history assets on one date. For Commodities,
it affects nine dates and at most two assets: MODEC1 and SCO1 enter with nine usable
observations on 2016-03-31 and first reach all 48 on 2016-12-31. This is documented as a
separate convention issue and was not silently changed in the ethanol-exclusion rerun.

## Verification

Fail-first collection before implementation:

```text
ModuleNotFoundError: No module named
'papers.cluster_lineage_2026.replication.run_futures_commodity_pnl_attribution'
```

The transaction-cost omission was then deliberately reintroduced. The focused test
failed on the first instrument contribution (`2.0` measured versus `1.8` expected). After
restoring costs:

The owner-exclusion test was also shown to fail while CUA1 remained eligible, then passed
after the centralized point-in-time mask was installed.

```text
........                                                                 [100%]
```

Direct lint:

```text
All checks passed!
```

## Artifacts

- Runner: `papers/cluster_lineage_2026/replication/run_futures_commodity_pnl_attribution.py`
- Test: `papers/cluster_lineage_2026/replication/futures_commodity_pnl_attribution_test.py`
- Ranked output: `e5b/futures_asset_class_long_short_u1_window/commodity_global_q020_attribution/ranking.csv`
- Acceptance: `e5b/futures_asset_class_long_short_u1_window/commodity_global_q020_attribution/acceptance.csv`
- Availability: `e5b/futures_asset_class_long_short_u1_window/commodity_global_q020_attribution/availability_by_date.csv`
- Contract histories: `e5b/futures_asset_class_long_short_u1_window/commodity_global_q020_attribution/contract_history.csv`
- Determinism: `e5b/futures_asset_class_long_short_u1_window/commodity_global_q020_attribution/determinism.csv`
