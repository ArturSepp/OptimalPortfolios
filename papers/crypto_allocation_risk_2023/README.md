# Optimal Allocation to Cryptocurrencies in Diversified Portfolios

This directory contains the simulations for:

Sepp A. (2023), "Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
*Risk*, October 2023, 1–6. [SSRN 4217841](https://ssrn.com/abstract=4217841).

## Layout

- `paper/` — working-paper TeX and rendered PDF.
- `replication/` — all Python data, analysis, reporting, and verification code.
- `update_2026/` — the 2026 update analysis and roadmap.
- `data/` — redistributable inputs plus gitignored licensed Bloomberg snapshots.
- `outputs/` — gitignored generated tables, manifests, and reports.

## Published-paper update first

`replication/published_update.py` is the canonical update runner. It keeps the acquired data,
analysis code, tables, manifests, and reports inside this paper directory. Its default engine
freezes the August 2024 implementation at OptimalPortfolios commit `6038fba`; `current_v7_1` is an
explicitly labelled diagnostic, not the published-paper result. See
[`update_2026/PUBLISHED_UPDATE_2026.md`](update_2026/PUBLISHED_UPDATE_2026.md) for the findings and
next-study roadmap.

Use the repository's external Python environment. Never use or create a `.venv` under OneDrive:

```powershell
C:\Python\OptimalPortfolios312\Scripts\python.exe -m papers.crypto_allocation_risk_2023.replication.published_update all --as-of 2026-09-04 --engine published_2024
```

The `all` action first creates an immutable Bloomberg snapshot and verifies it before running the
analysis. To rerun an existing verified snapshot without another terminal request:

```powershell
C:\Python\OptimalPortfolios312\Scripts\python.exe -m papers.crypto_allocation_risk_2023.replication.published_update analyse --snapshot bbg_20260904 --engine published_2024
```

The primary result retains the paper's legacy ETH convention: Bloomberg `XETUSD Curncy` is
backfilled before its first observation by scaled Bloomberg BTC. The observed-ETH sensitivity uses
only actual `XETUSD Curncy` history and applies a common 60-month warm-up to all four methods:

```powershell
C:\Python\OptimalPortfolios312\Scripts\python.exe -m papers.crypto_allocation_risk_2023.replication.published_update analyse --snapshot bbg_20260904 --engine published_2024 --observed-eth --no-report
```

## Bloomberg data contract

The update route is Bloomberg-only. It requests adjusted `PX_LAST` for the investable equity/ETF
legs (`SPY`, `IEF`, `PSP`, `IYR`, `REET`, `GSG`, `COMT`, and `GLD`) and unadjusted `PX_LAST` for
`XBTUSD Curncy`, `XETUSD Curncy`, `HFRXGL Index`, `HFRIMDT Index`, `NEIXCTA Index`, and `GB3 Govt`.
The 3-month bill yield is converted from percentage points to a decimal rate. `HFRIMDT` is the
Macro series used by the August 2024 update.

Licensed snapshots live under `data/bloomberg/<snapshot-tag>/` and are gitignored. Each snapshot
contains raw Bloomberg observations, both derived ETH panels, the risk-free series, a coverage
table, and a SHA-256 manifest. Verification rebuilds every derived panel from the saved raw data and
checks schema, hashes, cutoff, history, staleness, and gaps.

Generated artifacts default to `outputs/<snapshot-tag>/<engine>/<eth-mode>/` under this directory
and are also gitignored. The JSON analysis manifest records the exact data-manifest hash, engine,
runtime versions, reporting starts, parameters, output hashes, and portfolio validation metrics.

## Verification

The offline contracts cover the Bloomberg request specification, immutable snapshot validation,
derivations, optimizer constraints, no-look-ahead estimator inputs, and short-history eligibility:

```powershell
C:\Python\OptimalPortfolios312\Scripts\python.exe -m pytest papers\crypto_allocation_risk_2023\replication\update_test.py papers\crypto_allocation_risk_2023\replication\parity_2024_test.py -q
```

When the private archived 2024 panel and workbook are available, opt into the full 16-case numerical
replay against the workbook oracle:

```powershell
$env:RUN_CRYPTO_PARITY_GOLDEN = '1'
C:\Python\OptimalPortfolios312\Scripts\python.exe -m pytest papers\crypto_allocation_risk_2023\replication\parity_2024_test.py -k private_archived_panel -q
Remove-Item Env:RUN_CRYPTO_PARITY_GOLDEN
```

## Historical reproduction route

The older `replication/article_figures.py`,
`replication/backtest_portfolios_for_article.py`, and Yahoo/manual-data helpers remain for historical
reproduction. They are not the Bloomberg-only update pipeline. Redistributable frozen inputs remain
in `data/`; the two licensed SG workbooks required by the old mixed-source route cannot be shipped.
