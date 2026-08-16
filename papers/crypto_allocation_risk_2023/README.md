# Optimal Allocation to Cryptocurrencies in Diversified Portfolios

This directory contains the simulations for:

Sepp A. (2023), "Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
*Risk*, October 2023, 1–6. [SSRN 4217841](https://ssrn.com/abstract=4217841).

## Environment

From a repository checkout, install the optional data and reporting dependencies explicitly:

```bash
uv sync --extra dev --extra data --extra reports
```

The scripts use the current checkout. They do not carry PEP 723 metadata that could silently install
a released `optimalportfolios` version instead.

## Data

The `data/` directory includes the CSV inputs that can be redistributed. Rebuilding the combined
price file with `update_prices_with_yf()` additionally requires two licensed Société Générale index
workbooks that the repository cannot ship:

- `CTA_Historical.xlsx` — [SG CTA Index](https://wholesale.banking.societegenerale.com/en/prime-services-indices/)
- `Macro_Trading_Index_Historical.xlsx` — [SG Macro Trading Index](https://wholesale.banking.societegenerale.com/fileadmin/indices_feeds/Macro_Trading_Index_Historical.xls)

Place authorised `.xlsx` copies beside the CSV files in `data/`. The loader checks for both before
starting any Yahoo download and reports their exact paths when either is absent. The Bloomberg route
requires a Bloomberg terminal and `bbg-fetch`; it is not the public reproduction path.

Yahoo price histories are downloaded at runtime and may be revised by the provider. A fresh download
therefore reproduces the method, not necessarily the exact published values. The committed combined
CSV files are the stable inputs for comparing the stored analysis.

## Running the scripts

```bash
uv run --no-sync python papers/crypto_allocation_risk_2023/load_prices.py
uv run --no-sync python papers/crypto_allocation_risk_2023/article_figures.py
uv run --no-sync python papers/crypto_allocation_risk_2023/backtest_portfolios_for_article.py
```

Generated reports default to the gitignored
`outputs/crypto_allocation_risk_2023/` directory at the repository root. Override it without editing
the scripts by setting `OPTIMALPORTFOLIOS_OUTPUT_DIR`; for example in PowerShell:

```powershell
$env:OPTIMALPORTFOLIOS_OUTPUT_DIR = 'D:\portfolio-reports\crypto'
```
