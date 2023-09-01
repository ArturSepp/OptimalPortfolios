Implementation of simulations for paper:

Sepp A. (2023) Optimal Allocation to Cryptocurrencies in Diversified Portfolios, Risk (forthcoming)
Available at SSRN: https://ssrn.com/abstract=4217841

The analysis presented in the paper can be replicated or extended using this module

Implementation steps:
1) Populate the time series of asset prices in the investable universe using
```python 
optimaportfolios/examples/crypto_allocation/load_prices.py
```

Price data for some assets can be fetched from local csv files, some can be generated on the fly 

Run
```python 
update_prices() 
```

2) Generate article figures using unit tests in
 ```python 
optimaportfolios/examples/crypto_allocation/article_figures.py
```

3) Generate reports of simulated investment portfolios as reported in the article
 ```python 
optimaportfolios/examples/crypto_allocation/backtest_crypto_portfolios.py
```

