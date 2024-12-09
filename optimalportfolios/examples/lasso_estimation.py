# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from scipy.optimize import minimize
from enum import Enum

import yfinance as yf
import qis as qis

from optimalportfolios.utils.lasso import solve_lasso, compute_r2

asset = 'HYG'
tickers = ['SPY', 'IEF', 'LQD', 'USO', 'GLD', 'UUP']
prices = yf.download(tickers+[asset], start=None, end=None)['Adj Close'].dropna()
print(prices)

returns = qis.to_returns(prices, freq='ME', drop_first=True)

x = returns[tickers].to_numpy()
y = returns[asset].to_numpy()

span = 24
beta = solve_lasso(x=x, y=y, reg_lambda=1e-8, span=span, nonneg=True)
beta = pd.Series(beta, index=tickers)
print(beta)
print(compute_r2(x, y, beta))

