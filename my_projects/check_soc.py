import numpy as np
import pandas as pd
import yfinance as yf
import warnings

import cvxpy as cp
from timeit import default_timer as timer

warnings.filterwarnings("ignore")

yf.pdr_override()
pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2016-01-01'
end = '2019-12-30'

# Tickers of assets
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
          'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
          'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
assets.sort()

# Downloading data
data = yf.download(assets, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets

# Calculating returns
Y = data[assets].pct_change().dropna()

print(Y)


####################################
# Minimizing Portfolio Variance
####################################

# Defining initial inputs
mu = Y.mean().to_numpy().reshape(1,-1)
sigma = Y.cov().to_numpy()

# Defining initial variables
x = cp.Variable((mu.shape[1], 1))

# Budget and weights constraints
constraints = [cp.sum(x) == 1,
               x <= 1.0,
               x >= 0.0]

# Defining risk objective
risk = cp.quad_form(x, sigma)
objective = cp.Minimize(risk)

weights = pd.DataFrame([])
# Solving the problem with several solvers
prob = cp.Problem(objective, constraints)
solvers = ['ECOS', 'SCS']
for i in solvers:
    prob.solve(solver=i)
    # Showing Optimal Weights
    weights_1 = pd.DataFrame(x.value, index=assets, columns=[i])
    weights = pd.concat([weights, weights_1], axis=1)

print(weights)


#########################################
# Maximizing Portfolio Return with
# Variance Constraint
#########################################

# Defining initial inputs
mu = Y.mean().to_numpy().reshape(1,-1)
sigma = Y.cov().to_numpy()

# Defining initial variables
x = cp.Variable((mu.shape[1], 1))
sigma_hat = 15 / (252**0.5 * 100)
ret = mu @ x

# Budget and weights constraints
constraints = [cp.sum(x) == 1,
               x <= 1.0,
               x >= 0.0]

# Defining risk constraint and objective
risk = cp.quad_form(x, sigma)
constraints += [risk <= sigma_hat**2] # variance constraint
objective = cp.Maximize(ret)

weights = pd.DataFrame([])
# Solving the problem with several solvers
prob = cp.Problem(objective, constraints)
solvers = ['ECOS', 'SCS']
for i in solvers:
    prob.solve(solver=i)
    # Showing Optimal Weights
    weights_1 = pd.DataFrame(x.value, index=assets, columns=[i])
    weights = pd.concat([weights, weights_1], axis=1)

print(weights)