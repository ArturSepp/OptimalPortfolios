We continue the development of cluster linage methodology for my OSS stack and paper for Quantitative Finance

Read background here:

C:\\Users\\artur\\OneDrive\\analytics\\my\_github\\OptimalPortfolios\\papers\\cluster\_lineage\_2026\\agents

here is my outline for the paper

1\) motivation of the paper (1-2 pages):

&#x09;Clustering is important and acknowledged for finance (HCRP)

&#x09;Stability of clusters has never been studies for portfolio optimisation (check it)

&#x09;Frequently changed clusters increase turnover

&#x09;Cluster interpretability is also important for PM, risk, etc but nothing is available yet for finance (check it)

2\) our contribution (1-2 pages):

&#x09;We must be bold in what we propose here – ideally we can claim an original smoothing/labelling algo that bears my name specifically:

&#x09;Cluster smoothing must be somehow linked to the span of covariance matrix estimator

&#x09;Another idea – right now we look at the EWMA covar: could we do adjustment by the first PC and look at the covariance of residuals – could it be more stable?

&#x09;Then labelling – in our MATF framework clusters can be associated with factors and we derived labels from there. Can be this presented as actual algo?

3\) method development (up to 10 pages):

&#x09;Robust smoothing for clustering

&#x09;Robust labelling of clusters

4\) empirical section (up to 10 pages):

We must show that our clusters are stable and make sense for different datasets

I aligned 3 datasets in paper folder data

i)	Mac funds log-returns for ME and QE – this will come along with CUSTOM MATF risk model – our application of MATF-CMA

ii)	S\&P500 returns (proxied using MSCI US data) – this come along with FF factors data (yet to be collected) – this is classic application for equity quant modeling. The target frequency here is W-WED or ME dependent on frequency of FF factors

iii)	Futures return – this is interesting  because of global markets – the factor model can be EQUITY REGIONAL or CUSTOM MATF 

(for claude agent: check rosaa/data/matf)

Frequency can be ‘B’ – we can test different frequencies for our empirical pipeline

Then for actual illustration of our clustering stability – I am thinking of actually running a similar analysis to C:\\Users\\artur\\OneDrive\\analytics\\my\_github\\OptimalPortfolios\\rosaa\\products\\funds\\analysis\\run\_cross\_mandate\_analysis.py

for each universe, we do clustering and for clusters we do ranks based on momentum and low beta and then we create top quantile long portfolio. Then for each clustering method we can analyse the performance and turnover, etc to emphasize the importance of stable clustering

we do this analysis for our three universes to generalise our results

5\) conclusion (1 page)

6\) Appendicex – keep to minimum – only algos

7\) Replication code in optimalportfolios/papers

&#x09;





