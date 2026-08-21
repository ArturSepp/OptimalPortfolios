

here are my ideas on the cluster and classification framework

we consider covar/corr matrix Sigma\_t computed using EWMA with span N

we can remove PC-1 component

let mapping f(p) with some set of parameters P to project Sigma\_t into set of Q clusters, where Q is function of Sigma\_t and p:
Q(Sigma\_t, p)

I am certain that under specific conditions that number of assets increase to infinity there must be a stationary 

distribution of Q(Sigma\_t, P)

todo: check what literature exist for this

No le't consider changes in delta Sigma\_t = Sigma\_t - Sigma\_{t-1}


These changes are function of span N and generic process for returns generation

Let's assume iid for returns and AR-1 or IGARCH or GARCH for volatility with some parameter set Theta



Now delta Sigma\_t = f(Theta, P; X) where f is some explicit or implicit function and X is data vector

I am sure there must be some asymptotic law of delta Sigma\_t

now we can also then derive from here the asymptotic law of changes delta Q =  Q(Sigma\_t, P) -  Q(Sigma\_{t-1}, P)

with delta Q being function of P, Theta and data vector X

what we want is to minimize the variance of delta Q using the parameter set P

todo: check if my idea is feasible and literature exists on this subject

