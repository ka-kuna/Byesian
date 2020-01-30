import numpy as np
import matplotlib.pyplot as plt
import random
import pymc3 as pm
import math
import theano

n = 100
x = np.linspace(-np.pi, np.pi, n)
def y(x):
  return x**3

num = []
for i in range(10):
  num.append(random.uniform(-10,10))
num = np.array(num)[:, None]

f = y(num)
f = f.reshape(10,)
sigma = 40*np.random.normal(0, 0.1, 10)
mu = f + sigma

with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
    η = pm.HalfCauchy("η", beta=5)
    cov = η**2 * pm.gp.cov.Matern52(1, ℓ)

    gp = pm.gp.Latent(cov_func=cov)

    f = gp.prior("f", X=num)

    σ = pm.HalfCauchy("σ", beta=5)
    ν = pm.Gamma("ν", alpha=2, beta=0.1)
    y_ = pm.StudentT("y", mu=f, lam=1.0/σ, nu=ν, observed=mu)

    trace = pm.sample(500)

n_new = 200
X_new = np.linspace(10, -10, n_new)[:,None]

with model:
    f_pred = gp.conditional("f_pred", X_new)
    pred_samples = pm.sample_posterior_predictive(trace, vars=[f_pred], samples=1000)

from pymc3.gp.util import plot_gp_dist
plot_gp_dist(plt, pred_samples["f_pred"], X_new)
num =  num[:,0]
plt.scatter(num, mu, c='r')

#plt.xlim(-5, 5)
#plt.ylim(y(-5), y(5))


plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graphs')

plt.show()