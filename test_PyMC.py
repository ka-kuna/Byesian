import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

basic_model = pm.Model()
samples = np.random.binomial(1, 0.5, size=(100000,))

with basic_model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    k = pm.Bernoulli('k', p=theta, observed=samples)
    trace = pm.sample(100000)

pm.plot_posterior(trace[50000:],['theta'])