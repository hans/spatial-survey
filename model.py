import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm


types = ["near", "next to"]
assignments = [0, 0, 1, 1]
points = [100, 120, 30, 25]


model = pm.Model()
with model:
    types = ["near", "next to"]

    dist_means = pm.Normal("dist_means", mu=[100] * len(types), sd=50, shape=len(types))
    dist_sd = pm.Uniform("dist_sd", lower=0, upper=50)

    # Likelihood for observed assignments.
    points = pm.Normal("points", mu=dist_means[assignments], sd=dist_sd,
                       observed=points)

    # Fit.
    mh = pm.Metropolis(vars=[dist_means, dist_sd])
    result = pm.sample(1000, step=mh)

    pm.traceplot(result[250:])
    plt.tight_layout()
    plt.savefig("./out.png")
