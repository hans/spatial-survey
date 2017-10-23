import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from pymc3.plots.kdeplot import fast_kde
from scipy import integrate
from theano import shared


def eig_datum(model, trace, steps, x):
    """
    Calculate the estimated expected information gain of observing a new datum.
    """

    # First try computing probability over assignments.
    # Hmm, this isn't doing conditional inference. Maybe obvious in retrospect..
    print(trace["dist_means"].mean(axis=0))
    d_points.set_value([x] * len(d_points_0))
    ppc = pm.sample_ppc(trace, vars=[model["assignments"]], samples=1000)
    print(ppc["assignments"].mean())

    # Add data point.
    d_points.set_value(d_points_0 + [x])

    for assignment in [0, 1]:
        d_assignments.set_value(d_assignments_0 + [assignment])
        result = pm.sample(2000, step=steps)

        # Drop first N hundred.
        result = result[500:]

        # Get KDE of posterior over means and estimate entropy.
        for distr in [0, 1]:
            kde_density, kde_min, kde_max = fast_kde(result["dist_means"][:, distr])
            # Convert to differential entropy
            kde_density = kde_density * np.log(kde_density + 1e-6)

            xs = np.linspace(kde_min, kde_max, len(kde_density))
            kde_entropy = integrate.simps(kde_density, xs)
            print(distr, kde_entropy)


types = ["near", "next to"]
d_assignments_0 = [0, 0, 0, 0, 1, 1, 1, 1]
d_assignments = shared(np.array(d_assignments_0, dtype=np.int32))
d_points_0 = [100, 120, 140, 110, 30, 25, 19, 36]
d_points = shared(np.array(d_points_0, dtype=np.float32))
n = len(d_points_0)#shared(np.array(d_points_0, dtype=np.int32))


model = pm.Model()
with model:
    dist_means = pm.Normal("dist_means", mu=[100] * len(types), sd=50, shape=len(types))
    dist_sd = 25#pm.Uniform("dist_sd", lower=0, upper=50)

    p = pm.Dirichlet("p", a=np.array([1., 1.]), shape=len(types))
    assignments = pm.Categorical("assignments", p=p, observed=d_assignments)

    # Likelihood for observed assignments.
    points = pm.Normal("points", mu=dist_means[assignments], sd=dist_sd,
                       observed=d_points)

    # Fit.
    mh = pm.Metropolis(vars=[dist_means, p])
    # sample = pm.ElemwiseCategorical(vars=[assignments], values=[0, 1])
    steps = [mh]
    result = pm.sample(2000, step=steps)
    result = result[500:]

    # pm.traceplot(result[2000:])
    # plt.tight_layout()
    # plt.savefig("./out.png")

    # #########

    # probability_datum(model, result, 30)
    eig_datum(model, result, steps, 30)

