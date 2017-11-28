import itertools

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from pymc3.plots.kdeplot import kdeplot
from pymc3.plots.artists import kdeplot_op
from theano import shared
from tqdm import tqdm, trange

from spatial.active import EIGPredictor


def active_loop(model, data_sampler, assignment_fn,
                batch_size=10, **eig_args):
  """
  Run an active learning loop, incrementally requesting labels for potential
  new samples.

  Args:
    model: pymc3 model
    data_sampler: generator which yields observed data points
    assignment_fn: function which provides labels for sampled data points.
      (We're aiming to minimize calls to this function.)
    batch_size: Number of samples to compare on each iteration before
      requesting a single label.
  """

  with model:
    # Inference steps
    steps = [pm.Metropolis()]

    i = 0
    while True:
      # Run a first MH inference.
      result = pm.sample(2000, step=steps)
      # Burn-in.
      result = result[1000:]

      # Plot result at this iter.
      fig, ax = plt.subplots()
      kdeplot_op(ax, result["dist_means"])
      fig.tight_layout()
      fig.savefig("active.%02i.png" % i)

      eig_predictor = EIGPredictor(model, k, result, steps, **eig_args)

      # Sample and score candidates.
      samples = []
      for _, sample in tqdm(zip(range(batch_size), data_sampler),
                            desc="Drawing and scoring samples",
                            total=batch_size):
        score = eig_predictor.eig(sample)
        samples.append((score, sample))

      # Request label for max-scoring candidate.
      _, sample = max(samples, key=lambda x: x[0])
      label = assignment_fn(sample)

      # Add to dataset.
      # TODO: generalize
      d_points.set_value(
          np.concatenate([d_points.get_value(), [sample]]).astype(np.float32))
      d_assignments.set_value(
          np.concatenate([d_assignments.get_value(), [label]]).astype(np.int32))

      i += 1


def eig_demo(model):
  """
  Basic EIG demo for this model. Saves a figure `after.png`.
  """
  with model:
    # Fit.
    mh = pm.Metropolis(vars=[term_means, term_sd])
    steps = [mh]
    result = pm.sample(2000, step=steps)
    result = result[1000:]

    # #########

    xs = np.tile(np.linspace(-20, 150, 2)[:, np.newaxis],
                 (1, d))

    eig_predictor = EIGPredictor(model, k, result, steps,
                                 opt_vars=[term_means, term_sd])
    eigs = np.array([eig_predictor.eig(x)
                     for x in tqdm(xs, desc="EIG at points")])

    from pprint import pprint
    pprint(list(zip(xs, eigs)))

    # ##########

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("density")

    # Plot KDE of dist_means
    kdeplot_op(ax1, result["dist_means"])

    # Plot EIG samples.
    ax2 = ax1.twinx()
    ax2.set_ylabel("EIG")
    ax2.scatter(xs, eigs, c='r')

    plt.tight_layout()
    plt.savefig("after.png")


types = ["near", "next to"]
# d_assignments_0 = [0, 0, 0, 0, 1, 1, 1, 1]
# d_points_0 = [100, 120, 140, 110, 30, 25, 19, 36]
d_assignments_0 = [0, 1]
d_points_0 = np.array([[100, 5],
                       [30, 10]])

d_assignments = shared(np.array(d_assignments_0, dtype=np.int32))
d_points = shared(d_points_0.astype(np.float32))

# n: # observed points
# d: dimensionality of observations
# k: number of underlying spatial relations
n, d = d_points_0.shape
k = len(types)


model = pm.Model()
with model:
  term_means = pm.MvNormal("dist_means",
      mu=50. * np.ones((k, d)),
      cov=50. * np.eye(d),
      shape=(k, d))
  term_sd = pm.Uniform("dist_sd", lower=0, upper=10)

  assignments = pm.Categorical("assignments",
      p=np.ones(k), observed=d_assignments)

  # Likelihood for observed assignments.
  points = pm.MvNormal("points",
      mu=term_means[assignments],
      cov=np.eye(d),
      observed=d_points)


if __name__ == '__main__':
  eig_demo(model)

#   # Infinite random number generator.
#   data_sampler = (np.random.random() * 100 for _ in itertools.count())

#   def assignment_fn(sample):
#     # Get rid of potential tqdm mess.
#     print()

#     label = input("Label of %.3f? > " % sample)
#     label = int(label.strip())
#     return label

#   active_loop(model, data_sampler, assignment_fn, opt_vars=[dist_means, dist_sd])
