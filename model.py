from contextlib import contextmanager
import itertools

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from pymc3.plots.kdeplot import fast_kde, kdeplot
from pymc3.plots.artists import kdeplot_op
from scipy import integrate
from scipy.stats import norm
from theano import shared
from tqdm import tqdm, trange


@contextmanager
def temp_set(var, value):
  """
  Context manager which temporarily updates the value of a shared
  variable.

  Yields value of shared variable before temporary update is performed.
  """
  old_value = var.get_value()
  var.set_value(value)
  yield old_value
  var.set_value(old_value)


@contextmanager
def temp_append(var, x):
  """
  Context manager which temporarily appends a value to a shared array.
  """
  old_value = var.get_value()
  var.set_value(
      np.concatenate([old_value, [np.cast[old_value.dtype](x)]]))
  yield
  var.set_value(old_value)


class EIGPredictor(object):
  """
  Reusable EIG predictor applying to an instance of a sampled model.

  Caches quantities which don't change w.r.t. the EIG argument.
  """

  def __init__(self, model, k, trace, steps):
    self.model = model
    self.k = k
    self.orig_trace = trace
    self.steps = steps

    self._ppcs = {}
    self._sample_ppc()
    self._calculate_entropy_pre()

  def _sample_ppc(self):
    for assignment in range(k):
      with temp_set(d_assignments, [assignment] * len(d_assignments_0)):
        self._ppcs[assignment] = pm.sample_ppc(self.orig_trace, samples=1000, vars=[model["points"]])["points"]

  def _kde_entropy(self, sample):
    """
    Estimate entropy of a sample of a continuous RV by building a KDE.
    """
    kde_density, kde_min, kde_max = fast_kde(sample)
    # Convert to entropy.
    kde_density = kde_density * np.log(kde_density + 1e-6)

    # Integrate.
    xs = np.linspace(kde_min, kde_max, len(kde_density))
    kde_entropy = -integrate.simps(kde_density, xs)
    return kde_entropy

  def _calculate_entropy_pre(self):
    total_pre_entropy = 0
    for assignment in range(self.k):
      total_pre_entropy += self._kde_entropy(self.orig_trace["dist_means"][:, assignment])

    self._entropy_pre = total_pre_entropy

  def eig(self, x):
    """
    Estimate the expected information gain of observing a new datum.
    """
    # First compute p(x) under current model's posterior predictive
    # (estimate with Gaussian)
    p_assignment = np.array([
      norm.pdf(x, loc=self._ppcs[idx].mean(), scale=self._ppcs[idx].std())
      for idx in range(self.k)])
    p_assignment /= p_assignment.sum()

    # Add data point.
    with temp_append(d_points, x):
      assignment_kl = np.zeros(self.k)
      for assignment in trange(self.k, desc="Enumerating assignments"):
        with temp_append(d_assignments, assignment):
          result = pm.sample(2000, step=self.steps, trace=self.orig_trace)
          # Drop first bit
          result = result[500:]

        # Get KDE of posterior over latents and estimate entropy.
        total_post_entropy = 0
        for distr in range(self.k):
          total_post_entropy += self._kde_entropy(result["dist_means"][:, distr])

        assignment_kl[assignment] = total_post_entropy - self._entropy_pre

    return (p_assignment * assignment_kl).sum()


def active_loop(model, data_sampler, assignment_fn,
                batch_size=10):
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
    steps = [pm.Metropolis(vars=[dist_means, dist_sd, p])]

    i = 0
    while True:
      # Run a first MH inference.
      result = pm.sample(5000, step=steps)
      # Burn-in.
      result = result[1000:]

      # Plot result at this iter.
      fig, ax = plt.subplots()
      kdeplot_op(ax, result["dist_means"])
      fig.tight_layout()
      fig.savefig("active.%02i.png" % i)

      eig_predictor = EIGPredictor(model, k, result, steps)

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

types = ["near", "next to"]
# d_assignments_0 = [0, 0, 0, 0, 1, 1, 1, 1]
# d_points_0 = [100, 120, 140, 110, 30, 25, 19, 36]
d_assignments_0 = [0, 1]
d_points_0 = [100, 30]

d_assignments = shared(np.array(d_assignments_0, dtype=np.int32))
d_points = shared(np.array(d_points_0, dtype=np.float32))
n = len(d_points_0)#shared(np.array(d_points_0, dtype=np.int32))
k = len(types)


model = pm.Model()
with model:
  dist_means = pm.Normal("dist_means", mu=[100] * len(types), sd=50, shape=len(types))
  dist_sd = pm.Uniform("dist_sd", lower=0, upper=10)

  p = pm.Dirichlet("p", a=np.array([1., 1.]), shape=len(types))
  assignments = pm.Categorical("assignments", p=p, observed=d_assignments)

  # Likelihood for observed assignments.
  points = pm.Normal("points", mu=dist_means[d_assignments], sd=dist_sd,
             observed=d_points)

  # # Fit.
  # mh = pm.Metropolis(vars=[dist_means, dist_sd, p])
  # steps = [mh]
  # result = pm.sample(5000, step=steps)
  # result = result[1000:]

  # # #########

  # eig_predictor = EIGPredictor(model, k, result, steps)
  # xs = np.linspace(-20, 150, 50)
  # eigs = np.array([eig_predictor.eig(x) for x in xs])

  # from pprint import pprint
  # pprint(list(zip(xs, eigs)))

  # # ##########

  # fig, ax1 = plt.subplots()
  # ax1.set_ylabel("density")

  # # Plot KDE of dist_means
  # kdeplot_op(ax1, result["dist_means"])

  # # Plot EIG samples.
  # ax2 = ax1.twinx()
  # ax2.set_ylabel("EIG")
  # ax2.scatter(xs, eigs, c='r')

  # plt.tight_layout()
  # plt.savefig("after.png")

if __name__ == '__main__':
  # Infinite random number generator.
  data_sampler = (np.random.random() * 100 for _ in itertools.count())

  def assignment_fn(sample):
    # Get rid of potential tqdm mess.
    print()

    label = input("Label of %.3f? > " % sample)
    label = int(label.strip())
    return label

  active_loop(model, data_sampler, assignment_fn)
