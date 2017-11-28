"""
Active learning routines.
"""

import numpy as np
import pymc3 as pm
from pymc3.model import ObservedRV
from pymc3.plots.kdeplot import fast_kde
from scipy import integrate
from scipy.stats import norm
from tqdm import trange

from spatial.util import temp_set, temp_append, extract_shared


class EIGPredictor(object):
  """
  Reusable EIG predictor applying to an instance of a sampled model.

  Caches quantities which don't change w.r.t. the EIG argument.
  """

  def __init__(self, model, k, trace, steps, assignment_var=None,
               query_vars=None, opt_vars=None):
    """
    Args:
      model: pymc3 model
      k: number of possible assignments
      orig_trace: model fit trace with which to initialize EIG fit
        calculations
      steps: model fit steps
      assignment_var: observed RV in the model which assigns instances to
        classes. By default, will look for an observed RV named
        "assignments."
      query_vars: input variables with which we are querying EIG. By
        default, all observed RVs of the model except for the
        assignment var.
      opt_vars: random variables whose information we want to gain! By
        default, all unobserved RVs of the model.
    """

    assert isinstance(model, pm.Model)

    self.model = model
    self.k = k
    self.orig_trace = trace
    self.steps = steps

    try:
      self.assignment_var = assignment_var or model.named_vars["assignments"]
    except KeyError:
      raise ValueError(
          "assignment_var not provided and model has no RV "
          "named 'assignments'")
    if not isinstance(self.assignment_var, ObservedRV):
      raise ValueError("assignment_var (name '%s') is not an ObservedRV"
                       % self.assignment_var.name)

    self.query_vars = list(filter(lambda x: x != self.assignment_var,
                                  query_vars or model.observed_RVs))
    self.opt_vars = opt_vars or model.free_RVs

    # Hack: grab shared variable from assignment var
    self.assignment_shared = extract_shared(self.assignment_var)
    if self.assignment_shared is None:
      raise ValueError("failed to extract shared assignments data. make "
          "sure you are using a shared variable in observations.")
    self._assignment_shared_size = len(self.assignment_shared.get_value())

    # Hack: grab shared variable from query var
    # Just supports one right now..
    self.query_shared = extract_shared(self.query_vars[0])
    if self.query_shared is None:
      raise ValueError("failed to extract shared value data.")

    self._ppcs = {}
    self._sample_ppc()
    self._entropy_pre = self._calculate_opt_var_entropy(self.orig_trace)

  def _sample_ppc(self):
    for assignment in range(self.k):
      with temp_set(self.assignment_shared, [assignment] * self._assignment_shared_size):
        # TODO: remove magic "points"
        self._ppcs[assignment] = pm.sample_ppc(self.orig_trace, samples=1000, vars=self.query_vars)["points"]

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

  def _calculate_opt_var_entropy(self, trace):
    total = 0
    for opt_var in self.opt_vars:
      trace_data = self.orig_trace[opt_var.name]

      if trace_data.ndim == 2:
        # var is not shared across assignments.
        assert trace_data.shape[1] == self.k
      else:
        # Fake a second axis to make the computation nice and uniform.
        trace_data = trace_data[:, np.newaxis]

      for assignment in range(trace_data.shape[1]):
        total += self._kde_entropy(trace_data[:, assignment])

    return total

  def eig(self, x):
    """
    Estimate the expected information gain of observing a new datum.

    Args:
      x: a possible assignment of `query_vars`
    """
    # First compute p(x) under current model's posterior predictive
    # (estimate with Gaussian)
    # TODO generalize to non-scalar input points.
    p_assignment = np.array([
      norm.pdf(x, loc=self._ppcs[idx].mean(), scale=self._ppcs[idx].std())
      for idx in range(self.k)])
    p_assignment /= p_assignment.sum()

    # Add data point.
    with temp_append(self.query_shared, x):
      assignment_kl = np.zeros(self.k)
      for assignment in trange(self.k, desc="Enumerating assignments"):
        with temp_append(self.assignment_shared, assignment):
          result = pm.sample(2000, step=self.steps, trace=self.orig_trace)
          # Drop first bit
          result = result[500:]

        # Get KDE of posterior over latents and estimate entropy.
        total_post_entropy = self._calculate_opt_var_entropy(result)

        assignment_kl[assignment] = total_post_entropy - self._entropy_pre

    return (p_assignment * assignment_kl).sum()

