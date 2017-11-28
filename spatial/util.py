from contextlib import contextmanager

import numpy as np
import pymc3 as pm
from pymc3.model import ObservedRV
from theano.tensor.sharedvar import TensorSharedVariable


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


def extract_shared(pm_observed_rv):
  """
  Extract the Theano shared variable from a PyMC3 ObservedRV.

  Returns `None` if the observed RV does not have a shared variable as
  data.
  """
  assert isinstance(pm_observed_rv, ObservedRV)

  if isinstance(pm_observed_rv.observations, TensorSharedVariable):
    return pm_observed_rv.observations

  try:
    ret = pm_observed_rv.observations.owner.inputs[0]
    assert isinstance(ret, TensorSharedVariable)
    return ret
  except:
    return None
