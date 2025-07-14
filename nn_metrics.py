import numpy as np
import torch
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored
from .util import *


# ---------------------------------------------------------------
# Main loss and assessment functions
# ---------------------------------------------------------------


def neg_cox_log_likelihood(log_hazards, survival_event, survival_time, c_inds):
  """
  Calculate the average Cox negative partial log-likelihood to be minimized by PyTorch
  Input:
      log_hazards: linear predictors from trained model. n, num_conditions
      survival_event: survival event from ground truth: 1 for event and 0 for censored. n, num_conditions
      survival_time: survival time from ground truth (age). n, num_conditions
      c_inds (optional): specify subset of condition indexes
  """

  hazards = torch.exp(log_hazards)
  n, num_conditions = hazards.shape

  R_matrix = get_R_matrix(survival_time)  # n, n, c
  assert R_matrix.shape == (n, n, num_conditions)

  risk_set_sum = (R_matrix * hazards.unsqueeze(0)).sum(
    dim=1)  # sum along j/cols; summed risk of everyone later/eq than i/row
  assert risk_set_sum.shape == (n, num_conditions)

  diff = log_hazards - torch.log(risk_set_sum)  # (n, num_conditions)

  diff_observed = diff * survival_event  # (n, num_conditions)

  n_observed = survival_event.sum(dim=0)
  assert n_observed.shape == (num_conditions,)
  n_observed[n_observed == 0] = 1  # zero div
  loss = - (diff_observed.sum(dim=0) / n_observed)
  assert loss.shape == (num_conditions,)

  if c_inds == []:
    return loss.mean()  # average over all conditions
  else:
    loss = loss[c_inds]
    assert loss.shape == (len(c_inds),)
    return loss.mean()


def compute_c_index(hazards, survival_event, survival_time, compute_mean=True):
  """
  Computes concordance index given predictions and ground truth
  Input:
      hazards: exponent of linear predictors from trained model. n, num_conditions
      survival_event: survival event from ground truth: 1 for event and 0 for censored. n, num_conditions
      survival_time: survival time from ground truth (age). n, num_conditions
      compute_mean (optional): take the mean over conditions
  """

  survival_time_i = survival_time.unsqueeze(dim=1)
  survival_time_j = survival_time.unsqueeze(dim=0)
  j_surv_greater = (survival_time_j > survival_time_i).to(torch.float32)  # n, n, num_cond

  survival_event_i = survival_event.unsqueeze(dim=1)

  hazards_i = hazards.unsqueeze(dim=1)
  hazards_j = hazards.unsqueeze(dim=0)
  j_hazard_less = (hazards_j < hazards_i).to(torch.float32)
  j_jazard_eq = (hazards_j == hazards_i).to(torch.float32) * 0.5

  concord = (survival_event_i * j_surv_greater * (j_hazard_less + j_jazard_eq)).sum(dim=0).sum(dim=0)

  totals = (survival_event_i * j_surv_greater).sum(dim=0).sum(dim=0)  # n, n, num_cond
  assert totals.shape == (num_conditions,) and concord.shape == (num_conditions,)
  totals[totals == 0] = 1.  # zero div
  c_indexes = torch.divide(concord, totals)

  if compute_mean:
    return c_indexes.mean()
  else:
    return c_indexes


def get_R_matrix(survival_time):
  """
  Create an indicator matrix of risk sets, where T_j >= T_i.

  # Alternate
  batch_length = survival_time.shape[0]
  R_matrix = np.zeros([batch_length, batch_length], dtype=int)
  for i in range(batch_length):
    for j in range(batch_length):
      R_matrix[i, j] = survival_time[j] >= survival_time[i]
  return R_matrix
  """

  surv_i = survival_time.unsqueeze(dim=1)  # n, 1, num_conditions
  surv_j = survival_time.unsqueeze(dim=0)  # 1, n, num_conditions
  return (surv_j >= surv_i).to(torch.float32)  # whether person col ind age/event time is later/eq than person row ind


# ---------------------------------------------------------------
# Auxiliary functions
# ---------------------------------------------------------------


def compute_c_index_lib(hazards, survival_event, survival_time):
  raise NotImplementedError
  # Across disease, expects numpy
  # Use lifelines concordance_index
  # https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index
  assert hazards.shape[1] == num_conditions
  c_index = 0.
  for c in range(hazards.shape[1]):
    c_index_curr = concordance_index(survival_time[:, c], hazards[:, c], survival_event[:, c].astype(int))
    c_index += c_index_curr
  return c_index / hazards.shape[1]


def compute_c_index_sk(hazards, survival_event, survival_time):
  assert hazards.shape[1] == num_conditions
  assert np.logical_or((survival_event == 0), (survival_event == 1)).all()
  survival_event = survival_event.astype(bool)
  c_index = 0.
  for c in range(hazards.shape[1]):
    c_index_curr, _, _, _, _ = concordance_index_censored(survival_event[:, c], survival_time[:, c], hazards[:, c])
    c_index += c_index_curr
  return c_index / hazards.shape[1]


def compute_c_index_orig(hazards, survival_event, survival_time):
  # Across disease, expects numpy
  c_index = 0.
  assert hazards.shape[1] == num_conditions
  for c in range(hazards.shape[1]):
    concord = 0.
    total = 0.
    N_test = survival_event.shape[0]
    events = np.asarray(survival_event, dtype=bool)
    for i in range(N_test):
      if events[i, c] == 1:
        for j in range(N_test):

          if survival_time[j, c] > survival_time[i, c]:
            total = total + 1
            if hazards[j, c] < hazards[i, c]:
              concord = concord + 1
            elif hazards[j, c] == hazards[i, c]:
              concord = concord + 0.5
    c_index_curr = (concord / max(1, total))
    c_index += c_index_curr
  return c_index / hazards.shape[1]


def compute_logrank_p(hazards, survival_event, survival_time):
  # Per disease, expects numpy
  # Use lifelines logrank_test
  assert len(hazards.shape) == 1 and len(survival_time.shape) == 1 and len(survival_time.shape) == 1
  hazards_median = np.median(hazards)  # print('Median:', hazards_median)
  hazards_dichotomize = np.zeros([len(hazards)], dtype=int)
  hazards_dichotomize[hazards > hazards_median] = 1  # set low risk group as 0, high risk group as 1

  idx = hazards_dichotomize == 0
  T1 = survival_time[idx]  # low risk group
  T2 = survival_time[~idx]  # high risk group
  E1 = survival_event[idx]
  E2 = survival_event[~idx]
  results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
  return results.p_value


def compute_AUC(age, hazards, survival_event, survival_time, compute_mean=True):
  # Across disease, expects numpy
  # Ground truth is whether event occured before age
  # Use sklearn.metrics roc_auc_score
  # Uncensored_idx = survival_event == 1

  if compute_mean:
    survival_bin = survival_time < age  # event occured before specified age
    return roc_auc_score(survival_bin.flatten(), hazards.flatten())
  else:
    res = np.zeros(num_conditions) * np.nan
    survival_bin = survival_time < age
    for c in range(num_conditions):
      surv_c = survival_bin[:, c]
      if surv_c.sum() > 0:  # actually got disease before age
        res[c] = roc_auc_score(surv_c, hazards[:, c])  # do not care if censored or not
    return res


def get_R_matrix_orig(survival_time):
  batch_length = survival_time.shape[0]
  R_matrix = np.zeros([batch_length, batch_length], dtype=int)
  for i in range(batch_length):
    for j in range(batch_length):
      R_matrix[i, j] = survival_time[j] >= survival_time[i]
  return R_matrix


def neg_par_log_likelihood_orig(log_hazards, survival_event, survival_time):
  # Across disease, expects torch
  hazards = torch.exp(log_hazards)

  loss = 0.
  assert hazards.shape[1] == num_conditions
  for c in range(hazards.shape[1]):
    n_observed = survival_event[:, c].unsqueeze(dim=1).sum(0)
    # print(n_observed)
    R_matrix = get_R_matrix_orig(survival_time[:, c].unsqueeze(dim=1))
    R_matrix = torch.Tensor(R_matrix)
    risk_set_sum = R_matrix.mm(hazards[:, c].unsqueeze(dim=1))
    # print("risk_set_sum", risk_set_sum)
    diff = log_hazards[:, c].unsqueeze(dim=1) - torch.log(risk_set_sum)
    # print("diff", diff)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(survival_event[:, c].unsqueeze(dim=1))
    # print("sum_diff_in_observed", sum_diff_in_observed)
    loss += (- (sum_diff_in_observed) / n_observed).reshape((-1,))

  return loss / num_conditions


# ---------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------


if __name__ == "__main__":
  set_seed(1)

  print("Testing loss")
  survival_time = torch.Tensor([[500], [1200], [1200], [300]])
  survival_event = torch.Tensor([[1], [0], [0], [1]])
  pred = torch.Tensor([[0.5], [0.3], [0.3], [0.7]])
  print(survival_time.shape, survival_event.shape, pred.shape)

  R_matrix = get_R_matrix(survival_time)
  loss = neg_cox_log_likelihood(pred, survival_event, survival_time)
  print(loss)

  print("Higher loss")
  survival_time = torch.Tensor([[500], [1200], [1200], [300]])
  survival_event = torch.Tensor([[1], [0], [0], [1]])
  pred = torch.Tensor([[0.1], [0.3], [0.5], [0.05]])

  R_matrix = get_R_matrix(survival_time)
  loss = neg_cox_log_likelihood(pred, survival_event, survival_time)
  print(loss)

  print("Testing c_index")

  n = 4
  num_conditions = 3  # set global
  survival_time = torch.cat([survival_time, survival_time * 2, survival_time * 0.5], dim=1)
  survival_time[1, 1] = 300  # assymmetry for person who actually got it
  survival_event = torch.cat([survival_event, (1 - survival_event), survival_event], dim=1)
  assert survival_time.shape == (n, num_conditions) and survival_event.shape == (n, num_conditions)
  hazards = torch.rand(n, num_conditions)

  print(survival_time)
  print(survival_event)
  print(hazards)

  print("orig")
  print(compute_c_index_orig(hazards.numpy(), survival_event.numpy(), survival_time.numpy()))
  print("sk")
  print(compute_c_index_sk(hazards.numpy(), survival_event.numpy(), survival_time.numpy()))
  print("pytorch")
  print(compute_c_index(hazards, survival_event, survival_time))

  print("testing loss")
  log_hazards = torch.randn(n, num_conditions)
  print("orig")
  print(neg_par_log_likelihood_orig(log_hazards, survival_event, survival_time))
  print("pytorch")
  print(neg_cox_log_likelihood(log_hazards, survival_event, survival_time))
