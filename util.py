import json
import numpy as np
import os
import pyreadr
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from datetime import datetime
import random
import pyarrow.feather as feather
import gzip
import statsmodels.api as sm
import pandas as pd
import torch
import nibabel as nib
from matplotlib.pyplot import cm
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import seaborn as sns

from .consts import *


def set_seed(seed=1234):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)


def sprint(s):
  print(s)
  sys.stdout.flush()


def trim_landmark_index(ind):
  new_ind = ind - trim_len
  assert new_ind <= num_slices_raw - trim_len - 1
  return new_ind


def register_graphs(graph, midthigh, hip, iliopsoas, shoulder):
  n, nc, l = graph.shape
  assert nc == len(channels) and l == graph_len

  # Find the average section lengths for each sex
  diffs = []
  midthigh_avg = int(midthigh.mean())
  hip_avg = int(hip.mean())
  iliopsoas_avg = int(iliopsoas.mean())
  shoulder_avg = int(shoulder.mean())
  points = [0, midthigh_avg, hip_avg, iliopsoas_avg, shoulder_avg, graph_len]  # low to high excl

  for li in range(len(points) - 1):
    assert points[li] < points[li + 1]
    diffs.append(points[li + 1] - points[li])

  print("(register_graphs) points {}, diffs {}".format(points, diffs))
  assert sum(diffs) == graph_len

  # Register the graphs
  new_graph = []
  graphindex = np.arange(graph_len)
  for ni in range(n):
    points_i = [0, midthigh[ni], hip[ni], iliopsoas[ni], shoulder[ni], graph_len]

    curve_reg = [[] for _ in range(len(channels))]
    for section in range(len(points_i) - 1):
      reinds = np.linspace(start=points_i[section], stop=points_i[section + 1], num=diffs[section],
                           endpoint=False)
      for ci in range(len(channels)):
        curve_section = np.interp(reinds, xp=graphindex, fp=graph[ni, ci, :])
        curve_reg[ci].append(curve_section)

    for c in range(len(channels)):
      curve_reg[c] = np.concatenate(curve_reg[c], axis=0)
      assert curve_reg[c].shape == (graph_len,)

    # Zero supra hip for TMAT
    TMAT_channel = channels.index("TMAT")
    graph[ni, TMAT_channel, hip[ni]:] = 0
    curve_reg[TMAT_channel][points[2]:] = 0

    new_graph.append(np.stack(curve_reg, axis=0))

  new_graph = np.stack(new_graph, axis=0)
  assert new_graph.shape == (n, nc, l)
  return graph, new_graph, points


def get_outliers(landmark_inds):
  assert len(landmark_inds.shape) == 1
  if landmark_inds.shape[0] == 1:
    return np.array([False])  # no outliers if 1 element
  outliers = np.abs(landmark_inds - landmark_inds.mean()) >= 4 * landmark_inds.std()
  return outliers


def get_VAT_outliers(graph):
  assert channels.index("VAT") == 3
  graph_VAT = graph[:, channels.index("VAT"), :]

  fst_nonzero = get_first_nonzero(graph_VAT, axis=1, invalid_val=-1)
  lst_nonzero = get_last_nonzero(graph_VAT, axis=1, invalid_val=-1)

  assert fst_nonzero.shape == (graph_VAT.shape[0],) and lst_nonzero.shape == (graph_VAT.shape[0],)
  print("(get_VAT_outliers) nonzeros")
  print(fst_nonzero)
  print(lst_nonzero)
  print("(get_VAT_outliers) metrics")
  print(fst_nonzero.mean(), fst_nonzero.std(), fst_nonzero.min(), fst_nonzero.max())
  print(lst_nonzero.mean(), lst_nonzero.std(), lst_nonzero.min(), lst_nonzero.max())

  all_zeros = np.logical_or(fst_nonzero == -1, lst_nonzero == -1)
  fst_has_mag = fst_nonzero[np.logical_not(all_zeros)]
  lst_has_mag = lst_nonzero[np.logical_not(all_zeros)]

  print("(get_VAT_outliers) num all zeros being removed")
  print(all_zeros.sum())

  fst_outliers = np.abs(fst_nonzero - fst_has_mag.mean()) >= 4 * fst_has_mag.std()
  last_outliers = np.abs(lst_nonzero - lst_has_mag.mean()) >= 4 * lst_has_mag.std()
  assert fst_nonzero.shape == (graph_VAT.shape[0],) and lst_nonzero.shape == (
  graph_VAT.shape[0],) and all_zeros.shape == (graph_VAT.shape[0],)

  return np.logical_or(np.logical_or(fst_outliers, last_outliers), all_zeros)


def get_first_nonzero(arr, axis, invalid_val=-1):
  mask = arr != 0
  return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)  # argmax gets index of first occurence


def get_last_nonzero(arr, axis, invalid_val=-1):
  mask = arr != 0
  val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
  return np.where(mask.any(axis=axis), val, invalid_val)


def plot_nifti(nifti_fname, id_val, instance, orig_landmarks, landmark_colours, labels, suff):
  fig, axs = plt.subplots(1, figsize=(15, 10), facecolor='w')
  ip_img = np.asanyarray(nib.load(nifti_fname).dataobj)

  axs.imshow(np.rot90(ip_img[:, 80, :]), 'gray',
             aspect=250 / 174)  # Rotated middle view # removing trim as with indexing

  if orig_landmarks is not None:
    for li, v in enumerate(orig_landmarks):
      plt.axhline(y=(graph_len - v), color=landmark_colours[li], label="orig_{} ({})".format(labels[li], v),
                  linestyle="--")

    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.tight_layout()
  plt.savefig("{}/data/{}_{}_before_nifti{}.png".format(root_dir, id_val, instance, suff), bbox_inches='tight')
  plt.close('all')


def plot_registration(i, ids, instances, graph, midthigh, hip, iliopsoas, shoulder, new_graph, avg_landmarks,
                      flat_graphs, lengths, offsets):
  orig_landmarks = [midthigh[i], hip[i], iliopsoas[i], shoulder[i]]
  avg_landmarks = avg_landmarks[1:-1]
  landmark_colours = cm.rainbow(np.linspace(0, 1, len(avg_landmarks)))
  labels = ["midthigh", "hip", "iliopsoas", "shoulder"]

  # before nifti
  nifti_fname = os.path.join(raw_dir, "{}_{}/nifti/water.nii.gz".format(ids[i], instances[i]))
  plot_nifti(nifti_fname, ids[i], instances[i], orig_landmarks, landmark_colours, labels, "_water")
  for c_i, c in enumerate(channels):
    c_path = resource_paths[c].format("{}_{}".format(ids[i], instances[i]))
    plot_nifti(c_path, ids[i], instances[i], None, None, None, "_{}".format(c))

  # before graph
  fig, axs = plt.subplots(1, figsize=(15, 10), facecolor='w')
  for ci, c in enumerate(channels):
    axs.plot(graph[i, ci], np.arange(graph[i, ci].shape[0]), label=c)

  for li, v in enumerate(orig_landmarks):
    plt.axhline(y=v, color=landmark_colours[li], label="orig_{} ({})".format(labels[li], v), linestyle="--")

  box = axs.get_position()
  axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.tight_layout()
  plt.savefig("{}/data/{}_{}_before_graph.png".format(root_dir, ids[i], instances[i]), bbox_inches='tight')
  plt.close('all')

  # after graph
  fig, axs = plt.subplots(1, figsize=(15, 10), facecolor='w')
  for ci, c in enumerate(channels):
    axs.plot(new_graph[i, ci], np.arange(new_graph[i, ci].shape[0]), label=c)

  for li, v in enumerate(avg_landmarks):
    plt.axhline(y=v, color=landmark_colours[li], label="avg_{} ({})".format(labels[li], v), linestyle="--")

  box = axs.get_position()
  axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.tight_layout()
  plt.savefig("{}/data/{}_{}_after_graph.png".format(root_dir, ids[i], instances[i]), bbox_inches='tight')
  plt.close('all')

  # flat graphs
  fig, axs = plt.subplots(1, figsize=(15, 10), facecolor='w')
  flat_graph = flat_graphs[i]
  curr_ind = 0
  for ci, c in enumerate(channels):
    curr_end_excl = curr_ind + lengths[ci]
    axs.plot(flat_graph[curr_ind:curr_end_excl], offsets[ci] + np.arange(flat_graph[curr_ind:curr_end_excl].shape[0]),
             label="flat_{}".format(c))
    curr_ind = curr_end_excl

  axs.legend()
  plt.tight_layout()
  plt.savefig("{}/data/{}_{}_flat_graph.png".format(root_dir, ids[i], instances[i]), bbox_inches='tight')
  plt.close('all')


def flatten_format_graphs(graph, landmarks):
  n, nc, l = graph.shape
  assert nc == len(channels) and l == graph_len

  new_graph = []
  avg_hip = landmarks[2]
  offsets = []
  lengths = []
  for ci, c in enumerate(channels):
    graph_c = graph[:, ci, :]

    if c == "TMAT":
      graph_c = graph_c[:, :avg_hip]
      offsets.append(0)
    elif c == "VAT":
      """
      nonzero0, nonzero1 = np.nonzero(graph_c) # the id inds x the non zero inds for each id
      fst_nonzero = nonzero1.min()
      last_nonzero = nonzero1.max()
      print("(flatten_format_graphs) VAT",
            nonzero0.min(), nonzero0.max(), len(nonzero0),
            nonzero1.min(), nonzero1.max(), len(nonzero1),
            )
      """
      fst_nonzero = get_first_nonzero(graph_c, axis=1, invalid_val=-1)
      lst_nonzero = get_last_nonzero(graph_c, axis=1, invalid_val=-1)
      assert not ((fst_nonzero == -1).any() or (lst_nonzero == -1).any())  # all zeros have been removed
      assert fst_nonzero.shape == (n,)
      assert lst_nonzero.shape == (n,)
      fst_nonzero_val = fst_nonzero.min()
      lst_nonzero_val = lst_nonzero.max()

      print("(flatten_format_graphs) VAT")
      print(fst_nonzero_val, lst_nonzero_val)
      print(fst_nonzero.mean(), fst_nonzero.std(), fst_nonzero.min(), fst_nonzero.max())
      print(lst_nonzero.mean(), lst_nonzero.std(), lst_nonzero.min(), lst_nonzero.max())
      graph_c = graph_c[:, fst_nonzero_val:(lst_nonzero_val + 1)]
      offsets.append(fst_nonzero_val)
    else:
      offsets.append(0)

    new_graph.append(graph_c)
    lengths.append(graph_c.shape[1])

  new_graph = np.concatenate(new_graph, axis=1)
  lengths = np.array(lengths)
  print("(flatten_format_graphs) {}".format(new_graph.shape))
  assert new_graph.shape == (n, l * 3 + avg_hip + (lst_nonzero_val + 1 - fst_nonzero_val))
  print("(flatten_format_graphs) Offsets should be cumulative lengths")
  print(offsets)
  print(lengths)
  return new_graph, lengths, offsets


def compute_acc(thresh, all_scores, all_labels):
  n, = all_scores.shape
  assert all_labels.shape == (n,)
  pred = all_scores > thresh
  acc = (pred == all_labels).sum() / float(n)
  return acc


def print_full(x):
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 2000)
  pd.set_option('display.float_format', '{:20,.2f}'.format)
  pd.set_option('display.max_colwidth', None)
  print(x)
  pd.reset_option('display.max_rows')
  pd.reset_option('display.max_columns')
  pd.reset_option('display.width')
  pd.reset_option('display.float_format')
  pd.reset_option('display.max_colwidth')


def cross_validate(K, X_pos_train, cv, verbose=0):
  # covariance_type="diag"
  models = [GaussianMixture(n_components=K, random_state=i, init_params='kmeans', max_iter=100, verbose=verbose) for i
            in range(cv)]
  n, l = X_pos_train.shape

  new_order = np.arange(n)
  np.random.shuffle(new_order)
  X_pos_train = X_pos_train[new_order, :]

  chunk = int(np.floor(n / float(cv)))
  data = []

  for i in range(cv):
    start = i * chunk
    if i < cv - 1:
      end_excl = (i + 1) * chunk
    else:
      end_excl = n
    data.append(X_pos_train[start:end_excl, :])
  """
  for i in range(cv):
    start = i * chunk
    end_excl = min((i + 1) * chunk, n)
    data.append(X_pos_train[start:end_excl, :])
  """

  test_scores = []
  train_scores = []
  for i in range(cv):
    train = np.concatenate(data[:i] + data[(i + 1):], axis=0)
    test = data[i]
    print("train test cv {} {}".format(train.shape, test.shape))

    model = models[i].fit(train)
    # train_score = model.score(train)
    # test_score = model.score(test)

    train_score = model._estimate_log_prob(train).max(axis=1)
    assert train_score.shape == (train.shape[0],)
    train_score = train_score.mean()

    test_score = model._estimate_log_prob(test).max(axis=1)
    assert test_score.shape == (test.shape[0],)
    test_score = test_score.mean()

    train_prob = model.predict_proba(train)  # n, k
    test_prob = model.predict_proba(test)

    if verbose:
      print("(cv)", i, train_score, test_score,
            train.shape, test.shape,
            get_stats(train_prob),
            get_stats(test_prob),
            train_prob.shape, test_prob.shape,
            )
      print(prob_analysis(train_prob))
      print(prob_analysis(test_prob))
    test_scores.append(test_score)
    train_scores.append(train_score)

  return np.array(train_scores), np.array(test_scores)


def cross_validate(K, X_pos_train, cv, verbose=0):
  # covariance_type="diag"
  models = [GaussianMixture(n_components=K, random_state=i, init_params='kmeans', max_iter=100, verbose=verbose) for i
            in range(cv)]
  n, l = X_pos_train.shape

  new_order = np.arange(n)
  np.random.shuffle(new_order)
  X_pos_train = X_pos_train[new_order, :]

  chunk = int(np.floor(n / float(cv)))
  data = []

  for i in range(cv):
    start = i * chunk
    if i < cv - 1:
      end_excl = (i + 1) * chunk
    else:
      end_excl = n
    data.append(X_pos_train[start:end_excl, :])
  """
  for i in range(cv):
    start = i * chunk
    end_excl = min((i + 1) * chunk, n)
    data.append(X_pos_train[start:end_excl, :])
  """

  test_scores = []
  train_scores = []
  for i in range(cv):
    train = np.concatenate(data[:i] + data[(i + 1):], axis=0)
    test = data[i]
    print("train test cv {} {}".format(train.shape, test.shape))

    model = models[i].fit(train)
    train_score = model.score(train)
    test_score = model.score(test)

    train_prob = model.predict_proba(train)  # n, k
    test_prob = model.predict_proba(test)

    if verbose:
      print("(cv)", i, train_score, test_score,
            train.shape, test.shape,
            get_stats(train_prob),
            get_stats(test_prob),
            train_prob.shape, test_prob.shape,
            )
      print(prob_analysis(train_prob))
      print(prob_analysis(test_prob))
    test_scores.append(test_score)
    train_scores.append(train_score)

  return np.array(train_scores), np.array(test_scores)


def cross_validate_binary(K, X_pos, X_neg, cv, verbose=0):
  # covariance_type="diag"
  models = [GaussianMixture(n_components=K, random_state=i, init_params='kmeans', max_iter=100, verbose=verbose) for i
            in range(cv)]
  n, l = X_pos.shape

  new_order = np.arange(n)
  np.random.shuffle(new_order)
  X_pos = X_pos[new_order, :]

  chunk = int(np.floor(n / float(cv)))
  data = []

  for i in range(cv):
    start = i * chunk
    if i < cv - 1:
      end_excl = (i + 1) * chunk
    else:
      end_excl = n
    data.append(X_pos[start:end_excl, :])

  test_scores = []
  train_scores = []
  for i in range(cv):
    train = np.concatenate(data[:i] + data[(i + 1):], axis=0)
    test = data[i]
    print("train test cv {} {}".format(train.shape, test.shape))

    model = models[i].fit(train)
    train_score = model.score(train)
    test_score = model.score(test)

    train_prob = model.predict_proba(train)  # n, k
    test_prob = model.predict_proba(test)

    if verbose:
      print("(cv)", i, train_score, test_score,
            train.shape, test.shape,
            get_stats(train_prob),
            get_stats(test_prob),
            train_prob.shape, test_prob.shape,
            )
      print(prob_analysis(train_prob))
      print(prob_analysis(test_prob))
    test_scores.append(test_score)
    train_scores.append(train_score)

  return np.array(train_scores), np.array(test_scores)


def get_stats(x):
  return [x.mean(), x.std(), x.min(), x.max()]


def prob_analysis(probs):
  assert len(probs.shape) == 2
  top_probs = probs.max(axis=1)
  top_clusters = probs.argmax(axis=1)
  # print("Diff of sum cluster probs with 1:", get_stats(- 1))
  assert (np.abs(probs.sum(axis=1) - 1) <= 1e-10).all()
  return {"cluster assigned probs": get_stats(top_probs), "cluster counts": np.unique(top_clusters, return_counts=True)}


def plot_roc(all_labels_train, all_scores_train, best_thresh, suff):
  fpr, tpr, threshold = metrics.roc_curve(all_labels_train, all_scores_train)
  roc_auc = metrics.auc(fpr, tpr)

  tn, fp, fn, tp = metrics.confusion_matrix(all_labels_train, (all_scores_train >= best_thresh).astype(int)).ravel()
  fpr_best = fp / float(fp + tn)
  tpr_best = tp / float(tp + fn)

  print("(plot_roc) fpr_best {} tpr_best {}".format(fpr_best, tpr_best))
  plt.title('ROC {}\n best: {} {} ({})'.format(suff, fpr_best, tpr_best, best_thresh))
  plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
  plt.scatter(fpr_best, tpr_best, marker="*")
  plt.legend(loc='lower right')
  plt.plot([0, 1], [0, 1], 'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.tight_layout()
  plt.savefig(os.path.join(root_dir, "train_gmm/roc_{}.png".format(suff)), bbox_inches='tight')

  plt.close("all")


def plot_roc_direct(fpr, tpr, fpr_best, tpr_best, roc_auc, suff):
  plt.title('direct ROC {}\n best: {} {} ({})'.format(suff, fpr_best, tpr_best, roc_auc))

  plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
  plt.scatter(fpr_best, tpr_best, marker="*")
  plt.legend(loc='lower right')
  plt.plot([0, 1], [0, 1], 'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.tight_layout()
  plt.savefig(os.path.join(root_dir, "train_gmm/direct_roc_{}.png".format(suff)), bbox_inches='tight')

  plt.close("all")


def check_volumes(graph_data, ids, instances):
  lengths = graph_data["tissue_lengths"]
  graphs = graph_data["graphs"]
  flat_graphs = graph_data["flat_graphs"]
  assert lengths.shape == (5,)
  records_other_traits = feather.read_feather(os.path.join(root_dir, "data/records_other_traits.feather"))
  print(list(records_other_traits.columns.values))
  print(records_other_traits.shape)
  print(records_other_traits.describe())
  print("------")
  for i in range(10):
    id = ids[i]
    instance = instances[i]
    traits = records_other_traits[
      (records_other_traits["individual_id"] == id) & (records_other_traits["instance_id"] == instance)
      & (records_other_traits["array_id"] == 0)]
    print_full(traits)
    print(traits.shape)
    if traits.shape[0] == 1:
      t_musc = traits["MRI_total_muscle"].values[0]
      t_sat = traits["MRI_SAT"].values[0]
      t_vat = traits["MRI_VAT"].values[0]

      graph = graphs[i]
      assert graph.shape == (len(channels), graph_len)
      sat = graph[1].sum()
      musc = graph[2].sum()
      vat = graph[3].sum()
      print("Person {} {} instance {}:".format(i, id, instance))
      print("grph sat ml {} {} ({} {})".format((sat - t_sat) / t_sat, sat - t_sat, sat, t_sat))
      print("grph musc ml {} {} ({} {})".format((musc - t_musc) / t_musc, (musc - t_musc), musc, t_musc))
      print("grph vat ml {} {} ({} {})".format((vat - t_vat) / t_vat, (vat - t_vat), vat, t_vat))

      flat_graph = flat_graphs[i]
      assert len(flat_graph.shape) == 1
      sat = flat_graph[lengths[0]: lengths[:2].sum()].sum()
      musc = flat_graph[lengths[:2].sum(): lengths[:3].sum()].sum()
      vat = flat_graph[lengths[:3].sum(): lengths[:4].sum()].sum()
      print("flat sat ml {} {} ({} {})".format((sat - t_sat) / t_sat, sat - t_sat, sat, t_sat))
      print("flat musc ml {} {} ({} {})".format((musc - t_musc) / t_musc, (musc - t_musc), musc, t_musc))
      print("flat vat ml {} {} ({} {})".format((vat - t_vat) / t_vat, (vat - t_vat), vat, t_vat))
    print("---")

  exit(0)


def comp_normvals(arr):
  # always take mean, std along 0th dimension

  if len(arr.shape) == 1:
    means = np.mean(arr)
    stds = np.std(arr)
  elif len(arr.shape) == 2:
    means = np.mean(arr, axis=0, keepdims=True)
    assert means.shape == (1, arr.shape[1])
    stds = np.std(arr, axis=0, keepdims=True)
  elif len(arr.shape) == 3:  # mean and std per channel and location
    means = np.mean(arr, axis=0, keepdims=True)
    assert means.shape == (1, arr.shape[1], arr.shape[2])
    stds = np.std(arr, axis=0, keepdims=True)
  else:
    raise NotImplementedError

  return (means, stds)


def single_norm(arr, means, stds, eps=1e-10):
  if len(arr.shape) == 1:
    assert isinstance(means, np.float32) and isinstance(stds, np.float32)
  else:
    assert (means.shape == (1,) + arr.shape[1:]) and (stds.shape == means.shape)

  no_var = stds < eps
  num_no_var = no_var.sum()
  if num_no_var > 0:
    print("np_norm found variable with no var, adjusting std: {}".format(num_no_var))
    if len(arr.shape) == 1:
      stds = 1.
    else:
      stds[no_var] = 1.  # no variation

  return (arr - means) / stds


def np_norm(arr):
  means, stds = comp_normvals(arr)
  return single_norm(arr, means, stds)


def save_heatmap(mat, rownames, colnames, rowtitle, coltitle, valtitle, fname, fmt=None):
  assert len(mat.shape) == 2
  d = []  # list of row lists
  for ri in range(mat.shape[0]):
    for ci in range(mat.shape[1]):
      d.append([rownames[ri], colnames[ci], mat[ri, ci]])
  df = pd.DataFrame(data=d, columns=[rowtitle, coltitle, valtitle])

  sns.set(font_scale=3)

  fig, ax = plt.subplots(figsize=(80, 60))  # Sample figsize in inches

  df = df.pivot(index=rowtitle, columns=coltitle, values=valtitle)
  if fmt is not None:
    sns_plot = sns.heatmap(df, annot=False, fmt=fmt, ax=ax)
  else:
    sns_plot = sns.heatmap(df, annot=False, ax=ax)

  sns_plot.set_title(valtitle)  # fontsize=50
  sns_plot.figure.savefig(os.path.join(root_dir, fname))


def plot_clusters(avg_graph, graph2, title, filename, symmetric):
  fig, axs = plt.subplots(1, 1, figsize=(10, 10), facecolor='w')
  for c, channel in enumerate(channels):
    p = axs.plot(avg_graph[c], range(avg_graph[c].shape[0]), label=channel)  # horizontal values, vertical inds
    if symmetric:
      axs.fill_betweenx(range(avg_graph[c].shape[0]), avg_graph[c] - graph2[c],
                        avg_graph[c] + graph2[c], facecolor=p[-1].get_color(), alpha=0.3)
    else:
      axs.fill_betweenx(range(avg_graph[c].shape[0]), avg_graph[c],
                        graph2[c], facecolor=p[-1].get_color(), alpha=0.3)

  axs.set_ylabel('Height', fontsize=18)
  axs.set_xlabel('Value (L)', fontsize=18)
  axs.set_title(title)

  legend_ax = axs.inset_axes([.85, 0.1, 0.1, 0.8])  # Adjust the position and size as needed
  legend_ax.legend(*axs.get_legend_handles_labels(), loc='lower right', fontsize=14)
  legend_ax.axis('off')

  plt.tight_layout()
  plt.savefig(filename, bbox_inches='tight')
  plt.close('all')


def compute_one_hot(targets, nb_classes):
  res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
  return res.reshape(list(targets.shape) + [nb_classes])


def pretty(l):
  if l == []:
    return "None"

  elif l is None:
    return "None"

  elif isinstance(l, str):
    return "".join([ch if ch.isalnum() else "_" for ch in l])

  else:
    s = ""
    for c in l:
      s += "".join([ch if ch.isalnum() else "_" for ch in c])
    return s


def pretty_print(l):
  return l


def count_params(model):
  return sum(p.numel() for p in model.parameters())
