import glob
from collections import defaultdict
import numpy as np, scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import traceback
import pyarrow.feather as feather
import torch
import argparse
from d3blocks import D3Blocks
import scipy

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import csv

import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx
import pylab

from ..nn_util import *
from .util import *

parser = argparse.ArgumentParser()
parser.add_argument('--run_types', type=str, nargs="+", default=[])
parser.add_argument('--fmodel', type=str, default="nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None")
parser.add_argument('--correction', default=False, action='store_true')
parser.add_argument('--corr_correction', default=False, action='store_true')
parser.add_argument('--remove_statins', default=False, action='store_true')
args = parser.parse_args()

print("Check dirs", root_dir, raw_dir)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Activating tex in all labels globally
plt.rc('text', usetex=True)
# Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': 14.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

fname = os.path.join(root_dir, "models", args.fmodel + ".pgz")
print(fname)

sns.set_theme(style='white') # white

with gzip.open(fname, "rb") as f:
  results = pickle.load(f)

# run all instance 2 data through model, get risk per disease
# store raw and quartiles in same order as nn_data dataset segments
model_args = results["args"]
int_trainvals, int_test, ext_test, conditions = get_nn_data(model_args, cond=True)

if "collect_data" in args.run_types:
  train = int_trainvals["train"][0] # all folds have same data
  val = int_trainvals["val"][0]
  model = results["best_model"] #.to(device)
  model.eval()

  sex_ind_start = DEMOG_fields.index(("sex", 2))
  assert sex_ind_start == 0

  log_hazards = []
  ids = []
  sex = []
  cond_observed = []
  cond_diag_age = []
  bbc_age, MASS, SAT, MUSC, VAT, TMAT = [], [], [], [], [], []
  data_segment = []
  for dl_i, dl in enumerate([train, val, int_test, ext_test]):
    for batch_idx, data in enumerate(dl):
      print("dl {} batch {}/{} {}".format(dl_i, batch_idx, len(dl), datetime.now()))
      sys.stdout.flush()
      #data = tuple([d.to(device) for d in data])
      ids_curr = data[nn_data_types.index("ids")]

      sex_curr = data[nn_data_types.index("demog")][:, sex_ind_start + 1] # 0 for male, 1 for female

      cond_observed_curr = data[nn_data_types.index("observed")] # n, conds
      cond_diag_age_curr = data[nn_data_types.index("diag_age")]

      # bbc and age
      bbc_age_curr = data[nn_data_types.index("age_imaging")].squeeze()
      MASS_curr = data[nn_data_types.index("bbc")][:, channels.index("MASS")]
      SAT_curr = data[nn_data_types.index("bbc")][:, channels.index("SAT")]
      MUSC_curr = data[nn_data_types.index("bbc")][:, channels.index("MUSC")]
      VAT_curr = data[nn_data_types.index("bbc")][:, channels.index("VAT")]
      TMAT_curr = data[nn_data_types.index("bbc")][:, channels.index("TMAT")]
      #print(bbc_age_curr.shape, MASS_curr.shape)
      assert bbc_age_curr.shape == MASS_curr.shape

      data_segment_curr = np.ones(ids_curr.shape[0]) * dl_i

      with torch.no_grad():
        log_hazards_curr = model("log_hazards", data)
      log_hazards.append(log_hazards_curr.detach().numpy())

      ids.append(ids_curr.squeeze().numpy().astype(int))
      sex.append(sex_curr.squeeze().numpy())
      cond_observed.append(cond_observed_curr.numpy())
      cond_diag_age.append(cond_diag_age_curr.numpy())
      bbc_age.append(bbc_age_curr)
      MASS.append(MASS_curr)
      SAT.append(SAT_curr)
      MUSC.append(MUSC_curr)
      VAT.append(VAT_curr)
      TMAT.append(TMAT_curr)
      data_segment.append(data_segment_curr)

  ids = np.concatenate(ids)
  log_hazards = np.concatenate(log_hazards, axis=0) # n, num_cond
  sex = np.concatenate(sex)
  cond_observed = np.concatenate(cond_observed, axis=0)
  cond_diag_age = np.concatenate(cond_diag_age, axis=0)
  bbc_age = np.concatenate(bbc_age)
  MASS = np.concatenate(MASS)
  SAT = np.concatenate(SAT)
  MUSC = np.concatenate(MUSC)
  VAT = np.concatenate(VAT)
  TMAT = np.concatenate(TMAT)
  data_segment = np.concatenate(data_segment)

  sex_is, sex_c = np.unique(sex)
  print("check sex info", sex_is, sex_c, data[nn_data_types.index("demog")].shape)

  # one column per condition containing factor with quartile
  collected_data = {"individual_id": ids}
  conditions_list = []
  print_qbin = True
  if print_qbin: print("------------- qbins ------------- ")

  for ci, c in enumerate(conditions):
    conditions_list.append(c[0])

    log_hazards_c = log_hazards[:, ci]
    hazards_q = np.ones(ids.shape[0], dtype=int) * np.nan
    qbin_tup = []
    for s in range(2):
      sex_i = sex == s
      mask_curr = np.logical_and(sex_i, data_segment <= 1) # sex specific quartiles based on training data. (Same as before reindexing)
      _, qbins = pd.qcut(log_hazards_c[mask_curr], q=4, labels=False, retbins=True) #log_hazards_c[sex_i]
      assert qbins.shape == (5,)
      qbin_tup = qbin_tup + qbins[1:-1].tolist()

      quartiles_curr = np.ones(log_hazards_c[sex_i].shape, dtype=int) * np.nan # 0 indexed
      for qi in range(4):
        quartiles_curr[np.logical_and(log_hazards_c[sex_i] >= qbins[qi], log_hazards_c[sex_i] < qbins[qi + 1])] = qi
      quartiles_curr[log_hazards_c[sex_i] < qbins[0]] = 0 # before first training point
      quartiles_curr[log_hazards_c[sex_i] >= qbins[4]] = 3 # after last training point

      assert np.isfinite(quartiles_curr).all() and (quartiles_curr <= 3).all() and (quartiles_curr >= 0).all()
      hazards_q[sex_i] =  quartiles_curr

    if print_qbin:
      print("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(c[0], *tuple(qbin_tup)))

    assert np.isfinite(hazards_q).all()

    collected_data["{} log hazard".format(c[0])] = log_hazards_c
    collected_data["{} log hazard q".format(c[0])] = hazards_q

    collected_data["{} cond_observed".format(c[0])] = cond_observed[:, ci]
    collected_data["{} cond_diag_age".format(c[0])] = cond_diag_age[:, ci]

  if print_qbin: print("-------------------------- ")

  collected_data["bbc_age"] = bbc_age
  collected_data["MASS"] = MASS
  collected_data["SAT"] = SAT
  collected_data["MUSC"] = MUSC
  collected_data["VAT"] = VAT
  collected_data["TMAT"] = TMAT
  collected_data["data_segment"] = data_segment

  for k, v in collected_data.items():
    print(k, v[:5], v.shape)
    assert(len(v.shape) == 1)

  collected_data_df = pd.DataFrame(collected_data)
  print(collected_data_df.head())
  ids_ordered_df = feather.read_feather(os.path.join(root_dir, "data/ids_main.feather")) # numpy array
  # left join, count no na, save
  collected_data_df = pd.merge(ids_ordered_df, collected_data_df, how="left", on=["individual_id"])
  assert collected_data_df["individual_id"].equals(ids_ordered_df["individual_id"]) and not collected_data_df.isnull().values.any()

  # save as R accessible feather file
  feather.write_feather(collected_data_df, os.path.join(root_dir, "analysis/collected_data_{}.feather".format(args.fmodel)))

  print("Check conditions order same")
  print(len(conditions_list))
  print(conditions_list)
  conditions_df = pd.DataFrame({"condition_group": conditions_list})
  print(conditions_df.head())
  feather.write_feather(conditions_df, os.path.join(root_dir, "analysis/conditions_ordered_{}.feather".format(args.fmodel)))

# ---------------------------------------------------------------
# adjusted and adjusted+ hazard ratio of q4 ci1 for ci2 disease
# ---------------------------------------------------------------

def get_pthresh(quartiles_suff, model_pref, correct, base=0.05):
  if not correct:
    return base
  counts_file = feather.read_feather(
    os.path.join(root_dir, "analysis/uncensored_counts_{}_{}.feather".format(model_pref, quartiles_suff)))
  counts_uncensored = counts_file["counts"].to_numpy()
  num_comparisons = 0
  for ci1_ord, cname1 in enumerate(conds_ordered):  # row
    ci1 = conditions_df.index(cname1)
    for ci2_ord, cname2 in enumerate(conds_ordered):  # col
      ci2 = conditions_df.index(cname2)
      if counts_uncensored[ci2] < 50:
        continue
      # R has 1+ indexing
      res1 = feather.read_feather(
        os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res1_{}.feather".format(model_pref, ci1 + 1, ci2 + 1,
                                                                                     quartiles_suff)))
      num_comparisons += res1.shape[0]

  pthresh = base / num_comparisons
  print("Num comparisons and pthresh", quartiles_suff, model_pref, num_comparisons, pthresh)
  return pthresh


if "hazards" in args.run_types:

  conds_ordered = []
  for cgroup in cgroups_ord: # excludes Average
    for ci1, c1 in enumerate(conditions):
      cname1 = c1[0]

      assert not cname1 == "Average"
      if condition_supergroups[cname1] == cgroup:
        conds_ordered.append(cname1)
  print(conds_ordered)

  conditions_df = feather.read_feather(os.path.join(root_dir, "analysis/conditions_ordered_{}.feather".format(args.fmodel)))
  conditions_df = list(conditions_df["condition_group"].to_numpy())
  print("R conditions ordered")
  print(conditions_df)

  num_sch_fail, num_sch_pass = 0, 0
  for quartiles_suff in ["q", "h", "continuous"]: #
    discrete_type = {"q": "quartile", "h": "binary", "continuous": "continuous"}[quartiles_suff]

    for model_pref in ["basicplus"]:
      pthresh = get_pthresh(quartiles_suff, model_pref, correct=args.correction)

      print("Doing setting", quartiles_suff, model_pref)
      counts_file = feather.read_feather(os.path.join(root_dir, "analysis/uncensored_counts_{}_{}.feather".format(model_pref, quartiles_suff)))
      counts_uncensored = counts_file["counts"].to_numpy()
      print("Counts:")
      print(counts_uncensored)

      hazards = [] # row is from, col is to
      flat_conds = []
      cgroups = []
      full_significant = []
      pval_pass_both = 0
      all_pass_hazard_all, all_pass_hazard = [], []
      acmort_hazards, acmort_hazards_strong, acmort_hazards_sch, self_hazard, self_hazard_strong = [], [], [], [], []
      for ci1_ord, cname1 in enumerate(conds_ordered): # row
        ci1 = conditions_df.index(cname1)

        hazards.append([])
        full_significant.append([])
        flat_conds.append(cname1)
        cgroups.append(condition_supergroups[cname1])
        for ci2_ord, cname2 in enumerate(conds_ordered): # col
          ci2 = conditions_df.index(cname2)

          if counts_uncensored[ci2] < 50:
            print("Skipping", cname2)
            hazards[-1].append(np.nan)
            full_significant[-1].append(0)
            continue

          # R has 1+ indexing
          res1 = feather.read_feather(
            os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res1_{}.feather".format(model_pref, ci1 + 1, ci2 + 1, quartiles_suff)))

          if quartiles_suff == "q":
            log_hr = res1[res1["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\")3"]["estimate"].squeeze()
            pval = res1[res1["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\")3"]["p.value"].squeeze()
            sch_name = "relevel(as.factor(log_hazard_{}), ref = \"0\")".format(quartiles_suff)
          elif quartiles_suff == "h":
            log_hr = res1[res1["term"] == "relevel(as.factor(log_hazard_h), ref = \"0\")1"]["estimate"].squeeze()
            pval = res1[res1["term"] == "relevel(as.factor(log_hazard_h), ref = \"0\")1"]["p.value"].squeeze()
            sch_name = "relevel(as.factor(log_hazard_{}), ref = \"0\")".format(quartiles_suff)
          elif quartiles_suff == "continuous":
            log_hr = res1[res1["term"] == "scale(log_hazard)"]["estimate"].squeeze()
            pval = res1[res1["term"] == "scale(log_hazard)"]["p.value"].squeeze()
            sch_name = "scale(log_hazard)"

          sch_pass = None
          if pval <= pthresh:
            all_pass_hazard_all.append(np.exp(log_hr))
            if cname2 == "All cause mortality":
              acmort_hazards.append(np.exp(log_hr))

            res2 = feather.read_feather(os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res2_{}.feather".format(model_pref, ci1 + 1, ci2 + 1, quartiles_suff)))
            res3 = os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res3_{}.feather".format(model_pref, ci1 + 1, ci2 + 1, quartiles_suff))
            if os.path.exists(res3):
              schf = feather.read_feather(res3)
              sch = schf[schf["term_name"] == sch_name]["p"]
              assert sch.shape == (1,)
              sch = sch.squeeze()
              sch_pass = sch > pthresh
              if sch_pass:
                num_sch_pass += 1
                all_pass_hazard.append(np.exp(log_hr))

                if cname1 == cname2:
                  self_hazard.append(np.exp(log_hr))
                  if np.exp(log_hr) >= 3.:
                    self_hazard_strong.append(np.exp(log_hr))

                if cname2 == "All cause mortality":
                  acmort_hazards_sch.append(np.exp(log_hr))
                  if np.exp(log_hr) >= 3.:
                    acmort_hazards_strong.append(np.exp(log_hr))

              else:
                num_sch_fail += 1

            else:
              sch_pass = False
              num_sch_fail += 1

            if sch_pass:
              hazards[-1].append(log_hr)
            else:
              hazards[-1].append(np.nan)

          else:
            hazards[-1].append(np.nan)
          #print("Result for {} - {}: {} {} {} {}".format(cname1, cname2, pval, sch_pass, counts_uncensored[ci2], log_hr))

          # get significance after full adjustment
          r2 = feather.read_feather(
            os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res1_{}.feather".format("full", ci1 + 1, ci2 + 1, quartiles_suff)))
          if quartiles_suff == "q":
            pval2 = r2[r2["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\")3"]["p.value"].squeeze()
          elif quartiles_suff == "h":
            pval2 = r2[r2["term"] == "relevel(as.factor(log_hazard_h), ref = \"0\")1"]["p.value"].squeeze()
          elif quartiles_suff == "continuous":
            pval2 = r2[r2["term"] == "scale(log_hazard)"]["p.value"].squeeze()

          if pval2 <= pthresh:
            full_significant[-1].append(1)
            if (pval <= pthresh):
              pval_pass_both += 1
          else:
            full_significant[-1].append(0)

          if cname2 == "All cause mortality":
            assert os.path.exists(res3)
            print("ACMort hazards: {}, log hr {}, pval {} ({}), pval2 {}, sch {}".format(cname1, log_hr, pval, pval <= pthresh, pval2, sch_pass))

      print("-----")
      print("pval_pass_both", quartiles_suff, pval_pass_both)
      print("-----")
      acmort_hazards = np.array(acmort_hazards)
      acmort_hazards_strong = np.array(acmort_hazards_strong)
      acmort_hazards_sch = np.array(acmort_hazards_sch)
      self_hazard = np.array(self_hazard)
      self_hazard_strong = np.array(self_hazard_strong)
      all_pass_hazard = np.array(all_pass_hazard)
      all_pass_hazard_all = np.array(all_pass_hazard_all)
      print("All hazards stats pass:", quartiles_suff, all_pass_hazard_all.shape, all_pass_hazard_all.mean(), all_pass_hazard_all.std(), all_pass_hazard_all.min(), all_pass_hazard_all.max())
      print("All hazards stats pass sch:", quartiles_suff, all_pass_hazard.shape, all_pass_hazard.mean(), all_pass_hazard.std(), all_pass_hazard.min(), all_pass_hazard.max())
      print("-----")
      print("All cause mortality hazard stats pass:", quartiles_suff, acmort_hazards.shape, acmort_hazards.mean(), acmort_hazards.std(), acmort_hazards.min(), acmort_hazards.max())
      print("All cause mortality hazard stats pass sch:", quartiles_suff, acmort_hazards_sch.shape, acmort_hazards_sch.mean(), acmort_hazards_sch.std(), acmort_hazards_sch.min(), acmort_hazards_sch.max())
      print("All cause mortality hazard stats pass sch strong:", quartiles_suff, acmort_hazards_strong.shape, acmort_hazards_strong.mean(), acmort_hazards_strong.std(), acmort_hazards_strong.min(), acmort_hazards_strong.max())
      print("-----")
      print("Self hazards stats pass sch:", quartiles_suff, self_hazard.shape, self_hazard.mean(), self_hazard.std(), self_hazard.min(), self_hazard.max())
      print("Self hazards stats pass sch strong:", quartiles_suff, self_hazard_strong.shape, self_hazard_strong.mean(), self_hazard_strong.std(), self_hazard_strong.min(), self_hazard_strong.max())
      print("-----")


      hazards_np = np.array(hazards)
      print("hazards and mask stats")
      print(get_stats(hazards_np))
      full_significant_np = np.array(full_significant)
      print(get_stats(full_significant_np))
      full_significant_np = full_significant_np.astype(bool)
      print("num_sch pass {} fail {}".format(num_sch_pass, num_sch_fail))

      full_significant_both = np.logical_and(full_significant_np, np.isfinite(hazards_np))
      print("Number of significant {}, still significant after full adjustment {} ".format(np.isfinite(hazards_np).sum(), full_significant_both.sum()))

      row_names = flat_conds #["{} (in)".format(f) for f in flat_conds]
      col_names = flat_conds #["{} (out)".format(f) for f in flat_conds]
      hazards_df = pd.DataFrame(hazards_np, index=row_names, columns=col_names) # hazards

      print(hazards_df.head())
      print(hazards_df.shape)
      print(hazards_df)

      # show insignificant as grey, annotate with different df containing strings with significance
      sns.set_theme(rc={'figure.figsize': (18, 18)})
      sns.set(font_scale=0.8) # 0.7
      #sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
      #                             "ytick.labelsize": 14})

      cmap = sns.color_palette("vlag", as_cmap=True)
      cmap.set_bad("lightgray")

      data_vals = []
      for hazard, full_sig in zip(hazards_np.ravel(), full_significant_np.ravel()):
        if full_sig:
          data_vals.append(r'\underline{' + "{:.2f}".format(hazard) + '}')
        else:
          data_vals.append("{:.2f}".format(hazard))

      hazard_plot = sns.heatmap(data=hazards_df, center=0, linewidth=0.15,
                                cmap=cmap,
                                annot=np.array(data_vals).reshape(np.shape(hazards_np)),
                                fmt='',
                                #cbar =False,
                                square=True,
                                cbar_kws=dict(use_gridspec=False,location="right",pad=0.02,shrink=0.8))

      # draw dashed lines
      curr_cond = 0
      for cgi, cg in enumerate(cgroups_ord): # the order the condition groups are printed, excluding Average
        if cgi == len(cgroups_ord) - 1:
          break

        curr_cond += cg_counts[cg]
        line_ind = curr_cond #- 0.5
        hazard_plot.axes.axvline(line_ind, color="dimgray", linestyle="--") # end of this one #  linewidth=0.15
        hazard_plot.axes.axhline(line_ind, color="dimgray", linestyle="--") # end of this one # linewidth=0.15

      #plt.tight_layout()
      # todo
      hazard_plot.set_xticklabels(hazard_plot.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
      hazard_plot.set_yticklabels(hazard_plot.get_yticklabels(), rotation=(-30), ha="right", rotation_mode='anchor')

      hazard_plot.set_xlim(0, sum(list(cg_counts.values())) - 1) # also sets axis ordering
      hazard_plot.set_ylim(0, sum(list(cg_counts.values())) - 1)
      hazard_plot.set(xlabel="Event (output)", ylabel="OnsetNet score {} (input)".format(discrete_type))

      plt.subplots_adjust(left=0.2)
      hazard_plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_{}_hazards_{}_{}.png".format(model_pref, quartiles_suff, args.correction)))
      hazard_plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_{}_hazards_{}_{}.eps".format(model_pref, quartiles_suff, args.correction)), format="eps")

      plt.clf()
      plt.close('all')


if "hazards_condensed" in args.run_types:
  conds_ordered = []
  for cgroup in cgroups_ord:  # excludes Average
    for ci1, c1 in enumerate(conditions):
      cname1 = c1[0]

      assert not cname1 == "Average"
      if condition_supergroups[cname1] == cgroup:
        conds_ordered.append(cname1)
  print(conds_ordered)

  conditions_df = feather.read_feather(
    os.path.join(root_dir, "analysis/conditions_ordered_{}.feather".format(args.fmodel)))
  conditions_df = list(conditions_df["condition_group"].to_numpy())
  print("R conditions ordered")
  print(conditions_df)

  num_sch_fail, num_sch_pass = 0, 0
  for quartiles_suff in ["q", "h"]:  # continuous
    discrete_type = {"q": "quartile", "h": "binary", "continuous": "continuous"}[quartiles_suff]

    for model_pref in ["basicplus"]:
      pthresh = get_pthresh(quartiles_suff, model_pref, correct=args.correction)

      print("Doing setting", quartiles_suff, model_pref)
      counts_file = feather.read_feather(
        os.path.join(root_dir, "analysis/uncensored_counts_{}_{}.feather".format(model_pref, quartiles_suff)))
      counts_uncensored = counts_file["counts"].to_numpy()
      print("Counts:")
      print(counts_uncensored)

      hazards = []  # row is from, col is to
      flat_conds = []
      cgroups = []
      full_significant = []
      pval_pass_both = 0
      all_pass_hazard_all, all_pass_hazard = [], []
      acmort_hazards, acmort_hazards_strong, acmort_hazards_sch, self_hazard, self_hazard_strong = [], [], [], [], []
      for ci1_ord, cname1 in enumerate(conds_ordered):  # row
        ci1 = conditions_df.index(cname1)

        hazards.append([])
        full_significant.append([])
        flat_conds.append(cname1)
        cgroups.append(condition_supergroups[cname1])
        for ci2_ord, cname2 in enumerate(conds_ordered):  # col
          ci2 = conditions_df.index(cname2)

          if counts_uncensored[ci2] < 50:
            print("Skipping", cname2)
            hazards[-1].append(np.nan)
            full_significant[-1].append(0)
            continue

          # R has 1+ indexing
          res1 = feather.read_feather(
            os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res1_{}.feather".format(model_pref, ci1 + 1, ci2 + 1,
                                                                                         quartiles_suff)))

          if quartiles_suff == "q":
            log_hr = res1[res1["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\")3"]["estimate"].squeeze()
            pval = res1[res1["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\")3"]["p.value"].squeeze()
            sch_name = "relevel(as.factor(log_hazard_{}), ref = \"0\")".format(quartiles_suff)
          elif quartiles_suff == "h":
            log_hr = res1[res1["term"] == "relevel(as.factor(log_hazard_h), ref = \"0\")1"]["estimate"].squeeze()
            pval = res1[res1["term"] == "relevel(as.factor(log_hazard_h), ref = \"0\")1"]["p.value"].squeeze()
            sch_name = "relevel(as.factor(log_hazard_{}), ref = \"0\")".format(quartiles_suff)
          elif quartiles_suff == "continuous":
            log_hr = res1[res1["term"] == "scale(log_hazard)"]["estimate"].squeeze()
            pval = res1[res1["term"] == "scale(log_hazard)"]["p.value"].squeeze()
            sch_name = "scale(log_hazard)"

          sch_pass = None
          if pval <= pthresh:
            all_pass_hazard_all.append(np.exp(log_hr))
            if cname2 == "All cause mortality":
              acmort_hazards.append(np.exp(log_hr))

            res2 = feather.read_feather(os.path.join(root_dir,
                                                     "analysis/linear_cox/{}_{}_{}_res2_{}.feather".format(model_pref,
                                                                                                           ci1 + 1,
                                                                                                           ci2 + 1,
                                                                                                           quartiles_suff)))
            res3 = os.path.join(root_dir,
                                "analysis/linear_cox/{}_{}_{}_res3_{}.feather".format(model_pref, ci1 + 1, ci2 + 1,
                                                                                      quartiles_suff))
            if os.path.exists(res3):
              schf = feather.read_feather(res3)
              sch = schf[schf["term_name"] == sch_name]["p"]
              assert sch.shape == (1,)
              sch = sch.squeeze()
              sch_pass = sch > pthresh
              if sch_pass:
                num_sch_pass += 1
                all_pass_hazard.append(np.exp(log_hr))

                if cname1 == cname2:
                  self_hazard.append(np.exp(log_hr))
                  if np.exp(log_hr) >= 3.:
                    self_hazard_strong.append(np.exp(log_hr))

                if cname2 == "All cause mortality":
                  acmort_hazards_sch.append(np.exp(log_hr))
                  if np.exp(log_hr) >= 3.:
                    acmort_hazards_strong.append(np.exp(log_hr))

              else:
                num_sch_fail += 1

            else:
              sch_pass = False
              num_sch_fail += 1

            if sch_pass:
              hazards[-1].append(log_hr)
            else:
              hazards[-1].append(np.nan)

          else:
            hazards[-1].append(np.nan)
          # print("Result for {} - {}: {} {} {} {}".format(cname1, cname2, pval, sch_pass, counts_uncensored[ci2], log_hr))

          # get significance after full adjustment
          r2 = feather.read_feather(
            os.path.join(root_dir, "analysis/linear_cox/{}_{}_{}_res1_{}.feather".format("full", ci1 + 1, ci2 + 1,
                                                                                         quartiles_suff)))
          if quartiles_suff == "q":
            pval2 = r2[r2["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\")3"]["p.value"].squeeze()
          elif quartiles_suff == "h":
            pval2 = r2[r2["term"] == "relevel(as.factor(log_hazard_h), ref = \"0\")1"]["p.value"].squeeze()
          elif quartiles_suff == "continuous":
            pval2 = r2[r2["term"] == "scale(log_hazard)"]["p.value"].squeeze()

          if pval2 <= pthresh:
            full_significant[-1].append(1)
            if (pval <= pthresh):
              pval_pass_both += 1
          else:
            full_significant[-1].append(0)

          if cname2 == "All cause mortality":
            assert os.path.exists(res3)
            print("ACMort hazards: {}, log hr {}, pval {} ({}), pval2 {}, sch {}".format(cname1, log_hr, pval,
                                                                                         pval <= pthresh, pval2,
                                                                                         sch_pass))

      print("-----")
      print("pval_pass_both", quartiles_suff, pval_pass_both)
      print("-----")
      acmort_hazards = np.array(acmort_hazards)
      acmort_hazards_strong = np.array(acmort_hazards_strong)
      acmort_hazards_sch = np.array(acmort_hazards_sch)
      self_hazard = np.array(self_hazard)
      self_hazard_strong = np.array(self_hazard_strong)
      all_pass_hazard = np.array(all_pass_hazard)
      all_pass_hazard_all = np.array(all_pass_hazard_all)
      print("All hazards stats pass:", quartiles_suff, all_pass_hazard_all.shape, all_pass_hazard_all.mean(),
            all_pass_hazard_all.std(), all_pass_hazard_all.min(), all_pass_hazard_all.max())
      print("All hazards stats pass sch:", quartiles_suff, all_pass_hazard.shape, all_pass_hazard.mean(),
            all_pass_hazard.std(), all_pass_hazard.min(), all_pass_hazard.max())
      print("-----")
      print("All cause mortality hazard stats pass:", quartiles_suff, acmort_hazards.shape, acmort_hazards.mean(),
            acmort_hazards.std(), acmort_hazards.min(), acmort_hazards.max())
      print("All cause mortality hazard stats pass sch:", quartiles_suff, acmort_hazards_sch.shape,
            acmort_hazards_sch.mean(), acmort_hazards_sch.std(), acmort_hazards_sch.min(), acmort_hazards_sch.max())
      print("All cause mortality hazard stats pass sch strong:", quartiles_suff, acmort_hazards_strong.shape,
            acmort_hazards_strong.mean(), acmort_hazards_strong.std(), acmort_hazards_strong.min(),
            acmort_hazards_strong.max())
      print("-----")
      print("Self hazards stats pass sch:", quartiles_suff, self_hazard.shape, self_hazard.mean(), self_hazard.std(),
            self_hazard.min(), self_hazard.max())
      print("Self hazards stats pass sch strong:", quartiles_suff, self_hazard_strong.shape, self_hazard_strong.mean(),
            self_hazard_strong.std(), self_hazard_strong.min(), self_hazard_strong.max())
      print("-----")

      hazards_np = np.array(hazards)
      print("hazards and mask stats")
      print(get_stats(hazards_np))
      full_significant_np = np.array(full_significant)
      print(get_stats(full_significant_np))
      full_significant_np = full_significant_np.astype(bool)
      print("num_sch pass {} fail {}".format(num_sch_pass, num_sch_fail))

      full_significant_both = np.logical_and(full_significant_np, np.isfinite(hazards_np))
      print(
        "Number of significant {}, still significant after full adjustment {} ".format(np.isfinite(hazards_np).sum(),
                                                                                       full_significant_both.sum()))

      row_names = flat_conds  # ["{} (in)".format(f) for f in flat_conds]
      col_names = flat_conds  # ["{} (out)".format(f) for f in flat_conds]

      # Predictors on x axis, outcomes on y axis. Outcomes without significant HRs ommitted
      hazards_np = np.transpose(hazards_np)
      full_significant_np = np.transpose(full_significant_np)
      all_sig = np.logical_not(np.logical_not(np.isfinite(hazards_np)).astype(int).sum(axis=1) == hazards_np.shape[1]) # collapse cols
      sig_conds = []
      for cond_i, sig in enumerate(all_sig):
        if sig:
          sig_conds.append(row_names[cond_i])
      print("all sig:", all_sig, sig_conds, len(all_sig), len(sig_conds))
      print("Prev shape:", hazards_np.shape, full_significant_np.shape)
      hazards_np = hazards_np[all_sig, :]
      full_significant_np = full_significant_np[all_sig, :]
      row_names = sig_conds
      print("New shape:", hazards_np.shape, full_significant_np.shape)

      hazards_df = pd.DataFrame(hazards_np, index=row_names, columns=col_names)  # hazards

      print(hazards_df.head())
      print(hazards_df.shape)
      print(hazards_df)

      # show insignificant as grey, annotate with different df containing strings with significance
      sns.set_theme(rc={'figure.figsize': (19, 20)})
      #sns.set(font_scale=0.8)  # 0.7
      sns.set_context("paper", rc={"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 14,
                                   "xtick.labelsize": 14, "ytick.labelsize": 14})

      cmap = sns.color_palette("vlag", as_cmap=True)
      cmap.set_bad("lightgray")

      data_vals = []
      for hazard, full_sig in zip(hazards_np.ravel(), full_significant_np.ravel()):
        if full_sig:
          data_vals.append(r'\underline{' + "{:.2f}".format(hazard) + '}')
        else:
          data_vals.append("{:.2f}".format(hazard))

      hazard_plot = sns.heatmap(data=hazards_df, center=0, linewidth=0.6,
                                cmap=cmap,
                                annot=np.array(data_vals).reshape(np.shape(hazards_np)),
                                fmt='',
                                square=True,
                                cbar=False)

      hazard_plot.set_xlim(0, sum(list(cg_counts.values())) - 1)  # also sets axis ordering
      hazard_plot.set_ylim(0, len(sig_conds))
      hazard_plot.set(xlabel="OnsetNet score {} (input)".format(discrete_type), ylabel="Event (output)")

      plt.subplots_adjust(left=0.2)
      hazard_plot.figure.savefig(os.path.join(root_dir,
                                              "analysis/part_2_{}_hazards_condensed_{}_{}.png".format(model_pref, quartiles_suff,
                                                                                            args.correction)))
      hazard_plot.figure.savefig(os.path.join(root_dir,
                                              "analysis/part_2_{}_hazards_condensed_{}_{}.eps".format(model_pref, quartiles_suff,
                                                                                            args.correction)), format="eps")

      plt.clf()
      plt.close('all')

# ---------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------

if "correlations" in args.run_types:

  conds_ordered = []
  for cgroup in cgroups_ord: # excludes Average
    for ci1, c1 in enumerate(conditions):
      cname1 = c1[0]

      assert not cname1 == "Average"
      if condition_supergroups[cname1] == cgroup:
        conds_ordered.append(cname1)
  print(conds_ordered)

  perf_thresh = 0.6
  perfs = feather.read_feather(os.path.join(root_dir, "analysis/part_1_perf_Internal test.feather"))
  perfs = perfs[perfs["Input type"] == "demog+bb+bbc+sbc+blood"]
  perfs_d = {}
  removed_conds = []
  for cond in conds_ordered:
    res_cond = perfs[perfs["Condition"] == cond]["Internal test C-index"].to_numpy()
    print("getting perf:", cond, res_cond.shape, res_cond.mean())
    perfs_d[cond] = res_cond.mean()
    if perfs_d[cond] < perf_thresh:
      removed_conds.append(cond)
  print("removed conds for thresh", perf_thresh, len(removed_conds), removed_conds)
  num_kept = len(conds_ordered) - len(removed_conds)
  if args.corr_correction:
    corr_pthresh = 0.05 / (num_kept ** 2)
  else:
    corr_pthresh = 0.05
  print("corr_pthresh", args.corr_correction, corr_pthresh)

  #corrs_np_triangle = np.ones((num_conditions, num_conditions)) * - np.nan
  corrs_np = np.ones((num_conditions, num_conditions)) * - np.nan
  corr_pvals = np.ones((num_conditions, num_conditions)) * - np.nan
  all_pvals = []
  all_pvals_thresh = []
  collected_data = feather.read_feather(os.path.join(root_dir, "analysis/collected_data_{}.feather".format(args.fmodel)))
  print("collected_data:", collected_data.shape)
  if args.remove_statins:
    statin_users = feather.read_feather(os.path.join(root_dir, "analysis/statin_users.feather"))
    statin_users = statin_users["individual_id"].to_numpy()
    collected_data = collected_data[~collected_data["individual_id"].isin(statin_users)]
    print("removed statins, collected_data:", collected_data.shape)

  data_vals = []
  for ci1_ord, c1 in enumerate(conds_ordered): # row
    for ci2_ord, c2 in enumerate(conds_ordered): # col
      corr_val, corr_p = scipy.stats.pearsonr(collected_data["{} log hazard q".format(c1)].to_numpy(),
                                              collected_data["{} log hazard q".format(c2)].to_numpy())
      print("pearson:", c1, c2, corr_val, corr_p)

      # p value fits, and performance is good enough
      if (perfs_d[c1] >= perf_thresh) and (perfs_d[c2] >= perf_thresh):
        all_pvals.append(corr_p)
        if corr_p <= corr_pthresh:
          corrs_np[ci1_ord, ci2_ord] = corr_val
          corr_pvals[ci1_ord, ci2_ord] = corr_p
          all_pvals_thresh.append(corr_p)

          #if ci2_ord <= ci1_ord: # only add triangle
          #  corrs_np_triangle[ci1_ord, ci2_ord] = corr_val

      data_vals.append("{:.2f}".format(corr_val)) # append always

  assert len(all_pvals) == num_kept ** 2
  corr_df = pd.DataFrame(corrs_np, index=conds_ordered, columns=conds_ordered)
  print(corr_df.head())
  print(corr_df.shape)
  pval_df = pd.DataFrame(corr_pvals, index=conds_ordered, columns=conds_ordered)

  all_pvals = np.array(all_pvals)
  all_pvals_thresh = np.array(all_pvals_thresh)
  print("All corr pval accepted conds", all_pvals.shape, (all_pvals <= corr_pthresh).sum(), all_pvals.mean(), all_pvals.std(), all_pvals.min(), all_pvals.max())
  print("All corr pval accepted conds within thresh", all_pvals_thresh.shape, (all_pvals_thresh <= corr_pthresh).sum(), all_pvals_thresh.mean(), all_pvals_thresh.std(), all_pvals_thresh.min(), all_pvals_thresh.max())

  top_val_np = np.argsort(-corrs_np, axis=1) # along columns, max first
  all_rows, ps = [], []
  with open(os.path.join(root_dir, "analysis/part_2_corr_top_{}_{}.csv".format(args.corr_correction, args.remove_statins)), 'w') as topf:
    wr = csv.writer(topf, quoting=csv.QUOTE_ALL)
    wr.writerow([] * (num_conditions + 1))
    for ci1_ord, c1 in enumerate(conds_ordered):
      if perfs_d[c1] > perf_thresh:
        c1_row = [c1]
        for j in range(num_conditions):
          curr_ind = top_val_np[ci1_ord, j]
          if j == 0:
            assert curr_ind == ci1_ord
            continue
          else:
            if np.isfinite(corrs_np[ci1_ord, curr_ind]):
              row_str = "{} ({:.2f})".format(conds_ordered[curr_ind], corrs_np[ci1_ord, curr_ind])
              if corrs_np[ci1_ord, curr_ind] >= 0.65:
                row_str = "***" + row_str
              c1_row.append(row_str) #  np.format_float_scientific(corr_pvals[ci1_ord, curr_ind], precision=2, exp_digits=1)
              ps.append(corr_pvals[ci1_ord, curr_ind])
        all_rows.append(c1_row)
        wr.writerow(c1_row)

  # latex
  print("=====")
  print("Latex corr_top table:")
  top5_exceeded = 0
  top5_pvals = []
  for ci1_ord, c1 in enumerate(conds_ordered):
    if perfs_d[c1] > perf_thresh:
      c1_row = f"{c1} " #& "
      for j in range(6): # print top 5
        curr_ind = top_val_np[ci1_ord, j]
        if j == 0:
          assert curr_ind == ci1_ord
          continue
        else:
          if np.isfinite(corrs_np[ci1_ord, curr_ind]):
            #if j > 1:
            #  lower_name = conds_ordered[curr_ind][0].lower() + conds_ordered[curr_ind][1:]
            #else:
            #  lower_name = conds_ordered[curr_ind]
            row_str = " & {} ({:.2f})".format(conds_ordered[curr_ind], corrs_np[ci1_ord, curr_ind])
            #if j < 5:
            #  row_str += ", "
            c1_row += row_str
            top5_pvals.append(corr_pvals[ci1_ord, curr_ind])
            if corr_pvals[ci1_ord, curr_ind] > corr_pthresh:
              top5_exceeded += 1
          else:
            assert False
      c1_row += " \\\\"
      print(c1_row)
  print(f"Latex corr_top table finised, pval exceeded {corr_pthresh} {top5_exceeded}. Mean, std:")
  top5_pvals = np.array(top5_pvals)
  print(top5_pvals.mean(), top5_pvals.std())
  print("=====")

  ps = np.array(ps)
  print("Pvals in table: ", ps.mean(), ps.std(), ps.min(), ps.max())
  # print_latex(all_rows)

  # show insignificant as grey, annotate with different df containing strings with significance
  sns.set_theme(rc={'figure.figsize': (20, 21)})
  #sns.set(font_scale=0.8) # 0.7
  sns.set_context("paper", rc={"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                               "ytick.labelsize": 14})

  cmap = sns.color_palette("vlag", as_cmap=True)
  cmap.set_bad("lightgray")

  corr_plot = sns.heatmap(data=corr_df, center=0, linewidth=0.15,
                            cmap=cmap,
                            annot=np.array(data_vals).reshape(np.shape(corrs_np)),
                            fmt='',
                            cbar=False,
                            square=True,
                            cbar_kws=dict(use_gridspec=False,location="right",pad=0.02,shrink=0.8))

  # draw dashed lines
  curr_cond = 0
  for cgi, cg in enumerate(cgroups_ord): # the order the condition groups are printed, excluding Average
    if cgi == len(cgroups_ord) - 1:
      break

    curr_cond += cg_counts[cg]
    line_ind = curr_cond #- 0.5
    corr_plot.axes.axvline(line_ind, color="dimgray", linestyle="--") # end of this one #  linewidth=0.15
    corr_plot.axes.axhline(line_ind, color="dimgray", linestyle="--") # end of this one # linewidth=0.15

  #corr_plot.set_xticklabels(corr_plot.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
  #corr_plot.set_yticklabels(corr_plot.get_yticklabels(), rotation=(-30), ha="right", rotation_mode='anchor')

  corr_plot.set_xlim(0, sum(list(cg_counts.values())) - 1) # also sets axis ordering
  corr_plot.set_ylim(0, sum(list(cg_counts.values())) - 1)
  corr_plot.set(xlabel="OnsetNet score quartile", ylabel="OnsetNet score quartile")

  plt.subplots_adjust(left=0.2)
  plt.tight_layout()
  corr_plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_corr_correct_{}_stat_{}.png".format(args.corr_correction, args.remove_statins)))
  corr_plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_corr_correct_{}_stat_{}.eps".format(args.corr_correction, args.remove_statins)), format="eps")
  plt.clf()
  plt.close('all')

  if perf_thresh == 0:
    exit(0)

  # cluster map
  # remove columns with all nan
  fontsz = 14
  sns.set_context("paper", rc={"font.size": fontsz, "axes.titlesize": 18, "axes.labelsize": fontsz, "xtick.labelsize": fontsz,
                               "ytick.labelsize": fontsz})
  group_colors = sns.color_palette("husl", n_colors=len(cgroups_ord)).as_hex()
  cond_row_colors = [group_colors[cgroups_ord.index(condition_supergroups[conds_ordered[ci]])] for ci in range(num_conditions)]
  cond_col_colors = [group_colors[cgroups_ord.index(condition_supergroups[conds_ordered[ci]])] for ci in range(num_conditions)]

  keep_cols = []
  keep_colnames = []
  cond_col_colors_keep = []
  for ci in range(corrs_np.shape[1]):
    if np.isfinite(corrs_np[:, ci]).any():
      keep_cols.append(ci)
      keep_colnames.append(conds_ordered[ci])
      cond_col_colors_keep.append(cond_col_colors[ci])

  keep_rows = []
  keep_rownames = []
  cond_row_colors_keep = []
  for ri in range(corrs_np.shape[0]):
    if np.isfinite(corrs_np[ri, :]).any():
      keep_rows.append(ri)
      keep_rownames.append(conds_ordered[ri])
      cond_row_colors_keep.append(cond_col_colors[ri])

  corrs_clus_keep = corrs_np[:, keep_cols]
  corrs_clus_keep = corrs_clus_keep[keep_rows, :]
  print("keep clus shape:", corrs_clus_keep.shape, len(keep_rows), len(keep_cols))
  corrs_clus_keep_df = pd.DataFrame(corrs_clus_keep, index=keep_rownames, columns=keep_colnames)

  # replace nan with 0
  corrs_clus_keep_df.fillna(0, inplace=True)
  corrs_clus_keep_df.replace(- np.nan, 0, inplace=True)
  print(np.isfinite(corrs_clus_keep_df.to_numpy()).all())
  #sns.set(font_scale=1.05)
  cluster_plot = sns.clustermap(figsize=(15, 15), data=corrs_clus_keep_df, cmap=cmap, center=0, # cbar_kws=dict(shrink=0.5)
                                row_colors=cond_row_colors_keep, col_colors=cond_col_colors_keep,
                                metric="cosine", method="average", dendrogram_ratio=(.15, .15)) # cbar_pos=None,
  cluster_plot.ax_row_dendrogram.set_visible(False)
  cluster_plot.ax_heatmap.tick_params(left=False, bottom=False, right=False)

  #plt.setp(cluster_plot.ax_heatmap.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
  #plt.setp(cluster_plot.ax_heatmap.get_yticklabels(), rotation=(30), ha="left", rotation_mode='anchor')

  cluster_plot.ax_heatmap.set(xlabel="OnsetNet score quartile", ylabel="OnsetNet score quartile")
  #cluster_plot.ax_cbar.set_position([0.79, 0.88, 0.03, cluster_plot.ax_col_dendrogram.get_position().height * 0.7])
  cluster_plot.ax_cbar.set_position([0.79, 0.88, 0.02, cluster_plot.ax_col_dendrogram.get_position().height * 0.67])
  plt.tight_layout()
  cluster_plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_corr_clus_correct_{}_stat_{}.png".format(args.corr_correction, args.remove_statins)))
  cluster_plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_corr_clus_correct_{}_stat_{}.eps".format(args.corr_correction, args.remove_statins)), format="eps")
  plt.clf()
  plt.close('all')

  # chord diagram
  chord_thresh = 0.65 # 0.6
  palette = sns.color_palette("husl", n_colors=(len(cgroups_ord) + 1)).as_hex()
  palette = [palette[(j + 2) % len(palette)] for j in range(len(palette))]
  print(conds_ordered)
  print(palette)
  corrs_np_orig = np.copy(corrs_np)
  corrs_np[corrs_np < chord_thresh] = - np.nan
  chord_max = corrs_np[np.isfinite(corrs_np)].max()
  chord_min = corrs_np[np.isfinite(corrs_np)].min() - 0.03 # to give something to render
  corrs_np = (corrs_np - chord_min) / (chord_max - chord_min)

  render = {}
  chord = defaultdict(list)
  chord_color = []
  for ri in range(corrs_np.shape[0]):
    for ci in range(ri):  # triangle
      if np.isfinite(corrs_np[ri, ci]): #  and corrs_np[ri, ci] >= chord_thresh
        chord["source"].append(conds_ordered[ri])
        chord["target"].append(conds_ordered[ci])
        chord["weight"].append(corrs_np[ri, ci])
        chord_color.append(
          palette[cgroups_ord.index(condition_supergroups[conds_ordered[ci]])])  # assign color of link to target
        render[conds_ordered[ri]] = True
        render[conds_ordered[ci]] = True

  render_conds_ordered = []
  for cond in conds_ordered:
    if cond in render:
      render_conds_ordered.append(cond)

  print("added this many colors: ", len(chord_color))
  chord_df = pd.DataFrame(dict(chord))
  d3 = D3Blocks(frame=True)  # frame=False
  fpath = os.path.join(root_dir, "analysis/part_2_corr_chord_correct_{}_stat_{}.html".format(args.corr_correction, args.remove_statins))
  d3.chord(chord_df,
           ordering=render_conds_ordered,
           filepath=fpath, save_button=True,
           color='source', opacity='source', cmap="vlag", arrowhead=-1)  # auto colour scheme
  d3.show()  # coloring determined by d3

  print(list(d3.node_properties))
  print(d3.node_properties.index)
  print(d3.edge_properties.index)

  key_conds = {}
  for ci in range(corrs_np.shape[1]):
    ci_name = conds_ordered[ci]
    if ci_name in render:
      d3.node_properties.loc[d3.node_properties["label"] == ci_name, "color"] = palette[
        cgroups_ord.index(condition_supergroups[ci_name])] # last k
      d3.edge_properties.loc[d3.edge_properties["source"] == ci_name, "color"] = "#C0C0C0"
      d3.node_properties.loc[d3.node_properties["label"] == ci_name, "opacity"] = 0.98
      d3.edge_properties.loc[d3.edge_properties["source"] == ci_name, "opacity"] = 0.98
      if not condition_supergroups[ci_name] == "Composite":
        key_conds[condition_supergroups[ci_name]] = True

  # first two group colours for composite
  composite_color = palette[cgroups_ord.index(condition_supergroups["All cause mortality"])]
  d3.node_properties.loc[d3.node_properties["label"] == "All cause morbidity", "color"] = palette[-1]  # extra one
  d3.edge_properties.loc[d3.edge_properties["source"] == "All cause morbidity", "color"] = palette[-1]
  d3.edge_properties.loc[d3.edge_properties["target"] == "All cause morbidity", "color"] = palette[-1]
  key_conds["All cause morbidity"] = True

  d3.node_properties.loc[d3.node_properties["label"] == "All cause mortality", "color"] = composite_color
  d3.edge_properties.loc[d3.edge_properties["source"] == "All cause mortality", "color"] = composite_color
  d3.edge_properties.loc[d3.edge_properties["target"] == "All cause mortality", "color"] = composite_color
  key_conds["All cause mortality"] = True

  d3.show()
  print("saved to {}".format(fpath))

  # colour legend
  figdata = pylab.figure(figsize=(3,2))
  ax = pylab.gca()
  for cond in key_conds:
    if cond == "All cause mortality":
      cg_c = composite_color
    elif cond == "All cause morbidity":
      cg_c = palette[-1]
    else:
      cg_c = palette[cgroups_ord.index(cond)]
    pylab.plot(range(5), range(5), label=cond, color=cg_c) # dummy
  figlegend = pylab.figure(figsize=(10, 3)) # 3, 4
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left', ncol=4, facecolor='white', framealpha=1, edgecolor='white')
  figlegend.savefig(os.path.join(root_dir, 'analysis/part_2_chord_legend_{}_{}.png'.format(args.corr_correction, args.remove_statins)))
  figlegend.savefig(os.path.join(root_dir, 'analysis/part_2_chord_legend_{}_{}.eps'.format(args.corr_correction, args.remove_statins)), format="eps")

  # networkx
  del corrs_np

  for seed in [17]:
    set_seed(seed)
    def make_edge(x, y, text, width, color):
      return go.Scatter(x=x,
                        y=y,
                        line=dict(width = width,
                                     color = color),
                        hoverinfo='text',
                        text=([text]),
                        mode='lines')

    def corr_standard(corr):
      return corr >= 0.3

    midsummer = nx.Graph()

    render = {}
    for ri in range(corrs_np_orig.shape[0]):
      for ci in range(ri):  # triangle
        if np.isfinite(corrs_np_orig[ri, ci]) and corr_standard(corrs_np_orig[ri, ci]): # perf and p both ok
          render[conds_ordered[ri]] = True
          render[conds_ordered[ci]] = True

    dict_corrs = {}
    for ri in range(corrs_np_orig.shape[0]):
      for ci in range(ri):  # triangle
        if np.isfinite(corrs_np_orig[ri, ci]) and corr_standard(corrs_np_orig[ri, ci]):
          midsummer.add_edge(conds_ordered[ri], conds_ordered[ci], weight=corrs_np_orig[ri, ci])
          dict_corrs["{}_{}".format(conds_ordered[ri], conds_ordered[ci])] = corrs_np_orig[ri, ci]

    render_cg = {}
    for node_name in list(render.keys()):
      midsummer.add_node(node_name, size=2.5)
      render_cg[condition_supergroups[node_name]] = True

    palette_nodes = sns.color_palette("husl", n_colors=len(cgroups_ord)).as_hex()
    palette_edges = sns.color_palette("vlag", n_colors=5).as_hex()

    pos_orig = nx.spring_layout(midsummer, seed=seed)

    for node in ["Osteoporosis", "Prostatic hyperplasia", "Lower limb varicosis","Parkinson’s disease", "Lipid metabolism disorders",
                 "All cause morbidity", "Hypertension", "Cerebral ischemia/chronic stroke", "Chronic gastritis/GERD"]:
      print(node, pos_orig[node])

    if not (args.remove_statins):
      pos_new = {"Prostatic hyperplasia": (-0.4, -0.1),
                 "Lower limb varicosis": (0.15, 0.25),
                 "Osteoporosis": (-0.33, 0.4),
                 "Hyperuricemia/gout": (-0.42, 0.3),
                 "Parkinson’s disease": (0.42, -0.195),
                 "Lipid metabolism disorders": (-0.39, 0.158)
                 }
    else:
      pos_new = {"Prostatic hyperplasia": (-0.3, -0.1),
                 "Lower limb varicosis": (0.15, 0.25),
                 "Osteoporosis": (-0.33, 0.4),
                 "Chronic gastritis/GERD": (0.12, 0.07523029),
                 }

    for k, v in pos_new.items():
      pos_orig[k] = tuple(v)

    pos_ = nx.spring_layout(midsummer, pos=pos_orig, fixed=list(pos_orig.keys()))
    #else:
    #  print("Using original pos")
    #  pos_ = pos_orig

    edge_trace = []
    for edge in midsummer.edges():
      if midsummer.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]
        key_1 = "{}_{}".format(char_1, char_2)
        key_2 = "{}_{}".format(char_2, char_1)

        if key_1 in dict_corrs:
          corr_val = dict_corrs[key_1]
        else:
          assert key_2 in dict_corrs
          corr_val = dict_corrs[key_2]

        if (condition_supergroups[char_1] == "Composite") or (condition_supergroups[char_2] == "Composite"):
          corr_color = "#000000"
        else:
          corr_color = palette_edges[-1]

        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]

        text = char_1 + '--' + char_2 + ': ' + str(midsummer.edges()[edge]['weight'])
        trace = make_edge([x0, x1, None], [y0, y1, None], text,
                          midsummer.edges()[edge]['weight'] ** 3., color=corr_color)

        edge_trace.append(trace)

    # Make a node trace
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            textposition = "top center",
                            textfont_size = 10, # todo
                            mode      = 'markers+text',
                            hoverinfo = 'none',
                            marker    = dict(color = [],
                                             size  = [],
                                             line  = None))

    # For each node in midsummer, get the position and size and add to the node_trace
    for node in midsummer.nodes():
      if condition_supergroups[node] == "Composite":
        node_color = "#000000"
      else:
        node_color = palette_nodes[cgroups_ord.index(condition_supergroups[node])]

      x, y = pos_[node]
      node_trace['x'] += tuple([x])
      node_trace['y'] += tuple([y])
      node_trace['marker']['color'] += tuple([node_color])
      node_trace['marker']['size'] += tuple([5*midsummer.nodes()[node]['size']])
      node_trace['text'] += tuple(['<b>' + node + '</b>'])

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(layout = layout)

    for trace in edge_trace:
        fig.add_trace(trace)

    fig.add_trace(node_trace)
    fig.update_layout(showlegend = False)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(template='simple_white')
    fig.show()
    #py.plot(fig, filename=os.path.join(root_dir, 'analysis/part_2_midsummer_network_correct_{}_stat_{}_{}.html'.format(seed)))
    py.plot(fig, image = 'svg', image_filename="part_2_midsummer_network_correct_{}_stat_{}_{}".format(args.corr_correction, args.remove_statins, seed),
              output_type='file', image_width=1400, image_height=650,
              filename=os.path.join(root_dir, 'analysis/part_2_midsummer_network_correct_{}_stat_{}_{}.html'.format(args.corr_correction, args.remove_statins, seed)))

  figdata = pylab.figure(figsize=(3,2))
  ax = pylab.gca()
  for cg_i, cg in enumerate(cgroups_ord):
    if cg in render_cg:
      if not cg == "Composite":
        cg_c = palette_nodes[cg_i]
      else:
        cg_c = "black"
      pylab.plot(range(5), range(5), label=cg, color=cg_c) # dummy
  figlegend = pylab.figure(figsize=(3, 4))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
  figlegend.savefig(os.path.join(root_dir, 'analysis/part_2_midsummer_network_legend.png'))
  figlegend.savefig(os.path.join(root_dir, 'analysis/part_2_midsummer_network_legend.svg'), format="svg")

# ---------------------------------------------------------------
# Subtype illstrations
# ---------------------------------------------------------------

if "subtypes" in args.run_types:
  suff = "q"

  conditions_df = feather.read_feather(os.path.join(root_dir, "analysis/conditions_ordered_{}.feather".format(args.fmodel)))
  conditions_df = list(conditions_df["condition_group"].to_numpy())
  print("R conditions ordered")
  print(conditions_df)

  # ids, log_hazards already computed
  collected_data_df = feather.read_feather(os.path.join(root_dir, "analysis/collected_data_{}.feather".format(args.fmodel)))
  # individual_id x conditions: risk quartile

  blood_ind = []
  for bf in render_blood_vars:
    blood_ind.append(BLOOD_fields.index((bf, None)))
  blood_ind = np.array(blood_ind)

  with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
    nn_data = pickle.load(f)

  # get people with statin usage
  statin_users = feather.read_feather(os.path.join(root_dir, "analysis/statin_users.feather"))
  statin_users = statin_users["individual_id"].to_numpy()
  print("statin shape", statin_users.shape)

  # find average distribution for race, edu, townsend per sex
  demog_varnames = ["sex", "race", "education", "townsend_deprivation"]
  data_varnames = ["bb", "sbc", "bbc", "age_imaging", "blood"]

  raw_vars = defaultdict(list)
  for dname in ["int_trainval", "int_test", "ext_test"]: # not normed
    raw_vars["ids"].append(nn_data[dname]["ids"])

    curr_field_ind = 0
    for field in demog_varnames:
      shape = -1
      for demog_name, demog_shape in DEMOG_fields:
        if demog_name == field:
          shape = demog_shape
          break
      assert shape != -1

      if shape is None:
        curr_field_end = curr_field_ind + 1
      else:
        curr_field_end = curr_field_ind + shape
      print(field, curr_field_ind, curr_field_end)
      raw_vars[field].append(nn_data[dname]["demog"][:, curr_field_ind:curr_field_end])
      curr_field_ind = curr_field_end

    for varname in data_varnames:
      raw_vars[varname].append(nn_data[dname][varname])

  # compute average template demog distribution
  all_vars = {}
  for varname in ["ids"] + demog_varnames + data_varnames:
    all_vars[varname] = np.concatenate(raw_vars[varname], axis=0).squeeze()

  print(all_vars["ids"].shape)
  _, id_counts = np.unique(all_vars["ids"], return_counts=True)
  assert (id_counts == 1).all()

  ids_df = pd.DataFrame({"individual_id": all_vars["ids"]})
  collected_data_df = pd.merge(ids_df, collected_data_df, how="left", on=["individual_id"])
  assert ((collected_data_df["individual_id"].to_numpy() == all_vars["ids"]).all()) and not collected_data_df.isnull().values.any()

  num_townsend = 10
  all_vars["townsend_deprivation"] = pd.qcut(all_vars["townsend_deprivation"], q=num_townsend, labels=False) # discretize
  assert isinstance(all_vars["townsend_deprivation"], np.ndarray)
  assert all_vars["townsend_deprivation"].min() == 0 and all_vars["townsend_deprivation"].max() == num_townsend - 1

  distr = defaultdict(int) # counts per combo
  distr_combos = [] # combo index per person
  combo_list = [] # fixed order of combo index
  for _, _, race, ed, td in zip(all_vars["ids"], all_vars["sex"], all_vars["race"], all_vars["education"], all_vars["townsend_deprivation"]):
    combo = "{}_{}_{}".format(race, ed, td)
    distr[combo] += 1
    if not combo in combo_list:
      combo_list.append(combo)
    distr_combos.append(combo_list.index(combo))

  combos = list(distr.keys())
  distr_np = np.array([distr[combo] for combo in combo_list])
  distr_np = distr_np/distr_np.sum() # distribution over combos
  distr_combos = np.array(distr_combos) # combo index per person

  # render averages
  print("Beginning to render conditions")
  print(conditions)
  for ci_ord, co in enumerate(conditions):
    c = co[0]
    ci = conditions_df.index(c)
    print("Doing tableau", ci_ord, c, ci)

    netriskq_colname = "{} log hazard q".format(c)
    netriskq_c = collected_data_df[netriskq_colname].to_numpy()  # already ordered
    print(netriskq_c.shape, get_stats(netriskq_c))

    netrisk_colname = "{} log hazard".format(c)
    netrisk_c = np.exp(collected_data_df[netrisk_colname].to_numpy())  # already ordered

    hazard_res_c = feather.read_feather(
      os.path.join(root_dir,"analysis/linear_cox/{}_{}_{}_res1_{}.feather".format("basicplus", ci + 1, ci + 1, suff)))

    hazard_res_c_full_adj = feather.read_feather(
      os.path.join(root_dir,"analysis/linear_cox/{}_{}_{}_res1_{}.feather".format("full", ci + 1, ci + 1, suff)))

    sex_c = [0, 1]
    results = {0: defaultdict(dict), 1:  defaultdict(dict)}
    if c in sex_f_conds:
      sex_c = [1]
      results = {1: defaultdict(dict)}
    if c in sex_m_conds:
      sex_c = [0]
      results = {0: defaultdict(dict)}

    no_statins = (np.expand_dims(all_vars["ids"], 1) == np.expand_dims(statin_users, 0)).sum(axis=1) == 0
    print("statin info", no_statins.shape, all_vars["ids"].shape, no_statins.shape, statin_users.shape, no_statins.sum())

    for s in sex_c:
      s_bool = all_vars["sex"][:, s] == 1 # s is discrete 2 column bool

      for q in range(4):
        # sample types according to general distribution
        q_s_bool = np.logical_and(np.logical_and(netriskq_c == q, s_bool), no_statins)

        q_s_ids = all_vars["ids"][q_s_bool]
        print(c, s, q, q_s_bool.shape, q_s_ids.shape)
        print("with this quartile", all_vars["ids"][netriskq_c == q].shape)
        print(all_vars["sex"][netriskq_c == q][:5])

        # compute marginal distribution p(x) = sum_combo p(x | combo) p(combo) = p_combo/n_curr_for_combo
        q_s_combos = distr_combos[q_s_bool] # num people q_s. combo id per person
        q_s_probs = distr_np[q_s_combos] # num people q_s. original combo marginal per person
        assert q_s_probs.shape == q_s_ids.shape
        probsum1 = q_s_probs.sum()
        combo_ids, q_s_count_per_combo_part = np.unique(q_s_combos, return_counts=True)
        assert (combo_ids >= 0).all() and (combo_ids <= len(combo_list) - 1).all()
        q_s_count_per_combo = np.zeros(len(combos))
        q_s_count_per_combo[combo_ids] = q_s_count_per_combo_part # may not have certain combo present at all
        q_s_counts_expanded = q_s_count_per_combo[q_s_combos] # num people q_s
        assert q_s_counts_expanded.shape == q_s_ids.shape and (q_s_counts_expanded > 0).all()

        q_s_probs = np.divide(q_s_probs, q_s_counts_expanded)

        print("shapes", distr_combos.shape, q_s_combos.shape, distr_np.shape, q_s_probs.shape)
        # from set of all q_s people, choose set of same size according to non-uniform q_s_probs
        probsum2 = q_s_probs.sum()
        q_s_probs = q_s_probs/q_s_probs.sum() # small correction
        print("q_s_probs", probsum1, probsum2, q_s_probs.sum())

        resampled = np.random.choice(q_s_ids.shape[0], size=q_s_ids.shape[0], p=q_s_probs, replace=False) # q_s_ids.shape[0]
        r_ids = q_s_ids[resampled]
        assert r_ids.shape == q_s_ids.shape

        r_ids_bool = (np.expand_dims(all_vars["ids"], axis=1) == np.expand_dims(r_ids, axis=0)).sum(axis=1) # all, selected
        assert r_ids_bool.shape == all_vars["ids"].shape
        nu, nc = np.unique(r_ids_bool, return_counts=True)
        print(nu, nc)
        assert np.logical_or(r_ids_bool == 1, r_ids_bool == 0).all()
        r_ids_bool = r_ids_bool.astype(bool)
        print(r_ids_bool.shape, r_ids.shape)
        print(r_ids_bool.sum())

        # compute averages
        netrisk_q_c = netrisk_c[r_ids_bool]
        results[s][q]["netrisk"] = (netrisk_q_c.mean(), netrisk_q_c.std())

        # hazard ratio (exp log)
        assert suff == "q"
        if q > 0:
          log_hr = hazard_res_c[hazard_res_c["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\"){}".format(q)]["estimate"].squeeze()
          pval = hazard_res_c[hazard_res_c["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\"){}".format(q)]["p.value"].squeeze()

          log_hr_full_adj = hazard_res_c_full_adj[hazard_res_c_full_adj["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\"){}".format(q)]["estimate"].squeeze()
          pval_full_adj = hazard_res_c_full_adj[hazard_res_c_full_adj["term"] == "relevel(as.factor(log_hazard_q), ref = \"0\"){}".format(q)]["p.value"].squeeze()

          results[s][q]["hazard_ratio"] = (np.exp(log_hr), pval)
          results[s][q]["hazard_ratio_ful_adj"] = (np.exp(log_hr_full_adj), pval_full_adj)
        else:
          results[s][q]["hazard_ratio"] = (-np.inf, -np.inf) # placeholder
          results[s][q]["hazard_ratio_ful_adj"] = (-np.inf, -np.inf) # placeholder

        # SBC
        sbc_curr = all_vars["sbc"][r_ids_bool]
        assert len(sbc_curr.shape) == 3
        results[s][q]["sbc"] = (sbc_curr.mean(axis=0), sbc_curr.std(axis=0))

        # age at imaging
        age_curr = all_vars["age_imaging"][r_ids_bool]
        results[s][q]["age"] = (age_curr.mean(), age_curr.std())

        # BBC
        bbc_curr = all_vars["bbc"][r_ids_bool]
        results[s][q]["bbc"] = (bbc_curr.mean(axis=0), bbc_curr.std(axis=0))

        # BB
        bb_curr = all_vars["bb"][r_ids_bool]
        curr_bbi = 0
        for bbf, bb_shape in BB_fields:
          if bb_shape is not None:
            curr_bbi_end = curr_bbi + bb_shape
            if "age" in bbf:
              raise ValueError # age always scalar

            bbf_curr = bb_curr[:, curr_bbi:curr_bbi_end]
            bff_inds, bbf_curr_flat = bbf_curr.nonzero()
            assert (bff_inds == np.arange(bb_curr.shape[0])).all()
            bff_mode, bff_count = st.mode(bbf_curr_flat, axis=None, keepdims=False)
            if bbf == "alcohol":
              bff_mode_text = alcohol_fields[bff_mode]
            elif bbf == "smoking":
              bff_mode_text = smoking_fields[bff_mode]
            else:
              raise NotImplementedError

            results[s][q][bbf] = (bff_mode_text, "{:10.1f}".format(100.0 * bff_count / float(bb_curr.shape[0])))
          else:
            curr_bbi_end = curr_bbi + 1
            if "age" in bbf:
              curr_bbi = curr_bbi_end
              continue

            bbf_curr = bb_curr[:, curr_bbi:curr_bbi_end].squeeze()
            results[s][q][bbf] = (bbf_curr.mean(), bbf_curr.std())

          curr_bbi = curr_bbi_end

        # BLOOD (previous instance)
        blood_curr = all_vars["blood"][r_ids_bool]
        results[s][q]["blood"] = (blood_curr[:, blood_ind].mean(axis=0), blood_curr[:, blood_ind].std(axis=0))

    render_condition(ci, c, sex_c, results)

# ------------------------------------------------------------------
# Traj illstrations - distribution of demographics standardised too
# ------------------------------------------------------------------

if "render_traj" in args.run_types:
  #sns.set_theme(rc={'figure.figsize': (8, 10)})
  sns.set(font_scale=0.8)
  count_thresh = 100 #100

  collected_data_df = feather.read_feather(os.path.join(root_dir, "analysis/collected_data_{}.feather".format(args.fmodel)))

  blood_ind = []
  for bf in render_blood_vars:
    blood_ind.append(BLOOD_fields.index((bf, None)))
  blood_ind = np.array(blood_ind)

  with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
    nn_data = pickle.load(f)

  statin_users = feather.read_feather(os.path.join(root_dir, "analysis/statin_users.feather"))
  statin_users = statin_users["individual_id"].to_numpy()
  print("statin shape", statin_users.shape)

  # find average distribution for race, edu, townsend per sex
  demog_varnames = ["sex", "race", "education", "townsend_deprivation"]
  data_varnames = ["bb", "sbc", "bbc", "age_imaging", "blood"]

  raw_vars = defaultdict(list)
  for dname in ["int_trainval", "int_test", "ext_test"]: # not normed
    raw_vars["ids"].append(nn_data[dname]["ids"])

    curr_field_ind = 0
    for field in demog_varnames:
      shape = -1
      for demog_name, demog_shape in DEMOG_fields:
        if demog_name == field:
          shape = demog_shape
          break
      assert shape != -1

      if shape is None:
        curr_field_end = curr_field_ind + 1
      else:
        curr_field_end = curr_field_ind + shape
      print(field, curr_field_ind, curr_field_end)
      raw_vars[field].append(nn_data[dname]["demog"][:, curr_field_ind:curr_field_end])
      curr_field_ind = curr_field_end

    for varname in data_varnames:
      raw_vars[varname].append(nn_data[dname][varname])

  all_vars = {}
  for varname in ["ids"] + demog_varnames + data_varnames:
    all_vars[varname] = np.concatenate(raw_vars[varname], axis=0).squeeze()
  add_non_imaging_fields(all_vars)

  # compute average template demog distribution
  print(all_vars["ids"].shape)
  _, id_counts = np.unique(all_vars["ids"], return_counts=True)
  assert (id_counts == 1).all()

  ids_df = pd.DataFrame({"individual_id": all_vars["ids"]})
  collected_data_df = pd.merge(ids_df, collected_data_df, how="left", on=["individual_id"])
  assert ((collected_data_df["individual_id"].to_numpy() == all_vars["ids"]).all()) and not collected_data_df.isnull().values.any()

  num_townsend = 10
  all_vars["townsend_deprivation"] = pd.qcut(all_vars["townsend_deprivation"], q=num_townsend, labels=False) # discretize
  assert isinstance(all_vars["townsend_deprivation"], np.ndarray)
  assert all_vars["townsend_deprivation"].min() == 0 and all_vars["townsend_deprivation"].max() == num_townsend - 1

  distr = defaultdict(int) # counts
  distr_combos = [] # combo index per person
  combo_list = [] # fixed order of combo index
  for _, _, race, ed, td in zip(all_vars["ids"], all_vars["sex"], all_vars["race"], all_vars["education"], all_vars["townsend_deprivation"]):
    combo = "{}_{}_{}".format(race, ed, td)
    distr[combo] += 1
    if not combo in combo_list:
      combo_list.append(combo)
    distr_combos.append(combo_list.index(combo))

  combos = list(distr.keys())
  distr_np = np.array([distr[combo] for combo in combo_list])
  distr_np = distr_np/distr_np.sum()
  distr_combos = np.array(distr_combos)

  # render average trajectories
  print("Beginning to render trajectories")
  print(conditions)

  no_statins = (np.expand_dims(all_vars["ids"], 1) == np.expand_dims(statin_users, 0)).sum(axis=1) == 0
  print("statin info", no_statins.shape, all_vars["ids"].shape, no_statins.shape, statin_users.shape, no_statins.sum())

  for ci, co in enumerate(conditions):
    c = co[0]
    netriskq_colname = "{} log hazard q".format(c)
    netriskq_c = collected_data_df[netriskq_colname].to_numpy()  # already ordered
    print(netriskq_c.shape, get_stats(netriskq_c))

    sex_c = [0, 1]
    if c in sex_f_conds:
      sex_c = [1]
    if c in sex_m_conds:
      sex_c = [0]

    for s in sex_c:
      s_bool = all_vars["sex"][:, s] == 1 # s is discrete 2 column bool
      render_c_s = defaultdict(list)
      max_vals = {}

      stds = {"x": defaultdict(list), "y0": defaultdict(list), "y1": defaultdict(list), "color_ind": {}}
      stds_norm = {"x": {}, "y0": {}, "y1": {}}

      plot_quartiles = ["Q1", "Q2+"]
      for q_name in plot_quartiles: #[1, 4]: # range(4)
        # sample types according to general distribution
        if q_name == plot_quartiles[0]:
          q_s_bool = np.logical_and(netriskq_c == 0, s_bool)
        else:
          q_s_bool = np.logical_and(netriskq_c > 0, s_bool)
        q_s_bool = np.logical_and(q_s_bool, no_statins)

        q_s_ids = all_vars["ids"][q_s_bool]

        # compute marginal distribution p(x) = sum_combo p(x | combo) p(combo) = p_combo/n_curr_for_combo
        q_s_combos = distr_combos[q_s_bool] # num people q_s
        q_s_probs = distr_np[q_s_combos] # num people q_s
        assert q_s_probs.shape == q_s_ids.shape
        probsum1 = q_s_probs.sum()
        combo_ids, q_s_count_per_combo_part = np.unique(q_s_combos, return_counts=True)
        assert (combo_ids >= 0).all() and (combo_ids <= len(combo_list) - 1).all()
        q_s_count_per_combo = np.zeros(len(combos))
        q_s_count_per_combo[combo_ids] = q_s_count_per_combo_part # may not have certain combo present at all
        q_s_counts_expanded = q_s_count_per_combo[q_s_combos] # num people q_s
        assert q_s_counts_expanded.shape == q_s_ids.shape and (q_s_counts_expanded > 0).all()

        q_s_probs = np.divide(q_s_probs, q_s_counts_expanded)

        print("shapes", distr_combos.shape, q_s_combos.shape, distr_np.shape, q_s_probs.shape)
        # from set of all q_s people, choose set of same size according to non-uniform q_s_probs
        probsum2 = q_s_probs.sum()
        q_s_probs = q_s_probs /q_s_probs.sum() # small correction
        print("q_s_probs", probsum1, probsum2, q_s_probs.sum())

        resampled = np.random.choice(q_s_ids.shape[0], size=q_s_ids.shape[0], p=q_s_probs, replace=False) # q_s_ids.shape[0]
        r_ids = q_s_ids[resampled]
        assert r_ids.shape == q_s_ids.shape

        r_ids_bool = (np.expand_dims(all_vars["ids"], axis=1) == np.expand_dims(r_ids, axis=0)).sum(axis=1) # all, selected
        assert r_ids_bool.shape == all_vars["ids"].shape
        nu, nc = np.unique(r_ids_bool, return_counts=True)
        print(nu, nc)
        assert np.logical_or(r_ids_bool == 1, r_ids_bool == 0).all()
        r_ids_bool = r_ids_bool.astype(bool)
        print(r_ids_bool.shape, r_ids.shape)
        print(r_ids_bool.sum())
        n_r = r_ids_bool.sum()

        # assemble the fields
        age_bbc = all_vars["age_imaging"][r_ids_bool]
        bbc_curr = all_vars["bbc"][r_ids_bool] # n, num_subfields
        assert bbc_curr.shape == (n_r, len(channels))
        assert (bbc_curr >= 0).all()

        age_bb = all_vars["bb_flat_age"][r_ids_bool]
        bb_curr = all_vars["bb_flat"][r_ids_bool]
        assert age_bb.shape == (n_r, len(render_bb_vars)) and bb_curr.shape == (n_r, len(render_bb_vars))

        age_blood = all_vars["blood_flat_age"][r_ids_bool]
        blood_curr = all_vars["blood_flat"][r_ids_bool]
        assert age_blood.shape == (n_r, len(render_blood_vars)) and blood_curr.shape == (n_r, len(render_blood_vars))

        subfield_max = {} # over all ages
        for age_mid in [42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5]:
          age_i = age_mid - 2.5
          age_j = age_mid + 2.5
          # BBC
          bbc_age_bool = np.logical_and(age_bbc >= age_i, age_bbc < age_j)
          if bbc_age_bool.sum() >= count_thresh:
            bbc_age_avg = bbc_curr[bbc_age_bool, :].mean(axis=0)
            bbc_age_std = bbc_curr[bbc_age_bool, :].std(axis=0)

            for bbc_i, bbc_subfield in enumerate(channels):
              simple_text_name = simple_text_names[bbc_subfield]
              render_c_s["Biomarker_type"].append("bbc")
              render_c_s["Biomarker"].append(simple_text_name)
              render_c_s["Quartile"].append(q_name)
              render_c_s["Value"].append(bbc_age_avg[bbc_i])
              render_c_s["Age"].append(age_mid)
              render_c_s["Count"].append(bbc_age_bool.sum())

              if not simple_text_name in max_vals:
                max_vals[simple_text_name] = bbc_age_avg[bbc_i]
              else:
                max_vals[simple_text_name] = max(max_vals[simple_text_name], bbc_age_avg[bbc_i])

              stds["y0"]["{} {}".format(simple_text_name, q_name)].append(bbc_age_avg[bbc_i] + 2 * bbc_age_std[bbc_i])
              stds["y1"]["{} {}".format(simple_text_name, q_name)].append(bbc_age_avg[bbc_i] - 2 * bbc_age_std[bbc_i])
              stds["x"]["{} {}".format(simple_text_name, q_name)].append(age_mid)
              stds["color_ind"][simple_text_name] = bbc_i

          # BB
          for bbi, bb_subfield in enumerate(render_bb_vars):
            simple_text_name = simple_text_names[bb_subfield]
            bb_age_bool = np.logical_and(age_bb[:, bbi] >= age_i, age_bb[:, bbi] < age_j)
            if bb_age_bool.sum() >= count_thresh:
              #print(bb_curr.shape, bb_age_bool.shape)
              bb_age_avg = bb_curr[bb_age_bool, bbi].mean(axis=0)
              bb_age_std = bb_curr[bb_age_bool, bbi].std(axis=0)

              render_c_s["Biomarker_type"].append("bb")
              render_c_s["Biomarker"].append(simple_text_name)
              render_c_s["Quartile"].append(q_name)
              render_c_s["Value"].append(bb_age_avg)
              render_c_s["Age"].append(age_mid)
              render_c_s["Count"].append(bb_age_bool.sum())

              if not simple_text_name in max_vals:
                max_vals[simple_text_name] = bb_age_avg
              else:
                max_vals[simple_text_name] = max(max_vals[simple_text_name], bb_age_avg)

              stds["y0"]["{} {}".format(simple_text_name, q_name)].append(bb_age_avg + 2 * bb_age_std)
              stds["y1"]["{} {}".format(simple_text_name, q_name)].append(bb_age_avg - 2 * bb_age_std)
              stds["x"]["{} {}".format(simple_text_name, q_name)].append(age_mid)
              stds["color_ind"][simple_text_name] = bbi

          # BLOOD
          per_panel = int(np.ceil(len(render_blood_vars) / 3.))
          for bloodi, blood_subfield in enumerate(render_blood_vars):
            simple_text_name = simple_text_names[blood_subfield]
            blood_age_bool = np.logical_and(age_blood[:, bloodi] >= age_i, age_blood[:, bloodi] < age_j)
            if blood_age_bool.sum() >= count_thresh:
              blood_age_avg = blood_curr[blood_age_bool, bloodi].mean(axis=0)
              blood_age_std = blood_curr[blood_age_bool, bloodi].std(axis=0)

              blood_fig = int(np.floor(bloodi / float(per_panel))) # int(bloodi <= int(len(render_blood_vars) / 2))
              render_c_s["Biomarker_type"].append("blood_{}".format(blood_fig))
              render_c_s["Biomarker"].append(simple_text_name)
              render_c_s["Quartile"].append(q_name)
              render_c_s["Value"].append(blood_age_avg)
              render_c_s["Age"].append(age_mid)
              render_c_s["Count"].append(blood_age_bool.sum())

              if not simple_text_name in max_vals:
                max_vals[simple_text_name] = blood_age_avg
              else:
                max_vals[simple_text_name] = max(max_vals[simple_text_name], blood_age_avg)

              stds["y0"]["{} {}".format(simple_text_name, q_name)].append(blood_age_avg + blood_age_std)
              stds["y1"]["{} {}".format(simple_text_name, q_name)].append(blood_age_avg - blood_age_std)
              stds["x"]["{} {}".format(simple_text_name, q_name)].append(age_mid)
              stds["color_ind"][simple_text_name] = bloodi % per_panel

              #if blood_fig == 1:
              #  stds["color_ind"][simple_text_name] = bloodi
              #else:
              #  stds["color_ind"][simple_text_name] = bloodi - (int(len(render_blood_vars) / 2) + 1)

      for i in range(len(render_c_s["Value"])):
        render_c_s["Value"][i] = render_c_s["Value"][i] / max_vals[render_c_s["Biomarker"][i]]

      print("max_vals", list(max_vals.keys()))
      for k in ["x", "y0", "y1"]:
        for simple_text_name_q, l in stds[k].items():
          if k == "x":
            stds_norm[k][simple_text_name_q] = np.array(l)
          else:
            simple_text_name = " ".join(simple_text_name_q.split(" ")[:-1])
            stds_norm[k][simple_text_name_q] = np.array(l) / max_vals[simple_text_name]

      render_c_s = dict(render_c_s)
      render_c_s_df = pd.DataFrame(render_c_s)

      for biomarker_type, num_fields in [("bb", all_vars["bb_flat"].shape[1]),
                                         ("bbc", all_vars["bbc"].shape[1]),
                                         ("blood_0", per_panel),
                                         ("blood_1", per_panel),
                                         ("blood_2", len(render_blood_vars) - 2 * per_panel)]:
        sns.set_theme(style='white')  # white
        fig, axarr = plt.subplots(2, figsize=(6, 10), height_ratios=[6, 1]) # sharex=True

        df_curr = render_c_s_df[render_c_s_df["Biomarker_type"] == biomarker_type]

        print("num fields ours and true", biomarker_type, num_fields, df_curr["Biomarker"].unique().shape)
        palette = sns.color_palette("Paired", num_fields)
        #if num_fields == 13:
        #  palette[-1] = (0.5, 0.5, 0.5) # repeat
        ax = sns.lineplot(ax=axarr[0], data=df_curr, x='Age', y='Value', style='Quartile', hue='Biomarker', palette=palette, markers=False) # muted
        ax.get_legend().set_title(None)

        #print("color palette length", len(palette))
        for biomarker in df_curr["Biomarker"].unique():
          for qname in plot_quartiles:
            biomarker_q = "{} {}".format(biomarker, qname)
            axarr[0].fill_between(stds_norm["x"][biomarker_q], stds_norm["y0"][biomarker_q],  stds_norm["y1"][biomarker_q],
                                  color=palette[stds["color_ind"][biomarker]], alpha=0.1)

        axarr[0].spines[['right', 'top']].set_visible(False)

        sns.barplot(ax=axarr[1], ci=None, palette=sns.color_palette("Paired", len(plot_quartiles)),
                            data=df_curr, x="Age", y="Count", hue="Quartile")
        axarr[1].invert_yaxis()
        axarr[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        #axarr[1].tick_params(axis='both', which='major', labelbottom=True, bottom=False, top=False, labeltop=False)
        axarr[1].spines[['right', 'bottom']].set_visible(False)
        axarr[1].set(xticklabels=[])
        axarr[1].set(xlabel=None)

        ftitle = "{}, {}".format(c, sex_dict[s])
        ax.set_title(ftitle)  # fontsize=50
        #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        sns.move_legend(ax, "lower left")

        plt.tight_layout()
        ax.figure.savefig(os.path.join(root_dir, "analysis/part_2_traj_{}_{}_{}.png".format(ci, s, biomarker_type)))
        ax.figure.savefig(os.path.join(root_dir, "analysis/part_2_traj_{}_{}_{}.eps".format(ci, s, biomarker_type)), format="eps")
        plt.close('all')


# ------------------------------------------------------------------
# Kaplan meier
# ------------------------------------------------------------------

if "survival_curves" in args.run_types:
  sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                               "ytick.labelsize": 14})

  for shortform in [False]:
    collected_data = feather.read_feather(os.path.join(root_dir, "analysis/collected_data_{}.feather".format(args.fmodel)))
    p_vals = []
    print("Doing survival curves {} {}".format(shortform, conditions))

    avg_ages = defaultdict(list)
    for ci, c in enumerate(conditions):
      observed_c = collected_data["{} cond_observed".format(c[0])].to_numpy()
      diag_age_c = collected_data["{} cond_diag_age".format(c[0])].to_numpy()
      log_hazard_q_c = collected_data["{} log hazard q".format(c[0])].to_numpy()

      if shortform:
        fig, ax = plt.subplots(1, figsize=(3, 3)) #3, 3
      else:
        fig, ax = plt.subplots(1, figsize=(7.5, 7.5)) #6, 6

      kmfs = []
      for q in range(4):
        kmf = KaplanMeierFitter()
        q_bool = log_hazard_q_c == q
        kmf.fit(durations=diag_age_c[q_bool], event_observed=observed_c[q_bool], label="OnsetNet Q{}".format(q + 1))
        kmf.plot_survival_function(ax=ax, at_risk_counts=False, loc=slice(0., 90.))

        print((~np.isfinite(diag_age_c[q_bool])).sum())
        avg_set = diag_age_c[q_bool][observed_c[q_bool].astype(bool)]
        avg_ages[q].append((avg_set.mean(), avg_set.shape, c))

        leg = ax.get_legend()
        if not shortform:
          #leg.set_bbox_to_anchor((0.65, 0.8))
          pass
        else:
          if c[0] == "All cause mortality":
            leg.set_bbox_to_anchor((0.65, 0.9)) # 0.2 too left, 0.5 too right. it's by where the centre is
          else:
            leg.remove()

        kmfs.append(kmf)

      results = multivariate_logrank_test(event_durations=diag_age_c, groups=log_hazard_q_c, event_observed=observed_c)

      def split_name(cname):
        if shortform and "/" in cname:
          return "/\n".join(cname.split("/"))
        elif shortform and cname in ["Gynecological problems"]:
          splt = cname.split(" ")
          fst = " ".join(splt[:1])
          snd = " ".join(splt[1:])
          return fst + "\n" + snd
        elif shortform and cname in ["All cause morbidity", "Chronic ischemic heart disease", "Diabetes mellitus non-T1",
                       "Lipid metabolism disorders"]:
          splt = cname.split(" ")
          fst = " ".join(splt[:2])
          snd = " ".join(splt[2:])
          return fst + "\n" + snd
        else:
          return cname

      if not shortform:
        add_at_risk_counts(kmfs[0], kmfs[1], kmfs[2], kmfs[3], ax=ax)
        print("Adding title: {} {} {}".format(ci, c, split_name(c[0])))
        plt.title(
          "{} (log rank p-val {})".format(split_name(c[0]),  # np.format_float_scientific(results.test_statistic, precision=2, exp_digits=1),
                             np.format_float_scientific(results.p_value, precision=2, exp_digits=1)),
          loc='right')
      else:
        ax.xaxis.get_label().set_visible(False)
        plt.text(0.02, 0.04, "{}\nlog rank p-val {}".format(split_name(c[0]),  np.format_float_scientific(results.p_value, precision=2, exp_digits=1)), transform=ax.transAxes)

        #plt.title("{}\np-val {}".format(c[0],  np.format_float_scientific(results.p_value, precision=2, exp_digits=1)), loc='right')

      p_vals.append(results.p_value)

      ax.spines[['right', 'top']].set_visible(False)

      plt.tight_layout()
      fig.savefig(os.path.join(root_dir, "analysis/part_2_curves_{}_{}.png".format(shortform, ci)))
      fig.savefig(os.path.join(root_dir, "analysis/part_2_curves_{}_{}.eps".format(shortform, ci)), format="eps")
      plt.close('all')

    print(p_vals)
    p_vals = np.array(p_vals)
    print("p-val survival: ", p_vals.mean(), p_vals.std(), p_vals.min(), p_vals.max())

    for q in range(4):
      sorted_avgs = sorted(avg_ages[q], key=lambda t: t[0])
      print("Sorted for q", q)
      print(sorted_avgs)

# ---------------------------------
# Summary stats
# ---------------------------------

if "summary_stats" in args.run_types:
  suff = "q"

  conditions_df = feather.read_feather(
    os.path.join(root_dir, "analysis/conditions_ordered_{}.feather".format(args.fmodel)))
  conditions_df = list(conditions_df["condition_group"].to_numpy())
  print("R conditions ordered")
  print(conditions_df)

  blood_ind = []
  for bf in render_blood_vars:
    blood_ind.append(BLOOD_fields.index((bf, None)))
  blood_ind = np.array(blood_ind)

  with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
    nn_data = pickle.load(f)

  # one plot per demog and bb and bbc and blood variable. Discrete -> bar chart, continuous -> violin
  # (color = dataset)

  demog_varnames = ["sex", "race", "education", "townsend_deprivation"]
  data_varnames = ["bb", "bbc", "blood"]
  discrete_dict = {"sex": ["Male", "Female"],
                   "race": ["White", "Other", "Asian or Asian British", "Mixed", "Black or Black British"],
                   "education": ["College or University degree",
                    "A levels/AS levels or equivalent",
                    "O levels/GCSEs or equivalent",
                    "CSEs or equivalent",
                    "NVQ or HND or HNC or equivalent",
                    "Other professional qualifications eg: nursing, teaching",
                    "None of the above",
                    "Prefer not to answer",],
                   "alcohol": ["Daily or almost daily", "Three or four times a week", "Once or twice a week", "One to three times a month", "Special occasions only", "Never", "Prefer not to answer"],
                   "smoking": ["Prefer not to answer", "Never", "Previous", "Current"]
  }

  dname_dict = {"int_trainval": "TrainVal", "int_test": "Internal test", "ext_test": "External test"}

  figure_ind = 0

  # demog
  curr_field_ind = 0
  for field in demog_varnames:
    print("Doing", field)
    all_df = defaultdict(list)

    print_field = field
    if field in render_text_subfields:
      print_field = render_text_subfields[field]

    shape = -1
    for demog_name, demog_shape in DEMOG_fields: # find shape
      if demog_name == field:
        shape = demog_shape
        break
    assert shape != -1

    if shape is None: # continuous demog
      curr_field_end = curr_field_ind + 1
    else: # discrete demog
      curr_field_end = curr_field_ind + shape

    for dname in ["int_trainval", "int_test", "ext_test"]:  # not normed
      pretty_name = dname_dict[dname]

      curr_demog_var = nn_data[dname]["demog"][:, curr_field_ind:curr_field_end]
      print(dname, curr_demog_var.shape)

      if shape is None:
        for indiv_i in range(curr_demog_var.shape[0]):
          all_df["Dataset"].append(pretty_name)
          append_val = curr_demog_var[indiv_i].squeeze()
          assert append_val.shape == ()
          all_df[print_field].append(float(append_val))

      else:
        assert np.logical_or(curr_demog_var == 0, curr_demog_var == 1).all()
        for val_i, valname in enumerate(discrete_dict[field]):
          all_df["Dataset"].append(pretty_name)
          all_df["Value"].append(valname)
          all_df["Percentage (\%)"].append(curr_demog_var[:, val_i].sum() / float(curr_demog_var.shape[0]))

    curr_field_ind = curr_field_end

    df = pd.DataFrame(all_df)
    print(df.head())
    print(df.dtypes)

    if not (shape is None): # discrete
      hue_plot_params = {
        'data': df,
        'y': "Percentage (\%)",
        'x': "Value",
        "hue": "Dataset",
        "palette": "muted"
      }
      plot = sns.barplot(**hue_plot_params) # ci=None,
    else:
      hue_plot_params = {
        'data': df,
        'y': print_field,
        'x': "Dataset",
        "palette": "muted"
      }
      plot = sns.violinplot(**hue_plot_params)

    plot.set_xticklabels(plot.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
    plt.tight_layout()
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.png".format(figure_ind)))
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.eps".format(figure_ind)), format="eps")
    plt.clf()
    plt.close('all')
    figure_ind += 1

  # BB
  curr_bbi = 0
  for bbf, bb_shape in BB_fields:
    all_df = defaultdict(list)

    print_field = bbf
    if bbf in render_text_subfields:
      print_field = render_text_subfields[bbf]

    if bb_shape is not None:
      curr_bbi_end = curr_bbi + bb_shape
      if "age" in bbf:
        raise ValueError  # age always scalar

    else:
      curr_bbi_end = curr_bbi + 1
      if "age" in bbf:
        curr_bbi = curr_bbi_end
        continue

    for dname in ["int_trainval", "int_test", "ext_test"]:  # not normed
      pretty_name = dname_dict[dname]
      curr_demog_var = nn_data[dname]["bb"][:, curr_bbi:curr_bbi_end]

      if bb_shape is None:
        for indiv_i in range(curr_demog_var.shape[0]):
          all_df["Dataset"].append(pretty_name)
          append_val = curr_demog_var[indiv_i].squeeze()
          assert append_val.shape == ()
          all_df[print_field].append(float(append_val))

      else:
        assert np.logical_or(curr_demog_var == 0, curr_demog_var == 1).all()
        for val_i, valname in enumerate(discrete_dict[bbf]):
          all_df["Dataset"].append(pretty_name)
          all_df["Value"].append(valname)
          all_df["Percentage (\%)"].append(curr_demog_var[:, val_i].sum() / float(curr_demog_var.shape[0]))

    curr_bbi = curr_bbi_end

    df = pd.DataFrame(all_df)
    print(df.head())
    print(df.dtypes)

    if not (bb_shape is None): # discrete
      hue_plot_params = {
        'data': df,
        'y': "Percentage (\%)",
        'x': "Value",
        "hue": "Dataset",
        "palette": "muted"
      }
      plot = sns.barplot(**hue_plot_params) # ci=None,
    else:
      hue_plot_params = {
        'data': df,
        'y': print_field,
        'x': "Dataset",
        "palette": "muted"
      }
      plot = sns.violinplot(**hue_plot_params)

    plot.set_xticklabels(plot.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
    plt.tight_layout()
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.png".format(figure_ind)))
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.eps".format(figure_ind)), format="eps")
    plt.clf()
    plt.close('all')
    figure_ind += 1

  # BBC
  for trait_i, trait_name in enumerate(channels):
    all_df = defaultdict(list)

    for dname in ["int_trainval", "int_test", "ext_test"]:
      pretty_name = dname_dict[dname]
      curr_demog_var = nn_data[dname]["bbc"][:, trait_i]

      for indiv_i in range(curr_demog_var.shape[0]):
        all_df["Dataset"].append(pretty_name)
        append_val = curr_demog_var[indiv_i].squeeze()
        assert append_val.shape == ()
        all_df[trait_name].append(float(append_val))

    df = pd.DataFrame(all_df)
    print(df.head())
    print(df.dtypes)

    hue_plot_params = {
      'data': df,
      'y': trait_name,
      'x': "Dataset",
      "palette": "muted"
    }
    plot = sns.violinplot(**hue_plot_params)

    plot.set_xticklabels(plot.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
    plt.tight_layout()
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.png".format(figure_ind)))
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.eps".format(figure_ind)), format="eps")
    plt.clf()
    plt.close('all')
    figure_ind += 1

  # blood
  curr_i = 0
  for trait_name, trait_shape in BLOOD_fields:
    assert trait_shape is None

    if "age" in trait_name:
      continue

    all_df = defaultdict(list)
    print_field = render_blood_vars_names[trait_name]

    for dname in ["int_trainval", "int_test", "ext_test"]:
      pretty_name = dname_dict[dname]
      curr_demog_var = nn_data[dname]["blood"][:, curr_i]

      for indiv_i in range(curr_demog_var.shape[0]):
        all_df["Dataset"].append(pretty_name)
        append_val = curr_demog_var[indiv_i].squeeze()
        assert append_val.shape == ()
        all_df[print_field].append(float(append_val))

    df = pd.DataFrame(all_df)
    print(df.head())
    print(df.dtypes)

    hue_plot_params = {
      'data': df,
      'y': print_field,
      'x': "Dataset",
      "palette": "muted"
    }
    plot = sns.violinplot(**hue_plot_params)

    plot.set_xticklabels(plot.get_xticklabels(), rotation=(-30), ha="left", rotation_mode='anchor')
    plt.tight_layout()
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.png".format(figure_ind)))
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_2_summary_stat_{}.eps".format(figure_ind)), format="eps")
    plt.clf()
    plt.close('all')
    figure_ind += 1

    curr_i += 1