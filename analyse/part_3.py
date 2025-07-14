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
from scipy.stats import entropy

from ..util import *
from ..nn_util import *
from .util import *

parser = argparse.ArgumentParser()
# nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None
# nn_demog_bb_bbc_blood_1e-05_1e-05_0.0_512_0_None
parser.add_argument('--fmodel', type=str, default="nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None")
parser.add_argument('--run_types', type=str, nargs="+", default=[])

args = parser.parse_args()

device = "cpu"

def shorten_model(x):
  return x.replace("_", "")

fname = os.path.join(root_dir, "models", args.fmodel + ".pgz")
print(fname)

sns.set_theme(style='white', rc={'figure.figsize':(18, 14)}) # 16, 13
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

with gzip.open(fname, "rb") as f:
  results = pickle.load(f)

# run all instance 2 data through model, get abs partial derivative per disease for men and women

model_args = results["args"]
int_trainvals, int_test, ext_test, conditions = get_nn_data(model_args, cond=True)

train = int_trainvals["train"][0] # all folds have same data
val = int_trainvals["val"][0]
model = results["best_model"].to(device)
model.eval()

sex_ind_start = DEMOG_fields.index(("sex", 2))
assert sex_ind_start == 0
saliency_types = ["bb", "bbc", "blood"]
if "sbc" in args.fmodel:
  saliency_types.append("sbc")

if "first_run" in args.run_types:
  feat_sz = None
  log_hazards = []
  ids = []
  sex = []
  grad_all = {0: defaultdict(list), 1: defaultdict(list)}
  sbc_grad_all = {0: defaultdict(list), 1: defaultdict(list)}

  for dl_i, dl in enumerate([train, val, int_test, ext_test]):
    for batch_idx, data in enumerate(dl):
      print("dl {} batch {}/{} {}".format(dl_i, batch_idx, len(dl), datetime.now()))
      sys.stdout.flush()

      data = tuple([d.to(device) for d in data])

      for d in data:
        d.requires_grad = True

      sex_curr = data[nn_data_types.index("demog")][:, sex_ind_start + 1] # 0 for male, 1 for female

      for sex in range(2):
        sex_i = sex_curr == sex
        data_s = tuple([d[sex_i] for d in data])

        for ci in range(num_conditions):
          data_s_ci = data_s
          model.zero_grad()

          log_hazards_curr = model("log_hazards", data_s_ci)
          log_hazards_ci = log_hazards_curr[:, ci]
          loss = - log_hazards_ci.sum() # increase the hazard for everyone
          # optimise to increase the risk
          grad, sbc_grad = [], None
          for nn_input in saliency_types:
            curr_grad = torch.autograd.grad(loss, data_s_ci[nn_data_types.index(nn_input)], retain_graph=(nn_input != saliency_types[-1]))[0]
            if nn_input != "sbc":
              grad.append(curr_grad.flatten(start_dim=1).abs().detach().cpu())
            else:
              sbc_grad = curr_grad.abs().detach().cpu() # n, ch, len

          if feat_sz is None:
            feat_sz = [g.shape[1] for g in grad]
          grad = torch.cat(grad, dim=1) # concatenate across dimensions. n, num_feat
          assert grad.shape == (log_hazards_ci.shape[0], sum(feat_sz))
          grad = torch.divide(grad, grad.sum(dim=1, keepdim=True)) # average abs partial deriv
          assert grad.shape == (log_hazards_ci.shape[0], sum(feat_sz))
          grad[torch.logical_not(torch.isfinite(grad))] = 0
          grad_all[sex][ci].append(grad)

          if "sbc" in args.fmodel:
            sbc_grad = torch.divide(sbc_grad, sbc_grad.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)) # divide by total sbc grad
            sbc_grad[torch.logical_not(torch.isfinite(sbc_grad))] = 0
            sbc_grad_all[sex][ci].append(sbc_grad)

  print(feat_sz)
  grads_processed = {0:[], 1:[]}
  sbc_grads_processed = {0:[], 1:[]}
  for sex in range(2):
    cols = defaultdict(list)
    for ci in range(num_conditions):
      grad_ci_sex = torch.cat(grad_all[sex][ci], dim=0).numpy()
      grad_ci_sex = grad_ci_sex.mean(axis=0)
      assert grad_ci_sex.shape == (sum(feat_sz),)
      grads_processed[sex].append(grad_ci_sex)

      if "sbc" in args.fmodel:
        sbc_grad_ci_sex = torch.cat(sbc_grad_all[sex][ci], dim=0).numpy()
        sbc_grad_ci_sex = sbc_grad_ci_sex.mean(axis=0)
        assert sbc_grad_ci_sex.shape == (len(channels), graph_len)
        sbc_grads_processed[sex].append(sbc_grad_ci_sex)

  with gzip.open(os.path.join(root_dir, "data/part_3_grads_{}.gz".format(args.fmodel)), "wb") as f:
    pickle.dump({"grads_processed": grads_processed, "sbc_grads_processed": sbc_grads_processed}, f)


  exit(0)

if "render_grads" in args.run_types:
  for top_k in [5, 10]:
    with gzip.open(os.path.join(root_dir, "data/part_3_grads_{}.gz".format(args.fmodel)), "rb") as f:
      saved_grads = pickle.load(f)
      grads_processed = saved_grads["grads_processed"]
      sbc_grads_processed = saved_grads["sbc_grads_processed"]

    # non-sbc grads
    print("Starting")
    sys.stdout.flush()
    f_keep = {}
    curr_i = 0
    for f in saliency_types:
      if not f == "sbc":
        f_desc = {"bb": BB_fields, "bbc":  list(zip(channels, [None] * len(channels))), "blood": BLOOD_fields}[f]
        for fsubname, fsubshape in f_desc:
          fsub_len = 1
          if not (fsubshape is None):
            fsub_len = fsubshape

          for i in range(fsub_len):
            if not ("age" in fsubname):
              if fsub_len == 1:
                f_keep[curr_i] = r"{}".format(simple_text_names[fsubname])
              else:
                f_keep[curr_i] = r"{}$_{}$".format(simple_text_names[fsubname], i)
            else:
              f_keep[curr_i] = None
            curr_i += 1

    print("keeping feature inds")
    print(f_keep)
    sys.stdout.flush()

    sex_excl = {0: [], 1:[]}
    for sex in range(2):
      cols = defaultdict(list)
      annot = defaultdict(list)
      avg = []
      for ci in range(num_conditions + 1):
        if ci < num_conditions:
          cond_name = r"{}".format(conditions[ci][0])
        else:
          cond_name = "Average"

        cols["Condition"].append(cond_name)
        annot["Condition"].append(cond_name)

        if str(cond_name) in sex_conds[1 - sex]:  # belongs to other sex
          print("found member", cond_name, sex)
          sex_excl[sex].append(ci)
          for i in range(grads_processed[sex][0].shape[0]): # go through biomarker columns
            if not (f_keep[i] is None):
              cols[f_keep[i]].append(0)
              annot[f_keep[i]].append("")

        else:
          if ci < num_conditions: # add to data and average
            grad_ci_sex = grads_processed[sex][ci] # vector of saliency
            print("should sum to 1 approx", grad_ci_sex.sum())
            avg.append(grad_ci_sex)
          else:
            grad_ci_sex = np.stack(avg, axis=0).mean(axis=0)

          top = np.argsort(grad_ci_sex)[-top_k:]
          for i in range(grad_ci_sex.shape[0]): # go through biomarker columns
            if not (f_keep[i] is None):
              cols[f_keep[i]].append(grad_ci_sex[i]) # add to biomarker column if chosen

              if i in top:
                annot[f_keep[i]].append("{:.2f}".format(grad_ci_sex[i]).lstrip("0"))
              else:
                annot[f_keep[i]].append("")

      all_conds_df = pd.DataFrame(dict(cols)) #columns=["Condition"] + [str(i) for i in range(sum(feat_sz))]
      all_conds_df.set_index("Condition", drop=True, inplace=True)

      annot_df = pd.DataFrame(dict(annot))
      annot_df.set_index("Condition", drop=True, inplace=True)
      sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                                   "ytick.labelsize": 14})

      _, axarr = plt.subplots(1, figsize=(19.25, 14))
      plot = sns.heatmap(all_conds_df, cmap="vlag", annot=annot_df, fmt='', xticklabels=1, yticklabels=1, square=True, cbar=True, ax=axarr,
                         vmin=0, vmax=0.2)
      #plot.set_xticklabels(plot.get_xticklabels(), rotation=(-40), ha="left", rotation_mode='anchor')
      #plot.set_yticklabels(plot.get_yticklabels(), rotation=(-30), ha="right", rotation_mode='anchor')

      plot.axhline(47, color="dimgray", linestyle="--")  # end of this one
      plot.set(xlabel="Trait")
      plt.tight_layout()

      plot.figure.savefig(os.path.join(root_dir, "analysis/part_3_{}_{}_top{}.png".format(shorten_model(args.fmodel), sex, top_k)))
      plot.figure.savefig(os.path.join(root_dir, "analysis/part_3_{}_{}_top{}.eps".format(shorten_model(args.fmodel), sex, top_k)), format='eps')

      plt.clf()
      plt.close('all')

  # sbc grads
  with gzip.open(os.path.join(root_dir, "data/graph_data.gz"), "rb") as f:
    graph_data = pickle.load(f)
  avg_landmarks = graph_data["avg_landmarks"]
  tissue_lengths = graph_data["tissue_lengths"]
  offsets = graph_data["offsets"]

  print(avg_landmarks)
  print(tissue_lengths)
  print(offsets)
  landmark_names = ["midthigh", "hip", "T12", "shoulder"]

  if "sbc" in args.fmodel:
    print("Doing sbc grads")
    sys.stdout.flush()
    cond_names_sq = [c[0] for c in conditions] + ["Average"]

    for sex in range(2):
      sbc_res = defaultdict(list)
      for ci in range(num_conditions):
        cond_name = r"{}".format(conditions[ci][0])
        if cond_name in sex_conds[1 - sex]: # belongs to other sex
          for ch_i, ch in enumerate(channels):
            sbc_res[ch].append(np.zeros(sbc_grads_processed[sex][0][ch_i, :].shape))
        else:
          for ch_i, ch in enumerate(channels):
            sbc_res[ch].append(sbc_grads_processed[sex][ci][ch_i, :])

      all_avgs = []
      for ch_i, ch in enumerate(channels):
        fig, axarr = plt.subplots(1, figsize=(20, 12))
        np_res_ch = np.stack(sbc_res[ch], axis=0) # cond, len
        avg_ch = np_res_ch.mean(axis=0, keepdims=True)
        np_res_ch = np.concatenate([np_res_ch, avg_ch], axis=0)
        print(sex, ch, np_res_ch.shape)
        assert np_res_ch.shape == (num_conditions + 1, graph_len)
        sys.stdout.flush()

        # tmat and vat
        np_res_ch[:, :offsets[ch_i]] = 0
        np_res_ch[:, (offsets[ch_i] + tissue_lengths[ch_i]):] = 0

        avg_ch[:, :offsets[ch_i]] = 0
        avg_ch[:, (offsets[ch_i] + tissue_lengths[ch_i]):] = 0
        all_avgs.append(avg_ch)

        sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                                     "ytick.labelsize": 14})
        plot = sns.heatmap(np_res_ch, cmap="rocket_r", xticklabels=1, yticklabels=cond_names_sq, ax=axarr,
                           vmin=0.0, vmax=0.003)
        plot.set(xticklabels=[])

        # draw vertical lines at landmarks, remove offset
        for li, landmark_x in enumerate(avg_landmarks[1:-1]):
          axarr.axvline(landmark_x, color="dimgray", linestyle="--")  # end of this one
          #axarr.text(x=(landmark_x + 0.1), y=0, s=landmark_names[li], rotation=30)

        axarr.axhline(47, color="dimgray", linestyle="--")  # end of this one
        axarr.set(xlabel="Height", ylabel="Condition")

        #print(cond_names_sq)
        #plot.set_yticks(cond_names_sq)
        plt.tight_layout()
        fig.savefig(os.path.join(root_dir, "analysis/part_3_sbc_{}_{}_{}.png".format(shorten_model(args.fmodel), sex, ch)))
        fig.savefig(os.path.join(root_dir, "analysis/part_3_sbc_{}_{}_{}.eps".format(shorten_model(args.fmodel), sex, ch)), format='eps')

        plt.clf()
        plt.close('all')

      # render average spatial
      avg_np = np.concatenate(all_avgs, axis=0)
      print(avg_np.shape)
      fig, axarr = plt.subplots(1, figsize=(6, 4))
      sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                                   "ytick.labelsize": 14})
      plot = sns.heatmap(avg_np, cmap="rocket_r", xticklabels=1, yticklabels=channels, ax=axarr, cbar=True,
                         vmin=0.0, vmax=0.002)
      plot.set(xticklabels=[])
      for li, landmark_x in enumerate(avg_landmarks[1:-1]):
        axarr.axvline(landmark_x, color="dimgray", linestyle="--")  # end of this one
        #axarr.text(x=(landmark_x + 0.1), y=0, s=landmark_names[li], rotation=30)
      plot.set(xlabel="Height", ylabel="Body Composition")
      plt.tight_layout()
      plot.figure.savefig(os.path.join(root_dir, "analysis/part_3_sbc_{}_{}_avg.png".format(shorten_model(args.fmodel), sex)))
      plot.figure.savefig(os.path.join(root_dir, "analysis/part_3_sbc_{}_{}_avg.eps".format(shorten_model(args.fmodel), sex)), format='eps')

      plt.clf()
      plt.close('all')