import glob
import numpy as np, scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from select import select
from statannotations.Annotator import Annotator
import traceback
import argparse
import torch
from scipy import stats
import pyarrow.feather as feather

from ..consts import *
from ..util import *
from ..nn_util import *
from .util import *


def legend_without_duplicate_labels(ax, loc="upper right"):
  handles, labels = ax.get_legend_handles_labels()
  unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
  ax.legend(*zip(*unique), loc=loc, fontsize=14)


def generate_exact_mean_std_sample(desired_mean, desired_std_dev, num_samples):
  samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples)
  actual_mean = np.mean(samples)
  zero_mean_samples = samples - (actual_mean)

  zero_mean_std = np.std(zero_mean_samples)

  scaled_samples = zero_mean_samples * (desired_std_dev / zero_mean_std)

  final_samples = scaled_samples + desired_mean
  final_mean = np.mean(final_samples)
  final_std = np.std(final_samples)
  print("Final samples stats : mean = {}, {}, stdv = {}, {}".format(final_mean, desired_mean, final_std, desired_std_dev))
  return final_samples

# -------------------------------------------------------------------------------------------------------------------

model_inputs_list = [["demog"], ["demog", "bb"],
                     ["demog", "bb", "bbc"],
                     ["demog", "bb", "sbc"], ["demog", "bb", "blood"],
                     ["demog", "bb", "bbc", "blood"], ["demog", "bb", "sbc", "blood"],
                     ["demog", "bb", "bbc", "sbc", "blood"]]

top3_names = ["demog+bb+bbc+blood", "demog+bb+sbc+blood", "demog+bb+bbc+sbc+blood"]

num_nn_folds = 5
num_linear_sim_folds = 100

# -------------------------------------------------------------------------------------------------------------------

set_seed(1)
#root_dir = "/home/mica/storage/healthspan/v4.2"

print("Root dir {}".format(root_dir))

parser = argparse.ArgumentParser()
parser.add_argument('--run_types', type=str, nargs="+", default=[]) # "first_run"
parser.add_argument('--sing_cond', type=str, nargs="+", default=[]) # for model set
parser.add_argument('--best_age_model', type=str, default="nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None")
args = parser.parse_args()
print(args)

print("model_inputs_list orig")
print(model_inputs_list)

hue_order = []
for model_inputs in model_inputs_list:
  input_type = "+".join(model_inputs)  # comparison group
  hue_order.append(input_type)

sns.set(font_scale=8) # 3.5 original
sns.set_theme(style='ticks') # white
# https://www.practicalpythonfordatascience.com/ap_seaborn_palette
palette = sns.color_palette(palette="icefire", n_colors=len(model_inputs_list)) # set2. gist_stern_r, Spectral, n_colors=len(cg_order), icefire
print(palette)

results_types = ["Internal test", "External test", "Train", "Val"] # ["Internal test"] #
select_type = "Val" # for selection of results
indices = {"Train": 0, "Val": 1, "Internal test": 2, "External test": 3} # saving order in train_nn.py
select_index = indices[select_type]

with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
  nn_data = pickle.load(f)
condition_names = nn_data["int_trainval_norm"]["conditions"]
condition_counts_dict = {}
condition_counts_dict["Train"] = nn_data["int_trainval"]["observed"].sum(axis=0) # trainval
assert condition_counts_dict["Train"].shape == (num_conditions,)
condition_counts_dict["Val"] = nn_data["int_trainval"]["observed"].sum(axis=0) # trainval
condition_counts_dict["Internal test"] = nn_data["int_test"]["observed"].sum(axis=0)
condition_counts_dict["External test"] = nn_data["ext_test"]["observed"].sum(axis=0)
print(condition_names)
print(condition_counts_dict)
print("--------")

# -------------------------------------------------------------------------------------------------------------------
all_best_select_cindex = []
for _ in range(num_conditions + 1):
  all_best_select_cindex.append({})
  for model_inputs in model_inputs_list:
    input_type = "+".join(model_inputs)
    all_best_select_cindex[-1][input_type] = -np.inf

all_best_select_model_name = [{} for _ in range(num_conditions)]
all_best_fold_results = [{} for _ in range(num_conditions)]
best_val_fold = [{} for _ in range(num_conditions)]
all_print_res = {}

if "first_run" in args.run_types:
  print("Selecting results type {} model based on table_type {}: ".format(results_types, select_type))
  skipped = []
  input_type_counts = defaultdict(int)
  num_params = []
  for model_inputs in model_inputs_list:
    input_type = "+".join(model_inputs)  # comparison group
    nn_versions = [0]
    for nn_version in nn_versions:
      for lr in [0.00005, 0.00001, 0.0001]:
        for weight_decay in [0.00005, 0.00001, 0.0001]: # [0.00001]: #
          for dropout in [0.3, 0.0, 0.5]:
            for feat_sz in [512, 256, 128]:
              fname = "{}/models/nn_{}_{}_{}_{}_{}_{}_{}.pgz".format(root_dir, "_".join(model_inputs), lr,
                                                                    weight_decay, dropout, feat_sz,
                                                                    nn_version, pretty(args.sing_cond))
              split_name = fname.split("/")
              model_name = split_name[-1].split(".pgz")[0]

              try:
                with gzip.open(fname, "rb") as f:
                  results = pickle.load(f)
              except Exception as e:
                if "nn_demog_bb_bbc_blood_5e-05_0.0001_0.3_256_0_None" in fname:
                  print("Caught exception for model")
                  print(traceback.format_exc())

                print(e)
                skipped.append(model_name)
                continue

              if results["cross_val_results"] is not None:
                model = results["best_model"]
                num_params.append(count_params(model))

                input_type_counts[input_type] += 1
                print("Doing {}".format(model_name))
                cross_val_results = results["cross_val_results"]
                args_curr = results["args"]
                best_fold_val = results["best_fold"]
                assert args_curr.crossval_folds == num_nn_folds

                # all folds and disease
                select_curr = np.zeros((args_curr.crossval_folds, num_conditions))
                results_curr = np.zeros((len(results_types), args_curr.crossval_folds, num_conditions))

                for f in range(args_curr.crossval_folds):
                  best_epoch = results["epochs"][f] # this is based on best overall performance, we keep 1 best model overall
                  select_curr[f, :] = cross_val_results[f][indices[select_type]][best_epoch]["c_index_condition"]

                  for ri, results_type in enumerate(results_types): # on best epoch already
                    if results_type in ["Train", "Val"]:
                      results_curr[ri, f, :] = cross_val_results[f][indices[results_type]][best_epoch]["c_index_condition"]
                    else:
                      results_curr[ri, f, :] = cross_val_results[f][indices[results_type]]["c_index_condition"]

                print_res = {}
                print_res["model_name"] = model_name
                print_res["Best hyperparameters"] = " ".join([str(v) for v in [lr, weight_decay, dropout, feat_sz]])
                for ri, results_type in enumerate(results_types):
                  print_res["{} mean".format(results_type)] = results_curr[ri, :, :].mean()
                  print_res["{} std".format(results_type)] = results_curr[ri, :, :].std()
                  print_res["{} 95% CI".format(results_type)] = st.t.interval(0.95, args_curr.crossval_folds - 1,
                                                                            loc=print_res["{} mean".format(results_type)],
                                                                            scale=print_res["{} std".format(results_type)])

                all_conds_perf = select_curr.mean(axis=1)
                assert best_fold_val == all_conds_perf.argmax()
                select_perf_all = select_curr[best_fold_val, :].mean()
                if select_perf_all > all_best_select_cindex[num_conditions][input_type]: # avg over all conditions, last index
                  all_best_select_cindex[num_conditions][input_type] = select_perf_all
                  all_print_res[input_type] = print_res

                for ci in range(num_conditions):
                  #select_perf = select_curr[:, ci].mean() # best val perf across folds is chosen on per condition basis TODO
                  select_perf = select_curr.mean() # chosen based on all conditions; same model for all conditions
                  if select_perf > all_best_select_cindex[ci][input_type]:
                    all_best_select_cindex[ci][input_type] = select_perf
                    all_best_select_model_name[ci][input_type] = model_name
                    all_best_fold_results[ci][input_type] = results_curr[:, :, ci] # results type, fold, condition
                    best_val_fold[ci][input_type] = results_curr[results_types.index("Val"), :, ci].argmax()
              else:
                print("Skipping")

  print("Skipped")
  print(len(skipped))
  print(skipped)

  print("Input type counts")
  print(input_type_counts)

  print("all_best_select_model cindex name")
  print(all_best_select_cindex)
  print(all_best_select_model_name)

  print("Num params")
  num_params = np.array(num_params)
  print(num_params.shape, num_params.mean(), num_params.std(), num_params.min(), num_params.max())

  # Performance summary table
  res = []
  for input_type in all_print_res:
    res_curr = {}
    res_curr["Inputs"] = input_type
    for k, v in all_print_res[input_type].items():
      res_curr[k] = v
    res.append(res_curr)

  res = pd.DataFrame(res)
  res.to_csv(os.path.join(root_dir, "analysis/part_1_table_{}.csv".format(pretty(args.sing_cond))), sep='\t', encoding='utf-8', index=False, header=True)

  print("Wrote table")
  sys.stdout.flush()

  #print("all_results demog+bb+bbc+blood")
  #print(all_results["demog+bb+bbc+blood"][:, :, select_ind_one_index]) # int test

  with gzip.open(os.path.join(root_dir, "analysis/part_1_{}.gz".format(pretty(args.sing_cond))), "wb") as f:
    pickle.dump({"all_best_select_cindex": all_best_select_cindex,
                 "all_best_select_model_name": all_best_select_model_name,
                 "all_best_fold_results": all_best_fold_results,
                 "best_val_fold": best_val_fold}, f)

# -------------------------------------------------------------------------------------------------------------------

top3 = False

if "performance" in args.run_types:
  with gzip.open(os.path.join(root_dir, "analysis/part_1_{}.gz".format(pretty(args.sing_cond))), "rb") as f:
    res_load = pickle.load(f)
    all_best_fold_results = res_load["all_best_fold_results"]

  for ri, results_type in enumerate(results_types):
    val_min = np.inf
    val_max = -np.inf

    data = []
    best_fold_data = defaultdict(list)
    best_fold_avgs = defaultdict(list)
    all_means = []
    for condition_ind in range(num_conditions):
      cond_name = condition_names[condition_ind][0]
      condition_count = condition_counts_dict[results_type][condition_ind]

      if condition_count > 0:
        for input_type, d in all_best_fold_results[condition_ind].items():
          results_curr = d[ri, :]
          assert results_curr.shape == (5,)

          # store result per fold
          for f in range(results_curr.shape[0]):
            val = results_curr[f]
            data_entry = {"Input type": input_type} # "Condition group ind": group_ind, "Condition group name": group_name
            data_entry["Condition"] = cond_name
            data_entry["{} C-index".format(results_type)] = val
            data_entry["Num. uncensored (log)"] = np.log(condition_count)
            data.append(data_entry)

            data_entry_avg = {"Input type": input_type}
            data_entry_avg["Condition"] = "Average" # all folds all conditions
            data_entry_avg["{} C-index".format(results_type)] = val
            data_entry_avg["Num. uncensored (log)"] = np.log(condition_counts_dict[results_type].mean())
            data.append(data_entry_avg)

          mean_val = results_curr.mean()
          if results_type == "Internal test" and input_type == "demog+bb+bbc+sbc+blood":
            print("Plot results demog+bb+bbc+sbc+blood:", cond_name, mean_val, results_curr.std())
            all_means.append(mean_val)
          val_min = min(val_min, mean_val)
          val_max = max(val_max, mean_val)

          # store best value according to best val fold
          best_fold_data["Input type"].append(input_type)
          best_fold_data["Condition"].append(cond_name)
          best_fold_data["{} C-index".format(results_type)].append(results_curr.max()) #
          best_fold_avgs[input_type].append(results_curr.max()) # results_curr.max()

    if results_type == "Internal test":
      min_i = all_means.index(min(all_means))
      max_i = all_means.index(max(all_means))
      print("min max i", min_i, max_i)
      print(all_means[min_i], all_means[max_i])
      print(condition_names[min_i], condition_names[max_i])

    # average of all for best
    if not top3:
      avg_list = ["+".join(x) for x in model_inputs_list]
    else:
      avg_list = top3_names

    for input_type in avg_list:
      best_fold_data["Input type"].append(input_type)
      best_fold_data["Condition"].append("Average")
      best_fold_data["{} C-index".format(results_type)].append(np.array(best_fold_avgs[input_type]).mean())

    data = pd.DataFrame(data)
    print("Input types found")
    print(pd.unique(data["Input type"]))
    feather.write_feather(data, os.path.join(root_dir, "analysis/part_1_perf_{}.feather".format(results_type)))

    # write averages for inspection
    d_avg = data[data["Condition"] == "Average"].groupby(["Input type", "Condition"]).mean().reset_index()
    d_avg_std = data[data["Condition"] == "Average"].groupby(["Input type", "Condition"]).std().reset_index()
    d_avg = pd.merge(d_avg, d_avg_std, on=["Input type", "Condition"])
    #feather.write_feather(d_avg, os.path.join(root_dir, "analysis/part_1_perf_avg_{}.feather".format(results_type)))
    d_avg.to_csv(os.path.join(root_dir, "analysis/part_1_perf_avg_{}.csv".format(results_type)), sep='\t', encoding='utf-8', index=False, header=True)

    if results_type == "Internal test": # results type
      print("Printing performance differences and p-vals for internal test:")
      d_avg_inttest = data[data["Condition"] == "Average"] # condition
      print(d_avg_inttest.head())
      print(d_avg_inttest.shape)
      print(d_avg_inttest.describe())

      unique_input_types = d_avg_inttest["Input type"].unique()

      def input_type_in(input_type1, input_type2):
        split1 = input_type1.split("+")
        split2 = input_type2.split("+")
        for sp1 in split1:
          if not (sp1 in split2):
            return False
        return True

      def break_plus(x):
        split1 = x.split("+")
        if len(split1) > 3:
          return "+".join(split1[:3]) + " " + "+".join(split1[3:])
        else:
          return x

      num_comp = 0
      for input_type1 in unique_input_types:
        for input_type2 in unique_input_types:
          if input_type_in(input_type1, input_type2) and len(input_type2) > len(input_type1):
            num_comp += 1
      print(num_comp, 0.05 / num_comp)

      for input_type1 in unique_input_types:
        for input_type2 in unique_input_types:
          if input_type_in(input_type1, input_type2) and len(input_type2) > len(input_type1):
            results1 = d_avg_inttest[d_avg_inttest["Input type"] == input_type1]["Internal test C-index"].to_numpy() # 47 conditions, 5 folds = 235 results
            results2 = d_avg_inttest[d_avg_inttest["Input type"] == input_type2]["Internal test C-index"].to_numpy()
            assert results1.shape == (47 * 5,) and results2.shape == (47 * 5,)

            stat_results = stats.ttest_ind(results1, results2)
            padd = ""
            if stat_results.pvalue <= (0.05 / num_comp):
              padd = "\\textsuperscript{**}"
            elif stat_results.pvalue <= 0.05:
              padd = "\\textsuperscript{*}"
            pstring = "{} & {} & {} & {} & {} & {} {} \\\\".format(break_plus(input_type1),
                                                                     np.format_float_scientific(results1.mean(),
                                                                                                precision=3,
                                                                                                exp_digits=1).replace("e", "E"),
                                                                     break_plus(input_type2),
                                                                     np.format_float_scientific(results2.mean(),
                                                                                                precision=3, exp_digits=1).replace("e", "E"),
                                                                     np.format_float_scientific(results2.mean() - results1.mean(), precision=3, exp_digits=1).replace("e", "E"),
                                                                     np.format_float_scientific(stat_results.pvalue, precision=3, exp_digits=1).replace("e", "E"),
                                                                     padd)

            print(pstring)

      print("------")

    print("best_fold_data")

    best_fold_data = pd.DataFrame(best_fold_data)
    best_fold_data["Condition"].astype(str)

    perfsuff = ""
    if top3:
      perfsuff = "top3"
      data = data[data["Input type"].isin(
        top3_names
      )]
      hue_order = [x for x in hue_order if x in top3_names]

    hue_plot_params = {
      'data': data,
      'x': "Condition",
      'y': "{} C-index".format(results_type),
      "order": c_group_order,
      "hue": "Input type",
      "hue_order": hue_order,

      "ci": 95,
      "errcolor": "black",

      "errwidth": 0.2, # todo
      "capsize": 0.1
    }

    if top3:
      hue_plot_params["errwidth"] = 0.4
      hue_plot_params["capsize"] = 0.2

    sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})

    fig, axarr = plt.subplots(2, figsize=(18, 15), sharex=True, height_ratios=[6, 1]) # (20, 16) # 18, 14.4

    plot = sns.barplot(ax=axarr[0], palette=palette, **hue_plot_params) # ci=None,
    legend_without_duplicate_labels(plot)

    y_max_lim = min(val_max + 0.7, 1)
    axarr[0].set_ylim(round(val_min - 0.05, 1), y_max_lim) # 0.15 -> 0.1, 0.19 -> 0.1, 0.2 -> 0.1 or 0.2
    axarr[0].set_xlim(-1, sum(list(cg_counts.values()))) # 0.15 -> 0.1, 0.19 -> 0.1, 0.2 -> 0.1 or 0.2
    axarr[0].spines[['right', 'top']].set_visible(False)
    axarr[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axarr[0].tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=False)

    #annotator = Annotator(axarr[0], pairs, **hue_plot_params)
    #annotator.configure(test="t-test_ind", pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]]).apply_and_annotate() # comparisons_correction="bonferroni")

    counts_data = data[data["Input type"] == top3_names[0]] # any input type
    hue_plot_params2 = {
      'data': counts_data,
      'x': "Condition",
      'y': "Num. uncensored (log)",
      "order": c_group_order,
      "color": "grey"
      #"hue": "Input type",
      #"hue_order": hue_order
    }
    plot2 = sns.barplot(ax=axarr[1], ci=None, **hue_plot_params2)
    plot2.set_xticklabels(plot2.get_xticklabels(), rotation=(90), ha="center") # -30, , rotation_mode='anchor'
    axarr[1].set_xlim(-1, sum(list(cg_counts.values()))) # 0.15 -> 0.1, 0.19 -> 0.1, 0.2 -> 0.1 or 0.2
    axarr[1].invert_yaxis()
    axarr[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axarr[1].tick_params(axis='both', which='major', labelbottom=True, bottom=False, top=False, labeltop=False)
    axarr[1].spines[['right', 'bottom']].set_visible(False)

    curr_ylim = axarr[0].get_ylim()
    print("ylims")
    print(curr_ylim, y_max_lim)

    # draw dashed lines
    curr_cond = 0
    for cgi, cg in enumerate(cg_order): # the order the condition groups are printed
      if cgi == len(cg_order) - 1:
        break

      curr_cond += cg_counts[cg]
      x_vline = curr_cond - 0.5
      axarr[0].axvline(x_vline, color="dimgray", linestyle="--") # end of this one
      print("Plot at " + str(x_vline))
      axarr[1].axvline(x_vline, color="dimgray", linestyle="--") # end of this one

      # text at top #
      axarr[0].text(x=(x_vline + 0.1), y=(curr_ylim[1] + 0.005), s=cg_order[cgi + 1], rotation=90) # next label curr_ylim[1]

    plot.set(xlabel=None)


    plt.tight_layout()
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_1_plot_{}_{}{}.png".format(pretty(results_type), pretty(args.sing_cond), perfsuff)))
    plot.figure.savefig(os.path.join(root_dir, "analysis/part_1_plot_{}_{}{}.eps".format(pretty(results_type), pretty(args.sing_cond), perfsuff)), format="eps")
    plt.clf()
    plt.close('all')

# -------------------------------------------------------------------------------------------------------------------

if "ablation_graph" in args.run_types:
  # plot performance for  demog_bb_bbc_blood nn, full inputs nn, demog_bb_bbc_blood linear. Mean and 95% conf across folds (test_cvar)
  results_types_ablation = [("Internal test", "int_test"), ("External test", "ext_test")]

  if "print" in args.run_types:
    nn_ablation_types = ["demog", "demog+bb",
                         "demog+bb+bbc", "demog+bb+sbc", "demog+bb+blood",
                         "demog+bb+bbc+blood", "demog+bb+sbc+blood",
                         "demog+bb+bbc+sbc+blood"]
  else:
    nn_ablation_types = ["demog+bb+bbc+blood", "demog+bb+bbc+sbc+blood"]

  linear_ablation_type = "demog+bb+bbc+blood"
  hue_order_ablation = ["OnsetNet demog+bb+bbc+blood", "OnsetNet demog+bb+bbc+sbc+blood", "Linear demog+bb+bbc+blood"]

  with gzip.open(os.path.join(root_dir, "analysis/part_1_{}.gz".format(pretty(args.sing_cond))), "rb") as f:
    res_load = pickle.load(f)
    all_best_fold_results = res_load["all_best_fold_results"]

  all_data = {}
  for ri, (results_type, short_type) in enumerate(results_types_ablation):
    #print("Adding results type", results_type)
    val_min = np.inf
    val_max = -np.inf

    data = []
    seen_conds = []
    for condition_ind in range(num_conditions):
      cond_name = condition_names[condition_ind][0]
      condition_count = condition_counts_dict[results_type][condition_ind]
      pretty_name = pretty(cond_name)

      if condition_count > 0:
        seen_conds.append(cond_name)
        for input_type in nn_ablation_types:
          d = all_best_fold_results[condition_ind][input_type]
          results_curr = d[ri, :]
          assert results_curr.shape == (5,)
          input_type_print = "OnsetNet {}".format(input_type)

          # store result per fold
          for f in range(results_curr.shape[0]):
            #if f == 0: print("Adding results type nn", results_type)

            val = results_curr[f]
            data_entry = {"Input type": input_type_print}  # "Condition group ind": group_ind, "Condition group name": group_name
            data_entry["Condition"] = cond_name
            data_entry["{} C-index".format(results_type)] = val
            data_entry["Num. uncensored (log)"] = np.log(condition_count)
            data_entry["Fold"] = f
            data.append(data_entry)

            data_entry_avg = {"Input type": input_type_print}
            data_entry_avg["Condition"] = "Average"  # all folds all conditions
            data_entry_avg["{} C-index".format(results_type)] = val
            data_entry_avg["Num. uncensored (log)"] = np.log(condition_counts_dict[results_type].mean())
            data_entry_avg["Fold"] = f
            data.append(data_entry_avg)

          mean_val = results_curr.mean()
          val_min = min(val_min, mean_val)
          val_max = max(val_max, mean_val)

        # add the linear model and dummy data for variance
        input_type_print = "Linear {}".format(linear_ablation_type)
        linear_res2_fname = os.path.join(root_dir, "analysis/linear_cox/ablate_age_{}_res2_{}.feather".format(pretty_name, linear_ablation_type))
        linear_res2 = feather.read_feather(linear_res2_fname)
        linear_c_index = linear_res2["{}_concordance".format(short_type)].squeeze()
        linear_c_index_var = linear_res2["{}_cvar".format(short_type)].squeeze()
        lindata = generate_exact_mean_std_sample(linear_c_index, np.sqrt(linear_c_index_var), num_linear_sim_folds)

        for f in range(num_linear_sim_folds):
          data_entry = {"Input type": input_type_print}
          data_entry["Condition"] = cond_name
          data_entry["{} C-index".format(results_type)] = lindata[f]
          data_entry["Num. uncensored (log)"] = np.log(condition_count)
          data_entry["Fold"] = f
          data.append(data_entry)

          data_entry_avg = {"Input type": input_type_print}
          data_entry_avg["Condition"] = "Average"  # all folds all conditions
          data_entry_avg["{} C-index".format(results_type)] = lindata[f]
          data_entry_avg["Num. uncensored (log)"] =  np.log(condition_counts_dict[results_type].mean())
          data_entry_avg["Fold"] = f
          data.append(data_entry_avg)

      else:
        data_entry = {"Input type": input_type_print}  # "Condition group ind": group_ind, "Condition group name": group_name
        data_entry["Condition"] = cond_name
        data_entry["{} C-index".format(results_type)] = np.nan
        data_entry["Num. uncensored (log)"] = np.log(condition_count)
        data_entry["Fold"] = f
        data.append(data_entry)

    data = pd.DataFrame(data)
    print("Input types found")
    print(pd.unique(data["Input type"]))

    pd.set_option('display.max_columns', None)
    print(data.shape)
    print(data.head())

    all_data[results_type] = data

    feather.write_feather(data, os.path.join(root_dir, "analysis/part_1_ablation_table_{}.feather".format(results_type)))
    summary = data.groupby(["Input type", "Condition"]).mean().reset_index()
    summary_std = data.groupby(["Input type", "Condition"]).std().reset_index().rename({"{} C-index".format(results_type): "std"}, axis=1)[["Input type", "Condition", "std"]]
    summary = pd.merge(summary, summary_std, on=["Input type", "Condition"])
    print("Results for {}, models {}".format(results_type, hue_order_ablation))
    counts = defaultdict(list)
    for condition_ind in range(num_conditions + 1): # not average
      if condition_ind < num_conditions:
        cond_name = condition_names[condition_ind][0]
      else:
        cond_name = "Average"
      over_models = []
      for model_name in hue_order_ablation:
        res_row = summary[(summary["Condition"] == cond_name) & (summary["Input type"] == model_name)]
        if res_row.shape[0] > 0 and np.isfinite(res_row["Num. uncensored (log)"].squeeze()):
          over_models.append((res_row["{} C-index".format(results_type)].squeeze(), res_row["std"].squeeze()))
        else:
          print("Skipping", cond_name)
      if len(over_models) > 0:
        if not len(over_models) == len(hue_order_ablation):
          print(over_models)
          assert False

        best_cond = hue_order_ablation[np.array([tup[0] for tup in over_models]).argmax()]
        print("Perf result", cond_name, best_cond, over_models)
        counts[best_cond].append(cond_name)

    print("Best conditions {}:".format(results_type))
    print(counts)
    for k, v in counts.items():
      print((k, len(v)))

    sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})

    hue_plot_params = {
      'data': data,
      'x': "Condition",
      'y': "{} C-index".format(results_type),
      "order": c_group_order,
      "hue": "Input type",
      "hue_order": hue_order_ablation,

      "ci": 95,
      "errcolor": "black",
      "errwidth": 0.4,
      "capsize": 0.2  # 0.2
    }

    print("Linear avg result", "{} C-index".format(results_type))
    print(data.loc[(data["Input type"] == "Linear {}".format(linear_ablation_type))  & (data["Condition"] == "Average"), "{} C-index".format(results_type)].to_numpy().mean())
    print(data.loc[(data["Input type"] == "OnsetNet demog+bb+bbc+sbc+blood")  & (data["Condition"] == "Average"), "{} C-index".format(results_type)].to_numpy().mean())


    fig, axarr = plt.subplots(1, figsize=(18, 14.4))  # (20, 16)
    plot = sns.barplot(ax=axarr, palette=palette, **hue_plot_params)  # ci=None,
    legend_without_duplicate_labels(plot)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=(90), ha="center") # -30

    y_max_lim = min(val_max + 0.7, 1)
    axarr.set_ylim(round(val_min - 0.05, 1), y_max_lim)  # 0.15 -> 0.1, 0.19 -> 0.1, 0.2 -> 0.1 or 0.2
    axarr.set_xlim(-1, sum(list(cg_counts.values())))  # 0.15 -> 0.1, 0.19 -> 0.1, 0.2 -> 0.1 or 0.2
    axarr.spines[['right', 'top']].set_visible(False)
    axarr.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #axarr.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=False)
    axarr.tick_params(axis='both', which='major', labelbottom=True, bottom=False, top=False, labeltop=False)

    """
    pairs = []
    for c in c_group_order:
      if c in seen_conds:
        pairs.append([(c, "OnsetNet demog+bb+bbc+blood"), (c,  "OnsetNet demog+bb+bbc+sbc+blood")])
        pairs.append([(c, "OnsetNet demog+bb+bbc+sbc+blood"), (c, "Linear demog+bb+bbc+blood")])
        pairs.append([(c, "OnsetNet demog+bb+bbc+blood"), (c, "Linear demog+bb+bbc+blood")])
    #annotator = Annotator(axarr, pairs, **hue_plot_params)
    #annotator.configure(test="t-test_ind", pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]]).apply_and_annotate() # comparisons_correction="bonferroni")
    # comparisons_correction="Bonferroni",
    """
    curr_ylim = axarr.get_ylim()
    print("ylims")
    print(curr_ylim, y_max_lim)

    # draw dashed lines
    curr_cond = 0
    for cgi, cg in enumerate(cg_order):  # the order the condition groups are printed
      if cgi == len(cg_order) - 1:
        break

      curr_cond += cg_counts[cg]
      x_vline = curr_cond - 0.5
      axarr.axvline(x_vline, color="dimgray", linestyle="--")  # end of this one
      print("Plot at " + str(x_vline))
      # text at top
      axarr.text(x=(x_vline + 0.1), y=(curr_ylim[1] + 0.005), s=cg_order[cgi + 1], rotation=90)  # next label curr_ylim[1] #

    # plot.set(xlabel=None)

    # axarr[0].legend()
    # sns.move_legend(axarr[0], "upper right") # bbox_to_anchor=(1, 1)
    # axarr[1].get_legend().set_visible(False)

    plt.tight_layout()
    plot.figure.savefig(
      os.path.join(root_dir, "analysis/part_1_ablation_{}_{}.png".format(pretty(results_type), pretty(args.sing_cond))))
    plot.figure.savefig(
      os.path.join(root_dir, "analysis/part_1_ablation_{}_{}.eps".format(pretty(results_type), pretty(args.sing_cond))), format="eps")

    plt.clf()
    plt.close('all')


  #### print perf diff
  if "print" in args.run_types:
    for results_type in all_data:
      print("Linear avg result after", "{} C-index".format(results_type))
      data = all_data[results_type]
      print(data.loc[(data["Input type"] == "Linear {}".format(linear_ablation_type)) & (
          data["Condition"] == "Average"), "{} C-index".format(results_type)].to_numpy().mean())
      print(data.loc[(data["Input type"] == "OnsetNet demog+bb+bbc+sbc+blood") & (
          data["Condition"] == "Average"), "{} C-index".format(results_type)].to_numpy().mean())


    methods_ordered = ["Linear {}".format(linear_ablation_type)] + ["OnsetNet {}".format(input_type) for input_type in nn_ablation_types]
    printed = {}
    for method in methods_ordered:
      parts = method.split(" ")
      fst = parts[0]
      snd = " ".join(parts[1:])
      if not fst in printed:
        print(f"{fst} \\\\")
        printed[fst] = True

      row_val = [snd]
      row2_val = [" "]
      row3_val = [" "]

      all_perfs_method = []
      all_perfs_comp = []

      # get internal and external test, add to row
      for ri, (results_type, _) in enumerate(results_types_ablation):
        results_name = "{} C-index".format(results_type)
        data = all_data[results_type]

        perfs = data.loc[(data["Input type"] == method) & (data["Condition"] == "Average"), results_name].to_numpy()
        perfs_comp = data.loc[(data["Input type"] == "Linear {}".format(linear_ablation_type))  & (data["Condition"] == "Average"), results_name].to_numpy()

        perf_mean = perfs.mean()
        perf_std = perfs.std()
        row_val.append(f"${perf_mean:.3E}$")  # 3E
        row2_val.append(f"$\pm {perf_std:.2E}$")

        all_perfs_method.append(perfs)
        all_perfs_comp.append(perfs_comp)

        if not (method == "Linear {}".format(linear_ablation_type)):
          pval_with_linear = stats.ttest_ind(perfs, perfs_comp, equal_var=False).pvalue
          row3_val.append(f"($p$={pval_with_linear:.2E})")

      # get internal and external avg, add to row
      all_perfs_method = np.concatenate(all_perfs_method)
      all_perfs_comp = np.concatenate(all_perfs_comp)
      all_perfs_mean = all_perfs_method.mean()
      all_perfs_std = all_perfs_method.std()
      row_val.append(f"${all_perfs_mean:.3E} $")
      row2_val.append(f"$\pm {all_perfs_std:.2E}$")

      all_pval_with_linear = stats.ttest_ind(all_perfs_method, all_perfs_comp, equal_var=False).pvalue
      row3_val.append(f"($p$={all_pval_with_linear:.2E})")

      print(" & ".join(row_val) + "\\\\")
      print(" & ".join(row2_val) + "\\\\")
      if not (method == "Linear {}".format(linear_ablation_type)):
        print(" & ".join(row3_val) + "\\\\")

