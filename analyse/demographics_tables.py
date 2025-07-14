import glob
from collections import defaultdict, OrderedDict

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

from ..util import *
from ..nn_util import *
from .util import *


# alcohol
print("\\toprule")
print("\\textbf{Sex} & \\textbf{Alcohol intake frequency} & \\textbf{Percentage (\\%)} & \\textbf{n} \\\\")
print("\\midrule")
for sex_i, fname in enumerate(["alcohol_m", "alcohol_f"]):
  df = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format(fname)))
  for index, row in df.iterrows():
    if index == 0:
      fst_entry = sex_dict[sex_i]
    else:
      fst_entry = ""

    print("{} & {} & {:.2f} & {} \\\\".format(fst_entry, row["alcohol"], row["freq"] * 100, row["nuniq"]))
  if sex_i == 0:
    print("\\midrule")
print("\\bottomrule")

print("-----")

# smoking

print("\\toprule")
print("\\textbf{Sex} & \\textbf{Smoking status} & \\textbf{Percentage (\\%)} & \\textbf{n} \\\\")
print("\\midrule")
for sex_i, fname in enumerate(["smoking_m", "smoking_f"]):
  df = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format(fname)))
  for index, row in df.iterrows():
    if index == 0:
      fst_entry = sex_dict[sex_i]
    else:
      fst_entry = ""

    print("{} & {} & {:.2f} & {} \\\\".format(fst_entry, row["smoking"], row["freq"] * 100, row["nuniq"]))
  if sex_i == 0:
    print("\\midrule")
print("\\bottomrule")

print("-----")

# race

print("\\toprule")
print("\\textbf{Sex} & \\textbf{Race} & \\textbf{Percentage (\\%)} & \\textbf{n} \\\\")
print("\\midrule")
for sex_i, fname in enumerate(["race_m", "race_f"]):
  df = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format(fname)))
  for index, row in df.iterrows():
    if index == 0:
      fst_entry = sex_dict[sex_i]
    else:
      fst_entry = ""

    print("{} & {} & {:.2f} & {} \\\\".format(fst_entry, row["race"], row["freq"] * 100, row["nuniq"]))
  if sex_i == 0:
    print("\\midrule")
print("\\bottomrule")

print("-----")


# education

print("\\toprule")
print("\\textbf{Sex} & \\textbf{Education} & \\textbf{Percentage (\\%)} & \\textbf{n} \\\\")
print("\\midrule")
for sex_i, fname in enumerate(["education_m", "education_f"]):
  df = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format(fname)))
  for index, row in df.iterrows():
    if index == 0:
      fst_entry = sex_dict[sex_i]
    else:
      fst_entry = ""

    print("{} & {} & {:.2f} & {} \\\\".format(fst_entry, row["education"], row["freq"] * 100, row["nuniq"]))
  if sex_i == 0:
    print("\\midrule")
print("\\bottomrule")

print("-----")


# continuous demographic traits
df = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format("demog_traits_continuous")))
cont_traits = OrderedDict({
"sex": "Sex",
"mean_DOB_year": "Mean DOB year",
"sd_DOB_year": "Std DOB year",
"min_DOB_year": "Min DOB year",
"max_DOB_year": "Max DOB year",
"iqr_DOB_year": "IQR DOB year",
"mean_townsend_deprivation": "Mean Townsend deprivation index"
})

print("\\toprule")
for (col, col_name) in cont_traits.items():
  col_res = df[col].to_list()
  if col == "sex":
    col_name = ""
    print("{} & \\textbf{{{}}} & \\textbf{{{}}} \\\\".format(col_name, col_res[1], col_res[0]))
    print("\\midrule")
  else:
    print("{} & {:.2f} & {:.2f} \\\\".format(col_name, col_res[1], col_res[0]))

print("\\bottomrule")

# centres

print("\\toprule")
print("\\textbf{Centre} & \\textbf{Percentage (\\%)} & \\textbf{n} \\\\")
print("\\midrule")
df = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format("centres_instance_2")))
for index, row in df.iterrows():
  centre_name = row["centre"]
  if row["centre"] == "Unknown":
    centre_name = "Unknown (imaging)"
  print("{} & {:.2f} & {} \\\\".format(centre_name, row["freq"] * 100, row["nuniq"]))
print("\\bottomrule")

print("-----")


# disease counts
df_f = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format("d_f")))
df_m = pd.read_csv(os.path.join(root_dir, "analysis/demogr/{}.csv".format("d_m")))
conds = df_f["Var1"].to_list()
print("\\toprule")
print(" & \\textbf{Male} & \\textbf{Female} \\\\")
print("\\midrule")
for cond in conds:
  print("{} & {} & {} \\\\".format(cond, df_m[df_m["Var1"] == cond]["Freq"].squeeze(),
                                   df_f[df_f["Var1"] == cond]["Freq"].squeeze()))

print("\\bottomrule")


