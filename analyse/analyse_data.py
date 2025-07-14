
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

from ..util import *
from ..nn_util import *
from .util import *

run = 1

if run == 0:
  sex_ind_start = DEMOG_fields.index(("sex", 2))
  assert sex_ind_start == 0

  with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
    nn_data = pickle.load(f)

  int_trainval = nn_data["int_trainval"]
  int_test = nn_data["int_test"]
  ext_test = nn_data["ext_test"]

  all = defaultdict(list)
  print("Data details:")
  for dl_i, dl in enumerate([int_trainval, int_test, ext_test]):
    sex = dl["demog"][:, sex_ind_start + 1] # 1 for female
    age = dl["age_imaging"]
    diag_ag = dl["diag_age"]
    all["sex"].append(sex)
    all["age"].append(age)
    # ages of people with diag after imaging
    #print(age.shape, diag_ag.shape)
    diffs = diag_ag - age
    diffs[diffs <= 0] = 0 #keep positives only
    all["diag_diff"].append(diffs)

    print(age.shape)
    assert len(age.shape) == 2 and age.shape[1] == 1
    print(dl_i, (sex == 0).sum(), (sex == 1).sum(), age.mean(), age.std())

  print("total")
  for k, v in all.items():
    vs = np.concatenate(v, axis=0)
    print(k, (vs == 0).sum(), (vs == 1).sum(), vs.mean(), vs.std(), vs.min(), vs.max())

