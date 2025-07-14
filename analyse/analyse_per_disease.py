

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

from ..util import *
from ..nn_util import *
from .util import *
from ..consts import *

parser = argparse.ArgumentParser()
parser.add_argument('--fmodel', type=str, default="nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None")
args = parser.parse_args()

fname = os.path.join(root_dir, "models", args.fmodel + ".pgz")
print(fname)

with gzip.open(fname, "rb") as f:
  results = pickle.load(f)

args = results["args"]

int_trainvals, int_test, ext_test, conditions = get_nn_data(args, cond=True)

cross_val_results = results["cross_val_results"]
all_int_test = []
for fold in range(args.crossval_folds):
  int_test_results = cross_val_results[fold][2]["c_index_condition"]
  assert int_test_results.shape == (num_conditions,)
  all_int_test.append(int_test_results)

all_int_test = np.stack(all_int_test, axis=0)
print(all_int_test.shape) # num folds, num conditions

for ci, c in enumerate(conditions):
  #if c[0] in ["All cause mortality", "Diabetes mellitus"]:
  all_int_test_c = all_int_test[:, ci]

  interval = st.t.interval(0.95, len(all_int_test)-1, loc=np.mean(all_int_test_c), scale=st.sem(all_int_test_c))
  print(c, ci, all_int_test_c.mean(), interval, all_int_test_c.max())