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

old_model = "nn_demog_bb_sbc_blood_5e-05_1e-05_0.0_256_0" # 4.1
old_model_fname = "{}/models/{}.pgz".format("/mnt/mica01/healthspan/v4.1", old_model)

new_model = "nn_demog_bb_bbc_sbc_blood_5e-05_5e-05_0.3_128_0_None" # 4.4
new_model_fname = "{}/models/{}.pgz".format("/home/mica/storage/healthspan/v4.4", new_model)
print(old_model, new_model)

# get data from new
with gzip.open(new_model_fname, "rb") as f:
  new_results = pickle.load(f)
new_args = new_results["args"]
int_trainvals, int_test, ext_test, conditions = get_nn_data(new_args, cond=True)

# test model
with gzip.open(old_model_fname, "rb") as f:
  results = pickle.load(f)
model = results["best_model"].eval() #.to(device)
model.c_inds = []
print("old int test results")
old_int_test = results["cross_val_results"][results["best_fold"]][2]
print(old_int_test)
#print(list(results.keys()))
for c_i, c in enumerate(conditions):
  print(c_i, c, old_int_test["c_index_condition"][c_i])

int_test_results = eval_nn(new_args, model, int_test, curr_device="cpu")
ext_test_results = eval_nn(new_args, model, ext_test, curr_device="cpu")
train_val_results = eval_nn(new_args, model, int_trainvals["train"][0], curr_device="cpu")

print("---")
print("int_test_results")
print(int_test_results)
print(int_test_results["c_index_condition"].shape)
print(conditions.shape)
#assert int_test_results["c_index_condition"].shape == conditions.shape
for c_i, c in enumerate(conditions):
  print(c_i, c, int_test_results["c_index_condition"][c_i])
