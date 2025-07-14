import numpy as np
import pyarrow.feather as feather
import pandas as pd
from collections import defaultdict
import glob
import SimpleITK as sitk
from recoh.quantify import compute_volume_of_mask
from recoh.util import sitk_bbox
from datetime import datetime
import traceback
from scipy import stats
from .util import *

with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
  nn_data = pickle.load(f)
assessment_info = feather.read_feather(os.path.join(root_dir, "data/assessment_info.feather"))
assessment_info = assessment_info[assessment_info["instance_id"] == 0]

# for each variable, get the trainval set, int test set, ext test set
raw = defaultdict(dict)
vals = defaultdict(dict)
stds = defaultdict(dict)
pvals = defaultdict(dict)

printvals = OrderedDict()

for datatype in ["int_trainval", "int_test", "ext_test"]: # ordered
  datatype_ids = nn_data[datatype]["ids"].squeeze(axis=1) # numpy
  print(f"n for {datatype}: {datatype_ids.shape[0]}")
  #print("shape", nn_data[datatype]["ids"].shape, datatype_ids.shape[0], datatype_ids[:3], assessment_info["individual_id"].dtype)

  # age
  varname = "Age at baseline (yrs)"
  printvals[varname] = 1
  ids_df = pd.DataFrame({"individual_id": datatype_ids})
  datatype_assessment_info = ids_df.merge(assessment_info, how="inner", on="individual_id")
  assert datatype_assessment_info["age"].isna().to_numpy().sum() == 0
  ages = datatype_assessment_info["age"].to_numpy()
  assert ages.shape == (datatype_ids.shape[0],) # one per person

  assert ages.shape[0] == datatype_ids.shape[0] and len(ages.shape) == 1
  raw[datatype][varname] = ages
  vals[datatype][varname] = ages.mean()
  stds[datatype][varname] = ages.std()
  pvals[datatype][varname]  = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  ## demog
  demog = nn_data[datatype]["demog"]

  # sex
  varname = "Sex"
  printvals[varname] = 1
  assert sex_dict[1] == "Female" and BB_scalar_inds["sex"] == 1
  raw[datatype][varname] = demog[:, 0:(0 + 2)].mean(axis=0) * 100.0  # percentage each sex
  assert abs(raw[datatype][varname].sum() - 100.0) <= 1e-4 and raw[datatype][varname].shape == (2,)
  vals[datatype][varname] = None
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  for subi, subvar in enumerate(["Male", "Female"]):
    if subvar == "Female":
      subvarname = f"\hspace{{2mm}} {subvar}"
      printvals[subvarname] = 1
      vals[datatype][subvarname] = raw[datatype][varname][subi]
      stds[datatype][subvarname] = None
      pvals[datatype][subvarname] = None

  # Race
  varname = "Race"
  printvals[varname] = 1
  raw[datatype][varname] = demog[:, 2:(2+5)].mean(axis=0) * 100.0
  assert abs(raw[datatype][varname].sum() - 100.0) <= 1e-4 and raw[datatype][varname].shape == (5,)
  vals[datatype][varname] = None
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  for subi, subvar in enumerate(["White", "Other", "Asian or Asian British", "Mixed", "Black or Black British"]):
    if subvar == "White":
      subvarname = f"\hspace{{2mm}} {subvar}"
      printvals[subvarname] = 1
      vals[datatype][subvarname] = raw[datatype][varname][subi]
      stds[datatype][subvarname] = None
      pvals[datatype][subvarname] = None

  # Education
  varname = "Education"
  printvals[varname] = 1
  raw[datatype][varname] = demog[:, 7:(7+8)].mean(axis=0) * 100.0
  assert abs(raw[datatype][varname].sum() - 100.0) <= 1e-4 and raw[datatype][varname].shape == (8,)
  vals[datatype][varname] = None
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  for subi, subvar in enumerate(["College or University degree",
    "A levels/AS levels or equivalent",
    "O levels/GCSEs or equivalent",
    "CSEs or equivalent",
    "NVQ or HND or HNC or equivalent",
    "Other professional qualifications eg: nursing, teaching",
    "None of the above",
    "Prefer not to answer"]):
    if subvar == "College or University degree":
      subvarname = f"\hspace{{2mm}} {subvar}"
      printvals[subvarname] = 1
      vals[datatype][subvarname] = raw[datatype][varname][subi]
      stds[datatype][subvarname] = None
      pvals[datatype][subvarname] = None

  # deprivation
  varname = "Townsend deprivation index"
  townsend = demog[:, 2+5+8]
  raw[datatype][varname] = townsend
  vals[datatype][varname] = townsend.mean()
  stds[datatype][varname] = townsend.std()
  pvals[datatype][varname]  = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue


  ## basic body
  bb = nn_data[datatype]["bb"]

  # Smoking
  varname = "Smoking"
  printvals[varname] = 1
  raw[datatype][varname] = bb[:, 18:(18+4)].mean(axis=0) * 100.0
  #print(raw[datatype][varname] , raw[datatype][varname].sum(), abs(raw[datatype][varname].sum() - 100.0))
  assert abs(raw[datatype][varname].sum() - 100.0) <= 1e-4 and raw[datatype][varname].shape == (4,)
  vals[datatype][varname] = None
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  for subi, subvar in enumerate(["Prefer not to answer", "Never", "Previous", "Current"]):
    if subvar == "Never":
      subvarname = f"\hspace{{2mm}} {subvar}"
      printvals[subvarname] = 1
      vals[datatype][subvarname] = raw[datatype][varname][subi]
      stds[datatype][subvarname] = None
      pvals[datatype][subvarname] = None

  # alcohol
  varname = "Alcohol"
  printvals[varname] = 1
  raw[datatype][varname] = bb[:, 10:(10+7)].mean(axis=0) * 100.0
  #print(raw[datatype][varname], abs(raw[datatype][varname].sum() - 100.0))
  assert abs(raw[datatype][varname].sum() - 100.0) <= 1e-4 and raw[datatype][varname].shape == (7,)
  vals[datatype][varname] = None
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  for subi, subvar in enumerate(["Daily or almost daily", "Three or four times a week", "Once or twice a week",
                                 "One to three times a month", "Special occasions only", "Never", "Prefer not to answer"]):
    if subvar == "Three or four times a week":
      subvarname = f"\hspace{{2mm}} {subvar}"
      printvals[subvarname] = 1
      vals[datatype][subvarname] = raw[datatype][varname][subi]
      stds[datatype][subvarname] = None
      pvals[datatype][subvarname] = None

  # bmi
  varname = "BMI $(Kg/m^2)$"
  printvals[varname] = 1
  bmi = bb[:, 23]
  raw[datatype][varname] = bmi
  vals[datatype][varname] = bmi.mean()
  stds[datatype][varname] = bmi.std()
  pvals[datatype][varname]  = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  # grip strength
  varname = "Grip strength $(Kg)$"
  printvals[varname] = 1
  grip = bb[:, 6]
  raw[datatype][varname] = grip
  vals[datatype][varname] = grip.mean()
  stds[datatype][varname] = grip.std()
  pvals[datatype][varname]  = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue


  ##blood
  blood = nn_data[datatype]["blood"]

  # hba1c
  varname = "HbA1c ($mmol/mol$)"
  printvals[varname] = 1
  assert BLOOD_fields[26] == ("HbA1c", None)
  bloodvar = blood[:, 26]
  raw[datatype][varname] = bloodvar
  vals[datatype][varname] = bloodvar.mean()
  stds[datatype][varname] = bloodvar.std()
  pvals[datatype][varname] = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  # hdl-c
  varname = "HDL-C ($mmol/L$)"
  printvals[varname] = 1
  assert BLOOD_fields[28] == ("HDL_cholesterol", None)
  bloodvar = blood[:, 28]
  raw[datatype][varname] = bloodvar
  vals[datatype][varname] = bloodvar.mean()
  stds[datatype][varname] = bloodvar.std()
  pvals[datatype][varname] = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  # apoB
  varname = "ApoB ($g/L$)"
  printvals[varname] = 1
  assert BLOOD_fields[8] == ("apolipoprotein_B", None)
  bloodvar = blood[:, 8]
  raw[datatype][varname] = bloodvar
  vals[datatype][varname] = bloodvar.mean()
  stds[datatype][varname] = bloodvar.std()
  pvals[datatype][varname] = stats.ttest_ind(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  ## outcomes
  conditions = list(t[0] for t in nn_data[datatype]["conditions"])
  observed = nn_data[datatype]["observed"]
  print("conditions", conditions)

  # hypertension
  varname = "Hypertension (\%)"
  printvals[varname] = 1
  varind = conditions.index("Hypertension")
  #print(varname, varind)
  orig = observed[:, varind]
  onehot = np.zeros((orig.shape[0], 2))
  onehot[:, 1] = orig
  onehot[:, 0] = 1 - orig
  raw[datatype][varname] = onehot.mean(axis=0) * 100.0
  vals[datatype][varname] = orig.mean() * 100.0
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  # non-T1 diabetes
  varname = "Diabetes mellitus non-T1 (\%)"
  printvals[varname] = 1
  varind = conditions.index("Diabetes mellitus non-T1")
  #print(varname, varind)
  orig = observed[:, varind]
  onehot = np.zeros((orig.shape[0], 2))
  onehot[:, 1] = orig
  onehot[:, 0] = 1 - orig
  raw[datatype][varname] = onehot.mean(axis=0) * 100.0
  vals[datatype][varname] = orig.mean() * 100.0
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

  # depression
  varname = "Depression (\%)"
  printvals[varname] = 1
  varind = conditions.index("Depression")
  #print(varname, varind)
  orig = observed[:, varind]
  onehot = np.zeros((orig.shape[0], 2))
  onehot[:, 1] = orig
  onehot[:, 0] = 1 - orig
  raw[datatype][varname] = onehot.mean(axis=0) * 100.0
  vals[datatype][varname] = orig.mean() * 100.0
  stds[datatype][varname] = None
  pvals[datatype][varname] = stats.chisquare(raw[datatype][varname], raw["int_trainval"][varname]).pvalue

# for each variable, print the trainval mean std, int test mean std, int test pval, ext test mean std, ext test pval
for v in printvals:

  if not (stds['int_trainval'][v] is None): # continuous
    if vals['int_trainval'][v] < 1000 and vals['int_trainval'][v] >= 0.01: # 2 dp
      print(f"{v} & ${vals['int_trainval'][v]:.2f} \pm {stds['int_trainval'][v]:.2f}$ & "
          f"${vals['int_test'][v]:.2f} \pm {stds['int_test'][v]:.2f}$ & {pvals['int_test'][v]:.2E} & "
          f"${vals['ext_test'][v]:.2f} \pm {stds['ext_test'][v]:.2f}$ & {pvals['ext_test'][v]:.2E} \\\\")
    else: # sci notation
      print(f"{v} & ${vals['int_trainval'][v]:.2E} \pm {stds['int_trainval'][v]:.2E}$ & "
          f"${vals['int_test'][v]:.2E} \pm {stds['int_test'][v]:.2E}$ & {pvals['int_test'][v]:.2E} & "
          f"${vals['ext_test'][v]:.2E} \pm {stds['ext_test'][v]:.2E}$ & {pvals['ext_test'][v]:.2E} \\\\")

  else: # discrete, 2 dp
    if "\%" in v: # condition percentage
      print(f"{v} & ${vals['int_trainval'][v]:.2f} $ & "
          f"${vals['int_test'][v]:.2f} $ & {pvals['int_test'][v]:.2E} & "
          f"${vals['ext_test'][v]:.2f} $ & {pvals['ext_test'][v]:.2E} \\\\")
    elif "\hspace" in v: # entry
      print(f"{v} & ${vals['int_trainval'][v]:.2f}$ & "
            f"${vals['int_test'][v]:.2f}$ &  & "
            f"${vals['ext_test'][v]:.2f}$ & \\\\")
    else: # header
      print(f"{v} &  & "
            f" & {pvals['int_test'][v]: .2E} & "
            f" & {pvals['ext_test'][v]: .2E} \\\\")
