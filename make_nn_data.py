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

from .util import *

def make_dataset(ids, records_traits, records_other, graph_instance, graph_data, assessment_info, records_outcomes):
  # Indexing for individual id is ids (same in same dataset). Indexing for disease/input var field same across all datasets
  # Return ids; observed, diag_age; demog, bb, sbc, bbc, blood

  ids_df = pd.DataFrame({"individual_id": ids})
  n_ids = ids.shape[0]

  # DEMOG
  demog = []
  records_traits = pd.merge(ids_df, records_traits, how="left", on=["individual_id"])
  assert((records_traits["individual_id"].to_numpy() == ids).all()) and not records_traits.isnull().values.any()
  demog_len = 0
  for field, num_class in DEMOG_fields:
    field_data = records_traits[field].to_numpy()
    if not (num_class is None):
      field_data = field_one_hot(field_data, num_class)
    else:
      assert len(field_data.shape) == 1
      field_data = np.expand_dims(field_data, 1)
    demog_len += field_data.shape[1]
    demog.append(field_data)
  demog = np.concatenate(demog, axis=1).astype(np.float32)
  assert demog.shape == (n_ids, demog_len)
  print("demog shape", demog.shape)

  # BB
  bb = []
  records_other = pd.merge(ids_df, records_other, how="left", on=["individual_id"])
  assert((records_other["individual_id"].to_numpy() == ids).all()) and not records_other.isnull().values.any()
  print(records_other.columns.values.tolist())
  bb_len = 0
  for field, num_class in BB_fields:
    field_data = records_other[field].to_numpy()
    if not (num_class is None):
      field_data = field_one_hot(field_data, num_class)
    else:
      assert len(field_data.shape) == 1
      field_data = np.expand_dims(field_data, 1)
    bb_len += field_data.shape[1]
    bb.append(field_data)
  bb = np.concatenate(bb, axis=1).astype(np.float32)
  assert bb.shape == (n_ids, bb_len)
  print("bb shape", bb.shape)

  # BLOOD
  blood = []
  for field, num_class in BLOOD_fields:
    field_data = records_other[field].to_numpy()
    assert num_class is None
    assert len(field_data.shape) == 1
    field_data = np.expand_dims(field_data, 1)
    blood.append(field_data)
  blood = np.concatenate(blood, axis=1).astype(np.float32)
  assert blood.shape == (n_ids, len(BLOOD_fields))

  # SBC
  flat_graphs = graph_data["flat_graphs"]  # n, lflat
  lengths = graph_data["tissue_lengths"]
  offsets = graph_data["offsets"]
  graph_ids = graph_data["ids"]
  graph_instances = graph_data["instances"]
  n_graph_id_inst = graph_ids.shape[0]
  print("n_graph", n_graph_id_inst)  # not all graph data nec used

  sprint("Reordering graphs")
  data_kept_ids_exp = np.expand_dims(ids, axis=1)  # n, 1
  assert data_kept_ids_exp.shape == (n_ids, 1)

  id_match = data_kept_ids_exp == np.expand_dims(graph_ids, axis=0)  # n, ngii
  instances_match = graph_instance == np.expand_dims(graph_instances, axis=0)  # 1, ngii

  id_instances_match = np.logical_and(id_match, instances_match) # n, ngii
  assert id_instances_match.shape == (n_ids, n_graph_id_inst) and (id_instances_match.sum(axis=1) == 1).all() # individual and instance must exist in graph data
  mx, my = np.nonzero(id_instances_match)
  assert (mx == np.arange(n_ids)).all()
  assert np.unique(my).shape == (n_ids,)  # no repeat matches
  reordered_graphs = flat_graphs[my, :]
  assert reordered_graphs.shape == (n_ids, flat_graphs.shape[1])
  sprint("Reordered graphs")

  sbc = np.zeros((n_ids, len(channels), num_slices_raw), dtype=np.float32)
  curr_ind = 0
  for ci, c in enumerate(channels):
    curr_end_excl = curr_ind + lengths[ci]
    assert reordered_graphs[:, curr_ind:curr_end_excl].shape[1] == lengths[ci]
    sbc[:, ci, offsets[ci]:(offsets[ci] + lengths[ci])] = reordered_graphs[:, curr_ind:curr_end_excl]
    curr_ind = curr_end_excl
  print("sbc shape", sbc.shape)

  # BBC
  bbc = sbc.sum(axis=2)
  assert bbc.shape == (n_ids, len(channels))
  print("bbc shape", bbc.shape)

  # age_imaging, for both SBC, BBC
  assessment_info_img = assessment_info[assessment_info["instance_id"] == graph_instance]  # get centre for imaging visit
  age_imaging = pd.merge(ids_df, assessment_info_img, how="left", on=["individual_id"])
  assert((age_imaging["individual_id"].to_numpy() == ids).all())
  is_null_bool = (age_imaging.isnull().to_numpy().sum(axis=1) > 0)
  print(age_imaging.head(n=50))
  print("---")
  print(is_null_bool.sum(), age_imaging.isnull().to_numpy().shape)
  print((age_imaging.isnull().to_numpy().sum(axis=0)))

  assert(not age_imaging.isnull().values.any())
  age_imaging = age_imaging["age"].to_numpy().astype(np.float32)

  # observed_mat, event_time_mat
  # records_outcomes: individual_id, diagnosis_date, condition_group=factor, condition_ICD10=factor
  # records_traits: DOB_month, DOB_year
  observed = []
  diag_age = []
  conditions = np.unique(records_outcomes["condition_group"].to_numpy()) # sorted
  for cond in conditions:
    has_cond = records_outcomes[records_outcomes["condition_group"] == cond]
    has_cond_kept = pd.merge(records_traits, has_cond, how="left", on=["individual_id"]) # ordered
    assert (np.equal(has_cond_kept["individual_id"].to_numpy(), ids).all()) # has na

    has_cond_kept["observed"] = pd.to_numeric(has_cond_kept["condition_group"] == cond) # as opposed to NA
    not_obs = has_cond_kept[has_cond_kept["observed"] == 0]["individual_id"].to_numpy()
    assert has_cond_kept["observed"].sum() == pd.to_numeric(has_cond_kept["condition_group"] == cond).sum()

    has_cond_kept["year"] = records_traits["DOB_year"].astype(int) # already ordered right
    has_cond_kept["month"] = records_traits["DOB_month"].astype(int) + 1 # R 1 indexing
    has_cond_kept["day"] = 1
    has_cond_kept["birth"] = pd.to_datetime(has_cond_kept[["year", "month", "day"]])

    has_cond_kept["diag_age"] = (has_cond_kept["diagnosis_date"] - has_cond_kept["birth"]).dt.days
    has_cond_kept["diag_age"] = (has_cond_kept["diag_age"] / 365.).apply(np.floor)
    has_cond_kept["diag_age"] = has_cond_kept["diag_age"].fillna(has_cond_kept.pop("censor_age")) # set censored/unobserved age

    assert not has_cond_kept[["observed", "diag_age"]].isnull().values.any()
    observed.append(has_cond_kept["observed"].to_numpy())
    diag_age.append(has_cond_kept["diag_age"].to_numpy())

  observed = np.stack(observed, axis=1).astype(np.float32)
  diag_age = np.stack(diag_age, axis=1).astype(np.float32)
  assert observed.shape == (n_ids, conditions.shape[0])
  assert diag_age.shape == (n_ids, conditions.shape[0])

  return {"ids": ids,
          "observed": observed, "diag_age": diag_age,
          "demog": demog, "bb": bb, "sbc": sbc, "bbc": bbc, "age_imaging": age_imaging, "blood": blood,
          "conditions": conditions}


def get_normvals(dataset):
  stats = {}

  # do separately when there are mixed continuous and discrete
  for fi, (field, num_class) in enumerate(DEMOG_fields):
    if num_class is None:
      m, s = comp_normvals(dataset["demog"][:, fi])
      stats["demog_{}".format(field)] = (m, s)

  for fi, (field, num_class) in enumerate(BB_fields):
    if num_class is None:
      m, s = comp_normvals(dataset["bb"][:, fi])
      stats["bb_{}".format(field)] = (m, s)

  stats["blood"] = comp_normvals(dataset["blood"])
  stats["sbc"] = comp_normvals(dataset["sbc"])
  stats["bbc"] = comp_normvals(dataset["bbc"])
  stats["age_imaging"] = comp_normvals(dataset["age_imaging"])
  return stats


def apply_norm(dataset, stats):
  # do separately when there are mixed continuous and discrete
  norm_dataset = deepcopy(dataset)

  for fi, (field, num_class) in enumerate(DEMOG_fields):
    if num_class is None:
      m, s = stats["demog_{}".format(field)]
      #print(m.shape, s.shape, dataset["demog"][:, fi].shape)
      norm_dataset["demog"][:, fi] = single_norm(dataset["demog"][:, fi], m, s)

  for fi, (field, num_class) in enumerate(BB_fields):
    if num_class is None:
      m, s = stats["bb_{}".format(field)]
      norm_dataset["bb"][:, fi] = single_norm(dataset["bb"][:, fi], m, s)

  norm_dataset["blood"] = single_norm(dataset["blood"], stats["blood"][0], stats["blood"][1])
  norm_dataset["sbc"] = single_norm(dataset["sbc"], stats["sbc"][0], stats["sbc"][1])
  norm_dataset["bbc"] = single_norm(dataset["bbc"], stats["bbc"][0], stats["bbc"][1])
  norm_dataset["age_imaging"] = single_norm(dataset["age_imaging"], stats["age_imaging"][0], stats["age_imaging"][1])
  return norm_dataset


def field_one_hot(field_data, num_class):
  n = field_data.shape[0]
  new_data = np.zeros((n, num_class), dtype=int)
  for a_i in range(num_class):
    new_data[field_data == a_i, a_i] = 1
  return new_data


def check_condition_indexing(int_trainval, d):
  assert (int_trainval["conditions"] == d["conditions"]).all()


# ------------------------------------------------
# data load
# ------------------------------------------------
set_seed(0)

graph_data_kept = feather.read_feather(os.path.join(root_dir, "data/graph_data_kept.feather")) # individual_id, instance_id
with gzip.open(os.path.join(root_dir, "data/graph_data.gz"), "rb") as f:
  graph_data = pickle.load(f)

print("Graph kept data info")
print(graph_data_kept.shape)
graph_ids = graph_data_kept["individual_id"].to_numpy()
graph_instances = graph_data_kept["instance_id"].to_numpy()
print(graph_ids.shape, graph_instances.shape)
print("Graph unique ids:")
ugraph_ids = np.unique(graph_ids)
print(ugraph_ids.shape)
instances_unique, instance_count = np.unique(graph_instances, return_counts=True)
print("Graph unique instances:")
print(instances_unique)
print(instance_count)

records_traits = feather.read_feather(os.path.join(root_dir, "data/records_demog_traits.feather")) #no instance
records_outcomes = feather.read_feather(os.path.join(root_dir, "data/records_outcomes.feather"))
print("Converting date dtype")
print(records_outcomes["diagnosis_date"].dtypes)
records_outcomes["diagnosis_date"] = pd.to_datetime(records_outcomes["diagnosis_date"])
print(records_outcomes["diagnosis_date"].dtypes)

records_other = feather.read_feather(os.path.join(root_dir, "data/records_other_traits.feather"))
assessment_info = feather.read_feather(os.path.join(root_dir, "data/assessment_info.feather"))
print("Original data shapes")
print(records_traits.shape, records_outcomes.shape, records_other.shape)

# get intersection of ids in records_traits (should contain all), records_other, graph_ids as kept_ids
utraits_ids = np.unique(records_traits["individual_id"].to_numpy())
uother_ids = np.unique(records_other["individual_id"].to_numpy())
print("Traits and other unique ids")
print(utraits_ids.shape, uother_ids.shape)

# first criteria, to have demographic info and other traits info
kept_ids = np.intersect1d(np.intersect1d(utraits_ids, ugraph_ids), uother_ids)
print("kept_ids 1", kept_ids.shape)
# second criteria, to have assessment info for instance_id >= 2
assessment_info_ids = np.unique(assessment_info[assessment_info["instance_id"] >= 2]["individual_id"].to_numpy())
kept_ids = np.intersect1d(kept_ids, assessment_info_ids)
print("kept_ids 2", kept_ids.shape)

longitudinal_ids_2 = np.intersect1d(np.unique(graph_data_kept[graph_data_kept["instance_id"] == 2]["individual_id"].to_numpy()), kept_ids)
longitudinal_ids_3 = np.intersect1d(np.unique(graph_data_kept[graph_data_kept["instance_id"] == 3]["individual_id"].to_numpy()), kept_ids)
longitudinal_ids = np.intersect1d(longitudinal_ids_2, longitudinal_ids_3) # anyone included who has longitudinal scan and original scan whether in bristol or not
print("Longitudinal:", longitudinal_ids.shape)

# get ids: int_trainval (instance 2, other centres), int_test (instance 2, other centres), ext_test (instance 2, specific centre), longitudinal (instance 3)
ext_test_centre = "Bristol" # may need to try others
assessment_info = assessment_info[assessment_info["individual_id"].isin(longitudinal_ids_2)]
assessment_info_img = assessment_info[assessment_info["instance_id"] == 2] # get centre for imaging visit
ext_test_ids = np.unique(assessment_info_img[assessment_info_img["centre"].str.contains(ext_test_centre)]["individual_id"].to_numpy()) # people who have a scan at bristol
assessment_info_img_ext_test = assessment_info_img[assessment_info_img["individual_id"].isin(ext_test_ids)]
print("ext_test_ids:", ext_test_ids.shape)
print(list(pd.unique(assessment_info_img_ext_test["centre"]).to_numpy()))
print(list(pd.unique(assessment_info_img["centre"]).to_numpy()))
int_ids = kept_ids[~np.isin(kept_ids, ext_test_ids)] # people who don't have a scan at bristol
int_ids = np.intersect1d(int_ids, longitudinal_ids_2)
print("int_ids:", int_ids.shape)

# for stats
"""
first_assessment = pd.to_datetime(assessment_info[assessment_info["instance_id"] == 0]["assessment_date"])
print(first_assessment.describe())
#first_assessment_mean = pd.to_datetime(first_assessment.values.astype(np.int64).mean().date)
#print("first_assessment_mean", first_assessment_mean)
censor_date = np.datetime64("2022-11-30")
date_diffs = censor_date - first_assessment
print(date_diffs.describe())
"""

first_assessment = assessment_info[assessment_info["instance_id"] == 0][["individual_id", "assessment_date"]]
first_assessment["fst_assessment_date"] = first_assessment["assessment_date"]
second_assessment = assessment_info[assessment_info["instance_id"] == 2][["individual_id", "assessment_date"]]
second_assessment["snd_assessment_date"] = second_assessment["assessment_date"]
merged = pd.merge(first_assessment, second_assessment, how='inner', on="individual_id")
diff = merged["snd_assessment_date"] - first_assessment["fst_assessment_date"]
print(diff.describe())
print("---")
date_df = pd.to_datetime(second_assessment["snd_assessment_date"])
print(date_df.describe())
date_df = date_df.values.astype(np.int64)
print(pd.to_datetime(date_df.mean()))
print(pd.to_datetime(date_df.min()))
print(pd.to_datetime(date_df.max()))
exit(0)

n_int = int_ids.shape[0]
np.random.shuffle(int_ids)
int_test_ids = int_ids[:int(n_int * test_pc)]
int_trainval_ids = int_ids[int(n_int * test_pc):]

# make datasets
print("int_trainval")
int_trainval = make_dataset(int_trainval_ids, records_traits, records_other, 2, graph_data, assessment_info, records_outcomes)
print("int_test")
int_test = make_dataset(int_test_ids, records_traits, records_other, 2, graph_data, assessment_info, records_outcomes)
print("ext_test")
ext_test = make_dataset(ext_test_ids, records_traits, records_other, 2, graph_data, assessment_info, records_outcomes)
print("longitudinal")
longitudinal_2 = make_dataset(longitudinal_ids, records_traits, records_other, 2, graph_data, assessment_info, records_outcomes)
longitudinal_3 = make_dataset(longitudinal_ids, records_traits, records_other, 3, graph_data, assessment_info, records_outcomes)

# get averages for normalization of continuous values from int_trainval, make normalised cont var versions of all datasets
normvals = get_normvals(int_trainval)
int_trainval_norm = apply_norm(int_trainval, normvals)
int_test_norm = apply_norm(int_test, normvals)
ext_test_norm = apply_norm(ext_test, normvals)
longitudinal_2_norm = apply_norm(longitudinal_2, normvals)
longitudinal_3_norm = apply_norm(longitudinal_3, normvals)

for d in [int_test, ext_test, longitudinal_2, longitudinal_3, int_trainval_norm, int_test_norm, longitudinal_2_norm, longitudinal_3_norm]:
  check_condition_indexing(int_trainval, d)

# For all ids instance 2, check disease counts over int_trainval, int_test, ext_test same as in orig outcomes
print("Checking instance 2 disease counts")
all_observed_2 = np.concatenate([int_trainval["observed"], int_test["observed"], ext_test["observed"]], axis=0)
assert all_observed_2.shape == (longitudinal_ids_2.shape[0], num_conditions) # everyone accounted for
records_outcomes_sub = records_outcomes[records_outcomes["individual_id"].isin(longitudinal_ids_2)]
for ci, cond in enumerate(int_trainval["conditions"]):
  ct1 = int(all_observed_2[:, ci].sum())
  ct2 = (records_outcomes_sub["condition_group"] == cond).sum()
  print(ct1, ct2)
  assert(ct1 == ct2)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

ids_main = np.concatenate([int_trainval_ids, int_test_ids, ext_test_ids], axis=0)
assert ids_main.shape == np.unique(ids_main).shape
df = pd.DataFrame({"individual_id": ids_main})
feather.write_feather(df, os.path.join(root_dir, "data/ids_main.feather"))

print("results_dict shapes:")
results_dict = {
  "int_trainval": int_trainval,
  "int_test": int_test,
  "ext_test": ext_test,
  "longitudinal_2": longitudinal_2,
  "longitudinal_3": longitudinal_3,

  "int_trainval_norm": int_trainval_norm,
  "int_test_norm": int_test_norm,
  "ext_test_norm": ext_test_norm,
  "longitudinal_2_norm": longitudinal_2_norm,
  "longitudinal_3_norm": longitudinal_3_norm
}

print("Storing and resetting shapes:")
for k, v in results_dict.items():
  print("---------")
  print(k)
  print("---------")

  for k2, v2 in v.items():
    if isinstance(v2, np.ndarray):
      print(k2, v2.shape, v2.dtype)
      if v2.dtype == np.float32 or v2.dtype == float:
        print("   stats", get_stats(v2))
      if len(v2.shape) == 1:
        results_dict[k][k2] = np.expand_dims(v2, axis=1)
    else:
      print(k2, v2.__class__)

with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "wb") as f:
  pickle.dump(results_dict, f)


print("Completed")