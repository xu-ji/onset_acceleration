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

set_seed(1)

logs = defaultdict(int)

shoulder = []
hip = []
iliopsoas = []
midthigh = []
graph = []
instances = []
ids = []

gl_raw = glob.glob("{}/*".format(raw_dir))
print("Glob size {}".format(len(gl_raw)))
subject_dirs = []
for path_i, path in enumerate(gl_raw):
  if path_i < 5 or path_i % (len(gl_raw) // 100) == 0:
    print("Doing {} {} / {}, {}, {}".format(path_i, path, len(gl_raw), len(ids), datetime.now()))
    sys.stdout.flush()

  #if path_i >= 5: # 5000
  #  break # early

  path_last = path.split("/")[-1]
  if path_last[0].isnumeric() and path_last[-1].isnumeric() and path_last[-2] == "_":
    path_parts = path_last.split("_")
    assert len(path_parts) == 2
    id = int(path_parts[0])
    instance = int(path_parts[1])

    # Body tissues
    graph_value = np.zeros((len(channels), graph_len))
    try:
      for c_i, c in enumerate(channels):
        c_path = resource_paths[c].format(path_last)
        # milliliters = cubic centimeters
        curve, _ = np.array(compute_volume_of_mask(sitk.ReadImage(c_path), by_slice=True), dtype=np.int64)
        assert isinstance(curve, np.ndarray) and curve.shape == (num_slices_raw,)
        if trim_len > 0:
          assert np.abs(curve[:trim_len]).sum() + np.abs(curve[-trim_len:]).sum() == 0
          curve = curve[trim_len:-trim_len]
        graph_value[c_i, :] = curve
    except Exception as e:
      err = "Landmark loading curves {}".format(e)
      if path_i < 20:
        print("Example error: {} {} {} {}".format(path_last, c_i, c, c_path))
        print(traceback.format_exc())
      logs[err] += 1
      continue

    #sex_value = int(records.loc[records["individual_id"] == id]["sex"].values[0] == "Female")

    # Landmarks
    try:
      # Shoulder, hip
      bone_joint_file_path = resource_paths["bone_joints"].format("{}".format(path_last))
      with open(bone_joint_file_path, 'r') as f:
        json_data = json.load(f)
      np_array = np.array(list(json_data.values()))
      shoulder_value = trim_landmark_index((np_array[0, 2] + np_array[3, 2]) // 2)
      hip_value = trim_landmark_index((np_array[1, 2] + np_array[2, 2]) // 2)

    except Exception as e:
      err = "Landmark loading error bone joints {}".format(e)
      logs[err] += 1
      continue

    try:
      # T12 - iliopsoas
      c_fname_left = resource_paths["iliopsoas_left"].format("{}".format(path_last))
      c_fname_right = resource_paths["iliopsoas_right"].format("{}".format(path_last))

      if (not os.path.exists(c_fname_left)) or (not os.path.exists(c_fname_right)):
        raise FileNotFoundError

      _, _, _, x_max_left, y_max_left, iliopsoas_value_left = sitk_bbox(sitk.ReadImage(c_fname_left))
      _, _, _, x_max_right, y_max_right, iliopsoas_value_right = sitk_bbox(sitk.ReadImage(c_fname_right))
      iliopsoas_value = trim_landmark_index((iliopsoas_value_left + iliopsoas_value_right) // 2)

    except Exception as e:
      err = "Landmark loading error iliopsoas {}".format(e)
      logs[err] += 1
      continue

    try:
      # midthigh
      c_fname_left = resource_paths["midthigh_left"].format("{}".format(path_last))
      c_fname_right = resource_paths["midthigh_right"].format("{}".format(path_last))

      if (not os.path.exists(c_fname_left)) or (not os.path.exists(c_fname_right)):
        raise FileNotFoundError

      _, _, midthigh_min_left, _, _, midthigh_max_left = sitk_bbox(sitk.ReadImage(c_fname_left))
      _, _, midthigh_min_right, _, _, midthigh_max_right = sitk_bbox(sitk.ReadImage(c_fname_right))
      midthigh_left = (midthigh_min_left + midthigh_max_left) // 2
      midthigh_right = (midthigh_min_right + midthigh_max_right) // 2
      midthigh_value = trim_landmark_index((midthigh_left + midthigh_right) // 2)

    except Exception as e:
      err = "Landmark loading error midthigh {}".format(e)
      logs[err] += 1
      continue

    shoulder.append(shoulder_value)
    hip.append(hip_value)
    iliopsoas.append(iliopsoas_value)
    midthigh.append(midthigh_value)
    graph.append(graph_value)
    instances.append(instance)
    ids.append(id)
  else:
    logs["Non-patient dir"] += 1
    continue

  #if len(ids) == 5:
  #  break

print(logs)

shoulder = np.array(shoulder)
hip = np.array(hip)
iliopsoas = np.array(iliopsoas)
midthigh = np.array(midthigh)
graph = np.stack(graph, axis=0)
ids = np.array(ids)
instances = np.array(instances)

print("Pre-outlier shapes:")
for v in [shoulder, hip, iliopsoas, midthigh, graph, ids, instances]:
  print(v.shape)

# remove people with landmarks > 4 stddev
outliers_shoulder = get_outliers(shoulder)
outliers = outliers_shoulder
print("outliers_shoulder {}".format(outliers_shoulder.sum()))
print("Outliers + shoulder {}".format(outliers.sum()))

outliers_hip = get_outliers(hip)
outliers = np.logical_or(outliers, outliers_hip)
print("outliers_hip {}".format(outliers_hip.sum()))
print("Outliers + hip {}".format(outliers.sum()))

outliers_iliopsoas = get_outliers(iliopsoas)
outliers = np.logical_or(outliers, outliers_iliopsoas)
print("outliers_iliopsoas {}".format(outliers_iliopsoas.sum()))
print("Outliers + iliopsoas {}".format(outliers.sum()))

outliers_midthigh = get_outliers(midthigh)
outliers = np.logical_or(outliers, outliers_midthigh)
print("outliers_midthigh {}".format(outliers_midthigh.sum()))
print("Outliers + midthigh {}".format(outliers.sum()))

outliers_VAT = get_VAT_outliers(graph)
outliers = np.logical_or(outliers, outliers_VAT)
print("outliers_VAT {}".format(outliers_VAT.sum()))
print("Outliers + VAT (total) {}".format(outliers.sum()))

keep = np.logical_not(outliers)

shoulder = shoulder[keep]
hip = hip[keep]
iliopsoas = iliopsoas[keep]
midthigh = midthigh[keep]
graph = graph[keep]
instances = instances[keep]
ids = ids[keep]

# Register and format
graph, new_graph, avg_landmarks = register_graphs(graph, midthigh, hip, iliopsoas, shoulder)

# Format by changing dimensions
flat_graphs, lengths, offsets = flatten_format_graphs(new_graph, avg_landmarks)

graph_data = {
  "ids": ids,
  "instances": instances,

  "shoulder": shoulder,
  "hip": hip,
  "iliopsoas": iliopsoas,
  "midthigh": midthigh,

  "unregistered_graphs": graph,
  "graphs": new_graph,
  "flat_graphs": flat_graphs,

  "avg_landmarks": avg_landmarks,
  "tissue_lengths": lengths,
  "offsets": offsets
}
# Curves fully loaded
# Remember there will be duplicate ids because of longitudinal data
print("ID longitudinals check: first two same num all scans, third value num first scans only:")
print(ids.shape)
pair_string = lambda t : "{}_{}".format(t[0], t[1])
print(len(list(set([pair_string(a) for a in zip(ids, instances)]))))
print(np.unique(ids).shape)

with gzip.open(os.path.join(root_dir, "data/graph_data.gz"), "wb") as f:
  pickle.dump(graph_data, f)


# Store feather format for demographics file - ids and instances of people who made it. Landmarks summary and outliers info printed here already
pd_data = {"individual_id": ids,
          "instance_id": instances}
df = pd.DataFrame(pd_data)
feather.write_feather(df, os.path.join(root_dir, "data/graph_data_kept.feather"))

# Render example before and after for a few people. To check landmarks and nifti and graphs all align
# Can also call this in post processing!
print("Completed")