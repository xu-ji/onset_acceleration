from ..consts import *
import os
import pyarrow.feather as feather
import numpy as np

temp_root_dir = "/mnt/mica01/healthspan/v4.2"
raw_dir = "/home/mica/slurm_bucket/healthspan"

#dl_ids = [ for f in os.scandir(data_dir) if f.is_dir()]
resource_paths = {
"MASS": "{}/{{}}/analysis/mask.body_arms_removed.nii.gz".format(raw_dir),
"SAT": "{}/{{}}/analysis/mask.subcutaneous_fat_arms_removed.nii.gz".format(raw_dir),
"MUSC": "{}/{{}}/analysis/otsu_prob_argmax_total_muscle_arms_removed.nii.gz".format(raw_dir), # otsu name
"VAT": "{}/{{}}/analysis/mask.visceral_fat.nii.gz".format(raw_dir),
"TMAT": "{}/{{}}/analysis/mask.internal_fat_arms_removed.nii.gz".format(raw_dir),

"bone_joints": "{}/{{}}/landmarks/bone_joints.json".format(raw_dir), # yes
"iliopsoas_left": "{}/{{}}/analysis/model_iliopsoas_muscle_left.nii.gz".format(raw_dir),
"iliopsoas_right": "{}/{{}}/analysis/model_iliopsoas_muscle_right.nii.gz".format(raw_dir),
"midthigh_left": "{}/{{}}/analysis/mask.thigh_bones_left.nii.gz".format(raw_dir),
"midthigh_right": "{}/{{}}/analysis/mask.thigh_bones_right.nii.gz".format(raw_dir),
}

for vname in ["mri_v4b", "mri_v5"]:
  print("Doing ", vname)
  vf = os.path.join(temp_root_dir, "{}{}{}".format("data/analyse_slurm_", vname, ".feather"))
  vres = feather.read_feather(vf)
  r_ids = vres["individual_id"].to_numpy()

  dl_ids = []
  full_ids = []
  joints_ids = []
  r_and_full_ids = []
  r_and_joints_ids = []
  for f in os.scandir(raw_dir):
    if f.is_dir():
      id_scan = f.path.split("/")[-1]
      id = int(id_scan[:-2])
      dl_ids.append(id)

      has_all = True
      for k, v in resource_paths.items():
        has_all = has_all and os.path.exists(v.format(id_scan))
      if has_all:
        full_ids.append(id)

      has_r = id in r_ids

      if has_all and has_r:
        r_and_full_ids.append(id)

      if os.path.exists(resource_paths["bone_joints"].format(id_scan)):
        joints_ids.append(id)
        if has_r:
          r_and_joints_ids.append(id)

  for v in [dl_ids, full_ids, joints_ids, r_and_full_ids, r_and_joints_ids]:
    v_np = np.unique(np.array(v))
    print(v_np.shape, v_np[:5])

