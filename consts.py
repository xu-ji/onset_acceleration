from collections import defaultdict, OrderedDict

raw_dir = "/home/mica/slurm_bucket/healthspan"
root_dir = "/home/mica/storage/healthspan/v6.2"

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

test_pc = 0.15

trim_len = 0
num_slices_raw = 370
graph_len = num_slices_raw - 2 * trim_len
channels = ["MASS", "SAT", "MUSC", "VAT", "TMAT"]
sex_dict = {0: "Male", 1: "Female"}

DEMOG_fields = [("sex", 2), ("race", 5), ("education", 8), ("townsend_deprivation", None)]
# 4

BB_fields = [("height", None), ("age_height", None),
             ("weight", None), ("age_weight", None),
             ("waist_circumference", None), ("age_waist_circumference", None),
             ("grip_strength", None),  ("age_grip_strength", None),
             ("blood_pressure_systolic", None), ("age_blood_pressure_systolic", None), # 10
              ("alcohol", 7), ("age_alcohol", None), # 8
             ("smoking", 4),   ("age_smoking", None), # 5
             ("BMI", None), ("age_BMI", None) # 2
             ] # 14 + 11 = 25

BLOOD_fields = [
  ("alanine_aminotransferase", None), ("age_alanine_aminotransferase", None),
  ("albumin", None), ("age_albumin", None),
  ("alkaline_phosphatase", None), ("age_alkaline_phosphatase", None),
  ("apolipoprotein_A", None), ("age_apolipoprotein_A", None),
  ("apolipoprotein_B", None), ("age_apolipoprotein_B", None),
  ("aspartate_aminotransferase", None), ("age_aspartate_aminotransferase", None),
  ("c_reactive_protein", None), ("age_c_reactive_protein", None),
  ("calcium", None), ("age_calcium", None),
  ("cholesterol", None), ("age_cholesterol", None),
  ("creatinine", None), ("age_creatinine", None),
  ("cystatin_c", None), ("age_cystatin_c", None),
  ("gamma_glutamyltransferase", None), ("age_gamma_glutamyltransferase", None),
  ("glucose", None), ("age_glucose", None),
  ("HbA1c", None), ("age_HbA1c", None),
  ("HDL_cholesterol", None), ("age_HDL_cholesterol", None),
  ("IGF1", None), ("age_IGF1", None),
  ("LDL_direct", None), ("age_LDL_direct", None),
  ("phosphate", None), ("age_phosphate", None),
  ("SHBG", None), ("age_SHBG", None),
  ("total_bilirubin", None), ("age_total_bilirubin", None),
  ("total_protein", None), ("age_total_protein", None),
  ("triglycerides", None), ("age_triglycerides", None),
  ("urate", None), ("age_urate", None),
  ("urea", None), ("age_urea", None),
  ("vitamin_D", None), ("age_vitamin_D", None)
]
#25, 50 total

# 1 for SBC graph

# 5 for BBC components

def get_feat_len(fields):
  total = 0
  for f, ft in fields:
    if ft is None:
      total += 1
    else:
      assert isinstance(ft, int)
      total += ft
  return total

nn_data_types = ["ids", "observed", "diag_age", "demog", "bb", "sbc", "bbc", "age_imaging", "blood"]
nn_module_order = ["demog", "bb", "sbc", "bbc", "blood"]
nn_module_input_len = {"demog": get_feat_len(DEMOG_fields),
                       "bb": get_feat_len(BB_fields),
                       "sbc": ((len(channels), num_slices_raw), 1),
                       "bbc": len(channels) + 1,
                       "blood": get_feat_len(BLOOD_fields)}

model_inputs_list = [["demog"], ["demog", "bb"],
                     ["demog", "bb", "bbc"],
                     ["demog", "bb", "sbc"], ["demog", "bb", "blood"],
                     ["demog", "bb", "bbc", "blood"], ["demog", "bb", "sbc", "blood"],
                     ["demog", "bb", "bbc", "sbc", "blood"]]

# ranked by best fold
condition_supergroups = {
  "All cause mortality": "Composite",
  "All cause morbidity": "Composite",

  "Diabetes mellitus non-T1": "Metabolic/endocrine/nutritional",
  "Hyperuricemia/gout": "Metabolic/endocrine/nutritional",
  "Obesity": "Metabolic/endocrine/nutritional",
  "Lipid metabolism disorders": "Metabolic/endocrine/nutritional",
  "Thyroid diseases": "Metabolic/endocrine/nutritional",

  "Renal insufficiency": "Genitourinary",
  "Prostatic hyperplasia": "Genitourinary",
  "Gynecological problems": "Genitourinary",
  "Urinary incontinence": "Genitourinary",
  "Urinary tract calculi": "Genitourinary",

  "Cardiac insufficiency": "Circulatory",
  "Chronic ischemic heart disease": "Circulatory",
  "Hypertension": "Circulatory",
  "Hypotension": "Circulatory",
  "Cerebral ischemia/chronic stroke": "Circulatory",
  "Lower limb varicosis": "Circulatory",
  "Cardiac arrhythmias": "Circulatory",
  "Hemorrhoids": "Circulatory",
  "Cardiac valve disorders": "Circulatory",
  "Atherosclerosis/PAOD": "Circulatory",

  "Liver disease": "Digestive",
  "Chronic cholecystitis/gallstones": "Digestive",
  "Intestinal diverticulosis": "Digestive",
  "Chronic gastritis/GERD": "Digestive",

  "Dementia": "Mental/behavioural/mixed",
  "Sexual dysfunction": "Mental/behavioural/mixed",
  "Anxiety": "Mental/behavioural/mixed",
  "Insomnia": "Mental/behavioural/mixed",
  "Depression": "Mental/behavioural/mixed",
  "Somatoform disorders": "Mental/behavioural/mixed",

  "Anemia": "Anemia",

  "Parkinson’s disease": "Nervous system",
  "Migraine/chronic headache": "Nervous system",
  "Neuropathies": "Nervous system",

  "Cancer": "Cancer",

  "Bronchitis/COPD": "Respiratory",

  "Osteoporosis": "Musculoskeletal",
  "Joint arthritis": "Musculoskeletal",
  "Rheumatoid arthritis/chronic polyarthritis": "Musculoskeletal",
  "Chronic low back pain": "Musculoskeletal",

  "Psoriasis": "Allergy/skin",
  "Allergy": "Allergy/skin",

  "Severe vision reduction": "Eye",

  "Dizziness": "Ear",
  "Severe hearing loss": "Ear"
}

num_conditions = len(condition_supergroups)
print("num_conditions", num_conditions)
assert num_conditions == 47

# printing order
# cgroups_ord doesn't contain Average, cg_order does
cgroups_ord = ['Composite', 'Metabolic/endocrine/nutritional', 'Digestive',
               'Genitourinary', 'Circulatory', 'Nervous system', 'Musculoskeletal',
                'Mental/behavioural/mixed', 'Anemia',
               'Ear', 'Allergy/skin', 'Cancer',
                'Respiratory', 'Eye']

# average first
c_group_order = ["Average"]
cg_counts = defaultdict(int)
cg_counts["Average"] += 1 # empty string
cg_order = ["Average"]

# composite next
cg_order.append("Composite")
for c, cgc in condition_supergroups.items():
  if cgc == "Composite":
    c_group_order.append(c)
    cg_counts[cgc] += 1

# our order, after average and composite
for cg in cgroups_ord[1:]:
  cg_order.append(cg)
  for c, cgc in condition_supergroups.items():
    if cgc == cg:
      c_group_order.append(c)
      cg_counts[cgc] += 1

print("cg_counts")
print(cg_counts)
print("cg_order")
print(cg_order)
print("c_group_order")
print(c_group_order)


render_blood_vars_names = OrderedDict([
  ("LDL_direct", "LDL-C $mmol/L$"),
  ("triglycerides", "Triglycerides $mmol/L$"),
  ("apolipoprotein_A", "ApoA $g/L$"),
  ("apolipoprotein_B", "ApoB $g/L$"),
  ("cholesterol", "Cholesterol $mmol/L$"),
  ("HDL_cholesterol", "HDL-C $mmol/L$"),
  ("glucose", "Glucose $mmol/L$"),
  ("HbA1c", "HbA1c $mmol/mol$"),
  ("IGF1", "IGF-1 $nmol/L$"),
  ("urate", "Urate $umol/L$"),
  ("urea", "Urea $mmol/L$"),
  ("total_protein", "tot. Protein $g/L$"),
  ("alanine_aminotransferase", "ALT $U/L$"),

  ("albumin",  "ALB $g/L$"),
  ("alkaline_phosphatase", "ALP $U/L$"),
  ("aspartate_aminotransferase", "AST $U/L$"),
  ("c_reactive_protein",  "CRP $mg/L$"),
  ("calcium", "Ca $mmol/L$"),
  ("creatinine", "Creatinine $umol/L$"),
  ("cystatin_c", "Cystatin C $mg/L$"),
  ("gamma_glutamyltransferase", "GGT $U/L$"),
  ("phosphate", "Phosphate $mmol/L$"),
  ("SHBG", "SHBG $nmol/L$"),
  ("total_bilirubin", "tot. Bilirubin $umol/L$"),
  ("vitamin_D", "Vit D $nmol/L$"),
  ]
)

# both ordered
render_blood_vars = list(render_blood_vars_names.keys())
render_bb_vars = []
for bbf, _ in BB_fields:
  if (not ("age" in bbf)) and (not ("alcohol" in bbf) and (not ("smoking") in bbf)):
    render_bb_vars.append(bbf)

print("render vars:")
print(render_blood_vars)
print(render_bb_vars)

# from make_trait_data.r
alcohol_fields = ["Daily", "3-4/wk", "1-2/wk", "1-3/mth", "Occasion", "Never", "N/A"]
smoking_fields = ["N/A", "Never", "Prev.", "Curr."]

# index of discrete values in BB; if 1 is in this index, scalar val is 1, else 0
BB_scalar_inds = {
  "sex": 1, # female
  "race": 0, # white
  "education": 0, # college
  "alcohol": alcohol_fields.index("Never"),
  "smoking": smoking_fields.index("Never")
}

# single sex diseases, match hazards script
sex_f_conds = ["Thyroid diseases", "Gynecological problems", "Osteoporosis"]
sex_m_conds = ["Prostatic hyperplasia", "Hyperuricemia/gout", "Sexual dysfunction"]
sex_conds = [sex_m_conds, sex_f_conds]

# similar to part 2 but no units and flat
simple_text_names = {"height": "Height",
                     "weight": "Weight",
                     "waist_circumference": "Waist",
                     "grip_strength": "Grip str.",
                     "blood_pressure_systolic": "BP sys.",
                     "alcohol": "Alcohol",
                     "smoking": "Smoking",
                     "BMI": "BMI",

                     "MASS": "tot. MASS",
                     "SAT": "tot. SAT",
                     "MUSC": "tot. MUSC",
                     "VAT": "tot. VAT",
                     "TMAT": "tot. TMAT",
                     }
for bl, bl_name in render_blood_vars_names.items():
  simple_text_names[bl] = bl_name.split("$")[0][:-1]

render_text_subfields = {"netrisk": "OnsetNet score",
                         "hazard_ratio": "HR (p-val)",
                         "hazard_ratio_ful_adj": "HR adj+ (p-val)",

                         "age": "Assess. age $yrs$",  # 3
                         "height": "Height $cm$",
                         "weight": "Weight $kg$",
                         "waist_circumference": "Waist $cm$",
                         "grip_strength": "Grip str. $kg$",
                         "blood_pressure_systolic": "BP sys. $mmHg$",
                         "alcohol": "Alcohol mode ($\%$)",
                         "smoking": "Smoking mode ($\%$)",
                         "BMI": "BMI $kg/m^2$",  # 8 + 3

                         "bbc": ["tot. {} $cm^3$".format(ch) for ch in channels],  # 16
                         "blood": [render_blood_vars_names[blood_var] for blood_var in render_blood_vars]  # 25
                         }
