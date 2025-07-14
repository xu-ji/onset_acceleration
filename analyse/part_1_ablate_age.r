library(tidyverse)
library(plyr)
library(arrow)
library(survival)
library(broom)
library(caret)
library(stringr)

r_pretty <- function(l) {
  return(str_replace_all(l, "[^[:alnum:]]", "_"))
}

args <- commandArgs(trailingOnly = TRUE)
print(args)
multimorbid_acmort_input_type_ind = args[1]
print(multimorbid_acmort_input_type_ind)

find_corr = args[2] == "True"
print(find_corr)

data_root <- "/home/mica/storage/healthspan/v6.2"
fmodel <- "nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None" # can be any
print(fmodel)

conditions <- c(
  "All cause mortality",
  "All cause morbidity",
  "Diabetes mellitus non-T1",
  "Hyperuricemia/gout",
  "Obesity",
  "Lipid metabolism disorders",
  "Thyroid diseases",
  "Renal insufficiency",
  "Prostatic hyperplasia",
  "Gynecological problems",
  "Urinary incontinence",
  "Urinary tract calculi",
  "Cardiac insufficiency",
  "Chronic ischemic heart disease",
  "Hypertension",
  "Hypotension",
  "Cerebral ischemia/chronic stroke",
  "Lower limb varicosis",
  "Cardiac arrhythmias",
  "Hemorrhoids",
  "Cardiac valve disorders",
  "Atherosclerosis/PAOD",
  "Liver disease",
  "Chronic cholecystitis/gallstones",
  "Intestinal diverticulosis",
  "Chronic gastritis/GERD",
  "Dementia",
  "Sexual dysfunction",
  "Anxiety",
  "Insomnia",
  "Depression",
  "Somatoform disorders",
  "Anemia",
  "Parkinson’s disease",
  "Migraine/chronic headache",
  "Neuropathies",
  "Cancer",
  "Bronchitis/COPD",
  "Osteoporosis",
  "Joint arthritis",
  "Rheumatoid arthritis/chronic polyarthritis",
  "Chronic low back pain",
  "Psoriasis",
  "Allergy",
  "Severe vision reduction",
  "Dizziness",
  "Severe hearing loss"
)
print(conditions)
print(length(conditions))
stopifnot(length(conditions) == 47)

trait_path <- file.path(data_root, "data/make_trait_data.rda")
load(file=trait_path)

ids_main <- arrow::read_feather(file.path(data_root, "data/ids_main.feather"))
print("ids_main")
print(dim(ids_main))

collected_data <- arrow::read_feather(file.path(data_root, paste0("analysis/collected_data_", fmodel, ".feather")))
print("collected_data")
print(dim(collected_data)) # number of people x num conditions 47
print(head(collected_data))
collected_data <- left_join(ids_main, collected_data, by=join_by(individual_id))
stopifnot(sum(is.na(collected_data)) == 0)

# combine ids_main with demog and other traits
dim1 <- dim(ids_main)[1]
records_key_traits <- left_join(ids_main, records_demog_traits, by = join_by(individual_id))
records_key_traits <- left_join(records_key_traits, records_other_traits, by=join_by(individual_id))
print(colnames(records_key_traits))
records_key_traits <- records_key_traits %>% select(individual_id,
  # censor info
  censor_age, DOB_approx,

  # demog
  sex, race, education, townsend_deprivation,

  # bb, ages
  height, weight, waist_circumference, grip_strength, blood_pressure_systolic, alcohol, smoking, BMI,
  #age_height, age_weight, age_waist_circumference, age_grip_strength, age_blood_pressure_systolic, age_alcohol, age_smoking, age_BMI,
  bb_age,

  # blood, ages
  alanine_aminotransferase, albumin, alkaline_phosphatase, apolipoprotein_A, apolipoprotein_B, aspartate_aminotransferase,
  c_reactive_protein, calcium, cholesterol, creatinine, cystatin_c, gamma_glutamyltransferase, glucose,
  HbA1c, HDL_cholesterol, IGF1, LDL_direct, phosphate, SHBG, total_bilirubin, total_protein, triglycerides, urate,
  urea, vitamin_D,
  #age_alanine_aminotransferase, age_albumin, age_alkaline_phosphatase, age_apolipoprotein_A, age_apolipoprotein_B, age_aspartate_aminotransferase,
  #age_c_reactive_protein, age_calcium, age_cholesterol, age_creatinine, age_cystatin_c, age_gamma_glutamyltransferase, age_glucose,
  #age_HbA1c, age_HDL_cholesterol, age_IGF1, age_LDL_direct, age_phosphate, age_SHBG, age_total_bilirubin, age_total_protein, age_triglycerides, age_urate,
  #age_urea, age_vitamin_D
  blood_age
  )

# todo remove
print("Check for bb_age, blood_age")
print(colnames(records_key_traits))
stopifnot("bb_age" %in% colnames(records_key_traits))
stopifnot("blood_age" %in% colnames(records_key_traits))

# print correlated values
if (find_corr) {
  traits_sub <- records_key_traits[ , !(names(records_key_traits) %in% c("individual_id", "censor_age", "DOB_approx", "sex", "race", "education", "alcohol", "smoking"))]
  corr_df = cor(traits_sub)
  hc = findCorrelation(corr_df, cutoff=0.9, names=TRUE)
  #hc = sort(hc)
  print("Original columns:")
  print(colnames(records_key_traits))
  print("Correlated columns:")
  print(hc)
}

stopifnot(sum(is.na(records_key_traits)) == 0)
dim2 <- dim(records_key_traits)[1]
print(c(dim(ids_main), dim(records_key_traits)))
stopifnot(dim1 == dim2)

print("Check date type")
print(head(select(records_outcomes, individual_id, condition_group, diagnosis_date)))

for (condition in conditions) {
  print("Doing condition:")
  condition_pretty <- r_pretty(condition)
  print(condition_pretty)

  outcomes_ci2 <- records_outcomes %>% filter(condition_group == condition) %>% select(individual_id, diagnosis_date)
  outcomes_ci2 <- left_join(records_key_traits, outcomes_ci2, by = join_by(individual_id)) # can only have 1 entry per disease
  print(dim(outcomes_ci2))

  # time is age at death, censored age is already stored
  outcomes_ci2$diagnosis_age <- round(time_length(outcomes_ci2$diagnosis_date - outcomes_ci2$DOB_approx, unit="years")) # some na
  outcomes_ci2$status <- ifelse(is.na(outcomes_ci2$diagnosis_age), 0, 1) # if it's na, status is 0 else 1
  print(c("is na (no diagnosis) should be same", sum(is.na(outcomes_ci2$diagnosis_age)), sum(outcomes_ci2$status == 0), "total shape", dim(outcomes_ci2$diagnosis_age)))
  outcomes_ci2$time <- ifelse(is.na(outcomes_ci2$diagnosis_age), outcomes_ci2$censor_age, outcomes_ci2$diagnosis_age)

  collected_data_select <- collected_data[c("individual_id", "bbc_age", "MASS", "SAT", "MUSC", "VAT", "TMAT", "data_segment")]
  outcomes_ci2 <- left_join(outcomes_ci2, collected_data_select, by = join_by(individual_id))
  print(head(outcomes_ci2))
  print(dim(outcomes_ci2))

  train_data = outcomes_ci2 %>% filter(data_segment <= 1)
  int_test_data = outcomes_ci2 %>% filter(data_segment == 2)
  ext_test_data = outcomes_ci2 %>% filter(data_segment == 3)

  print(dim(train_data))
  print(dim(int_test_data))
  print(dim(ext_test_data))

  if (multimorbid_acmort_input_type_ind == "demog+bb+bbc+blood") {
    res_cox <- coxph(Surv(time, status) ~ relevel(
      # demog
      as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
      # bb
      scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI) +
      scale(bb_age) +
      #scale(age_height) + scale(age_waist_circumference) + scale(age_grip_strength) + scale(age_blood_pressure_systolic) + scale(age_alcohol) + scale(age_smoking) + scale(age_BMI) +

      # bbc
      scale(MASS) + scale(SAT) + scale(MUSC) + scale(VAT) + scale(TMAT) +
      scale(bbc_age) +

      # blood
      scale(alanine_aminotransferase) + scale(albumin) + scale(alkaline_phosphatase) + scale(apolipoprotein_A) + scale(apolipoprotein_B) + scale(aspartate_aminotransferase) +
      scale(c_reactive_protein) + scale(calcium) + scale(cholesterol) + scale(creatinine) + scale(cystatin_c) + scale(gamma_glutamyltransferase) + scale(glucose) +
      scale(HbA1c) + scale(HDL_cholesterol) + scale(IGF1) + scale(LDL_direct) + scale(phosphate) + scale(SHBG) + scale(total_bilirubin) + scale(total_protein) + scale(triglycerides) + scale(urate) +
      scale(urea) + scale(vitamin_D) +
      scale(blood_age)
      #scale(age_alanine_aminotransferase) + scale(age_albumin) + scale(age_alkaline_phosphatase) + scale(age_apolipoprotein_A) + scale(age_apolipoprotein_B) + scale(age_aspartate_aminotransferase) +
      #scale(age_c_reactive_protein) + scale(age_calcium) + scale(age_cholesterol) + scale(age_creatinine) + scale(age_cystatin_c) + scale(age_gamma_glutamyltransferase) + scale(age_glucose) +
      #scale(age_HbA1c) + scale(age_HDL_cholesterol) + scale(age_IGF1) + scale(age_LDL_direct) + scale(age_phosphate) + scale(age_SHBG) + scale(age_total_bilirubin) + scale(age_total_protein) + scale(age_triglycerides) + scale(age_urate) +
      #scale(age_urea) + scale(age_vitamin_D)
      ,
      data = train_data)
  }

  res1 <- as.data.frame(tidy(res_cox))
  res2 <- as.data.frame(glance(res_cox))

  train_data$infer_pred <- predict(res_cox, train_data, type="lp") # lp
  print("First train data dims")
  print(dim(train_data))

  skip_res3 <- TRUE
  tryCatch(
    expr = {
      zp <- cox.zph(res_cox, terms = TRUE)
      res3 <- as.data.frame(zp$table)
      res3$term_name <- row.names(res3) # store explicitly for feather
      skip_res3 <- FALSE
    },
    error = function(e){
      message('Caught zph error')
      print(e)
    }
  )

  # int_test results
  cindex_int_test <- concordance(object=res_cox, newdata=int_test_data)
  print(cindex_int_test)
  print(names(unclass(cindex_int_test)))
  res2$int_test_concordance = cindex_int_test$concordance
  # https://cran.r-project.org/web/packages/survival/survival.pdf
  # https://dominicmagirr.github.io/post/2021-12-14-notes-on-concordance/
  res2$int_test_cvar = cindex_int_test$cvar
  res2$int_test_n = cindex_int_test$n

  print("res2 after int_test")
  print(res2)

  int_test_pred <- predict(res_cox, int_test_data, type="lp") # , reference="sample"
  print("int_test dims")
  print(dim(int_test_pred))
  print(dim(int_test_data))
  int_test_data$infer_pred <- int_test_pred
  print(dim(int_test_data))
  print(head(int_test_data[c("individual_id", "infer_pred", "data_segment")]))
  stopifnot(all(int_test_data$data_segment == 2))

  # ext_test results
  cindex_ext_test <- concordance(object=res_cox, newdata=ext_test_data)
  print(cindex_ext_test)
  print(names(unclass(cindex_ext_test)))
  res2$ext_test_concordance = cindex_ext_test$concordance
  res2$ext_test_cvar = cindex_ext_test$cvar
  res2$ext_test_n = cindex_ext_test$n

  print("res2 after ext_test")
  print(res2)

  ext_test_pred <- predict(res_cox, ext_test_data, type="lp") # , reference="sample"
  print("ext_test dims")
  print(dim(ext_test_pred))
  print(dim(ext_test_data))
  ext_test_data$infer_pred <- ext_test_pred
  print(dim(ext_test_data))
  print(head(ext_test_data[c("individual_id", "infer_pred", "data_segment")]))
  stopifnot(all(ext_test_data$data_segment == 3))

  # combine

  all_pred <- do.call("rbind", list(train_data, int_test_data, ext_test_data))
  print("Final dims")
  print(colnames(all_pred))
  print(dim(all_pred))

  arrow::write_feather(all_pred, file.path(data_root, paste0("analysis/linear_cox/ablate_age_", condition_pretty, "_all_pred_", multimorbid_acmort_input_type_ind, ".feather")))

  arrow::write_feather(res1, file.path(data_root, paste0("analysis/linear_cox/ablate_age_", condition_pretty, "_res1_", multimorbid_acmort_input_type_ind, ".feather")))
  arrow::write_feather(res2, file.path(data_root, paste0("analysis/linear_cox/ablate_age_", condition_pretty, "_res2_", multimorbid_acmort_input_type_ind, ".feather")))
  write.csv(res2, file.path(data_root, paste0("analysis/part_1_ablate_age_", condition_pretty, "_res2_", multimorbid_acmort_input_type_ind, ".csv")), row.names = FALSE)

  if (!skip_res3) {
    arrow::write_feather(res3, file.path(data_root, paste0("analysis/linear_cox/ablate_age_", condition_pretty, "_res3_", multimorbid_acmort_input_type_ind, ".feather")))
  }
}

