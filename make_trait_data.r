library(tidyverse)
library(plyr)
library(ukbb)
library(arrow)
library(lubridate)

data_root <- "/home/mica/storage/healthspan/v6.2"
instances <- c(0, 1, 2, 3)
scan_instances <- c(2, 3)

# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------

add_ICD10 <- function(records_outcomes, fields_disease, ICD10_disease, disease_category) {
  results_disease = list()
  for (i in c(1:length(fields_disease))) {
    stopifnot((fields_disease[[i]] %% 2) == 0) # all date fields should be even
    stopifnot(str_length(ICD10_disease[[i]]) == 3)
    name <- strsplit((ukbb_env$field_df %>% filter(field_id == (fields_disease[[i]])))$field_name[[1]],  " ")[[1]]
    stopifnot((name[1] == "Date") & (name[2] %in% ICD10_disease) & (paste(name[3:4], collapse=" ") == "first reported"))

    curr_disease <- get_field(fields_disease[[i]]) %>%
      select(individual_id, instance_id, array_id, diagnosis_date=value) %>%
      mutate(condition_ICD10=factor(ICD10_disease[[i]]), condition_group=factor(disease_category))

    curr_disease <- check_drop_zero_instance_array(curr_disease)
    results_disease[[i]] <- curr_disease
  }

  records_outcomes_new <- bind_rows(results_disease)
  records_outcomes <- rbind(records_outcomes, records_outcomes_new)
  return(records_outcomes)
}

add_cancer <- function(records_outcomes, cancer_records, cancer_codes, cancer_category, cancer_orig_code_type, cancer_names) {
  # individual_id, diagnosis_date=value, condition_ICD10=ICD10_disease[[i]], condition_group=disease_category

  for (i in c(1:length(cancer_codes))) {
    cc <- cancer_codes[[i]]

    if ((cancer_orig_code_type == "ICD9") & (nchar(cc) == 4)) {
      #print(paste("Already full code: ", cc))
      all_codes <- c(paste0("^", cc, "$"))
    } else {
        all_codes <- c(paste0("^", cc, "$"), paste0("^", cc, ".$")) # the 4 character codes don't overlap with any 3 character ones
    }

    print(all_codes)
    results_disease <- cancer_records %>% filter(orig_code_type == cancer_orig_code_type) %>% filter(grepl(paste(all_codes, collapse="|"), code))

    results_disease <- results_disease %>%
        select(individual_id, diagnosis_date) %>%
        mutate(condition_group = factor(cancer_category), condition_ICD10=factor(cc))

    results_disease <- check_drop_zero_instance(results_disease) # has multi array values
    records_outcomes <- rbind(records_outcomes, results_disease)
  }

  return(records_outcomes)
}


add_death <- function(records_outcomes) {
  # individual_id, diagnosis_date=value, condition_ICD10=ICD10_disease[[i]], condition_group=disease_category

  results_disease <- get_field(40000) %>%
    select(individual_id, instance_id, array_id, diagnosis_date=value) %>%
    mutate(condition_ICD10 = factor("All cause mortality"), condition_group = factor("All cause mortality"))

  results_disease <- unique_drop_instance_array(results_disease) # has nonzero instances
  records_outcomes <- rbind(records_outcomes, results_disease)
  return(records_outcomes)
}

check_drop_zero_instance_array <- function(df) {
  stopifnot(all(df$instance_id == 0))

  if (! all(df$array_id == 0)) {
    print("Found non-zero array_id")
    print(unique(df$array_id))
  }

  stopifnot(all(df$array_id == 0))
  df <- unique_drop_instance_array(df)
  return(df)
}


check_drop_zero_instance <- function(df) {
  stopifnot(all(df$instance_id == 0))
  df <- unique_drop_instance_array(df)
  return(df)
}


unique_drop_instance_array <- function(df) {
  if (("diagnosis_date" %in% colnames(df)) & ("condition_ICD10" %in% colnames(df))) {
    df <- distinct(df, individual_id, diagnosis_date, condition_ICD10, .keep_all=TRUE)
  } else {
    df <- distinct(df, individual_id, .keep_all=TRUE)
  }
  df <- df[, !(names(df) %in% c("instance_id", "array_id"))]
  return(df)
}

setup_ukbb("mri_v5")


# ----------------------------------------------------------------------------------------------------------------------
# DEMOG traits
# ----------------------------------------------------------------------------------------------------------------------

sex <- get_field(31) %>% select(individual_id, instance_id, array_id, sex=value) %>% mutate(sex = factor(sex))
sex_from <- c("Male", "Female")
print("Check correspond and numeric type:")
print(head(sex))
sex$sex <- as.numeric(as.vector(mapvalues(sex$sex, from = sex_from, to = c(0, 1))))
print(head(sex))

education <- get_field(6138) %>% select(individual_id, instance_id, array_id, education=value) %>% mutate(education = factor(education))
education_from <- c(
  "College or University degree",
  "A levels/AS levels or equivalent",
  "O levels/GCSEs or equivalent",
  "CSEs or equivalent",
  "NVQ or HND or HNC or equivalent",
  "Other professional qualifications eg: nursing, teaching",
  "None of the above",
  "Prefer not to answer")
print("Check correspond and numeric type:")
print(head(education))
education$education <- as.numeric(as.vector(mapvalues(education$education, from = education_from, to = c(0, 1, 2, 3, 4, 5, 6, 7))))
print(head(education))

race <- get_field(21000) %>% select(individual_id, instance_id, array_id, race=value)
race$race <- as.vector(mapvalues(race$race, from = c("White","British", "Any other white background","Other ethnic group","Pakistani","Irish","Chinese","Indian","White and Asian",
                                                                                           "Any other mixed background","White and Black Caribbean","Mixed","White and Black African","Caribbean","Bangladeshi",
                                                                                           "Any other Asian background","Asian or Asian British","African","Any other Black background","Prefer not to answer", "Do not know"),
                                       to = c("White","White", "White","Other","Asian or Asian British","White","Asian or Asian British","Asian or Asian British","Mixed","Mixed","Mixed","Mixed","Mixed","Black or Black British",
                                             "Asian or Asian British","Asian or Asian British","Asian or Asian British","Black or Black British","Black or Black British","Other","Other")
                       ))
race <- race %>% mutate(race = factor(race))
race_from <- c("White", "Other", "Asian or Asian British", "Mixed", "Black or Black British")
print("Check correspond and numeric type:")
print(head(race))
race$race <- as.numeric(as.vector(mapvalues(race$race, from = race_from, to = c(0, 1, 2, 3, 4))))
print("Check correspond and numeric type:")
print(head(race))

townsend_deprivation <- get_field(22189)  %>% select(individual_id, instance_id, array_id, townsend_deprivation=value)

DOB_year <- get_field(34)  %>% select(individual_id, instance_id, array_id, DOB_year=value)
DOB_month <- get_field(52)  %>% select(individual_id, instance_id, array_id, DOB_month=value)
DOB_month_from <- c("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
print("Check correspond and numeric type:")
print(head(DOB_month))
DOB_month$DOB_month <- as.numeric(as.vector(mapvalues(DOB_month$DOB_month, from = DOB_month_from, to = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))))
print(head(DOB_month))

#censor_ages <- censor_ages %>% distinct(individual_id, .keep_all=TRUE)

print("Drop sizes:")
records_demog_traits <- sex
print(dim(records_demog_traits))
records_demog_traits <- left_join(records_demog_traits, education, by = join_by(individual_id, instance_id, array_id)) %>% drop_na()
print(dim(records_demog_traits))
records_demog_traits <- left_join(records_demog_traits, race, by = join_by(individual_id, instance_id, array_id)) %>% drop_na()
print(dim(records_demog_traits))
records_demog_traits <- left_join(records_demog_traits, townsend_deprivation, by = join_by(individual_id, instance_id, array_id)) %>% drop_na()
print(dim(records_demog_traits))

records_demog_traits <- left_join(records_demog_traits, DOB_year, by = join_by(individual_id, instance_id, array_id)) %>% drop_na()
print(dim(records_demog_traits))
records_demog_traits <- left_join(records_demog_traits, DOB_month, by = join_by(individual_id, instance_id, array_id)) %>% drop_na()
print(dim(records_demog_traits))

records_demog_traits$DOB_approx_day <- 1
records_demog_traits <- records_demog_traits %>% mutate(DOB_approx = make_date(year=DOB_year, month=(DOB_month + 1), day=DOB_approx_day))
records_demog_traits$censor_date <- as.Date("2022-11-30") # https://biobank.ndph.ox.ac.uk/ukb/exinfo.cgi?src=Data_providers_and_dates
records_demog_traits$censor_age <- floor(time_length(records_demog_traits$censor_date - records_demog_traits$DOB_approx, unit="years"))

#records_demog_traits <- left_join(records_demog_traits, censor_ages, by = join_by(individual_id)) %>% drop_na()
print("Check censor ages and dates")
print(head(select(records_demog_traits, DOB_year, DOB_month, DOB_approx_day, DOB_approx, censor_date, censor_age)))
print(dim(records_demog_traits))

# finish
records_demog_traits <- check_drop_zero_instance_array(records_demog_traits)
print(colnames(records_demog_traits))
summary(records_demog_traits)

stopifnot(sum(is.na(records_demog_traits)) == 0)

# ----------------------------------------------------------------------------------------------------------------------
# Assessment info
# ----------------------------------------------------------------------------------------------------------------------

assessment_info <- list(
    # age <- get_field(21003) %>% filter(array_id == 0) %>% select(individual_id, instance_id, age=value),
    centre <- get_field(54) %>% filter(array_id == 0) %>% select(individual_id, instance_id, centre=value) %>% mutate(centre = factor(centre)),
    assessment_date <- get_field(53) %>% filter(array_id == 0) %>% select(individual_id, instance_id, assessment_date=value) %>% mutate(assessment_date = as.Date(assessment_date))
)
assessment_info <- Reduce(function(...) full_join(..., join_by(individual_id, instance_id)), assessment_info) %>% distinct() %>% drop_na()
print(dim(assessment_info))
#assessment_info <- assessment_info %>% mutate(orig_date=1)

imaging_info <- get_idp("scan_date") %>% select(individual_id, instance_id, assessment_date=scan_date) %>%
  mutate(assessment_date = as.Date(assessment_date), centre="Unknown")

# get number of repeats
print("Original shapes")
print(dim(assessment_info))
print(dim(imaging_info))
print(head(imaging_info))

shared <- inner_join(assessment_info, imaging_info, by=c("individual_id", "instance_id")) %>% drop_na()
print("Shared ids, instances")
print(dim(shared))
shared$eq = shared$assessment_date.x == shared$assessment_date.y

neq <- shared %>% filter(eq == FALSE)
print("Shared not equal, for info")
print(dim(neq))
print(head(neq, n=20))

# remove the rows in shared
shared <- shared %>% select(individual_id, instance_id) %>% mutate(in_shared = 1)
imaging_info <- left_join(imaging_info, shared, by=c("individual_id", "instance_id"))
print(head(imaging_info))
print(table(imaging_info$instance_id))

#imaging_info_add <- imaging_info[imaging_info$in_shared != 1, ]
imaging_info_add <- imaging_info[is.na(imaging_info$in_shared), ]

print(head(imaging_info_add))
print(table(imaging_info_add$instance_id))

stopifnot(all(is.na(imaging_info_add$in_shared)))
imaging_info_add <- imaging_info_add[, !(names(imaging_info_add) %in% c("in_shared"))]

print(head(imaging_info_add))
print(table(imaging_info_add$instance_id))

print("New sizes")
assessment_info$orig_date <- 1
imaging_info_add$orig_date <- 0
print(colnames(assessment_info))
print(colnames(imaging_info_add))
print(table(assessment_info$instance_id))
print(head(imaging_info_add))
print(table(imaging_info_add$instance_id))
assessment_info <- rbind(assessment_info, imaging_info_add)
print(dim(assessment_info))
print(table(assessment_info$instance_id))

# drop those without demog
assessment_info <- assessment_info %>% filter(individual_id %in% records_demog_traits$individual_id)
print(dim(assessment_info))
print(table(assessment_info$instance_id))

# add age from original source
age <- get_field(21003) %>% filter(array_id == 0) %>% select(individual_id, instance_id, original_age=value)
assessment_info <- left_join(assessment_info, age, by=c("individual_id", "instance_id")) # with na
print("Num missing ages")
print(sum(is.na(assessment_info)))
print(sum(!complete.cases(assessment_info)))

assessment_info <- left_join(assessment_info, records_demog_traits %>% select(individual_id, DOB_approx), by="individual_id")
stopifnot(sum(is.na(assessment_info$DOB_approx)) == 0)
stopifnot(sum(is.na(assessment_info$assessment_date)) == 0)
assessment_info$comp_age <- floor(time_length(assessment_info$assessment_date - assessment_info$DOB_approx, unit="years"))
assessment_info$age <- ifelse(is.na(assessment_info$original_age), assessment_info$comp_age, assessment_info$original_age)
assessment_info$orig_age <- ifelse(is.na(assessment_info$original_age), 0, 1)
print("Check age comp")
print(head(assessment_info %>% filter(orig_age == 0)))

assessment_info <- assessment_info[, !(names(assessment_info) %in% c("comp_age", "original_age", "DOB_approx"))]

print("Final size")
print(dim(assessment_info))
print(colnames(assessment_info)) # id, instance id, centre, assessment date, age, orig_date, orig_age
print(table(assessment_info$instance_id))
print(table(assessment_info$orig_date))
print(table(assessment_info$orig_age))
stopifnot(sum(is.na(assessment_info)) == 0)

# ----------------------------------------------------------------------------------------------------------------------
# Health outcomes
# ----------------------------------------------------------------------------------------------------------------------

records_outcomes <- data.frame()

non_cancer_fields_path <- file.path(data_root, "data/v5_non_cancer_fields.rda")
if (!file.exists(non_cancer_fields_path)) {
  # get all fields corresponding to the chosen non-cancer ICD10 codes. Do not count at this stage
  # 44 from paperm excludes cancer and tobacco abuse

  # easy specification format, group -> member
  non_cancer_conditions = list(
  list("Hypertension", c("I10", "I11", "I12", "I13", "I14", "I15")),
  list("Lipid metabolism disorders", c("E78")),
  list("Chronic low back pain", c("M40", "M41", "M42", "M43", "M44", "M45", "M47", "M48", "M50", "M51", "M52", "M53", "M54")),
  list("Severe vision reduction", c("H17", "H18", "H25", "H26", "H27", "H28", "H31", "H33", "H34", "H35", "H36", "H40", "H43", "H47", "H54")),
  list("Joint arthritis", c("M15", "M16", "M17", "M18", "M19")),
  list("Diabetes mellitus non-T1", c("E11", "E12", "E13", "E14")),
  list("Chronic ischemic heart disease", c("I20", "I25", "I21")),
  list("Thyroid diseases", c("E01", "E02", "E03", "E04", "E05", "E06", "E07")),
  list("Cardiac arrhythmias", c("I44", "I45", "I46", "I47", "I48", "I49")),
  list("Obesity", c("E66")),
  list("Hyperuricemia/gout", c("E79", "M10")),
  list("Prostatic hyperplasia", c("N40")),
  list("Lower limb varicosis", c("I83", "I87")),
  list("Liver disease", c("K70", "K71", "K72", "K73", "K74", "K76")),
  list("Depression", c("F32", "F33")),
  list("Bronchitis/COPD", c("J40", "J41", "J42", "J43", "J44", "J47")),
  list("Gynecological problems", c("N81", "N84", "N85", "N86", "N87", "N88", "N89", "N90", "N93", "N95")),
  list("Atherosclerosis/PAOD", c("I65", "I66", "I67", "I70", "I73")),
  list("Osteoporosis", c("M80", "M81", "M82")),
  list("Renal insufficiency", c("N18", "N19")),
  list("Cerebral ischemia/chronic stroke", c("I60", "I61", "I62", "I63", "I64", "I69", "G45")),
  list("Cardiac insufficiency", c("I50")),
  list("Severe hearing loss", c("H90", "H91")),
  list("Chronic cholecystitis/gallstones", c("K80", "K81")),
  list("Somatoform disorders", c("F45")),
  list("Hemorrhoids", c("I84")),
  list("Intestinal diverticulosis", c("K57")),
  list("Rheumatoid arthritis/chronic polyarthritis", c("M05", "M06", "M79")),
  list("Cardiac valve disorders", c("I34", "I37")),
  list("Neuropathies", c("G50", "G51", "G52", "G53", "G54", "G55", "G56", "G57", "G58", "G59", "G60", "G61", "G62", "G63", "G64")),
  list("Dizziness", c("H81", "H82", "R42")),
  list("Dementia", c("F00", "F01", "F02", "F03", "F05", "G30", "G31", "R54")),
  list("Urinary incontinence", c("N39", "R32")),
  list("Urinary tract calculi", c("N20")),
  list("Anemia", c("D50", "D51", "D52", "D53", "D55", "D56", "D57", "D58", "D59", "D60", "D61", "D63", "D64")),
  list("Anxiety", c("F40", "F41")),
  list("Psoriasis", c("L40")),
  list("Migraine/chronic headache", c("G43", "G44")),
  list("Parkinson’s disease", c("G20", "G21", "G22")),
  list("Allergy", c("H01", "J30", "L23", "L27", "L56", "K52", "K90", "T78", "T88")),
  list("Chronic gastritis/GERD", c("K21", "K25", "K26", "K27", "K28", "K29")),
  list("Sexual dysfunction", c("F52", "N48")),
  list("Insomnia", c("G47", "F51")),
  list("Hypotension", c("I95"))
  )

  # intermediate format
  non_cancer_str <- list()
  i <- 1
  cnt <- 0
  for (tup in non_cancer_conditions) {
    name <- tup[[1]]
    codes <- tup[[2]]
    codes_str <- paste0(codes, collapse=", ")
    non_cancer_str[[i]] <- c(name, codes_str)
    i <- i + 1
    cnt <- cnt + length(codes)
  }
  print(c("cnt num non cancer ICD10 codes: ", cnt))
  #  non_cancer_str_df <- data.frame(non_cancer_str)
  non_cancer_str_df <- as.data.frame(do.call(rbind, non_cancer_str))
  colnames(non_cancer_str_df) <- c("condition_name", "ICD10_codes")
  print(head(non_cancer_str_df))

  # member -> group name
  non_cancer_df <- data.frame()
  cs <- c(2401, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417)
  ds <- ukbb_env$field_df %>% filter(category_id %in% cs, grepl("Date", field_name)) %>% select(field_id, field_name)
  ds <- unique(ds)
  for (d in unique(ds$field_id)) {
    field_name <- ds[ds$field_id == d, "field_name"][[1]]
    #print("-----")
    #print(c(d, field_name))
    name_split <- strsplit(field_name,  " ")[[1]]
    ICD10_name <- name_split[2]
    stopifnot(str_length(ICD10_name) == 3)
    stopifnot((name_split[1] == "Date") & (paste(name_split[3:4], collapse=" ") == "first reported"))
    field_res <- non_cancer_str_df %>% filter(grepl(ICD10_name, ICD10_codes))
    #print(head(field_res))
    #print("----")
    nr <- nrow(field_res)
    if (nr > 0) {
      stopifnot(nrow(field_res) == 1)
      non_cancer_df <- rbind(non_cancer_df, data.frame(condition_ICD10=ICD10_name, condition_group=field_res$condition_name, condition_field=d))
    }
  }

  print(head(non_cancer_df))
  save(non_cancer_df, non_cancer_conditions, file=non_cancer_fields_path)
  print("Saved non_cancer_fields; rerun script")
  stop()
} else {
  load(file=non_cancer_fields_path)
}

print("loaded non_cancer_df")
print(dim(non_cancer_df))
print(head(non_cancer_df))

print("loaded non_cancer_conditions")
print(length(non_cancer_conditions))
print(non_cancer_conditions[[1]])

##### Add non-cancer disease records

print("Adding non-cancer")
for (tup in non_cancer_conditions) {
  condition_group_curr <- tup[[1]]
  condition_ICD10_codes <- sort(tup[[2]])
  condition_fields <- non_cancer_df %>% filter(condition_group == condition_group_curr)
  condition_fields <- condition_fields[order(condition_fields$condition_ICD10),]
  condition_fields <- condition_fields$condition_field
  records_outcomes <- add_ICD10(records_outcomes, condition_fields, condition_ICD10_codes, condition_group_curr)
  print(c(tup, dim(records_outcomes)))
}

##### Add cancer records
print("Adding cancer")
# https://en.wikipedia.org/wiki/List_of_ICD-9_codes_140%E2%80%93239:_neoplasms

cancer_ICD10 <- get_field(41270, raw=TRUE) %>% select(individual_id, instance_id, array_id, code=value)
cancer_dates <- get_field(41280) %>% select(individual_id, instance_id, array_id, diagnosis_date=value)
cancer_ICD10 <- inner_join(cancer_ICD10, cancer_dates, join_by(individual_id, instance_id, array_id)) %>% mutate(orig_code_type = factor("ICD10")) %>% drop_na() # linked by instance and array_id

print("Cancer summary")
print(head(cancer_ICD10))
print(summary(cancer_ICD10))

print("After removing instance and array")
cancer_ICD10 <- check_drop_zero_instance(cancer_ICD10)
print(head(cancer_ICD10))
print(summary(cancer_ICD10))

disease_category <- "Cancer"
ICD10_codes = c(
"C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", #C00–C14,
"C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", #C15–C26,
"C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", #C30–C39,
"C40", "C41", #C40–C41,
"C43", "C44", #C43–C44,
"C45", "C46", "C47", "C48", "C49", #C45–C49,
"C50", #C50,
"C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", #C51–C58,
"C60", "C61", "C62", "C63", #C60–C63,
"C64", "C65", "C66", "C67", "C68", #C64–C68,
"C69", "C70", "C71", "C72", #C69–C72,
"C73", "C74", "C75", #C73–C75,

"C76", "C77", "C78", "C79", "C80",  #C76–C80,
"C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C89", "C90", "C91", "C92", "C93", "C94", "C95", "C96", #C81–C96,
"C97", #C97,

"D00", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", #D00–D09,
"D37", "D38", "D39", "D40", "D41", "D42", "D43", "D44", "D45", "D46", "D47", "D48" #D37–D48
)

records_outcomes <- add_cancer(records_outcomes, cancer_ICD10, ICD10_codes, disease_category, "ICD10", ICD10_codes)

print(dim(records_outcomes))


#### Any condition excluding death

# individual_id, diagnosis_date=value, condition_ICD10=ICD10_disease[[i]], condition_group=disease_category

print("Adding all cause morbidity")
print(dim(records_outcomes))

prev_len <- length(unique(records_outcomes$individual_id))
print("Len:")
print(prev_len)

any_condition <- records_outcomes %>%
  mutate(diagnosis_date = as.Date(diagnosis_date)) %>%
  group_by(individual_id) %>%
  filter(diagnosis_date == min(diagnosis_date)) %>%
  slice(1) %>% # takes the first occurrence if there is a tie
  ungroup() %>%
  mutate(condition_ICD10 = factor("All cause morbidity"), condition_group=factor("All cause morbidity"))
print(dim(any_condition))

print("Should be same len as before:")
new_len <- length(unique(any_condition$individual_id))
print(new_len)
stopifnot(new_len == prev_len)

records_outcomes <- rbind(records_outcomes, any_condition)

##### Add death (all cause mortality)
print("Adding death/all cause mortality")
records_outcomes <- add_death(records_outcomes)
print(dim(records_outcomes))

##### De duplication

print("Number of people with health outcomes before dedup")
indiv_outcomes <- length(unique(records_outcomes$individual_id))
indiv_outcomes_dim <- dim(records_outcomes)
print(indiv_outcomes)
print(indiv_outcomes_dim)

# Remove duplication of diagnoses, take first one
print("Removing duplicate first instance disease diagnoses")
records_outcomes <- records_outcomes %>%
  mutate(diagnosis_date = as.Date(diagnosis_date)) %>%
  arrange(diagnosis_date) %>%
  distinct(individual_id, condition_group, .keep_all = TRUE) # fine
print(dim(records_outcomes))

print("Number of people with health outcomes after dedup")
indiv_outcomes_dedup <- length(unique(records_outcomes$individual_id))
indiv_outcomes_dedup_dim <- dim(records_outcomes)
print(indiv_outcomes_dedup)
print(indiv_outcomes_dedup_dim)
stopifnot(indiv_outcomes_dedup == indiv_outcomes)


# ----------------------------------------------------------------------------------------------------------------------
# All other traits (array 0, all instances - only df that retains instance_id)
# ----------------------------------------------------------------------------------------------------------------------

other_traits_list <- list(
    # basic body 1
    weight <- get_field(21002) %>% filter(array_id == 0) %>% select(individual_id, instance_id, weight=value),
    height <- get_field(50) %>% filter(array_id == 0) %>% select(individual_id, instance_id, height=value),
    waist_circumference <- get_field(48) %>% filter(array_id == 0) %>% select(individual_id, instance_id, waist_circumference=value),
    grip_strength_L <- get_field(46) %>% filter(array_id == 0) %>% select(individual_id, instance_id, grip_strength_L=value),
    grip_strength_R <- get_field(47) %>% filter(array_id == 0) %>% select(individual_id, instance_id, grip_strength_R=value),

    blood_pressure_systolic <- get_field(4080) %>% filter(array_id == 0) %>% select(individual_id, instance_id, blood_pressure_systolic=value),
    #MET <- get_field(22040) %>% filter(array_id == 0) %>% select(individual_id, instance_id, MET=value),
    alcohol <- get_field(1558) %>% filter(array_id == 0) %>% select(individual_id, instance_id, alcohol=value) %>% mutate(alcohol = factor(alcohol)),
    smoking <-get_field(20116) %>% filter(array_id == 0) %>% select(individual_id, instance_id, smoking=value) %>% mutate(smoking = factor(smoking)),

    # blood https://biobank.ctsu.ox.ac.uk/crystal/label.cgi?id=17518
    alanine_aminotransferase <- get_field(30620) %>% filter(array_id == 0) %>% select(individual_id, instance_id, alanine_aminotransferase=value),
    albumin <- get_field(30600) %>% filter(array_id == 0) %>% select(individual_id, instance_id, albumin=value),
    alkaline_phosphatase <- get_field(30610) %>% filter(array_id == 0) %>% select(individual_id, instance_id, alkaline_phosphatase=value),
    apolipoprotein_A <- get_field(30630) %>% filter(array_id == 0) %>% select(individual_id, instance_id, apolipoprotein_A=value),
    apolipoprotein_B <- get_field(30640) %>% filter(array_id == 0) %>% select(individual_id, instance_id, apolipoprotein_B=value),
    aspartate_aminotransferase <- get_field(30650) %>% filter(array_id == 0) %>% select(individual_id, instance_id, aspartate_aminotransferase=value),
    c_reactive_protein <- get_field(30710) %>% filter(array_id == 0) %>% select(individual_id, instance_id, c_reactive_protein=value),
    calcium <- get_field(30680) %>% filter(array_id == 0) %>% select(individual_id, instance_id, calcium=value),
    cholesterol <- get_field(30690) %>% filter(array_id == 0) %>% select(individual_id, instance_id, cholesterol=value),
    creatinine <- get_field(30700) %>% filter(array_id == 0) %>% select(individual_id, instance_id, creatinine=value),
    cystatin_c <- get_field(30720) %>% filter(array_id == 0) %>% select(individual_id, instance_id, cystatin_c=value),
    #direct_bilirubin <- get_field(30660) %>% filter(array_id == 0) %>% select(individual_id, instance_id, direct_bilirubin=value),
    gamma_glutamyltransferase <- get_field(30730) %>% filter(array_id == 0) %>% select(individual_id, instance_id, gamma_glutamyltransferase=value),
    glucose <- get_field(30740) %>% filter(array_id == 0) %>% select(individual_id, instance_id, glucose=value),
    HbA1c <- get_field(30750) %>% filter(array_id == 0) %>% select(individual_id, instance_id, HbA1c=value),
    HDL_cholesterol <- get_field(30760) %>% filter(array_id == 0) %>% select(individual_id, instance_id, HDL_cholesterol=value),
    IGF1 <- get_field(30770) %>% filter(array_id == 0) %>% select(individual_id, instance_id, IGF1=value),
    LDL_direct <- get_field(30780) %>% filter(array_id == 0) %>% select(individual_id, instance_id, LDL_direct=value),
    #lipoprotein_A <- get_field(30790) %>% filter(array_id == 0) %>% select(individual_id, instance_id, lipoprotein_A=value),
    #oestradiol <- get_field(30800) %>% filter(array_id == 0) %>% select(individual_id, instance_id, oestradiol=value),
    phosphate <- get_field(30810) %>% filter(array_id == 0) %>% select(individual_id, instance_id, phosphate=value),
    #rheumatoid_factor <- get_field(30820) %>% filter(array_id == 0) %>% select(individual_id, instance_id, rheumatoid_factor=value),
    SHBG <- get_field(30830) %>% filter(array_id == 0) %>% select(individual_id, instance_id, SHBG=value),
    #testosterone <- get_field(30850) %>% filter(array_id == 0) %>% select(individual_id, instance_id, testosterone=value),
    total_bilirubin <- get_field(30840) %>% filter(array_id == 0) %>% select(individual_id, instance_id, total_bilirubin=value),
    total_protein <- get_field(30860) %>% filter(array_id == 0) %>% select(individual_id, instance_id, total_protein=value),
    triglycerides <- get_field(30870) %>% filter(array_id == 0) %>% select(individual_id, instance_id, triglycerides=value),
    urate <- get_field(30880) %>% filter(array_id == 0) %>% select(individual_id, instance_id, urate=value),
    urea <- get_field(30670) %>% filter(array_id == 0) %>% select(individual_id, instance_id, urea=value),
    vitamin_D <- get_field(30890) %>% filter(array_id == 0) %>% select(individual_id, instance_id, vitamin_D=value)
)

print("Other traits raw count (individual_id count should be approx same before and after):")
for (i in 1:length(other_traits_list)) {
  print(names(other_traits_list[[i]]))

  len_before <- length(unique(other_traits_list[[i]]$individual_id))
  print(len_before)
  print(dim(other_traits_list[[i]]))
  print(unique(other_traits_list[[i]]$instance_id))

  # for each individual and instance, add age and assessment centre
  trait_name <- colnames(other_traits_list[[i]])
  trait_name <- trait_name[!trait_name %in% c("individual_id", "instance_id")][[1]]
  print(trait_name) # check
  inst_name <- paste("age", trait_name, sep="_")
  inst_centre <- paste("centre", trait_name, sep="_")
  inst_assessment_date <- paste("assessment_date", trait_name, sep="_")
  inst_orig_date <- paste("orig_date", trait_name, sep="_")
  inst_orig_age <- paste("orig_age", trait_name, sep="_")

  other_traits_list[[i]] <-  left_join(other_traits_list[[i]], assessment_info, by = join_by(individual_id, instance_id)) %>%
                              drop_na()
  colnames(other_traits_list[[i]])[which(names(other_traits_list[[i]]) == "age")] <- inst_name
  colnames(other_traits_list[[i]])[which(names(other_traits_list[[i]]) == "centre")] <- inst_centre
  colnames(other_traits_list[[i]])[which(names(other_traits_list[[i]]) == "assessment_date")] <- inst_assessment_date
  colnames(other_traits_list[[i]])[which(names(other_traits_list[[i]]) == "orig_date")] <- inst_orig_date
  colnames(other_traits_list[[i]])[which(names(other_traits_list[[i]]) == "orig_age")] <- inst_orig_age

  print(head(other_traits_list[[i]]))

  # for each individual, get latest entry up to instance 2
  other_traits_list[[i]] <- other_traits_list[[i]] %>%
    filter(instance_id <= 2) %>%
    group_by(individual_id) %>%
    filter(row_number(desc(instance_id)) == 1) %>%
    ungroup()

  print(unique(other_traits_list[[i]]$instance_id))
  other_traits_list[[i]] <- unique_drop_instance_array(other_traits_list[[i]])

  len_after <- length(unique(other_traits_list[[i]]$individual_id))
  print(c(len_after, len_after - len_before))
  print(dim(other_traits_list[[i]]))
  print("----")
}

# keep NAs/uneven class sizes in dataset
records_other_traits <- Reduce(function(...) full_join(..., join_by(individual_id)), other_traits_list)

# basic body 2
records_other_traits$BMI <- (records_other_traits$weight / (records_other_traits$height / 100.0)^2)
records_other_traits$age_BMI <- records_other_traits$age_weight
records_other_traits$centre_BMI <- records_other_traits$centre_weight

records_other_traits$grip_strength <- (records_other_traits$grip_strength_L +  records_other_traits$grip_strength_R) / 2.0
records_other_traits$age_grip_strength <- records_other_traits$age_grip_strength_L
records_other_traits$centre_grip_strength <- records_other_traits$centre_grip_strength_L

print(colnames(records_other_traits))
records_other_traits$bb_age <- rowMeans(records_other_traits %>% select(age_height, age_weight, age_waist_circumference, age_grip_strength, age_blood_pressure_systolic, age_alcohol, age_smoking, age_BMI))
records_other_traits$blood_age <- rowMeans(records_other_traits %>% select(
age_alanine_aminotransferase,
age_albumin,
age_alkaline_phosphatase,
age_apolipoprotein_A,
age_apolipoprotein_B,
age_aspartate_aminotransferase,
age_c_reactive_protein,
age_calcium,
age_cholesterol,
age_creatinine,
age_cystatin_c,
age_gamma_glutamyltransferase,
age_glucose,
age_HbA1c,
age_HDL_cholesterol,
age_IGF1,
age_LDL_direct,
age_phosphate,
age_SHBG,
age_total_bilirubin,
age_total_protein,
age_triglycerides,
age_urate,
age_urea,
age_vitamin_D))

# convert factors
alcohol_from = c("Daily or almost daily", "Three or four times a week", "Once or twice a week", "One to three times a month", "Special occasions only", "Never", "Prefer not to answer")
print("Check correspond and numeric type:")
print(head(records_other_traits["alcohol"]))
records_other_traits$alcohol <- as.numeric(as.vector(mapvalues(records_other_traits$alcohol, from = alcohol_from, to = c(0, 1, 2, 3, 4, 5, 6))))
print(head(records_other_traits["alcohol"]))

smoking_from = c("Prefer not to answer", "Never", "Previous", "Current")
print("Check correspond and numeric type:")
print(head(records_other_traits["smoking"]))
records_other_traits$smoking <- as.numeric(as.vector(mapvalues(records_other_traits$smoking, from = smoking_from, to = c(0, 1, 2, 3))))
print(head(records_other_traits["smoking"]))

print(dim(records_other_traits))
print(summary(records_other_traits))

cs <- colnames(records_other_traits)
#include_traits <- c()
for (i in 1:length(cs)) {
  curr_name <- cs[[i]]
  num_na <- sum(!complete.cases(records_other_traits[curr_name]))
  if ((num_na >= 75000) & !grepl("age", curr_name, fixed=TRUE) & !grepl("centre", curr_name, fixed=TRUE) & !grepl("assessment_date", curr_name, fixed=TRUE)) {
    print("Should have excluded this; halting")
    print(curr_name)
    print(num_na)
    stopifnot(TRUE)
  }
}

# Drop anyone who doesn't have full metrics
print("Dropping na for other_traits:")
records_other_traits <- records_other_traits %>% drop_na()
print(dim(records_other_traits))
print(summary(records_other_traits))

# ----------------------------------------------------------------------------------------------------------------------
# Summarise and save
# ----------------------------------------------------------------------------------------------------------------------

print("Saving, summaries:")
print(dim(records_demog_traits))
print(dim(records_other_traits))
print(dim(records_outcomes))
print(dim(assessment_info))

print(summary(records_demog_traits))
print(summary(records_other_traits))
print(summary(records_outcomes))
print(summary(assessment_info))

save(records_demog_traits, records_other_traits, records_outcomes,
  assessment_info, sex_from, education_from, race_from, alcohol_from, smoking_from,
  file = file.path(data_root, "data/make_trait_data.rda"))

arrow::write_feather(assessment_info, file.path(data_root, "data/assessment_info.feather"))
arrow::write_feather(records_demog_traits, file.path(data_root, "data/records_demog_traits.feather"))
arrow::write_feather(records_other_traits, file.path(data_root, "data/records_other_traits.feather"))
arrow::write_feather(records_outcomes, file.path(data_root, "data/records_outcomes.feather"))

