library(tidyverse)
library(plyr)
library(arrow)
library(janitor)

data_root <- "/home/mica/storage/healthspan/v6.2"

trait_path <- file.path(data_root, "data/make_trait_data.rda")
load(file=trait_path)

# Convert back to factors
records_demog_traits$sex <- as.vector(mapvalues(records_demog_traits$sex, to = sex_from, from = (seq(1:length(sex_from)) - 1) ))
records_demog_traits$education <- as.vector(mapvalues(records_demog_traits$education, to = education_from, from = (seq(1:length(education_from)) - 1) ))
records_demog_traits$race <- as.vector(mapvalues(records_demog_traits$race, to = race_from, from = (seq(1:length(race_from)) - 1) ))
records_demog_traits <- records_demog_traits %>% mutate(sex = factor(sex), education = factor(education), race = factor(race))

records_other_traits$alcohol <- as.vector(mapvalues(records_other_traits$alcohol, to = alcohol_from, from = (seq(1:length(alcohol_from)) - 1) ))
records_other_traits$smoking <- as.vector(mapvalues(records_other_traits$smoking, to = smoking_from, from = (seq(1:length(smoking_from)) - 1) ))
records_other_traits <- records_other_traits %>% mutate(alcohol = factor(alcohol), smoking = factor(smoking))

# ----------------------------------------------------------------------------
# Analyse
# ----------------------------------------------------------------------------
print("(1)")

# get main_ids and analyse
ids_main <- arrow::read_feather(file.path(data_root, "data/ids_main.feather"))
print("ids_main")
print(dim(ids_main))

print("checking membership")
print(ids_main %>% filter(individual_id %in% c(1001483, 1000313, 1000012)))
print(records_demog_traits %>% filter(individual_id %in% c(1001483, 1000313, 1000012)))
print(records_other_traits %>% filter(individual_id %in% c(1001483, 1000313, 1000012)))

print("Instances (all)")
print(dim(records_demog_traits))

records_demog_traits <- inner_join(records_demog_traits, ids_main, by="individual_id")
print("Num unique individuals, should be same as ids_main:")
kept_unique_ids <- unique(records_demog_traits$individual_id)
total_unique <- length(kept_unique_ids)
stopifnot(total_unique == length(ids_main$individual_id))
print(total_unique)

write.csv(total_unique, file.path(data_root, "analysis/demogr/total_unique.csv"), row.names = FALSE)

records_demog_traits_f <- records_demog_traits %>% filter(sex=="Female")
records_demog_traits_m <- records_demog_traits %>% filter(sex=="Male")

print("Female")
print(head(records_demog_traits_f))
print(summary(records_demog_traits_f))
print(dim(records_demog_traits_f))

total_unique_f <- length(unique(records_demog_traits_f$individual_id))
print(total_unique_f)
write.csv(total_unique_f, file.path(data_root, "analysis/demogr/total_unique_f.csv"), row.names = FALSE)

print("Male")
print(head(records_demog_traits_m))
print(summary(records_demog_traits_m))
print(dim(records_demog_traits_m))

total_unique_m <- length(unique(records_demog_traits_m$individual_id))
print(total_unique_m)
write.csv(total_unique_m, file.path(data_root, "analysis/demogr/total_unique_m.csv"), row.names = FALSE)

race_f <- records_demog_traits_f %>%
  group_by(race) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>% # .groups="keep"
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")

race_f
write.csv(race_f, file.path(data_root, "analysis/demogr/race_f.csv"), row.names = FALSE)

race_m <- records_demog_traits_m %>%
  group_by(race) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>% # .groups="keep"
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
race_m
write.csv(race_m, file.path(data_root, "analysis/demogr/race_m.csv"), row.names = FALSE)

education_f <- records_demog_traits_f %>%
  group_by(education) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>% # .groups="keep"
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
education_f
write.csv(education_f, file.path(data_root, "analysis/demogr/education_f.csv"), row.names = FALSE)

education_m <- records_demog_traits_m %>%
  group_by(education) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>% # .groups="keep"
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
education_m
write.csv(education_m, file.path(data_root, "analysis/demogr/education_m.csv"), row.names = FALSE)

demog_traits_continuous <- records_demog_traits %>%
  group_by(sex) %>%
  dplyr::summarize(
            mean_DOB_year = mean(DOB_year, na.rm = TRUE),
            sd_DOB_year = sd(DOB_year, na.rm = TRUE),
            min_DOB_year = min(DOB_year, na.rm = TRUE),
            max_DOB_year = max(DOB_year, na.rm = TRUE),
            iqr_DOB_year = IQR(DOB_year, na.rm = TRUE),
            unique_DOB_years = n_distinct(DOB_year),

            mean_townsend_deprivation = mean(townsend_deprivation, na.rm=FALSE),
  )
demog_traits_continuous
write.csv(demog_traits_continuous, file.path(data_root, "analysis/demogr/demog_traits_continuous.csv"), row.names = FALSE)

# diseases
print("(2)")

# records_demog_traits is already ids subset
records_outcomes_join <- left_join(records_outcomes, records_demog_traits, by="individual_id", unmatched="drop", relationship="many-to-one")
print(head(records_outcomes_join))
print(dim(records_outcomes_join))

print("Number of unique diseases in cleaned records_outcomes_join")
print(length(unique(records_outcomes_join$condition_group)))

print("Disease counts (F, M):")
records_outcomes_f <- records_outcomes_join %>% filter(sex=="Female")
d_f <- table(as.character(records_outcomes_f$condition_group)) # set as character otherwise prints factors without count
d_f
write.csv(d_f, file.path(data_root, "analysis/demogr/d_f.csv"), row.names = FALSE)

records_outcomes_m <- records_outcomes_join %>% filter(sex=="Male")
d_m <- table(as.character(records_outcomes_m$condition_group))
d_m
write.csv(d_m, file.path(data_root, "analysis/demogr/d_m.csv"), row.names = FALSE)


#  3 records_traits_join - one per individual id
print("(3)")
print("Other traits")
print(dim(records_other_traits))
# records_demog_traits already ids subset

print("Counting NAs")
print(sum(is.na(records_other_traits)))
print(sum(is.na(records_demog_traits)))

records_traits_join <- left_join(records_demog_traits, records_other_traits, by="individual_id", unmatched="drop", relationship="one-to-one")
print(sum(is.na(records_traits_join)))
print(records_traits_join %>% filter(individual_id %in% c(1001483, 1000313, 1000012)) %>% select(c("individual_id", "sex")))

#records_traits_join[c("individual_id", "sex")] %>% as_tibble() %>% print(n=200)
print(colnames(records_traits_join))

# get centre distribution for instance 2 of kept ids (ie imaging variable)
centres_instance_2 <- assessment_info %>%
  filter(individual_id %in% kept_unique_ids, instance_id == 2) %>%
  group_by(centre) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>%
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
centres_instance_2
write.csv(centres_instance_2, file.path(data_root, "analysis/demogr/centres_instance_2.csv"), row.names = FALSE)

records_traits_join_f <- records_traits_join %>% filter(sex=="Female")
records_traits_join_m <- records_traits_join %>% filter(sex=="Male")

# for these 2, keep matched ids only - redundant really
smoking_f <- records_traits_join_f %>%
  filter(individual_id %in% kept_unique_ids) %>%
  group_by(smoking) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>%
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
smoking_f
write.csv(smoking_f, file.path(data_root, "analysis/demogr/smoking_f.csv"), row.names = FALSE)

smoking_m <- records_traits_join_m %>%
  filter(individual_id %in% kept_unique_ids) %>%
  group_by(smoking) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>%
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
smoking_m
write.csv(smoking_m, file.path(data_root, "analysis/demogr/smoking_m.csv"), row.names = FALSE)

alcohol_f <- records_traits_join_f %>%
  filter(individual_id %in% kept_unique_ids) %>%
  group_by(alcohol) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>%
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
alcohol_f
write.csv(alcohol_f, file.path(data_root, "analysis/demogr/alcohol_f.csv"), row.names = FALSE)

alcohol_m <- records_traits_join_m %>%
  filter(individual_id %in% kept_unique_ids) %>%
  group_by(alcohol) %>%
  dplyr::summarise(n = n(), nuniq = n_distinct(individual_id)) %>%
  mutate(freq = n / sum(n)) %>%
  adorn_totals("row")
alcohol_m
write.csv(alcohol_m, file.path(data_root, "analysis/demogr/alcohol_m.csv"), row.names = FALSE)

print(colnames(records_traits_join))
print("All traits")
# non-factor
traits_summary <- records_traits_join %>%
  group_by(sex) %>%
  dplyr::summarize(
            n = n(),
            nuniq = n_distinct(individual_id),
            mean_height = mean(height, na.rm = TRUE),
            mean_weight = mean(weight, na.rm = TRUE),
            mean_waist_circumference = mean(waist_circumference, na.rm = TRUE),
            mean_blood_pressure_systolic = mean(blood_pressure_systolic, na.rm = TRUE),
            mean_grip_strength = mean(grip_strength, na.rm = TRUE),
            mean_BMI = mean(BMI, na.rm = TRUE),

            # blood markers
            mean_alanine_aminotransferase = mean(alanine_aminotransferase, na.rm = TRUE),
            mean_albumin = mean(albumin, na.rm = TRUE),
            mean_alkaline_phosphatase= mean(alkaline_phosphatase, na.rm = TRUE),
            mean_apolipoprotein_A = mean(apolipoprotein_A, na.rm = TRUE),
            mean_apolipoprotein_B = mean(apolipoprotein_B, na.rm = TRUE),
            mean_aspartate_aminotransferase = mean(aspartate_aminotransferase, na.rm = TRUE),

            mean_c_reactive_protein = mean(c_reactive_protein, na.rm = TRUE),
            mean_calcium = mean(calcium, na.rm = TRUE),
            mean_cholesterol = mean(cholesterol, na.rm = TRUE),
            mean_creatinine = mean(creatinine, na.rm = TRUE),
            mean_cystatin_c = mean(cystatin_c, na.rm = TRUE),
            #mean_direct_bilirubin = mean(direct_bilirubin, na.rm = TRUE),
            mean_gamma_glutamyltransferase = mean(gamma_glutamyltransferase, na.rm = TRUE),
            mean_glucose = mean(glucose, na.rm = TRUE),
            mean_HbA1c = mean(HbA1c, na.rm = TRUE),
            mean_HDL_cholesterol = mean(HDL_cholesterol, na.rm = TRUE),
            mean_IGF1 = mean(IGF1, na.rm = TRUE),

            mean_LDL_direct= mean(LDL_direct, na.rm = TRUE),
            #mean_lipoprotein_A = mean(lipoprotein_A, na.rm = TRUE),
            #mean_oestradiol = mean(oestradiol, na.rm = TRUE),
            mean_phosphate = mean(phosphate, na.rm = TRUE),
            #mean_rheumatoid_factor = mean(rheumatoid_factor, na.rm = TRUE),
            mean_SHBG = mean(SHBG, na.rm = TRUE),
            #mean_testosterone = mean(testosterone, na.rm = TRUE),
            mean_total_bilirubin = mean(total_bilirubin, na.rm = TRUE),
            mean_total_protein = mean(total_protein, na.rm = TRUE),
            mean_triglycerides = mean(triglycerides, na.rm = TRUE),
            mean_urate = mean(urate, na.rm = TRUE),
            mean_urea = mean(urea, na.rm = TRUE),
            mean_vitamin_D = mean(vitamin_D, na.rm = TRUE),

            # ages
            mean_age_height = mean(age_height, na.rm = TRUE),
            mean_age_weight = mean(age_weight, na.rm = TRUE),
            mean_age_waist_circumference = mean(age_waist_circumference, na.rm = TRUE),
            mean_age_blood_pressure_systolic = mean(age_blood_pressure_systolic, na.rm = TRUE),
            mean_age_grip_strength = mean(age_grip_strength, na.rm = TRUE),
            mean_age_BMI = mean(age_BMI, na.rm = TRUE),

            # blood markers
            mean_age_alanine_aminotransferase = mean(age_alanine_aminotransferase, na.rm = TRUE),
            mean_age_albumin = mean(age_albumin, na.rm = TRUE),
            mean_age_alkaline_phosphatase= mean(age_alkaline_phosphatase, na.rm = TRUE),
            mean_age_apolipoprotein_A = mean(age_apolipoprotein_A, na.rm = TRUE),
            mean_age_apolipoprotein_B = mean(age_apolipoprotein_B, na.rm = TRUE),
            mean_age_aspartate_aminotransferase = mean(age_aspartate_aminotransferase, na.rm = TRUE),

            mean_age_c_reactive_protein = mean(age_c_reactive_protein, na.rm = TRUE),
            mean_age_calcium = mean(age_calcium, na.rm = TRUE),
            mean_age_cholesterol = mean(age_cholesterol, na.rm = TRUE),
            mean_age_creatinine = mean(age_creatinine, na.rm = TRUE),
            mean_age_cystatin_c = mean(age_cystatin_c, na.rm = TRUE),
            #mean_age_direct_bilirubin = mean(age_direct_bilirubin, na.rm = TRUE),
            mean_age_gamma_glutamyltransferase = mean(age_gamma_glutamyltransferase, na.rm = TRUE),
            mean_age_glucose = mean(age_glucose, na.rm = TRUE),
            mean_age_HbA1c = mean(age_HbA1c, na.rm = TRUE),
            mean_age_HDL_cholesterol = mean(age_HDL_cholesterol, na.rm = TRUE),
            mean_age_IGF1 = mean(age_IGF1, na.rm = TRUE),

            mean_age_LDL_direct= mean(age_LDL_direct, na.rm = TRUE),
            #mean_age_lipoprotein_A = mean(age_lipoprotein_A, na.rm = TRUE),
            #mean_age_oestradiol = mean(age_oestradiol, na.rm = TRUE),
            mean_age_phosphate = mean(age_phosphate, na.rm = TRUE),
            #mean_age_rheumatoid_factor = mean(age_rheumatoid_factor, na.rm = TRUE),
            mean_age_SHBG = mean(age_SHBG, na.rm = TRUE),
            #mean_age_testosterone = mean(age_testosterone, na.rm = TRUE),
            mean_age_total_bilirubin = mean(age_total_bilirubin, na.rm = TRUE),
            mean_age_total_protein = mean(age_total_protein, na.rm = TRUE),
            mean_age_triglycerides = mean(age_triglycerides, na.rm = TRUE),
            mean_age_urate = mean(age_urate, na.rm = TRUE),
            mean_age_urea = mean(age_urea, na.rm = TRUE),
            mean_age_vitamin_D = mean(age_vitamin_D, na.rm = TRUE)

      )
traits_summary
write.csv(traits_summary, file.path(data_root, "analysis/demogr/traits_summary.csv"), row.names = FALSE)
