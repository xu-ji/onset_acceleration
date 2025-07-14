library(tidyverse)
library(plyr)
library(arrow)
library(survival)
library(broom)


data_root <- "/home/mica/storage/healthspan/v6.2"
fmodel <- "nn_demog_bb_bbc_sbc_blood_5e-05_1e-05_0.0_256_0_None"

print(fmodel)

for (adjust_name in c("basicplus", "full")) {
  for (quartiles_suff in c("q", "h", "continuous")) {

    print(paste0("doing setting ", adjust_name, " ", quartiles_suff))

    trait_path <- file.path(data_root, "data/make_trait_data.rda")
    load(file=trait_path)

    ids_main <- arrow::read_feather(file.path(data_root, "data/ids_main.feather"))
    print("ids_main")
    print(dim(ids_main))

    hazards_main <- arrow::read_feather(file.path(data_root, paste0("analysis/collected_data_", fmodel, ".feather")))
    print("hazards_main")
    print(dim(hazards_main)) # number of people x num conditions 47
    #print(head(hazards_main))
    hazards_main <- left_join(ids_main, hazards_main, by=join_by(individual_id))
    stopifnot(sum(is.na(hazards_main)) == 0)

    conditions_ordered <- arrow::read_feather(file.path(data_root, paste0("analysis/conditions_ordered_", fmodel, ".feather")))
    print("conditions_ordered")
    print(dim(conditions_ordered))
    print(conditions_ordered)
    conditions <- conditions_ordered$condition_group
    num_conditions = length(conditions)
    print(num_conditions)
    print(conditions)

    sex_f <- c("Thyroid diseases", "Gynecological problems", "Osteoporosis")
    sex_m <- c("Prostatic hyperplasia", "Hyperuricemia/gout", "Sexual dysfunction")
    sex_conds <- rep(NA, num_conditions)
    for (ci1 in 1:num_conditions) {
      if (conditions[ci1] %in% sex_f) {
        sex_conds[ci1] <- 1
      }
      if (conditions[ci1] %in% sex_m) {
        sex_conds[ci1] <- 0
      }
    }
    print(c(sex_f, sex_m))
    print(sex_conds)
    print(sex_conds[1])

    # get age and date at instance 2, add to ids_main
    assessment_info <- assessment_info %>% filter(instance_id == 2)
    ids_main <-  left_join(ids_main, assessment_info, by = join_by(individual_id))
    print(colnames(ids_main))
    ids_main <- ids_main %>% select(individual_id, age, assessment_date)
    stopifnot(sum(is.na(assessment_info)) == 0)

    # combine ids_main with demog and other traits
    dim1 <- dim(ids_main)[1]
    records_key_traits <- left_join(ids_main, records_demog_traits, by = join_by(individual_id))
    records_key_traits <- left_join(records_key_traits, records_other_traits, by=join_by(individual_id))
    print(colnames(records_key_traits))
    records_key_traits <- records_key_traits %>% select(individual_id,
      age, assessment_date, # instance 2 date
      sex, race, education, townsend_deprivation, censor_date,
      height, weight, waist_circumference, grip_strength, blood_pressure_systolic, alcohol, smoking, BMI)
    stopifnot(sum(is.na(records_key_traits)) == 0)
    dim2 <- dim(records_key_traits)[1]
    print(c(dim(ids_main), dim(records_key_traits)))
    stopifnot(dim1 == dim2)

    print("Check date type")
    print(head(select(records_outcomes, individual_id, condition_group, diagnosis_date)))
    uncensored_counts <- rep(NA, num_conditions)

    # same sex model
    for (ci1 in 1:num_conditions) { # predict from
      for (ci2 in 1:num_conditions) { # to

        outcomes_ci2 <- records_outcomes %>% filter(condition_group == conditions[ci2]) %>% select(individual_id, diagnosis_date)
        print(c("doing", ci1, ci2, conditions[ci1], conditions[ci2]))

        print(dim(outcomes_ci2))
        print(sum(is.na(outcomes_ci2$diagnosis_date)))

        outcomes_ci2 <- left_join(records_key_traits, outcomes_ci2, by = join_by(individual_id)) # can only have 1 entry per disease
        print(dim(outcomes_ci2))
        print(sum(is.na(outcomes_ci2$diagnosis_date))) # people without this condition

        # preexisting diagnosis
        pre_existing_ci2 <- outcomes_ci2 %>% drop_na() %>% filter(diagnosis_date <= assessment_date) %>% distinct(individual_id)
        pre_existing_ci2 <- pre_existing_ci2$individual_id
        outcomes_ci2 <- outcomes_ci2 %>% filter(!(individual_id %in% pre_existing_ci2))
        print(dim(outcomes_ci2))
        print(sum(is.na(outcomes_ci2$diagnosis_date)))

        post_existing_ci2 <- outcomes_ci2 %>% drop_na()
        post_existing_ci2 <- post_existing_ci2 %>% filter(diagnosis_date > assessment_date) # not nec
        post_existing_ci2 <- post_existing_ci2 %>% distinct(individual_id) # not nec

        #print(dim(post_existing_ci2))
        ## todo
        #if (conditions[ci2] == "All cause morbidity") {
        #  hazard_colname <- paste0(conditions[ci1], " log hazard q", collapse=" ")
        #  hazards_ci1 <- hazards_main[c("individual_id", hazard_colname)]
        #  outcomes_ci2 <- left_join(post_existing_ci2, hazards_ci1, by = join_by(individual_id))
        #  colnames(outcomes_ci2)[colnames(outcomes_ci2) == hazard_colname] <- "log_hazard_q"
        #  print("Number of outcomes in top quartile for All cause morbidity:")
        #  print(dim(outcomes_ci2 %>% filter(log_hazard_q == 3)))
        #  stopifnot(FALSE)
        #}
        post_existing_ci2 <- outcomes_ci2$individual_id
        uncensored_counts[[ci2]] <- length(post_existing_ci2)

        outcomes_ci2$censor_time <- round(time_length(outcomes_ci2$censor_date - outcomes_ci2$assessment_date, unit="months")) # non na all
        stopifnot(sum(is.na(outcomes_ci2$censor_time)) == 0)
        outcomes_ci2$diagnosis_time <- round(time_length(outcomes_ci2$diagnosis_date - outcomes_ci2$assessment_date, unit="months")) # some na
        outcomes_ci2$status <- ifelse(is.na(outcomes_ci2$diagnosis_time), 0, 1) # if it's na, status is 0 else 1
        print(c("is na (no diagnosis) should be same", sum(is.na(outcomes_ci2$diagnosis_time)), sum(outcomes_ci2$status == 0), "total shape", dim(outcomes_ci2$diagnosis_time)))

        outcomes_ci2$time <- ifelse(is.na(outcomes_ci2$diagnosis_time), outcomes_ci2$censor_time, outcomes_ci2$diagnosis_time)

        #print(colnames(hazards_main))
        #print(paste0(conditions[ci1], " log hazard q", collapse=" "))
        hazard_colname <- paste0(conditions[ci1], " log hazard q", collapse=" ")
        hazards_ci1 <- hazards_main[c("individual_id", hazard_colname)]
        outcomes_ci2 <- left_join(outcomes_ci2, hazards_ci1, by = join_by(individual_id))
        colnames(outcomes_ci2)[colnames(outcomes_ci2) == hazard_colname] <- "log_hazard_q"

        stopifnot(all(outcomes_ci2$log_hazard_q >= 0))
        stopifnot(all(outcomes_ci2$log_hazard_q <= 3))
        outcomes_ci2$log_hazard_h <- as.integer(outcomes_ci2$log_hazard_q >= 2)

        hazard_colname_cont <- paste0(conditions[ci1], " log hazard", collapse=" ")
        hazards_ci1_cont <- hazards_main[c("individual_id", hazard_colname_cont)]
        outcomes_ci2 <- left_join(outcomes_ci2, hazards_ci1_cont, by = join_by(individual_id))
        colnames(outcomes_ci2)[colnames(outcomes_ci2) == hazard_colname_cont] <- "log_hazard"
        #  reference levels are for healthy

        # subselect if either variable is sex specific, preferentially with the outcome sex

        sc <- NULL
        if (conditions[ci2] %in% c(sex_f, sex_m)) {
          sc <- sex_conds[ci2]
          print("Found 1")
        } else if (conditions[ci1] %in% c(sex_f, sex_m)) {
          sc <- sex_conds[ci1]
          print("Found 2")

        }
        print(c("sex cond", sc))

        if (quartiles_suff == "q") {
          if (adjust_name == "full") {
            if ((conditions[ci2] %in% c(sex_f, sex_m))  | (conditions[ci1] %in% c(sex_f, sex_m))) {
              print("Training with sex cond")
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_q), ref="0") +
                scale(age) + scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI),
                data = outcomes_ci2, subset=(sex==sc))
            } else {
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_q), ref="0") +
                scale(age) + scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI),
                data = outcomes_ci2)
            }
          } else if (adjust_name == "basicplus") {
            if ((conditions[ci2] %in% c(sex_f, sex_m))  | (conditions[ci1] %in% c(sex_f, sex_m))) {
              print("Training with sex cond")
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_q), ref="0") +
                scale(age) + scale(height) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1"),
                data = outcomes_ci2, subset=(sex==sc))
            } else {
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_q), ref="0") +
                scale(age) + scale(height) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1"),
                data = outcomes_ci2)
            }
          }

        } else if (quartiles_suff == "h") {
          if (adjust_name == "full") {
            if ((conditions[ci2] %in% c(sex_f, sex_m))  | (conditions[ci1] %in% c(sex_f, sex_m))) {
              print("Training with sex cond")
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_h), ref="0") +
                scale(age) + scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI),
                data = outcomes_ci2, subset=(sex==sc))
            } else {
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_h), ref="0") +
                scale(age) + scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI),
                data = outcomes_ci2)
            }
          } else if (adjust_name == "basicplus") {
            if ((conditions[ci2] %in% c(sex_f, sex_m))  | (conditions[ci1] %in% c(sex_f, sex_m))) {
              print("Training with sex cond")
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_h), ref="0") +
                scale(age) + scale(height) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1"),
                data = outcomes_ci2, subset=(sex==sc))
            } else {
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                relevel(as.factor(log_hazard_h), ref="0") +
                scale(age) + scale(height) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1"),
                data = outcomes_ci2)
            }
          }
        } else if (quartiles_suff == "continuous") {
          if (adjust_name == "full") {
            if ((conditions[ci2] %in% c(sex_f, sex_m))  | (conditions[ci1] %in% c(sex_f, sex_m))) {
              print("Training with sex cond")
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                scale(log_hazard) +
                scale(age) + scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI),
                data = outcomes_ci2, subset=(sex==sc))
            } else {
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                scale(log_hazard) +
                scale(age) + scale(height) + scale(waist_circumference) + scale(grip_strength) + scale(blood_pressure_systolic) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1") + scale(BMI),
                data = outcomes_ci2)
            }
          } else if (adjust_name == "basicplus") {
            if ((conditions[ci2] %in% c(sex_f, sex_m))  | (conditions[ci1] %in% c(sex_f, sex_m))) {
              print("Training with sex cond")
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                scale(log_hazard) +
                scale(age) + scale(height) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1"),
                data = outcomes_ci2, subset=(sex==sc))
            } else {
              res_cox <- coxph(Surv(time, status) ~ relevel(as.factor(sex), ref="0") + relevel(as.factor(race), ref="0") + relevel(as.factor(education), ref="0") + scale(townsend_deprivation) +
                scale(log_hazard) +
                scale(age) + scale(height) + relevel(as.factor(alcohol), ref="5") + relevel(as.factor(smoking), ref="1"),
                data = outcomes_ci2)
            }
          }
        } else {
            stopifnot(FALSE)
        }

        #print(summary(res_cox))
        res1 <- as.data.frame(tidy(res_cox))
        res2 <- as.data.frame(glance(res_cox))

        if (conditions[ci2] == "Diabetes mellitus non-T1") {
          print("Example diabetes:")
          print(adjust_name)
          print(quartiles_suff)
          print(res1)
          print(res2)
        }

        tryCatch(
          expr = {
            zp <- cox.zph(res_cox, terms = TRUE)
            res3 <- as.data.frame(zp$table)
            res3$term_name <- row.names(res3) # store explicitly for feather
            #print(res3)
            #print(res3[res3$term_name == "relevel(as.factor(log_hazard_h), ref = \"0\")", ]$p)
            skip_res3 <- FALSE
          },
          error = function(e){
            message('Caught zph error')
            print(e)
            skip_res3 <- TRUE
          }
        )

        arrow::write_feather(res1, file.path(data_root, paste0("analysis/linear_cox/", adjust_name, "_", ci1, "_", ci2, "_res1_", quartiles_suff, ".feather")))
        arrow::write_feather(res2, file.path(data_root, paste0("analysis/linear_cox/", adjust_name, "_", ci1, "_", ci2, "_res2_", quartiles_suff, ".feather")))
        if (!skip_res3) {
          arrow::write_feather(res3, file.path(data_root, paste0("analysis/linear_cox/", adjust_name, "_", ci1, "_", ci2, "_res3_", quartiles_suff, ".feather")))
        }

      }
    }

    uncensored_counts_df <- data.frame(counts=uncensored_counts)
    print("uncensored counts")
    print(adjust_name)
    print(quartiles_suff)
    print(uncensored_counts_df)
    arrow::write_feather(uncensored_counts_df, file.path(data_root, paste0("analysis/uncensored_counts_", adjust_name, "_", quartiles_suff, ".feather")))

  }
}


