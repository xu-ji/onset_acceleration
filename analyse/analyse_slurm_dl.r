library(tidyverse)
library(plyr)
library(ukbb)
library(arrow)
library(lubridate)

# get all ids and demog available either ukbb version
data_root <- "/mnt/mica01/healthspan/v4.2"

vname <- "mri_v5" #
print(vname)

setup_ukbb(vname)

sex <- get_field(31) %>% select(individual_id, sex=value) %>% distinct(individual_id, .keep_all=TRUE)
arrow::write_feather(sex, file.path(data_root, paste0("data/analyse_slurm_", vname, ".feather")))

