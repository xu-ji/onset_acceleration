library(tidyverse)
library(plyr)
library(ukbb)
library(arrow)
library(lubridate)

setup_ukbb("mri_v5")
data_root <- "/home/mica/storage/healthspan/v6.2"
meds <- get_field(20003)
take_statin <- meds %>% filter(grepl("statin", value)) %>% select(individual_id) %>% distinct(individual_id)
arrow::write_feather(take_statin, file.path(data_root, "analysis/statin_users.feather"))
