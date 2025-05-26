library(haven)

ddpath  <- "/Users/liyuanxu/PycharmProjects/KGF/code/data/job_corps_raw/"
setwd(ddpath) 
job_corps_dataset = read_sas("fu12_raw.sas7bdat")

tmp = read_sas("fu30_raw.sas7bdat") #add this for id selection
job_corps_dataset = merge(job_corps_dataset, tmp, by="MPRID")

tmp = read_sas("impact.sas7bdat")
job_corps_dataset = merge(job_corps_dataset, tmp, by="MPRID")

tmp = read_sas("key_vars.sas7bdat")
job_corps_dataset = merge(job_corps_dataset, tmp, by="MPRID")

tmp = read_sas("base_raw.sas7bdat")
job_corps_dataset = merge(job_corps_dataset, tmp, by="MPRID")

tmp = read_sas("rand_dat.sas7bdat")
job_corps_dataset = merge(job_corps_dataset, tmp, by="MPRID")


names(job_corps_dataset) <- tolower(names(job_corps_dataset))
job_corps_dataset = job_corps_dataset[order(job_corps_dataset$mprid),]
save(job_corps_dataset, file = "my_job_corps_dataset.rdata")

