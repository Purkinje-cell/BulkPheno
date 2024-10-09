library(tidyverse)
library(preprocessCore)
library(BiocManager)
library(clusterProfiler)
library(org.Hs.eg.db)
library(TCGAbiolinks)
query.exp.hg38 <- GDCquery(
    project = "TCGA-LUAD", 
    data.category = "Transcriptome Profiling", 
    data.type = "Gene Expression Quantification", 
    workflow.type = "STAR - Counts",
)
GDCdownload(query.exp.hg38)
expdat <- GDCprepare(
    query = query.exp.hg38,
)

data <- read_table("BulkPheno/TCGA/TCGA-LIHC.htseq_counts.tsv")

exp2 <- function(x){
    x <- as.numeric(x)
    x <- 2^x - 1
    x <- as.integer(x)
    return(x)
}
data <- data %>% mutate(across(starts_with("TCGA"), exp2))

data <- data %>% mutate(Ensembl_ID = str_extract(Ensembl_ID, "ENSG[0-9]+"))
data <- data %>% filter(!is.na(Ensembl_ID))
gene_symbol <- bitr(data$Ensembl_ID, fromType = "ENSEMBL", toType = "SYMBOL", OrgDb = org.Hs.eg.db)
data <- data %>% left_join(gene_symbol, by = c("Ensembl_ID" = "ENSEMBL"))

data <- data %>% dplyr::select(-Ensembl_ID) %>% group_by(SYMBOL) %>% summarise(across(everything(), sum))
data <- data %>% filter(!is.na(SYMBOL))
write_tsv(data, "BulkPheno/TCGA/TCGA-LIHC.htseq_counts_clean.tsv")

clinical_data <- read_delim("BulkPheno/TCGA/TCGA-LIHC.GDC_phenotype.tsv", delim="\t")
colnames(clinical_data)

exp_data <- read.table("BulkPheno/TCGA/TCGA-LIHC.htseq_counts_clean.tsv", header=TRUE, row.names=1)
pseudo_bulk_data <- read.csv("BulkPheno/Liver/pseudo_bulk.csv", header=TRUE, row.names=1)

common <- intersect(rownames(exp_data), rownames(pseudo_bulk_data))

survival_data <- read.table("BulkPheno/TCGA/TCGA-LIHC.survival.tsv", header=TRUE, row.names=1)
rownames(survival_data) <- gsub("-", ".", rownames(survival_data))
common_sample <- intersect(rownames(survival_data), colnames(exp_data))
survival_data <- survival_data[common_sample,]
exp_data <- exp_data[,common_sample]

exp_data <- exp_data[common,]
pseudo_bulk_data <- pseudo_bulk_data[common,]

combined_data <- cbind(exp_data, pseudo_bulk_data)
normalized_combined_data <- normalize.quantiles(as.matrix(combined_data), keep.names=TRUE)
normalized_combined_data <- as.data.frame(t(as.data.frame(normalized_combined_data)))
normalized_combined_data <- normalized_combined_data %>% mutate(across(where(is.numeric), floor))

normalized_bulk_expression <- normalized_combined_data[1:ncol(exp_data),]
normalized_pseudo_bulk_expression <- normalized_combined_data[(ncol(exp_data)+1):ncol(normalized_combined_data),]
write.csv(normalized_bulk_expression, "BulkPheno/Cleaned_data/Liver/TCGA-LIHC.htseq_counts_clean_normalized.csv")
write.csv(normalized_pseudo_bulk_expression, "BulkPheno/Cleaned_data/Liver/pseudo_bulk_normalized.csv")

### Survival


head(survival_data)
library(survival)
survfit <- survfit(Surv(OS.time, OS) ~ 1, data = survival_data)

# 计算中位生存期
median_survival <- summary(survfit)$table['median']
pattern <- "(?<=\\.)[A-Za-z0-9]+$"

survival_data$barcode <- rownames(survival_data)
survival_data <- survival_data %>% filter(!is.na(OS.time)) %>% mutate(median_surv = ifelse(OS.time <= median_survival, "short", "long"), sample_type_code = str_extract(str_extract(barcode, pattern), "[0-9]+"), sample_type = case_when(sample_type_code == '01' ~ 'Tumor', sample_type_code == '02' ~ 'Tumor', sample_type_code == '11'~'Normal')) %>% dplyr::select(-sample_type_code)

write_csv(survival_data, "BulkPheno/Cleaned_data/Liver/TCGA-LIHC.survival_clean.csv")

tumor_barcode <- survival_data %>% filter(sample_type == "Tumor") %>% pull(barcode)
normal_barcode <- survival_data %>% filter(sample_type == "Normal") %>% pull(barcode)

normalized_bulk_tumor <- normalized_bulk_expression[tumor_barcode, ]
normalized_bulk_normal <- normalized_bulk_expression[normal_barcode, ]
write.csv(normalized_bulk_tumor, "BulkPheno/Cleaned_data/Liver/TCGA-LIHC.htseq_counts_clean_normalized_tumor.csv")
write.csv(normalized_bulk_normal, "BulkPheno/Cleaned_data/Liver/TCGA-LIHC.htseq_counts_clean_normalized_normal.csv")

survival_data_tumor <- survival_data %>% filter(sample_type == "Tumor")
survival_data_normal <- survival_data %>% filter(sample_type == "Normal")
write.csv(survival_data_tumor, "BulkPheno/Cleaned_data/Liver/TCGA-LIHC.survival_clean_tumor.csv")
write.csv(survival_data_normal, "BulkPheno/Cleaned_data/Liver/TCGA-LIHC.survival_clean_normal.csv")
