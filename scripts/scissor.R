library(Scissor)
library(tidyverse)
library(Seurat)
options(Seurat.object.assay.version = "v3")

seurat_object <- readRDS("/home/DingchengYi/BulkPheno/data/MERFISH/HC1.rds")

# 随机抽样十分之一的细胞
num_cells <- ncol(seurat_object)
sample_size <- ceiling(num_cells / 20)
sc_dataset <- subset(seurat_object, cells = sample(colnames(seurat_object), size = sample_size))

VariableFeatures(sc_dataset) <- rownames(sc_dataset@assays$RNA@counts)
sc_dataset <- FindNeighbors(sc_dataset, dims = 1:10)
sc_dataset <- FindClusters(sc_dataset, resolution = 0.6)

bulk_dataset <- read_csv('data/TCGA/Processed/exp_data.csv')
clinical <- read_csv('data/TCGA/Processed/clinical_data.csv')
survival <- clinical %>% select(...1, OS.time, OS)
survival
all(bulk_dataset$...1 == survival$...1)
survival <- survival %>% select(-...1)
colnames(survival) <- c("time", "status")

rm(list = setdiff(ls(), c("sc_dataset", "bulk_dataset", "survival")))
gc()
bulk_dataset <- bulk_dataset %>% select(-...1) %>% t()
phenotype <- survival
infos <- Scissor(as.matrix(bulk_dataset), sc_dataset, phenotype, alpha = 0.05, cutoff = 0.05,
                     family = "cox", Save_file = 'Scissor_HC1_survival_tmp.RData')

Scissor_select <- rep(0, ncol(sc_dataset))
names(Scissor_select) <- colnames(sc_dataset)
Scissor_select[infos$Scissor_pos] <- "Worse"
Scissor_select[infos$Scissor_neg] <- "Better"
sc_dataset <- AddMetaData(sc_dataset, metadata = Scissor_select, col.name = "scissor")
saveRDS(sc_dataset, "Scissor_HC1.rds")

numbers <- length(infos$Scissor_pos) + length(infos$Scissor_neg)
result <- reliability.test(X, Y, network, alpha = 0.05, family = "cox", cell_num = numbers, n = 10, nfold = 10)
save(result,file="Scissor_HC1_reliability.RData")
