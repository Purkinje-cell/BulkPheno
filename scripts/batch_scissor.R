library(Scissor)
library(tidyverse)
library(Seurat)
library(sceasy)
library(reticulate)
options(Seurat.object.assay.version = "v3")

# Setup Python environment
use_condaenv("R")

# Load data
seurat_object <- readRDS("data/MERFISH/HC1.rds")
bulk_dataset <- read_csv('data/TCGA/Processed/exp_data_tumor.csv')
clinical <- read_csv('data/TCGA/Processed/clinical_data_tumor.csv')

# Prepare survival data
survival <- clinical %>% select(...1, OS.time, OS)
# Verify alignment
stopifnot(all(bulk_dataset$...1 == survival$...1))
survival <- survival %>% select(-...1)
colnames(survival) <- c("time", "status")

# Prepare bulk data
bulk_dataset <- bulk_dataset %>% select(-...1) %>% t()
phenotype <- survival

# Batch processing
batch_size <- 20000
all_cells <- colnames(seurat_object)
num_batches <- ceiling(length(all_cells) / batch_size)

for (batch_idx in 1:num_batches) {
  cat("Processing batch", batch_idx, "of", num_batches, "\n")
  
  # Get cells for this batch
  start_idx <- (batch_idx - 1) * batch_size + 1
  end_idx <- min(batch_idx * batch_size, length(all_cells))
  batch_cells <- all_cells[start_idx:end_idx]
  
  # Subset Seurat object
  sc_batch <- subset(seurat_object, cells = batch_cells)
  
  # Process batch
  VariableFeatures(sc_batch) <- rownames(sc_batch@assays$RNA@counts)
  sc_batch <- FindNeighbors(sc_batch, dims = 1:10)
  sc_batch <- FindClusters(sc_batch, resolution = 0.6)
  
  # Run Scissor
  infos <- Scissor(as.matrix(bulk_dataset), sc_batch, phenotype, alpha = 0.05, cutoff = 0.05,
                   family = "cox", Save_file = paste0('Scissor_tumor_HC1_survival_batch_', batch_idx, '.RData'))
  
  # Add Scissor metadata
  Scissor_select <- rep(0, ncol(sc_batch))
  names(Scissor_select) <- colnames(sc_batch)
  Scissor_select[infos$Scissor_pos] <- "Worse"
  Scissor_select[infos$Scissor_neg] <- "Better"
  sc_batch <- AddMetaData(sc_batch, metadata = Scissor_select, col.name = "scissor")
  
  # Save as anndata
  output_file <- paste0('HC1_scissor_tumor_batch_', batch_idx, '.h5ad')
  sceasy::convertFormat(sc_batch, from="seurat", to="anndata", outFile=output_file)
  cat("Saved batch", batch_idx, "to", output_file, "\n")
  
  # Clean up to free memory
  rm(sc_batch, infos, Scissor_select)
  gc()
}

cat("All batches processed successfully.\n")
