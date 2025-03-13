library(splatter)
library(scater)
library(tidyverse)
library(reticulate)
use_condaenv("pyg")
library(sceasy)
params.groups <- newSplatParams(batchCells = 2000, nGenes = 500)

# One small group, one big group
sim1 <- splatSimulateGroups(
    params.groups,
    group.prob = c(0.8, 0.1, 0.1),
    de.prob = c(0.1, 0.03, 0.03),
    verbose = TRUE
)
sim1 <- logNormCounts(sim1)
sim1 <- runPCA(sim1)
sim1 <- runUMAP(sim1)
plotPCA(sim1, colour_by = "Group") + ggtitle("two small group, one big group")

plotUMAP(sim1, colour_by = "Group") + ggtitle("two small group, one big group")
ggsave("BulkPheno/data/sim1.png")
sceasy::convertFormat(sim1, from="sce", to="anndata",outFile='data/Simulation/sim1.h5ad', main_layer = "counts")
SummarizedExperiment::assay(sim1, "counts")
