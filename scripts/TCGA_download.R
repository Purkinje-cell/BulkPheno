library(tidyverse)
library(TCGAbiolinks)
query <- GDCquery(
    project = c("TCGA-LUAD", "TCGA-BRCA"),
    data.category = "Transcriptome Profiling",
)
datatable(
    getResults(query), 
    filter = 'top',
    options = list(scrollX = TRUE, keys = TRUE, pageLength = 5), 
    rownames = FALSE
)

query <- GDCquery(
    project = "TCGA-LUAD",
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification", 
    workflow.type = "STAR - Counts",
)
GDCdownload(query = query)
data <- GDCprepare(query = query)