paquetes <- c("tree", "rpart", "rpart.plot", "dplyr", "tidyr", "ggplot2", "tidymodels", "ranger", "doParallel", "recipes", "workflows", "tibble", "jsonlite")
faltan <- setdiff(paquetes, rownames(installed.packages()))
if (length(faltan)) install.packages(faltan, repos = "https://cloud.r-project.org")
