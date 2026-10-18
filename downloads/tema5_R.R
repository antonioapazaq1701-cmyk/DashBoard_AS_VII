# %% 00 — Librerías y configuración
#' Random Forest para regresión de resistencia a compresión del concreto.
#' Adaptación de Random Forest/randomforestenR.r. Se conservan la partición
#' 80/20, el bosque base, las búsquedas OOB y CV de num.trees, la búsqueda de
#' mtry, Grid Search, la importancia por permutación y la comparación con rpart.
directorio_proyecto <- getwd()
directorio_tarea <- getwd()

library(dplyr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(ranger)
library(rpart)
library(rpart.plot)
library(doParallel)
library(recipes)
library(workflows)
library(tibble)

options(scipen = 999)
set.seed(123)
cat("Bibliotecas cargadas | Random Forest para Concrete_Data\n")


# %% 01 — Carga y preparación de datos
#' Concrete_Data.csv contiene ocho predictores de la mezcla y su edad.
#' La variable respuesta es la resistencia a compresión medida en MPa.
ruta_datos <- file.path(directorio_proyecto, "Concrete_Data.csv")
if (!file.exists(ruta_datos)) {
  stop("No se encontró Concrete_Data.csv en la raíz del proyecto.")
}

concreto <- read.csv(ruta_datos, head = TRUE, check.names = FALSE)
columnas_concreto <- c(
  "cemento",
  "escoria",
  "ceniza_volante",
  "agua",
  "superplastificante",
  "agregado_grueso",
  "agregado_fino",
  "edad_dias",
  "resistencia_mpa"
)
if (ncol(concreto) != length(columnas_concreto)) {
  stop(sprintf("Se esperaban 9 columnas; se encontraron %d.", ncol(concreto)))
}
names(concreto) <- columnas_concreto

head(concreto)
str(concreto)
cat("\nFilas:", nrow(concreto), " | Columnas:", ncol(concreto), "\n")
cat("Duplicados exactos:", sum(duplicated(concreto)), "\n")
cat("Valores nulos:", sum(is.na(concreto)), "\n")
summary(concreto$resistencia_mpa)


# %% 02 — Partición de los datos
#' Se mantiene la división aleatoria del ejemplo: 80% entrenamiento y 20% prueba.
set.seed(123)
train <- sample(
  1:nrow(concreto),
  size = floor(nrow(concreto) * 0.8)
)
data_train <- concreto[train, ]
data_test <- concreto[-train, ]

cat("Train:", nrow(data_train), "    Test:", nrow(data_test), "\n")
cat(
  "Media respuesta — Train:", round(mean(data_train$resistencia_mpa), 3),
  " | Test:", round(mean(data_test$resistencia_mpa), 3), "MPa\n"
)
cat(
  "Desviación estándar — Train:", round(sd(data_train$resistencia_mpa), 3),
  " | Test:", round(sd(data_test$resistencia_mpa), 3), "MPa\n"
)


# %% 03 — Modelo base
#' El bosque de referencia conserva los 12 árboles del ejemplo original.
modelo_base <- ranger(
  formula = resistencia_mpa ~ .,
  data = data_train,
  num.trees = 12,
  seed = 123
)
print(modelo_base)
cat("Predictores:", ncol(concreto) - 1, "\n")
summary(modelo_base)

pred_base <- predict(modelo_base, data = data_test)$predictions
rmse_base <- sqrt(mean((pred_base - data_test$resistencia_mpa)^2))
cat("RMSE Modelo Base (12 árboles):", round(rmse_base, 3), "MPa\n\n")


# %% 04 — Optimización de num.trees con OOB
#' El error OOB estima el desempeño con observaciones fuera de la muestra bootstrap.
num_trees_range <- seq(1, 400, 20)
results_oob <- tibble(
  n_arboles = num_trees_range,
  train_rmse = NA_real_,
  oob_rmse = NA_real_
)

for (i in seq_along(num_trees_range)) {
  modelo <- ranger(
    formula = resistencia_mpa ~ .,
    data = data_train,
    num.trees = num_trees_range[i],
    oob.error = TRUE,
    seed = 123
  )
  pred_train <- predict(modelo, data = data_train)$predictions
  results_oob$train_rmse[i] <- sqrt(mean((pred_train - data_train$resistencia_mpa)^2))
  results_oob$oob_rmse[i] <- sqrt(modelo$prediction.error)
}

optimal_trees_oob <- results_oob$n_arboles[which.min(results_oob$oob_rmse)]
p1 <- ggplot(results_oob, aes(x = n_arboles)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE"), linewidth = 1) +
  geom_line(aes(y = oob_rmse, color = "OOB RMSE"), linewidth = 1) +
  geom_vline(xintercept = optimal_trees_oob, linetype = "dashed",
             color = "red", alpha = 0.7) +
  labs(
    title = "Optimización: número de árboles (OOB)",
    x = "Número de árboles",
    y = "RMSE (MPa)",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p1)
cat("Óptimo num.trees (OOB):", optimal_trees_oob, "\n\n")
print(results_oob)


# %% 05 — Optimización de num.trees con validación cruzada
#' Para cada cantidad de árboles se calcula el RMSE de entrenamiento y CV de 5 pliegues.
num_trees_range <- seq(1, 400, 10)
results_cv_trees <- tibble(
  n_arboles = num_trees_range,
  train_rmse = NA_real_,
  cv_rmse = NA_real_
)

for (i in seq_along(num_trees_range)) {
  modelo <- rand_forest(
    mode = "regression",
    trees = num_trees_range[i]
  ) %>%
    set_engine("ranger", seed = 123)

  set.seed(1234)
  cv_folds <- vfold_cv(data_train, v = 5)
  cv_fit <- fit_resamples(
    preprocessor = resistencia_mpa ~ .,
    object = modelo,
    resamples = cv_folds,
    metrics = metric_set(rmse)
  )
  cv_rmse <- collect_metrics(cv_fit)$mean[1]

  modelo_fit <- fit(modelo, resistencia_mpa ~ ., data = data_train)
  pred_train <- predict(modelo_fit, new_data = data_train)$.pred
  train_rmse <- sqrt(mean((pred_train - data_train$resistencia_mpa)^2))
  results_cv_trees$train_rmse[i] <- train_rmse
  results_cv_trees$cv_rmse[i] <- cv_rmse
}

optimal_trees_cv <- results_cv_trees$n_arboles[which.min(results_cv_trees$cv_rmse)]
p2 <- ggplot(results_cv_trees, aes(x = n_arboles)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE"), linewidth = 1) +
  geom_line(aes(y = cv_rmse, color = "CV RMSE"), linewidth = 1) +
  geom_vline(xintercept = optimal_trees_cv, linetype = "dashed",
             color = "red", alpha = 0.7) +
  labs(
    title = "Optimización: número de árboles (5-Fold CV)",
    x = "Número de árboles",
    y = "RMSE (MPa)",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p2)
cat("Óptimo num.trees (CV):", optimal_trees_cv, "\n\n")
print(results_cv_trees)


# %% 06 — Optimización de mtry con OOB
#' mtry indica cuántos predictores se consideran en cada división del bosque.
mtry_range <- 1:(ncol(data_train) - 1)
results_oob_mtry <- tibble(
  mtry = mtry_range,
  train_rmse = NA_real_,
  oob_rmse = NA_real_
)

for (i in seq_along(mtry_range)) {
  modelo <- ranger(
    formula = resistencia_mpa ~ .,
    data = data_train,
    num.trees = 50,
    mtry = mtry_range[i],
    oob.error = TRUE,
    seed = 123
  )
  pred_train <- predict(modelo, data = data_train)$predictions
  results_oob_mtry$train_rmse[i] <- sqrt(mean((pred_train - data_train$resistencia_mpa)^2))
  results_oob_mtry$oob_rmse[i] <- sqrt(modelo$prediction.error)
}

optimal_mtry_oob <- results_oob_mtry$mtry[which.min(results_oob_mtry$oob_rmse)]
p3 <- ggplot(results_oob_mtry, aes(x = mtry)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE"), linewidth = 1) +
  geom_line(aes(y = oob_rmse, color = "OOB RMSE"), linewidth = 1) +
  geom_vline(xintercept = optimal_mtry_oob, linetype = "dashed",
             color = "red", alpha = 0.7) +
  scale_x_continuous(breaks = mtry_range) +
  labs(
    title = "Optimización de mtry (OOB)",
    x = "mtry",
    y = "RMSE (MPa)",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p3)
cat("Óptimo mtry (OOB):", optimal_mtry_oob, "\n\n")
print(results_oob_mtry)


# %% 07 — Grid Search con validación cruzada
#' Se conservan los hiperparámetros y la cuadrícula del ejemplo R de Boston.
modelo_tune <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune()
) %>%
  set_engine(
    engine = "ranger",
    max.depth = tune(),
    seed = 123
  )

transformer <- recipe(resistencia_mpa ~ ., data = data_train)
workflow_modelado <- workflow() %>%
  add_recipe(transformer) %>%
  add_model(modelo_tune)

hiperpar_grid <- expand_grid(
  trees = c(50, 100, 500, 1000),
  mtry = c(3, 5, 7, ncol(data_train) - 1),
  max.depth = c(1, 3, 10, 20)
)
cat("Combinaciones en Grid Search:", nrow(hiperpar_grid), "\n")

set.seed(1234)
cv_folds <- vfold_cv(data_train, v = 5, strata = resistencia_mpa)
numero_nucleos <- parallel::detectCores()
if (is.na(numero_nucleos)) numero_nucleos <- 2L
numero_trabajadores <- min(2L, max(1L, numero_nucleos - 1L))
cl <- makePSOCKcluster(numero_trabajadores)
registerDoParallel(cl)
cat("Trabajadores paralelos:", numero_trabajadores, "\n")
grid_fit <- tune_grid(
  object = workflow_modelado,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  grid = hiperpar_grid,
  control = control_grid(verbose = FALSE)
)
stopCluster(cl)
foreach::registerDoSEQ()

best_params <- show_best(grid_fit, metric = "rmse", n = 5)
print(best_params)


# %% 08 — Modelo final y evaluación en prueba
cat("\n=== Entrenando Modelo Final ===\n")
mejores_hiperpar <- select_best(grid_fit, metric = "rmse")
print(mejores_hiperpar)

modelo_final_fit <- finalize_workflow(
  workflow_modelado,
  parameters = mejores_hiperpar
) %>%
  fit(data = data_train) %>%
  extract_fit_parsnip()

predicciones <- predict(modelo_final_fit, new_data = data_test, type = "numeric")
resultados_test <- predicciones %>%
  bind_cols(data_test %>% select(resistencia_mpa))
rmse_final <- rmse(
  data = resultados_test,
  truth = resistencia_mpa,
  estimate = .pred,
  na_rm = TRUE
)
mae_final <- mae(
  data = resultados_test,
  truth = resistencia_mpa,
  estimate = .pred,
  na_rm = TRUE
)
r2_final <- rsq_trad(
  data = resultados_test,
  truth = resistencia_mpa,
  estimate = .pred,
  na_rm = TRUE
)
cat("RMSE Final en Test:", round(rmse_final$.estimate, 3), "MPa\n")
cat("MAE Final en Test:", round(mae_final$.estimate, 3), "MPa\n")
cat("R² Final en Test:", round(r2_final$.estimate, 4), "\n")
cat(
  "Mejora vs Modelo Base:",
  round((1 - rmse_final$.estimate / rmse_base) * 100, 1), "%\n"
)
print(resultados_test)


# %% 09 — Importancia de predictores por permutación
cat("\n=== Importancia de Predictores ===\n")
modelo_importance <- rand_forest(mode = "regression") %>%
  set_engine(
    engine = "ranger",
    importance = "permutation",
    seed = 123
  ) %>%
  finalize_model(mejores_hiperpar) %>%
  fit(resistencia_mpa ~ ., data = data_train)

importancia_pred <- modelo_importance$fit$variable.importance %>%
  enframe(name = "predictor", value = "importancia") %>%
  arrange(desc(importancia))
print(importancia_pred)

p_importance <- ggplot(
  importancia_pred,
  aes(x = reorder(predictor, importancia), y = importancia, fill = importancia)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Importancia de predictores (permutación)",
    x = "Predictor",
    y = "Importancia"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
print(p_importance)


# %% 10 — Comparación con árbol de decisión
#' Se conserva rpart con método ANOVA y los controles del archivo de referencia.
cat("\n=== Comparación con Árbol de Decisión (rpart) ===\n")
arbol_optimal <- rpart(
  formula = resistencia_mpa ~ .,
  data = data_train,
  method = "anova",
  control = list(minsplit = 6, maxdepth = 7, cp = 0.01)
)
pred_arbol <- predict(arbol_optimal, newdata = data_test)
rmse_arbol <- sqrt(mean((pred_arbol - data_test$resistencia_mpa)^2))

comparacion_modelos <- tibble(
  Modelo = c("Árbol de Decisión", "Random Forest"),
  RMSE_Test = c(rmse_arbol, rmse_final$.estimate),
  Mejora_vs_Base = c(
    (1 - rmse_arbol / rmse_base) * 100,
    (1 - rmse_final$.estimate / rmse_base) * 100
  )
)
print(comparacion_modelos)
rpart.plot(arbol_optimal, main = "Árbol de decisión (rpart) — resistencia del concreto")

mezcla_nueva <- as.data.frame(lapply(data_train %>% select(-resistencia_mpa), median))
cat("\nPredicción del árbol para una mezcla con predictores medianos:\n")
print(mezcla_nueva)
cat("Resistencia estimada (rpart):",
    round(predict(arbol_optimal, newdata = mezcla_nueva), 3), "MPa\n")
cat(
  "La comparación usa el mismo conjunto de prueba; el árbol se incluye como referencia",
  "del ejemplo de curso.\n"
)
