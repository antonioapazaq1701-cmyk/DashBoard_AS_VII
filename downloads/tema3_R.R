# %% Carga, estructura y preparación
# RLMmejorecuacion — Concrete_Data.csv
# Seleccion de predictores y comparacion de ecuaciones para resistencia del concreto.
# Ejecutar desde Tarea3 o desde la raiz del proyecto.

opciones <- c("Concrete_Data.csv", file.path("..", "..", "Concrete_Data.csv"),
              file.path("..", "Concrete_Data.csv"))
archivo <- opciones[file.exists(opciones)][1]
if (is.na(archivo)) stop("No se encontro Concrete_Data.csv desde el directorio actual.")

datos <- read.csv(archivo, check.names = FALSE)
nombres <- c("cemento", "escoria", "ceniza_volante", "agua", "superplastificante",
             "agregado_grueso", "agregado_fino", "edad_dias", "resistencia_mpa")
if (ncol(datos) != length(nombres)) stop("Se esperaban 9 columnas en Concrete_Data.csv.")
names(datos) <- nombres
cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")
cat("Faltantes por variable:\n"); print(colSums(is.na(datos)))
cat("Duplicados exactos:", sum(duplicated(datos)), "\n")
cat("Resumen descriptivo:\n"); print(summary(datos))

# %% Correlaciones de Pearson para exploracion (no implican causalidad).
cor_y <- sort(cor(datos)[, "resistencia_mpa"][-ncol(datos)], decreasing = TRUE)
cat("\nCorrelacion con resistencia (Pearson):\n"); print(round(cor_y, 3))

# %% Separar entrenamiento y prueba antes de seleccionar variables.
set.seed(42)
idx <- sample(seq_len(nrow(datos)), size = floor(.8 * nrow(datos)))
train <- datos[idx, ]; test <- datos[-idx, ]
predictores <- setdiff(names(datos), "resistencia_mpa")
rmse <- function(real, pred) sqrt(mean((real - pred)^2))
mae <- function(real, pred) mean(abs(real - pred))
r2_test <- function(real, pred) 1 - sum((real - pred)^2) / sum((real - mean(real))^2)

# %% Modelo base con todos los predictores.
modelo_completo <- lm(resistencia_mpa ~ ., data = train)
p_completo <- predict(modelo_completo, newdata = test)
cat("\n===== MODELO COMPLETO =====\n"); print(summary(modelo_completo))

# %% Ecuacion mejorada por criterio AIC: pasos hacia atras y hacia adelante.
modelo_atras <- step(modelo_completo, direction = "backward", trace = 0)
modelo_nulo <- lm(resistencia_mpa ~ 1, data = train)
modelo_adelante <- step(modelo_nulo,
                        scope = list(lower = formula(modelo_nulo), upper = formula(modelo_completo)),
                        direction = "forward", trace = 0)
p_atras <- predict(modelo_atras, newdata = test)
p_adelante <- predict(modelo_adelante, newdata = test)
cat("\n===== SELECCION AIC HACIA ATRAS =====\n")
print(formula(modelo_atras)); print(summary(modelo_atras))
cat("\n===== SELECCION AIC HACIA ADELANTE =====\n")
print(formula(modelo_adelante)); print(summary(modelo_adelante))

# %% VIF calculado con regresiones auxiliares, sin paquetes adicionales.
calcular_vif <- function(df) {
  sapply(names(df), function(variable) {
    otros <- setdiff(names(df), variable)
    if (!length(otros)) return(1)
    ajuste <- lm(reformulate(otros, response = variable), data = df)
    1 / (1 - summary(ajuste)$r.squared)
  })
}
vars_vif <- predictores
repeat {
  valores_vif <- calcular_vif(train[, vars_vif, drop = FALSE])
  if (max(valores_vif, na.rm = TRUE) <= 10 || length(vars_vif) <= 2) break
  quitar <- names(which.max(valores_vif))
  cat(sprintf("VIF: se retira %s (%.2f)\n", quitar, valores_vif[quitar]))
  vars_vif <- setdiff(vars_vif, quitar)
}
modelo_vif <- lm(reformulate(vars_vif, response = "resistencia_mpa"), data = train)
p_vif <- predict(modelo_vif, newdata = test)

# %% Eliminacion iterativa por p-valor (umbral 0.05).
vars_p <- predictores
repeat {
  fit_p <- lm(reformulate(vars_p, response = "resistencia_mpa"), data = train)
  pvals <- coef(summary(fit_p))[-1, "Pr(>|t|)"]
  if (!length(pvals) || max(pvals, na.rm = TRUE) <= .05 || length(vars_p) <= 1) break
  quitar <- names(which.max(pvals))
  cat(sprintf("P-valores: se retira %s (p=%.4g)\n", quitar, pvals[quitar]))
  vars_p <- setdiff(vars_p, quitar)
}
modelo_p <- lm(reformulate(vars_p, response = "resistencia_mpa"), data = train)
p_p <- predict(modelo_p, newdata = test)

# %% Estandarizar usando solo entrenamiento. Ridge y Lasso se ajustan en esta escala.
x_train <- as.matrix(train[, predictores])
x_test <- as.matrix(test[, predictores])
centros <- colMeans(x_train); escalas <- apply(x_train, 2, sd); escalas[escalas == 0] <- 1
z_train <- scale(x_train, center = centros, scale = escalas)
z_test <- scale(x_test, center = centros, scale = escalas)
y_train <- train$resistencia_mpa; y_test <- test$resistencia_mpa
y_media <- mean(y_train); y_sd <- sd(y_train)
y_std <- (y_train - y_media) / y_sd

# %% Ridge con validacion cruzada de 5 pliegues para elegir lambda.
set.seed(123)
fold <- sample(rep(1:5, length.out = nrow(z_train)))
lambdas <- 10^seq(-3, 3, length.out = 60)
cv_ridge <- sapply(lambdas, function(lambda) {
  mean(sapply(1:5, function(k) {
    ajustar <- fold != k; validar <- !ajustar
    # Ajustar escalas exclusivamente con el subentrenamiento de este pliegue.
    mu_k <- colMeans(x_train[ajustar, , drop = FALSE])
    sd_k <- apply(x_train[ajustar, , drop = FALSE], 2, sd)
    sd_k[sd_k == 0] <- 1
    z_fit <- scale(x_train[ajustar, , drop = FALSE], mu_k, sd_k)
    z_val <- scale(x_train[validar, , drop = FALSE], mu_k, sd_k)
    ym_k <- mean(y_train[ajustar]); ys_k <- sd(y_train[ajustar])
    yz_k <- (y_train[ajustar] - ym_k) / ys_k
    b <- solve(crossprod(z_fit) + lambda * diag(ncol(z_fit)), crossprod(z_fit, yz_k))
    pred <- ym_k + ys_k * as.vector(z_val %*% b)
    rmse(y_train[validar], pred)
  }))
})
lambda_ridge <- lambdas[which.min(cv_ridge)]
b_ridge <- solve(crossprod(z_train) + lambda_ridge * diag(ncol(z_train)), crossprod(z_train, y_std))
p_ridge <- y_media + y_sd * as.vector(z_test %*% b_ridge)
cat("\nLambda Ridge elegida:", lambda_ridge, "\n")

# %% Lasso por descenso de coordenadas con respuesta y predictores estandarizados.
soft <- function(x, lambda) sign(x) * pmax(abs(x) - lambda, 0)
ajustar_lasso <- function(X, y, lambda, maxit = 5000, tol = 1e-8) {
  beta <- rep(0, ncol(X)); n <- nrow(X)
  for (iter in seq_len(maxit)) {
    anterior <- beta
    for (j in seq_len(ncol(X))) {
      residuo <- y - as.vector(X %*% beta) + X[, j] * beta[j]
      rho <- sum(X[, j] * residuo) / n
      beta[j] <- soft(rho, lambda) / (sum(X[, j]^2) / n)
    }
    if (max(abs(beta - anterior)) < tol) break
  }
  beta
}
lambda_lasso <- .03
b_lasso <- ajustar_lasso(z_train, y_std, lambda_lasso)
p_lasso <- y_media + y_sd * as.vector(z_test %*% b_lasso)

# %% Comparacion de errores en observaciones que no se usaron para entrenar.
comparacion <- data.frame(
  modelo = c("Completo", "AIC atras", "AIC adelante", "VIF", "P-valores", "Ridge", "Lasso"),
  RMSE_test = c(rmse(y_test,p_completo), rmse(y_test,p_atras), rmse(y_test,p_adelante),
                rmse(y_test,p_vif), rmse(y_test,p_p), rmse(y_test,p_ridge), rmse(y_test,p_lasso)),
  MAE_test = c(mae(y_test,p_completo), mae(y_test,p_atras), mae(y_test,p_adelante),
               mae(y_test,p_vif), mae(y_test,p_p), mae(y_test,p_ridge), mae(y_test,p_lasso)),
  R2_test = c(r2_test(y_test,p_completo), r2_test(y_test,p_atras), r2_test(y_test,p_adelante),
               r2_test(y_test,p_vif), r2_test(y_test,p_p), r2_test(y_test,p_ridge), r2_test(y_test,p_lasso))
)
cat("\n===== COMPARACION DE ECUACIONES =====\n")
print(comparacion[order(comparacion$RMSE_test), ], row.names = FALSE)

# %% Diagnosticos de residuos de la ecuacion AIC hacia atras.
par(mfrow = c(2, 2)); plot(modelo_atras); par(mfrow = c(1, 1))

# %% Random Forest opcional; requiere el paquete randomForest.
if (requireNamespace("randomForest", quietly = TRUE)) {
  set.seed(42)
  modelo_rf <- randomForest::randomForest(resistencia_mpa ~ ., data = train,
                                         ntree = 500, importance = TRUE)
  p_rf <- predict(modelo_rf, newdata = test)
  cat(sprintf("\\nRandom Forest: RMSE test=%.3f MPa | R2 test=%.4f\\n",
              rmse(y_test,p_rf), r2_test(y_test,p_rf)))
  print(randomForest::importance(modelo_rf))
} else {
  cat("\\nRandom Forest omitido; opcionalmente instale el paquete randomForest.\\n")
}
