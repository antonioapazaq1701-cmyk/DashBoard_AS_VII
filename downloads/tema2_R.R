# %% Carga, estructura y preparación
# TAREA 2 — REGRESION LINEAL SIMPLE Y MULTIPLE
# Ejecutar desde Regresion lineal simple y multiple/Tarea2
opciones <- c("Concrete_Data.csv", file.path("..", "..", "Concrete_Data.csv"),
              file.path("..", "Concrete_Data.csv"))
archivo <- opciones[file.exists(opciones)][1]
if (is.na(archivo)) stop("No se encontro Concrete_Data.csv desde el directorio actual.")
datos <- read.csv(archivo, check.names = FALSE)
names(datos) <- c("cemento", "escoria", "ceniza_volante", "agua", "superplastificante",
                  "agregado_grueso", "agregado_fino", "edad_dias", "resistencia_mpa")
set.seed(42)
indices <- sample(seq_len(nrow(datos)), size = floor(.8 * nrow(datos)))
train <- datos[indices, ]
test <- datos[-indices, ]
rmse <- function(real, pred) sqrt(mean((real - pred)^2))
mae <- function(real, pred) mean(abs(real - pred))

# %% Regresion simple: resistencia explicada por cemento
modelo_simple <- lm(resistencia_mpa ~ cemento, data = train)
pred_simple <- predict(modelo_simple, newdata = test)
cat("===== REGRESION LINEAL SIMPLE =====\n")
print(summary(modelo_simple))
cat(sprintf("Test: RMSE = %.3f MPa | MAE = %.3f MPa | R2 = %.3f\n",
            rmse(test$resistencia_mpa, pred_simple),
            mae(test$resistencia_mpa, pred_simple),
            1 - sum((test$resistencia_mpa - pred_simple)^2) /
              sum((test$resistencia_mpa - mean(test$resistencia_mpa))^2)))

# %% Regresion multiple: todas las variables de mezcla y edad
modelo_multiple <- lm(resistencia_mpa ~ ., data = train)
pred_multiple <- predict(modelo_multiple, newdata = test)
cat("\n===== REGRESION LINEAL MULTIPLE =====\n")
print(summary(modelo_multiple))
cat(sprintf("Test: RMSE = %.3f MPa | MAE = %.3f MPa | R2 = %.3f\n",
            rmse(test$resistencia_mpa, pred_multiple),
            mae(test$resistencia_mpa, pred_multiple),
            1 - sum((test$resistencia_mpa - pred_multiple)^2) /
              sum((test$resistencia_mpa - mean(test$resistencia_mpa))^2)))
comparacion <- data.frame(
  modelo = c("Simple: cemento", "Multiple: todos los predictores"),
  RMSE_test = c(rmse(test$resistencia_mpa, pred_simple), rmse(test$resistencia_mpa, pred_multiple)),
  MAE_test = c(mae(test$resistencia_mpa, pred_simple), mae(test$resistencia_mpa, pred_multiple)),
  R2_test = c(1 - sum((test$resistencia_mpa - pred_simple)^2) / sum((test$resistencia_mpa - mean(test$resistencia_mpa))^2),
              1 - sum((test$resistencia_mpa - pred_multiple)^2) / sum((test$resistencia_mpa - mean(test$resistencia_mpa))^2))
)
print(comparacion, row.names = FALSE)

# %% Diagnosticos clasicos sobre el modelo multiple
par(mfrow = c(2, 2))
plot(modelo_multiple)
par(mfrow = c(1, 1))

# %% Prediccion para una mezcla de ejemplo
mezcla <- data.frame(cemento = 350, escoria = 100, ceniza_volante = 0,
                     agua = 180, superplastificante = 8, agregado_grueso = 1000,
                     agregado_fino = 750, edad_dias = 28)
cat("Prediccion para mezcla de ejemplo (MPa):\n")
print(predict(modelo_multiple, newdata = mezcla, interval = "prediction"))
