# Tarea 4 - Arboles de regresion con Concrete_Data.csv
# Adaptacion directa de Arboles/arbolesderegresion.r.
# Los comentarios # %% delimitan las celdas del cuaderno HTML.

# %% 1. Librerias y lectura de la base
#' Se mantiene la secuencia del ejemplo R: lectura, descripcion, tree, validacion y poda, y finalmente rpart con metodo ANOVA. La respuesta se adapta a la resistencia del concreto en MPa. Se cargan solo las librerias que se usan; las funciones descriptivas de R base sustituyen a las librerias auxiliares del ejemplo.
directorio_tarea <- getwd()
library(tree)
library(rpart)
library(rpart.plot)

archivo_base <- file.path(directorio_tarea, "Concrete_Data.csv")
datos <- read.csv(archivo_base, header = TRUE, sep = ",", check.names = FALSE)
stopifnot(ncol(datos) == 9)
diccionario <- data.frame(
  Nombre_original = names(datos),
  Nombre_usado = c("Cemento", "Escoria", "Ceniza", "Agua", "Superplastificante",
                   "Agregado_grueso", "Agregado_fino", "Edad", "Resistencia"),
  Unidad = c(rep("kg/m3", 7), "dias", "MPa")
)
names(datos) <- diccionario$Nombre_usado
print(diccionario, row.names = FALSE)
head(datos)
str(datos)

# %% 2. Estadistica descriptiva
#' La base del concreto tiene variables numericas, por lo que no corresponde convertir una columna a factor como chas en Boston. Se conserva summary del ejemplo y se revisan nulos y duplicados sin modificar las observaciones originales.
summary(datos)
colSums(is.na(datos))
cat("Filas exactamente duplicadas:", sum(duplicated(datos)), "\n")
stopifnot(all(vapply(datos, is.numeric, logical(1))))
stopifnot(all(is.finite(as.matrix(datos))))

# %% 3. Datos de entrenamiento (80%) y prueba (20%)
#' Se mantiene set.seed(123) y el muestreo por filas del script base. Esta particion aleatoria puede repartir mediciones repetidas entre los conjuntos, por lo que los resultados describen este protocolo de clase y no una validacion independiente por formulacion.
set.seed(123)
train <- sample(1:nrow(datos), size = nrow(datos) * 0.8)
data_train <- datos[train, ]
data_test <- datos[-train, ]
dim(data_train)
dim(data_test)

# %% 4. Arbol tree con valores por defecto
#' Se usa tree con split="deviance", como en el ejemplo. Cada division reduce la suma de cuadrados y cada hoja predice la media de resistencia de sus observaciones.
set.seed(123)
arbol_regresion1 <- tree(
  formula = Resistencia ~ .,
  data = data_train,
  split = "deviance"
)
summary(arbol_regresion1)
arbol_regresion1

# %% 5. Estructura del arbol inicial
#| fig-width: 18
#| fig-height: 9
#' Se muestra el arbol real completo. Las figuras amplias permiten desplazamiento horizontal para leer las etiquetas sin reemplazar el modelo por otro arbol ilustrativo.
par(mfrow = c(1, 1))
plot(arbol_regresion1, type = "proportional")
text(arbol_regresion1, splits = TRUE, pretty = 0, cex = 0.80, col = "firebrick")
title(main = "Arbol tree inicial - Resistencia del concreto (MPa)")

# %% 6. Correlacion de ajuste y error en prueba
y_hat <- predict(arbol_regresion1, newdata = data_train)
cor(y_hat, data_train$Resistencia)
predicciones <- predict(arbol_regresion1, newdata = data_test)
test_rmse <- sqrt(mean((predicciones - data_test$Resistencia)^2))
cat("RMSE de prueba del arbol inicial:", round(test_rmse, 3), "MPa\n")

# %% 7. Prediccion de una mezcla nueva
#' Se conserva el bloque de nuevos_datos del ejemplo, adaptado a los componentes del concreto y su edad. Las cantidades son kg/m3 y la edad se expresa en dias; la resistencia predicha se expresa en MPa.
nuevos_datos <- data.frame(
  Cemento = 300,
  Escoria = 100,
  Ceniza = 0,
  Agua = 180,
  Superplastificante = 6,
  Agregado_grueso = 1000,
  Agregado_fino = 800,
  Edad = 28
)
nuevos_datos
predict(arbol_regresion1, newdata = nuevos_datos)

# %% 8. Control de criterios de tree
#' Se conservan mincut=5, minsize=10 y mindev=0.01 del ejemplo. Son restricciones del arbol tree, no los parametros de rpart.
set.seed(123)
arbol_regresion <- tree::tree(
  formula = Resistencia ~ .,
  data = data_train,
  split = "deviance",
  mincut = 5,
  minsize = 10,
  mindev = 0.01
)
summary(arbol_regresion)
arbol_regresion

# %% 9. Arbol controlado: ramas proporcionales a devianza
#| fig-width: 18
#| fig-height: 9
plot(arbol_regresion, type = "proportional")
text(arbol_regresion, splits = TRUE, all = TRUE, pretty = 0,
     cex = 0.75, col = "firebrick")
title(main = "Arbol tree controlado - Ramas proporcionales")

# %% 10. Arbol controlado: ramas uniformes
#| fig-width: 18
#| fig-height: 9
plot(arbol_regresion, type = "uniform")
text(arbol_regresion, splits = TRUE, all = FALSE, pretty = 0,
     cex = 0.80, col = "firebrick")
title(main = "Arbol tree controlado - Ramas uniformes")

# %% 11. Validacion cruzada de tree: 10 pliegues
#' Se conserva cv.tree con FUN=prune.tree y K=10. La seleccion del tamano de poda usa entrenamiento, como en el ejemplo.
set.seed(123)
cv_arbol <- cv.tree(
  object = arbol_regresion,
  FUN = prune.tree,
  K = 10
)
cv_arbol

# %% 12. Curva de error y tamano optimo
#| fig-width: 9
#| fig-height: 5
plot(cv_arbol$size, cv_arbol$dev, type = "b", pch = 19, col = "steelblue",
     xlab = "Numero de nodos terminales", ylab = "Devianza (error CV)",
     main = "Seleccion del tamano optimo del arbol tree")
size_optimo <- rev(cv_arbol$size)[which.min(rev(cv_arbol$dev))]
cat("Numero optimo de nodos terminales:", size_optimo, "\n")

# %% 13. Poda de tree con el tamano seleccionado
#| fig-width: 18
#| fig-height: 9
#' Correccion puntual: el ejemplo calcula size_optimo pero luego fija best=8. Aqui se utiliza best=size_optimo para aplicar la seleccion obtenida con la misma validacion del ejemplo.
arbol_podado <- prune.tree(tree = arbol_regresion, best = size_optimo)
summary(arbol_podado)
arbol_podado
if (nrow(arbol_podado$frame) > 1) {
  plot(arbol_podado, type = "uniform")
  text(arbol_podado, splits = TRUE, all = TRUE, pretty = 0,
       cex = 0.80, col = "firebrick")
  title(main = "Arbol tree podado - Resistencia del concreto")
} else {
  plot.new()
  text(0.5, 0.5, paste("Arbol de una hoja. Prediccion:",
                       round(arbol_podado$frame$yval[1], 3), "MPa"))
}

# %% 14. Comparacion del error: tree original vs. podado
rmse <- function(real, pred) sqrt(mean((real - pred)^2))
pred_original <- predict(arbol_regresion, newdata = data_test)
pred_podado <- predict(arbol_podado, newdata = data_test)
comparacion_tree <- data.frame(
  Modelo = c("tree controlado", "tree podado"),
  RMSE_MPa = c(rmse(data_test$Resistencia, pred_original),
               rmse(data_test$Resistencia, pred_podado))
)
comparacion_tree

# %% 15. Regresion con rpart: metodo ANOVA
#' Se mantiene el segundo metodo del ejemplo: rpart con method="anova", entrenado usando subset=train. El arbol inicial conserva los controles por defecto de rpart.
set.seed(123)
m1 <- rpart(Resistencia ~ ., data = datos, subset = train, method = "anova")
m1

# %% 16. Visualizacion del primer arbol rpart
#| fig-width: 18
#| fig-height: 10
rpart.plot(m1, main = "rpart inicial - Metodo ANOVA", cex = 0.85)

# %% 17. Curva de complejidad de rpart inicial
#| fig-width: 9
#| fig-height: 5
plotcp(m1)
m1$cptable

# %% 18. Extension de rpart y validacion cruzada
#' Se conserva cp=0 y xval=10 del ejemplo. Se imprime la tabla de complejidad para estudiar la poda y se marca el cp de menor xerror. La linea arbitraria abline(v=12) del ejemplo se sustituye por este valor calculado.
set.seed(123)
m2 <- rpart(
  formula = Resistencia ~ .,
  data = datos,
  subset = train,
  method = "anova",
  control = list(cp = 0, xval = 10)
)
m2$cptable
cp_optimo <- m2$cptable[which.min(m2$cptable[, "xerror"]), "CP"]
cat("cp con menor xerror en m2:", cp_optimo, "\n")

# %% 19. Curva de complejidad del arbol extendido
#| fig-width: 9
#| fig-height: 5
plotcp(m2)
if (cp_optimo > 0) abline(v = cp_optimo, lty = "dashed", col = "firebrick")

# %% 20. Modelo rpart con los controles del ejemplo
#' Se conserva el bloque llamado optimal_tree y sus controles: minsplit=6, maxdepth=7, cp=0.01. El nombre procede del ejemplo; estos valores no son una garantia de optimum en esta base. La evaluacion se corrige para usar data_test: pasar subset=train a predict no restringe de forma fiable las predicciones del ejemplo original.
set.seed(123)
optimal_tree <- rpart(
  formula = Resistencia ~ .,
  data = datos,
  subset = train,
  method = "anova",
  control = list(minsplit = 6, maxdepth = 7, cp = 0.01)
)
pred <- predict(optimal_tree, newdata = data_test)
rmse_test_rpart <- rmse(data_test$Resistencia, pred)
cat("RMSE de prueba del rpart con controles:", round(rmse_test_rpart, 3), "MPa\n")
optimal_tree

# %% 21. Estructura del rpart con controles
#| fig-width: 18
#| fig-height: 10
rpart.plot(optimal_tree, main = "rpart - minsplit=6, maxdepth=7, cp=0.01", cex = 0.85)

# %% 22. Prediccion de la mezcla nueva con los modelos finales
prediccion_nueva <- data.frame(
  Modelo = c("tree inicial", "tree podado", "rpart con controles"),
  Resistencia_predicha_MPa = c(
    predict(arbol_regresion1, nuevos_datos),
    predict(arbol_podado, nuevos_datos),
    predict(optimal_tree, nuevos_datos)
  )
)
prediccion_nueva

# %% 23. Importancia y diagrama circular complementario
#| fig-width: 11
#| fig-height: 6
#' Se conserva el circular solicitado como complemento posterior al procedimiento del ejemplo R. Se normaliza la importancia de optimal_tree para sumar 100%; los porcentajes no representan efectos causales ni partes de la resistencia.
importancia <- sort(optimal_tree$variable.importance, decreasing = TRUE)
importancia <- importancia[importancia > 0]
importancia_relativa <- importancia / sum(importancia)
data.frame(Variable = names(importancia_relativa),
           Porcentaje = round(100 * importancia_relativa, 2), row.names = NULL)
colores <- c("#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
             "#fdb462", "#b3de69", "#fccde5")
layout(matrix(c(1, 2), nrow = 1), widths = c(2, 1))
par(mar = c(1, 1, 4, 1))
pie(importancia_relativa, labels = paste0(round(100 * importancia_relativa, 1), "%"),
    col = colores[seq_along(importancia_relativa)], border = "white",
    clockwise = TRUE, init.angle = 90, main = "Importancia relativa de variables - R")
par(mar = c(1, 0, 4, 0))
plot.new()
legend("center", names(importancia_relativa), fill = colores[seq_along(importancia_relativa)],
       border = NA, bty = "n", cex = 0.9)
layout(1)

# %% 24. Resultados, conclusiones y archivos complementarios
#' La tabla resume los errores sobre la misma prueba para todos los modelos del procedimiento. R2, MAE y RMSE son medidas descriptivas de esta particion. Se mantiene el protocolo del ejemplo R, por lo que sus numeros no son directamente comparables con los del cuaderno Python, que usa otra semilla y otra particion.
modelos <- list(
  "tree inicial" = arbol_regresion1,
  "tree controlado" = arbol_regresion,
  "tree podado" = arbol_podado,
  "rpart inicial" = m1,
  "rpart extendido" = m2,
  "rpart con controles" = optimal_tree
)
metricas <- function(modelo) {
  y_real <- data_test$Resistencia
  y_pred <- as.numeric(predict(modelo, newdata = data_test))
  c(MAE = mean(abs(y_real - y_pred)), MSE = mean((y_real - y_pred)^2),
    RMSE = rmse(y_real, y_pred),
    R2 = 1 - sum((y_real - y_pred)^2) / sum((y_real - mean(y_real))^2))
}
resultados <- as.data.frame(t(vapply(modelos, metricas, numeric(4))))
round(resultados, 4)
cat("Tamano tree seleccionado por CV:", size_optimo, "hojas.\n")
cat("RMSE tree original/podado:", round(comparacion_tree$RMSE_MPa, 3), "MPa.\n")
if (nrow(arbol_podado$frame) == nrow(arbol_regresion$frame)) {
  cat("La validacion eligio conservar todas las hojas: la poda no redujo el arbol ni cambio las predicciones.\n")
}
cat("RMSE del rpart con controles:", round(rmse_test_rpart, 3), "MPa.\n")
if (max(abs(predict(optimal_tree, data_test) - pred_original)) < 1e-10) {
  cat("En esta base, tree controlado y rpart con controles obtuvieron las mismas predicciones de prueba.\n")
}
cat("Predictores de mayor importancia en rpart:",
    paste(head(names(importancia), 3), collapse = ", "), ".\n")
cat("Estos resultados no garantizan desempeno en nuevas formulaciones.\n")
write.csv(resultados, file.path(directorio_tarea, "metricas_arboles_concrete_R.csv"),
          fileEncoding = "UTF-8")
predicciones_test <- data.frame(
  Fila_original = as.integer(rownames(data_test)),
  Real_MPa = data_test$Resistencia,
  Predicho_tree_podado_MPa = as.numeric(pred_podado),
  Predicho_rpart_controles_MPa = as.numeric(pred),
  Residuo_rpart_MPa = data_test$Resistencia - as.numeric(pred)
)
write.csv(predicciones_test, file.path(directorio_tarea, "predicciones_prueba_concrete_R.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
reglas <- c("TREE PODADO", capture.output(print(arbol_podado)),
            "", "RPART CON CONTROLES", capture.output(print(optimal_tree)))
writeLines(reglas, file.path(directorio_tarea, "reglas_arbol_concrete_R.txt"), useBytes = TRUE)

# %% 25. Reproducibilidad
cat(R.version.string, "\n")
cat("tree:", as.character(packageVersion("tree")), "\n")
cat("rpart:", as.character(packageVersion("rpart")), "\n")
cat("rpart.plot:", as.character(packageVersion("rpart.plot")), "\n")
cat("Semilla: 123 | train/test: 80/20 | cv.tree: 10 pliegues\n")
