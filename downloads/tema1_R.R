# %% Carga, estructura y preparación
# TAREA 1 — LIMPIEZA DE Concrete_Data.csv
# Ejecutar desde esta carpeta: Limipieza de datos/Tarea1

opciones <- c("Concrete_Data.csv", file.path("..", "..", "Concrete_Data.csv"),
              file.path("..", "Concrete_Data.csv"))
archivo <- opciones[file.exists(opciones)][1]
if (is.na(archivo)) stop("No se encontro Concrete_Data.csv desde el directorio actual.")

datos <- read.csv(archivo, check.names = FALSE, stringsAsFactors = FALSE)
nombres <- c("cemento", "escoria", "ceniza_volante", "agua", "superplastificante",
             "agregado_grueso", "agregado_fino", "edad_dias", "resistencia_mpa")
if (ncol(datos) != length(nombres)) stop("La base no tiene las 9 columnas esperadas.")
names(datos) <- nombres

datos[] <- lapply(datos, function(x) suppressWarnings(as.numeric(as.character(x))))
cat("Dimensiones originales:", nrow(datos), "filas x", ncol(datos), "columnas\n")
cat("\nEstructura:\n")
str(datos)
cat("\nResumen estadistico:\n")
print(summary(datos))
cat("\nFaltantes por variable:\n")
print(colSums(is.na(datos)))
cat("\nPorcentaje faltante (%):\n")
print(round(colMeans(is.na(datos)) * 100, 2))

# %% Duplicados exactos: cuantificar antes de decidir si se eliminan.
n_duplicados <- sum(duplicated(datos))
cat("\nFilas duplicadas exactas:", n_duplicados, "\n")

# %% Deteccion IQR por columna. Los registros se conservan: los extremos pueden
# corresponder a mezclas y edades reales del concreto.
resumen_iqr <- do.call(rbind, lapply(names(datos), function(variable) {
  x <- datos[[variable]]
  q <- quantile(x, c(.25, .75), na.rm = TRUE)
  amplitud <- q[2] - q[1]
  inferior <- q[1] - 1.5 * amplitud
  superior <- q[2] + 1.5 * amplitud
  data.frame(variable = variable, limite_inferior = inferior,
             limite_superior = superior,
             n_atipicos = sum(x < inferior | x > superior, na.rm = TRUE))
}))
cat("\nDeteccion de valores atipicos (IQR):\n")
print(resumen_iqr, row.names = FALSE)

# %% Exportar la base limpia
# La base no contiene faltantes; se retiran solo duplicados exactos si existen.
datos_limpios <- unique(datos)
cat("\nDimensiones luego de retirar duplicados exactos:",
    nrow(datos_limpios), "x", ncol(datos_limpios), "\n")
cat("Faltantes luego de limpieza:", sum(is.na(datos_limpios)), "\n")
salida <- "Concrete_Data_limpia.csv"
write.csv(datos_limpios, salida, row.names = FALSE, na = "")
cat("Archivo generado:", salida, "\n")
