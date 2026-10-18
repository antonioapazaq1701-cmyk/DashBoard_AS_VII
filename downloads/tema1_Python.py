# Exportación del cuaderno ejecutado. display reproduce las salidas de sus expresiones.

from IPython.display import display

# %% Tarea 1 — Limpieza de datos: concreto
from pathlib import Path
import pandas as pd
import numpy as np

candidatos = [Path("Concrete_Data.csv"), Path("../../Concrete_Data.csv"), Path("../Concrete_Data.csv")]
archivo = next((p for p in candidatos if p.exists()), None)
if archivo is None:
    raise FileNotFoundError("No se encontró Concrete_Data.csv; abre el notebook desde una carpeta del proyecto.")

nombres = ["cemento", "escoria", "ceniza_volante", "agua", "superplastificante",
           "agregado_grueso", "agregado_fino", "edad_dias", "resistencia_mpa"]
datos = pd.read_csv(archivo)
if datos.shape[1] != len(nombres):
    raise ValueError("Se esperaban 9 columnas en Concrete_Data.csv")
datos.columns = nombres
datos = datos.apply(pd.to_numeric, errors="coerce")
print(f"Dimensiones: {datos.shape[0]} filas × {datos.shape[1]} columnas")
display(datos.head())

# %% Diagnóstico inicial
print("Tipos de datos:")
print(datos.dtypes)
print("\nFaltantes por columna:")
print(datos.isna().sum())
print("\nPorcentaje faltante (%):")
print((datos.isna().mean() * 100).round(2))
print("\nDuplicados exactos:", int(datos.duplicated().sum()))
display(datos.describe().T.round(2))

# %% Detección IQR y limpieza conservadora
filas_iqr = []
for columna in datos.columns:
    q1, q3 = datos[columna].quantile([0.25, 0.75])
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mascara = datos[columna].lt(inferior) | datos[columna].gt(superior)
    filas_iqr.append({"variable": columna, "limite_inferior": inferior,
                      "limite_superior": superior, "n_atipicos": int(mascara.sum())})
resumen_iqr = pd.DataFrame(filas_iqr)
display(resumen_iqr.round(2))

# %% Detección IQR y limpieza conservadora
datos_limpios = datos.drop_duplicates().copy()
print("Filas retiradas por duplicación exacta:", len(datos) - len(datos_limpios))
print("Faltantes restantes:", int(datos_limpios.isna().sum().sum()))
salida = Path("Concrete_Data_limpia.csv")
datos_limpios.to_csv(salida, index=False)
print("Guardado en:", salida.name)