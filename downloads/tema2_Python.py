# Exportación del cuaderno ejecutado. display reproduce las salidas de sus expresiones.

from IPython.display import display

# %% Tarea 2 — Regresión lineal simple y múltiple
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
datos.columns = nombres
print("Dimensiones:", datos.shape)
display(datos.describe().T.round(2))

# %% División entrenamiento/prueba y métricas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

variables = nombres[:-1]
train, test = train_test_split(datos, test_size=0.2, random_state=42)
y_train, y_test = train.resistencia_mpa, test.resistencia_mpa
modelo_simple = LinearRegression().fit(train[["cemento"]], y_train)
modelo_multiple = LinearRegression().fit(train[variables], y_train)
pred_simple = modelo_simple.predict(test[["cemento"]])
pred_multiple = modelo_multiple.predict(test[variables])
print("Entrenamiento:", len(train), "Prueba:", len(test))
print("Correlación de Pearson en entrenamiento:", pearsonr(train.cemento, y_train))
print("Intercepto simple:", modelo_simple.intercept_)
print("Pendiente simple:", modelo_simple.coef_[0])
print(pd.Series(modelo_multiple.coef_, index=variables))
def metricas(y, pred):
    return {"RMSE (MPa)": np.sqrt(mean_squared_error(y,pred)),
            "MAE (MPa)": mean_absolute_error(y,pred), "R²": r2_score(y,pred)}
display(pd.DataFrame([metricas(y_test,pred_simple), metricas(y_test,pred_multiple)],
                     index=["Simple: cemento", "Múltiple: todos"]).round(3))
fig, ax = plt.subplots(figsize=(9,5))
ax.scatter(train.cemento, y_train, alpha=.3, label="Datos de entrenamiento")
x = np.linspace(train.cemento.min(), train.cemento.max(), 100)
ax.plot(x, modelo_simple.predict(pd.DataFrame({"cemento":x})), color="#d56b32", label="Recta estimada")
ax.set(xlabel="Cemento (kg/m³)", ylabel="Resistencia (MPa)")
ax.legend(); plt.tight_layout(); plt.show()

# %% Interpretación y diagnósticos
residuos = y_test - pred_multiple
print("Media de residuos en prueba:", round(float(residuos.mean()), 4))
print("Correlación observación-predicción:", round(float(np.corrcoef(y_test, pred_multiple)[0, 1]), 4))
for real, estimado in zip(y_test[:8], pred_multiple[:8]):
    print(f"Real: {real:6.2f} MPa | Predicción: {estimado:6.2f} MPa | Residuo: {real-estimado:6.2f}")

# %% Predicción de una mezcla nueva
mezcla = pd.DataFrame([{
    "cemento": 350, "escoria": 100, "ceniza_volante": 0, "agua": 180,
    "superplastificante": 8, "agregado_grueso": 1000,
    "agregado_fino": 750, "edad_dias": 28,
}])
prediccion = float(modelo_multiple.predict(mezcla[variables])[0])
print(f"Resistencia estimada: {prediccion:.2f} MPa")