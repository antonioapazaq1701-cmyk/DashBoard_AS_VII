# Exportado del cuaderno ejecutado. Inicia en la carpeta que contiene Concrete_Data.csv.

# %% Entorno de ejecución
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight"})
print(f"Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
import xgboost as xgb
from xgboost import XGBRegressor
from joblib import parallel_backend
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score

SEED = 42
print(f"XGBoost {xgb.__version__}")

# %% Carga, limpieza y agrupación
ruta = next(
    (carpeta / "Concrete_Data.csv" for carpeta in (Path.cwd(), *Path.cwd().parents)
     if (carpeta / "Concrete_Data.csv").is_file()),
    None,
)
if ruta is None:
    raise FileNotFoundError("Abre el cuaderno dentro del proyecto que contiene Concrete_Data.csv.")

original = pd.read_csv(ruta)
columnas = [
    "cemento", "escoria", "ceniza_volante", "agua", "superplastificante",
    "agregado_grueso", "agregado_fino", "edad_dias", "resistencia_mpa",
]
if original.shape[1] != len(columnas):
    raise ValueError(f"Se esperaban 9 columnas, se encontraron {original.shape[1]}.")

df = original.set_axis(columnas, axis=1).apply(pd.to_numeric)
if not np.isfinite(df.to_numpy()).all():
    raise ValueError("Se encontraron valores faltantes o no finitos.")

print(f"Archivo: {ruta.name} | Filas originales: {len(df)} | Columnas: {df.shape[1]}")
print(f"Valores faltantes: {df.isna().sum().sum()} | Duplicados exactos: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
TARGET = "resistencia_mpa"
X = df.drop(columns=TARGET)
y = df[TARGET]
# Identificador de receta basado únicamente en las siete cantidades de mezcla.
grupos = df.groupby(columnas[:7], sort=False).ngroup()
print(f"Filas analizadas: {len(df)} | Recetas distintas: {grupos.nunique()}")
display(df.head(3))
display(df.describe().T.round(2))

# %% Distribución de la resistencia
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
axes[0].hist(y, bins=30, color="#336b87", edgecolor="white")
axes[0].set(title="Distribución de la resistencia", xlabel="Resistencia (MPa)", ylabel="Frecuencia")
sns.boxplot(y=y, ax=axes[1], color="#a3cbd9")
axes[1].set(title="Diagrama de caja", ylabel="Resistencia (MPa)")
plt.show()

# %% Correlaciones
fig, ax = plt.subplots(figsize=(10, 6.8), layout="constrained")
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1,
            cmap="RdBu_r", square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlación de Pearson · Concrete Data")
plt.show()

# %% Partición y modelo base
def dividir(indices, proporcion, semilla):
    divisor = GroupShuffleSplit(n_splits=1, test_size=proporcion, random_state=semilla)
    parte_a, parte_b = next(divisor.split(X.iloc[indices], groups=grupos.iloc[indices]))
    a, b = indices[parte_a], indices[parte_b]
    assert set(grupos.iloc[a]).isdisjoint(grupos.iloc[b])
    return a, b


def mostrar_particiones(**particiones):
    display(pd.DataFrame([
        {
            "Conjunto": nombre,
            "Filas": len(indices),
            "% de filas": 100 * len(indices) / len(df),
            "Recetas": grupos.iloc[indices].nunique(),
        }
        for nombre, indices in particiones.items()
    ]).round(2))

entrenamiento, prueba = dividir(np.arange(len(df)), 0.20, SEED)
X_train, X_test = X.iloc[entrenamiento], X.iloc[prueba]
y_train, y_test = y.iloc[entrenamiento], y.iloc[prueba]
grupos_train = grupos.iloc[entrenamiento]
mostrar_particiones(Entrenamiento=entrenamiento, Prueba=prueba)

def medir(observado, predicho):
    mse = mean_squared_error(observado, predicho)
    return {
        "MSE (MPa²)": mse,
        "RMSE (MPa)": np.sqrt(mse),
        "MAE (MPa)": mean_absolute_error(observado, predicho),
        "R²": r2_score(observado, predicho),
    }

def evaluar(observado, predicho):
    metricas = medir(observado, predicho)
    if (observado != 0).all():
        metricas["MAPE (%)"] = 100 * np.mean(np.abs((observado - predicho) / observado))
    return metricas


config = dict(objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0)
xgb_base = XGBRegressor(**config)
xgb_base.fit(X_train, y_train)
pred_base = xgb_base.predict(X_test)
metricas_base = evaluar(y_test, pred_base)
display(pd.Series(metricas_base, name="XGBoost base").round(4).to_frame())

# %% Búsqueda de seis hiperparámetros
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}
busqueda = GridSearchCV(
    XGBRegressor(**config), param_grid, cv=GroupKFold(n_splits=5),
    scoring="neg_root_mean_squared_error", n_jobs=4, error_score="raise",
)
with parallel_backend("threading"):
    busqueda.fit(X_train, y_train, groups=grupos_train)

cv_tabla = pd.DataFrame(busqueda.cv_results_["params"])
cv_tabla["RMSE CV (MPa)"] = -busqueda.cv_results_["mean_test_score"]
print(f"Combinaciones: {len(cv_tabla)} | Ajustes CV: {len(cv_tabla) * busqueda.n_splits_}")
print(f"Mejor RMSE medio de validación: {-busqueda.best_score_:.4f} MPa")
display(cv_tabla.nsmallest(8, "RMSE CV (MPa)").round(4).reset_index(drop=True))

# %% Optimización con parada temprana
ajuste, validacion = dividir(entrenamiento, 0.20, SEED + 1)
mostrar_particiones(Ajuste=ajuste, Validación=validacion, Prueba=prueba)
seleccion = XGBRegressor(
    **busqueda.best_params_, **config, eval_metric="rmse", early_stopping_rounds=20,
)
seleccion.fit(
    X.iloc[ajuste], y.iloc[ajuste],
    eval_set=[(X.iloc[ajuste], y.iloc[ajuste]), (X.iloc[validacion], y.iloc[validacion])],
    verbose=False,
)
arboles = seleccion.best_iteration + 1
historial = seleccion.evals_result()
print(f"Mejor iteración: {arboles} | Rondas ejecutadas: {len(historial['validation_1']['rmse'])}")
print(f"RMSE de validación en la mejor iteración: {seleccion.best_score:.4f} MPa")

params_finales = {**busqueda.best_params_, "n_estimators": arboles}
xgb_opt = XGBRegressor(**params_finales, **config)
xgb_opt.fit(X_train, y_train)
pred_opt = xgb_opt.predict(X_test)
metricas_opt = evaluar(y_test, pred_opt)
display(pd.Series(metricas_opt, name="XGBoost optimizado").round(4).to_frame())

# %% Diagnóstico e importancia
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
residuos = y_test.to_numpy() - pred_opt
lo, hi = min(y_test.min(), pred_opt.min()), max(y_test.max(), pred_opt.max())

ax = axes[0, 0]
ax.scatter(y_test, pred_opt, alpha=0.60, color="#336b87")
ax.plot([lo, hi], [lo, hi], "--", color="#b24832")
ax.set(title=f"Observado vs. predicho · R² = {metricas_opt['R²']:.3f}",
       xlabel="Observado (MPa)", ylabel="Predicho (MPa)")

ax = axes[0, 1]
ax.scatter(pred_opt, residuos, alpha=0.60, color="#336b87")
ax.axhline(0, color="#b24832", linestyle="--")
ax.set(title="Residuos en prueba", xlabel="Predicho (MPa)", ylabel="Residuo (MPa)")

ax = axes[0, 2]
ax.hist(residuos, bins=25, color="#336b87", edgecolor="white")
ax.axvline(0, color="#b24832", linestyle="--")
ax.set(title="Distribución de residuos", xlabel="Residuo (MPa)", ylabel="Frecuencia")

ganancia = pd.Series(xgb_opt.feature_importances_, index=X.columns).sort_values()
ganancia.plot.barh(ax=axes[1, 0], color="#336b87")
axes[1, 0].set(title="Importancia por ganancia", xlabel="Ganancia normalizada")

ax = axes[1, 1]
for clave, etiqueta in (("validation_0", "Ajuste"), ("validation_1", "Validación")):
    rmse = historial[clave]["rmse"]
    ax.plot(np.arange(1, len(rmse) + 1), rmse, label=etiqueta)
ax.axvline(arboles, color="#b24832", linestyle="--", label="Mejor ronda")
ax.set(title="Curva de aprendizaje", xlabel="Ronda de boosting", ylabel="RMSE (MPa)")
ax.legend(fontsize=8)

comparacion = pd.DataFrame({"Base": metricas_base, "Optimizado": metricas_opt})
comparacion.loc[["RMSE (MPa)", "MAE (MPa)"]].plot.bar(ax=axes[1, 2], rot=0,
                                                     color=["#336b87", "#b24832"])
axes[1, 2].set(title="Comparación en prueba", ylabel="Error (MPa)")
plt.show()

# %% Validación final y comparación
with parallel_backend("threading"):
    rmse_cv = -cross_val_score(
        XGBRegressor(**params_finales, **config),
        X_train, y_train, groups=grupos_train, cv=GroupKFold(n_splits=10),
        scoring="neg_root_mean_squared_error", n_jobs=4, error_score="raise",
    )
display(pd.Series(rmse_cv, index=np.arange(1, 11), name="RMSE (MPa)")
        .rename_axis("Pliegue").round(4).to_frame())
print(f"RMSE CV medio: {rmse_cv.mean():.4f} MPa | Desviación estándar: {rmse_cv.std(ddof=1):.4f} MPa")
display(comparacion.round(4))

diferencia = metricas_base["RMSE (MPa)"] - metricas_opt["RMSE (MPa)"]
mejor = "el optimizado" if diferencia > 0 else "el base" if diferencia < 0 else "ambos"
print(f"Conclusión · Menor RMSE en prueba: {mejor}. "
      f"Base: {metricas_base['RMSE (MPa)']:.2f} MPa. Optimizado: {metricas_opt['RMSE (MPa)']:.2f} MPa "
      f"(MAE {metricas_opt['MAE (MPa)']:.2f} MPa, R² {metricas_opt['R²']:.4f}).")
print(f"Mayor ganancia normalizada en el ajuste optimizado: {ganancia.idxmax()}.")

# %% Predicción de nuevas mezclas
mezclas = pd.DataFrame([
    [300, 90, 75, 175, 7, 1000, 760, 7],
    [300, 90, 75, 175, 7, 1000, 760, 28],
    [300, 90, 75, 175, 7, 1000, 760, 90],
], columns=X.columns, index=["7 días", "28 días", "90 días"])

registro = mezclas.loc[["28 días"]]
prediccion = xgb_opt.predict(registro)[0]
print(f"Predicción individual a 28 días: {prediccion:.2f} MPa")
display(mezclas.assign(prediccion_mpa=xgb_opt.predict(mezclas)).round(2))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
axes[0].hist(y, bins=28, color="#336b87", edgecolor="white")
axes[0].axvline(prediccion, color="#b24832", linestyle="--", label=f"{prediccion:.1f} MPa")
axes[0].set(title="Distribución observada", xlabel="Resistencia (MPa)", ylabel="Frecuencia")
axes[0].legend()

percentil = 100 * (y < prediccion).mean()
axes[1].barh(["Predicción"], [percentil], color="#336b87")
axes[1].set(xlim=(0, 100), xlabel="% de ensayos con menor resistencia", title=f"Percentil empírico: {percentil:.1f}")

claves = ganancia.nlargest(5).index
minimos, rangos = X[claves].min(), X[claves].max() - X[claves].min()
perfiles = pd.DataFrame({
    "Mezcla a 28 días": (registro.iloc[0][claves] - minimos) / rangos,
    "Promedio de la base": (X[claves].mean() - minimos) / rangos,
})
perfiles.plot.barh(ax=axes[2], color=["#336b87", "#b24832"])
axes[2].set(title="Perfil frente al promedio", xlabel="Valor normalizado (0–1)")
axes[2].legend(fontsize=8)
plt.show()
