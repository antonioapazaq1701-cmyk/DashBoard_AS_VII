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
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

# %% Distribuciones y dispersión
df.hist(bins=20, figsize=(11, 8), color="#336b87", edgecolor="white")
plt.suptitle("Distribución de las nueve variables")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), layout="constrained", sharey=True)
for ax, columna in zip(axes.flat, X.columns):
    unidad = "días" if columna == "edad_dias" else "kg/m³"
    ax.scatter(X[columna], y, alpha=0.38, s=12, color="#336b87")
    ax.set_xlabel(f"{columna} ({unidad})")
for ax in axes[:, 0]:
    ax.set_ylabel("Resistencia (MPa)")
fig.suptitle("Predictores y resistencia a compresión")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), layout="constrained")
for ax, columna in zip(axes.flat, X.columns):
    sns.boxplot(y=X[columna], ax=ax, color="#a3cbd9")
    ax.set(title=columna, ylabel="días" if columna == "edad_dias" else "kg/m³")
fig.suptitle("Distribución de cada predictor")
plt.show()

# %% Correlación de Pearson
fig, ax = plt.subplots(figsize=(10, 6.8), layout="constrained")
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1,
            cmap="RdBu_r", square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlación de Pearson · Concrete Data")
plt.show()

# %% Separación por receta y Pipeline
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

entrenamiento, prueba = dividir(np.arange(len(df)), 0.20, 1234)
X_train, X_test = X.iloc[entrenamiento], X.iloc[prueba]
y_train, y_test = y.iloc[entrenamiento], y.iloc[prueba]
grupos_train = grupos.iloc[entrenamiento]
mostrar_particiones(Entrenamiento=entrenamiento, Prueba=prueba)

cv = GroupKFold(n_splits=5)


def buscar(kernel, rejilla, **fijos):
    estimador = make_pipeline(StandardScaler(), SVR(kernel=kernel, **fijos))
    busqueda = GridSearchCV(
        estimador, {f"svr__{k}": v for k, v in rejilla.items()},
        scoring="neg_root_mean_squared_error", cv=cv, error_score="raise",
    )
    busqueda.fit(X_train, y_train, groups=grupos_train)
    tabla = pd.DataFrame(busqueda.cv_results_["params"])
    tabla.columns = tabla.columns.str.removeprefix("svr__")
    tabla["RMSE CV (MPa)"] = -busqueda.cv_results_["mean_test_score"]
    print({k.removeprefix("svr__"): v for k, v in busqueda.best_params_.items()})
    display(tabla.sort_values("RMSE CV (MPa)").round(4).reset_index(drop=True))
    return busqueda, tabla


def diagnostico(predicho, titulo):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    minimo = min(y_test.min(), predicho.min())
    maximo = max(y_test.max(), predicho.max())
    axes[0].scatter(y_test, predicho, alpha=0.6, color="#336b87")
    axes[0].plot([minimo, maximo], [minimo, maximo], "--", color="#b24832")
    axes[0].set(xlabel="Observado (MPa)", ylabel="Predicho (MPa)", title="Observado frente a predicho")
    axes[1].scatter(predicho, y_test.to_numpy() - predicho, alpha=0.6, color="#336b87")
    axes[1].axhline(0, linestyle="--", color="#b24832")
    axes[1].set(xlabel="Predicho (MPa)", ylabel="Residuo (MPa)", title="Residuos en prueba")
    fig.suptitle(titulo)
    plt.show()

def medir(observado, predicho):
    mse = mean_squared_error(observado, predicho)
    return {
        "MSE (MPa²)": mse,
        "RMSE (MPa)": np.sqrt(mse),
        "MAE (MPa)": mean_absolute_error(observado, predicho),
        "R²": r2_score(observado, predicho),
    }

mezclas = pd.DataFrame([
    [300, 90, 75, 175, 7, 1000, 760, 7],
    [300, 90, 75, 175, 7, 1000, 760, 28],
    [300, 90, 75, 175, 7, 1000, 760, 90],
], columns=X.columns, index=["7 días", "28 días", "90 días"])

# %% Kernel lineal
lineal, tabla_lineal = buscar("linear", {"C": list(range(1, 50, 2))})
fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
ax.plot(tabla_lineal["C"], tabla_lineal["RMSE CV (MPa)"], "o-", color="#336b87")
ax.axvline(lineal.best_params_["svr__C"], color="#b24832", linestyle="--", label="Mejor C")
ax.set(xlabel="C", ylabel="RMSE CV (MPa)", title="Selección de C · SVR lineal")
ax.legend()
plt.show()

pred_lineal = lineal.predict(X_test)
display(pd.Series(medir(y_test, pred_lineal), name="SVR lineal").round(4).to_frame())
diagnostico(pred_lineal, "SVR lineal")
display(mezclas.assign(prediccion_mpa=lineal.predict(mezclas)).round(2))

# %% Kernel polinomial
poli, _ = buscar("poly", {"C": [0.01, 0.05, 0.13, 1, 5, 15, 30, 60]}, degree=2, gamma="auto")
pred_poli = poli.predict(X_test)
display(pd.Series(medir(y_test, pred_poli), name="SVR polinomial").round(4).to_frame())
display(mezclas.assign(prediccion_mpa=poli.predict(mezclas)).round(2))

# %% Kernel gaussiano (RBF)
rbf_c, _ = buscar("rbf", {"C": [2, 5, 10, 20, 40, 80, 160, 320]}, gamma=0.125, epsilon=0.5)
c_rbf = rbf_c.best_params_["svr__C"]
rbf_gamma, _ = buscar(
    "rbf", {"gamma": [0.0019, 0.0099, 0.02, 0.05, 0.08, 0.125, 0.2, 0.35, 0.6]},
    C=c_rbf, epsilon=0.5,
)
gamma_rbf = rbf_gamma.best_params_["svr__gamma"]
rbf, _ = buscar("rbf", {"epsilon": [0.01, 0.1, 0.4, 0.5, 0.65, 1, 2]}, C=c_rbf, gamma=gamma_rbf)
pred_rbf = rbf.predict(X_test)
display(pd.Series(medir(y_test, pred_rbf), name="SVR RBF").round(4).to_frame())

# %% Kernel sigmoidal
sigmoide, _ = buscar("sigmoid", {"C": [0.1, 0.5, 1, 2, 5, 10, 14], "gamma": [0.01, 0.125, 1]})
pred_sigmoide = sigmoide.predict(X_test)
display(pd.Series(medir(y_test, pred_sigmoide), name="SVR sigmoidal").round(4).to_frame())

# %% Comparación y nuevas mezclas
busquedas = {"Lineal": lineal, "Polinomial": poli, "RBF": rbf, "Sigmoidal": sigmoide}
predicciones = dict(zip(busquedas, [pred_lineal, pred_poli, pred_rbf, pred_sigmoide]))
comparacion = pd.DataFrame([
    {"Kernel": nombre, "RMSE CV (MPa)": -busquedas[nombre].best_score_,
     **medir(y_test, predicho)}
    for nombre, predicho in predicciones.items()
]).set_index("Kernel").sort_values("RMSE CV (MPa)")
display(comparacion.round(4))
display(pd.DataFrame({"Observado (MPa)": y_test, **predicciones}).head(10).round(2))

ganador = comparacion.index[0]
diagnostico(pred_rbf, "SVR gaussiano (RBF)")
if ganador not in {"RBF", "Lineal"}:
    diagnostico(predicciones[ganador], f"SVR {ganador}")
display(mezclas.assign(prediccion_mpa=busquedas[ganador].predict(mezclas)).round(2))

resultado = comparacion.loc[ganador]
print(f"Conclusión · {ganador} obtuvo el menor RMSE en CV ({resultado['RMSE CV (MPa)']:.2f} MPa).")
print(f"En prueba: RMSE {resultado['RMSE (MPa)']:.2f} MPa, "
      f"MAE {resultado['MAE (MPa)']:.2f} MPa, R² {resultado['R²']:.4f}.")
