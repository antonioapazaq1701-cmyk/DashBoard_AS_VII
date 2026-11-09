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
from tempfile import TemporaryDirectory

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers

SEED = 42
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | Semilla {SEED}")

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

# %% Matriz de correlaciones
fig, ax = plt.subplots(figsize=(10, 6.8), layout="constrained")
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1,
            cmap="RdBu_r", square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlación de Pearson · Concrete Data")
plt.show()

# %% Distribución y asociación con la respuesta
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
axes[0].hist(y, bins=30, color="#336b87", edgecolor="white")
axes[0].axvline(y.mean(), color="#b24832", linestyle="--", label=f"Media: {y.mean():.1f} MPa")
axes[0].axvline(y.median(), color="#87611e", linestyle=":", label=f"Mediana: {y.median():.1f} MPa")
axes[0].set(title="Distribución de la resistencia", xlabel="Resistencia (MPa)", ylabel="Frecuencia")
axes[0].legend()
corr_y = df.corr()[TARGET].drop(TARGET).sort_values()
corr_y.plot.barh(ax=axes[1], color=["#b24832" if r < 0 else "#336b87" for r in corr_y])
axes[1].axvline(0, color="0.4", linewidth=0.8)
axes[1].set(title="Correlación con la resistencia", xlabel="Coeficiente de Pearson")
plt.show()

# %% Partición por receta y escalado
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

resto, prueba = dividir(np.arange(len(df)), 0.15, SEED)
entrenamiento, validacion = dividir(resto, 0.15 / 0.85, SEED)
assert set(grupos.iloc[entrenamiento]).isdisjoint(grupos.iloc[prueba])
assert set(grupos.iloc[validacion]).isdisjoint(grupos.iloc[prueba])
mostrar_particiones(Entrenamiento=entrenamiento, Validación=validacion, Prueba=prueba)

X_train, X_val, X_test = X.iloc[entrenamiento], X.iloc[validacion], X.iloc[prueba]
y_train, y_val, y_test = y.iloc[entrenamiento], y.iloc[validacion], y.iloc[prueba]

scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
X_train_sc, X_val_sc, X_test_sc = [
    scaler_X.transform(parte).astype("float32") for parte in (X_train, X_val, X_test)
]
y_train_sc, y_val_sc = [
    scaler_y.transform(parte.to_numpy().reshape(-1, 1)).astype("float32")
    for parte in (y_train, y_val)
]
print(f"Media de entrenamiento usada para escalar y: {scaler_y.mean_[0]:.3f} MPa")
print(f"Desviación usada para escalar y: {scaler_y.scale_[0]:.3f} MPa")

# %% Arquitectura densa
def construir_red(dimension):
    modelo = keras.Sequential([
        keras.Input(shape=(dimension,)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="oculta_1"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="oculta_2"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu", name="oculta_3"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu", name="oculta_4"),
        layers.Dense(1, activation="linear", name="salida"),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return modelo


modelo = construir_red(X_train_sc.shape[1])
modelo.summary(line_length=85)

# %% Entrenamiento y callbacks
opciones = tf.data.Options()
opciones.threading.private_threadpool_size = 1

lotes_train = (
    tf.data.Dataset.from_tensor_slices((X_train_sc, y_train_sc))
    .shuffle(len(X_train_sc), seed=SEED)
    .batch(32)
    .with_options(opciones)
)
lotes_val = tf.data.Dataset.from_tensor_slices((X_val_sc, y_val_sc)).batch(32).with_options(opciones)

parada = callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1)
reducir_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1,
)
with TemporaryDirectory() as carpeta:
    archivo = Path(carpeta) / "mejor_modelo_concrete.keras"
    checkpoint = callbacks.ModelCheckpoint(archivo, monitor="val_loss", save_best_only=True)
    historia = modelo.fit(
        lotes_train, validation_data=lotes_val, epochs=500,
        callbacks=[parada, reducir_lr, checkpoint], shuffle=False, verbose=0,
    )
    modelo = keras.models.load_model(archivo)

historial = pd.DataFrame(historia.history, index=pd.RangeIndex(1, len(historia.epoch) + 1, name="Época"))
mejor_epoca = historial["val_loss"].idxmin()
print(f"Épocas ejecutadas: {len(historial)} | Menor pérdida de validación en época {mejor_epoca}")
display(historial.loc[[mejor_epoca]].round(5))
display(historial.tail().round(5))

# %% Curvas de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
for prefijo, etiqueta in (("", "Entrenamiento"), ("val_", "Validación")):
    axes[0].plot(historial.index, historial[f"{prefijo}loss"], label=etiqueta)
    axes[1].plot(historial.index, historial[f"{prefijo}mae"] * scaler_y.scale_[0], label=etiqueta)
axes[0].set(title="Pérdida por época", ylabel="MSE estandarizado + L2")
axes[1].set(title="Error absoluto medio por época", ylabel="MAE (MPa)")
for ax in axes:
    ax.axvline(mejor_epoca, color="0.4", linestyle="--", label="Mejor época")
    ax.set_xlabel("Época")
    ax.legend(fontsize=8)
plt.show()

# %% Evaluación en prueba
def medir(observado, predicho):
    mse = mean_squared_error(observado, predicho)
    return {
        "MSE (MPa²)": mse,
        "RMSE (MPa)": np.sqrt(mse),
        "MAE (MPa)": mean_absolute_error(observado, predicho),
        "R²": r2_score(observado, predicho),
    }

def predecir_mpa(entradas_escaladas):
    pred_escalada = modelo(entradas_escaladas, training=False).numpy()
    return scaler_y.inverse_transform(pred_escalada).ravel()


y_pred = predecir_mpa(X_test_sc)
metricas = medir(y_test, y_pred)
if (y_test != 0).all():
    metricas["MAPE (%)"] = 100 * np.mean(np.abs((y_test - y_pred) / y_test))
display(pd.Series(metricas, name="Red neuronal · Prueba").round(4).to_frame())
print(f"El error absoluto medio de la predicción es {metricas['MAE (MPa)']:.2f} MPa.")

# %% Gráficos de diagnóstico
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
residuos = y_test.to_numpy() - y_pred
lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())

axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color="#336b87")
axes[0, 0].plot([lo, hi], [lo, hi], "--", color="#b24832")
axes[0, 0].set(title=f"Observado vs. predicho · R² = {metricas['R²']:.3f}",
               xlabel="Observado (MPa)", ylabel="Predicho (MPa)")

axes[0, 1].hist(residuos, bins=25, color="#336b87", edgecolor="white")
axes[0, 1].axvline(0, color="#b24832", linestyle="--")
axes[0, 1].axvline(residuos.mean(), color="#87611e", linestyle=":",
                  label=f"Media: {residuos.mean():.2f} MPa")
axes[0, 1].set(title="Distribución de residuos", xlabel="Residuo (MPa)", ylabel="Frecuencia")
axes[0, 1].legend(fontsize=8)

axes[1, 0].scatter(y_pred, residuos, alpha=0.6, color="#336b87")
axes[1, 0].axhline(0, color="#b24832", linestyle="--")
axes[1, 0].set(title="Residuos frente a valores predichos", xlabel="Predicho (MPa)", ylabel="Residuo (MPa)")

n = min(40, len(y_test))
axes[1, 1].plot(np.arange(1, n + 1), y_test.iloc[:n], "o-", markersize=3, label="Observado")
axes[1, 1].plot(np.arange(1, n + 1), y_pred[:n], "s--", markersize=3, label="Predicho")
axes[1, 1].set(title=f"Primeros {n} ensayos de prueba", xlabel="Ensayo", ylabel="Resistencia (MPa)")
axes[1, 1].legend(fontsize=8)
plt.show()

# %% Pesos de la primera capa
pesos = np.abs(modelo.get_layer("oculta_1").get_weights()[0]).mean(axis=1)
importancia_pesos = pd.Series(100 * pesos / pesos.sum(), index=X.columns, name="Peso relativo (%)").sort_values()
display(importancia_pesos.sort_values(ascending=False).round(3).to_frame())
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
importancia_pesos.plot.barh(ax=ax, color="#336b87")
ax.set(title="Pesos absolutos medios de la primera capa", xlabel="Peso relativo (%)")
plt.show()

# %% Predicción de recetas hipotéticas
mezclas = pd.DataFrame([
    [300, 90, 75, 175, 7, 1000, 760, 7],
    [300, 90, 75, 175, 7, 1000, 760, 28],
    [300, 90, 75, 175, 7, 1000, 760, 90],
], columns=X.columns, index=["7 días", "28 días", "90 días"])

def predecir_mezclas(nuevas):
    escaladas = scaler_X.transform(nuevas.loc[:, X.columns]).astype("float32")
    return nuevas.assign(prediccion_mpa=predecir_mpa(escaladas))


display(predecir_mezclas(mezclas).round(2))
personalizada = pd.DataFrame(
    [[340, 40, 55, 165, 8, 1010, 750, 28]], columns=X.columns, index=["Personalizada"],
)
display(predecir_mezclas(personalizada).round(2))

# %% Resumen del ajuste
print(f"Conclusión · Red densa de Keras con {len(historial)} épocas ejecutadas y mejor época {mejor_epoca}.")
print(f"En prueba: RMSE {metricas['RMSE (MPa)']:.2f} MPa, "
      f"MAE {metricas['MAE (MPa)']:.2f} MPa, R² {metricas['R²']:.4f}.")
print(f"Recetas de prueba: {grupos.iloc[prueba].nunique()} | Ensayos de prueba: {len(y_test)}")

# %% Importancia por permutación
rng = np.random.default_rng(SEED)
repeticiones = 15
caidas = {}

for i, columna in enumerate(X.columns):
    perdidas_r2 = []
    for _ in range(repeticiones):
        X_permutado = X_test_sc.copy()
        X_permutado[:, i] = rng.permutation(X_test_sc[:, i])
        pred_permutada = predecir_mpa(X_permutado)
        perdidas_r2.append(metricas["R²"] - r2_score(y_test, pred_permutada))
    caidas[columna] = perdidas_r2

tabla_perm = pd.DataFrame(caidas).agg(["mean", "std"]).T
tabla_perm.columns = ["Caída media de R²", "Desv. estándar"]
tabla_perm = tabla_perm.sort_values("Caída media de R²")
display(tabla_perm.iloc[::-1].round(4))

fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
ax.barh(tabla_perm.index, tabla_perm["Caída media de R²"], xerr=tabla_perm["Desv. estándar"],
        color="#336b87", capsize=3)
ax.axvline(0, color="0.4", linewidth=0.8)
ax.set(title=f"Importancia por permutación · {repeticiones} repeticiones",
       xlabel="Caída de R² (media ± desviación estándar)")
plt.show()
print(f"Mayor caída media al permutar: {tabla_perm.index[-1]}.")
