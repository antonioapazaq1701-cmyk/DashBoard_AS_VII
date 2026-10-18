# Exportación del cuaderno ejecutado. display reproduce las salidas de sus expresiones.

from IPython.display import display

# %% 1. Librerías y carga de datos
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

candidatos = [Path("Concrete_Data.csv"), Path("../../Concrete_Data.csv"), Path("../Concrete_Data.csv")]
archivo = next((p for p in candidatos if p.exists()), None)
if archivo is None:
    raise FileNotFoundError("No se encontró Concrete_Data.csv; ejecuta el notebook desde el proyecto o Tarea3.")

nombres = ["cemento", "escoria", "ceniza_volante", "agua", "superplastificante",
           "agregado_grueso", "agregado_fino", "edad_dias", "resistencia_mpa"]
datos = pd.read_csv(archivo)
if datos.shape[1] != len(nombres):
    raise ValueError(f"Se esperaban 9 columnas y se encontraron {datos.shape[1]}.")
datos.columns = nombres
datos = datos.apply(pd.to_numeric, errors="coerce")
print(f"Archivo: {archivo.name}")
print(f"Dimensiones: {datos.shape[0]} filas × {datos.shape[1]} columnas")
display(datos.head())

# %% 2. Revisión exploratoria
print("Tipos de datos:")
print(datos.dtypes)
print("\nFaltantes:")
print(datos.isna().sum())
print("Duplicados exactos:", int(datos.duplicated().sum()))
display(datos.describe().T.round(2))

# %% 2. Revisión exploratoria
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.flat, datos.columns):
    sns.histplot(data=datos, x=col, kde=True, ax=ax, color="steelblue")
    ax.set_title(col.replace("_", " "))
plt.tight_layout()
plt.show()

# %% 3. Correlación con la resistencia
correlaciones = datos.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=.4, ax=ax)
ax.set_title("Correlaciones entre variables")
plt.tight_layout()
plt.show()

corr_y = correlaciones["resistencia_mpa"].drop("resistencia_mpa").sort_values(key=np.abs, ascending=False)
print("Correlación de Pearson con resistencia:")
print(corr_y.round(3))

# %% 3. Correlación con la resistencia
top_vars = corr_y.abs().head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, top_vars):
    sns.regplot(data=datos, x=col, y="resistencia_mpa", ax=ax,
                scatter_kws={"alpha": .35, "s": 18}, line_kws={"color": "darkred"})
    ax.set_title(f"Resistencia vs. {col.replace('_', ' ')}")
plt.tight_layout()
plt.show()

# %% 4. Preparación y modelo base
respuesta = "resistencia_mpa"
feature_names = [c for c in datos.columns if c != respuesta]
X = datos[feature_names]
y = datos[respuesta]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=42, shuffle=True
)
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas")
display(X_train_sc.head())

# %% 4. Preparación y modelo base
resultados = []
def evaluar_modelo(Xtr, Xte, ytr, yte, nombre, estimador=None):
    modelo = estimador if estimador is not None else LinearRegression()
    modelo.fit(Xtr, ytr)
    pred_train = modelo.predict(Xtr)
    pred_test = modelo.predict(Xte)
    cv = cross_val_score(LinearRegression(), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    fila = {"Método": nombre,
            "Variables": list(Xtr.columns),
            "n_variables": Xtr.shape[1],
            "R²_train": r2_score(ytr, pred_train),
            "R²_test": r2_score(yte, pred_test),
            "RMSE_test": np.sqrt(mean_squared_error(yte, pred_test)),
            "MAE_test": mean_absolute_error(yte, pred_test),
            "CV_R²_media": cv.mean(),
            "CV_R²_sd": cv.std()}
    resultados.append(fila)
    print(f"{nombre}: R² test={fila['R²_test']:.4f} | RMSE={fila['RMSE_test']:.3f} MPa | CV R²={cv.mean():.4f} ± {cv.std():.4f}")
    return modelo

print("MODELO BASE — todos los predictores")
modelo_base = evaluar_modelo(X_train_sc, X_test_sc, y_train, y_test, "Base: todas")

# %% 5. Métodos de filtro
print("Correlación de Pearson absoluta con la respuesta (train):")
corr_train = X_train_sc.corrwith(y_train).abs().sort_values(ascending=False)
print(corr_train.round(3))

selector_f = SelectKBest(score_func=f_regression, k="all").fit(X_train_sc, y_train)
ftest = pd.DataFrame({"variable": feature_names, "F": selector_f.scores_, "p_valor": selector_f.pvalues_}).sort_values("F", ascending=False)
print("\nF-test:")
print(ftest.round(4).to_string(index=False))

selector_mi = SelectKBest(score_func=mutual_info_regression, k="all").fit(X_train_sc, y_train)
mutua = pd.Series(selector_mi.scores_, index=feature_names, name="informacion_mutua").sort_values(ascending=False)
print("\nInformación mutua:")
print(mutua.round(4))

# %% 5. Métodos de filtro
def calcular_vif(X_df):
    matriz = sm.add_constant(X_df, has_constant="add")
    return pd.DataFrame({"variable": X_df.columns,
                         "VIF": [variance_inflation_factor(matriz.to_numpy(), i + 1)
                                 for i in range(X_df.shape[1])]})

X_vif = X_train_sc.copy()
while X_vif.shape[1] > 2:
    tabla_vif = calcular_vif(X_vif).replace([np.inf, -np.inf], np.inf)
    max_row = tabla_vif.loc[tabla_vif["VIF"].idxmax()]
    if max_row["VIF"] <= 10:
        break
    print(f"Se retira {max_row['variable']} (VIF={max_row['VIF']:.2f})")
    X_vif = X_vif.drop(columns=max_row["variable"])
print("\nVIF final:")
print(calcular_vif(X_vif).round(3).to_string(index=False))
modelo_vif = evaluar_modelo(X_train_sc[X_vif.columns], X_test_sc[X_vif.columns],
                            y_train, y_test, "Filtro VIF")

# %% 6. Métodos wrapper
n_objetivo = max(1, len(feature_names) // 2)
rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_objetivo)
rfe.fit(X_train_sc, y_train)
vars_rfe = [v for v, conservar in zip(feature_names, rfe.support_) if conservar]
print("RFE:", vars_rfe)
modelo_rfe = evaluar_modelo(X_train_sc[vars_rfe], X_test_sc[vars_rfe], y_train, y_test, "RFE")

rfecv = RFECV(estimator=LinearRegression(), step=1,
              cv=KFold(5, shuffle=True, random_state=42), scoring="r2",
              min_features_to_select=1)
rfecv.fit(X_train_sc, y_train)
vars_rfecv = [v for v, conservar in zip(feature_names, rfecv.support_) if conservar]
print("RFECV seleccionó:", vars_rfecv)
modelo_rfecv = evaluar_modelo(X_train_sc[vars_rfecv], X_test_sc[vars_rfecv],
                              y_train, y_test, "RFECV")

# %% 6. Métodos wrapper
sfs_forward = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=n_objetivo, direction="forward",
    scoring="r2", cv=KFold(5, shuffle=True, random_state=42), n_jobs=1)
sfs_forward.fit(X_train_sc, y_train)
vars_forward = list(X_train_sc.columns[sfs_forward.get_support()])
print("Forward:", vars_forward)
modelo_forward = evaluar_modelo(X_train_sc[vars_forward], X_test_sc[vars_forward],
                                y_train, y_test, "Wrapper forward")

sfs_backward = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=n_objetivo, direction="backward",
    scoring="r2", cv=KFold(5, shuffle=True, random_state=42), n_jobs=1)
sfs_backward.fit(X_train_sc, y_train)
vars_backward = list(X_train_sc.columns[sfs_backward.get_support()])
print("Backward:", vars_backward)
modelo_backward = evaluar_modelo(X_train_sc[vars_backward], X_test_sc[vars_backward],
                                 y_train, y_test, "Wrapper backward")

# %% Selección hacia adelante y hacia atrás por AIC
def ajustar_ols(columnas):
    X_sm = sm.add_constant(X_train_sc[columnas], has_constant="add")
    return sm.OLS(y_train, X_sm).fit()

def seleccion_forward_aic():
    disponibles = list(feature_names)
    elegidas = []
    actual = ajustar_ols(elegidas).aic
    while disponibles:
        candidatos = [(ajustar_ols(elegidas + [v]).aic, v) for v in disponibles]
        mejor_aic, variable = min(candidatos)
        if mejor_aic >= actual:
            break
        elegidas.append(variable)
        disponibles.remove(variable)
        actual = mejor_aic
        print(f"Agrega {variable}: AIC={actual:.2f}")
    return elegidas

def seleccion_backward_aic():
    elegidas = list(feature_names)
    actual = ajustar_ols(elegidas).aic
    while len(elegidas) > 1:
        candidatos = [(ajustar_ols([v for v in elegidas if v != quitar]).aic, quitar) for quitar in elegidas]
        mejor_aic, variable = min(candidatos)
        if mejor_aic >= actual:
            break
        elegidas.remove(variable)
        actual = mejor_aic
        print(f"Retira {variable}: AIC={actual:.2f}")
    return elegidas

vars_forward_aic = seleccion_forward_aic()
vars_backward_aic = seleccion_backward_aic()
print("\nForward AIC:", vars_forward_aic)
print("Backward AIC:", vars_backward_aic)
modelo_forward_aic = evaluar_modelo(X_train_sc[vars_forward_aic], X_test_sc[vars_forward_aic],
                                   y_train, y_test, "AIC forward")
modelo_backward_aic = evaluar_modelo(X_train_sc[vars_backward_aic], X_test_sc[vars_backward_aic],
                                    y_train, y_test, "AIC backward")

# %% 7. Métodos embebidos: Lasso y Ridge
lasso_cv = LassoCV(cv=KFold(5, shuffle=True, random_state=42), random_state=42,
                   max_iter=50000, alphas=100).fit(X_train_sc, y_train)
vars_lasso = [v for v, coef in zip(feature_names, lasso_cv.coef_) if abs(coef) > 1e-8]
print(f"Alpha Lasso: {lasso_cv.alpha_:.6g}")
print(pd.Series(lasso_cv.coef_, index=feature_names, name="coeficiente").sort_values(key=np.abs, ascending=False).round(4))
if not vars_lasso:
    vars_lasso = [feature_names[int(np.argmax(np.abs(lasso_cv.coef_)))]]
pred_lasso = lasso_cv.predict(X_test_sc)
cv_lasso = cross_val_score(LassoCV(cv=5, random_state=42, max_iter=50000),
                           X_train_sc, y_train,
                           cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
resultados.append({"Método": "LassoCV", "Variables": vars_lasso,
                   "n_variables": len(vars_lasso),
                   "R²_train": lasso_cv.score(X_train_sc, y_train),
                   "R²_test": r2_score(y_test, pred_lasso),
                   "RMSE_test": np.sqrt(mean_squared_error(y_test, pred_lasso)),
                   "MAE_test": mean_absolute_error(y_test, pred_lasso),
                   "CV_R²_media": cv_lasso.mean(), "CV_R²_sd": cv_lasso.std()})
print(f"LassoCV real: R² test={r2_score(y_test,pred_lasso):.4f} | RMSE={np.sqrt(mean_squared_error(y_test,pred_lasso)):.3f} MPa | CV R²={cv_lasso.mean():.4f}")

ridge_cv = RidgeCV(alphas=np.logspace(-3, 4, 100), cv=KFold(5, shuffle=True, random_state=42))
ridge_cv.fit(X_train_sc, y_train)
y_pred_ridge = ridge_cv.predict(X_test_sc)
ridge_fila = {"Método": "Ridge", "Variables": feature_names, "n_variables": len(feature_names),
              "R²_train": ridge_cv.score(X_train_sc, y_train),
              "R²_test": r2_score(y_test, y_pred_ridge),
              "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
              "MAE_test": mean_absolute_error(y_test, y_pred_ridge),
              "CV_R²_media": np.nan, "CV_R²_sd": np.nan}
resultados.append(ridge_fila)
print(f"Alpha Ridge: {ridge_cv.alpha_:.6g} | R² test={ridge_fila['R²_test']:.4f} | RMSE={ridge_fila['RMSE_test']:.3f} MPa")

# %% Eliminación iterativa por p-valor
vars_p = list(feature_names)
while len(vars_p) > 1:
    fit = sm.OLS(y_train, sm.add_constant(X_train_sc[vars_p], has_constant="add")).fit()
    pvals = fit.pvalues.drop("const")
    variable = pvals.idxmax()
    if pvals[variable] <= 0.05:
        break
    print(f"Se elimina {variable}: p={pvals[variable]:.4g}")
    vars_p.remove(variable)
print("Variables por p-valor:", vars_p)
modelo_p = evaluar_modelo(X_train_sc[vars_p], X_test_sc[vars_p], y_train, y_test,
                          "P-valores (OLS)")
print(sm.OLS(y_train, sm.add_constant(X_train_sc[vars_p], has_constant="add")).fit().summary())

# %% 8. Bosque aleatorio y comparación final
bosque_base = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
bosque_base.fit(X_train, y_train)
p_bosque_base = bosque_base.predict(X_test)
print(f"Bosque base | R² test={r2_score(y_test,p_bosque_base):.4f} | RMSE={np.sqrt(mean_squared_error(y_test,p_bosque_base)):.3f} MPa")

param_grid = {"n_estimators": [200, 400], "max_depth": [None, 8],
              "min_samples_leaf": [1, 3]}
busqueda = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1),
                        param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42),
                        scoring="neg_root_mean_squared_error", n_jobs=1)
busqueda.fit(X_train, y_train)
mejor_bosque = busqueda.best_estimator_
p_bosque = mejor_bosque.predict(X_test)
rf_fila = {"Método": "Random Forest (CV)", "Variables": feature_names, "n_variables": len(feature_names),
           "R²_train": mejor_bosque.score(X_train,y_train), "R²_test": r2_score(y_test,p_bosque),
           "RMSE_test": np.sqrt(mean_squared_error(y_test,p_bosque)),
           "MAE_test": mean_absolute_error(y_test,p_bosque),
           "CV_R²_media": np.nan, "CV_R²_sd": np.nan}
resultados.append(rf_fila)
print("Mejores parámetros:", busqueda.best_params_)
importancias = pd.Series(mejor_bosque.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nImportancia de variables:")
print(importancias.round(4))

# %% 8. Bosque aleatorio y comparación final
tabla_resultados = pd.DataFrame(resultados).drop_duplicates(subset="Método", keep="last")
tabla_resultados = tabla_resultados.sort_values("RMSE_test").reset_index(drop=True)
columnas = ["Método", "n_variables", "R²_train", "R²_test", "RMSE_test", "MAE_test", "CV_R²_media", "CV_R²_sd"]
display(tabla_resultados[columnas].round(4))

# %% 9. Ecuación mejorada y diagnóstico
vars_mejor = vars_backward_aic
modelo_mejor = sm.OLS(y_train, sm.add_constant(X_train_sc[vars_mejor], has_constant="add")).fit()
pred_mejor = modelo_mejor.predict(sm.add_constant(X_test_sc[vars_mejor], has_constant="add"))
print("Variables de la ecuación mejorada:", vars_mejor)
print(modelo_mejor.summary())
print(f"\nDesempeño de la ecuación mejorada: R² test={r2_score(y_test,pred_mejor):.4f} | RMSE={np.sqrt(mean_squared_error(y_test,pred_mejor)):.3f} MPa | MAE={mean_absolute_error(y_test,pred_mejor):.3f} MPa")

# %% 9. Ecuación mejorada y diagnóstico
residuos = y_test.to_numpy() - np.asarray(pred_mejor)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(pred_mejor, residuos, alpha=.55, color="steelblue")
axes[0].axhline(0, color="darkred", linestyle="--")
axes[0].set(xlabel="Predicción (MPa)", ylabel="Residuo (MPa)", title="Residuos vs. predicción")
axes[1].hist(residuos, bins=25, color="slategray", edgecolor="white")
axes[1].set(xlabel="Residuo (MPa)", title="Distribución de residuos")
plt.tight_layout()
plt.show()
print("Media de residuos en prueba:", round(float(residuos.mean()), 4))