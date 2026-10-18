# Exportación del cuaderno ejecutado. display reproduce las salidas de sus expresiones.

from IPython.display import display

# %% Tarea 4 · Árboles de regresión
import sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, GroupKFold,
    GridSearchCV, RandomizedSearchCV, validation_curve, learning_curve, cross_val_score)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
rutas = [Path('../../Concrete_Data.csv'), Path('Concrete_Data.csv')]
ruta = next((p for p in rutas if p.exists()), None)
if ruta is None: raise FileNotFoundError('Coloca Concrete_Data.csv en la raíz del proyecto.')
datos_originales = pd.read_csv(ruta)
nombres = ['Cemento', 'Escoria', 'Ceniza', 'Agua', 'Superplastificante',
           'Agregado_grueso', 'Agregado_fino', 'Edad', 'Resistencia']
assert datos_originales.shape[1] == 9
datos = datos_originales.copy()
datos.columns = nombres
print('Python:', sys.version.split()[0], '| scikit-learn:', sklearn.__version__)
print('SHA256 de la base:', hashlib.sha256(ruta.read_bytes()).hexdigest())
print('Dimensiones:', datos.shape)
display(pd.DataFrame({'Nombre original': datos_originales.columns, 'Nombre usado': nombres,
                      'Unidad': ['kg/m³']*7 + ['días', 'MPa']}))
display(datos.head())

# %% 1. Inspección y estadística descriptiva
display(datos.describe().T.round(3))
print('Nulos por columna:'); print(datos.isna().sum().to_string())
print('Filas exactamente duplicadas:', datos.duplicated().sum())
assert not datos.isna().any().any()
assert np.isfinite(datos.to_numpy()).all()
assert (datos >= 0).all().all()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(datos.Resistencia, bins=25, color='steelblue', edgecolor='white')
axes[0].set(xlabel='Resistencia (MPa)', ylabel='Frecuencia', title='Distribución de la respuesta')
corr = datos.corr()['Resistencia'].drop('Resistencia').sort_values()
corr.plot.barh(ax=axes[1], color='teal')
axes[1].set(xlabel='Correlación de Pearson', title='Relación marginal con la resistencia')
plt.tight_layout(); plt.show()

# %% 2. Separación de entrenamiento y prueba
X = datos.drop(columns='Resistencia')
y = datos.Resistencia
grupos = pd.util.hash_pandas_object(X, index=False).to_numpy()
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(split.split(X, y, groups=grupos))
X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
g_train = grupos[idx_train]
assert not set(g_train).intersection(grupos[idx_test])
cv = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=g_train))
print(f'Entrenamiento: {len(X_train)} ({len(X_train)/len(X):.1%})')
print(f'Prueba: {len(X_test)} ({len(X_test)/len(X):.1%})')
print('Predictores:', X.shape[1], '| Grupos compartidos entre train y test: 0')

# %% 3. Árbol inicial y referencia
modelos = {
    'Media de entrenamiento': DummyRegressor(strategy='mean'),
    'Árbol sin restricciones': DecisionTreeRegressor(random_state=42),
    'Árbol inicial': DecisionTreeRegressor(max_depth=5, min_samples_split=20,
                                         min_samples_leaf=10, random_state=42)
}
for modelo in modelos.values(): modelo.fit(X_train, y_train)
inicial = modelos['Árbol inicial']
print('Profundidad inicial:', inicial.get_depth(), '| Hojas:', inicial.get_n_leaves())
scores = -cross_val_score(inicial, X_train, y_train, cv=cv,
                         scoring='neg_root_mean_squared_error', n_jobs=1)
print('RMSE por fold:', np.round(scores, 3))
print(f'RMSE CV: {scores.mean():.3f} ± {scores.std():.3f} MPa')

# %% 4. Curva de validación y optimización
profundidades = np.arange(1, 21)
tr, va = validation_curve(DecisionTreeRegressor(random_state=42), X_train, y_train,
    param_name='max_depth', param_range=profundidades, cv=cv,
    scoring='neg_root_mean_squared_error', n_jobs=1)
plt.figure(figsize=(9, 4))
plt.plot(profundidades, -tr.mean(axis=1), 'o-', label='Entrenamiento')
plt.plot(profundidades, -va.mean(axis=1), 'o-', label='Validación CV')
plt.xlabel('Profundidad máxima'); plt.ylabel('RMSE (MPa)')
plt.title('Complejidad y generalización'); plt.legend(); plt.tight_layout(); plt.show()
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), {
    'max_depth': [3, 5, 8, 12, None], 'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 3, 5, 10], 'max_features': [None, 'sqrt'],
    'splitter': ['best', 'random']}, cv=cv,
    scoring='neg_root_mean_squared_error', n_jobs=1, error_score='raise')
grid.fit(X_train, y_train)
random = RandomizedSearchCV(DecisionTreeRegressor(random_state=42), {
    'max_depth': randint(3, 21), 'min_samples_split': randint(2, 31),
    'min_samples_leaf': randint(1, 11), 'max_features': [None, 'sqrt', 'log2'],
    'min_impurity_decrease': uniform(0, 0.1), 'splitter': ['best', 'random']},
    n_iter=80, random_state=42, cv=cv, scoring='neg_root_mean_squared_error',
    n_jobs=1, error_score='raise')
random.fit(X_train, y_train)
for nombre, busqueda in [('GridSearchCV', grid), ('RandomizedSearchCV', random)]:
    print(nombre, '| combinaciones:', len(busqueda.cv_results_['params']))
    print('Mejores parámetros:', busqueda.best_params_)
    print(f'RMSE CV: {-busqueda.best_score_:.4f} MPa')
display(pd.DataFrame(grid.cv_results_).sort_values('rank_test_score')
        [['params', 'mean_test_score', 'std_test_score']].head(5))

# %% 5. Poda por coste y complejidad
poda = GridSearchCV(DecisionTreeRegressor(random_state=42),
    {'ccp_alpha': np.r_[0, np.logspace(-3, 1, 40)]}, cv=cv,
    scoring='neg_root_mean_squared_error', n_jobs=1, error_score='raise')
poda.fit(X_train, y_train)
print('Mejor alpha:', poda.best_params_['ccp_alpha'])
print(f'RMSE CV de poda: {-poda.best_score_:.4f} MPa')
print('Profundidad podada:', poda.best_estimator_.get_depth(),
      '| Hojas:', poda.best_estimator_.get_n_leaves())
plt.figure(figsize=(9, 4))
plt.plot(poda.cv_results_['param_ccp_alpha'].astype(float),
         -poda.cv_results_['mean_test_score'], 'o-')
plt.xscale('symlog', linthresh=0.001); plt.xlabel('ccp_alpha'); plt.ylabel('RMSE CV (MPa)')
plt.title('Selección de la poda'); plt.tight_layout(); plt.show()
busquedas = {'GridSearchCV': grid, 'RandomizedSearchCV': random, 'Árbol podado': poda}
seleccion = max(busquedas, key=lambda nombre: busquedas[nombre].best_score_)
mejor = busquedas[seleccion].best_estimator_
modelos.update({nombre: b.best_estimator_ for nombre, b in busquedas.items()})
print('Modelo elegido antes de evaluar prueba:', seleccion)

# %% 6. Evaluación final en prueba
def metricas(real, pred):
    return {'MAE': mean_absolute_error(real, pred), 'MSE': mean_squared_error(real, pred),
            'RMSE': np.sqrt(mean_squared_error(real, pred)), 'R²': r2_score(real, pred)}
filas = []
for nombre, modelo in modelos.items():
    fila = {'Modelo': nombre, **metricas(y_test, modelo.predict(X_test)),
            'RMSE train': np.sqrt(mean_squared_error(y_train, modelo.predict(X_train)))}
    filas.append(fila)
resultados = pd.DataFrame(filas).set_index('Modelo')
display(resultados.round(4))
pred = mejor.predict(X_test)
comparacion = pd.DataFrame({'Real (MPa)': y_test, 'Predicho (MPa)': pred,
                           'Residuo (MPa)': y_test - pred})
display(comparacion.head(10).round(3))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, nombre in zip(axes, ['Árbol sin restricciones', 'GridSearchCV', 'RandomizedSearchCV']):
    p = modelos[nombre].predict(X_test)
    ax.scatter(y_test, p, alpha=.5, s=15)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax.set(xlabel='Real (MPa)', ylabel='Predicho (MPa)', title=f'{nombre}\nR² = {r2_score(y_test,p):.3f}')
plt.tight_layout(); plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(pred, y_test-pred, s=15, alpha=.5)
axes[0].axhline(0, color='black', linestyle='--')
axes[0].set(xlabel='Predicho (MPa)', ylabel='Residuo (MPa)', title='Residuos del modelo elegido')
axes[1].hist(y_test-pred, bins=25, edgecolor='white')
axes[1].set(xlabel='Residuo (MPa)', ylabel='Frecuencia', title='Distribución de errores')
plt.tight_layout(); plt.show()

# %% 7. Curva de aprendizaje
sizes, st, sv = learning_curve(mejor, X_train, y_train, cv=cv,
    train_sizes=np.linspace(.2, 1, 6), scoring='neg_root_mean_squared_error', n_jobs=1)
plt.figure(figsize=(9, 4))
plt.plot(sizes, -st.mean(axis=1), 'o-', label='Entrenamiento')
plt.plot(sizes, -sv.mean(axis=1), 'o-', label='Validación')
plt.fill_between(sizes, -sv.mean(axis=1)-sv.std(axis=1),
                 -sv.mean(axis=1)+sv.std(axis=1), alpha=.15)
plt.xlabel('Muestras de entrenamiento'); plt.ylabel('RMSE (MPa)')
plt.title('Curva de aprendizaje'); plt.legend(); plt.tight_layout(); plt.show()

# %% 8. Interpretación del árbol y de sus variables
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(mejor, feature_names=list(X.columns), max_depth=3,
          filled=True, rounded=True, fontsize=9, precision=2, ax=ax)
ax.set_title(f'Resistencia del concreto: {seleccion} (vista truncada a 3 niveles)')
plt.tight_layout(); plt.show()
reglas = export_text(mejor, feature_names=list(X.columns), max_depth=100, decimals=3)
Path('reglas_arbol_concrete.txt').write_text(reglas, encoding='utf-8')
print('Profundidad:', mejor.get_depth(), '| Hojas:', mejor.get_n_leaves())
print('Primeras 50 líneas de las reglas:')
print('\n'.join(reglas.splitlines()[:50]))
imp = pd.Series(mejor.feature_importances_, index=X.columns).sort_values(ascending=False)
perm = permutation_importance(mejor, X_test, y_test, n_repeats=20, random_state=42,
                              scoring='neg_root_mean_squared_error', n_jobs=1)
tabla_imp = pd.DataFrame({'Impureza': imp, 'Aumento RMSE por permutación':
                         pd.Series(perm.importances_mean, index=X.columns),
                         'Desviación permutación': pd.Series(perm.importances_std, index=X.columns)})
display(tabla_imp.round(4))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
imp.sort_values().plot.barh(ax=axes[0], color='teal')
axes[0].set(title='Importancia por impureza', xlabel='Reducción relativa de MSE')
orden = np.argsort(perm.importances_mean)
axes[1].barh(np.array(X.columns)[orden], perm.importances_mean[orden],
             xerr=perm.importances_std[orden], color='steelblue')
axes[1].set(title='Importancia por permutación', xlabel='Aumento del RMSE (MPa)')
plt.tight_layout(); plt.show()
# Circular del ejemplo: distribución de importancia por impureza.
imp_positiva = imp[imp > 0]
fig, ax = plt.subplots(figsize=(10, 6))
porciones, _, _ = ax.pie(imp_positiva, autopct=lambda p: f'{p:.1f}%' if p >= 3 else '', startangle=90,
    colors=plt.cm.Set3(np.linspace(0, 1, len(imp_positiva))),
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.2})
etiquetas = [f'{nombre}: {valor / imp_positiva.sum():.1%}' for nombre, valor in imp_positiva.items()]
ax.legend(porciones, etiquetas, title='Variables', loc='center left',
          bbox_to_anchor=(1, 0.5), frameon=False)
ax.set_title('Importancia relativa de las variables (%) — Python')
ax.set_aspect('equal')
plt.tight_layout(); plt.show()
print('El circular reparte la importancia por impureza entre variables. No representa el porcentaje de resistencia explicado por cada una.')

# %% 9. Predicción de una mezcla propuesta
nueva = pd.DataFrame([[300, 100, 0, 180, 6, 1000, 800, 28]], columns=X.columns)
display(nueva)
print(f'Resistencia estimada: {mejor.predict(nueva)[0]:.3f} MPa')
for col in X.columns:
    if not X_train[col].min() <= nueva[col].iloc[0] <= X_train[col].max():
        print('Fuera del rango de entrenamiento:', col)

# %% 10. Conclusiones basadas en los resultados
m = resultados.loc[seleccion]
base = resultados.loc['Árbol sin restricciones']
print(f'1. Se eligió {seleccion} por el menor RMSE en validación cruzada de entrenamiento.')
print(f'2. En prueba: MAE = {m.MAE:.3f} MPa, RMSE = {m.RMSE:.3f} MPa y R² = {m["R²"]:.4f}.')
cambio = (base.RMSE-m.RMSE)/base.RMSE*100
sentido = 'redujo' if cambio >= 0 else 'aumentó'
print(f'3. Frente al árbol sin restricciones, el RMSE {sentido} {abs(cambio):.2f}%. La optimización por CV no garantiza mejora en esta prueba.')
print('4. Variables principales por impureza:', ', '.join(imp.head(3).index)+'.')
print('5. La importancia describe el uso predictivo de las variables; no establece efectos causales.')
print('6. Los resultados se limitan a una partición de esta base. Se requiere validación externa para otras formulaciones.')
resultados.to_csv('metricas_arboles_concrete.csv', encoding='utf-8-sig')
comparacion.to_csv('predicciones_prueba_concrete.csv', encoding='utf-8-sig', index_label='Fila original')

# %% Árbol completo del modelo seleccionado
fig, ax = plt.subplots(figsize=(max(22, mejor.get_n_leaves() * .32), max(10, mejor.get_depth() * 1.4)))
plot_tree(mejor, feature_names=list(X.columns), filled=True, rounded=True, fontsize=5, precision=2, ax=ax)
ax.set_title("Árbol completo del modelo seleccionado")
fig.savefig("arbol_completo_python.svg", bbox_inches="tight")
plt.close(fig)
print("Árbol completo guardado en arbol_completo_python.svg")