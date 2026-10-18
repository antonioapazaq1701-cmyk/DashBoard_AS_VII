from pathlib import Path
import json, re, shutil
import nbformat

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'dashboard'
OUT = SITE / 'downloads'
if (OUT / 'tema1_Python.ipynb').exists():
    raise SystemExit('Las copias finales ya existen. prepare.py solo importa en una carpeta vacía.')
OUT.mkdir(parents=True, exist_ok=True)
(SITE / 'assets').mkdir(exist_ok=True)
(SITE / 'data').mkdir(exist_ok=True)
shutil.copy2(ROOT / 'Concrete_Data.csv', OUT / 'Concrete_Data.csv')
shutil.copy2(ROOT / 'Logo_UNAP.png', SITE / 'assets/logo-unap.png')
specs = [
 ('01*/Tarea1/limpieza_concrete.ipynb', '01*/Tarea1/limpieza_concrete_R.R'),
 ('02*/Tarea2/regresion_concrete.ipynb','02*/Tarea2/regresion_concrete_R.R'),
 ('03*/Tarea3/RLMmejorecuacion_concrete.ipynb','03*/Tarea3/RLMmejorecuacion_concrete.R'),
 ('04*/Tarea4/arboles_concrete.ipynb','04*/Tarea4/arboles_concrete_R.R'),
 ('05*/Tarea5/randomforest_concrete_python.ipynb','05*/Tarea5/randomforest_concrete_R.R')
]
manifest=[]
for number, (pyglob, rglob) in enumerate(specs, 1):
    p=next(ROOT.glob(pyglob)); rp=next(ROOT.glob(rglob))
    n=nbformat.read(p, as_version=4)
    for c in n.cells:
        if c.cell_type!='code': continue
        c.outputs=[]; c.execution_count=None
        c.source=c.source.replace('ROOT = Path(__file__).resolve().parents[2]', 'ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()')
        # Descargas portables: ejecutar desde la carpeta que contiene la base.
        c.source=c.source.replace('n_jobs=-1', 'n_jobs=2').replace('"n_jobs": -1','"n_jobs": 2')
        c.source=c.source.replace('optuna.create_study(direction="maximize")','optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))')
    if number==2:
        # La tarea previa usaba pseudoinversa: recuperar LinearRegression del ejemplo Python.
        n.cells[3].source='''from sklearn.linear_model import LinearRegression
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
ax.legend(); plt.tight_layout(); plt.show()'''
        n.cells[7].source=n.cells[7].source.replace('vector = np.r_[1.0, mezcla[variables].to_numpy().ravel()]\nprediccion = float(vector @ beta_multiple)', 'prediccion = float(modelo_multiple.predict(mezcla[variables])[0])')
    if number==4:
        n.cells.append(nbformat.v4.new_markdown_cell('## Árbol completo del modelo seleccionado\nSe conserva el mismo modelo; la figura anterior es una vista truncada. Esta figura permite inspeccionar todos los nodos.'))
        n.cells.append(nbformat.v4.new_code_cell('''fig, ax = plt.subplots(figsize=(max(22, mejor.get_n_leaves() * .32), max(10, mejor.get_depth() * 1.4))))
plot_tree(mejor, feature_names=list(X.columns), filled=True, rounded=True, fontsize=5, precision=2, ax=ax)
ax.set_title("Árbol completo del modelo seleccionado")
fig.savefig("arbol_completo_python.svg", bbox_inches="tight")
plt.close(fig)
print("Árbol completo guardado en arbol_completo_python.svg")'''.replace('* 1.4))))','* 1.4)))')))
    # Notebook y script comparten exactamente las mismas celdas.
    ip=OUT/f'tema{number}_Python.ipynb'; nbformat.write(n,ip)
    title='Preparación'; blocks=[]
    for c in n.cells:
        if c.cell_type=='markdown':
            lines=[x.strip('# *') for x in c.source.splitlines() if x.strip()]
            if lines: title=lines[0]
        else: blocks.append('# %% '+title+'\n'+c.source)
    (OUT/f'tema{number}_Python.py').write_text('from IPython.display import display\n\n'+'\n\n'.join(blocks),encoding='utf-8')
    r=rp.read_text(encoding='utf-8')
    if number==4:
        begin=r.index('if (!exists("directorio_tarea"')
        end=r.index('library(tree)')
        r=r[:begin]+'directorio_tarea <- getwd()\n'+r[end:]
        r=r.replace('file.path(directorio_tarea, "..", "..", "Concrete_Data.csv")','file.path(directorio_tarea, "Concrete_Data.csv")')
    if number==5:
        begin=r.index('directorio_proyecto <-'); end=r.index('library(dplyr)')
        r=r[:begin]+'directorio_proyecto <- getwd()\ndirectorio_tarea <- getwd()\n\n'+r[end:]
        r=r.replace('min(4L,','min(2L,')
        # R² predictivo coherente con sklearn; rsq era correlación al cuadrado.
        r=re.sub(r'\brsq\(', 'rsq_trad(', r)
    if number<4:
        # Dividir en bloques solo al nivel superior de comentarios existentes.
        r='# %% Carga, estructura y preparación\n'+r
        r=re.sub(r'^# (?!TAREA|RLMmejorecuacion|Ejecutar|Seleccion|La base no|corresponder|Deteccion|Duplicados)(?=[A-ZÁÉÍÓÚ])(.+)$',r'# %% \1',r,flags=re.M)
        # Evitar cortar una función: estos scripts no tienen comentarios internos a columna cero salvo los bloques.
    r=r.replace('normalizePath(salida, winslash = "/", mustWork = FALSE)','salida')
    (OUT/f'tema{number}_R.R').write_text(r,encoding='utf-8')
    manifest.append({'id':number,'python_source':str(p.relative_to(ROOT)), 'r_source':str(rp.relative_to(ROOT))})
(SITE/'data/provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('10 programas preparados en',OUT)
