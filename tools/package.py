from pathlib import Path
import json, shutil, hashlib, zipfile, importlib.metadata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[2];SITE=ROOT/'dashboard';DOWN=SITE/'downloads'
for page in ['tema','detallado','datos','descargas']:shutil.copyfile(SITE/'index.html',SITE/(page+'.html'))
df=pd.read_csv(DOWN/'Concrete_Data.csv')
labels=['Cemento','Escoria','Ceniza volante','Agua','Superplastificante','Agregado grueso','Agregado fino','Edad','Resistencia']
df.columns=['cemento','escoria','ceniza_volante','agua','superplastificante','agregado_grueso','agregado_fino','edad_dias','resistencia_mpa']
train,test=train_test_split(df,test_size=.2,random_state=42)
model=LinearRegression().fit(train.iloc[:,:8],train.iloc[:,8])
data={'labels':labels,'rows':df.values.tolist(),'min':df.min().tolist(),'max':df.max().tolist(),'medians':df.median().tolist(),'missing':int(df.isna().sum().sum()),'duplicates':int(df.duplicated().sum()),'sha256':hashlib.sha256((DOWN/'Concrete_Data.csv').read_bytes()).hexdigest(),'linear':{'intercept':float(model.intercept_),'coef':model.coef_.tolist(),'rmse':float(np.sqrt(mean_squared_error(test.iloc[:,8],model.predict(test.iloc[:,:8]))))}}
(SITE/'data/dataset.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
df.drop_duplicates().to_csv(DOWN/'Concrete_Data_limpia.csv',index=False)
libraries=['numpy','pandas','matplotlib','seaborn','scipy','scikit-learn','statsmodels','optuna','nbformat','nbclient','ipykernel','notebook']
lines=[]
for lib in libraries:
 try:lines.append(lib+'=='+importlib.metadata.version(lib))
 except importlib.metadata.PackageNotFoundError:lines.append(lib)
(DOWN/'requirements.txt').write_text('\n'.join(lines)+'\n')
(DOWN/'instalar_paquetes_R.R').write_text('paquetes <- c("tree", "rpart", "rpart.plot", "dplyr", "tidyr", "ggplot2", "tidymodels", "ranger", "doParallel", "recipes", "workflows", "tibble", "jsonlite")\nfaltan <- setdiff(paquetes, rownames(installed.packages()))\nif (length(faltan)) install.packages(faltan, repos = "https://cloud.r-project.org")\n',encoding='utf-8')
(DOWN/'LEEME.md').write_text('''# Concrete Lab — programas y datos

Extraer todos los archivos en una misma carpeta y ejecutar desde ella.
Python: crear entorno virtual, instalar `pip install -r requirements.txt`, ejecutar `python tema2_Python.py`.
R: establecer la carpeta como directorio de trabajo, ejecutar `source("instalar_paquetes_R.R")` una vez y luego `source("tema2_R.R", echo=TRUE)`.
Los cuadernos ipynb incluyen código y salidas ejecutadas; ejecutar sus celdas de arriba hacia abajo.
La base original conserva 1030 registros. La copia limpia tiene 1005 registros con nombres abreviados.
Los temas 2–5 usan la original para conservar el protocolo de las tareas.
Los programas guardan archivos de resultados en el directorio de trabajo; pueden sobrescribir sus propios archivos de una ejecución anterior.
R y Python usan métodos y particiones diferentes: no comparar directamente sus métricas como un ranking controlado.
Las predicciones son académicas y no sustituyen ensayos de resistencia.

Fuente: I-Cheng Yeh. Concrete Compressive Strength. UCI Machine Learning Repository. DOI 10.24432/C5PK67. CC BY 4.0.
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
''',encoding='utf-8')
with zipfile.ZipFile(DOWN/'Concrete_Lab_codigos_y_datos.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in DOWN.iterdir():
  if p.is_file() and p.suffix!='.zip' and not p.name.startswith('tema4_R_fig-') and not p.name.startswith('tema5_R_fig-'):z.write(p,p.name)
print('Datos, páginas y descargas preparados. RMSE simulador:',data['linear']['rmse'])
