# Concrete Lab — programas y datos

Extraer todos los archivos en una misma carpeta y ejecutar desde ella.
Python: crear entorno virtual, instalar `pip install -r requirements.txt`, ejecutar `python tema2_Python.py`.
Redes neuronales: en un entorno con Python 3.12, instalar `pip install -r requirements-redes.txt` y ejecutar `python tema8_Python.py`.
R: establecer la carpeta como directorio de trabajo, ejecutar `source("instalar_paquetes_R.R")` una vez y luego `source("tema2_R.R", echo=TRUE)`.
Los cuadernos ipynb incluyen código y salidas ejecutadas; ejecutar sus celdas de arriba hacia abajo.
La base original conserva 1030 registros. La copia limpia tiene 1005 registros con nombres abreviados.
Los temas 2–5 usan la original para conservar el protocolo de las tareas.
SVR, XGBoost y Redes neuronales leen el CSV original, eliminan duplicados y separan recetas por composición. Los tres cuadernos se entregan con sus salidas originales en Python.
Los programas guardan archivos de resultados en el directorio de trabajo; pueden sobrescribir sus propios archivos de una ejecución anterior.
R y Python usan métodos y particiones diferentes: no comparar directamente sus métricas como un ranking controlado.
Las predicciones son académicas y no sustituyen ensayos de resistencia.

Fuente: I-Cheng Yeh. Concrete Compressive Strength. UCI Machine Learning Repository. DOI 10.24432/C5PK67. CC BY 4.0.
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
