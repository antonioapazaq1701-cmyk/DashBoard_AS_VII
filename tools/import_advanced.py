"""Importa las celdas ejecutadas de las tareas 6–8 sin reentrenar modelos."""
import ast
import base64
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
DOWNLOADS = SITE / 'downloads'
SOURCES = {
    6: '06Maquina de soporte de vectores/Tarea6/svr_concrete_Python.ipynb',
    7: '07XGFBoost/Tarea7/xgboost_concrete_Python.ipynb',
    8: '08Redes Neuronales/Tarea8/red_neuronal_concrete_Python.ipynb',
}

SETUP = ('Entorno de ejecución', 'Bibliotecas y opciones del cuaderno.', 'Versiones impresas y herramientas listas para el análisis.')
LOAD = ('Carga, limpieza y agrupación', 'Concrete_Data.csv: 1.030 filas y nueve columnas.', '1.005 filas, 428 recetas y tablas descriptivas. La edad no define el grupo.')
# Título, entrada y salida interpretada, en el orden de las celdas ejecutadas.
NOTES = {
    6: [SETUP, LOAD,
        ('Distribuciones y dispersión', 'Variables numéricas de las 1.005 filas.', 'Histogramas, dispersión y diagramas de caja: sirven para revisar escala, forma y extremos.'),
        ('Correlación de Pearson', 'Nueve variables numéricas.', 'Mapa de asociaciones lineales. El color no demuestra causalidad ni importancia en SVR.'),
        ('Separación por receta y Pipeline', 'Predictores, respuesta y grupos de composición.', '781 filas de entrenamiento y 224 de prueba; el escalador se ajusta dentro de cada pliegue de CV.'),
        ('Kernel lineal', 'Entrenamiento y valores candidatos de C.', 'Tabla y curva de RMSE CV, diagnóstico y predicciones. El RMSE de prueba del kernel lineal fue 12,288 MPa.'),
        ('Kernel polinomial', 'Kernel de grado 2 y rejilla de C.', 'La tabla ordena C por RMSE CV. En prueba, el error del polinomial fue 16,045 MPa.'),
        ('Kernel gaussiano (RBF)', 'Valores candidatos de C, gamma y epsilon.', 'Tres búsquedas sucesivas: cada etapa conserva la elección previa. No es una búsqueda conjunta exhaustiva.'),
        ('Kernel sigmoidal', 'Rejilla de C y gamma.', 'El mejor candidato de esta familia obtiene 10,372 MPa en CV y 12,032 MPa en prueba.'),
        ('Comparación y nuevas mezclas', 'Cuatro kernels ajustados, prueba reservada y recetas hipotéticas.', 'RBF se elige por CV (6,683 MPa); su RMSE en prueba es 6,823 MPa. La diagonal del gráfico indica predicción igual al valor medido.')],
    7: [SETUP, LOAD,
        ('Distribución de la resistencia', 'Resistencias medidas en MPa.', 'Histograma y caja: resumen la distribución de la respuesta, no el error de un modelo.'),
        ('Correlaciones', 'Nueve variables numéricas.', 'La matriz muestra asociaciones lineales por pares; no mide la ganancia de los árboles.'),
        ('Partición y modelo base', '1.005 filas agrupadas por composición, en unidades originales.', '799 filas de entrenamiento y 206 de prueba. El XGBoost base alcanza un RMSE de 6,640 MPa en prueba.'),
        ('Búsqueda de seis hiperparámetros', 'Rejilla y cinco pliegues agrupados dentro de entrenamiento.', 'Los parámetros se ordenan por RMSE CV; la mejor combinación registra 5,403 MPa.'),
        ('Optimización con parada temprana', 'Subconjuntos de ajuste (627) y validación (172), más prueba reservada (206).', 'La validación decide cuándo parar. En la misma prueba, el RMSE baja de 6,640 a 6,030 MPa.'),
        ('Diagnóstico e importancia', 'Predicciones y residuos de prueba, historial de aprendizaje e importancias.', 'Panel de ajuste: la diagonal indica coincidencia, los residuos se leen respecto a cero y la ganancia mide uso predictivo.'),
        ('Validación final y comparación', 'Configuración elegida y diez pliegues de entrenamiento; métricas de prueba.', 'El RMSE CV medio es 5,350 MPa. La tabla Base/Optimizado compara ambos modelos en la misma prueba; la CV final no es anidada.'),
        ('Predicción de nuevas mezclas', 'Receta hipotética a 7, 28 y 90 días y distribución observada.', 'Estimaciones en MPa. El gráfico ubica la predicción en contexto; no representa un intervalo de confianza.')],
    8: [SETUP, LOAD,
        ('Matriz de correlaciones', 'Predictores y resistencia.', 'Pearson resume relaciones lineales por pares, no los pesos ni la importancia de la red.'),
        ('Distribución y asociación con la respuesta', 'Resistencia y ocho entradas.', 'Histograma y barras de correlación: permiten distinguir escala de resistencia y signo de cada asociación.'),
        ('Partición por receta y escalado', 'Composiciones y resistencia de las filas de entrenamiento.', '705/162/138 filas para entrenamiento, validación y prueba. Las medias y desviaciones de X e y se aprenden solo en entrenamiento.'),
        ('Arquitectura densa', 'Ocho entradas estandarizadas y configuración de Keras.', 'Resumen de cuatro capas ocultas ReLU (128, 64, 32, 16) y salida lineal. Los parámetros entrenables son pesos y sesgos, entre otros parámetros de las capas.'),
        ('Entrenamiento y callbacks', 'Lotes de entrenamiento y validación.', 'EarlyStopping conserva la mejor época (164); el historial permite revisar la pérdida y los cambios de tasa de aprendizaje.'),
        ('Curvas de aprendizaje', 'Historial por época y desviación de la resistencia.', 'La pérdida usa la respuesta escalada e incluye L2. El MAE se expresa en MPa tras convertir su escala.'),
        ('Evaluación en prueba', 'Predicciones estandarizadas y 138 resistencias reservadas.', 'Al recuperar MPa, RMSE = 6,893; MAE = 4,532; R² = 0,8544 en esta prueba.'),
        ('Gráficos de diagnóstico', 'Observados, predichos y residuos de prueba.', 'La diagonal indica predicción perfecta; el residuo es observado menos predicho. La dispersión muestra errores que el promedio oculta.'),
        ('Pesos de la primera capa', 'Pesos absolutos de cada entrada en la primera capa.', 'Los porcentajes normalizan esos pesos. Describen la primera capa y no el aporte global de cada predictor.'),
        ('Predicción de recetas hipotéticas', 'Composición, edad y escaladores aprendidos.', 'Estimaciones en MPa para edades y receta personalizada. No son mediciones físicas.'),
        ('Resumen del ajuste', 'Historial de entrenamiento y métricas reservadas.', 'Se informa la mejor época y el error de prueba de esta red en su propia partición.'),
        ('Importancia por permutación', 'Predictores de prueba y 15 permutaciones por variable.', 'Una caída de R² indica dependencia predictiva. Las barras no son efectos causales y sus valores pueden depender de predictores correlacionados.')],
}


def text(value):
    return ''.join(value) if isinstance(value, list) else value


def import_notebook(number, relative):
    source = ROOT / relative
    raw = source.read_bytes()
    notebook = json.loads(raw)
    key = f'tema{number}_Python'
    blocks, script = [], []
    prose = ''
    for cell in notebook['cells']:
        code = text(cell['source'])
        if cell['cell_type'] == 'markdown':
            prose = code
            continue
        if cell['cell_type'] != 'code':
            continue
        idx = len(blocks)
        title, entrada, salida = NOTES[number][idx]
        if cell.get('execution_count') is None or not cell.get('outputs'):
            raise ValueError(f'{relative}: bloque {idx + 1} sin ejecución guardada')
        outputs = []
        for j, out in enumerate(cell['outputs']):
            kind = out['output_type']
            if kind == 'error':
                raise ValueError(f'{relative}: bloque {idx + 1} contiene un error')
            if kind == 'stream':
                outputs.append({'type': 'text', 'value': text(out['text'])})
                continue
            data = out.get('data', {})
            if 'image/png' in data:
                name = f'{key}-{idx}-{j}.png'
                (SITE / 'assets' / name).write_bytes(base64.b64decode(text(data['image/png'])))
                outputs.append({'type': 'image', 'value': 'assets/' + name})
            elif 'text/html' in data and re.search(r'<table\b', text(data['text/html']), re.I):
                table = re.sub(r'<style\b[^>]*>.*?</style>', '', text(data['text/html']), flags=re.S | re.I)
                outputs.append({'type': 'html', 'value': table})
            elif 'text/plain' in data:
                outputs.append({'type': 'text', 'value': text(data['text/plain'])})
            else:
                raise ValueError(f'{relative}: salida sin representación compatible: {list(data)}')
        blocks.append(dict(title=title, prose=prose, code=code, outputs=outputs, input=entrada, output=salida))
        # Retira únicamente la configuración de visualización de Jupyter en el .py.
        plain = '\n'.join(line for line in code.splitlines() if not line.startswith(
            ('%matplotlib inline', 'from matplotlib_inline.backend_inline import set_matplotlib_formats', 'set_matplotlib_formats(')))
        script.append(f'# %% {title}\n{plain}')
        prose = ''
    assert len(blocks) == len(NOTES[number])
    py = '# Exportado del cuaderno ejecutado. Inicia en la carpeta que contiene Concrete_Data.csv.\n\n' + '\n\n'.join(script) + '\n'
    ast.parse(py)
    (DOWNLOADS / f'{key}.py').write_text(py, encoding='utf-8')
    shutil.copyfile(source, DOWNLOADS / f'{key}.ipynb')
    sha = hashlib.sha256(raw).hexdigest()
    (SITE / 'data' / f'{key}.json').write_text(json.dumps({
        'topic': number, 'language': 'Python', 'blocks': blocks, 'source': relative, 'sha256': sha,
    }, ensure_ascii=False), encoding='utf-8')
    return {'id': number, 'python_source': relative, 'sha256': sha}


if __name__ == '__main__':
    file = SITE / 'data/provenance.json'
    provenance = [p for p in json.loads(file.read_text(encoding='utf-8')) if p['id'] not in SOURCES]
    provenance.extend(import_notebook(i, p) for i, p in SOURCES.items())
    file.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(DOWNLOADS / 'Concrete_Lab_codigos_y_datos.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DOWNLOADS.iterdir()):
            if path.is_file() and path.suffix != '.zip' and not path.name.startswith(('tema4_R_fig-', 'tema5_R_fig-')):
                archive.write(path, path.name)
