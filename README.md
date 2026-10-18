# Concrete Lab · DashBoard_AS_VII

Dashboard académico de Aprendizaje Supervisado, VII semestre. Cinco temas con sus programas R y Python, resultados ejecutados, figuras ampliables y descargas.

## Abrir y publicar

Sitio estático sin instalación de JavaScript ni proceso de compilación. En GitHub: **Settings → Pages → Deploy from a branch → main → /(root) → Save**.

Para verlo localmente, desde esta carpeta: `python -m http.server 8788`, luego abrir `http://localhost:8788`.

## Recorrido

- Inicio: presentación del problema y acceso a los cinco temas.
- Explorar datos: filtros, dispersión, tabla completa y simulación de la ecuación lineal del tema 2 Python.
- Temas: un bloque por pantalla, con entrada, código, salida y explicación breve.
- Presentar: siete pantallas para una exposición.
- Detallado: enlace discreto del footer; teoría, lectura de los bloques y guía de líneas. Es una página pública, no un área protegida.
- Descargas: programas, cuadernos ejecutados, CSV original/limpio y ZIP.

Las ejecuciones están guardadas; Pages no ejecuta R ni Python. `data/tema*.json` contiene las salidas de cada ejecución. Los archivos de `downloads` corresponden al código mostrado.

## Métodos y alcance

Se usan las tareas originales de las cinco carpetas del curso. Las diferencias entre Python y R se conservan y explican. La partición por grupos del árbol Python difiere del muestreo por filas en R. No se presenta un ranking cruzado entre protocolos distintos. Los modelos usan Concrete_Data original; la copia limpia se entrega por separado.

Correcciones: rutas portables; regresión del tema 2 Python con sklearn como el ejemplo; figura del árbol Python completo; R² predictivo en el bosque R; estandarización interna por pliegue para Ridge R y denominador correcto de actualización Lasso; semilla de Optuna; control de paralelismo. Algunos selectores Python conservan el escalado previo a CV de la tarea: limitación explícita.

Los temas 1–3 omiten imputación innecesaria y bloques de categorías inexistentes. R mantiene sus implementaciones de selección AIC, VIF, p-valores y regularización; Python mantiene filtros, wrappers y embedded del ejemplo correspondiente.

## Reproducibilidad

Ver `downloads/LEEME.md`, `downloads/requirements.txt` e `instalar_paquetes_R.R`. Scripts del armado en `tools`. `prepare.py` representa la importación inicial; las copias finales corregidas son las de `downloads`. No regenerar con prepare.py si se desea conservar correcciones posteriores. El ejecutor necesita nbclient y jsonlite; Windows R requiere un directorio temporal sin espacios (`R_TEMP_DIR_DASHBOARD`). La ruta de R puede ajustarse en `tools/execute.py`.

## Créditos

Presentación: Antonio, Universidad Nacional del Altiplano, Puno. Diagramas vectoriales creados para este dashboard; figuras estadísticas generadas por los programas. Logo suministrado en la carpeta del curso.

Datos de referencia: I-Cheng Yeh, [Concrete Compressive Strength, UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), DOI 10.24432/C5PK67, licencia CC BY 4.0. Archivo CSV local conservado e identificado por SHA-256 en el explorador.

Las estimaciones son académicas y no sustituyen ensayos físicos.
