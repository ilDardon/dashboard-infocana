# Dashboard Infocaña (2015–2026) — Regresión Lineal y ETL

Proyecto Final de la materia **Big Data** (UNACH — Ingeniería en Desarrollo y Tecnologías de Software).

## Contenido del repositorio
- `etl_infocana.py`: Pipeline ETL (Extract, Transform, Load) que integra las zafras 2015–2026 de CONADESUCA (`datos.gob.mx`), normaliza encabezados, elimina nulos, calcula rendimientos derivados, guarda los concentrados en CSV y SQLite, y evalúa la regresión lineal (80% entrenamiento / 20% prueba).
- `app/app.py`: Dashboard interactivo en **Shiny for Python** (desplegado con **ShinyLive** en GitHub Pages) con indicadores KPI, gráfica de serie histórica y pronóstico, regresión lineal con variables operativas (`scikit-learn`) y tabla interactiva (`DataTable`).
- `app/infocana_zafras_limpio.csv` y `app/infocana_ingenios_limpio.csv`: Datasets limpios generados por el ETL.
- `.github/workflows/deploy-shinylive.yml`: Flujo de GitHub Actions que compila y publica automáticamente el dashboard en **GitHub Pages**.
