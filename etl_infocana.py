"""
Pipeline ETL y Regresión Lineal - Infocaña (CONADESUCA 2015-2026)
Extrae los archivos CSV semanales por zafra, normaliza columnas, limpia nulos,
calcula variables derivadas, agrega por zafra y entrena un modelo de regresión
lineal (80% entrenamiento / 20% prueba).
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Rutas de entrada y salida
BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR.parent / "datos"
APP_DIR = BASE_DIR / "app"
APP_DIR.mkdir(parents=True, exist_ok=True)

# Mapa de normalización de columnas entre zafras 2015-2025 y 2025-2026
RENAME_MAP = {
    "no_s_zafra": "semana",
    "azucar_total": "azucar_producida_total",
    "azucar_refinada": "azucar_producida_refinada",
    "azucar_estandar": "azucar_producida_estandar",
    "eficiencia_fabrica": "eficiencia_en_fabrica",
    "karbe_ton_cania_neta_teorico": "karbe_cana_neta",
}

COLUMNAS_BASE = [
    "ingenio",
    "zafra",
    "semana",
    "cana_molida_bruta",
    "cana_molida_neta",
    "superficie_cosechada",
    "azucar_producida_total",
]


def extract(datos_dir: Path) -> pd.DataFrame:
    """1. Extract: lee e integra todos los CSV de zafras 2015-2026."""
    archivos = sorted(datos_dir.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {datos_dir}")

    frames = []
    for archivo in archivos:
        df_tmp = pd.read_csv(archivo, encoding="utf-8-sig")
        df_tmp.columns = [c.strip().lower() for c in df_tmp.columns]
        df_tmp = df_tmp.rename(columns=RENAME_MAP)
        # Conservar solo columnas clave disponibles
        cols_presentes = [c for c in COLUMNAS_BASE if c in df_tmp.columns]
        frames.append(df_tmp[cols_presentes])

    df_raw = pd.concat(frames, ignore_index=True)
    print(f"[EXTRACT] Archivos integrados: {len(archivos)} | Registros totales: {len(df_raw):,}")
    return df_raw


def transform(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2. Transform: limpia nulos, convierte tipos, crea variables derivadas y agrega por zafra."""
    df = df_raw.copy()

    # Normalizar texto
    df["ingenio"] = df["ingenio"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    df["zafra"] = df["zafra"].astype(str).str.strip()

    # Conversión de tipos numéricos
    num_cols = ["semana", "cana_molida_bruta", "cana_molida_neta", "superficie_cosechada", "azucar_producida_total"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminación de valores nulos en columnas críticas
    antes = len(df)
    df = df.dropna(subset=["ingenio", "zafra", "semana", "cana_molida_bruta", "superficie_cosechada", "azucar_producida_total"]).copy()
    df["cana_molida_neta"] = df["cana_molida_neta"].fillna(0.0)
    print(f"[TRANSFORM] Filas eliminadas por nulos: {antes - len(df)} | Filas válidas: {len(df):,}")

    # Los reportes semanales de Infocaña son acumulados por zafra:
    # se toma la última semana reportada por cada ingenio en cada zafra
    df["anio_inicio"] = df["zafra"].str.split("-").str[0].astype(int)
    idx_ultima = df.groupby(["zafra", "ingenio"])["semana"].idxmax()
    df_ing = df.loc[idx_ultima].copy()
    df_ing["semana_cierre"] = df_ing["semana"].astype(int)

    # Variables derivadas por ingenio
    df_ing["rendimiento_fabrica_calc"] = np.where(
        df_ing["cana_molida_bruta"] > 0,
        (df_ing["azucar_producida_total"] / df_ing["cana_molida_bruta"]) * 100.0,
        0.0,
    ).round(3)
    df_ing["rendimiento_campo_calc"] = np.where(
        df_ing["superficie_cosechada"] > 0,
        df_ing["cana_molida_bruta"] / df_ing["superficie_cosechada"],
        0.0,
    ).round(2)
    df_ing["rendimiento_agroindustrial_calc"] = np.where(
        df_ing["superficie_cosechada"] > 0,
        df_ing["azucar_producida_total"] / df_ing["superficie_cosechada"],
        0.0,
    ).round(3)

    cols_ing = [
        "anio_inicio", "zafra", "ingenio", "semana_cierre",
        "cana_molida_bruta", "cana_molida_neta", "superficie_cosechada",
        "azucar_producida_total", "rendimiento_fabrica_calc",
        "rendimiento_campo_calc", "rendimiento_agroindustrial_calc",
    ]
    df_ing = df_ing[cols_ing].sort_values(["anio_inicio", "ingenio"]).reset_index(drop=True)

    # Agregación anual por zafra
    df_zafra = (
        df_ing.groupby(["anio_inicio", "zafra"], as_index=False)
        .agg(
            semana_max=("semana_cierre", "max"),
            ingenios_activos=("ingenio", "nunique"),
            cana_molida_bruta=("cana_molida_bruta", "sum"),
            cana_molida_neta=("cana_molida_neta", "sum"),
            superficie_cosechada=("superficie_cosechada", "sum"),
            azucar_producida_total=("azucar_producida_total", "sum"),
            rendimiento_promedio_ingenio=("rendimiento_fabrica_calc", "mean"),
            rendimiento_agroindustrial_prom=("rendimiento_agroindustrial_calc", "mean"),
        )
        .sort_values("anio_inicio")
        .reset_index(drop=True)
    )

    df_zafra.insert(0, "periodo", range(1, len(df_zafra) + 1))
    df_zafra["estatus"] = np.where(
        df_zafra["semana_max"] >= 30,
        "Completa",
        "En curso (sem. " + df_zafra["semana_max"].astype(str) + ")",
    )

    for col in ["cana_molida_bruta", "cana_molida_neta", "superficie_cosechada", "azucar_producida_total"]:
        df_zafra[col] = df_zafra[col].round(2)
    df_zafra["rendimiento_promedio_ingenio"] = df_zafra["rendimiento_promedio_ingenio"].round(3)
    df_zafra["rendimiento_agroindustrial_prom"] = df_zafra["rendimiento_agroindustrial_prom"].round(3)

    cols_zafra = [
        "periodo", "anio_inicio", "zafra", "estatus", "semana_max", "ingenios_activos",
        "cana_molida_bruta", "cana_molida_neta", "superficie_cosechada",
        "azucar_producida_total", "rendimiento_promedio_ingenio", "rendimiento_agroindustrial_prom",
    ]
    df_zafra = df_zafra[cols_zafra]
    print(f"[TRANSFORM] Zafras consolidadas: {len(df_zafra)} | Registros por ingenio-zafra: {len(df_ing)}")
    return df_zafra, df_ing


def load(df_zafra: pd.DataFrame, df_ing: pd.DataFrame) -> None:
    """3. Load: almacena los datasets limpios en CSV y en SQLite."""
    csv_zafra = APP_DIR / "infocana_zafras_limpio.csv"
    csv_ing = APP_DIR / "infocana_ingenios_limpio.csv"
    db_path = BASE_DIR / "infocana_consolidado.sqlite"

    df_zafra.to_csv(csv_zafra, index=False, encoding="utf-8")
    df_ing.to_csv(csv_ing, index=False, encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        df_zafra.to_sql("produccion_zafras", conn, if_exists="replace", index=False)
        df_ing.to_sql("produccion_ingenios", conn, if_exists="replace", index=False)

    print(f"[LOAD] CSV guardados en: {APP_DIR}")
    print(f"[LOAD] Base SQLite guardada en: {db_path}")


def entrenar_modelo_temporal(df_zafra: pd.DataFrame) -> None:
    """Entrena y evalúa la regresión lineal temporal (80% entrenamiento / 20% prueba)."""
    df_comp = df_zafra[df_zafra["estatus"] == "Completa"].copy()
    X = df_comp[["periodo"]].values
    y = df_comp["azucar_producida_total"].values

    corte = int(len(df_comp) * 0.8)  # 80% entrenamiento, 20% prueba
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    modelo_eval = LinearRegression().fit(X_train, y_train)
    pred_test = modelo_eval.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    # Ajuste final sobre las 10 zafras completas para pronóstico
    modelo_final = LinearRegression().fit(X, y)
    r2 = r2_score(y, modelo_final.predict(X))
    prox_25_26 = modelo_final.predict([[len(df_comp) + 1]])[0]
    prox_26_27 = modelo_final.predict([[len(df_comp) + 2]])[0]

    print("\n--- MODELO DE REGRESIÓN LINEAL (SERIE TEMPORAL) ---")
    print(f"División temporal: {len(X_train)} zafras entrenamiento (80%) | {len(X_test)} zafras prueba (20%)")
    print(f"MAE prueba:  {mae:,.2f} ton | RMSE prueba: {rmse:,.2f} ton")
    print(f"Ecuación final: Y = {modelo_final.intercept_:,.2f} + ({modelo_final.coef_[0]:,.2f}) * X")
    print(f"R² histórico (10 zafras completas): {r2:.4f}")
    print(f"Último valor real completo (2024-2025): {y[-1]:,.2f} ton")
    print(f"Pronóstico cierre zafra 2025-2026:      {prox_25_26:,.2f} ton")
    print(f"Pronóstico zafra 2026-2027:             {prox_26_27:,.2f} ton")


if __name__ == "__main__":
    raw = extract(DATOS_DIR)
    zafras, ingenios = transform(raw)
    load(zafras, ingenios)
    entrenar_modelo_temporal(zafras)
