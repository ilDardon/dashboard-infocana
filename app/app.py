from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from shiny import App, ui, render, reactive

# Cargar datasets limpios generados por el pipeline ETL
APP_DIR = Path(__file__).parent
df_zafras = pd.read_csv(APP_DIR / "infocana_zafras_limpio.csv")
df_ingenios = pd.read_csv(APP_DIR / "infocana_ingenios_limpio.csv")

df_completas_all = df_zafras[df_zafras["estatus"] == "Completa"].copy().reset_index(drop=True)
df_parcial = df_zafras[df_zafras["estatus"] != "Completa"].copy().reset_index(drop=True)
zafras_completas_set = set(df_completas_all["zafra"])
df_ing_comp_all = df_ingenios[df_ingenios["zafra"].isin(zafras_completas_set)].copy()

ANIO_MIN = int(df_completas_all["anio_inicio"].min())
ANIO_MAX = int(df_completas_all["anio_inicio"].max())

VARIABLES_X = {
    "periodo": "Periodo / Zafra (Serie temporal)",
    "cana_molida_bruta": "Caña molida bruta (ton)",
    "cana_molida_neta": "Caña molida neta (ton)",
    "superficie_cosechada": "Superficie cosechada (ha)",
}

CSS_CUSTOM = """
body { background-color: #f4f6f9; font-family: 'Segoe UI', Arial, sans-serif; }
.top-banner {
    background: linear-gradient(90deg, #1b3a63 0%, #244b7c 100%);
    color: #ffffff;
    padding: 13px 22px;
    border-radius: 6px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.top-banner h2 { margin: 0; font-size: 1.32rem; font-weight: 600; }
.top-banner .sub-info { font-size: 0.84rem; opacity: 0.92; text-align: right; line-height: 1.35; }
.kpi-card {
    border-radius: 8px;
    padding: 13px 16px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-blue { background-color: #eaf2fb; color: #163a63; }
.kpi-slate { background-color: #eef2f6; color: #1f2d3d; }
.kpi-green { background-color: #e8f5ec; color: #1b5e32; }
.kpi-label { font-size: 0.80rem; font-weight: 600; opacity: 0.85; margin-bottom: 3px; }
.kpi-value { font-size: 1.42rem; font-weight: 700; line-height: 1.2; }
.kpi-sub { font-size: 0.78rem; opacity: 0.82; margin-top: 3px; }
.model-box {
    background-color: #eef7f0;
    border: 1px solid #c6e3cc;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.84rem;
    color: #1d4d2b;
    margin-top: 8px;
    line-height: 1.48;
}
.warn-box {
    background-color: #fff8e6;
    border: 1px solid #f0cf85;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.82rem;
    color: #7c5400;
    margin-top: 6px;
}
shiny-data-frame table {
    font-size: 0.86rem !important;
}
shiny-data-frame th, shiny-data-frame td {
    white-space: nowrap !important;
    padding: 6px 10px !important;
}
.footer-credits {
    text-align: center;
    font-size: 0.82rem;
    color: #5c6773;
    margin: 14px 0 6px 0;
}
"""

app_ui = ui.page_fluid(
    ui.tags.style(CSS_CUSTOM),
    ui.div(
        ui.div(
            ui.h2("Dashboard - Pronóstico de Producción de Azúcar (Infocaña)"),
            ui.span("CONADESUCA 2015–2026 · Regresión Lineal (scikit-learn)", style="font-size:0.84rem; opacity:0.88;"),
        ),
        ui.div(
            ui.strong("Equipo: "),
            "Luis Antonio Dardón Alvarez · Ever Daniel Gomez Dominguez",
            ui.tags.br(),
            "UNACH · Big Data (ETL + ShinyLive)",
            class_="sub-info",
        ),
        class_="top-banner",
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Configuración del pronóstico", style="color:#1b3a63; font-weight:700; margin-bottom:10px;"),
            ui.input_select(
                "var_x",
                "Variable independiente (X):",
                choices=VARIABLES_X,
                selected="periodo",
            ),
            ui.input_slider(
                "rango_anios",
                "Rango histórico de zafras (año inicio):",
                min=ANIO_MIN,
                max=ANIO_MAX,
                value=[ANIO_MIN, ANIO_MAX],
                step=1,
                sep="",
            ),
            ui.panel_conditional(
                "input.var_x === 'periodo'",
                ui.input_radio_buttons(
                    "horizonte",
                    "Horizonte de pronóstico:",
                    choices={
                        "1": "1 zafra siguiente",
                        "2": "2 zafras siguientes",
                        "3": "3 zafras siguientes",
                    },
                    selected="2",
                ),
                ui.input_checkbox(
                    "mostrar_parcial",
                    "Mostrar avance parcial 2025-2026 (sem. 14)",
                    value=False,
                ),
            ),
            ui.input_select(
                "tabla_vista",
                "Vista en tabla de datos:",
                choices={
                    "zafras": "Concentrado anual por zafra",
                    "ingenios": "Detalle por ingenio y zafra",
                },
                selected="zafras",
            ),
            ui.hr(),
            ui.output_ui("panel_modelo"),
            width=310,
            bg="#ffffff",
        ),
        ui.output_ui("kpi_row"),
        ui.div(style="height: 12px;"),
        ui.card(
            ui.card_header(ui.output_text("titulo_grafica")),
            ui.output_plot("grafica_modelo", height="390px"),
            full_screen=True,
        ),
        ui.div(style="height: 12px;"),
        ui.card(
            ui.card_header(ui.output_text("titulo_tabla")),
            ui.output_data_frame("tabla_datos"),
        ),
    ),
    ui.div(
        "Elaborado por Luis Antonio Dardón Alvarez y Ever Daniel Gomez Dominguez · Datos abiertos CONADESUCA (Infocaña)",
        class_="footer-credits",
    ),
    title="Dashboard Infocaña - Regresión Lineal",
)


def server(input, output, session):

    @reactive.calc
    def zafras_filtradas():
        a_min, a_max = input.rango_anios()
        df_sub = df_completas_all[
            (df_completas_all["anio_inicio"] >= a_min)
            & (df_completas_all["anio_inicio"] <= a_max)
        ].copy().reset_index(drop=True)
        # Reindexar periodo dentro del rango seleccionado para la regresión
        df_sub["periodo_rango"] = np.arange(1, len(df_sub) + 1)
        return df_sub

    @reactive.calc
    def ingenios_filtrados():
        a_min, a_max = input.rango_anios()
        return df_ing_comp_all[
            (df_ing_comp_all["anio_inicio"] >= a_min)
            & (df_ing_comp_all["anio_inicio"] <= a_max)
        ].copy()

    @reactive.calc
    def resultado_modelo():
        var_x = input.var_x()
        df_sub = zafras_filtradas()

        if var_x == "periodo":
            horizonte = int(input.horizonte())
            if len(df_sub) < 3:
                return {"error": "Selecciona al menos 3 zafras en el rango histórico para entrenar y evaluar el modelo (80/20)."}

            X = df_sub[["periodo_rango"]].values
            y = df_sub["azucar_producida_total"].values

            # División temporal 80% entrenamiento / 20% prueba
            corte = max(2, int(len(df_sub) * 0.8))
            if corte >= len(df_sub):
                corte = len(df_sub) - 1
            X_train, X_test = X[:corte], X[corte:]
            y_train, y_test = y[:corte], y[corte:]

            mod_eval = LinearRegression().fit(X_train, y_train)
            pred_test = mod_eval.predict(X_test)
            mae = mean_absolute_error(y_test, pred_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

            # Ajuste final sobre el rango seleccionado para proyectar
            mod_full = LinearRegression().fit(X, y)
            r2 = r2_score(y, mod_full.predict(X))

            ult_periodo = int(df_sub["periodo_rango"].iloc[-1])
            ult_anio = int(df_sub["anio_inicio"].iloc[-1])
            ult_zafra = df_sub["zafra"].iloc[-1]
            ult_valor = float(y[-1])

            fut_periodos = np.arange(ult_periodo + 1, ult_periodo + horizonte + 1)
            fut_preds = mod_full.predict(fut_periodos.reshape(-1, 1))
            fut_zafras = [f"{ult_anio + i}-{ult_anio + i + 1}" for i in range(1, horizonte + 1)]

            return {
                "error": None,
                "modo": "temporal",
                "df_sub": df_sub,
                "horizonte": horizonte,
                "ult_zafra": ult_zafra,
                "ult_valor": ult_valor,
                "fut_periodos": fut_periodos,
                "fut_zafras": fut_zafras,
                "fut_preds": fut_preds,
                "b0": float(mod_full.intercept_),
                "b1": float(mod_full.coef_[0]),
                "r2": float(r2),
                "mae": float(mae),
                "rmse": rmse,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "modelo": mod_full,
            }
        else:
            df_op = ingenios_filtrados()
            df_op = df_op[df_op[var_x] > 0].copy()
            if len(df_op) < 10:
                return {"error": "Selecciona un rango más amplio para entrenar el modelo operativo."}

            X = df_op[[var_x]].values
            y = df_op["azucar_producida_total"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )
            modelo = LinearRegression().fit(X_train, y_train)
            pred_test = modelo.predict(X_test)
            r2 = r2_score(y_test, pred_test)
            mae = mean_absolute_error(y_test, pred_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

            x_prom = float(df_op[var_x].mean())
            y_pred_prom = float(modelo.predict([[x_prom]])[0])

            return {
                "error": None,
                "modo": "operativo",
                "var_x": var_x,
                "X_train": X_train.ravel(),
                "y_train": y_train,
                "X_test": X_test.ravel(),
                "y_test": y_test,
                "b0": float(modelo.intercept_),
                "b1": float(modelo.coef_[0]),
                "r2": float(r2),
                "mae": float(mae),
                "rmse": rmse,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "x_prom": x_prom,
                "y_pred_prom": y_pred_prom,
                "modelo": modelo,
            }

    @render.ui
    def panel_modelo():
        res = resultado_modelo()
        if res.get("error"):
            return ui.div(res["error"], class_="warn-box")
        signo = "+" if res["b1"] >= 0 else "-"
        return ui.div(
            ui.strong("Modelo: Regresión Lineal (sklearn)"),
            ui.tags.br(),
            ui.span(f"Split: 80% train ({res['n_train']}) / 20% test ({res['n_test']})"),
            ui.tags.br(),
            ui.span(f"Y = {res['b0']:,.1f} {signo} {abs(res['b1']):,.2f}·X"),
            ui.tags.br(),
            ui.span(f"R² = {res['r2']:.4f} | MAE = {res['mae']:,.0f} t"),
            ui.tags.br(),
            ui.span(f"RMSE = {res['rmse']:,.0f} t"),
            class_="model-box",
        )

    @render.ui
    def kpi_row():
        res = resultado_modelo()
        if res.get("error"):
            return ui.div(res["error"], class_="warn-box")
        if res["modo"] == "temporal":
            h = res["horizonte"]
            etiqueta_h = f"{h} zafra" if h == 1 else f"{h} zafras"
            zafra_obj = res["fut_zafras"][-1]
            pron_final = float(res["fut_preds"][-1])
            return ui.layout_columns(
                ui.div(
                    ui.div("Horizonte seleccionado", class_="kpi-label"),
                    ui.div(etiqueta_h, class_="kpi-value"),
                    ui.div(f"Proyección hasta {zafra_obj}", class_="kpi-sub"),
                    class_="kpi-card kpi-blue",
                ),
                ui.div(
                    ui.div("Último valor real del rango", class_="kpi-label"),
                    ui.div(f"{res['ult_valor']:,.0f} t", class_="kpi-value"),
                    ui.div(f"Zafra {res['ult_zafra']} ({res['ult_valor']/1e6:.2f} M ton)", class_="kpi-sub"),
                    class_="kpi-card kpi-slate",
                ),
                ui.div(
                    ui.div(f"Pronóstico ({zafra_obj})", class_="kpi-label"),
                    ui.div(f"{pron_final:,.0f} t", class_="kpi-value"),
                    ui.div(f"Pendiente: {res['b1']:+,.0f} t/zafra ({pron_final/1e6:.2f} M ton)", class_="kpi-sub"),
                    class_="kpi-card kpi-green",
                ),
                col_widths=[4, 4, 4],
            )
        else:
            df_sub = zafras_filtradas()
            ult_z = df_sub["zafra"].iloc[-1] if len(df_sub) else "-"
            ult_v = float(df_sub["azucar_producida_total"].iloc[-1]) if len(df_sub) else 0.0
            return ui.layout_columns(
                ui.div(
                    ui.div("Coeficiente de determinación (R²)", class_="kpi-label"),
                    ui.div(f"{res['r2']:.4f}", class_="kpi-value"),
                    ui.div("Evaluado en conjunto de prueba (20%)", class_="kpi-sub"),
                    class_="kpi-card kpi-blue",
                ),
                ui.div(
                    ui.div("Último valor real del rango", class_="kpi-label"),
                    ui.div(f"{ult_v:,.0f} t", class_="kpi-value"),
                    ui.div(f"Zafra completa {ult_z}", class_="kpi-sub"),
                    class_="kpi-card kpi-slate",
                ),
                ui.div(
                    ui.div("Pronóstico ingenio promedio", class_="kpi-label"),
                    ui.div(f"{res['y_pred_prom']:,.0f} t", class_="kpi-value"),
                    ui.div(f"Con X promedio = {res['x_prom']:,.0f}", class_="kpi-sub"),
                    class_="kpi-card kpi-green",
                ),
                col_widths=[4, 4, 4],
            )

    @render.text
    def titulo_grafica():
        a_min, a_max = input.rango_anios()
        if input.var_x() == "periodo":
            return f"Serie histórica ({a_min}–{a_max + 1}) y pronóstico de producción total de azúcar"
        return f"Regresión lineal ({a_min}–{a_max + 1}): Azúcar producida total vs {VARIABLES_X[input.var_x()]}"

    @render.text
    def titulo_tabla():
        a_min, a_max = input.rango_anios()
        if input.tabla_vista() == "zafras":
            return f"Concentrado anual procesado (ETL Infocaña · Rango {a_min}–{a_max + 1})"
        return f"Detalle por ingenio y zafra (ETL Infocaña · Rango {a_min}–{a_max + 1})"

    @render.plot
    def grafica_modelo():
        res = resultado_modelo()
        fig, ax = plt.subplots(figsize=(9.5, 4.3), dpi=110)

        if res.get("error"):
            ax.text(0.5, 0.5, res["error"], ha="center", va="center", fontsize=11, color="#7c5400")
            ax.axis("off")
            return fig

        if res["modo"] == "temporal":
            df_sub = res["df_sub"]
            x_hist = df_sub["periodo_rango"].values
            y_hist = df_sub["azucar_producida_total"].values / 1e6
            etiquetas_hist = df_sub["zafra"].tolist()

            x_fut = res["fut_periodos"]
            y_fut = res["fut_preds"] / 1e6
            etiquetas_fut = res["fut_zafras"]

            y_tend = res["modelo"].predict(x_hist.reshape(-1, 1)) / 1e6
            ax.plot(x_hist, y_tend, linestyle=":", color="#8a9ba8", linewidth=1.3, label="Tendencia lineal")

            ax.plot(x_hist, y_hist, marker="o", color="#1f77b4", linewidth=2.2, markersize=6, label="Datos históricos (rango seleccionado)")

            ax.plot(
                np.concatenate([[x_hist[-1]], x_fut]),
                np.concatenate([[y_hist[-1]], y_fut]),
                marker="o",
                linestyle="--",
                color="#2ca02c",
                linewidth=2.2,
                markersize=6,
                label="Pronóstico (Regresión Lineal)",
            )

            ax.axvline(x=x_hist[-1], color="#7f8c8d", linestyle="--", linewidth=1.1)
            y_min_plot = min(y_hist.min(), y_fut.min())
            y_max_plot = max(y_hist.max(), y_fut.max())
            ax.annotate(
                f"Último dato: {y_hist[-1]:.2f} M t",
                xy=(x_hist[-1], y_hist[-1]),
                xytext=(max(x_hist[0], x_hist[-1] - 1.8), min(y_max_plot + 0.25, y_hist[-1] + 0.42)),
                fontsize=8.6,
                fontweight="bold",
                color="#163a63",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2fb", edgecolor="#9ec1e6"),
                arrowprops=dict(arrowstyle="->", color="#163a63", lw=1),
            )

            for xf, yf in zip(x_fut, y_fut):
                ax.annotate(
                    f"{yf:.2f} M",
                    xy=(xf, yf),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#1b5e32",
                )

            mostrar_p = input.mostrar_parcial() and len(df_parcial) > 0 and input.rango_anios()[1] == ANIO_MAX
            if mostrar_p:
                p_val = float(df_parcial["azucar_producida_total"].iloc[0]) / 1e6
                sem_p = int(df_parcial["semana_max"].iloc[0])
                ax.scatter(
                    [x_hist[-1] + 1],
                    [p_val],
                    marker="D",
                    s=55,
                    color="#d97706",
                    zorder=5,
                    label=f"Avance parcial 2025-2026 (sem. {sem_p}: {p_val:.2f} M t)",
                )

            todos_x = np.concatenate([x_hist, x_fut])
            todas_etq = etiquetas_hist + etiquetas_fut
            ax.set_xticks(todos_x)
            ax.set_xticklabels(todas_etq, rotation=28, ha="right", fontsize=8.5)
            ax.set_xlabel("Zafra", fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Azúcar producida (millones de toneladas)", fontsize=9.5, fontweight="bold")
            low_lim = 0 if mostrar_p else max(0, y_min_plot - 0.65)
            ax.set_ylim(low_lim, y_max_plot + 0.75)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="lower left", fontsize=8.2, frameon=True)
        else:
            ax.scatter(res["X_train"], res["y_train"], alpha=0.55, s=26, color="#1f77b4", label="Entrenamiento (80%)")
            ax.scatter(res["X_test"], res["y_test"], alpha=0.80, s=34, color="#e67e22", label="Prueba (20%)")

            x_all = np.concatenate([res["X_train"], res["X_test"]])
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            y_line = res["modelo"].predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color="#2ca02c", linewidth=2.2, label=f"Recta de regresión (R²={res['r2']:.4f})")

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
            ax.set_xlabel(VARIABLES_X[res["var_x"]], fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Azúcar producida total (toneladas)", fontsize=9.5, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="upper left", fontsize=8.5, frameon=True)

        fig.tight_layout()
        return fig

    @render.data_frame
    def tabla_datos():
        a_min, a_max = input.rango_anios()
        if input.tabla_vista() == "zafras":
            df_t = df_zafras[
                (df_zafras["anio_inicio"] >= a_min)
                & ((df_zafras["anio_inicio"] <= a_max) | (a_max == ANIO_MAX))
            ].copy()
            vista = pd.DataFrame({
                "Zafra": df_t["zafra"].str.replace("-", "\u2011", regex=False),
                "Estatus": df_t["estatus"],
                "Ingenios": df_t["ingenios_activos"],
                "Caña Bruta (t)": df_t["cana_molida_bruta"].map(lambda v: f"{v:,.0f}"),
                "Caña Neta (t)": df_t["cana_molida_neta"].map(lambda v: f"{v:,.0f}"),
                "Superficie (ha)": df_t["superficie_cosechada"].map(lambda v: f"{v:,.0f}"),
                "Azúcar Total (t)": df_t["azucar_producida_total"].map(lambda v: f"{v:,.0f}"),
                "Rend. (%)": df_t["rendimiento_promedio_ingenio"].map(lambda v: f"{v:.2f}%"),
            })
            return render.DataTable(vista, width="100%", height="370px", filters=False, summary=True)
        else:
            df_t = ingenios_filtrados().sort_values(["anio_inicio", "azucar_producida_total"], ascending=[False, False])
            vista = pd.DataFrame({
                "Zafra": df_t["zafra"].str.replace("-", "\u2011", regex=False),
                "Ingenio": df_t["ingenio"],
                "Sem.": df_t["semana_cierre"],
                "Caña Bruta (t)": df_t["cana_molida_bruta"].map(lambda v: f"{v:,.0f}"),
                "Caña Neta (t)": df_t["cana_molida_neta"].map(lambda v: f"{v:,.0f}"),
                "Superficie (ha)": df_t["superficie_cosechada"].map(lambda v: f"{v:,.0f}"),
                "Azúcar Total (t)": df_t["azucar_producida_total"].map(lambda v: f"{v:,.0f}"),
                "Rend. (%)": df_t["rendimiento_fabrica_calc"].map(lambda v: f"{v:.2f}%"),
            })
            return render.DataTable(vista, width="100%", height="370px", filters=False, summary=True)


app = App(app_ui, server)
