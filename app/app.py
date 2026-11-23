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

df_completas = df_zafras[df_zafras["estatus"] == "Completa"].copy().reset_index(drop=True)
df_parcial = df_zafras[df_zafras["estatus"] != "Completa"].copy().reset_index(drop=True)
zafras_completas_set = set(df_completas["zafra"])
df_ing_comp = df_ingenios[df_ingenios["zafra"].isin(zafras_completas_set)].copy()

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
    padding: 14px 22px;
    border-radius: 6px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-banner h2 { margin: 0; font-size: 1.35rem; font-weight: 600; }
.top-banner span { font-size: 0.88rem; opacity: 0.9; }
.kpi-card {
    border-radius: 8px;
    padding: 14px 16px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-blue { background-color: #eaf2fb; color: #163a63; }
.kpi-slate { background-color: #eef2f6; color: #1f2d3d; }
.kpi-green { background-color: #e8f5ec; color: #1b5e32; }
.kpi-label { font-size: 0.82rem; font-weight: 600; opacity: 0.85; margin-bottom: 4px; }
.kpi-value { font-size: 1.45rem; font-weight: 700; line-height: 1.2; }
.kpi-sub { font-size: 0.78rem; opacity: 0.8; margin-top: 3px; }
.model-box {
    background-color: #eef7f0;
    border: 1px solid #c6e3cc;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.85rem;
    color: #1d4d2b;
    margin-top: 10px;
}
"""

app_ui = ui.page_fluid(
    ui.tags.style(CSS_CUSTOM),
    ui.div(
        ui.h2("Dashboard - Pronóstico de Producción de Azúcar (Infocaña)"),
        ui.span("CONADESUCA 2015-2026 | Regresión Lineal (scikit-learn)"),
        class_="top-banner",
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Configuración del pronóstico", style="color:#1b3a63; font-weight:700;"),
            ui.input_select(
                "var_x",
                "Variable independiente (X):",
                choices=VARIABLES_X,
                selected="periodo",
            ),
            ui.panel_conditional(
                "input.var_x === 'periodo'",
                ui.input_radio_buttons(
                    "horizonte",
                    "Horizonte de pronóstico:",
                    choices={"1": "1 zafra (2025-2026)", "2": "2 zafras (2026-2027)", "3": "3 zafras (2027-2028)"},
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
                "Dataset en tabla inferior:",
                choices={
                    "zafras": "Concentrado anual por zafra (2015-2026)",
                    "ingenios": "Detalle por ingenio y zafra",
                },
                selected="zafras",
            ),
            ui.hr(),
            ui.output_ui("panel_modelo"),
            width=300,
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
            ui.card_header("Conjunto de datos procesado (ETL Infocaña 2015-2026)"),
            ui.output_data_frame("tabla_datos"),
        ),
    ),
    title="Dashboard Infocaña - Regresión Lineal",
)


def server(input, output, session):

    @reactive.calc
    def resultado_modelo():
        var_x = input.var_x()

        if var_x == "periodo":
            horizonte = int(input.horizonte())
            X = df_completas[["periodo"]].values
            y = df_completas["azucar_producida_total"].values

            # División temporal 80% entrenamiento / 20% prueba
            corte = int(len(df_completas) * 0.8)
            X_train, X_test = X[:corte], X[corte:]
            y_train, y_test = y[:corte], y[corte:]

            mod_eval = LinearRegression().fit(X_train, y_train)
            pred_test = mod_eval.predict(X_test)
            mae = mean_absolute_error(y_test, pred_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

            # Ajuste completo sobre las 10 zafras completas para proyectar
            mod_full = LinearRegression().fit(X, y)
            r2 = r2_score(y, mod_full.predict(X))

            ult_periodo = int(df_completas["periodo"].iloc[-1])
            ult_anio = int(df_completas["anio_inicio"].iloc[-1])
            ult_zafra = df_completas["zafra"].iloc[-1]
            ult_valor = float(y[-1])

            fut_periodos = np.arange(ult_periodo + 1, ult_periodo + horizonte + 1)
            fut_preds = mod_full.predict(fut_periodos.reshape(-1, 1))
            fut_zafras = [f"{ult_anio + i}-{ult_anio + i + 1}" for i in range(1, horizonte + 1)]

            return {
                "modo": "temporal",
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
            df_op = df_ing_comp[df_ing_comp[var_x] > 0].copy()
            X = df_op[[var_x]].values
            y = df_op["azucar_producida_total"].values

            # División 80% entrenamiento / 20% prueba
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
        if res["modo"] == "temporal":
            h = res["horizonte"]
            etiqueta_h = f"{h} zafra" if h == 1 else f"{h} zafras"
            zafra_obj = res["fut_zafras"][-1]
            pron_final = float(res["fut_preds"][-1])
            return ui.layout_columns(
                ui.div(
                    ui.div("Horizonte seleccionado", class_="kpi-label"),
                    ui.div(etiqueta_h, class_="kpi-value"),
                    ui.div(f"Hasta zafra {zafra_obj}", class_="kpi-sub"),
                    class_="kpi-card kpi-blue",
                ),
                ui.div(
                    ui.div("Último valor real (zafra completa)", class_="kpi-label"),
                    ui.div(f"{res['ult_valor']:,.0f} t", class_="kpi-value"),
                    ui.div(f"Zafra {res['ult_zafra']} ({res['ult_valor']/1e6:.2f} M ton)", class_="kpi-sub"),
                    class_="kpi-card kpi-slate",
                ),
                ui.div(
                    ui.div(f"Pronóstico ({zafra_obj})", class_="kpi-label"),
                    ui.div(f"{pron_final:,.0f} t", class_="kpi-value"),
                    ui.div(f"Tendencia: {res['b1']:+,.0f} t/zafra ({pron_final/1e6:.2f} M ton)", class_="kpi-sub"),
                    class_="kpi-card kpi-green",
                ),
                col_widths=[4, 4, 4],
            )
        else:
            ult_z = df_completas["zafra"].iloc[-1]
            ult_v = float(df_completas["azucar_producida_total"].iloc[-1])
            return ui.layout_columns(
                ui.div(
                    ui.div("Coeficiente de determinación (R²)", class_="kpi-label"),
                    ui.div(f"{res['r2']:.4f}", class_="kpi-value"),
                    ui.div("Evaluado en conjunto de prueba (20%)", class_="kpi-sub"),
                    class_="kpi-card kpi-blue",
                ),
                ui.div(
                    ui.div("Último valor real nacional", class_="kpi-label"),
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
        if input.var_x() == "periodo":
            return "Serie histórica y pronóstico de producción total de azúcar (toneladas)"
        return f"Regresión lineal: Azúcar producida total vs {VARIABLES_X[input.var_x()]}"

    @render.plot
    def grafica_modelo():
        res = resultado_modelo()
        fig, ax = plt.subplots(figsize=(9.5, 4.3), dpi=110)

        if res["modo"] == "temporal":
            x_hist = df_completas["periodo"].values
            y_hist = df_completas["azucar_producida_total"].values / 1e6
            etiquetas_hist = df_completas["zafra"].tolist()

            x_fut = res["fut_periodos"]
            y_fut = res["fut_preds"] / 1e6
            etiquetas_fut = res["fut_zafras"]

            # Línea de tendencia histórica
            y_tend = res["modelo"].predict(x_hist.reshape(-1, 1)) / 1e6
            ax.plot(x_hist, y_tend, linestyle=":", color="#8a9ba8", linewidth=1.3, label="Tendencia lineal")

            # Datos históricos
            ax.plot(x_hist, y_hist, marker="o", color="#1f77b4", linewidth=2.2, markersize=6, label="Datos históricos (zafras completas)")

            # Conectar último punto real con pronóstico
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

            # Línea vertical en último dato real
            ax.axvline(x=x_hist[-1], color="#7f8c8d", linestyle="--", linewidth=1.1)
            ax.annotate(
                f"Último dato: {y_hist[-1]:.2f} M t",
                xy=(x_hist[-1], y_hist[-1]),
                xytext=(x_hist[-1] - 2.3, y_hist[-1] + 0.45),
                fontsize=8.8,
                fontweight="bold",
                color="#163a63",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2fb", edgecolor="#9ec1e6"),
                arrowprops=dict(arrowstyle="->", color="#163a63", lw=1),
            )

            # Etiquetas de valores pronosticados
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

            # Avance parcial 2025-2026 opcional
            if input.mostrar_parcial() and len(df_parcial) > 0:
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
            ax.set_xticklabels(todas_etq, rotation=30, ha="right", fontsize=8.5)
            ax.set_xlabel("Zafra", fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Azúcar producida (millones de toneladas)", fontsize=9.5, fontweight="bold")
            ax.set_ylim(0 if (input.mostrar_parcial() and len(df_parcial) > 0) else 3.8, 6.9)
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
        if input.tabla_vista() == "zafras":
            cols = [
                "periodo", "zafra", "estatus", "ingenios_activos",
                "cana_molida_bruta", "cana_molida_neta",
                "superficie_cosechada", "azucar_producida_total",
                "rendimiento_promedio_ingenio",
            ]
            return render.DataTable(df_zafras[cols], filters=True, height="320px")
        else:
            cols = [
                "zafra", "ingenio", "semana_cierre",
                "cana_molida_bruta", "cana_molida_neta",
                "superficie_cosechada", "azucar_producida_total",
                "rendimiento_fabrica_calc",
            ]
            return render.DataTable(df_ingenios[cols], filters=True, height="320px")


app = App(app_ui, server)
