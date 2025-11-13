# pages/3_regresiones.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from utils.data_loader import load_all_data

st.title("Regresión")

# ===================== CARGA Y LIMPIEZA =====================
RAW = load_all_data()

def _to_city_map(obj):
    """Convierte la salida de load_all_data en {ciudad: DataFrame}."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # columna fantasma
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # normalizar price a numérico
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return df

raw_city_map = _to_city_map(RAW)
dfs_ciudades = {k: clean_city_df(v) for k, v in raw_city_map.items()}

# ===================== SIDEBAR =====================
ciudades_reg_sel = st.sidebar.multiselect(
    "Ciudades para regresión",
    options=list(dfs_ciudades.keys()),
    default=[k for k in list(dfs_ciudades.keys())[:2]],
    max_selections=4
)

tipo_reg = st.sidebar.radio(
    "Tipo de regresión",
    options=["Regresión lineal simple", "Regresión lineal múltiple", "Regresión no lineal"],
    index=0
)

if not ciudades_reg_sel:
    st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
    st.stop()

# ===================== REGRESIÓN LINEAL (simple / múltiple) =====================
if tipo_reg in ("Regresión lineal simple", "Regresión lineal múltiple"):
    n = len(ciudades_reg_sel)
    cols = st.columns(min(3, n))

    for i, ciudad in enumerate(ciudades_reg_sel):
        if i > 0 and i % 3 == 0:
            cols = st.columns(min(3, n - i))

        df_ciudad = dfs_ciudades[ciudad]
        with cols[i % 3]:
            st.subheader(ciudad)

            if "price" not in df_ciudad.columns:
                st.info("Falta columna 'price'.")
                continue

            # ---------- Lineal simple: price ~ accommodates ----------
            if tipo_reg == "Regresión lineal simple":
                if "accommodates" not in df_ciudad.columns:
                    st.info("Falta columna 'accommodates'.")
                    continue

                tmp = df_ciudad[["accommodates", "price"]].astype(float).dropna()
                if len(tmp) < 3:
                    st.info("Datos insuficientes para ajustar el modelo.")
                    continue

                x = tmp["accommodates"].to_numpy()
                y = tmp["price"].to_numpy()
                a, b = np.polyfit(x, y, 1)  # β1, β0
                y_pred = a * x + b
                r2 = r2_score(y, y_pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{r2:.3f}")
                m2.metric("Pendiente (β1)", f"{a:.3f}")
                m3.metric("Intercepto (β0)", f"{b:.2f}")

                fig = px.scatter(
                    tmp, x="accommodates", y="price",
                    labels={"accommodates": "Accommodates", "price": "Price"},
                    title="Price ~ Accommodates (Regresión lineal)"
                )
                x_line = np.linspace(x.min(), x.max(), 50)
                fig.add_trace(go.Scatter(x=x_line, y=a * x_line + b, mode="lines", name="Ajuste"))
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # ---------- Lineal múltiple ----------
            elif tipo_reg == "Regresión lineal múltiple":
                num_cols = df_ciudad.select_dtypes(include="number").columns.tolist()
                num_cols = [c for c in num_cols if not c.lower().startswith("unnamed")]
                candidatas = [c for c in num_cols if c != "price"]

                if len(candidatas) < 2:
                    st.info("Se requieren al menos 2 variables numéricas distintas a 'price' en el dataset.")
                    continue

                default_preds = []
                if "accommodates" in candidatas:
                    default_preds.append("accommodates")
                for c in candidatas:
                    if c not in default_preds:
                        default_preds.append(c)
                    if len(default_preds) >= 2:
                        break

                predictores = st.multiselect(
                    "Variables explicativas (elige 2 o más)",
                    options=candidatas,
                    default=default_preds,
                    key=f"preds_mult_{ciudad}"
                )

                if len(predictores) < 2:
                    st.warning("Selecciona al menos 2 variables para ajustar la regresión múltiple.")
                    continue

                tmp = df_ciudad[predictores + ["price"]].dropna()
                if len(tmp) < len(predictores) + 1:
                    st.info("Datos insuficientes después de eliminar NA para ajustar el modelo.")
                    continue

                X = tmp[predictores].to_numpy(dtype=float)
                y = tmp["price"].to_numpy(dtype=float)

                modelo = LinearRegression()
                modelo.fit(X, y)

                r2 = modelo.score(X, y)
                intercepto = modelo.intercept_
                coefs = modelo.coef_

                m1, m2, m3 = st.columns(3)
                m1.metric("R² (múltiple)", f"{r2:.3f}")
                m2.metric("N° predictores", f"{len(predictores)}")
                m3.metric("Intercepto (β0)", f"{intercepto:.2f}")

                coef_df = pd.DataFrame({
                    "Variable": predictores,
                    "Coeficiente (β)": coefs
                })
                st.dataframe(coef_df, use_container_width=True)

                # Efecto parcial de una variable manteniendo las demás en su media
                var_plot = "accommodates" if "accommodates" in predictores else predictores[0]

                fig = px.scatter(
                    tmp[[var_plot, "price"]], x=var_plot, y="price",
                    labels={var_plot: var_plot, "price": "Price"},
                    title=f"Price ~ {var_plot} (Regresión múltiple, otros = media)"
                )

                x_line = np.linspace(tmp[var_plot].min(), tmp[var_plot].max(), 50)
                medias = tmp[predictores].mean()
                X_line = np.tile(medias.to_numpy(), (50, 1))
                idx_plot = predictores.index(var_plot)
                X_line[:, idx_plot] = x_line
                y_line = modelo.predict(X_line)

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Predicción (otros = media)"
                ))
                st.plotly_chart(fig, use_container_width=True)


# ===================== REGRESIÓN NO LINEAL (sólo si se elige) =====================
if tipo_reg == "Regresión no lineal":
    st.markdown("## Regresión no lineal")

    # ---- Catálogo de funciones ----
    def f_quadratic(x, a, b, c):          return a*x**2 + b*x + c
    def f_exp_decay(x, a, b, c):          return a*np.exp(-b*x) + c
    def f_inverse(x, a):                  return (1.0/np.where(a==0, 1e-9, a))*x
    def f_sine(x, a, b):                  return a*np.sin(x) + b
    def f_tangent(x, a, b):               return a*np.tan(x) + b
    def f_abs_linear(x, a, b, c):         return a*np.abs(x) + b*x + c
    def f_rational_poly2(x, a, b, c):
        denom = c*(x**2)
        return (a*x**2 + b) / np.where(denom == 0, 1e-9, denom)
    def f_log(x, a, b):                   return a*np.log(x) + b
    def f_linear_combo(x, a, b, c):       return (a + b + c) * x
    def f_inv_quad(x, a):                 return (1.0/np.where(a==0, 1e-9, a)) * (x**2)
    def f_inv_poly2(x, a, b, c):          return (a/np.where(b==0, 1e-9, b)) * (x**2) + c*x

    MODELS = {
        "Función cuadrática (a*x^2 + b*x + c)": (f_quadratic,  [1.0, 1.0, 0.0]),
        "Función exponencial (a*exp(-b*x)+c)":  (f_exp_decay,  [1.0, 0.1, 0.0]),
        "Función inversa ((1/a)*x)":            (f_inverse,    [1.0]),
        "Función senoidal (a*sin(x_s)+b)":      (f_sine,       [1.0, 0.0]),
        "Función tangencial (a*tan(x_s)+b)":    (f_tangent,    [0.5, 0.0]),
        "Valor absoluto (a*|x| + b*x + c)":     (f_abs_linear, [1.0, 0.0, 0.0]),
        "Cociente polinomios ((a*x^2+b)/(c*x^2))": (f_rational_poly2, [1.0, 1.0, 1.0]),
        "Logarítmica (a*log(x)+b)":             (f_log,        [1.0, 0.0]),
        "Lineal (a+b+c)*x":                     (f_linear_combo,[1.0, 1.0, 1.0]),
        "Cuadrática inversa ((1/a)*x^2)":       (f_inv_quad,   [1.0]),
        "Polinomial inversa ((a/b)*x^2 + c*x)": (f_inv_poly2,  [1.0, 1.0, 0.0]),
    }

    # ---- Controles ----
    df_ref_nl = dfs_ciudades[ciudades_reg_sel[0]]
    numeric_cols = df_ref_nl.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.lower().startswith("unnamed")]

    x_col = st.selectbox(
        "Variable X (predictor)",
        options=numeric_cols,
        index=numeric_cols.index("accommodates") if "accommodates" in numeric_cols else 0
    )
    y_col = st.selectbox(
        "Variable Y (objetivo)",
        options=numeric_cols,
        index=numeric_cols.index("price") if "price" in numeric_cols else 0
    )

    model_name = st.selectbox("Selecciona el modelo no lineal", options=list(MODELS.keys()))
    func, p0_default = MODELS[model_name]
    st.caption("Nota: para senoidal/tangencial se usa X escalado; para log/inversa se filtran X no válidos.")

    # ---- Visual por ciudad ----
    n = len(ciudades_reg_sel)
    cols_nl = st.columns(min(2, n))

    for i, ciudad in enumerate(ciudades_reg_sel):
        if i > 0 and i % 2 == 0:
            cols_nl = st.columns(min(2, n - i))
        with cols_nl[i % 2]:
            st.subheader(ciudad)

            dfc = dfs_ciudades[ciudad].copy()
            if x_col not in dfc.columns or y_col not in dfc.columns:
                st.info("Columnas seleccionadas no disponibles.")
                continue

            dfc = dfc[[x_col, y_col]].dropna()
            if dfc.empty:
                st.info("Sin datos para ajustar después de eliminar NA.")
                continue

            x_raw = dfc[x_col].astype(float).to_numpy()
            y = dfc[y_col].astype(float).to_numpy()

            mask = np.isfinite(x_raw) & np.isfinite(y)
            if "Logarítmica" in model_name:
                mask &= x_raw > 0
            if "inversa" in model_name or "Cociente" in model_name:
                mask &= x_raw != 0
            x_raw, y = x_raw[mask], y[mask]
            if len(x_raw) < 5:
                st.info("Datos insuficientes tras filtros de dominio.")
                continue

            use_scaled = ("senoidal" in model_name.lower()) or ("tangencial" in model_name.lower())
            if use_scaled:
                mu, sd = x_raw.mean(), x_raw.std() if x_raw.std() > 0 else 1.0
                x = (x_raw - mu) / sd
            else:
                mu, sd = 0.0, 1.0
                x = x_raw.copy()

            try:
                popt, _ = curve_fit(func, x, y, p0=p0_default, maxfev=10000)
                y_hat = func(x, *popt)
                r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
            except Exception as e:
                st.warning(f"No se pudo ajustar el modelo: {e}")
                continue

            m1, m2 = st.columns(2)
            m1.metric("R²", f"{r2:.3f}")
            m2.write("Parámetros:")
            m2.json({f"θ{i}": float(v) for i, v in enumerate(popt, start=1)})

            order = np.argsort(x_raw)
            x_plot_raw = x_raw[order]
            x_plot = ((x_plot_raw - mu)/sd) if use_scaled else x_plot_raw
            y_plot = func(x_plot, *popt)

            fig = px.scatter(
                x=x_raw, y=y,
                labels={"x": x_col, "y": y_col},
                title=f"{y_col} ~ {x_col} — {model_name}"
            )
            fig.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode="lines", name="Ajuste"))
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Diagnóstico de residuos", expanded=False):
                resid = y - func(x, *popt)
                c1, c2 = st.columns(2)
                with c1:
                    fr = px.scatter(
                        x=y_hat, y=resid,
                        labels={"x": "Predicción", "y": "Residuo"},
                        title="Residuos vs Predicción"
                    )
                    st.plotly_chart(fr, use_container_width=True)
                with c2:
                    fh = px.histogram(resid, nbins=30, title="Distribución de residuos")
                    st.plotly_chart(fh, use_container_width=True)


# ===================== HALLAZGOS GLOBALES (siempre visibles) =====================
def generar_hallazgos(ciudades):
    lines = []
    for c in ciudades:
        df = dfs_ciudades[c]
        if len(df) < 10:
            continue
        med_price = df["price"].median() if "price" in df.columns else np.nan
        rating = df["review_scores_rating"].mean() if "review_scores_rating" in df.columns else np.nan
        occ = df["estimated_occupancy_l365d"].mean() if "estimated_occupancy_l365d" in df.columns else np.nan
        reviews = df["number_of_reviews"].median() if "number_of_reviews" in df.columns else np.nan
        txt = f"{c} — Mediana precio: {med_price:,.0f} USD"
        if not np.isnan(rating):  txt += f", Rating prom.: {rating:.1f}"
        if not np.isnan(reviews): txt += f", Mediana #reviews: {reviews:,.0f}"
        if not np.isnan(occ):     txt += f", Ocupación est.: {occ:.1f}%"
        lines.append(txt)
    return lines

hallazgos = generar_hallazgos(ciudades_reg_sel)
if hallazgos:
    st.markdown("### 💡 Hallazgos principales")
    st.markdown("""
    <style>
    .hallazgo-box {
        background-color: #f7f7f8;
        border-left: 5px solid #ff5a5f;
        padding: 0.6em 1em;
        margin: 0.3em 0;
        border-radius: 0.4em;
    }
    </style>
    """, unsafe_allow_html=True)
    for h in hallazgos:
        st.markdown(f'<div class="hallazgo-box">{h}</div>', unsafe_allow_html=True)