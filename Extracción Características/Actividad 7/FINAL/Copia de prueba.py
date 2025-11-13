# Creamos el archivo de la APP en el interprete principal (Python)
### se corre con: streamlit run prueba.py
#####################################################
# Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
######################################################

# ===================== Configuración de página (primera llamada a st.*) =====================
st.set_page_config(
    page_title="Dashboard Airbnb",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== Datos (cache) =====================
@st.cache_resource
def load_data():
    base = "/Users/hijos/Desktop/Analitica Prof/Prof. Alfredo/FINAL/data"
    dfb  = pd.read_csv(f"{base}/Barcelona_Limpios.csv")
    dfc  = pd.read_csv(f"{base}/Cambridge_Limpios.csv")
    dfbo = pd.read_csv(f"{base}/Boston_Limpios.csv")
    dfh  = pd.read_csv(f"{base}/Hawai_Limpios.csv")
    dfbu = pd.read_csv(f"{base}/Budapest_Limpios.csv")
    return {"Barcelona": dfb, "Cambridge": dfc, "Boston": dfbo, "Hawái": dfh, "Budapest": dfbu}


def clean_data(dfb, dfc, dfbo, dfh, dfbu):
    # Limpieza homogénea de price
    for df in (dfb, dfc, dfbo, dfh, dfbu):
        if "price" in df.columns:
            df["price"] = (
                df["price"].astype(str)
                .str.replace(r"[^\d\.\-]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )
    return dfb, dfc, dfbo, dfh, dfbu


# ===================== Estilos =====================
st.markdown("""
<style>
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] > div:first-child { background: #734042; }
</style>
""", unsafe_allow_html=True)

# ===================== Sidebar =====================
logo_sidebar = 'airbnb.png'
col1, col2, col3 = st.sidebar.columns([1, 3, 1])
with col2:
    st.image(logo_sidebar)
    st.write("---")
st.sidebar.title("Análisis de Datos Airbnb")

# Widget 1: Selectbox (vista)
View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=["Extracción de Características", "Tablas comparativas", "Regresiones"],
)

# Widget 2: Checkbox (ver datos)
show_data = st.sidebar.checkbox(label="Mostrar Datos")

# ===================== Carga y mapeo de datos =====================
dfb, dfc, dfbo, dfh, dfbu = load_data()
dfb, dfc, dfbo, dfh, dfbu = clean_data(dfb, dfc, dfbo, dfh, dfbu)

dfs_ciudades = {
    "Barcelona": dfb,
    "Cambridge": dfc,
    "Boston": dfbo,
    "Hawái": dfh,
    "Budapest": dfbu,
}

# ===================== Mostrar datos (opcional) =====================
if show_data:
    st.subheader("Datos de Airbnb en Barcelona")
    st.dataframe(dfb.head(10))
    st.subheader("Datos de Airbnb en Cambridge")
    st.dataframe(dfc.head(10))
    st.subheader("Datos de Airbnb en Boston")
    st.dataframe(dfbo.head(10))
    st.subheader("Datos de Airbnb en Hawái")
    st.dataframe(dfh.head(10))
    st.subheader("Datos de Airbnb en Budapest")
    st.dataframe(dfbu.head(10))

# ===================== Filtros de ciudades =====================
ciudades_multiselect = st.sidebar.multiselect(
    label="Selecciona las Ciudades (Extracción)",
    options=list(dfs_ciudades.keys()),
    default=["Barcelona", "Cambridge"],
    max_selections=4
)

ciudad_regresion = st.sidebar.radio(
    label="Ciudad para regresión (default)",
    options=list(dfs_ciudades.keys()),
    index=0,
)

ciudades_reg_sel = st.sidebar.multiselect(
    label="Ciudades para regresión (comparar hasta 4)",
    options=list(dfs_ciudades.keys()),
    default=[ciudad_regresion],
    max_selections=4
)

# ===================== Helper de hallazgos =====================
def generar_hallazgos(ciudades):
    lines = []
    resumen = []
    for c in ciudades:
        df = dfs_ciudades[c]
        n = len(df)
        med_price = (df["price"].astype(float).median()
                     if "price" in df.columns else np.nan)
        corr_ap = (df[["accommodates", "price"]].corr().iloc[0, 1]
                   if set(["accommodates", "price"]).issubset(df.columns) else np.nan)
        resumen.append({"c": c, "n": n, "med_price": med_price, "corr": corr_ap})

    # mediana de price
    val = [r for r in resumen if not np.isnan(r["med_price"])]
    if val:
        top = max(val, key=lambda r: r["med_price"])
        low = min(val, key=lambda r: r["med_price"])
        if top["c"] != low["c"]:
            lines.append(
                f"• **{top['c']}** tiene la **mediana de precio** más alta (≈ {top['med_price']:.0f}); "
                f"**{low['c']}** la más baja (≈ {low['med_price']:.0f})."
            )

    # correlación accommodates–price
    valc = [r for r in resumen if not np.isnan(r["corr"])]
    if valc:
        strong = max(valc, key=lambda r: abs(r["corr"]))
        signo = "positiva" if strong["corr"] >= 0 else "negativa"
        lines.append(
            f"• La relación **accommodates–price** más marcada está en **{strong['c']}** "
            f"({signo}, r≈{strong['corr']:.2f})."
        )

    # tamaño de muestra
    topn = max(resumen, key=lambda r: r["n"])
    lines.append(f"• **{topn['c']}** cuenta con el **mayor número de anuncios** (n={topn['n']}).")
    return lines


###############################################################################
# 1) EXTRACCIÓN DE CARACTERÍSTICAS
###############################################################################
if View == "Extracción de Características":
    st.title("Extracción de Características")
    st.write("Análisis de características clave en los datos de Airbnb.")

    if not ciudades_multiselect:
        st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
    else:
        # ===================== CATEGORÍAS (selector único + Top-10) =====================
        st.markdown("## 🔎 Análisis global de variables categóricas")

        # Tomamos las columnas categóricas del primer df seleccionado
        primera_ciudad = ciudades_multiselect[0]
        df_referencia = dfs_ciudades[primera_ciudad]
        cat_cols = df_referencia.select_dtypes(include=["object", "category"]).columns.tolist()

        if not cat_cols:
            st.info(f"No hay variables categóricas en {primera_ciudad}.")
        else:
            # Selector único global
            cat_var = st.selectbox(
                "Variable categórica (todas las ciudades)",
                options=cat_cols,
                key="cat_global"
            )

            # Bucle para graficar cada ciudad, usando el mismo cat_var
            for ciudad in ciudades_multiselect:
                df_ciudad = dfs_ciudades[ciudad]

                if cat_var not in df_ciudad.columns:
                    st.warning(f"**{ciudad}** no tiene la columna **{cat_var}**.")
                    continue

                # Top-10 categorías por frecuencia
                serie = df_ciudad[cat_var].astype("string").fillna("NA")
                top10 = serie.value_counts().head(10).index
                df_cat = df_ciudad[serie.isin(top10)].copy()
                df_cat[cat_var] = df_cat[cat_var].astype("string").fillna("NA")

                st.subheader(f"{ciudad} — {cat_var} (Top-10)")
                col_a, col_b = st.columns(2)

                # 1) Frecuencia
                with col_a:
                    counts = df_cat[cat_var].value_counts().reset_index()
                    counts.columns = [cat_var, "count"]
                    fig1 = px.bar(
                        counts, x="count", y=cat_var, orientation="h",
                        title=f"Frecuencia de {cat_var} ({ciudad})"
                    )
                    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig1, use_container_width=True)

                # 2) Media de price (o primer numérico) por categoría
                with col_b:
                    num_cols_city = df_ciudad.select_dtypes(include="number").columns.tolist()
                    target = "price" if "price" in num_cols_city else (num_cols_city[0] if num_cols_city else None)
                    if target:
                        agg = (
                            df_cat.groupby(cat_var, dropna=False)[target]
                            .mean().reset_index().rename(columns={target: f"mean_{target}"})
                        )
                        agg = agg.sort_values(f"mean_{target}", ascending=False)
                        fig2 = px.bar(
                            agg, x=f"mean_{target}", y=cat_var, orientation="h",
                            title=f"Media de {target} por {cat_var} ({ciudad})"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No hay columnas numéricas para calcular medias.")

        st.markdown("---")
        st.markdown("## 📊 Análisis numérico por ciudad")

        # ===================== NUMÉRICO POR CIUDAD (se mantiene) =====================
        n = len(ciudades_multiselect)
        cols = st.columns(min(4, n))

        for i, ciudad in enumerate(ciudades_multiselect):
            if i > 0 and i % 4 == 0:
                cols = st.columns(min(4, n - i))

            df_ciudad = dfs_ciudades[ciudad]
            num_cols = df_ciudad.select_dtypes(include="number")

            with cols[i % 4]:
                st.subheader(ciudad)

                # Variables numéricas seleccionables
                if not num_cols.empty:
                    opciones = list(num_cols.columns)
                    default_vars = [c for c in opciones if c != "price"][:2] or [opciones[0]]
                    vars_sel = st.multiselect(
                        f"Variables numéricas para graficar ({ciudad})",
                        options=opciones,
                        default=default_vars,
                        key=f"vars_{ciudad}",
                    )

                    # 1. Histograma por variable
                    for v in vars_sel:
                        fig_hist = px.histogram(
                            df_ciudad.dropna(subset=[v]),
                            x=v,
                            nbins=30,
                            title=f"Distribución de {v} en {ciudad}",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # 2. Scatter vs price si existe
                        if "price" in df_ciudad.columns and v != "price":
                            tmp = df_ciudad[[v, "price"]].dropna()
                            if not tmp.empty:
                                fig_scatter = px.scatter(
                                    tmp, x=v, y="price", trendline="ols",
                                    title=f"{v} vs Price en {ciudad}",
                                    labels={v: v, "price": "Price"},
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)

                    # 3. Boxplot de price (si existe)
                    if "price" in df_ciudad.columns:
                        tmp_price = df_ciudad[["price"]].dropna()
                        if not tmp_price.empty:
                            fig_box = px.box(tmp_price, y="price", title=f"Distribución de Price en {ciudad}")
                            st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("No se detectaron columnas numéricas en esta ciudad.")

        # ---- Hallazgos ----
        st.markdown("### Hallazgos")
        for l in generar_hallazgos(ciudades_multiselect):
            st.markdown(l)


###############################################################################
# 2) TABLAS COMPARATIVAS
###############################################################################
elif View == "Tablas comparativas":
    st.title("Tablas Comparativas")
    st.write("Una tabla por ciudad; se muestran en filas de hasta 4.")

    ciudades_sel = st.multiselect(
        "Selecciona las ciudades a comparar",
        options=list(dfs_ciudades.keys()),
        default=["Barcelona", "Cambridge"],
        key="cmp_ciudades",
        max_selections=4
    )

    if not ciudades_sel:
        st.warning("Selecciona al menos una ciudad para comparar.")
    else:
        n = len(ciudades_sel)
        cols = st.columns(min(4, n))

        for i, ciudad in enumerate(ciudades_sel):
            if i > 0 and i % 4 == 0:
                cols = st.columns(min(4, n - i))

            df_ciudad = dfs_ciudades[ciudad]
            num_cols = df_ciudad.select_dtypes(include="number")

            with cols[i % 4]:
                st.subheader(ciudad)
                if num_cols.empty:
                    st.info("Sin columnas numéricas.")
                else:
                    tabla = num_cols.agg(["mean", "median", "std"]).T.rename(
                        columns={"mean": "Media", "median": "Mediana", "std": "DesvEst"}
                    )
                    st.dataframe(tabla, use_container_width=True)

        # ---- Hallazgos ----
        st.markdown("### Hallazgos")
        for l in generar_hallazgos(ciudades_sel):
            st.markdown(l)


###############################################################################
# 3) REGRESIONES (Lineal y Múltiple)
###############################################################################
elif View == "Regresiones":
    st.title("Regresión Lineal y Multiple")
    st.write("Comparación de modelos por ciudad (hasta 4 simultáneas).")

    # Guardamos la elección del usuario
    tipo_reg = st.selectbox(
        'Escoje el tipo de regresion',
        options=['Regresion Lineal', 'Regresion Multiple']
    )
    
    if not ciudades_reg_sel:
        st.warning("Selecciona al menos una ciudad para analizar.")
    else:
        n = len(ciudades_reg_sel)
        cols = st.columns(min(4, n))

        for i, ciudad in enumerate(ciudades_reg_sel):
            if i > 0 and i % 4 == 0:
                cols = st.columns(min(4, n - i))

            df_ciudad = dfs_ciudades[ciudad]

            with cols[i % 4]:
                st.subheader(ciudad)

                # Validación básica
                if "price" not in df_ciudad.columns:
                    st.info("Falta columna 'price'.")
                    continue
                if "accommodates" not in df_ciudad.columns:
                    st.info("Falta columna 'accommodates'.")
                    continue

                # ===================== REGRESIÓN LINEAL SIMPLE =====================
                if tipo_reg == "Regresion Lineal":
                    tmp = df_ciudad[["accommodates", "price"]].astype(float).dropna()
                    if len(tmp) < 3:
                        st.info("Datos insuficientes para ajustar el modelo.")
                        continue

                    x = tmp["accommodates"].to_numpy()
                    y = tmp["price"].to_numpy()

                    # Ajuste lineal simple
                    a, b = np.polyfit(x, y, 1)
                    y_pred = a * x + b

                    # Métrica R² (in-sample)
                    r2 = r2_score(y, y_pred)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("R²", f"{r2:.3f}")
                    m2.metric("Pendiente (β1)", f"{a:.3f}")
                    m3.metric("Intersección (β0)", f"{b:.2f}")

                    # Gráfica con recta
                    fig = px.scatter(
                        tmp, x="accommodates", y="price",
                        labels={"accommodates": "Accommodates", "price": "Price"},
                        title="Price ~ Accommodates (Regresión lineal)"
                    )
                    x_line = np.linspace(x.min(), x.max(), 50)
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=a * x_line + b,
                        mode="lines",
                        name="Predicción"
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                # ===================== REGRESIÓN MÚLTIPLE =====================
                else:  # "Regresion Multiple"
                    # Tomamos todas las numéricas como posibles predictores (excepto price)
                    num_cols = df_ciudad.select_dtypes(include="number").columns.tolist()
                    predictores = [c for c in num_cols if c != "price"]

                    # Necesitamos al menos 2 predictores para que sea "múltiple"
                    if "accommodates" not in predictores:
                        st.info("No se encontró 'accommodates' como variable numérica.")
                        continue

                    if len(predictores) < 2:
                        st.info("Se requieren al menos 2 variables explicativas numéricas para regresión múltiple.")
                        continue

                    tmp = df_ciudad[predictores + ["price"]].astype(float).dropna()
                    if len(tmp) < len(predictores) + 1:
                        st.info("Datos insuficientes para ajustar el modelo múltiple.")
                        continue

                    X = tmp[predictores].to_numpy()
                    y = tmp["price"].to_numpy()

                    modelo = LinearRegression()
                    modelo.fit(X, y)

                    r2 = modelo.score(X, y)
                    intercepto = modelo.intercept_
                    coefs = modelo.coef_

                    # Métricas generales
                    m1, m2, m3 = st.columns(3)
                    m1.metric("R² (múltiple)", f"{r2:.3f}")
                    m2.metric("N predictores", f"{len(predictores)}")
                    m3.metric("Intercepto (β0)", f"{intercepto:.2f}")

                    # Tabla con coeficientes
                    coef_df = pd.DataFrame({
                        "Variable": predictores,
                        "Coeficiente (β)": coefs
                    })
                    st.dataframe(coef_df, use_container_width=True)

                    # Gráfica: efecto de accommodates manteniendo las demás en su media
                    if "accommodates" in predictores:
                        fig = px.scatter(
                            tmp, x="accommodates", y="price",
                            labels={"accommodates": "Accommodates", "price": "Price"},
                            title="Price ~ Accommodates (Regresión múltiple)"
                        )

                        x_line = np.linspace(
                            tmp["accommodates"].min(),
                            tmp["accommodates"].max(),
                            50
                        )

                        # Vector con la media de cada predictor
                        medias = tmp[predictores].mean()
                        X_line = np.tile(medias.to_numpy(), (50, 1))

                        # Sustituimos accommodates por los valores de x_line
                        idx_acc = predictores.index("accommodates")
                        X_line[:, idx_acc] = x_line

                        y_line = modelo.predict(X_line)

                        fig.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="Predicción (otros = media)"
                        ))

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No se puede graficar porque no está 'accommodates' entre los predictores.")

        # ---- Hallazgos ----
        st.markdown("### Hallazgos")
        for l in generar_hallazgos(ciudades_reg_sel):
            st.markdown(l)