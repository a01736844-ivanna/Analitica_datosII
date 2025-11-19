import streamlit as st

# ==============Configuración pagina=====================
st.set_page_config(
    page_title="Dashboard Airbnb",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

#====== Definición de páginas=============================
pg_extraccion = st.Page("pages/1_extraccion.py", title="Extracción de Características", icon="🔎")
pg_tablas = st.Page("pages/2_tablas.py", title="Tablas comparativas",icon="📊")
pg_regresiones = st.Page("pages/3_regresiones.py", title="Regresiones", icon="📈")


# ============= Agrupar por secciones=====================
nav = st.navigation({
    "Análisis": [pg_extraccion, pg_tablas],
    "Modelado": [pg_regresiones],
})

nav.run()