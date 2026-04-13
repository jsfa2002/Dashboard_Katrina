"""
app.py — Dashboard de Inteligencia Comercial
Almacén Fábrica Sacos y Suéteres Katrina
=========================================
Desarrollado por Innovarte Consulting

Ejecutar localmente:
    pip install streamlit pandas numpy plotly openpyxl
    streamlit run app.py

Desplegado en: https://streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# ── Configuración de página ───────────────────────────────────────
st.set_page_config(
    page_title="Katrina Dashboard",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tema visual: navy + gold (igual que mockup) ───────────────────
NAVY    = "#1A1F3C"
NAVY_M  = "#252B4A"
NAVY_L  = "#2D3561"
GOLD    = "#D4A843"
GOLD_L  = "#F0C040"
WHITE   = "#FFFFFF"
GREEN   = "#27AE60"
RED     = "#E74C3C"
GRAY    = "#8892A4"
LIGHT_BROWN = "#D2B48C"   # Café claro

CSS = f"""
<style>
/* Fondo principal */
.stApp {{ background-color: {NAVY}; color: {WHITE}; }}
[data-testid="stSidebar"] {{ background-color: {NAVY_M}; }}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

/* Títulos */
h1, h2, h3, h4 {{ color: {GOLD} !important; font-family: 'Calibri', sans-serif; }}

/* Métricas */
[data-testid="stMetric"] {{
    background-color: {NAVY_M};
    border: 1px solid {NAVY_L};
    border-radius: 10px;
    padding: 12px 16px;
    border-top: 3px solid {GOLD};
}}
[data-testid="stMetricLabel"] {{ color: {GRAY} !important; font-size: 12px; }}
[data-testid="stMetricValue"] {{ color: {WHITE} !important; font-size: 28px; font-weight: bold; }}
[data-testid="stMetricDelta"] {{ font-size: 12px; }}

/* Botón principal (general) */
.stButton > button {{
    background-color: {GOLD};
    color: {NAVY};
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    transition: 0.2s;
}}
.stButton > button:hover {{ background-color: {GOLD_L}; }}

/* Botón de descarga de plantilla en el sidebar - forzamos estilo */
section[data-testid="stSidebar"] .stButton > button {{
    background-color: {LIGHT_BROWN} !important;
    color: {NAVY} !important;
    border: 1px solid {GOLD} !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #E0C090 !important;
}}

/* Expander "¿Cómo registrar los datos?" */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background-color: {LIGHT_BROWN} !important;
    border-radius: 10px !important;
    border: 1px solid {GOLD} !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    background-color: {LIGHT_BROWN} !important;
    color: {NAVY} !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
    background-color: {LIGHT_BROWN} !important;
    color: {NAVY} !important;
    border-radius: 10px !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] * {{
    color: {NAVY} !important;
}}

/* File uploader - fondo y borde */
[data-testid="stFileUploader"] {{
    background-color: {NAVY_M};
    border: 1px dashed {GOLD};
    border-radius: 10px;
    padding: 10px;
}}

/* Botón dentro del file uploader */
[data-testid="stFileUploader"] button {{
    background-color: {GOLD} !important;
    color: {NAVY} !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: bold !important;
    padding: 8px 20px !important;
}}
[data-testid="stFileUploader"] button:hover {{
    background-color: {GOLD_L} !important;
}}

/* Texto dentro del uploader */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{
    color: {WHITE} !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ background-color: {NAVY_M}; border-radius: 8px; }}
.stTabs [data-baseweb="tab"] {{ color: {GRAY}; }}
.stTabs [aria-selected="true"] {{ color: {GOLD} !important; border-bottom: 2px solid {GOLD}; }}

/* Separadores */
hr {{ border-color: {NAVY_L}; }}

/* DataFrames */
[data-testid="stDataFrame"] {{ background-color: {NAVY_M}; }}

/* Selectbox, multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background-color: {NAVY_M};
    color: {WHITE};
    border-color: {NAVY_L};
}}

/* Alert info */
.stAlert {{ background-color: {NAVY_M}; border-left: 4px solid {GOLD}; }}

/* Success */
.element-container .stSuccess {{ background-color: #0D2E1A; border-left: 4px solid {GREEN}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Helpers de formato ────────────────────────────────────────────
def fmt_cop(val):
    return f"${val:,.0f}"

def pct(val, total):
    if total == 0: return 0
    return round(val / total * 100, 1)

def fig_layout(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(color=GOLD, size=14)),
        paper_bgcolor=NAVY_M,
        plot_bgcolor=NAVY_M,
        font=dict(color=WHITE, family="Calibri"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=height,
        legend=dict(bgcolor=NAVY_M, bordercolor=NAVY_L, font=dict(color=WHITE)),
    )
    fig.update_xaxes(gridcolor=NAVY_L, linecolor=NAVY_L, tickfont=dict(color=GRAY))
    fig.update_yaxes(gridcolor=NAVY_L, linecolor=NAVY_L, tickfont=dict(color=GRAY))
    return fig

COLORES_SERIE = [GOLD, "#5B9BD5", "#70AD47", "#FF7043", "#AB47BC", "#26C6DA"]

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 12px 0 8px;'>
        <div style='font-size:32px;'>👔</div>
        <div style='color:{GOLD}; font-size:18px; font-weight:bold; line-height:1.2;'>
            Almacén Katrina
        </div>
        <div style='color:{GRAY}; font-size:11px;'>Inteligencia Comercial</div>
        <div style='color:{GRAY}; font-size:10px; margin-top:4px;'>
            Innovarte Consulting · 2025
        </div>
    </div>
    <hr style='border-color:{NAVY_L}; margin:8px 0;'>
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='color:{GOLD}; font-size:13px; font-weight:bold;'>📂 Cargar datos</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY}; font-size:11px;'>Sube uno o varios archivos CSV o Excel (uno por mes/periodo).</p>", unsafe_allow_html=True)

    archivos = st.file_uploader(
        label="Seleccionar archivos",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Puedes subir varios meses a la vez. El dashboard los unifica automáticamente.",
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2D3561; margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GOLD}; font-size:12px; font-weight:bold;'>🗂 Columnas requeridas</p>", unsafe_allow_html=True)
    columnas_req = [
        "fecha_pedido","referencia","categoria",
        "cantidad","total_venta","margen_bruto",
        "estado_pedido","canal","metodo_pago","talla"
    ]
    for c in columnas_req:
        st.markdown(f"<span style='color:{GRAY}; font-size:10px;'>· {c}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2D3561; margin:12px 0;'>", unsafe_allow_html=True)
    with st.expander("📥 ¿Cómo registrar los datos?", expanded=False):
        st.markdown(f"""
        <div style='color:{NAVY}; font-size:11px; line-height:1.7;'>
        <b style='color:{NAVY};'>Proceso recomendado:</b><br>
        1. Por cada pedido recibido por WhatsApp, diligenciar una fila en el Excel.<br>
        2. Al cerrar el mes, exportar como CSV.<br>
        3. Subir el archivo a este dashboard.<br><br>
        <b style='color:{NAVY};'>Frecuencia:</b> mensual o semanal<br>
        <b style='color:{NAVY};'>Plantilla:</b> descarga el archivo de ejemplo con el botón de abajo.
        </div>
        """, unsafe_allow_html=True)

# ── Cargar y unificar datos ───────────────────────────────────────
@st.cache_data
def cargar_archivos(archivos_bytes_lista):
    dfs = []
    errores = []
    nombres = []
    for nombre, contenido in archivos_bytes_lista:
        try:
            if nombre.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(contenido), encoding="utf-8-sig")
            else:
                df = pd.read_excel(io.BytesIO(contenido))
            df["_archivo_origen"] = nombre
            dfs.append(df)
            nombres.append(nombre)
        except Exception as e:
            errores.append(f"{nombre}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True), errores, nombres
    return None, errores, nombres

def preparar_df(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "fecha_pedido" in df.columns:
        df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"], errors="coerce")
        df["mes"] = df["fecha_pedido"].dt.month
        df["mes_nombre"] = df["fecha_pedido"].dt.strftime("%b %Y")
        df["semana"] = df["fecha_pedido"].dt.isocalendar().week.astype(int)
    if "estado_pedido" in df.columns:
        df["estado_pedido"] = df["estado_pedido"].str.strip()
    for col in ["total_venta", "margen_bruto", "precio_unitario", "costo_produccion", "cantidad"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "fecha_entrega_comprometida" in df.columns and "fecha_entrega_real" in df.columns:
        df["fecha_entrega_comprometida"] = pd.to_datetime(df["fecha_entrega_comprometida"], errors="coerce")
        df["fecha_entrega_real"] = pd.to_datetime(df["fecha_entrega_real"], errors="coerce")
        df["dias_retraso"] = (df["fecha_entrega_real"] - df["fecha_entrega_comprometida"]).dt.days.fillna(0)
        df["entregado_a_tiempo"] = df["dias_retraso"] <= 0
    return df

def generar_plantilla():
    ejemplo = pd.DataFrame([{
        "id_pedido": "KAT-202501-001",
        "fecha_pedido": "2025-01-15",
        "nombre_cliente": "Carlos Martínez",
        "canal": "WhatsApp",
        "referencia": "Camisa Clásica Blanca",
        "categoria": "Camisas",
        "talla": "M",
        "cantidad": 2,
        "precio_unitario": 69900,
        "costo_produccion": 32000,
        "total_venta": 139800,
        "margen_bruto": 75800,
        "estado_pedido": "Entregado",
        "fecha_entrega_comprometida": "2025-01-20",
        "fecha_entrega_real": "2025-01-20",
        "metodo_pago": "Nequi",
        "notas": "",
    }])
    return ejemplo.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

# ── Contenido principal ───────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.markdown(f"""
    <div style='background:{NAVY_M}; border-radius:12px; padding:14px; text-align:center;
         border:2px solid {GOLD}; margin-top:4px;'>
        <span style='font-size:36px;'>👔</span>
    </div>""", unsafe_allow_html=True)
with col_titulo:
    st.markdown(f"""
    <div style='padding:6px 0 0 8px;'>
        <span style='color:{GOLD}; font-size:26px; font-weight:bold;'>
            Dashboard de Inteligencia Comercial
        </span><br>
        <span style='color:{WHITE}; font-size:16px;'>
            Almacén Fábrica Sacos y Suéteres Katrina
        </span><br>
        <span style='color:{GRAY}; font-size:11px;'>
            Innovarte Consulting · Bogotá, Colombia
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2D3561; margin:10px 0 16px;'>", unsafe_allow_html=True)

# Botón plantilla siempre visible (dentro del sidebar)
with st.sidebar:
    st.download_button(
        label="📥 Descargar plantilla Excel",
        data=generar_plantilla(),
        file_name="plantilla_katrina.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Sin archivos: pantalla de bienvenida ──────────────────────────
if not archivos:
    st.markdown(f"""
    <div style='background:{NAVY_M}; border:1px solid {NAVY_L}; border-radius:14px;
         padding:36px; text-align:center; margin-top:20px;'>
        <div style='font-size:52px; margin-bottom:16px;'>📂</div>
        <h2 style='color:{GOLD}; margin-bottom:8px;'>Bienvenido al Dashboard de Katrina</h2>
        <p style='color:{WHITE}; font-size:15px; max-width:520px; margin:0 auto 16px;'>
            Sube tus archivos de ventas en el panel izquierdo para comenzar a explorar
            tus datos. Puedes subir varios meses a la vez.
        </p>
        <hr style='border-color:{NAVY_L}; max-width:400px; margin:16px auto;'>
        <div style='display:flex; justify-content:center; gap:40px; flex-wrap:wrap; margin-top:8px;'>
    """, unsafe_allow_html=True)

    pasos = [
        ("1️⃣", "Descarga la plantilla", "Usa el botón en el panel izquierdo"),
        ("2️⃣", "Llena tus datos", "Un fila = un pedido. Guarda como CSV."),
        ("3️⃣", "Sube al dashboard", "Selecciona uno o más archivos"),
        ("4️⃣", "Explora y decide", "Ventas, márgenes, canales y más"),
    ]
    cols_p = st.columns(4)
    for col, (emoji, titulo, desc) in zip(cols_p, pasos):
        with col:
            st.markdown(f"""
            <div style='background:{NAVY_L}; border-radius:10px; padding:16px 12px;
                 text-align:center; border-top:3px solid {GOLD};'>
                <div style='font-size:28px;'>{emoji}</div>
                <div style='color:{GOLD}; font-weight:bold; font-size:13px; margin:6px 0 4px;'>{titulo}</div>
                <div style='color:{GRAY}; font-size:11px;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ── Procesar archivos ─────────────────────────────────────────────
archivos_bytes = [(f.name, f.read()) for f in archivos]
df_raw, errores, nombres_ok = cargar_archivos(tuple(
    (n, b) for n, b in archivos_bytes
))

if errores:
    for e in errores:
        st.error(f"Error al leer: {e}")

if df_raw is None or df_raw.empty:
    st.warning("No se pudieron leer datos válidos. Verifica el formato de los archivos.")
    st.stop()

df = preparar_df(df_raw)

# ── Info de archivos cargados ─────────────────────────────────────
with st.expander(f"✅ {len(nombres_ok)} archivo(s) cargado(s) — {len(df):,} registros totales", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<b style='color:{GOLD};'>Archivos:</b>", unsafe_allow_html=True)
        for n in nombres_ok:
            sub = df[df["_archivo_origen"] == n]
            st.markdown(f"<span style='color:{GRAY};'>· {n} — {len(sub)} registros</span>", unsafe_allow_html=True)
    with col_b:
        if "fecha_pedido" in df.columns:
            fecha_min = df["fecha_pedido"].min()
            fecha_max = df["fecha_pedido"].max()
            st.markdown(f"<b style='color:{GOLD};'>Período:</b><br>"
                       f"<span style='color:{GRAY};'>{fecha_min.strftime('%d %b %Y')} → {fecha_max.strftime('%d %b %Y')}</span>",
                       unsafe_allow_html=True)

# ── Filtros globales ──────────────────────────────────────────────
st.markdown(f"<p style='color:{GOLD}; font-weight:bold; font-size:13px; margin-bottom:6px;'>🔍 Filtros</p>", unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns(4)

df_e = df[df["estado_pedido"] == "Entregado"].copy()

with fc1:
    if "categoria" in df.columns:
        cats = ["Todas"] + sorted(df["categoria"].dropna().unique().tolist())
        cat_sel = st.selectbox("Categoría", cats, key="cat_filter")
    else:
        cat_sel = "Todas"

with fc2:
    if "canal" in df.columns:
        canales = ["Todos"] + sorted(df["canal"].dropna().unique().tolist())
        canal_sel = st.selectbox("Canal", canales, key="canal_filter")
    else:
        canal_sel = "Todos"

with fc3:
    if "mes_nombre" in df.columns:
        meses_disp = sorted(df["mes_nombre"].dropna().unique().tolist(),
                            key=lambda x: pd.to_datetime(x, format="%b %Y"))
        meses_sel = st.multiselect("Meses", meses_disp, default=meses_disp, key="mes_filter")
    else:
        meses_sel = []

with fc4:
    if "talla" in df.columns:
        tallas = ["Todas"] + sorted(df["talla"].dropna().unique().tolist())
        talla_sel = st.selectbox("Talla", tallas, key="talla_filter")
    else:
        talla_sel = "Todas"

# Aplicar filtros
df_f = df.copy()
df_fe = df_e.copy()
if cat_sel != "Todas":
    df_f = df_f[df_f["categoria"] == cat_sel]
    df_fe = df_fe[df_fe["categoria"] == cat_sel]
if canal_sel != "Todos":
    df_f = df_f[df_f["canal"] == canal_sel]
    df_fe = df_fe[df_fe["canal"] == canal_sel]
if meses_sel:
    df_f = df_f[df_f["mes_nombre"].isin(meses_sel)]
    df_fe = df_fe[df_fe["mes_nombre"].isin(meses_sel)]
if talla_sel != "Todas":
    df_f = df_f[df_f["talla"] == talla_sel]
    df_fe = df_fe[df_fe["talla"] == talla_sel]

st.markdown("<hr style='border-color:#2D3561; margin:10px 0 14px;'>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen General",
    "🛍️ Productos",
    "📡 Canales & Pagos",
    "⏱️ Operaciones",
    "📋 Datos Crudos",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════
with tab1:
    # KPIs
    total_ventas   = df_fe["total_venta"].sum()
    total_margen   = df_fe["margen_bruto"].sum()
    total_pedidos  = len(df_f)
    entregados     = len(df_f[df_f["estado_pedido"] == "Entregado"])
    cancelados     = len(df_f[df_f["estado_pedido"] == "Cancelado"])
    tasa_conv      = pct(entregados, total_pedidos)
    margen_pct     = pct(total_margen, total_ventas)
    ticket_prom    = total_ventas / entregados if entregados > 0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("💰 Ventas totales", fmt_cop(total_ventas),
                  help="Suma de total_venta para pedidos Entregados")
    with k2:
        st.metric("📈 Margen bruto", fmt_cop(total_margen),
                  help="Ventas − Costo de producción")
    with k3:
        st.metric("% Margen", f"{margen_pct:.1f}%")
    with k4:
        st.metric("📦 Pedidos", f"{total_pedidos:,}")
    with k5:
        st.metric("✅ Tasa entrega", f"{tasa_conv:.1f}%",
                  delta=f"{cancelados} cancelados" if cancelados > 0 else None,
                  delta_color="inverse")
    with k6:
        st.metric("🎫 Ticket promedio", fmt_cop(ticket_prom))

    st.markdown("<br>", unsafe_allow_html=True)

    # Ventas por mes
    c_izq, c_der = st.columns([3, 2])
    with c_izq:
        if "mes_nombre" in df_fe.columns and not df_fe.empty:
            orden_meses = sorted(df_fe["mes_nombre"].unique(),
                                 key=lambda x: pd.to_datetime(x, format="%b %Y"))
            vxm = (df_fe.groupby("mes_nombre", observed=True)
                   .agg(ventas=("total_venta","sum"), margen=("margen_bruto","sum"),
                        pedidos=("total_venta","count"))
                   .reindex(orden_meses).reset_index())

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=vxm["mes_nombre"], y=vxm["ventas"],
                name="Ventas", marker_color=GOLD,
                text=[fmt_cop(v) for v in vxm["ventas"]],
                textposition="outside", textfont=dict(size=9, color=WHITE),
            ))
            fig.add_trace(go.Bar(
                x=vxm["mes_nombre"], y=vxm["margen"],
                name="Margen", marker_color="#5B9BD5",
            ))
            fig.add_trace(go.Scatter(
                x=vxm["mes_nombre"], y=vxm["pedidos"],
                name="Pedidos", mode="lines+markers",
                marker=dict(color=GREEN, size=8),
                line=dict(color=GREEN, width=2),
                yaxis="y2",
            ))
            fig.update_layout(
                barmode="group", yaxis2=dict(overlaying="y", side="right",
                showgrid=False, tickfont=dict(color=GREEN)),
            )
            fig_layout(fig, "Ventas y Margen por Mes", 360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos de fecha disponibles.")

    with c_der:
        if "estado_pedido" in df_f.columns and not df_f.empty:
            estado_cnt = df_f["estado_pedido"].value_counts().reset_index()
            estado_cnt.columns = ["Estado", "Cantidad"]
            colores_estado = {
                "Entregado": GREEN, "Pendiente": GOLD, "Cancelado": RED,
                "En producción": "#5B9BD5",
            }
            fig_d = go.Figure(go.Pie(
                labels=estado_cnt["Estado"],
                values=estado_cnt["Cantidad"],
                hole=0.55,
                marker_colors=[colores_estado.get(e, GRAY) for e in estado_cnt["Estado"]],
                textfont=dict(color=WHITE, size=11),
            ))
            fig_d.add_annotation(
                text=f"<b>{total_pedidos}</b><br><span style='font-size:10px'>pedidos</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color=WHITE),
            )
            fig_layout(fig_d, "Estado de Pedidos", 360)
            st.plotly_chart(fig_d, use_container_width=True)

    if "mes_nombre" in df_fe.columns and not df_fe.empty:
        st.markdown(f"<h4 style='color:{GOLD};'>Resumen mensual</h4>", unsafe_allow_html=True)
        orden_meses = sorted(df_fe["mes_nombre"].unique(),
                             key=lambda x: pd.to_datetime(x, format="%b %Y"))
        resumen_mes = (df_fe.groupby("mes_nombre", observed=True)
                       .agg(
                           Ventas=("total_venta","sum"),
                           Margen=("margen_bruto","sum"),
                           Pedidos=("total_venta","count"),
                           Ticket_Prom=("total_venta","mean"),
                       ).reindex(orden_meses).reset_index())
        resumen_mes.columns = ["Mes","Ventas (COP)","Margen (COP)","Pedidos","Ticket Prom (COP)"]
        resumen_mes["Margen %"] = (resumen_mes["Margen (COP)"] / resumen_mes["Ventas (COP)"] * 100).round(1)
        for col in ["Ventas (COP)","Margen (COP)","Ticket Prom (COP)"]:
            resumen_mes[col] = resumen_mes[col].apply(lambda x: fmt_cop(x))
        resumen_mes["Margen %"] = resumen_mes["Margen %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(resumen_mes, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCTOS
# ══════════════════════════════════════════════════════════════════
with tab2:
    if df_fe.empty:
        st.warning("Sin datos de pedidos entregados con los filtros seleccionados.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if "referencia" in df_fe.columns:
                top_ref = (df_fe.groupby("referencia", observed=True)
                           .agg(ventas=("total_venta","sum"), unidades=("cantidad","sum"),
                                margen=("margen_bruto","sum"))
                           .sort_values("ventas", ascending=True).tail(9).reset_index())
                fig_ref = go.Figure(go.Bar(
                    x=top_ref["ventas"], y=top_ref["referencia"],
                    orientation="h", marker_color=GOLD,
                    text=[fmt_cop(v) for v in top_ref["ventas"]],
                    textposition="outside", textfont=dict(size=9, color=WHITE),
                ))
                fig_layout(fig_ref, "Ventas por Referencia (COP)", 360)
                st.plotly_chart(fig_ref, use_container_width=True)

        with c2:
            if "categoria" in df_fe.columns:
                cat_v = (df_fe.groupby("categoria", observed=True)
                         ["total_venta"].sum().reset_index())
                fig_cat = go.Figure(go.Pie(
                    labels=cat_v["categoria"], values=cat_v["total_venta"],
                    hole=0.5,
                    marker_colors=[GOLD, "#5B9BD5", GREEN],
                    textfont=dict(color=WHITE, size=12),
                ))
                fig_layout(fig_cat, "Ventas por Categoría", 360)
                st.plotly_chart(fig_cat, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "referencia" in df_fe.columns:
                marg_ref = (df_fe.groupby("referencia", observed=True)
                            .agg(margen=("margen_bruto","sum"),
                                 ventas=("total_venta","sum"))
                            .reset_index())
                marg_ref["margen_pct"] = (marg_ref["margen"] / marg_ref["ventas"] * 100).round(1)
                marg_ref = marg_ref.sort_values("margen_pct", ascending=False)
                colors_bar = [GREEN if m >= 40 else GOLD if m >= 30 else RED
                              for m in marg_ref["margen_pct"]]
                fig_marg = go.Figure(go.Bar(
                    x=marg_ref["referencia"], y=marg_ref["margen_pct"],
                    marker_color=colors_bar,
                    text=[f"{m:.0f}%" for m in marg_ref["margen_pct"]],
                    textposition="outside", textfont=dict(size=9, color=WHITE),
                ))
                fig_marg.add_hline(y=30, line=dict(color=GOLD, dash="dash", width=1.5),
                                   annotation_text="Meta 30%",
                                   annotation_font=dict(color=GOLD, size=10))
                fig_layout(fig_marg, "Margen % por Referencia", 340)
                fig_marg.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_marg, use_container_width=True)

        with c4:
            if "talla" in df_fe.columns:
                talla_v = (df_fe.groupby("talla", observed=True)
                           ["cantidad"].sum().reset_index()
                           .sort_values("talla"))
                fig_talla = go.Figure(go.Bar(
                    x=talla_v["talla"], y=talla_v["cantidad"],
                    marker_color=GOLD,
                    text=talla_v["cantidad"], textposition="outside",
                    textfont=dict(size=11, color=WHITE),
                ))
                fig_layout(fig_talla, "Unidades Vendidas por Talla", 340)
                st.plotly_chart(fig_talla, use_container_width=True)

        if "referencia" in df_fe.columns:
            st.markdown(f"<h4 style='color:{GOLD};'>Rentabilidad por Producto</h4>", unsafe_allow_html=True)
            tabla_prod = (df_fe.groupby(["categoria","referencia"], observed=True)
                          .agg(
                              Unidades=("cantidad","sum"),
                              Ventas=("total_venta","sum"),
                              Margen=("margen_bruto","sum"),
                          ).reset_index())
            tabla_prod["Margen %"] = (tabla_prod["Margen"] / tabla_prod["Ventas"] * 100).round(1)
            tabla_prod["Ventas"] = tabla_prod["Ventas"].apply(fmt_cop)
            tabla_prod["Margen"] = tabla_prod["Margen"].apply(fmt_cop)
            tabla_prod["Margen %"] = tabla_prod["Margen %"].apply(lambda x: f"{x:.1f}%")
            tabla_prod.columns = ["Categoría","Referencia","Unidades","Ventas (COP)","Margen (COP)","Margen %"]
            st.dataframe(tabla_prod, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — CANALES & PAGOS
# ══════════════════════════════════════════════════════════════════
with tab3:
    if df_fe.empty:
        st.warning("Sin datos con los filtros seleccionados.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if "canal" in df_fe.columns:
                canal_v = (df_fe.groupby("canal", observed=True)
                           .agg(ventas=("total_venta","sum"),
                                pedidos=("total_venta","count"))
                           .sort_values("ventas", ascending=False).reset_index())
                fig_canal = go.Figure(go.Bar(
                    x=canal_v["canal"], y=canal_v["ventas"],
                    marker_color=[GOLD,"#5B9BD5",GREEN,RED],
                    text=[fmt_cop(v) for v in canal_v["ventas"]],
                    textposition="outside", textfont=dict(size=9, color=WHITE),
                ))
                fig_layout(fig_canal, "Ventas por Canal (COP)", 340)
                st.plotly_chart(fig_canal, use_container_width=True)

        with c2:
            if "metodo_pago" in df_fe.columns:
                pago_v = (df_fe.groupby("metodo_pago", observed=True)
                          ["total_venta"].sum().reset_index())
                fig_pago = go.Figure(go.Pie(
                    labels=pago_v["metodo_pago"], values=pago_v["total_venta"],
                    hole=0.5,
                    marker_colors=[GOLD,"#5B9BD5",GREEN,"#AB47BC"],
                    textfont=dict(color=WHITE, size=12),
                ))
                fig_layout(fig_pago, "Ventas por Método de Pago", 340)
                st.plotly_chart(fig_pago, use_container_width=True)

        if "canal" in df_fe.columns and "mes_nombre" in df_fe.columns:
            orden_meses = sorted(df_fe["mes_nombre"].unique(),
                                 key=lambda x: pd.to_datetime(x, format="%b %Y"))
            canal_mes = (df_fe.groupby(["mes_nombre","canal"], observed=True)
                         ["total_venta"].sum().unstack(fill_value=0)
                         .reindex(orden_meses))
            fig_evo = go.Figure()
            for idx, col_name in enumerate(canal_mes.columns):
                fig_evo.add_trace(go.Scatter(
                    x=canal_mes.index, y=canal_mes[col_name],
                    name=col_name, mode="lines+markers",
                    line=dict(color=COLORES_SERIE[idx % len(COLORES_SERIE)], width=2),
                    marker=dict(size=7),
                ))
            fig_layout(fig_evo, "Evolución de Ventas por Canal y Mes", 320)
            st.plotly_chart(fig_evo, use_container_width=True)

        if "canal" in df_fe.columns:
            st.markdown(f"<h4 style='color:{GOLD};'>Detalle por Canal</h4>", unsafe_allow_html=True)
            canal_det = (df_fe.groupby("canal", observed=True)
                         .agg(
                             Pedidos=("total_venta","count"),
                             Ventas=("total_venta","sum"),
                             Margen=("margen_bruto","sum"),
                             Ticket_Prom=("total_venta","mean"),
                         ).reset_index())
            canal_det["Participación %"] = (canal_det["Ventas"] / canal_det["Ventas"].sum() * 100).round(1)
            for col in ["Ventas","Margen","Ticket_Prom"]:
                canal_det[col] = canal_det[col].apply(fmt_cop)
            canal_det["Participación %"] = canal_det["Participación %"].apply(lambda x: f"{x:.1f}%")
            canal_det.columns = ["Canal","Pedidos","Ventas (COP)","Margen (COP)","Ticket Prom","Participación %"]
            st.dataframe(canal_det, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — OPERACIONES
# ══════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        if "fecha_pedido" in df_f.columns and not df_f.empty:
            df_f["dia_semana"] = df_f["fecha_pedido"].dt.day_name()
            orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dias_esp   = {"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles",
                          "Thursday":"Jueves","Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"}
            dias_v = (df_f.groupby("dia_semana")["id_pedido"].count()
                      .reindex(orden_dias).fillna(0).reset_index())
            dias_v["dia_esp"] = dias_v["dia_semana"].map(dias_esp)
            fig_dias = go.Figure(go.Bar(
                x=dias_v["dia_esp"], y=dias_v["id_pedido"],
                marker_color=GOLD,
                text=dias_v["id_pedido"].astype(int),
                textposition="outside", textfont=dict(color=WHITE, size=10),
            ))
            fig_layout(fig_dias, "Pedidos por Día de la Semana", 320)
            st.plotly_chart(fig_dias, use_container_width=True)

    with c2:
        if "entregado_a_tiempo" in df_fe.columns and not df_fe.empty:
            a_tiempo    = df_fe["entregado_a_tiempo"].sum()
            con_retraso = (~df_fe["entregado_a_tiempo"]).sum()
            fig_puntual = go.Figure(go.Pie(
                labels=["A tiempo","Con retraso"],
                values=[a_tiempo, con_retraso],
                hole=0.55,
                marker_colors=[GREEN, RED],
                textfont=dict(color=WHITE, size=13),
            ))
            pct_tiempo = pct(a_tiempo, a_tiempo + con_retraso)
            fig_puntual.add_annotation(
                text=f"<b>{pct_tiempo:.0f}%</b><br><span style='font-size:10px'>a tiempo</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color=WHITE),
            )
            fig_layout(fig_puntual, "Puntualidad de Entregas", 320)
            st.plotly_chart(fig_puntual, use_container_width=True)
        else:
            st.info("Necesitas columnas 'fecha_entrega_comprometida' y 'fecha_entrega_real' para ver puntualidad.")

    if "dias_retraso" in df_fe.columns and "referencia" in df_fe.columns and not df_fe.empty:
        retraso_ref = (df_fe.groupby("referencia", observed=True)["dias_retraso"]
                       .mean().reset_index().sort_values("dias_retraso", ascending=False))
        retraso_ref["dias_retraso"] = retraso_ref["dias_retraso"].round(1)
        colores_ret = [RED if d > 1 else GOLD if d > 0 else GREEN
                       for d in retraso_ref["dias_retraso"]]
        fig_ret = go.Figure(go.Bar(
            x=retraso_ref["referencia"], y=retraso_ref["dias_retraso"],
            marker_color=colores_ret,
            text=[f"{d:.1f}d" for d in retraso_ref["dias_retraso"]],
            textposition="outside", textfont=dict(size=9, color=WHITE),
        ))
        fig_ret.add_hline(y=0, line=dict(color=GREEN, dash="dash", width=1.5))
        fig_layout(fig_ret, "Retraso Promedio por Referencia (días)", 300)
        fig_ret.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown(f"<h4 style='color:{GOLD};'>Indicadores Operacionales</h4>", unsafe_allow_html=True)
    ko1, ko2, ko3, ko4 = st.columns(4)
    with ko1:
        if "dias_retraso" in df_fe.columns and not df_fe.empty:
            st.metric("⏱️ Retraso promedio", f"{df_fe['dias_retraso'].mean():.1f} días")
    with ko2:
        if "entregado_a_tiempo" in df_fe.columns and not df_fe.empty:
            st.metric("✅ % Entrega a tiempo", f"{pct(df_fe['entregado_a_tiempo'].sum(), len(df_fe)):.0f}%")
    with ko3:
        cancelados_f = len(df_f[df_f["estado_pedido"] == "Cancelado"])
        st.metric("❌ Pedidos cancelados", f"{cancelados_f}")
    with ko4:
        if "cantidad" in df_fe.columns and not df_fe.empty:
            prom_cant = df_fe["cantidad"].mean()
            st.metric("📦 Unidades/pedido", f"{prom_cant:.1f}")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — DATOS CRUDOS
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<h4 style='color:{GOLD};'>Tabla de datos ({len(df_f):,} registros)</h4>",
                unsafe_allow_html=True)

    buscar = st.text_input("🔍 Buscar (cliente, referencia, canal...)", placeholder="Escribir para filtrar...",
                           label_visibility="collapsed")
    if buscar:
        mask = df_f.astype(str).apply(lambda col: col.str.contains(buscar, case=False)).any(axis=1)
        df_mostrar = df_f[mask]
    else:
        df_mostrar = df_f

    cols_mostrar = [c for c in df_mostrar.columns if not c.startswith("_")]
    st.dataframe(df_mostrar[cols_mostrar].reset_index(drop=True),
                 use_container_width=True, height=400)

    csv_export = df_mostrar[cols_mostrar].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Descargar datos filtrados (CSV)",
        data=csv_export,
        file_name=f"katrina_datos_filtrados_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; margin-top:24px; padding:14px;
     background:{NAVY_M}; border-radius:10px; border-top:2px solid {GOLD};'>
    <span style='color:{GOLD}; font-size:12px; font-weight:bold;'>
        Innovarte Consulting
    </span>
    <span style='color:{GRAY}; font-size:11px;'>
        &nbsp;·&nbsp; Almacén Fábrica Sacos y Suéteres Katrina &nbsp;·&nbsp; 2025
    </span><br>
    <span style='color:{GRAY}; font-size:10px;'>
        Dashboard de Inteligencia Comercial — Propuesta E2E Digital
    </span>
</div>
""", unsafe_allow_html=True)
