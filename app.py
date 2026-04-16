"""
app.py — Dashboard de Inteligencia Comercial + IA Predictiva & Prescriptiva
Almacen Fabrica Sacos y Sueteres Katrina
=========================================
Desarrollado por Innovarte Consulting

pip install streamlit pandas numpy plotly openpyxl scikit-learn xgboost optuna
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, base64, warnings, zipfile
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Katrina Dashboard", page_icon="K", layout="wide", initial_sidebar_state="expanded")

NAVY="1A1F3C"; NAVY_M="252B4A"; NAVY_L="2D3561"
GOLD="D4A843"; GOLD_L="F0C040"; WHITE="FFFFFF"
GREEN="27AE60"; RED="E74C3C"; GRAY="8892A4"
LIGHT_BROWN="D2B48C"; PURPLE="8E44AD"; TEAL="16A085"
N=lambda c: f"#{c}"

CSS = f"""<style>
.stApp{{background-color:{N(NAVY)};color:{N(WHITE)}}}
[data-testid="stSidebar"]{{background-color:{N(NAVY_M)}}}
[data-testid="stSidebar"] *{{color:{N(WHITE)} !important}}
h1,h2,h3,h4{{color:{N(GOLD)} !important;font-family:Calibri,sans-serif}}
[data-testid="stMetric"]{{background-color:{N(NAVY_M)};border:1px solid {N(NAVY_L)};border-radius:10px;padding:12px 16px;border-top:3px solid {N(GOLD)}}}
[data-testid="stMetricLabel"]{{color:{N(GRAY)} !important;font-size:12px}}
[data-testid="stMetricValue"]{{color:{N(WHITE)} !important;font-size:28px;font-weight:bold}}
.stButton>button{{background-color:{N(GOLD)};color:{N(NAVY)};font-weight:bold;border:none;border-radius:6px;padding:8px 20px}}
.stButton>button:hover{{background-color:{N(GOLD_L)}}}
section[data-testid="stSidebar"] [data-testid="stExpander"]{{background-color:{N(LIGHT_BROWN)} !important;border-radius:10px !important;border:1px solid {N(GOLD)} !important}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary{{background-color:{N(LIGHT_BROWN)} !important;color:{N(NAVY)} !important;font-weight:bold !important;border-radius:10px !important}}
section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"]{{background-color:{N(LIGHT_BROWN)} !important;color:{N(NAVY)} !important;border-radius:10px !important}}
section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] *{{color:{N(NAVY)} !important}}
[data-testid="stFileUploader"]{{background-color:{N(NAVY_M)};border:1px dashed {N(GOLD)};border-radius:10px;padding:10px}}
[data-testid="stFileUploader"] button{{background-color:{N(GOLD)} !important;color:{N(NAVY)} !important;border:none !important;border-radius:6px !important;font-weight:bold !important}}
[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] p{{color:{N(WHITE)} !important}}
.stTabs [data-baseweb="tab-list"]{{background-color:{N(NAVY_M)};border-radius:8px}}
.stTabs [data-baseweb="tab"]{{color:{N(GRAY)}}}
.stTabs [aria-selected="true"]{{color:{N(GOLD)} !important;border-bottom:2px solid {N(GOLD)}}}
hr{{border-color:{N(NAVY_L)}}}
[data-testid="stDataFrame"]{{background-color:{N(NAVY_M)}}}
.stSelectbox>div>div,.stMultiSelect>div>div{{background-color:{N(NAVY_M)};color:{N(WHITE)};border-color:{N(NAVY_L)}}}
.stAlert{{background-color:{N(NAVY_M)};border-left:4px solid {N(GOLD)}}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

def fmt(v): return f"${v:,.0f}"
def pct(a, b): return round(a / b * 100, 1) if b else 0

def fl(fig, title="", h=360):
    fig.update_layout(
        title=dict(text=title, font=dict(color=N(GOLD), size=14)),
        paper_bgcolor=N(NAVY_M), plot_bgcolor=N(NAVY_M),
        font=dict(color=N(WHITE), family="Calibri"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10), height=h,
        legend=dict(bgcolor=N(NAVY_M), bordercolor=N(NAVY_L), font=dict(color=N(WHITE))),
    )
    fig.update_xaxes(gridcolor=N(NAVY_L), linecolor=N(NAVY_L), tickfont=dict(color=N(GRAY)))
    fig.update_yaxes(gridcolor=N(NAVY_L), linecolor=N(NAVY_L), tickfont=dict(color=N(GRAY)))
    return fig

SERIES = [N(GOLD), "#5B9BD5", "#70AD47", "#FF7043", "#AB47BC", "#26C6DA"]

def card_ia(icono, titulo, contenido, urgencia="media"):
    bgs = {"alta": "linear-gradient(135deg,#1B0A2A,#2D1B4E)",
           "media": "linear-gradient(135deg,#0D2E1A,#1A4A2E)",
           "alerta": "linear-gradient(135deg,#2E1A0D,#4A2E1A)"}
    borders = {"alta": N(PURPLE), "media": N(GREEN), "alerta": N(GOLD)}
    tcolors = {"alta": N(PURPLE), "media": N(GREEN), "alerta": N(GOLD)}
    return (f"<div style='background:{bgs.get(urgencia,bgs['media'])};border-left:4px solid "
            f"{borders.get(urgencia,N(GREEN))};border-radius:10px;padding:14px 16px;margin-bottom:10px;'>"
            f"<span style='color:{tcolors.get(urgencia,N(GREEN))};font-weight:bold;font-size:13px;'>{titulo}</span>"
            f"<div style='color:{N(WHITE)};font-size:12px;margin-top:6px;line-height:1.6;'>{contenido}</div></div>")

def info_box(texto, color=None):
    c = color or TEAL
    return (f"<div style='background:{N(NAVY_M)};border-left:3px solid {N(c)};border-radius:6px;"
            f"padding:10px 14px;font-size:12px;color:{N(WHITE)};margin-bottom:8px;'>{texto}</div>")

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""<div style='text-align:center;padding:12px 0 8px;'>
        <div style='color:{N(GOLD)};font-size:20px;font-weight:bold;letter-spacing:2px;'>KATRINA</div>
        <div style='color:{N(GRAY)};font-size:11px;'>Inteligencia Comercial</div>
        <div style='color:{N(GRAY)};font-size:10px;margin-top:4px;'>Innovarte Consulting · 2025</div>
    </div><hr style='border-color:{N(NAVY_L)};margin:8px 0;'>""", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{N(GOLD)};font-size:13px;font-weight:bold;'>Cargar datos</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{N(GRAY)};font-size:11px;'>Sube uno o varios CSV/Excel, uno por mes.</p>", unsafe_allow_html=True)
    archivos = st.file_uploader("files", type=["csv","xlsx","xls"], accept_multiple_files=True,
                                help="El dashboard los unifica automaticamente.", label_visibility="collapsed")
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{N(GOLD)};font-size:12px;font-weight:bold;'>Columnas requeridas</p>", unsafe_allow_html=True)
    for c in ["fecha_pedido","referencia","categoria","cantidad","total_venta","margen_bruto","estado_pedido","canal","metodo_pago","talla"]:
        st.markdown(f"<span style='color:{N(GRAY)};font-size:10px;'>- {c}</span>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:12px 0;'>", unsafe_allow_html=True)
    with st.expander("Como registrar los datos", expanded=False):
        st.markdown(f"""<div style='color:{N(NAVY)};font-size:11px;line-height:1.7;'>
        <b>Proceso recomendado:</b><br>
        1. Por cada pedido recibido por WhatsApp, diligenciar una fila en Excel.<br>
        2. Al cerrar el mes, exportar como CSV.<br>
        3. Subir el archivo a este dashboard.<br><br>
        <b>Frecuencia:</b> mensual o semanal</div>""", unsafe_allow_html=True)
    plantilla = pd.DataFrame([{"id_pedido":"KAT-202501-001","fecha_pedido":"2025-01-15","nombre_cliente":"Carlos Martinez",
        "canal":"WhatsApp","referencia":"Camisa Clasica Blanca","categoria":"Camisas","talla":"M","cantidad":2,
        "precio_unitario":69900,"costo_produccion":32000,"total_venta":139800,"margen_bruto":75800,
        "estado_pedido":"Entregado","fecha_entrega_comprometida":"2025-01-20","fecha_entrega_real":"2025-01-20","metodo_pago":"Nequi","notas":""}])
    b64p = base64.b64encode(plantilla.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64p}" download="plantilla_katrina.csv" style="display:block;width:100%;background-color:{N(LIGHT_BROWN)};color:{N(NAVY)};text-align:center;padding:8px 0;border-radius:6px;text-decoration:none;font-weight:bold;border:1px solid {N(GOLD)};margin-bottom:12px;">Descargar plantilla Excel</a>', unsafe_allow_html=True)

# ── Carga ──
@st.cache_data
def cargar(blist):
    dfs,err,noms=[],[],[]
    for n,b in blist:
        try:
            d = pd.read_csv(io.BytesIO(b), encoding="utf-8-sig") if n.endswith(".csv") else pd.read_excel(io.BytesIO(b))
            d["_src"]=n; dfs.append(d); noms.append(n)
        except Exception as e: err.append(f"{n}: {e}")
    return (pd.concat(dfs,ignore_index=True),err,noms) if dfs else (None,err,noms)

def prep(df):
    df=df.copy()
    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    if "fecha_pedido" in df.columns:
        df["fecha_pedido"]=pd.to_datetime(df["fecha_pedido"],errors="coerce")
        df["mes"]=df["fecha_pedido"].dt.month
        df["mes_nombre"]=df["fecha_pedido"].dt.strftime("%b %Y")
        df["semana"]=df["fecha_pedido"].dt.isocalendar().week.astype(int)
    if "estado_pedido" in df.columns: df["estado_pedido"]=df["estado_pedido"].str.strip()
    for c in ["total_venta","margen_bruto","precio_unitario","costo_produccion","cantidad"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    if "fecha_entrega_comprometida" in df.columns and "fecha_entrega_real" in df.columns:
        df["fecha_entrega_comprometida"]=pd.to_datetime(df["fecha_entrega_comprometida"],errors="coerce")
        df["fecha_entrega_real"]=pd.to_datetime(df["fecha_entrega_real"],errors="coerce")
        df["dias_retraso"]=(df["fecha_entrega_real"]-df["fecha_entrega_comprometida"]).dt.days.fillna(0)
        df["entregado_a_tiempo"]=df["dias_retraso"]<=0
    return df

# ── Funcion de limpieza reutilizable ──
def aplicar_limpieza(df_input, opciones):
    """
    Aplica las transformaciones de limpieza a un dataframe.
    Retorna (df_limpio, log_mensajes)
    """
    df_clean = df_input.copy()
    cols_utiles = [c for c in df_clean.columns if not c.startswith("_")]
    log = []

    op_dup        = opciones.get("op_dup", True)
    op_nulos_fecha= opciones.get("op_nulos_fecha", True)
    op_negativos  = opciones.get("op_negativos", True)
    op_strip      = opciones.get("op_strip", True)
    op_nulos_ref  = opciones.get("op_nulos_ref", False)
    op_nulos_canal= opciones.get("op_nulos_canal", False)
    op_nulos_estado=opciones.get("op_nulos_estado", False)
    op_calcular_margen=opciones.get("op_calcular_margen", False)
    rango_venta   = opciones.get("rango_venta", None)
    rango_cant    = opciones.get("rango_cant", None)
    rango_fecha   = opciones.get("rango_fecha", None)
    mapeos        = opciones.get("mapeos", {})

    if op_dup:
        antes=len(df_clean)
        df_clean=df_clean.drop_duplicates(subset=[c for c in cols_utiles if c not in ["id_pedido","_src"]],keep="first")
        elim=antes-len(df_clean)
        if elim: log.append(f"Duplicados eliminados: {elim} filas.")

    if op_nulos_fecha and "fecha_pedido" in df_clean.columns:
        antes=len(df_clean); df_clean=df_clean[df_clean["fecha_pedido"].notna()]
        elim=antes-len(df_clean)
        if elim: log.append(f"Filas sin fecha valida eliminadas: {elim}.")

    if op_negativos:
        antes=len(df_clean)
        for c in ["total_venta","cantidad"]:
            if c in df_clean.columns: df_clean=df_clean[df_clean[c]>=0]
        elim=antes-len(df_clean)
        if elim: log.append(f"Filas con valores negativos eliminadas: {elim}.")

    if op_strip:
        for c in ["referencia","canal","categoria","estado_pedido","metodo_pago","talla","nombre_cliente"]:
            if c in df_clean.columns: df_clean[c]=df_clean[c].astype(str).str.strip().str.title()
        log.append("Texto estandarizado (strip + Title Case).")

    if op_nulos_ref and "referencia" in df_clean.columns:
        n=df_clean["referencia"].isna().sum(); df_clean["referencia"]=df_clean["referencia"].fillna("Sin Referencia")
        if n: log.append(f"Referencias vacias rellenadas: {n}.")

    if op_nulos_canal and "canal" in df_clean.columns:
        n=df_clean["canal"].isna().sum(); df_clean["canal"]=df_clean["canal"].fillna("Presencial")
        if n: log.append(f"Canales vacios rellenados: {n}.")

    if op_nulos_estado and "estado_pedido" in df_clean.columns:
        n=df_clean["estado_pedido"].isna().sum(); df_clean["estado_pedido"]=df_clean["estado_pedido"].fillna("Pendiente")
        if n: log.append(f"Estados vacios rellenados: {n}.")

    if op_calcular_margen:
        cond=["precio_unitario","costo_produccion","cantidad","margen_bruto"]
        if all(c in df_clean.columns for c in cond):
            mask=df_clean["margen_bruto"]==0
            df_clean.loc[mask,"margen_bruto"]=(df_clean.loc[mask,"precio_unitario"]-df_clean.loc[mask,"costo_produccion"])*df_clean.loc[mask,"cantidad"]
            if mask.sum(): log.append(f"Margen recalculado en {mask.sum()} filas.")

    if rango_venta and "total_venta" in df_clean.columns:
        antes=len(df_clean); df_clean=df_clean[(df_clean["total_venta"]>=rango_venta[0])&(df_clean["total_venta"]<=rango_venta[1])]
        elim=antes-len(df_clean)
        if elim: log.append(f"Filas fuera del rango de venta eliminadas: {elim}.")

    if rango_cant and "cantidad" in df_clean.columns:
        antes=len(df_clean); df_clean=df_clean[(df_clean["cantidad"]>=rango_cant[0])&(df_clean["cantidad"]<=rango_cant[1])]
        elim=antes-len(df_clean)
        if elim: log.append(f"Filas fuera del rango de cantidad eliminadas: {elim}.")

    if rango_fecha and "fecha_pedido" in df_clean.columns and len(rango_fecha)==2:
        antes=len(df_clean); df_clean=df_clean[(df_clean["fecha_pedido"]>=pd.Timestamp(rango_fecha[0]))&(df_clean["fecha_pedido"]<=pd.Timestamp(rango_fecha[1]))]
        elim=antes-len(df_clean)
        if elim: log.append(f"Filas fuera del rango de fechas eliminadas: {elim}.")

    for col,mapa in mapeos.items():
        if col in df_clean.columns and mapa:
            df_clean[col]=df_clean[col].replace(mapa); log.append(f"Columna '{col}': {len(mapa)} valor(es) reemplazado(s).")

    return df_clean, log

# ── Header ──
cl,ct=st.columns([1,5])
with cl:
    st.markdown(f"<div style='background:{N(NAVY_M)};border-radius:12px;padding:14px;text-align:center;border:2px solid {N(GOLD)};margin-top:4px;'><span style='color:{N(GOLD)};font-size:28px;font-weight:bold;letter-spacing:3px;'>K</span></div>", unsafe_allow_html=True)
with ct:
    st.markdown(f"<div style='padding:6px 0 0 8px;'><span style='color:{N(GOLD)};font-size:26px;font-weight:bold;'>Dashboard de Inteligencia Comercial</span><br><span style='color:{N(WHITE)};font-size:16px;'>Almacen Fabrica Sacos y Sueteres Katrina</span><br><span style='color:{N(GRAY)};font-size:11px;'>Innovarte Consulting · Bogota, Colombia</span></div>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:10px 0 16px;'>", unsafe_allow_html=True)

if not archivos:
    st.markdown(f"<div style='background:{N(NAVY_M)};border:1px solid {N(NAVY_L)};border-radius:14px;padding:36px;text-align:center;margin-top:20px;'><h2 style='color:{N(GOLD)};margin-bottom:8px;'>Bienvenido al Dashboard de Katrina</h2><p style='color:{N(WHITE)};font-size:15px;max-width:520px;margin:0 auto 16px;'>Sube tus archivos de ventas en el panel izquierdo para comenzar.</p></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    pasos=[("1","Descarga la plantilla","Panel izquierdo"),("2","Llena los datos","Una fila = un pedido"),("3","Sube los archivos","Varios meses a la vez"),("4","Limpia y explora","Limpieza, dashboard e IA")]
    for c,(n,t,d) in zip(cols,pasos):
        with c: st.markdown(f"<div style='background:{N(NAVY_L)};border-radius:10px;padding:16px 12px;text-align:center;border-top:3px solid {N(GOLD)};'><div style='color:{N(GOLD)};font-size:24px;font-weight:bold;'>{n}</div><div style='color:{N(GOLD)};font-weight:bold;font-size:13px;margin:6px 0 4px;'>{t}</div><div style='color:{N(GRAY)};font-size:11px;'>{d}</div></div>", unsafe_allow_html=True)
    st.stop()

ab=[(f.name,f.read()) for f in archivos]
df_raw,errs,noms=cargar(tuple(ab))
for e in errs: st.error(f"Error: {e}")
if df_raw is None or df_raw.empty: st.warning("No se pudieron leer datos validos."); st.stop()
df_base=prep(df_raw)

# Guardar bytes originales por archivo para descargas individuales
archivos_bytes = {f.name: b for f.name, b in ab}

with st.expander(f"{len(noms)} archivo(s) cargado(s) — {len(df_base):,} registros totales", expanded=False):
    ca,cb=st.columns(2)
    with ca:
        st.markdown(f"<b style='color:{N(GOLD)};'>Archivos:</b>", unsafe_allow_html=True)
        for n in noms:
            sub=df_base[df_base["_src"]==n]; st.markdown(f"<span style='color:{N(GRAY)};'>- {n} — {len(sub)} registros</span>", unsafe_allow_html=True)
    with cb:
        if "fecha_pedido" in df_base.columns:
            st.markdown(f"<b style='color:{N(GOLD)};'>Periodo:</b><br><span style='color:{N(GRAY)};'>{df_base['fecha_pedido'].min().strftime('%d %b %Y')} a {df_base['fecha_pedido'].max().strftime('%d %b %Y')}</span>", unsafe_allow_html=True)

if "df_limpio" not in st.session_state: st.session_state["df_limpio"]=None
if "limpieza_aprobada" not in st.session_state: st.session_state["limpieza_aprobada"]=False
if "dfs_por_archivo" not in st.session_state: st.session_state["dfs_por_archivo"]={}

T0,T1,T2,T3,T4,T5,T6=st.tabs(["Limpieza y Transformacion","Resumen General","Productos","Canales y Pagos","Operaciones","IA Predictiva y Prescriptiva","Datos Crudos"])

# ═══════════════════════════════════
# TAB 0 — LIMPIEZA Y TRANSFORMACION
# ═══════════════════════════════════
with T0:
    st.markdown(f"<div style='background:linear-gradient(135deg,{N(NAVY_M)},{N(NAVY_L)});border-radius:12px;padding:18px 22px;border-left:4px solid {N(TEAL)};margin-bottom:18px;'><span style='color:{N(TEAL)};font-size:18px;font-weight:bold;'>Limpieza y Transformacion de Datos</span><br><span style='color:{N(GRAY)};font-size:12px;'>Las transformaciones se aplican a TODOS los archivos cargados de forma unificada. Al aprobar, el dashboard usa el dataset completo limpio y puedes descargar cada archivo por separado.</span></div>", unsafe_allow_html=True)

    df_work=df_base.copy()
    cols_utiles=[c for c in df_work.columns if not c.startswith("_")]

    # ── Resumen por archivo cargado ──
    st.markdown(f"<h4 style='color:{N(GOLD)};'>Archivos cargados ({len(noms)})</h4>", unsafe_allow_html=True)
    cols_arch = st.columns(min(len(noms), 4))
    for i, nom in enumerate(noms):
        sub = df_work[df_work["_src"]==nom]
        with cols_arch[i % len(cols_arch)]:
            nulos_sub = sub[cols_utiles].isnull().sum().sum()
            st.markdown(
                f"<div style='background:{N(NAVY_M)};border-radius:8px;padding:10px 14px;border-left:3px solid {N(TEAL)};margin-bottom:8px;'>"
                f"<div style='color:{N(GOLD)};font-size:12px;font-weight:bold;'>{nom}</div>"
                f"<div style='color:{N(WHITE)};font-size:13px;font-weight:bold;margin-top:2px;'>{len(sub):,} registros</div>"
                f"<div style='color:{N(GRAY)};font-size:11px;'>Nulos: {nulos_sub} | Cols: {len(cols_utiles)}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # 1. Diagnostico (sobre todo el dataset unificado)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>1. Diagnostico del dataset unificado</h4>", unsafe_allow_html=True)
    total_filas=len(df_work)
    nulos=df_work[cols_utiles].isnull().sum()
    nulos_pct=(nulos/total_filas*100).round(1)
    nulos_df=pd.DataFrame({"Columna":nulos.index,"Nulos":nulos.values,"Porcentaje":nulos_pct.values})
    nulos_df=nulos_df[nulos_df["Nulos"]>0].reset_index(drop=True)
    n_dup=df_work.duplicated(subset=[c for c in cols_utiles if c not in ["id_pedido","_src"]]).sum()
    fechas_nulas=df_work["fecha_pedido"].isnull().sum() if "fecha_pedido" in df_work.columns else 0
    neg={}
    for c in ["total_venta","margen_bruto","cantidad","precio_unitario","costo_produccion"]:
        if c in df_work.columns:
            n=(df_work[c]<0).sum()
            if n>0: neg[c]=n

    d1,d2,d3,d4=st.columns(4)
    with d1: st.metric("Total registros",f"{total_filas:,}")
    with d2: st.metric("Columnas con nulos",f"{len(nulos_df)}")
    with d3: st.metric("Filas duplicadas",f"{n_dup}")
    with d4: st.metric("Fechas invalidas",f"{fechas_nulas}")

    if len(nulos_df)>0:
        st.markdown(f"<h4 style='color:{N(GOLD)};'>Columnas con valores nulos</h4>", unsafe_allow_html=True)
        nulos_df["Porcentaje"]=nulos_df["Porcentaje"].apply(lambda x:f"{x:.1f}%")
        st.dataframe(nulos_df,use_container_width=True,hide_index=True)
    else:
        st.markdown(info_box("No se encontraron valores nulos en el dataset.",TEAL),unsafe_allow_html=True)

    if neg:
        st.markdown(f"<p style='color:{N(RED)};font-size:12px;font-weight:bold;'>Valores negativos detectados:</p>", unsafe_allow_html=True)
        for col,cnt in neg.items(): st.markdown(f"<span style='color:{N(GRAY)};font-size:12px;'>- {col}: {cnt} fila(s)</span>", unsafe_allow_html=True)

    # 2. Opciones
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>2. Opciones de limpieza</h4>", unsafe_allow_html=True)
    lo1,lo2=st.columns(2)
    with lo1:
        op_dup=st.checkbox("Eliminar filas duplicadas",value=True,key="op_dup")
        op_nulos_fecha=st.checkbox("Eliminar filas sin fecha valida",value=True,key="op_nf")
        op_negativos=st.checkbox("Eliminar filas con ventas o cantidades negativas",value=True,key="op_neg")
        op_strip=st.checkbox("Estandarizar mayusculas/minusculas en texto",value=True,key="op_strip")
    with lo2:
        op_nulos_ref=st.checkbox("Rellenar referencias vacias con Sin Referencia",value=False,key="op_nr")
        op_nulos_canal=st.checkbox("Rellenar canal vacio con Presencial",value=False,key="op_nc")
        op_nulos_estado=st.checkbox("Rellenar estado vacio con Pendiente",value=False,key="op_ne")
        op_calcular_margen=st.checkbox("Recalcular margen_bruto donde sea 0",value=False,key="op_marg")

    # 3. Filtros de rango
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>3. Filtros de rango de valores</h4>", unsafe_allow_html=True)
    fr1,fr2,fr3=st.columns(3)
    with fr1:
        if "total_venta" in df_work.columns and df_work["total_venta"].max()>0:
            min_v=float(df_work["total_venta"].min()); max_v=float(df_work["total_venta"].max())
            rango_venta=st.slider("Rango total_venta (COP)",min_value=min_v,max_value=max_v,value=(min_v,max_v),step=1000.0,key="rv",format="$%.0f")
        else: rango_venta=None
    with fr2:
        if "cantidad" in df_work.columns:
            min_c=int(df_work["cantidad"].min()); max_c=int(df_work["cantidad"].max())
            rango_cant=st.slider("Rango cantidad",min_value=min_c,max_value=max_c,value=(min_c,max_c),step=1,key="rc") if min_c<max_c else (min_c,max_c)
        else: rango_cant=None
    with fr3:
        if "fecha_pedido" in df_work.columns:
            fv=df_work["fecha_pedido"].dropna()
            if len(fv)>0:
                f_min=fv.min().date(); f_max=fv.max().date()
                rango_fecha=st.date_input("Rango de fechas",value=(f_min,f_max),min_value=f_min,max_value=f_max,key="rfd")
            else: rango_fecha=None
        else: rango_fecha=None

    # 4. Correccion valores categoricos
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>4. Correccion de valores en columnas clave</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>Edita los valores unicos detectados para corregir errores de escritura.</p>", unsafe_allow_html=True)
    mapeos={}
    cols_cat=[c for c in ["canal","categoria","estado_pedido","metodo_pago","talla"] if c in df_work.columns]
    for col in cols_cat:
        valores=sorted(df_work[col].dropna().unique().tolist())
        if not valores: continue
        with st.expander(f"Columna: {col} — {len(valores)} valor(es) unico(s)",expanded=False):
            mapa_col={}
            cols_edit=st.columns(min(3,len(valores)))
            for i,val in enumerate(valores):
                with cols_edit[i%len(cols_edit)]:
                    nuevo=st.text_input(f'"{val}"',value=str(val),key=f"remap_{col}_{i}")
                    if nuevo!=str(val): mapa_col[val]=nuevo
            if mapa_col: mapeos[col]=mapa_col

    # 5. Aplicar — ahora sobre TODOS los archivos
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>5. Aplicar y revisar</h4>", unsafe_allow_html=True)
    st.markdown(
        info_box(
            f"Las transformaciones se aplicaran sobre el dataset unificado de <b>{len(noms)} archivo(s)</b> "
            f"({total_filas:,} registros en total). Luego podras descargar el CSV unificado o un ZIP con cada archivo por separado.",
            TEAL
        ),
        unsafe_allow_html=True
    )

    if st.button("Aplicar todas las transformaciones",key="btn_limpiar"):
        opciones = dict(
            op_dup=op_dup,
            op_nulos_fecha=op_nulos_fecha,
            op_negativos=op_negativos,
            op_strip=op_strip,
            op_nulos_ref=op_nulos_ref,
            op_nulos_canal=op_nulos_canal,
            op_nulos_estado=op_nulos_estado,
            op_calcular_margen=op_calcular_margen,
            rango_venta=rango_venta,
            rango_cant=rango_cant,
            rango_fecha=rango_fecha if (rango_fecha and len(rango_fecha)==2) else None,
            mapeos=mapeos,
        )

        # ── Limpiar TODO el dataset unificado ──
        df_clean_total, log_total = aplicar_limpieza(df_work, opciones)

        # ── Limpiar CADA archivo por separado (para descargas individuales) ──
        dfs_por_archivo = {}
        for nom in noms:
            sub = df_work[df_work["_src"]==nom].copy()
            sub_clean, _ = aplicar_limpieza(sub, opciones)
            dfs_por_archivo[nom] = sub_clean

        st.session_state["df_limpio"] = df_clean_total
        st.session_state["dfs_por_archivo"] = dfs_por_archivo
        st.session_state["limpieza_aprobada"] = False

        st.success(
            f"Transformaciones aplicadas sobre {len(noms)} archivo(s). "
            f"Resultado unificado: {len(df_clean_total):,} filas."
        )
        for msg in log_total:
            st.markdown(info_box(msg, TEAL), unsafe_allow_html=True)

        # Resumen por archivo tras limpieza
        st.markdown(f"<h4 style='color:{N(GOLD)};'>Resultado por archivo</h4>", unsafe_allow_html=True)
        cols_res = st.columns(min(len(noms), 4))
        for i, nom in enumerate(noms):
            orig = df_work[df_work["_src"]==nom]
            clean = dfs_por_archivo[nom]
            diff = len(orig) - len(clean)
            with cols_res[i % len(cols_res)]:
                st.markdown(
                    f"<div style='background:{N(NAVY_M)};border-radius:8px;padding:10px 14px;"
                    f"border-left:3px solid {N(GREEN)};margin-bottom:8px;'>"
                    f"<div style='color:{N(GOLD)};font-size:11px;font-weight:bold;'>{nom}</div>"
                    f"<div style='color:{N(WHITE)};font-size:13px;'>{len(orig):,} → <b>{len(clean):,}</b> filas</div>"
                    f"<div style='color:{N(RED) if diff>0 else N(GREEN)};font-size:11px;'>"
                    f"{'−'+str(diff)+' eliminadas' if diff>0 else 'Sin cambios'}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    df_preview=st.session_state.get("df_limpio")
    dfs_por_arch_preview = st.session_state.get("dfs_por_archivo", {})

    if df_preview is not None:
        st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{N(GOLD)};'>Vista previa — datos transformados ({len(df_preview):,} filas totales)</h4>", unsafe_allow_html=True)
        bc1,bc2,bc3=st.columns(3)
        with bc1: st.metric("Filas originales",f"{len(df_base):,}")
        with bc2: st.metric("Filas limpias",f"{len(df_preview):,}")
        with bc3:
            diff=len(df_base)-len(df_preview); st.metric("Filas eliminadas",f"{diff}",delta=f"-{diff}" if diff else "0",delta_color="inverse")
        cm_prev=[c for c in df_preview.columns if not c.startswith("_")]
        st.dataframe(df_preview[cm_prev].head(20).reset_index(drop=True),use_container_width=True,hide_index=True)
        with st.expander("Estadisticas descriptivas del dataset limpio",expanded=False):
            num_cols=df_preview.select_dtypes(include=[np.number]).columns.tolist()
            num_cols=[c for c in num_cols if not c.startswith("_") and c not in ["mes","semana","dias_retraso"]]
            if num_cols: st.dataframe(df_preview[num_cols].describe().round(0),use_container_width=True)

        # ── DESCARGAS ──
        st.markdown(f"<h4 style='color:{N(GOLD)};margin-top:16px;'>Descargar datos limpios</h4>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)

        with dl1:
            # CSV unificado
            csv_limpio=df_preview[cm_prev].to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")
            b64_limpio=base64.b64encode(csv_limpio).decode()
            nombre_unif = f"katrina_datos_limpios_{datetime.now().strftime('%Y%m%d')}.csv"
            st.markdown(
                f'<a href="data:file/csv;base64,{b64_limpio}" download="{nombre_unif}" '
                f'style="display:block;background-color:{N(TEAL)};color:{N(WHITE)};padding:9px 20px;'
                f'border-radius:6px;text-decoration:none;font-weight:bold;font-size:13px;'
                f'text-align:center;margin-bottom:6px;">⬇ Descargar CSV unificado ({len(df_preview):,} filas)</a>',
                unsafe_allow_html=True
            )

        with dl2:
            # ZIP con un CSV por archivo original
            if dfs_por_arch_preview:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for nom, df_sub in dfs_por_arch_preview.items():
                        cm_sub = [c for c in df_sub.columns if not c.startswith("_")]
                        csv_sub = df_sub[cm_sub].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        nombre_csv = nom.replace(".xlsx","").replace(".xls","").replace(".csv","")
                        zf.writestr(f"{nombre_csv}_limpio.csv", csv_sub)
                zip_buffer.seek(0)
                b64_zip = base64.b64encode(zip_buffer.read()).decode()
                nombre_zip = f"katrina_archivos_limpios_{datetime.now().strftime('%Y%m%d')}.zip"
                st.markdown(
                    f'<a href="data:application/zip;base64,{b64_zip}" download="{nombre_zip}" '
                    f'style="display:block;background-color:{N(PURPLE)};color:{N(WHITE)};padding:9px 20px;'
                    f'border-radius:6px;text-decoration:none;font-weight:bold;font-size:13px;'
                    f'text-align:center;margin-bottom:6px;">⬇ Descargar ZIP ({len(dfs_por_arch_preview)} archivos separados)</a>',
                    unsafe_allow_html=True
                )

        # Descarga individual por archivo
        if dfs_por_arch_preview and len(dfs_por_arch_preview) > 1:
            with st.expander(f"Descargar archivo limpio individual ({len(dfs_por_arch_preview)} disponibles)", expanded=False):
                for nom, df_sub in dfs_por_arch_preview.items():
                    cm_sub = [c for c in df_sub.columns if not c.startswith("_")]
                    csv_sub = df_sub[cm_sub].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    b64_sub = base64.b64encode(csv_sub).decode()
                    nombre_ind = nom.replace(".xlsx","").replace(".xls","").replace(".csv","") + "_limpio.csv"
                    col_nom, col_btn = st.columns([3,2])
                    with col_nom:
                        st.markdown(f"<span style='color:{N(GRAY)};font-size:12px;'>{nom} → <b style='color:{N(WHITE)};'>{len(df_sub):,} filas</b></span>", unsafe_allow_html=True)
                    with col_btn:
                        st.markdown(
                            f'<a href="data:file/csv;base64,{b64_sub}" download="{nombre_ind}" '
                            f'style="display:inline-block;background-color:{N(NAVY_L)};color:{N(GOLD)};padding:5px 14px;'
                            f'border-radius:6px;text-decoration:none;font-size:12px;font-weight:bold;'
                            f'border:1px solid {N(GOLD)};">⬇ Descargar</a>',
                            unsafe_allow_html=True
                        )

        st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:16px 0;'>", unsafe_allow_html=True)
        if not st.session_state["limpieza_aprobada"]:
            st.markdown(info_box("Cuando estes conforme con los datos limpios, aprueba para activar el dashboard completo.",GOLD),unsafe_allow_html=True)
            if st.button("Aprobar datos limpios y activar el dashboard",key="btn_aprobar"):
                st.session_state["limpieza_aprobada"]=True
                st.success("Datos aprobados. Navega a las otras pestanas para ver el analisis completo.")
        else:
            st.success("Datos aprobados. El dashboard esta activo con los datos limpios.")
            if st.button("Resetear aprobacion y volver a limpiar",key="btn_reset"):
                st.session_state["limpieza_aprobada"]=False; st.session_state["df_limpio"]=None
                st.session_state["dfs_por_archivo"]={}; st.rerun()
    else:
        st.markdown(info_box("Configura las opciones y haz clic en Aplicar todas las transformaciones.",GRAY),unsafe_allow_html=True)

# ── Elegir dataset activo ──
if st.session_state.get("limpieza_aprobada") and st.session_state.get("df_limpio") is not None:
    df=prep(st.session_state["df_limpio"].copy()); banner_activo=True
else:
    df=df_base.copy(); banner_activo=False

estado_html=(
    f"<div style='background:#0D2E1A;border-left:3px solid {N(GREEN)};border-radius:6px;padding:8px 14px;font-size:12px;color:{N(WHITE)};margin-bottom:12px;'>Dashboard activo con datos limpios y aprobados ({len(df):,} registros).</div>"
    if banner_activo else
    f"<div style='background:{N(NAVY_M)};border-left:3px solid {N(GOLD)};border-radius:6px;padding:8px 14px;font-size:12px;color:{N(GRAY)};margin-bottom:12px;'>Mostrando datos sin limpiar ({len(df):,} registros). Ve a Limpieza y Transformacion para aprobar los datos limpios.</div>"
)

df_e=df[df["estado_pedido"]=="Entregado"].copy()

def filtros_tab(key_prefix):
    fc1,fc2,fc3,fc4=st.columns(4)
    with fc1: cats=["Todas"]+sorted(df["categoria"].dropna().unique().tolist()) if "categoria" in df.columns else ["Todas"]; cat_s=st.selectbox("Categoria",cats,key=f"cf_{key_prefix}")
    with fc2: cans=["Todos"]+sorted(df["canal"].dropna().unique().tolist()) if "canal" in df.columns else ["Todos"]; can_s=st.selectbox("Canal",cans,key=f"cf2_{key_prefix}")
    with fc3:
        if "mes_nombre" in df.columns:
            md=sorted(df["mes_nombre"].dropna().unique().tolist(),key=lambda x:pd.to_datetime(x,format="%b %Y")); ms_s=st.multiselect("Meses",md,default=md,key=f"cf3_{key_prefix}")
        else: ms_s=[]
    with fc4: ts=["Todas"]+sorted(df["talla"].dropna().unique().tolist()) if "talla" in df.columns else ["Todas"]; t_s=st.selectbox("Talla",ts,key=f"cf4_{key_prefix}")
    dff=df.copy(); dffe=df_e.copy()
    if cat_s!="Todas": dff=dff[dff["categoria"]==cat_s]; dffe=dffe[dffe["categoria"]==cat_s]
    if can_s!="Todos": dff=dff[dff["canal"]==can_s]; dffe=dffe[dffe["canal"]==can_s]
    if ms_s: dff=dff[dff["mes_nombre"].isin(ms_s)]; dffe=dffe[dffe["mes_nombre"].isin(ms_s)]
    if t_s!="Todas": dff=dff[dff["talla"]==t_s]; dffe=dffe[dffe["talla"]==t_s]
    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:10px 0 14px;'>", unsafe_allow_html=True)
    return dff,dffe

# ══════════════════
# TAB 1 — RESUMEN
# ══════════════════
with T1:
    st.markdown(estado_html,unsafe_allow_html=True)
    df_f,df_fe=filtros_tab("t1")
    tv=df_fe["total_venta"].sum(); tm=df_fe["margen_bruto"].sum()
    tp=len(df_f); te=len(df_f[df_f["estado_pedido"]=="Entregado"]); tc=len(df_f[df_f["estado_pedido"]=="Cancelado"])
    k1,k2,k3,k4,k5,k6=st.columns(6)
    with k1: st.metric("Ventas totales",fmt(tv))
    with k2: st.metric("Margen bruto",fmt(tm))
    with k3: st.metric("Margen %",f"{pct(tm,tv):.1f}%")
    with k4: st.metric("Pedidos",f"{tp:,}")
    with k5: st.metric("Tasa entrega",f"{pct(te,tp):.1f}%",delta=f"{tc} cancelados" if tc else None,delta_color="inverse")
    with k6: st.metric("Ticket prom.",fmt(tv/te if te else 0))
    st.markdown("<br>",unsafe_allow_html=True)
    ci,cd=st.columns([3,2])
    with ci:
        if "mes_nombre" in df_fe.columns and not df_fe.empty:
            om=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
            vxm=df_fe.groupby("mes_nombre",observed=True).agg(v=("total_venta","sum"),m=("margen_bruto","sum"),p=("total_venta","count")).reindex(om).reset_index()
            fig=go.Figure()
            fig.add_trace(go.Bar(x=vxm["mes_nombre"],y=vxm["v"],name="Ventas",marker_color=N(GOLD),text=[fmt(v) for v in vxm["v"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
            fig.add_trace(go.Bar(x=vxm["mes_nombre"],y=vxm["m"],name="Margen",marker_color="#5B9BD5"))
            fig.add_trace(go.Scatter(x=vxm["mes_nombre"],y=vxm["p"],name="Pedidos",mode="lines+markers",marker=dict(color=N(GREEN),size=8),line=dict(color=N(GREEN),width=2),yaxis="y2"))
            fig.update_layout(barmode="group",yaxis2=dict(overlaying="y",side="right",showgrid=False,tickfont=dict(color=N(GREEN))))
            fl(fig,"Ventas y Margen por Mes",360); st.plotly_chart(fig,use_container_width=True)
    with cd:
        if "estado_pedido" in df_f.columns and not df_f.empty:
            ec=df_f["estado_pedido"].value_counts().reset_index(); ec.columns=["E","C"]
            ce={"Entregado":N(GREEN),"Pendiente":N(GOLD),"Cancelado":N(RED)}
            fd=go.Figure(go.Pie(labels=ec["E"],values=ec["C"],hole=0.55,marker_colors=[ce.get(e,N(GRAY)) for e in ec["E"]],textfont=dict(color=N(WHITE),size=11)))
            fd.add_annotation(text=f"<b>{tp}</b><br><span style='font-size:10px'>pedidos</span>",x=0.5,y=0.5,showarrow=False,font=dict(size=18,color=N(WHITE)))
            fl(fd,"Estado de Pedidos",360); st.plotly_chart(fd,use_container_width=True)
    if "mes_nombre" in df_fe.columns and not df_fe.empty:
        st.markdown(f"<h4 style='color:{N(GOLD)};'>Resumen mensual</h4>",unsafe_allow_html=True)
        om=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
        rm=df_fe.groupby("mes_nombre",observed=True).agg(V=("total_venta","sum"),M=("margen_bruto","sum"),P=("total_venta","count"),T=("total_venta","mean")).reindex(om).reset_index()
        rm.columns=["Mes","Ventas (COP)","Margen (COP)","Pedidos","Ticket Prom (COP)"]
        rm["Margen %"]=(rm["Margen (COP)"]/rm["Ventas (COP)"]*100).round(1).apply(lambda x:f"{x:.1f}%")
        for c in ["Ventas (COP)","Margen (COP)","Ticket Prom (COP)"]: rm[c]=rm[c].apply(fmt)
        st.dataframe(rm,use_container_width=True,hide_index=True)

# ══════════════════
# TAB 2 — PRODUCTOS
# ══════════════════
with T2:
    st.markdown(estado_html,unsafe_allow_html=True)
    df_f,df_fe=filtros_tab("t2")
    if df_fe.empty: st.warning("Sin datos.")
    else:
        c1,c2=st.columns(2)
        with c1:
            if "referencia" in df_fe.columns:
                tr=df_fe.groupby("referencia",observed=True).agg(v=("total_venta","sum"),u=("cantidad","sum"),m=("margen_bruto","sum")).sort_values("v",ascending=True).tail(9).reset_index()
                fr=go.Figure(go.Bar(x=tr["v"],y=tr["referencia"],orientation="h",marker_color=N(GOLD),text=[fmt(v) for v in tr["v"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
                fl(fr,"Ventas por Referencia (COP)",360); st.plotly_chart(fr,use_container_width=True)
        with c2:
            if "categoria" in df_fe.columns:
                cv=df_fe.groupby("categoria",observed=True)["total_venta"].sum().reset_index()
                fc2v=go.Figure(go.Pie(labels=cv["categoria"],values=cv["total_venta"],hole=0.5,marker_colors=[N(GOLD),"#5B9BD5",N(GREEN)],textfont=dict(color=N(WHITE),size=12)))
                fl(fc2v,"Ventas por Categoria",360); st.plotly_chart(fc2v,use_container_width=True)
        c3,c4=st.columns(2)
        with c3:
            if "referencia" in df_fe.columns:
                mr=df_fe.groupby("referencia",observed=True).agg(m=("margen_bruto","sum"),v=("total_venta","sum")).reset_index()
                mr["mp"]=(mr["m"]/mr["v"]*100).round(1); mr=mr.sort_values("mp",ascending=False)
                cb=[N(GREEN) if m>=40 else N(GOLD) if m>=30 else N(RED) for m in mr["mp"]]
                fm2=go.Figure(go.Bar(x=mr["referencia"],y=mr["mp"],marker_color=cb,text=[f"{m:.0f}%" for m in mr["mp"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
                fm2.add_hline(y=30,line=dict(color=N(GOLD),dash="dash",width=1.5),annotation_text="Meta 30%",annotation_font=dict(color=N(GOLD),size=10))
                fl(fm2,"Margen % por Referencia",340); fm2.update_xaxes(tickangle=-35); st.plotly_chart(fm2,use_container_width=True)
        with c4:
            if "talla" in df_fe.columns:
                tv2=df_fe.groupby("talla",observed=True)["cantidad"].sum().reset_index().sort_values("talla")
                ft=go.Figure(go.Bar(x=tv2["talla"],y=tv2["cantidad"],marker_color=N(GOLD),text=tv2["cantidad"],textposition="outside",textfont=dict(size=11,color=N(WHITE))))
                fl(ft,"Unidades por Talla",340); st.plotly_chart(ft,use_container_width=True)
        if "referencia" in df_fe.columns:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Rentabilidad por Producto</h4>",unsafe_allow_html=True)
            tp2=df_fe.groupby(["categoria","referencia"],observed=True).agg(U=("cantidad","sum"),V=("total_venta","sum"),M=("margen_bruto","sum")).reset_index()
            tp2["M%"]=(tp2["M"]/tp2["V"]*100).round(1).apply(lambda x:f"{x:.1f}%")
            tp2["V"]=tp2["V"].apply(fmt); tp2["M"]=tp2["M"].apply(fmt)
            tp2.columns=["Categoria","Referencia","Unidades","Ventas","Margen","Margen %"]
            st.dataframe(tp2,use_container_width=True,hide_index=True)

# ════════════════════════
# TAB 3 — CANALES Y PAGOS
# ════════════════════════
with T3:
    st.markdown(estado_html,unsafe_allow_html=True)
    df_f,df_fe=filtros_tab("t3")
    if df_fe.empty: st.warning("Sin datos.")
    else:
        c1,c2=st.columns(2)
        with c1:
            if "canal" in df_fe.columns:
                cv=df_fe.groupby("canal",observed=True).agg(v=("total_venta","sum"),p=("total_venta","count")).sort_values("v",ascending=False).reset_index()
                fc3v=go.Figure(go.Bar(x=cv["canal"],y=cv["v"],marker_color=[N(GOLD),"#5B9BD5",N(GREEN),N(RED)],text=[fmt(v) for v in cv["v"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
                fl(fc3v,"Ventas por Canal (COP)",340); st.plotly_chart(fc3v,use_container_width=True)
        with c2:
            if "metodo_pago" in df_fe.columns:
                pv=df_fe.groupby("metodo_pago",observed=True)["total_venta"].sum().reset_index()
                fp3=go.Figure(go.Pie(labels=pv["metodo_pago"],values=pv["total_venta"],hole=0.5,marker_colors=[N(GOLD),"#5B9BD5",N(GREEN),"#AB47BC"],textfont=dict(color=N(WHITE),size=12)))
                fl(fp3,"Metodo de Pago",340); st.plotly_chart(fp3,use_container_width=True)
        if "canal" in df_fe.columns and "mes_nombre" in df_fe.columns:
            om=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
            cm=df_fe.groupby(["mes_nombre","canal"],observed=True)["total_venta"].sum().unstack(fill_value=0).reindex(om)
            fe3=go.Figure()
            for i,c in enumerate(cm.columns): fe3.add_trace(go.Scatter(x=cm.index,y=cm[c],name=c,mode="lines+markers",line=dict(color=SERIES[i%len(SERIES)],width=2),marker=dict(size=7)))
            fl(fe3,"Evolucion Ventas por Canal",320); st.plotly_chart(fe3,use_container_width=True)
        if "canal" in df_fe.columns:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Detalle por Canal</h4>",unsafe_allow_html=True)
            cd3=df_fe.groupby("canal",observed=True).agg(P=("total_venta","count"),V=("total_venta","sum"),M=("margen_bruto","sum"),T=("total_venta","mean")).reset_index()
            cd3["Part%"]=(cd3["V"]/cd3["V"].sum()*100).round(1).apply(lambda x:f"{x:.1f}%")
            for c in ["V","M","T"]: cd3[c]=cd3[c].apply(fmt)
            cd3.columns=["Canal","Pedidos","Ventas","Margen","Ticket Prom","Participacion %"]
            st.dataframe(cd3,use_container_width=True,hide_index=True)

# ════════════════════
# TAB 4 — OPERACIONES
# ════════════════════
with T4:
    st.markdown(estado_html,unsafe_allow_html=True)
    df_f,df_fe=filtros_tab("t4")
    c1,c2=st.columns(2)
    with c1:
        if "fecha_pedido" in df_f.columns and not df_f.empty:
            df_f["ds"]=df_f["fecha_pedido"].dt.day_name()
            od=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            de={"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miercoles","Thursday":"Jueves","Friday":"Viernes","Saturday":"Sabado","Sunday":"Domingo"}
            dv=df_f.groupby("ds")["id_pedido"].count().reindex(od).fillna(0).reset_index()
            dv["d"]=dv["ds"].map(de)
            fd4=go.Figure(go.Bar(x=dv["d"],y=dv["id_pedido"],marker_color=N(GOLD),text=dv["id_pedido"].astype(int),textposition="outside",textfont=dict(color=N(WHITE),size=10)))
            fl(fd4,"Pedidos por Dia de la Semana",320); st.plotly_chart(fd4,use_container_width=True)
    with c2:
        if "entregado_a_tiempo" in df_fe.columns and not df_fe.empty:
            at=df_fe["entregado_a_tiempo"].sum(); cr=(~df_fe["entregado_a_tiempo"]).sum()
            fp4=go.Figure(go.Pie(labels=["A tiempo","Con retraso"],values=[at,cr],hole=0.55,marker_colors=[N(GREEN),N(RED)],textfont=dict(color=N(WHITE),size=13)))
            pt=pct(at,at+cr)
            fp4.add_annotation(text=f"<b>{pt:.0f}%</b><br><span style='font-size:10px'>a tiempo</span>",x=0.5,y=0.5,showarrow=False,font=dict(size=18,color=N(WHITE)))
            fl(fp4,"Puntualidad de Entregas",320); st.plotly_chart(fp4,use_container_width=True)
    if "dias_retraso" in df_fe.columns and "referencia" in df_fe.columns and not df_fe.empty:
        rr=df_fe.groupby("referencia",observed=True)["dias_retraso"].mean().reset_index().sort_values("dias_retraso",ascending=False)
        rr["dias_retraso"]=rr["dias_retraso"].round(1)
        cr2=[N(RED) if d>1 else N(GOLD) if d>0 else N(GREEN) for d in rr["dias_retraso"]]
        fr2=go.Figure(go.Bar(x=rr["referencia"],y=rr["dias_retraso"],marker_color=cr2,text=[f"{d:.1f}d" for d in rr["dias_retraso"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
        fr2.add_hline(y=0,line=dict(color=N(GREEN),dash="dash",width=1.5))
        fl(fr2,"Retraso Promedio por Referencia (dias)",300); fr2.update_xaxes(tickangle=-35); st.plotly_chart(fr2,use_container_width=True)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>Indicadores Operacionales</h4>",unsafe_allow_html=True)
    ko1,ko2,ko3,ko4=st.columns(4)
    with ko1:
        if "dias_retraso" in df_fe.columns and not df_fe.empty: st.metric("Retraso prom.",f"{df_fe['dias_retraso'].mean():.1f} dias")
    with ko2:
        if "entregado_a_tiempo" in df_fe.columns and not df_fe.empty: st.metric("A tiempo",f"{pct(df_fe['entregado_a_tiempo'].sum(),len(df_fe)):.0f}%")
    with ko3: st.metric("Cancelados",f"{len(df_f[df_f['estado_pedido']=='Cancelado'])}")
    with ko4:
        if "cantidad" in df_fe.columns and not df_fe.empty: st.metric("Unid./pedido",f"{df_fe['cantidad'].mean():.1f}")

# ════════════════════════════════════════
# TAB 5 — IA PREDICTIVA Y PRESCRIPTIVA
# ════════════════════════════════════════
with T5:
    st.markdown(estado_html,unsafe_allow_html=True)
    df_f,df_fe=filtros_tab("t5")
    if "mes_nombre" not in df_fe.columns or df_fe.empty:
        st.warning("Se necesitan datos con fechas para activar los modelos de IA.")
    else:
        om_ia=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
        nm=len(om_ia)
        st.markdown(f"<div style='background:linear-gradient(135deg,{N(NAVY_M)},{N(NAVY_L)});border-radius:12px;padding:18px 22px;border-left:4px solid {N(PURPLE)};margin-bottom:18px;'><span style='color:{N(PURPLE)};font-size:18px;font-weight:bold;'>Motor de IA — Predictivo y Prescriptivo</span><br><span style='color:{N(GRAY)};font-size:12px;'>Analisis sobre <b style='color:{N(WHITE)};'>{nm} meses</b> de datos — Regresion Polinomial — XGBoost con Optuna — Clasificacion ABC — Clustering K-Means</span></div>",unsafe_allow_html=True)
        ia1,ia2,ia3,ia4=st.tabs(["Prediccion de Ventas","Clasificacion ABC","Segmentacion Clientes","Recomendaciones"])

        with ia1:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Prediccion Avanzada de Ventas</h4>",unsafe_allow_html=True)
            col_mod,col_hor=st.columns(2)
            with col_mod: modelo_pred=st.selectbox("Modelo",["Regresion Polinomial (base)","XGBoost con Optuna"],key="modelo_pred")
            with col_hor: horizonte=st.radio("Meses a predecir",[1,2,3],index=2,horizontal=True,key="horizonte")
            serie=df_fe.groupby("mes_nombre",observed=True)["total_venta"].sum().reindex(om_ia).reset_index()
            serie.columns=["mes","ventas"]; serie["t"]=np.arange(len(serie))
            if modelo_pred=="Regresion Polinomial (base)":
                X=serie["t"].values.reshape(-1,1); y=serie["ventas"].values
                grado=2 if nm>=4 else 1
                modelo=make_pipeline(PolynomialFeatures(grado),LinearRegression()); modelo.fit(X,y)
                ult=pd.to_datetime(om_ia[-1],format="%b %Y")
                mf=[(ult+pd.DateOffset(months=i+1)).strftime("%b %Y") for i in range(horizonte)]
                tf=np.arange(nm,nm+horizonte).reshape(-1,1)
                pf=np.maximum(modelo.predict(tf),0); ic_lo=pf*0.85; ic_hi=pf*1.15
            else:
                df_feat=serie.copy()
                for lag in [1,2,3]: df_feat[f"lag_{lag}"]=df_feat["ventas"].shift(lag)
                for w in [2,3]:
                    df_feat[f"rm_{w}"]=df_feat["ventas"].rolling(w).mean()
                    df_feat[f"rs_{w}"]=df_feat["ventas"].rolling(w).std()
                df_feat=df_feat.dropna().reset_index(drop=True)
                if len(df_feat)<4:
                    st.warning("Se requieren al menos 4 meses para XGBoost."); st.stop()
                feature_cols=[c for c in df_feat.columns if c not in ["mes","ventas","t"]]
                X=df_feat[feature_cols]; y=df_feat["ventas"]
                split=int(0.8*len(X)); X_train,X_test=X[:split],X[split:]; y_train,y_test=y[:split],y[split:]
                def objective(trial):
                    params={"n_estimators":trial.suggest_int("n_estimators",50,200),"max_depth":trial.suggest_int("max_depth",3,8),"learning_rate":trial.suggest_float("learning_rate",0.01,0.2,log=True),"subsample":trial.suggest_float("subsample",0.6,1.0),"colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),"reg_alpha":trial.suggest_float("reg_alpha",0,1),"reg_lambda":trial.suggest_float("reg_lambda",0,1)}
                    m=xgb.XGBRegressor(**params,random_state=42,verbosity=0); m.fit(X_train,y_train); return mean_absolute_error(y_test,m.predict(X_test))
                with st.spinner("Optimizando hiperparametros con Optuna..."):
                    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
                    study.optimize(objective,n_trials=20,show_progress_bar=False)
                best_model=xgb.XGBRegressor(**study.best_params,random_state=42,verbosity=0); best_model.fit(X_train,y_train)
                last_row=df_feat.iloc[-1:][feature_cols].copy(); preds=[]
                for _ in range(horizonte):
                    pred=best_model.predict(last_row)[0]; preds.append(pred)
                    new_row=last_row.copy()
                    for lag in [3,2,1]: new_row[f"lag_{lag}"]=pred if lag==1 else last_row[f"lag_{lag-1}"].values[0]
                    new_row["rm_2"]=(new_row["lag_1"]+new_row["lag_2"])/2; new_row["rs_2"]=abs(new_row["lag_1"]-new_row["lag_2"])/2
                    new_row["rm_3"]=(new_row["lag_1"]+new_row["lag_2"]+new_row["lag_3"])/3; new_row["rs_3"]=np.std([new_row["lag_1"].values[0],new_row["lag_2"].values[0],new_row["lag_3"].values[0]])
                    last_row=new_row
                pf=np.array(preds); ic_lo=pf*0.85; ic_hi=pf*1.15
                ult=pd.to_datetime(om_ia[-1],format="%b %Y"); mf=[(ult+pd.DateOffset(months=i+1)).strftime("%b %Y") for i in range(horizonte)]
                st.success(f"XGBoost optimizado — MAE en validacion: {fmt(study.best_value)}")
            cols_pred=st.columns(horizonte)
            for i,col in enumerate(cols_pred):
                with col: st.metric(mf[i],fmt(pf[i]),delta=f"{((pf[i]/serie['ventas'].iloc[-1])-1)*100:.1f}% vs {om_ia[-1]}" if i==0 else None)
            fig_pred=go.Figure()
            fig_pred.add_trace(go.Scatter(x=serie["mes"],y=serie["ventas"],name="Real",mode="lines+markers",line=dict(color=N(GOLD),width=3),marker=dict(size=9)))
            fig_pred.add_trace(go.Scatter(x=mf,y=pf,name="Prediccion",mode="lines+markers",line=dict(color=N(GREEN),width=3,dash="dash"),marker=dict(size=10,symbol="diamond")))
            fig_pred.add_trace(go.Scatter(x=mf+mf[::-1],y=list(ic_hi)+list(ic_lo[::-1]),fill="toself",fillcolor="rgba(39,174,96,0.12)",line=dict(color="rgba(0,0,0,0)"),name="Intervalo +/-15%"))
            fig_pred.add_vline(x=om_ia[-1],line=dict(color=N(GRAY),dash="dash",width=1))
            fl(fig_pred,f"Prediccion — {modelo_pred} — Proximos {horizonte} meses",420); fig_pred.update_yaxes(tickformat=",.0f",tickprefix="$"); st.plotly_chart(fig_pred,use_container_width=True)
            df_pt=pd.DataFrame({"Mes":mf,"Proyectado":[fmt(v) for v in pf],"Minimo -15%":[fmt(v) for v in ic_lo],"Maximo +15%":[fmt(v) for v in ic_hi],"Delta":[f"{((pf[i]/serie['ventas'].iloc[-1])-1)*100:.1f}%" for i in range(horizonte)]})
            st.dataframe(df_pt,use_container_width=True,hide_index=True)
            sp2=df_fe.groupby("mes_nombre",observed=True).size().reindex(om_ia).reset_index(); sp2.columns=["mes","pedidos"]; sp2["t"]=np.arange(len(sp2))
            model_p=make_pipeline(PolynomialFeatures(2 if nm>=4 else 1),LinearRegression()); model_p.fit(sp2["t"].values.reshape(-1,1),sp2["pedidos"].values)
            pp=np.maximum(model_p.predict(np.arange(nm,nm+horizonte).reshape(-1,1)),0).astype(int)
            st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>Pedidos estimados: {' - '.join([f'{pp[i]} ({mf[i]})' for i in range(horizonte)])}</p>",unsafe_allow_html=True)

        with ia2:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Analisis ABC de Productos</h4>",unsafe_allow_html=True)
            if "referencia" not in df_fe.columns: st.warning("Se necesita la columna referencia.")
            else:
                abc=df_fe.groupby("referencia",observed=True).agg(v=("total_venta","sum"),u=("cantidad","sum"),m=("margen_bruto","sum"),p=("total_venta","count")).reset_index().sort_values("v",ascending=False)
                abc["vacp"]=abc["v"].cumsum()/abc["v"].sum()*100; abc["mp"]=(abc["m"]/abc["v"]*100).round(1)
                abc["clase"]=abc["vacp"].apply(lambda x:"A" if x<=70 else("B" if x<=90 else "C"))
                ca={"A":N(GREEN),"B":N(GOLD),"C":N(RED)}
                ka,kb,kc=st.columns(3)
                for cl,col in zip(["A","B","C"],[ka,kb,kc]):
                    sub=abc[abc["clase"]==cl]
                    with col: st.markdown(f"<div style='background:{N(NAVY_M)};border-radius:10px;padding:14px;border-top:3px solid {ca[cl]};'><div style='color:{ca[cl]};font-size:22px;font-weight:bold;'>Clase {cl}</div><div style='color:{N(WHITE)};font-size:13px;margin-top:4px;'>{len(sub)} producto(s)</div><div style='color:{N(GRAY)};font-size:11px;'>{fmt(sub['v'].sum())} ({pct(sub['v'].sum(),abc['v'].sum()):.1f}%)</div><div style='color:{N(GRAY)};font-size:10px;'>Margen prom: {sub['mp'].mean():.1f}%</div></div>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                ca1,ca2=st.columns([3,2])
                with ca1:
                    fp2=go.Figure()
                    fp2.add_trace(go.Bar(x=abc["referencia"],y=abc["v"],name="Ventas",marker_color=[ca[c] for c in abc["clase"]],text=[f"Clase {c}" for c in abc["clase"]],textposition="outside",textfont=dict(size=8,color=N(WHITE))))
                    fp2.add_trace(go.Scatter(x=abc["referencia"],y=abc["vacp"],name="% Acumulado",mode="lines+markers",line=dict(color=N(PURPLE),width=2),marker=dict(size=6),yaxis="y2"))
                    fp2.add_hline(y=70,line=dict(color=N(GREEN),dash="dash",width=1),yref="y2"); fp2.add_hline(y=90,line=dict(color=N(GOLD),dash="dash",width=1),yref="y2")
                    fp2.update_layout(yaxis2=dict(overlaying="y",side="right",range=[0,110],ticksuffix="%",showgrid=False,tickfont=dict(color=N(PURPLE))))
                    fl(fp2,"Curva de Pareto",380); fp2.update_xaxes(tickangle=-35); st.plotly_chart(fp2,use_container_width=True)
                with ca2:
                    fs2=go.Figure()
                    for cl in ["A","B","C"]:
                        sub=abc[abc["clase"]==cl]
                        if sub.empty: continue
                        fs2.add_trace(go.Scatter(x=sub["u"],y=sub["mp"],mode="markers+text",name=f"Clase {cl}",marker=dict(size=sub["v"]/sub["v"].max()*30+10,color=ca[cl],opacity=0.85),text=sub["referencia"].str[:12],textposition="top center",textfont=dict(size=8)))
                    fs2.add_hline(y=30,line=dict(color=N(GRAY),dash="dot",width=1))
                    fl(fs2,"Volumen vs Margen %",380); st.plotly_chart(fs2,use_container_width=True)
                at=abc[["referencia","clase","v","u","mp","vacp","p"]].copy()
                at["v"]=at["v"].apply(fmt); at["mp"]=at["mp"].apply(lambda x:f"{x:.1f}%"); at["vacp"]=at["vacp"].apply(lambda x:f"{x:.1f}%")
                at.columns=["Referencia","Clase","Ventas","Unidades","Margen %","% Acumulado","Pedidos"]
                st.dataframe(at,use_container_width=True,hide_index=True)

        with ia3:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Segmentacion de Clientes — K-Means</h4>",unsafe_allow_html=True)
            if "nombre_cliente" not in df_fe.columns: st.warning("Se necesita la columna nombre_cliente.")
            else:
                rfm=df_fe.groupby("nombre_cliente",observed=True).agg(freq=("id_pedido","count"),gasto=("total_venta","sum"),ticket=("total_venta","mean"),margen=("margen_bruto","sum")).reset_index()
                nc=min(3,len(rfm)); sc=StandardScaler(); Xk=sc.fit_transform(rfm[["freq","gasto","ticket"]])
                km=KMeans(n_clusters=nc,random_state=42,n_init=10); rfm["seg_raw"]=km.fit_predict(Xk)
                rfm["grank"]=rfm["gasto"].rank(pct=True)
                rfm["segmento"]=pd.cut(rfm["grank"],bins=[0,0.33,0.66,1.01],labels=["Ocasional","Regular","VIP"])
                cs={"VIP":N(GOLD),"Regular":N(GREEN),"Ocasional":"#5B9BD5"}
                ks1,ks2,ks3=st.columns(3)
                for seg,col in zip(["VIP","Regular","Ocasional"],[ks1,ks2,ks3]):
                    sub=rfm[rfm["segmento"]==seg]
                    with col: st.markdown(f"<div style='background:{N(NAVY_M)};border-radius:10px;padding:14px;border-top:3px solid {cs[seg]};'><div style='color:{cs[seg]};font-size:13px;font-weight:bold;'>{seg}</div><div style='color:{N(WHITE)};font-size:20px;font-weight:bold;margin-top:4px;'>{len(sub)} clientes</div><div style='color:{N(GRAY)};font-size:11px;'>Gasto prom: {fmt(sub['gasto'].mean())}<br>Pedidos prom: {sub['freq'].mean():.1f}</div></div>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                cs1,cs2=st.columns(2)
                with cs1:
                    fsg=go.Figure()
                    for seg in rfm["segmento"].cat.categories:
                        sub=rfm[rfm["segmento"]==seg]
                        if sub.empty: continue
                        fsg.add_trace(go.Scatter(x=sub["freq"],y=sub["gasto"],mode="markers+text",name=str(seg),marker=dict(size=sub["ticket"]/sub["ticket"].max()*25+8,color=cs.get(str(seg),N(GRAY)),opacity=0.8),text=sub["nombre_cliente"].str.split().str[0],textposition="top center",textfont=dict(size=7.5)))
                    fl(fsg,"Frecuencia vs Gasto Total",400); st.plotly_chart(fsg,use_container_width=True)
                with cs2:
                    sr=rfm.groupby("segmento",observed=True).agg(cl=("nombre_cliente","count"),gt=("gasto","sum")).reset_index()
                    fsp=go.Figure(go.Pie(labels=sr["segmento"].astype(str),values=sr["gt"],hole=0.5,marker_colors=[cs.get(str(s),N(GRAY)) for s in sr["segmento"]],textfont=dict(color=N(WHITE),size=11)))
                    fl(fsp,"Participacion por Segmento",400); st.plotly_chart(fsp,use_container_width=True)
                tc=rfm.sort_values("gasto",ascending=False).head(10).copy()
                tc["gasto"]=tc["gasto"].apply(fmt); tc["ticket"]=tc["ticket"].apply(fmt); tc["margen"]=tc["margen"].apply(fmt)
                tc=tc.rename(columns={"nombre_cliente":"Cliente","freq":"Pedidos","segmento":"Segmento","gasto":"Gasto Total","ticket":"Ticket Prom","margen":"Margen Total"})[["Cliente","Segmento","Pedidos","Gasto Total","Ticket Prom","Margen Total"]]
                st.dataframe(tc,use_container_width=True,hide_index=True)

        with ia4:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Motor Prescriptivo — Recomendaciones Automaticas</h4>",unsafe_allow_html=True)
            tv2=df_fe["total_venta"].sum(); tm2=df_fe["margen_bruto"].sum(); mg=pct(tm2,tv2)
            sp3=df_fe.groupby("mes_nombre",observed=True)["total_venta"].sum().reindex(om_ia)
            trend=((sp3.iloc[-1]-sp3.iloc[0])/sp3.iloc[0]*100) if len(sp3)>=2 else 0
            cs3=df_fe.groupby("canal",observed=True)["total_venta"].sum()
            ctop=cs3.idxmax(); cpct=pct(cs3.max(),cs3.sum()); cmin=cs3.idxmin()
            if "referencia" in df_fe.columns:
                pm=df_fe.groupby("referencia",observed=True).apply(lambda x:(x["margen_bruto"].sum()/x["total_venta"].sum()*100)).sort_values(ascending=False)
                pbest=pm.index[0]; pbest_v=pm.iloc[0]; pworst=pm.index[-1]; pworst_v=pm.iloc[-1]
            else: pbest=pworst="N/A"; pbest_v=pworst_v=0
            punt=pct(df_fe["entregado_a_tiempo"].sum(),len(df_fe)) if "entregado_a_tiempo" in df_fe.columns else 100
            ttop=df_fe.groupby("talla",observed=True)["cantidad"].sum().idxmax() if "talla" in df_fe.columns else "M"
            mtop=sp3.idxmax(); mtop_v=sp3.max()
            recs=[]
            if trend>10:    recs.append(("alta",   "Aprovechar momentum de crecimiento",      f"Ventas crecieron <b>{trend:.1f}%</b>. Escalar el canal <b>{ctop}</b> ({cpct:.0f}%): Instagram Shop y Rappi Negocios. ROI estimado: +20% en mes 2."))
            elif trend<-5:  recs.append(("alerta", f"Ventas en descenso ({trend:.1f}%)",       f"Caida de <b>{abs(trend):.1f}%</b>. Contactar top 5 clientes inactivos y ofrecer descuento del 10%."))
            else:           recs.append(("media",  "Ventas estables — Oportunidad de crecer", f"Variacion de {trend:.1f}%. Canal digital aun bajo el 30% objetivo."))
            if mg<35:       recs.append(("alerta", f"Margen bajo ({mg:.1f}%)",                 f"Por debajo del 40%. Producto con menor margen: <b>{pworst}</b> ({pworst_v:.1f}%). Negociar tela (-5%), ajustar PVP o descontinuar."))
            else:           recs.append(("media",  f"Margen saludable ({mg:.1f}%)",            f"Producto mas rentable: <b>{pbest}</b> ({pbest_v:.1f}%). Priorizar produccion y visibilidad."))
            if cpct>60:     recs.append(("alerta", f"Dependencia de {ctop} ({cpct:.0f}%)",     f"Canal con menor participacion: <b>{cmin}</b>. Activar Instagram y WhatsApp Business."))
            else:           recs.append(("media",  "Canales diversificados",                   f"Canal dominante <b>{ctop}</b> con {cpct:.0f}%. Objetivo: 30% ventas digitales."))
            if punt<80:     recs.append(("alerta", f"Puntualidad critica ({punt:.0f}%)",       f"Tablero Notion ($0), alerta WhatsApp a los 3 dias sin avance, +1 dia de holgura en plazos."))
            elif punt>=90:  recs.append(("media",  f"Puntualidad excelente ({punt:.0f}%)",     f"Publicar en Instagram y solicitar resenas de Google a clientes satisfechos."))
            recs.append(("media", f"Planificar temporada alta: {mtop}",          f"Mes de mayor venta: <b>{mtop}</b> ({fmt(float(mtop_v))}). 3 semanas antes: +30% inventario, contratar temporal si >70 pedidos."))
            recs.append(("media", f"Optimizar inventario talla {ttop}",          f"Talla mas demandada: <b>{ttop}</b>. Bundle: 10% descuento en compra de 2 prendas de la misma talla."))
            recs.append(("alta",  "Implementar IA Generativa",                   f"ChatGPT-4o: 20 descripciones en 1h ($0). Canva AI + Remove.bg: fotos profesionales desde celular. Claude API: reporte ejecutivo automatico cada lunes."))
            le1,le2,le3=st.columns(3)
            with le1: st.markdown(f"<span style='color:{N(PURPLE)};font-size:11px;'>Alta prioridad</span>",unsafe_allow_html=True)
            with le2: st.markdown(f"<span style='color:{N(GOLD)};font-size:11px;'>Alerta — atencion inmediata</span>",unsafe_allow_html=True)
            with le3: st.markdown(f"<span style='color:{N(GREEN)};font-size:11px;'>Oportunidad — optimizacion</span>",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            orden={"alta":0,"alerta":1,"media":2}
            for urg,tit,cont in sorted(recs,key=lambda x:orden.get(x[0],3)):
                st.markdown(card_ia("",tit,cont,urg),unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(f"<div style='background:linear-gradient(135deg,{N(NAVY_M)},{N(NAVY_L)});border-radius:12px;padding:18px 22px;border-top:3px solid {N(GOLD)};'><h4 style='color:{N(GOLD)};margin-bottom:12px;'>Resumen Ejecutivo Automatico</h4><p style='color:{N(WHITE)};font-size:13px;line-height:1.8;'>Margen bruto: <b style='color:{N(GOLD)};'>{mg:.1f}%</b>. Tendencia: <b style='color:{'#27AE60' if trend>=0 else '#E74C3C'};'>{'positiva' if trend>=0 else 'negativa'} ({trend:+.1f}%)</b>. Canal principal: <b style='color:{N(GOLD)};'>{ctop}</b> ({cpct:.0f}%). Puntualidad: <b style='color:{'#27AE60' if punt>=85 else '#E74C3C'};'>{punt:.0f}%</b>. Producto mas rentable: <b style='color:{N(GOLD)};'>{pbest}</b> ({pbest_v:.1f}%).</p><p style='color:{N(GRAY)};font-size:11px;margin-top:8px;'>Generado automaticamente — {len(df_fe):,} transacciones — {nm} meses — {datetime.now().strftime('%d %b %Y %H:%M')}</p></div>",unsafe_allow_html=True)

# ══════════════════
# TAB 6 — DATOS
# ══════════════════
with T6:
    st.markdown(estado_html,unsafe_allow_html=True)
    df_f,df_fe=filtros_tab("t6")
    st.markdown(f"<h4 style='color:{N(GOLD)};'>Tabla de datos ({len(df_f):,} registros)</h4>",unsafe_allow_html=True)
    bq=st.text_input("Buscar",placeholder="Escribir para filtrar...",label_visibility="collapsed")
    dm=df_f[df_f.astype(str).apply(lambda c:c.str.contains(bq,case=False)).any(axis=1)] if bq else df_f
    cm=[c for c in dm.columns if not c.startswith("_")]
    st.dataframe(dm[cm].reset_index(drop=True),use_container_width=True,height=400)
    st.download_button("Descargar datos filtrados (CSV)",dm[cm].to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),file_name=f"katrina_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")

st.markdown(f"<div style='text-align:center;margin-top:24px;padding:14px;background:{N(NAVY_M)};border-radius:10px;border-top:2px solid {N(GOLD)};'><span style='color:{N(GOLD)};font-size:12px;font-weight:bold;'>Innovarte Consulting</span><span style='color:{N(GRAY)};font-size:11px;'> — Almacen Fabrica Sacos y Sueteres Katrina — 2025</span><br><span style='color:{N(GRAY)};font-size:10px;'>Dashboard de Inteligencia Comercial — Propuesta E2E Digital</span></div>",unsafe_allow_html=True)
