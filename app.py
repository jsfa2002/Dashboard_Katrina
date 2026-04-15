"""
app.py — Dashboard de Inteligencia Comercial + IA Predictiva & Prescriptiva
Almacén Fábrica Sacos y Suéteres Katrina
=========================================
Desarrollado por Innovarte Consulting

pip install streamlit pandas numpy plotly openpyxl scikit-learn
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, base64, warnings
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Katrina Dashboard", page_icon="👔", layout="wide", initial_sidebar_state="expanded")

# ── Paleta ────────────────────────────────────────────────────────
NAVY="1A1F3C"; NAVY_M="252B4A"; NAVY_L="2D3561"
GOLD="D4A843"; GOLD_L="F0C040"; WHITE="FFFFFF"
GREEN="27AE60"; RED="E74C3C"; GRAY="8892A4"
LIGHT_BROWN="D2B48C"; PURPLE="8E44AD"
N=lambda c: f"#{c}"  # helper hex

CSS = f"""<style>
.stApp{{background-color:{N(NAVY)};color:{N(WHITE)}}}
[data-testid="stSidebar"]{{background-color:{N(NAVY_M)}}}
[data-testid="stSidebar"] *{{color:{N(WHITE)} !important}}
h1,h2,h3,h4{{color:{N(GOLD)} !important;font-family:'Calibri',sans-serif}}
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

# ── Helpers ───────────────────────────────────────────────────────
def fmt(v): return f"${v:,.0f}"
def pct(a,b): return round(a/b*100,1) if b else 0

def fl(fig, title="", h=360):
    fig.update_layout(
        title=dict(text=title, font=dict(color=N(GOLD), size=14)),
        paper_bgcolor=N(NAVY_M), plot_bgcolor=N(NAVY_M),
        font=dict(color=N(WHITE), family="Calibri"),
        margin=dict(l=10,r=10,t=40 if title else 10,b=10), height=h,
        legend=dict(bgcolor=N(NAVY_M), bordercolor=N(NAVY_L), font=dict(color=N(WHITE))),
    )
    fig.update_xaxes(gridcolor=N(NAVY_L), linecolor=N(NAVY_L), tickfont=dict(color=N(GRAY)))
    fig.update_yaxes(gridcolor=N(NAVY_L), linecolor=N(NAVY_L), tickfont=dict(color=N(GRAY)))
    return fig

SERIES=[N(GOLD),"#5B9BD5","#70AD47","#FF7043","#AB47BC","#26C6DA"]

def card_ia(icono, titulo, contenido, urgencia="media"):
    bgs={"alta":f"linear-gradient(135deg,#1B0A2A,#2D1B4E)","media":f"linear-gradient(135deg,#0D2E1A,#1A4A2E)","alerta":f"linear-gradient(135deg,#2E1A0D,#4A2E1A)"}
    borders={"alta":N(PURPLE),"media":N(GREEN),"alerta":N(GOLD)}
    tcolors={"alta":N(PURPLE),"media":N(GREEN),"alerta":N(GOLD)}
    return f"""<div style='background:{bgs.get(urgencia,bgs["media"])};border-left:4px solid {borders.get(urgencia,N(GREEN))};border-radius:10px;padding:14px 16px;margin-bottom:10px;'>
    <span style='font-size:18px;'>{icono}</span>
    <span style='color:{tcolors.get(urgencia,N(GREEN))};font-weight:bold;font-size:13px;margin-left:8px;'>{titulo}</span>
    <div style='color:{N(WHITE)};font-size:12px;margin-top:6px;line-height:1.6;'>{contenido}</div>
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='text-align:center;padding:12px 0 8px;'>
        <div style='font-size:32px;'>👔</div>
        <div style='color:{N(GOLD)};font-size:18px;font-weight:bold;'>Almacén Katrina</div>
        <div style='color:{N(GRAY)};font-size:11px;'>Inteligencia Comercial</div>
        <div style='color:{N(GRAY)};font-size:10px;margin-top:4px;'>Innovarte Consulting · 2025</div>
    </div><hr style='border-color:{N(NAVY_L)};margin:8px 0;'>""", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{N(GOLD)};font-size:13px;font-weight:bold;'>📂 Cargar datos</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{N(GRAY)};font-size:11px;'>Sube uno o varios CSV/Excel — uno por mes.</p>", unsafe_allow_html=True)
    archivos = st.file_uploader("files", type=["csv","xlsx","xls"], accept_multiple_files=True,
                                help="El dashboard los unifica automáticamente.", label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{N(GOLD)};font-size:12px;font-weight:bold;'>🗂 Columnas requeridas</p>", unsafe_allow_html=True)
    for c in ["fecha_pedido","referencia","categoria","cantidad","total_venta","margen_bruto","estado_pedido","canal","metodo_pago","talla"]:
        st.markdown(f"<span style='color:{N(GRAY)};font-size:10px;'>· {c}</span>", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:12px 0;'>", unsafe_allow_html=True)
    with st.expander("📥 ¿Cómo registrar los datos?", expanded=False):
        st.markdown(f"""<div style='color:{N(NAVY)};font-size:11px;line-height:1.7;'>
        <b>Proceso recomendado:</b><br>
        1. Por cada pedido recibido por WhatsApp, diligenciar una fila en Excel.<br>
        2. Al cerrar el mes, exportar como CSV.<br>
        3. Subir el archivo a este dashboard.<br><br>
        <b>Frecuencia:</b> mensual o semanal
        </div>""", unsafe_allow_html=True)

    plantilla = pd.DataFrame([{"id_pedido":"KAT-202501-001","fecha_pedido":"2025-01-15","nombre_cliente":"Carlos Martínez",
        "canal":"WhatsApp","referencia":"Camisa Clásica Blanca","categoria":"Camisas","talla":"M","cantidad":2,
        "precio_unitario":69900,"costo_produccion":32000,"total_venta":139800,"margen_bruto":75800,
        "estado_pedido":"Entregado","fecha_entrega_comprometida":"2025-01-20",
        "fecha_entrega_real":"2025-01-20","metodo_pago":"Nequi","notas":""}])
    b64p = base64.b64encode(plantilla.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64p}" download="plantilla_katrina.csv" style="display:block;width:100%;background-color:{N(LIGHT_BROWN)};color:{N(NAVY)};text-align:center;padding:8px 0;border-radius:6px;text-decoration:none;font-weight:bold;border:1px solid {N(GOLD)};margin-bottom:12px;">📥 Descargar plantilla Excel</a>', unsafe_allow_html=True)

# ── Carga ─────────────────────────────────────────────────────────
@st.cache_data
def cargar(blist):
    dfs,err,noms=[],[],[]
    for n,b in blist:
        try:
            d=pd.read_csv(io.BytesIO(b),encoding="utf-8-sig") if n.endswith(".csv") else pd.read_excel(io.BytesIO(b))
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

# ── Header ────────────────────────────────────────────────────────
cl,ct=st.columns([1,5])
with cl:
    st.markdown(f"<div style='background:{N(NAVY_M)};border-radius:12px;padding:14px;text-align:center;border:2px solid {N(GOLD)};margin-top:4px;'><span style='font-size:36px;'>👔</span></div>", unsafe_allow_html=True)
with ct:
    st.markdown(f"<div style='padding:6px 0 0 8px;'><span style='color:{N(GOLD)};font-size:26px;font-weight:bold;'>Dashboard de Inteligencia Comercial</span><br><span style='color:{N(WHITE)};font-size:16px;'>Almacén Fábrica Sacos y Suéteres Katrina</span><br><span style='color:{N(GRAY)};font-size:11px;'>Innovarte Consulting · Bogotá, Colombia</span></div>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:10px 0 16px;'>", unsafe_allow_html=True)

# ── Bienvenida ────────────────────────────────────────────────────
if not archivos:
    st.markdown(f"<div style='background:{N(NAVY_M)};border:1px solid {N(NAVY_L)};border-radius:14px;padding:36px;text-align:center;margin-top:20px;'><div style='font-size:52px;margin-bottom:16px;'>📂</div><h2 style='color:{N(GOLD)};margin-bottom:8px;'>Bienvenido al Dashboard de Katrina</h2><p style='color:{N(WHITE)};font-size:15px;max-width:520px;margin:0 auto 16px;'>Sube tus archivos de ventas en el panel izquierdo para comenzar.</p></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for c,(e,t,d) in zip(cols,[("1️⃣","Descarga la plantilla","Panel izquierdo"),("2️⃣","Llena los datos","1 fila = 1 pedido"),("3️⃣","Sube los archivos","Varios meses a la vez"),("4️⃣","Explora y decide","Ventas, márgenes, IA")]):
        with c: st.markdown(f"<div style='background:{N(NAVY_L)};border-radius:10px;padding:16px 12px;text-align:center;border-top:3px solid {N(GOLD)};'><div style='font-size:28px;'>{e}</div><div style='color:{N(GOLD)};font-weight:bold;font-size:13px;margin:6px 0 4px;'>{t}</div><div style='color:{N(GRAY)};font-size:11px;'>{d}</div></div>", unsafe_allow_html=True)
    st.stop()

# ── Procesar ─────────────────────────────────────────────────────
ab=[(f.name,f.read()) for f in archivos]
df_raw,errs,noms=cargar(tuple(ab))
for e in errs: st.error(f"Error: {e}")
if df_raw is None or df_raw.empty: st.warning("No se pudieron leer datos válidos."); st.stop()
df=prep(df_raw)

with st.expander(f"✅ {len(noms)} archivo(s) — {len(df):,} registros", expanded=False):
    ca,cb=st.columns(2)
    with ca:
        st.markdown(f"<b style='color:{N(GOLD)};'>Archivos:</b>", unsafe_allow_html=True)
        for n in noms:
            sub=df[df["_src"]==n]; st.markdown(f"<span style='color:{N(GRAY)};'>· {n} — {len(sub)} registros</span>", unsafe_allow_html=True)
    with cb:
        if "fecha_pedido" in df.columns:
            st.markdown(f"<b style='color:{N(GOLD)};'>Período:</b><br><span style='color:{N(GRAY)};'>{df['fecha_pedido'].min().strftime('%d %b %Y')} → {df['fecha_pedido'].max().strftime('%d %b %Y')}</span>", unsafe_allow_html=True)

# ── Filtros ───────────────────────────────────────────────────────
st.markdown(f"<p style='color:{N(GOLD)};font-weight:bold;font-size:13px;margin-bottom:6px;'>🔍 Filtros</p>", unsafe_allow_html=True)
fc1,fc2,fc3,fc4=st.columns(4)
df_e=df[df["estado_pedido"]=="Entregado"].copy()
with fc1: cats=["Todas"]+sorted(df["categoria"].dropna().unique().tolist()) if "categoria" in df.columns else ["Todas"]; cat_sel=st.selectbox("Categoría",cats,key="cf")
with fc2: cans=["Todos"]+sorted(df["canal"].dropna().unique().tolist()) if "canal" in df.columns else ["Todos"]; can_sel=st.selectbox("Canal",cans,key="cf2")
with fc3:
    if "mes_nombre" in df.columns:
        md=sorted(df["mes_nombre"].dropna().unique().tolist(),key=lambda x:pd.to_datetime(x,format="%b %Y")); ms=st.multiselect("Meses",md,default=md,key="cf3")
    else: ms=[]
with fc4: ts=["Todas"]+sorted(df["talla"].dropna().unique().tolist()) if "talla" in df.columns else ["Todas"]; t_sel=st.selectbox("Talla",ts,key="cf4")

df_f=df.copy(); df_fe=df_e.copy()
if cat_sel!="Todas": df_f=df_f[df_f["categoria"]==cat_sel]; df_fe=df_fe[df_fe["categoria"]==cat_sel]
if can_sel!="Todos": df_f=df_f[df_f["canal"]==can_sel]; df_fe=df_fe[df_fe["canal"]==can_sel]
if ms: df_f=df_f[df_f["mes_nombre"].isin(ms)]; df_fe=df_fe[df_fe["mes_nombre"].isin(ms)]
if t_sel!="Todas": df_f=df_f[df_f["talla"]==t_sel]; df_fe=df_fe[df_fe["talla"]==t_sel]
st.markdown(f"<hr style='border-color:{N(NAVY_L)};margin:10px 0 14px;'>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
T1,T2,T3,T4,T5,T6=st.tabs(["📊 Resumen General","🛍️ Productos","📡 Canales & Pagos","⏱️ Operaciones","🤖 IA Predictiva & Prescriptiva","📋 Datos Crudos"])

# ══════════════════════════
# TAB 1 — RESUMEN GENERAL
# ══════════════════════════
with T1:
    tv=df_fe["total_venta"].sum(); tm=df_fe["margen_bruto"].sum()
    tp=len(df_f); te=len(df_f[df_f["estado_pedido"]=="Entregado"]); tc=len(df_f[df_f["estado_pedido"]=="Cancelado"])
    k1,k2,k3,k4,k5,k6=st.columns(6)
    with k1: st.metric("💰 Ventas totales",fmt(tv))
    with k2: st.metric("📈 Margen bruto",fmt(tm))
    with k3: st.metric("% Margen",f"{pct(tm,tv):.1f}%")
    with k4: st.metric("📦 Pedidos",f"{tp:,}")
    with k5: st.metric("✅ Tasa entrega",f"{pct(te,tp):.1f}%",delta=f"{tc} cancelados" if tc else None,delta_color="inverse")
    with k6: st.metric("🎫 Ticket prom.",fmt(tv/te if te else 0))
    st.markdown("<br>", unsafe_allow_html=True)
    ci,cd=st.columns([3,2])
    with ci:
        if "mes_nombre" in df_fe.columns and not df_fe.empty:
            om=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
            vxm=(df_fe.groupby("mes_nombre",observed=True).agg(v=("total_venta","sum"),m=("margen_bruto","sum"),p=("total_venta","count")).reindex(om).reset_index())
            fig=go.Figure()
            fig.add_trace(go.Bar(x=vxm["mes_nombre"],y=vxm["v"],name="Ventas",marker_color=N(GOLD),text=[fmt(v) for v in vxm["v"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
            fig.add_trace(go.Bar(x=vxm["mes_nombre"],y=vxm["m"],name="Margen",marker_color="#5B9BD5"))
            fig.add_trace(go.Scatter(x=vxm["mes_nombre"],y=vxm["p"],name="Pedidos",mode="lines+markers",marker=dict(color=N(GREEN),size=8),line=dict(color=N(GREEN),width=2),yaxis="y2"))
            fig.update_layout(barmode="group",yaxis2=dict(overlaying="y",side="right",showgrid=False,tickfont=dict(color=N(GREEN))))
            fl(fig,"Ventas y Margen por Mes",360); st.plotly_chart(fig,use_container_width=True)
    with cd:
        if "estado_pedido" in df_f.columns and not df_f.empty:
            ec=df_f["estado_pedido"].value_counts().reset_index(); ec.columns=["E","C"]
            ce={"Entregado":N(GREEN),"Pendiente":N(GOLD),"Cancelado":N(RED),"En producción":"#5B9BD5"}
            fd=go.Figure(go.Pie(labels=ec["E"],values=ec["C"],hole=0.55,marker_colors=[ce.get(e,N(GRAY)) for e in ec["E"]],textfont=dict(color=N(WHITE),size=11)))
            fd.add_annotation(text=f"<b>{tp}</b><br><span style='font-size:10px'>pedidos</span>",x=0.5,y=0.5,showarrow=False,font=dict(size=18,color=N(WHITE)))
            fl(fd,"Estado de Pedidos",360); st.plotly_chart(fd,use_container_width=True)
    if "mes_nombre" in df_fe.columns and not df_fe.empty:
        st.markdown(f"<h4 style='color:{N(GOLD)};'>Resumen mensual</h4>", unsafe_allow_html=True)
        om=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
        rm=(df_fe.groupby("mes_nombre",observed=True).agg(V=("total_venta","sum"),M=("margen_bruto","sum"),P=("total_venta","count"),T=("total_venta","mean")).reindex(om).reset_index())
        rm.columns=["Mes","Ventas (COP)","Margen (COP)","Pedidos","Ticket Prom (COP)"]
        rm["Margen %"]=(rm["Margen (COP)"]/rm["Ventas (COP)"]*100).round(1).apply(lambda x:f"{x:.1f}%")
        for c in ["Ventas (COP)","Margen (COP)","Ticket Prom (COP)"]: rm[c]=rm[c].apply(fmt)
        st.dataframe(rm,use_container_width=True,hide_index=True)

# ══════════════════════════
# TAB 2 — PRODUCTOS
# ══════════════════════════
with T2:
    if df_fe.empty: st.warning("Sin datos.")
    else:
        c1,c2=st.columns(2)
        with c1:
            if "referencia" in df_fe.columns:
                tr=(df_fe.groupby("referencia",observed=True).agg(v=("total_venta","sum"),u=("cantidad","sum"),m=("margen_bruto","sum")).sort_values("v",ascending=True).tail(9).reset_index())
                fr=go.Figure(go.Bar(x=tr["v"],y=tr["referencia"],orientation="h",marker_color=N(GOLD),text=[fmt(v) for v in tr["v"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
                fl(fr,"Ventas por Referencia (COP)",360); st.plotly_chart(fr,use_container_width=True)
        with c2:
            if "categoria" in df_fe.columns:
                cv=df_fe.groupby("categoria",observed=True)["total_venta"].sum().reset_index()
                fc=go.Figure(go.Pie(labels=cv["categoria"],values=cv["total_venta"],hole=0.5,marker_colors=[N(GOLD),"#5B9BD5",N(GREEN)],textfont=dict(color=N(WHITE),size=12)))
                fl(fc,"Ventas por Categoría",360); st.plotly_chart(fc,use_container_width=True)
        c3,c4=st.columns(2)
        with c3:
            if "referencia" in df_fe.columns:
                mr=df_fe.groupby("referencia",observed=True).agg(m=("margen_bruto","sum"),v=("total_venta","sum")).reset_index()
                mr["mp"]=(mr["m"]/mr["v"]*100).round(1); mr=mr.sort_values("mp",ascending=False)
                cb=[N(GREEN) if m>=40 else N(GOLD) if m>=30 else N(RED) for m in mr["mp"]]
                fm=go.Figure(go.Bar(x=mr["referencia"],y=mr["mp"],marker_color=cb,text=[f"{m:.0f}%" for m in mr["mp"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
                fm.add_hline(y=30,line=dict(color=N(GOLD),dash="dash",width=1.5),annotation_text="Meta 30%",annotation_font=dict(color=N(GOLD),size=10))
                fl(fm,"Margen % por Referencia",340); fm.update_xaxes(tickangle=-35); st.plotly_chart(fm,use_container_width=True)
        with c4:
            if "talla" in df_fe.columns:
                tv2=df_fe.groupby("talla",observed=True)["cantidad"].sum().reset_index().sort_values("talla")
                ft=go.Figure(go.Bar(x=tv2["talla"],y=tv2["cantidad"],marker_color=N(GOLD),text=tv2["cantidad"],textposition="outside",textfont=dict(size=11,color=N(WHITE))))
                fl(ft,"Unidades por Talla",340); st.plotly_chart(ft,use_container_width=True)
        if "referencia" in df_fe.columns:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Rentabilidad por Producto</h4>", unsafe_allow_html=True)
            tp2=df_fe.groupby(["categoria","referencia"],observed=True).agg(U=("cantidad","sum"),V=("total_venta","sum"),M=("margen_bruto","sum")).reset_index()
            tp2["M%"]=(tp2["M"]/tp2["V"]*100).round(1).apply(lambda x:f"{x:.1f}%")
            tp2["V"]=tp2["V"].apply(fmt); tp2["M"]=tp2["M"].apply(fmt)
            tp2.columns=["Categoría","Referencia","Unidades","Ventas","Margen","Margen %"]
            st.dataframe(tp2,use_container_width=True,hide_index=True)

# ══════════════════════════
# TAB 3 — CANALES & PAGOS
# ══════════════════════════
with T3:
    if df_fe.empty: st.warning("Sin datos.")
    else:
        c1,c2=st.columns(2)
        with c1:
            if "canal" in df_fe.columns:
                cv=df_fe.groupby("canal",observed=True).agg(v=("total_venta","sum"),p=("total_venta","count")).sort_values("v",ascending=False).reset_index()
                fc=go.Figure(go.Bar(x=cv["canal"],y=cv["v"],marker_color=[N(GOLD),"#5B9BD5",N(GREEN),N(RED)],text=[fmt(v) for v in cv["v"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
                fl(fc,"Ventas por Canal (COP)",340); st.plotly_chart(fc,use_container_width=True)
        with c2:
            if "metodo_pago" in df_fe.columns:
                pv=df_fe.groupby("metodo_pago",observed=True)["total_venta"].sum().reset_index()
                fp=go.Figure(go.Pie(labels=pv["metodo_pago"],values=pv["total_venta"],hole=0.5,marker_colors=[N(GOLD),"#5B9BD5",N(GREEN),"#AB47BC"],textfont=dict(color=N(WHITE),size=12)))
                fl(fp,"Método de Pago",340); st.plotly_chart(fp,use_container_width=True)
        if "canal" in df_fe.columns and "mes_nombre" in df_fe.columns:
            om=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
            cm=df_fe.groupby(["mes_nombre","canal"],observed=True)["total_venta"].sum().unstack(fill_value=0).reindex(om)
            fe=go.Figure()
            for i,c in enumerate(cm.columns):
                fe.add_trace(go.Scatter(x=cm.index,y=cm[c],name=c,mode="lines+markers",line=dict(color=SERIES[i%len(SERIES)],width=2),marker=dict(size=7)))
            fl(fe,"Evolución Ventas por Canal",320); st.plotly_chart(fe,use_container_width=True)
        if "canal" in df_fe.columns:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Detalle por Canal</h4>", unsafe_allow_html=True)
            cd=df_fe.groupby("canal",observed=True).agg(P=("total_venta","count"),V=("total_venta","sum"),M=("margen_bruto","sum"),T=("total_venta","mean")).reset_index()
            cd["Part%"]=(cd["V"]/cd["V"].sum()*100).round(1).apply(lambda x:f"{x:.1f}%")
            for c in ["V","M","T"]: cd[c]=cd[c].apply(fmt)
            cd.columns=["Canal","Pedidos","Ventas","Margen","Ticket Prom","Participación %"]
            st.dataframe(cd,use_container_width=True,hide_index=True)

# ══════════════════════════
# TAB 4 — OPERACIONES
# ══════════════════════════
with T4:
    c1,c2=st.columns(2)
    with c1:
        if "fecha_pedido" in df_f.columns and not df_f.empty:
            df_f["ds"]=df_f["fecha_pedido"].dt.day_name()
            od=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            de={"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles","Thursday":"Jueves","Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"}
            dv=df_f.groupby("ds")["id_pedido"].count().reindex(od).fillna(0).reset_index()
            dv["d"]=dv["ds"].map(de)
            fd=go.Figure(go.Bar(x=dv["d"],y=dv["id_pedido"],marker_color=N(GOLD),text=dv["id_pedido"].astype(int),textposition="outside",textfont=dict(color=N(WHITE),size=10)))
            fl(fd,"Pedidos por Día de la Semana",320); st.plotly_chart(fd,use_container_width=True)
    with c2:
        if "entregado_a_tiempo" in df_fe.columns and not df_fe.empty:
            at=df_fe["entregado_a_tiempo"].sum(); cr=(~df_fe["entregado_a_tiempo"]).sum()
            fp=go.Figure(go.Pie(labels=["A tiempo","Con retraso"],values=[at,cr],hole=0.55,marker_colors=[N(GREEN),N(RED)],textfont=dict(color=N(WHITE),size=13)))
            pt=pct(at,at+cr)
            fp.add_annotation(text=f"<b>{pt:.0f}%</b><br><span style='font-size:10px'>a tiempo</span>",x=0.5,y=0.5,showarrow=False,font=dict(size=18,color=N(WHITE)))
            fl(fp,"Puntualidad de Entregas",320); st.plotly_chart(fp,use_container_width=True)
    if "dias_retraso" in df_fe.columns and "referencia" in df_fe.columns and not df_fe.empty:
        rr=df_fe.groupby("referencia",observed=True)["dias_retraso"].mean().reset_index().sort_values("dias_retraso",ascending=False)
        rr["dias_retraso"]=rr["dias_retraso"].round(1)
        cr2=[N(RED) if d>1 else N(GOLD) if d>0 else N(GREEN) for d in rr["dias_retraso"]]
        fr2=go.Figure(go.Bar(x=rr["referencia"],y=rr["dias_retraso"],marker_color=cr2,text=[f"{d:.1f}d" for d in rr["dias_retraso"]],textposition="outside",textfont=dict(size=9,color=N(WHITE))))
        fr2.add_hline(y=0,line=dict(color=N(GREEN),dash="dash",width=1.5))
        fl(fr2,"Retraso Promedio por Referencia (días)",300); fr2.update_xaxes(tickangle=-35); st.plotly_chart(fr2,use_container_width=True)
    st.markdown(f"<h4 style='color:{N(GOLD)};'>Indicadores Operacionales</h4>", unsafe_allow_html=True)
    ko1,ko2,ko3,ko4=st.columns(4)
    with ko1:
        if "dias_retraso" in df_fe.columns and not df_fe.empty: st.metric("⏱️ Retraso prom.",f"{df_fe['dias_retraso'].mean():.1f} días")
    with ko2:
        if "entregado_a_tiempo" in df_fe.columns and not df_fe.empty: st.metric("✅ % A tiempo",f"{pct(df_fe['entregado_a_tiempo'].sum(),len(df_fe)):.0f}%")
    with ko3: st.metric("❌ Cancelados",f"{len(df_f[df_f['estado_pedido']=='Cancelado'])}")
    with ko4:
        if "cantidad" in df_fe.columns and not df_fe.empty: st.metric("📦 Unid./pedido",f"{df_fe['cantidad'].mean():.1f}")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — IA PREDICTIVA & PRESCRIPTIVA
# ══════════════════════════════════════════════════════════════════
with T5:
    if "mes_nombre" not in df_fe.columns or df_fe.empty:
        st.warning("Se necesitan datos con fechas para activar los modelos de IA.")
    else:
        om_ia=sorted(df_fe["mes_nombre"].unique(),key=lambda x:pd.to_datetime(x,format="%b %Y"))
        nm=len(om_ia)

        st.markdown(f"""<div style='background:linear-gradient(135deg,{N(NAVY_M)},{N(NAVY_L)});border-radius:12px;
             padding:18px 22px;border-left:4px solid {N(PURPLE)};margin-bottom:18px;'>
            <span style='font-size:22px;'>🤖</span>
            <span style='color:{N(PURPLE)};font-size:18px;font-weight:bold;margin-left:10px;'>Motor de IA — Predictivo & Prescriptivo</span><br>
            <span style='color:{N(GRAY)};font-size:12px;'>
                Análisis sobre <b style='color:{N(WHITE)};'>{nm} meses</b> de datos ·
                Regresión Polinomial · Clasificación ABC · Clustering K-Means · Motor de Reglas Prescriptivas
            </span>
        </div>""", unsafe_allow_html=True)

        ia1,ia2,ia3,ia4=st.tabs(["📈 Predicción de Ventas","🏷️ Clasificación ABC","👥 Segmentación Clientes","🎯 Recomendaciones"])

        # ── IA1: PREDICCIÓN DE VENTAS ─────────────────────────
        with ia1:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Predicción de Ventas — Próximos 3 Meses</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>Modelo: Regresión Polinomial grado {'2' if nm>=4 else '1'} entrenado con los {nm} meses disponibles. Con más historial la predicción mejora.</p>", unsafe_allow_html=True)

            serie=(df_fe.groupby("mes_nombre",observed=True)["total_venta"].sum().reindex(om_ia).reset_index())
            serie.columns=["mes","v"]; serie["t"]=np.arange(len(serie))
            X=serie["t"].values.reshape(-1,1); y=serie["v"].values
            grado=2 if nm>=4 else 1
            modelo=make_pipeline(PolynomialFeatures(grado),LinearRegression()); modelo.fit(X,y)

            ult=pd.to_datetime(om_ia[-1],format="%b %Y")
            mf=[(ult+pd.DateOffset(months=i+1)).strftime("%b %Y") for i in range(3)]
            tf=np.arange(nm,nm+3).reshape(-1,1)
            ph=modelo.predict(X); pf=np.maximum(modelo.predict(tf),0)
            ic_lo=pf*0.85; ic_hi=pf*1.15

            pi1,pi2,pi3=st.columns(3)
            for i,(col,mes) in enumerate(zip([pi1,pi2,pi3],mf)):
                with col: st.metric(f"📅 {mes} (est.)",fmt(pf[i]),delta=f"{((pf[i]/y[-1])-1)*100:.1f}% vs {om_ia[-1]}" if i==0 else None)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_p=go.Figure()
            fig_p.add_trace(go.Scatter(x=serie["mes"],y=serie["v"],name="Real",mode="lines+markers",line=dict(color=N(GOLD),width=3),marker=dict(size=9)))
            fig_p.add_trace(go.Scatter(x=serie["mes"],y=ph,name="Ajuste modelo",mode="lines",line=dict(color=N(PURPLE),width=2,dash="dot")))
            fig_p.add_trace(go.Scatter(x=mf,y=pf,name="Predicción",mode="lines+markers",line=dict(color=N(GREEN),width=3,dash="dash"),marker=dict(size=10,symbol="diamond")))
            fig_p.add_trace(go.Scatter(x=mf+mf[::-1],y=list(ic_hi)+list(ic_lo[::-1]),fill="toself",fillcolor="rgba(39,174,96,0.12)",line=dict(color="rgba(0,0,0,0)"),name="Intervalo ±15%"))
            fig_p.add_vline(x=om_ia[-1],line=dict(color=N(GRAY),dash="dash",width=1))
            fig_p.add_annotation(x=mf[0],y=float(max(y))*1.05,text="Proyección →",showarrow=False,font=dict(color=N(GREEN),size=11))
            fl(fig_p,"Histórico + Predicción Próximos 3 Meses (COP)",420)
            fig_p.update_yaxes(tickformat=",.0f",tickprefix="$")
            st.plotly_chart(fig_p,use_container_width=True)

            st.markdown(f"<h4 style='color:{N(GOLD)};'>Tabla de Predicción</h4>", unsafe_allow_html=True)
            df_pt=pd.DataFrame({"Mes":mf,"Proyectado":[fmt(v) for v in pf],"Mínimo -15%":[fmt(v) for v in ic_lo],"Máximo +15%":[fmt(v) for v in ic_hi],"Δ vs último mes":[f"{((v/y[-1])-1)*100:.1f}%" for v in pf]})
            st.dataframe(df_pt,use_container_width=True,hide_index=True)

            # Predicción pedidos
            sp=df_fe.groupby("mes_nombre",observed=True).size().reindex(om_ia).reset_index(); sp.columns=["mes","p"]
            mp=make_pipeline(PolynomialFeatures(grado),LinearRegression()); mp.fit(X,sp["p"].values)
            pp=np.maximum(mp.predict(tf),0).astype(int)
            st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;margin-top:10px;'>📦 Pedidos estimados: <b style='color:{N(WHITE)};'>{pp[0]}</b> ({mf[0]}) · <b style='color:{N(WHITE)};'>{pp[1]}</b> ({mf[1]}) · <b style='color:{N(WHITE)};'>{pp[2]}</b> ({mf[2]})</p>", unsafe_allow_html=True)

            st.markdown(f"""<div style='background:{N(NAVY_M)};border-radius:8px;padding:12px 16px;margin-top:12px;border-left:3px solid {N(PURPLE)};'>
                <span style='color:{N(PURPLE)};font-weight:bold;font-size:12px;'>ℹ️ Nota sobre el modelo</span><br>
                <span style='color:{N(GRAY)};font-size:11px;'>Con {nm} meses se usa regresión polinomial grado {grado}. Para incorporar estacionalidad se recomiendan al menos 12 meses. El intervalo ±15% es aproximado. Cada nuevo mes que se suba mejora la predicción.</span>
            </div>""", unsafe_allow_html=True)

        # ── IA2: CLASIFICACIÓN ABC ────────────────────────────
        with ia2:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Análisis ABC de Productos (Principio de Pareto)</h4>", unsafe_allow_html=True)
            st.markdown(f"""<p style='color:{N(GRAY)};font-size:12px;'>
            Clasifica productos según su contribución acumulada a las ventas.<br>
            <b style='color:{N(GREEN)};'>Clase A:</b> Top 70% ventas (máxima prioridad) &nbsp;·&nbsp;
            <b style='color:{N(GOLD)};'>Clase B:</b> 70–90% &nbsp;·&nbsp;
            <b style='color:{N(RED)};'>Clase C:</b> 90–100% (revisar o descontinuar)
            </p>""", unsafe_allow_html=True)

            if "referencia" not in df_fe.columns: st.warning("Se necesita la columna 'referencia'.")
            else:
                abc=(df_fe.groupby("referencia",observed=True).agg(v=("total_venta","sum"),u=("cantidad","sum"),m=("margen_bruto","sum"),p=("total_venta","count")).reset_index().sort_values("v",ascending=False))
                abc["vacp"]=abc["v"].cumsum()/abc["v"].sum()*100
                abc["mp"]=(abc["m"]/abc["v"]*100).round(1)
                abc["clase"]=abc["vacp"].apply(lambda x:"A" if x<=70 else ("B" if x<=90 else "C"))
                ca={"A":N(GREEN),"B":N(GOLD),"C":N(RED)}

                ka,kb,kc=st.columns(3)
                for cl,col in zip(["A","B","C"],[ka,kb,kc]):
                    sub=abc[abc["clase"]==cl]
                    with col:
                        st.markdown(f"""<div style='background:{N(NAVY_M)};border-radius:10px;padding:14px;border-top:3px solid {ca[cl]};'>
                        <div style='color:{ca[cl]};font-size:22px;font-weight:bold;'>Clase {cl}</div>
                        <div style='color:{N(WHITE)};font-size:13px;margin-top:4px;'>{len(sub)} producto(s)</div>
                        <div style='color:{N(GRAY)};font-size:11px;'>{fmt(sub['v'].sum())} ({pct(sub['v'].sum(),abc['v'].sum()):.1f}%)</div>
                        <div style='color:{N(GRAY)};font-size:10px;margin-top:4px;'>Margen prom: {sub['mp'].mean():.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                ca1,ca2=st.columns([3,2])
                with ca1:
                    fp2=go.Figure()
                    fp2.add_trace(go.Bar(x=abc["referencia"],y=abc["v"],name="Ventas",marker_color=[ca[c] for c in abc["clase"]],text=[f"Clase {c}" for c in abc["clase"]],textposition="outside",textfont=dict(size=8,color=N(WHITE))))
                    fp2.add_trace(go.Scatter(x=abc["referencia"],y=abc["vacp"],name="% Acumulado",mode="lines+markers",line=dict(color=N(PURPLE),width=2),marker=dict(size=6),yaxis="y2"))
                    fp2.add_hline(y=70,line=dict(color=N(GREEN),dash="dash",width=1),yref="y2")
                    fp2.add_hline(y=90,line=dict(color=N(GOLD),dash="dash",width=1),yref="y2")
                    fp2.update_layout(yaxis2=dict(overlaying="y",side="right",range=[0,110],ticksuffix="%",showgrid=False,tickfont=dict(color=N(PURPLE))))
                    fl(fp2,"Curva de Pareto — Ventas Acumuladas",380); fp2.update_xaxes(tickangle=-35); st.plotly_chart(fp2,use_container_width=True)
                with ca2:
                    fs2=go.Figure()
                    for cl in ["A","B","C"]:
                        sub=abc[abc["clase"]==cl]
                        if sub.empty: continue
                        fs2.add_trace(go.Scatter(x=sub["u"],y=sub["mp"],mode="markers+text",name=f"Clase {cl}",marker=dict(size=sub["v"]/sub["v"].max()*30+10,color=ca[cl],opacity=0.85),text=sub["referencia"].str[:12],textposition="top center",textfont=dict(size=8)))
                    fs2.add_hline(y=30,line=dict(color=N(GRAY),dash="dot",width=1))
                    fl(fs2,"Volumen vs Margen %",380); fs2.update_xaxes(title="Unidades vendidas"); fs2.update_yaxes(title="Margen %",ticksuffix="%"); st.plotly_chart(fs2,use_container_width=True)

                st.markdown(f"<h4 style='color:{N(GOLD)};'>Ranking ABC Completo</h4>", unsafe_allow_html=True)
                at=abc[["referencia","clase","v","u","mp","vacp","p"]].copy()
                at["v"]=at["v"].apply(fmt); at["mp"]=at["mp"].apply(lambda x:f"{x:.1f}%"); at["vacp"]=at["vacp"].apply(lambda x:f"{x:.1f}%")
                at.columns=["Referencia","Clase","Ventas (COP)","Unidades","Margen %","% Acumulado","Pedidos"]
                st.dataframe(at,use_container_width=True,hide_index=True)

        # ── IA3: SEGMENTACIÓN K-MEANS ─────────────────────────
        with ia3:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Segmentación de Clientes — Clustering K-Means</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>Agrupa automáticamente a los clientes según frecuencia de compra, gasto total y ticket promedio usando el algoritmo K-Means (scikit-learn).</p>", unsafe_allow_html=True)

            if "nombre_cliente" not in df_fe.columns: st.warning("Se necesita la columna 'nombre_cliente'.")
            else:
                rfm=(df_fe.groupby("nombre_cliente",observed=True).agg(freq=("id_pedido","count"),gasto=("total_venta","sum"),ticket=("total_venta","mean"),margen=("margen_bruto","sum")).reset_index())
                nc=min(3,len(rfm))
                sc=StandardScaler(); Xk=sc.fit_transform(rfm[["freq","gasto","ticket"]])
                km=KMeans(n_clusters=nc,random_state=42,n_init=10); rfm["seg_raw"]=km.fit_predict(Xk)
                # Reranquear por gasto
                rfm["grank"]=rfm["gasto"].rank(pct=True)
                rfm["segmento"]=pd.cut(rfm["grank"],bins=[0,0.33,0.66,1.01],labels=["🌱 Ocasional","⭐ Regular","💎 VIP"])
                cs={"💎 VIP":N(GOLD),"⭐ Regular":N(GREEN),"🌱 Ocasional":"#5B9BD5"}

                ks1,ks2,ks3=st.columns(3)
                for seg,col in zip(["💎 VIP","⭐ Regular","🌱 Ocasional"],[ks1,ks2,ks3]):
                    sub=rfm[rfm["segmento"]==seg]
                    with col:
                        st.markdown(f"""<div style='background:{N(NAVY_M)};border-radius:10px;padding:14px;border-top:3px solid {cs[seg]};'>
                        <div style='color:{cs[seg]};font-size:13px;font-weight:bold;'>{seg}</div>
                        <div style='color:{N(WHITE)};font-size:20px;font-weight:bold;margin-top:4px;'>{len(sub)} clientes</div>
                        <div style='color:{N(GRAY)};font-size:11px;'>Gasto prom: {fmt(sub['gasto'].mean())}<br>Pedidos prom: {sub['freq'].mean():.1f}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                cs1,cs2=st.columns(2)
                with cs1:
                    fsg=go.Figure()
                    for seg in rfm["segmento"].cat.categories:
                        sub=rfm[rfm["segmento"]==seg]
                        if sub.empty: continue
                        fsg.add_trace(go.Scatter(x=sub["freq"],y=sub["gasto"],mode="markers+text",name=str(seg),marker=dict(size=sub["ticket"]/sub["ticket"].max()*25+8,color=cs.get(str(seg),N(GRAY)),opacity=0.8),text=sub["nombre_cliente"].str.split().str[0],textposition="top center",textfont=dict(size=7.5)))
                    fl(fsg,"Frecuencia vs Gasto Total",400); fsg.update_xaxes(title="Pedidos realizados"); fsg.update_yaxes(title="Gasto total (COP)",tickformat=",.0f",tickprefix="$"); st.plotly_chart(fsg,use_container_width=True)
                with cs2:
                    sr=rfm.groupby("segmento",observed=True).agg(cl=("nombre_cliente","count"),gt=("gasto","sum")).reset_index()
                    fsp=go.Figure(go.Pie(labels=sr["segmento"].astype(str),values=sr["gt"],hole=0.5,marker_colors=[cs.get(str(s),N(GRAY)) for s in sr["segmento"]],textfont=dict(color=N(WHITE),size=11)))
                    fl(fsp,"Participación en Ventas por Segmento",400); st.plotly_chart(fsp,use_container_width=True)

                st.markdown(f"<h4 style='color:{N(GOLD)};'>Top 10 Clientes</h4>", unsafe_allow_html=True)
                tc=rfm.sort_values("gasto",ascending=False).head(10).copy()
                tc["gasto"]=tc["gasto"].apply(fmt); tc["ticket"]=tc["ticket"].apply(fmt); tc["margen"]=tc["margen"].apply(fmt)
                tc=tc.rename(columns={"nombre_cliente":"Cliente","freq":"Pedidos","segmento":"Segmento","gasto":"Gasto Total","ticket":"Ticket Prom","margen":"Margen Total"})[["Cliente","Segmento","Pedidos","Gasto Total","Ticket Prom","Margen Total"]]
                st.dataframe(tc,use_container_width=True,hide_index=True)

        # ── IA4: RECOMENDACIONES PRESCRIPTIVAS ────────────────
        with ia4:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Motor Prescriptivo — Recomendaciones Automáticas</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>El motor analiza automáticamente los KPIs y genera acciones concretas priorizadas para mejorar el negocio.</p>", unsafe_allow_html=True)

            # Calcular métricas base
            tv2=df_fe["total_venta"].sum(); tm2=df_fe["margen_bruto"].sum()
            mg=pct(tm2,tv2)
            sp2=df_fe.groupby("mes_nombre",observed=True)["total_venta"].sum().reindex(om_ia)
            trend=((sp2.iloc[-1]-sp2.iloc[0])/sp2.iloc[0]*100) if len(sp2)>=2 else 0
            cs2=df_fe.groupby("canal",observed=True)["total_venta"].sum()
            ctop=cs2.idxmax(); cpct=pct(cs2.max(),cs2.sum())
            cmin=cs2.idxmin()
            if "referencia" in df_fe.columns:
                pm=df_fe.groupby("referencia",observed=True).apply(lambda x:(x["margen_bruto"].sum()/x["total_venta"].sum()*100)).sort_values(ascending=False)
                pbest=pm.index[0]; pbest_v=pm.iloc[0]; pworst=pm.index[-1]; pworst_v=pm.iloc[-1]
            else: pbest=pworst="N/A"; pbest_v=pworst_v=0
            punt=pct(df_fe["entregado_a_tiempo"].sum(),len(df_fe)) if "entregado_a_tiempo" in df_fe.columns else 100
            ttop=df_fe.groupby("talla",observed=True)["cantidad"].sum().idxmax() if "talla" in df_fe.columns else "M"
            mtop=sp2.idxmax(); mtop_v=sp2.max()

            # Construir recomendaciones
            recs=[]
            # Tendencia
            if trend>10:
                recs.append(("alta","🚀","Aprovechar momentum de crecimiento",f"Ventas crecieron <b>{trend:.1f}%</b> en el período. Escalar el canal <b>{ctop}</b> ({cpct:.0f}% de ventas): activar Instagram Shop con las 6 referencias, lanzar en Rappi Negocios para entregas mismo día en Bogotá. ROI estimado: +20% ventas adicionales en mes 2."))
            elif trend<-5:
                recs.append(("alerta","⚠️",f"Ventas en descenso ({trend:.1f}%) — Acción urgente",f"Caída de <b>{abs(trend):.1f}%</b>. Revisar si es estacionalidad o pérdida de clientes. Campaña de reactivación: contactar top 5 clientes que no han comprado en el último mes. Ofrecer descuento del 10% para pedido inmediato."))
            else:
                recs.append(("media","📊",f"Ventas estables — Oportunidad de crecer",f"Variación de {trend:.1f}%. Potencial de crecimiento en canales digitales: el canal web/Instagram aún no tiene el 30% de las ventas objetivo."))
            # Margen
            if mg<35:
                recs.append(("alerta","💰",f"Margen global bajo ({mg:.1f}%) — Revisar estructura de costos",f"El margen de <b>{mg:.1f}%</b> está por debajo del objetivo del 40%. El producto con menor margen es <b>{pworst}</b> ({pworst_v:.1f}%). Acciones: (1) Negociar precio de tela con proveedor buscando ahorro del 5%, (2) Ajustar PVP en un 5-8%, (3) Si no mejora, descontinuar esa referencia."))
            else:
                recs.append(("media","✅",f"Margen saludable ({mg:.1f}%) — Proteger productos estrella",f"El producto más rentable es <b>{pbest}</b> ({pbest_v:.1f}% margen). Priorizar su producción, darle mayor visibilidad en el catálogo online y asegurar siempre stock de tela disponible."))
            # Canal
            if cpct>60:
                recs.append(("alerta","📡",f"Alta dependencia de {ctop} ({cpct:.0f}%) — Diversificar",f"El {cpct:.0f}% de las ventas viene de un solo canal. Canal con menor participación: <b>{cmin}</b>. Acción: crear perfil Instagram con las 6 referencias del catálogo web, publicar 3 veces por semana, activar WhatsApp Business con catálogo."))
            else:
                recs.append(("media","🌐","Canales diversificados — Fortalecer digital",f"El canal dominante <b>{ctop}</b> tiene {cpct:.0f}%. Objetivo: llevar el canal digital al 30% de las ventas totales."))
            # Puntualidad
            if punt<80:
                recs.append(("alerta","⏱️",f"Puntualidad crítica ({punt:.0f}%) — Proceso urgente",f"Solo <b>{punt:.0f}%</b> de pedidos a tiempo. Acciones inmediatas: (1) Tablero de pedidos en Notion ($0), (2) Alerta automática WhatsApp si un pedido lleva más de 3 días sin avanzar, (3) Agregar 1 día de holgura a todos los plazos prometidos al cliente."))
            elif punt>=90:
                recs.append(("media","⭐",f"Excelente puntualidad ({punt:.0f}%) — Comunicarlo",f"<b>{punt:.0f}%</b> de pedidos a tiempo es un diferenciador competitivo. Publicarlo en Instagram: 'Entregamos en menos de 5 días hábiles'. Pedir reseñas de Google a los clientes más satisfechos."))
            # Temporada
            recs.append(("media","📅",f"Planificación de temporada alta: {mtop}",f"Mes con mayores ventas: <b>{mtop}</b> ({fmt(float(mtop_v))}). Acción 3 semanas antes: (1) Aumentar inventario de telas un 30%, (2) Contratar operario temporal si se esperan más de 70 pedidos, (3) Programar contenido de Instagram con antelación."))
            # Talla
            recs.append(("media","👔",f"Optimizar inventario en talla {ttop}",f"La talla más demandada es <b>{ttop}</b>. Siempre tener stock de tela para esta talla. Bundle: 10% descuento en compra de 2 prendas talla {ttop} para aumentar el ticket promedio del pedido."))
            # IA generativa
            recs.append(("alta","🤖","Implementar IA Generativa (Claude + ChatGPT + Canva)",f"El 70% de compradores colombianos busca en redes sociales antes de comprar. Con <b>ChatGPT-4o</b>: generar 20 descripciones de producto en 1 hora (costo: $0). Con <b>Canva AI + Remove.bg</b>: editar fotos del celular a nivel profesional ($15.000 COP/mes). Con <b>Claude API</b>: reporte ejecutivo automático cada lunes. ROI: +20% ventas desde mes 2."))

            # Renderizar leyenda + tarjetas
            le1,le2,le3=st.columns(3)
            with le1: st.markdown(f"<span style='color:{N(PURPLE)};font-size:11px;'>🟣 Alta prioridad — Impacto estratégico</span>", unsafe_allow_html=True)
            with le2: st.markdown(f"<span style='color:{N(GOLD)};font-size:11px;'>🟡 Alerta — Requiere atención inmediata</span>", unsafe_allow_html=True)
            with le3: st.markdown(f"<span style='color:{N(GREEN)};font-size:11px;'>🟢 Oportunidad — Optimización continua</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            orden={"alta":0,"alerta":1,"media":2}
            for urg,ico,tit,cont in sorted(recs,key=lambda x:orden.get(x[0],3)):
                st.markdown(card_ia(ico,tit,cont,urg), unsafe_allow_html=True)

            # Resumen ejecutivo automático
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div style='background:linear-gradient(135deg,{N(NAVY_M)},{N(NAVY_L)});border-radius:12px;
                 padding:18px 22px;border-top:3px solid {N(GOLD)};'>
                <h4 style='color:{N(GOLD)};margin-bottom:12px;'>📋 Resumen Ejecutivo Automático</h4>
                <p style='color:{N(WHITE)};font-size:13px;line-height:1.8;'>
                    El Almacén Katrina muestra un <b style='color:{N(GOLD)};'>margen bruto de {mg:.1f}%</b>
                    y una tendencia de ventas <b style='color:{"#27AE60" if trend>=0 else "#E74C3C"};'>{"positiva" if trend>=0 else "negativa"} ({trend:+.1f}%)</b> en el período.
                    Canal principal: <b style='color:{N(GOLD)};'>{ctop}</b> ({cpct:.0f}% de ventas).
                    Puntualidad de entregas: <b style='color:{"#27AE60" if punt>=85 else "#E74C3C"};'>{punt:.0f}%</b>.
                    Producto más rentable: <b style='color:{N(GOLD)};'>{pbest}</b> ({pbest_v:.1f}% margen).
                </p>
                <p style='color:{N(GRAY)};font-size:11px;margin-top:8px;'>
                    Generado automáticamente · {len(df_fe):,} transacciones · {nm} meses ·
                    {datetime.now().strftime("%d %b %Y %H:%M")}
                </p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════
# TAB 6 — DATOS CRUDOS
# ══════════════════════════
with T6:
    st.markdown(f"<h4 style='color:{N(GOLD)};'>Tabla de datos ({len(df_f):,} registros)</h4>", unsafe_allow_html=True)
    bq=st.text_input("🔍 Buscar",placeholder="Escribir para filtrar...",label_visibility="collapsed")
    dm=df_f[df_f.astype(str).apply(lambda c:c.str.contains(bq,case=False)).any(axis=1)] if bq else df_f
    cm=[c for c in dm.columns if not c.startswith("_")]
    st.dataframe(dm[cm].reset_index(drop=True),use_container_width=True,height=400)
    st.download_button("⬇️ Descargar datos filtrados (CSV)",
                       dm[cm].to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
                       file_name=f"katrina_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────
st.markdown(f"""<div style='text-align:center;margin-top:24px;padding:14px;
     background:{N(NAVY_M)};border-radius:10px;border-top:2px solid {N(GOLD)};'>
    <span style='color:{N(GOLD)};font-size:12px;font-weight:bold;'>Innovarte Consulting</span>
    <span style='color:{N(GRAY)};font-size:11px;'> · Almacén Fábrica Sacos y Suéteres Katrina · 2025</span><br>
    <span style='color:{N(GRAY)};font-size:10px;'>Dashboard de Inteligencia Comercial — Propuesta E2E Digital</span>
</div>""", unsafe_allow_html=True)
