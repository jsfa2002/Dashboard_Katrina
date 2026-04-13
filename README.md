# 👔 Dashboard Katrina — Inteligencia Comercial

**Almacén Fábrica Sacos y Suéteres Katrina**  
Desarrollado por **Innovarte Consulting** · Bogotá, Colombia · 2025

---

## ¿Qué es esto?

Dashboard de inteligencia comercial construido en Python + Streamlit.
Permite a Katrina subir sus registros de ventas mensuales y explorar:

- Ventas totales, márgenes y ticket promedio
- Análisis por producto y categoría
- Canales de venta y métodos de pago
- Operaciones: puntualidad de entregas y pedidos por día
- Descarga de datos filtrados

---

## Estructura del proyecto

```
katrina_dashboard/
├── app.py                    ← Dashboard principal (Streamlit)
├── generar_datos_katrina.py  ← Simulación de datos (demo)
├── requirements.txt          ← Dependencias
├── datos_katrina/            ← CSVs generados (para demo)
│   ├── katrina_2025_01_enero.csv
│   ├── katrina_2025_02_febrero.csv
│   ├── katrina_2025_03_marzo.csv
│   ├── katrina_2025_04_abril.csv
│   ├── katrina_2025_05_mayo.csv
│   └── katrina_2025_06_junio.csv
└── README.md
```

---

## Cómo ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/katrina_dashboard.git
cd katrina_dashboard

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) Generar datos de demo
python generar_datos_katrina.py

# 4. Ejecutar el dashboard
streamlit run app.py
```

Abre tu navegador en: **http://localhost:8501**

---

## Cómo desplegar en Streamlit Cloud

1. Sube este repositorio a GitHub (público o privado)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona el repositorio y `app.py` como archivo principal
5. Haz clic en **Deploy** → listo 🎉

---

## Estructura de los datos (cómo llenar el Excel)

Cada archivo debe tener **una fila por pedido** con estas columnas:

| Columna | Ejemplo | Descripción |
|---------|---------|-------------|
| `id_pedido` | KAT-202501-001 | Identificador único |
| `fecha_pedido` | 2025-01-15 | Fecha YYYY-MM-DD |
| `nombre_cliente` | Carlos Martínez | Nombre del cliente |
| `canal` | WhatsApp | WhatsApp / Presencial / Instagram / Web |
| `referencia` | Camisa Clásica Blanca | Nombre del producto |
| `categoria` | Camisas | Camisas / Pantalones / Jeans |
| `talla` | M | S / M / L / XL / XXL |
| `cantidad` | 2 | Número de unidades |
| `precio_unitario` | 69900 | Precio sin puntos ni $, en COP |
| `costo_produccion` | 32000 | Costo de producir 1 unidad |
| `total_venta` | 139800 | precio_unitario × cantidad |
| `margen_bruto` | 75800 | (precio - costo) × cantidad |
| `estado_pedido` | Entregado | Entregado / Pendiente / Cancelado |
| `fecha_entrega_comprometida` | 2025-01-20 | Fecha prometida al cliente |
| `fecha_entrega_real` | 2025-01-20 | Fecha real de entrega |
| `metodo_pago` | Nequi | Efectivo / Nequi / Daviplata / Transferencia |
| `notas` | Cliente frecuente | Observaciones opcionales |

### Proceso recomendado para el equipo Katrina

```
Mes a mes (al cierre del mes):
1. Abrir el archivo Excel "Registro_Katrina_YYYYMM.xlsx"
2. Agregar una fila por cada pedido del mes
3. Exportar como CSV: katrina_2025_01_enero.csv
4. Subir al dashboard → explorar resultados
5. Guardar el archivo para el historial
```

---

## Preguntas de negocio que responde el dashboard

| Pregunta | Pestaña |
|----------|---------|
| ¿Cuánto vendí este mes? | Resumen General |
| ¿Qué producto deja más margen? | Productos |
| ¿Por qué canal vendo más? | Canales & Pagos |
| ¿Estoy entregando a tiempo? | Operaciones |
| ¿Cuál es mi ticket promedio? | Resumen General |
| ¿Qué talla se vende más? | Productos |

---

## Criterios del trabajo final cubiertos

| Criterio guía | Cómo lo cubre el dashboard |
|---------------|---------------------------|
| Consumo Digital | Google Sheets → CSV → Streamlit Cloud |
| Indicadores de Éxito (KPIs) | Margen %, tasa entrega, ventas por canal |
| Mockup de la Solución | Dashboard funcional como mockup interactivo |
| Gobernanza de Datos | Plantilla estandarizada + proceso mensual |
| Recolección de Feedback | Columna "notas" + encuesta post-entrega |

---

## Tecnologías

- **Python** · Pandas · NumPy
- **Streamlit** (frontend e infraestructura)
- **Plotly** (visualizaciones)
- **Streamlit Cloud** (despliegue gratuito)

---

*Innovarte Consulting © 2025*
