import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página con estilo (Premium / Profesional)
st.set_page_config(
    page_title="Motor de Decisiones de Retención",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para emular un entorno bancario corporativo
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    .stButton>button { background-color: #c5a059; color: white; border-radius: 5px; font-weight: bold; }
    .stButton>button:hover { background-color: #e5c07b; color: #0d1117; }
    h1, h2, h3 { color: #c5a059 !important; font-family: 'Helvetica Neue', sans-serif; }
    .metric-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); }
    .metric-title { font-size: 14px; color: #8b949e; text-transform: uppercase; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# Asegurar que la ruta raíz esté disponible para importar la clase src.inference si fuera necesario
sys.path.append(os.path.abspath(os.getcwd()))

# Función optimizada para cargar el pipeline unificado con caché de Streamlit
@st.cache_resource
def cargar_pipeline():
    ruta_pipeline = "outputs/sistema_retencion_completo.joblib"
    if os.path.exists(ruta_pipeline):
        return joblib.load(ruta_pipeline)
    else:
        return None

pipeline = cargar_pipeline()

# --- SIDEBAR: PANEL DE CONTROL Y ADVERTENCIAS ---
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png", width=60)
st.sidebar.title("Control Panel")
st.sidebar.markdown("### Gestión de Campañas de Churn")

if pipeline is None:
    st.sidebar.error("❌ No se encontró el archivo `sistema_retencion_completo.joblib` en la carpeta `outputs/`. Ejecuta primero tu `main.py` para generar el artefacto.")
else:
    st.sidebar.success("✅ Ecosistema de Inferencia cargado y congelado en frío.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Ajuste Dinámico de Parámetros (Simulador)")
st.sidebar.info("Define las premisas financieras y pulsa **Actualizar Parámetros** para recalcular el impacto estocástico de la base cargada.")

# Referencia de arquetipos SIEMPRE visible (sidebar) para guiar la fijación de precios de marketing
with st.sidebar.expander("📖 Referencia de Clústeres (para fijar precios)", expanded=True):
    # st.markdown (no st.sidebar.markdown) para que la tabla quede DENTRO del expander
    st.markdown(
        "| ID | Arquetipo | Regla de identificación |\n"
        "| :---: | --- | --- |\n"
        "| **0** | Financiados | Mayor `Avg_Utilization_Ratio` |\n"
        "| **1** | Súper Usuarios | Mayor `Total_Trans_Amt` |\n"
        "| **2** | VIP Pasivos | Mayor `Credit_Limit` |\n"
        "| **3** | Alerta Roja | Menor `Total_Revolving_Bal` |\n"
    )

expander_umbrales = st.sidebar.expander("🔍 Umbrales Optimizados en Vivo", expanded=True)

# Recuperar Unit Economics por defecto guardados en el objeto para inicializar los inputs
if pipeline and hasattr(pipeline, 'unit_economics'):
    ue_saved = pipeline.unit_economics
else:
    ue_saved = {
        0: {'costo_alerta': 35, 'ltv_rescate': 450},
        1: {'costo_alerta': 250, 'ltv_rescate': 2000},
        2: {'costo_alerta': 25, 'ltv_rescate': 600},
        3: {'costo_alerta': 2, 'ltv_rescate': 100}
    }

# Formulario que agrupa TODOS los parámetros: solo recalculan al pulsar el botón.
# Así un cambio individual ya no dispara un rerun que borraba la data cargada.
with st.sidebar.form("parametros_financieros"):
    # 1) Eficiencia de Retención PRIMERO (antes que los costos, como se pidió)
    st.markdown("### 🎯 Eficiencia de Retención")
    save_rate = st.number_input(
        "Tasa de Rescate (Save Rate)",
        min_value=0.05, max_value=1.00,
        value=0.30, step=0.05, format="%.2f",
        help="Tasa esperada de éxito de las campañas de contención.",
        key="save_rate_input"
    )

    st.markdown("---")
    st.markdown("### 💰 Unit Economics por Clúster")

    # 2) Costos y LTV por clúster mediante number_input (alternativa a los sliders)
    st.markdown("**C0: Financiados**")
    c0_cost = st.number_input("Costo Alerta (C0) $", min_value=5, max_value=100, value=int(ue_saved[0]['costo_alerta']), step=1, key="c0_c")
    c0_ltv = st.number_input("LTV Rescate (C0) $", min_value=100, max_value=1000, value=int(ue_saved[0]['ltv_rescate']), step=10, key="c0_l")

    st.markdown("**C1: Súper Usuarios**")
    c1_cost = st.number_input("Costo Alerta (C1) $", min_value=50, max_value=500, value=int(ue_saved[1]['costo_alerta']), step=5, key="c1_c")
    c1_ltv = st.number_input("LTV Rescate (C1) $", min_value=1000, max_value=5000, value=int(ue_saved[1]['ltv_rescate']), step=50, key="c1_l")

    st.markdown("**C2: VIP Pasivos**")
    c2_cost = st.number_input("Costo Alerta (C2) $", min_value=5, max_value=100, value=int(ue_saved[2]['costo_alerta']), step=1, key="c2_c")
    c2_ltv = st.number_input("LTV Rescate (C2) $", min_value=200, max_value=1500, value=int(ue_saved[2]['ltv_rescate']), step=10, key="c2_l")

    st.markdown("**C3: Alerta Roja**")
    c3_cost = st.number_input("Costo Alerta (C3) $", min_value=1, max_value=30, value=int(ue_saved[3]['costo_alerta']), step=1, key="c3_c")
    c3_ltv = st.number_input("LTV Rescate (C3) $", min_value=20, max_value=300, value=int(ue_saved[3]['ltv_rescate']), step=10, key="c3_l")

    # Botón que confirma TODA la selección de una sola vez (al fondo del formulario)
    st.form_submit_button("🔄 Actualizar Parámetros", width="stretch")

# Armar el diccionario de simulación dinámica con los valores confirmados del formulario
dynamic_ue = {
    0: {'costo_alerta': c0_cost, 'ltv_rescate': c0_ltv},
    1: {'costo_alerta': c1_cost, 'ltv_rescate': c1_ltv},
    2: {'costo_alerta': c2_cost, 'ltv_rescate': c2_ltv},
    3: {'costo_alerta': c3_cost, 'ltv_rescate': c3_ltv}
}

# =========================================================================
# MOTOR DE OPTIMIZACIÓN EMPÍRICA EN VIVO (BARRIDO MONTE CARLO)
# =========================================================================
def recalcular_umbrales_en_vivo(pipeline, dynamic_ue, save_rate=0.30):
    """
    Extrae la data de validación oculta en el artefacto y calcula por fuerza bruta
    los umbrales óptimos adaptados a los costos actuales seleccionados por el usuario.
    """
    X_val = pipeline.X_val
    y_val = np.asarray(pipeline.y_val)
    
    # Predecimos probabilidades sobre validación usando el LightGBM congelado
    proba_val = pipeline.model.predict_proba(X_val)[:, 1]
    cluster_val = X_val['Cluster_Cliente'].values
    grid = np.linspace(0.01, 0.99, 197)
    
    umbrales_dinamicos = {}
    
    for c, econ in dynamic_ue.items():
        mask = cluster_val == c
        n_churn = int(np.sum(y_val[mask] == 1))
        
        # 1. Recalcular el anclaje analítico Break-Even con los costos actuales del slider
        ub_breakeven = min(econ['costo_alerta'] / (save_rate * econ['ltv_rescate']), 1.0)
        
        # 2. Si el segmento es inestable, cae en el nuevo break-even analítico dinámico
        if n_churn < 10:
            umbrales_dinamicos[c] = ub_breakeven
        else:
            # 3. Barrido Empírico en Caliente por clúster
            v_tp = (save_rate * econ['ltv_rescate']) - econ['costo_alerta']
            v_fp = econ['costo_alerta']
            mejor_u, mejor_b = 1.0, 0.0
            
            for tau in grid:
                pred = proba_val[mask] > tau
                b = np.sum(pred & (y_val[mask] == 1)) * v_tp - np.sum(pred & (y_val[mask] == 0)) * v_fp
                if b > mejor_b:
                    mejor_b, mejor_u = b, tau
            umbrales_dinamicos[c] = mejor_u
            
    return umbrales_dinamicos

# --- PANTALLA PRINCIPAL ---
st.title("🏦 Ecosistema de Inferencia y Simulación Financiera")
st.markdown("Bienvenido al Portal Ejecutivo de Mitigación de Riesgos. Sube el lote de clientes para orquestar las campañas asimétricas.")

# 1. Creamos dos columnas: una para el cargador de archivos y otra para el botón demo
col_upload, col_demo = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader("📂 Arrastra aquí el layout transaccional crudo (.csv)", type=["csv"])

with col_demo:
    st.markdown("<br>", unsafe_allow_html=True) # Espaciador para alinear el botón
    usar_demo = st.button("🚀 Usar Dataset de Muestra (Kaggle)", width="stretch")
    st.link_button("🌐 Ver fuente en Kaggle", 
                   "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers", 
                   width="stretch")

# 2. Persistimos el DataFrame en session_state para que NO se borre al usar otros widgets.
#    El botón demo solo devuelve True el rerun en que se pulsa; sin session_state la data
#    desaparecía en cuanto se tocaba cualquier otro control.
if 'df_input' not in st.session_state:
    st.session_state.df_input = None

# 3. Validamos cuál de las dos opciones activó el usuario y lo guardamos en sesión
if uploaded_file is not None:
    st.session_state.df_input = pd.read_csv(uploaded_file)
elif usar_demo:
    ruta_demo = "data/BankChurners.csv"
    if os.path.exists(ruta_demo):
        st.session_state.df_input = pd.read_csv(ruta_demo)
        st.toast("🎯 Cargando el dataset completo de Kaggle para la demostración.", icon="📊")
    else:
        st.error(f"No se encontró el archivo demo en `{ruta_demo}`. Por favor, asegúrate de tener el archivo original ahí.")

# Recuperamos la data persistida (sobrevive a los reruns provocados por otros widgets)
df_input = st.session_state.df_input

# 4. Si logramos capturar datos (por subida o por demo), corremos toda la maquinaria abajo
if df_input is not None and pipeline is not None:
    
    # Procesamiento protegido con barra de carga para feedback visual
    with st.spinner("Re-optimizando umbrales en vivo y ejecutando inferencia..."):
        try:
            # =========================================================================
            # 🔥 INYECCIÓN DE LA RE-OPTIMIZACIÓN EN CALIENTE (BARRIDO EN VIVO)
            # =========================================================================
            # 1. Ejecutamos el barrido de 197 combinaciones sobre el set de Validación embebido
            umbrales_optimos_en_vivo = recalcular_umbrales_en_vivo(pipeline, dynamic_ue, save_rate=save_rate)
            
            # 2. Sobrescribimos temporalmente los umbrales estáticos del artefacto con los nuevos óptimos
            pipeline.umbrales = umbrales_optimos_en_vivo
            
            # 3. Desplegamos un visor dinámico en el sidebar para transparencia ejecutiva
            st.sidebar.markdown("---")
            with expander_umbrales:
                # Usamos st.write (no st.sidebar.write) para que el contenido caiga DENTRO del expander
                for cl, umb in umbrales_optimos_en_vivo.items():
                    st.write(f"Clúster {cl}: **{umb:.4f}**")
            # =========================================================================
            
            # Ahora el método procesar_e_inferir usará de forma nativa la política re-optimizada
            df_output = pipeline.procesar_e_inferir(df_input)
            
            # Reinyectar variables de visualización si existen en el original para métricas avanzadas de negocio
            if 'Target_Churn' in df_input.columns:
                df_output['Churn_Real'] = df_input['Target_Churn']
            elif 'Attrition_Flag' in df_input.columns:
                df_output['Churn_Real'] = df_input['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
                
            st.toast("🚀 Inferencia masiva ejecutada con éxito total.", icon="🔥")
            
            # --- CÁLCULO DE MÉTRICAS FINANCIERAS DINÁMICAS ---
            total_clientes = len(df_output)
            alertados = df_output['Prediccion_Final'].sum()
            
            # Calcular costos e ingresos basados en los parámetros confirmados en el formulario
            costo_total = 0.0
            ganancia_bruta = 0.0
            # save_rate proviene del formulario del sidebar (ya no está hardcodeado)
            
            # Si el archivo tiene la columna real, calculamos ROI puro, si no, proyectamos basado en predicciones
            tiene_real = 'Churn_Real' in df_output.columns
            
            for index, row in df_output.iterrows():
                c = int(row['Cluster_Cliente'])
                econ = dynamic_ue[c]
                is_alerted = row['Prediccion_Final'] == 1
                
                if is_alerted:
                    costo_total += econ['costo_alerta']
                    if tiene_real:
                        if row['Churn_Real'] == 1:
                            ganancia_bruta += save_rate * econ['ltv_rescate']
                    else:
                        # Si no hay data real (producción pura), proyectamos matemáticamente según la probabilidad calculada
                        ganancia_bruta += row['Probabilidad_Churn'] * save_rate * econ['ltv_rescate']
                        
            beneficio_neto = ganancia_bruta - costo_total

            # --- DISEÑO DE TARJETAS (KPI CARDS) ---
            st.markdown("### 📊 Indicadores de Impacto Financiero Operativo")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="metric-box"><div class="metric-title">Clientes Procesados</div><div class="metric-value">{total_clientes:,}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-box"><div class="metric-title">Alertas Disparadas</div><div class="metric-value" style="color: #ff4b4b;">{alertados:,} ({alertados/total_clientes:.1%})</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-box"><div class="metric-title">Inversión Operativa Campaña</div><div class="metric-value">${costo_total:,.2f}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-box"><div class="metric-title">Beneficio Neto Proyectado</div><div class="metric-value" style="color: #2ca02c;">${beneficio_neto:,.2f}</div></div>', unsafe_allow_html=True)
                
            st.markdown("---")
            
            # --- ZONA DE GRÁFICAS ---
            st.markdown("### 📈 Distribución Estratégica del Portafolio")
            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                # Gráfica de barras de acciones asignadas
                resumen_campanas = df_output['Estrategia_Asignada'].value_counts().reset_index()
                resumen_campanas.columns = ['Estrategia', 'Clientes']
                fig_bar = px.bar(
                    resumen_campanas, x='Clientes', y='Estrategia', orientation='h',
                    title='Asignación de Campañas de Mitigación',
                    color='Estrategia', color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color="white",
                    showlegend=False
                    )
                
                st.plotly_chart(fig_bar, width="stretch")
                
            with g_col2:
                # Gráfica de caja/distribución de probabilidades por clúster canónico

                nombres_clusters = {
                    0: 'C0: Financiados',
                    1: 'C1: Súper Usr.',
                    2: 'C2: VIP Pasivos',
                    3: 'C3: Alerta Roja'
                }
                df_output['Clúster Nombre'] = df_output['Cluster_Cliente'].map(nombres_clusters)
                
                # Definimos el orden explícito que queremos ver en el eje X
                orden_canonico = [
                    'C0: Financiados', 
                    'C1: Súper Usr.', 
                    'C2: VIP Pasivos', 
                    'C3: Alerta Roja'
                ]
                
                fig_box = px.box(
                    df_output, 
                    x='Clúster Nombre', 
                    y='Probabilidad_Churn',
                    color='Clúster Nombre', 
                    title='Densidad del Riesgo de Churn por Arquetipo',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    # FORZAMOS EL ORDEN DE LAS CATEGORÍAS AQUÍ:
                    category_orders={'Clúster Nombre': orden_canonico}
                )
                
                fig_box.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color="white", 
                    showlegend=False
                )
                st.plotly_chart(fig_box, width="stretch")
    
            # --- SIMULACIÓN MONTE CARLO EN VIVO (10,000 ESCENARIOS) ---
            st.markdown("---")
            st.markdown("### 🎲 Simulación Monte Carlo mediante Bootstrap de Riesgo en Vivo (10,000 Escenarios)")
            
            with st.spinner("Corriendo remuestreos Monte Carlo vectorizados..."):
                # Inicializamos el vector base de beneficio por cliente
                df_output['Beneficio_Base_Sim'] = 0.0
                tiene_real = 'Churn_Real' in df_output.columns
                
                for c_id, econ in dynamic_ue.items():
                    mask = (df_output['Cluster_Cliente'] == c_id) & (df_output['Prediccion_Final'] == 1)
                    
                    if tiene_real:
                        # MODO AUDITORÍA: Si hay datos reales, calculamos el pago exacto observado
                        cond_tp = mask & (df_output['Churn_Real'] == 1)
                        cond_fp = mask & (df_output['Churn_Real'] == 0)
                        df_output.loc[cond_tp, 'Beneficio_Base_Sim'] = (save_rate * econ['ltv_rescate']) - econ['costo_alerta']
                        df_output.loc[cond_fp, 'Beneficio_Base_Sim'] = -econ['costo_alerta']
                    else:
                        # MODO PRODUCCIÓN PURA (Sin etiquetas): Usamos el Valor Esperado Probabilístico
                        p_churn = df_output.loc[mask, 'Probabilidad_Churn']
                        df_output.loc[mask, 'Beneficio_Base_Sim'] = (p_churn * save_rate * econ['ltv_rescate']) - econ['costo_alerta']
                        
                # Extraemos el vector resultante (sea real o esperado) y corremos el Bootstrap masivo por LOTES
                beneficios_arr = df_output['Beneficio_Base_Sim'].values
                n_sim = 10000
                n_obs = len(beneficios_arr)
                batch_size = 1000  # Lotes de mil para proteger la RAM de Streamlit Cloud

                roi_simulados = np.zeros(n_sim)
                for i in range(0, n_sim, batch_size):
                    # Procesamos 1,000 escenarios a la vez (Pico de RAM marginal)
                    indices_boot = np.random.choice(n_obs, size=(batch_size, n_obs), replace=True)
                    roi_simulados[i:i+batch_size] = np.sum(beneficios_arr[indices_boot], axis=1)
                
                # Ocultamos el resto del formateo del histograma (conteos, cortes_bins, px.bar, etc. se quedan IGUAL)
                conteos, cortes_bins = np.histogram(roi_simulados, bins=40)
                df_hist_dinamico = pd.DataFrame({
                    'Beneficio Neto ($ USD)': (cortes_bins[:-1] + cortes_bins[1:]) / 2,
                    'Frecuencia (Simulaciones)': conteos
                })
                
                fig_hist = px.bar(
                    df_hist_dinamico, x='Beneficio Neto ($ USD)', y='Frecuencia (Simulaciones)',
                    color='Beneficio Neto ($ USD)', color_continuous_scale='viridis',
                    title="Distribución de Probabilidad del Retorno Financiero"
                )
                
                fig_hist.add_vline(x=np.mean(roi_simulados), line_dash="dash", line_color="gold", 
                                   annotation_text=f"Media: ${np.mean(roi_simulados):,.0f}", annotation_position="top right",annotation_font_color="gray")
                fig_hist.add_vline(x=np.percentile(roi_simulados, 2.5), line_dash="dot", line_color="red", 
                                   annotation_text=f"P 2.5%: ${np.percentile(roi_simulados, 2.5):,.0f}", annotation_position="top left",annotation_font_color="gray")
                
                fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
                st.plotly_chart(fig_hist, width="stretch")
                
            # --- TABLA DE DATOS Y DESCARGA ---
            st.markdown("---")
            st.markdown("### 📋 Vista Previa del Layout Operativo de Salida")
            
            # Mostramos las columnas solicitadas ordenadas
            cols_mostrar = ['CLIENTNUM', 'Cluster_Cliente', 'Probabilidad_Churn', 'Prediccion_Final', 'Estrategia_Asignada']
            st.dataframe(df_output[cols_mostrar].head(100), width="stretch")
            
            # Botón de descarga para el CSV operativo listo para Marketing
            csv_data = df_output[cols_mostrar].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Plan de Acción Comercial Completo (CSV)",
                data=csv_data,
                file_name="plan_accion_clientes_produccion.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento del archivo: {str(e)}")
            st.info("Asegúrate de que el archivo CSV contenga exactamente las columnas demográficas y transaccionales del Contrato de Datos original.")
else:
    if pipeline is not None:
        st.warning("💡 Por favor, carga un archivo CSV en el panel superior para activar el motor de inferencia.")
