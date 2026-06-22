import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def _beneficio_total(pred, y_true, clusters, unit_economics, save_rate):
    """Beneficio neto total (USD) de un vector de decisión binaria, dado el save_rate."""
    total = 0.0
    for c, econ in unit_economics.items():
        m = clusters == c
        tp = np.sum(pred & (y_true == 1) & m)
        fp = np.sum(pred & (y_true == 0) & m)
        total += tp * (save_rate * econ['ltv_rescate'] - econ['costo_alerta']) - fp * econ['costo_alerta']
    return total


def _barrer_umbral_cluster(proba, y_true, econ, save_rate, grid):
    """OPCIÓN 3: barre la grilla de umbrales y devuelve el que maximiza el beneficio
    real del clúster sobre el set de validación. Si ningún umbral es rentable,
    devuelve 1.0 (no alertar a nadie en ese segmento)."""
    valor_tp = save_rate * econ['ltv_rescate'] - econ['costo_alerta']
    valor_fp = econ['costo_alerta']
    mejor_u, mejor_b = 1.0, 0.0
    for tau in grid:
        pred = proba > tau
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        b = tp * valor_tp - fp * valor_fp
        if b > mejor_b:
            mejor_b, mejor_u = b, tau
    return mejor_u


def _barrer_umbral_global(proba, y_true, clusters, unit_economics, save_rate, grid):
    """Benchmark: mejor umbral GLOBAL único (campaña agresiva) calibrado en validación."""
    mejor_u, mejor_b = 1.0, 0.0
    for tau in grid:
        pred = proba > tau
        b = _beneficio_total(pred, y_true, clusters, unit_economics, save_rate)
        if b > mejor_b:
            mejor_b, mejor_u = b, tau
    return mejor_u


def generar_lista_estrategias(X_val, y_val, X_test, y_test, model, save_rate=0.30, output_dir="outputs"):
    """
    Motor de decisiones cost-sensitive. Calibra los umbrales por clúster sobre el
    set de VALIDACIÓN (sin tocar test) combinando dos criterios:
      - OPCIÓN 1: break-even analítico  p* = costo / (save_rate * LTV)
      - OPCIÓN 3: barrido empírico que maximiza el beneficio real en validación
    Luego evalúa la estrategia segmentada frente a una campaña global agresiva
    (también calibrada en validación) sobre el set de TEST intacto.
    """
    print("\nEjecutando el Motor de Decisiones Financieras Asimétricas (cost-sensitive)...")

    # ==========================================
    # DICCIONARIO DE UNIT ECONOMICS ($ USD)
    # ==========================================
    unit_economics = {
        0: {'costo_alerta': 35,  'ltv_rescate': 450},   # Financiados
        1: {'costo_alerta': 250, 'ltv_rescate': 2000},  # Súper Usuarios
        2: {'costo_alerta': 25,  'ltv_rescate': 600},   # VIP Pasivos
        3: {'costo_alerta': 2,   'ltv_rescate': 100}    # Alerta Roja
    }

    # ==========================================
    # CALIBRACIÓN DE UMBRALES SOBRE VALIDACIÓN
    # ==========================================
    proba_val = model.predict_proba(X_val)[:, 1]
    cluster_val = X_val['Cluster_Cliente'].values
    y_val_arr = np.asarray(y_val)
    grid = np.linspace(0.01, 0.99, 197)

    # OPCIÓN 1: ancla analítica break-even (acotada a 1.0 si el segmento no es rentable)
    umbrales_breakeven = {
        c: min(e['costo_alerta'] / (save_rate * e['ltv_rescate']), 1.0)
        for c, e in unit_economics.items()
    }

    # OPCIÓN 3: barrido empírico por clúster, con respaldo en el break-even
    # cuando el segmento tiene muy pocos churners en validación (poca evidencia).
    umbrales_segmentados, metodo = {}, {}
    for c, econ in unit_economics.items():
        mask = cluster_val == c
        n_churn = int(np.sum(y_val_arr[mask] == 1))
        if n_churn < 10:
            umbrales_segmentados[c] = umbrales_breakeven[c]
            metodo[c] = f"break-even (n_churn={n_churn})"
        else:
            umbrales_segmentados[c] = _barrer_umbral_cluster(
                proba_val[mask], y_val_arr[mask], econ, save_rate, grid
            )
            metodo[c] = "barrido empírico"

    # Benchmark: mejor campaña GLOBAL agresiva (umbral único) calibrada en validación
    umbral_global = _barrer_umbral_global(
        proba_val, y_val_arr, cluster_val, unit_economics, save_rate, grid
    )

    print(f"\n  Tasa de rescate realista (save_rate) aplicada: {save_rate:.0%}")
    print("  --- Umbrales calibrados sobre VALIDACIÓN (no se toca el test) ---")
    print(f"  {'Clúster':<8}{'Break-even (Op.1)':<19}{'Empírico (Op.3)':<18}{'Método':<26}")
    for c in unit_economics:
        print(f"  {c:<8}{umbrales_breakeven[c]:<19.4f}{umbrales_segmentados[c]:<18.4f}{metodo[c]:<26}")
    print(f"  Umbral GLOBAL agresivo (benchmark, calibrado en validación): {umbral_global:.4f}")

    # ==========================================
    # APLICACIÓN DE LA ESTRATEGIA SEGMENTADA SOBRE TEST
    # ==========================================
    df_decisiones = X_test.copy()
    df_decisiones['Probabilidad_Churn'] = model.predict_proba(X_test)[:, 1]
    df_decisiones['Churn_Real'] = np.asarray(y_test)

    df_decisiones['Estrategia_Asignada'] = "Sin Acción (Cliente Estable)"
    df_decisiones['Prediccion_Final'] = 0

    for cluster, umbral in umbrales_segmentados.items():
        cond_alarma = (df_decisiones['Cluster_Cliente'] == cluster) & (df_decisiones['Probabilidad_Churn'] > umbral)
        df_decisiones.loc[cond_alarma, 'Prediccion_Final'] = 1
        
        if cluster == 1:
            df_decisiones.loc[cond_alarma, 'Estrategia_Asignada'] = "ALERTA PREMIUM: Francotirador (Contacto Humano / Condonación)"
        elif cluster == 3:
            df_decisiones.loc[cond_alarma, 'Estrategia_Asignada'] = "CAMPAÑA MASIVA: Red de Arrastre (Email Automático / MSI)"
        elif cluster == 0:
            df_decisiones.loc[cond_alarma, 'Estrategia_Asignada'] = "ALIVIO FINANCIERO: Reestructuración (Descuento de Tasa / Pagos Fijos)"
        elif cluster == 2:
            df_decisiones.loc[cond_alarma, 'Estrategia_Asignada'] = "DESPERTAR VIP: Re-engagement (Bono de Cashback / Upgrade Tarjeta)"
            
    print("\n=== DISTRIBUCIÓN DE ACCIONES COMERCIALES ASIGNADAS ===")
    resumen_acciones = df_decisiones['Estrategia_Asignada'].value_counts().reset_index()
    resumen_acciones.columns = ['Acción Comercial', 'Cantidad de Clientes']
    resumen_acciones['% del Portafolio Set'] = (resumen_acciones['Cantidad de Clientes'] / len(df_decisiones)) * 100
    print(resumen_acciones.to_string(index=False))
    
    # Exportación del CSV de operaciones de marketing
    os.makedirs(output_dir, exist_ok=True)
    ruta_salida = os.path.join(output_dir, "data_plan_accion_clientes.csv")
    columnas_entrega = ['Cluster_Cliente', 'Credit_Limit', 'Total_Trans_Amt', 'Probabilidad_Churn', 'Estrategia_Asignada', 'Prediccion_Final', 'Churn_Real']
    df_decisiones[columnas_entrega].to_csv(ruta_salida)
    print(f"\n[ÉXITO] Lista de asignación comercial exportada a: {ruta_salida}")
    
    # --- GENERACIÓN DEL REPORTE MAESTRO DE MATRICES Y FINANZAS ---
    results = []
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Matriz Global (Arriba)
    ax_total = plt.subplot2grid((2, 4), (0, 1), colspan=2)
    matrix_global = metrics.confusion_matrix(df_decisiones['Churn_Real'], df_decisiones['Prediccion_Final'])
    labels_g = np.array([["{}\n({:.1%})".format(val, val/matrix_global.sum()) for val in fila] for fila in matrix_global])

    recall_global = matrix_global[1, 1] / (matrix_global[1, 1] + matrix_global[1, 0]) if (matrix_global[1, 1] + matrix_global[1, 0]) > 0 else 0
    prescision_global = matrix_global[1, 1] / (matrix_global[1, 1] + matrix_global[0, 1]) if (matrix_global[1, 1] + matrix_global[0, 1]) > 0 else 0
    f1_global = 2 * (prescision_global * recall_global) / (prescision_global + recall_global) if (prescision_global + recall_global) > 0 else 0
    accuracy_global = (matrix_global[0, 0] + matrix_global[1, 1]) / matrix_global.sum() if matrix_global.sum() > 0 else 0
    
    sns.heatmap(matrix_global, annot=labels_g, fmt='', cmap='Blues', ax=ax_total, cbar=False, square=True, annot_kws={"size": 12})
    ax_total.set_title('Matriz de Confusión Global\n(Estrategias Combinadas)', fontsize=13, fontweight='bold', pad=15)
    ax_total.set_xlabel('Predicción del Modelo')
    ax_total.set_ylabel('Realidad (Cliente)')

    # Variables acumulativas para el ROI Global
    roi_global_beneficio = 0
    roi_global_costo = 0
    roi_global_ganancia = 0
    
    # 2. Matrices por Segmento y Cálculo Financiero (Abajo)
    colores = ['Purples', 'Oranges', 'Greens', 'Reds']
    nombres_clusters = {
        0: 'C0: Financiados (Alivio)',
        1: 'C1: Súper Usr. (Contacto Premium)',
        2: 'C2: VIP Pasivos (Re-engagement)',
        3: 'C3: Alerta Roja (Red Arrastre)'
    }
    
    for idx, cluster in enumerate([0, 1, 2, 3]):
        ax = plt.subplot2grid((2, 4), (1, idx))
        df_c = df_decisiones[df_decisiones['Cluster_Cliente'] == cluster]
        
        matrix_c = metrics.confusion_matrix(df_c['Churn_Real'], df_c['Prediccion_Final'], labels=[0, 1])
        total_segmento = matrix_c.sum()
        
        labels_c = np.array([["{}\n({:.1%})".format(val, val/total_segmento if total_segmento > 0 else 0) for val in fila] for fila in matrix_c])
        sns.heatmap(matrix_c, annot=labels_c, fmt='', cmap=colores[idx], ax=ax, cbar=False, square=True, annot_kws={"size": 10})
        
        tp, fn, fp = matrix_c[1, 1], matrix_c[1, 0], matrix_c[0, 1]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (matrix_c[0, 0] + tp) / matrix_c.sum() if matrix_c.sum() > 0 else 0
        
        # --- CÁLCULO FINANCIERO DEL CLÚSTER ---
        costo_alerta = unit_economics[cluster]['costo_alerta']
        ltv_rescate = unit_economics[cluster]['ltv_rescate']
        
        costo_campana = (tp + fp) * costo_alerta
        ganancia_rescate = tp * save_rate * ltv_rescate
        beneficio_neto = ganancia_rescate - costo_campana
        
        roi_global_costo += costo_campana
        roi_global_ganancia += ganancia_rescate
        roi_global_beneficio += beneficio_neto
        
        results.append({
            'Segmento': nombres_clusters[cluster],
            'Clientes Totales': matrix_c.sum(),
            'Clientes Churn': matrix_c[1, :].sum(),
            'Alertados Correctos (TP)': tp,
            'Clientes Alertados': (tp + fp),
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Costo_Campana_USD': costo_campana,
            'Ganancia_Retencion_USD': ganancia_rescate,
            'Beneficio_Neto_ROI_USD': beneficio_neto
        })
        
        ax.set_title(f"{nombres_clusters[cluster]}\nRecall: {recall:.1%}\nPrecision: {precision:.1%}", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Predicción')
        if idx == 0:
            ax.set_ylabel('Realidad')

    # Insertamos el Global al inicio de la lista de resultados ahora que tenemos el ROI sumado
    results.insert(0, {
        'Segmento': 'Global',
        'Clientes Totales': matrix_global.sum(),
        'Clientes Churn': matrix_global[1, :].sum(),
        'Alertados Correctos (TP)': matrix_global[1, 1],
        'Clientes Alertados': matrix_global[:, 1].sum(),
        'Recall': recall_global,
        'Precision': prescision_global,
        'F1 Score': f1_global,
        'Accuracy': accuracy_global,
        'Costo_Campana_USD': roi_global_costo,
        'Ganancia_Retencion_USD': roi_global_ganancia,
        'Beneficio_Neto_ROI_USD': roi_global_beneficio
    })
            
    plt.suptitle('Resultados del Motor de Decisiones por Segmento Transaccional', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    ruta_viz = os.path.join(output_dir, "viz_resultados_estrategias.png")
    plt.savefig(ruta_viz, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Exportar el DataFrame numérico y financiero
    df_resultados = pd.DataFrame(results)
    ruta_resultados = os.path.join(output_dir, "data_resultados_financieros.csv")
    df_resultados.to_csv(ruta_resultados, index=False)
    print(f"[ÉXITO] Resultados financieros exportados a: {ruta_resultados}")
    
    # ==========================================
    # NUEVO: GRÁFICA DE BARRAS DE ROI DE NEGOCIO
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resultados[df_resultados['Segmento'] != 'Global'], 
                x='Segmento', 
                y='Beneficio_Neto_ROI_USD', 
                palette='viridis',
                hue='Beneficio_Neto_ROI_USD',
                )
    plt.title(f'Beneficio Neto (ROI) por Estrategia de Segmento\nBeneficio Total Proyectado: ${roi_global_beneficio:,.2f} USD', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Beneficio Neto ($ USD)')
    plt.xlabel('Clúster Conductual')
    plt.xticks(rotation=15)
    
    # Añadir etiquetas de valor encima de las barras
    for index, value in enumerate(df_resultados[df_resultados['Segmento'] != 'Global']['Beneficio_Neto_ROI_USD']):
        plt.text(index, value + (value*0.02), f'${value:,.0f}', ha='center', fontweight='bold')
        
    ruta_roi = os.path.join(output_dir, "viz_roi_financiero.png")
    plt.savefig(ruta_roi, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ÉXITO] Gráfica de Retorno de Inversión (ROI) exportada a: {ruta_roi}")

    # =========================================================================
    # NUEVO: ANÁLISIS DE RIESGO ESTOCÁSTICO (BOOTSTRAP DUAL COMPETITIVO)
    # =========================================================================
    print("\nIniciando simulación Bootstrap Dual (10,000 iteraciones) para medición de riesgo...")
    
    # 1. Vectorizar beneficios para ambas estrategias
    df_decisiones['Beneficio_Ind_A'] = 0.0 # Estrategia Segmentada (Nuestra, cost-sensitive)
    df_decisiones['Beneficio_Ind_B'] = 0.0 # Estrategia Agresiva Global (umbral único calibrado en validación)
    
    # Condiciones Estrategia A
    cond_tp_a = (df_decisiones['Prediccion_Final'] == 1) & (df_decisiones['Churn_Real'] == 1)
    cond_fp_a = (df_decisiones['Prediccion_Final'] == 1) & (df_decisiones['Churn_Real'] == 0)
    
    # Condiciones Estrategia B (aplica el umbral global agresivo a todo el mundo)
    cond_alarma_b = df_decisiones['Probabilidad_Churn'] > umbral_global
    cond_tp_b = cond_alarma_b & (df_decisiones['Churn_Real'] == 1)
    cond_fp_b = cond_alarma_b & (df_decisiones['Churn_Real'] == 0)
    
    for cluster, econ in unit_economics.items():
        c_mask = df_decisiones['Cluster_Cliente'] == cluster
        valor_tp = save_rate * econ['ltv_rescate'] - econ['costo_alerta']
        valor_fp = -econ['costo_alerta']
        
        # Llenado Estrategia A
        df_decisiones.loc[c_mask & cond_tp_a, 'Beneficio_Ind_A'] = valor_tp
        df_decisiones.loc[c_mask & cond_fp_a, 'Beneficio_Ind_A'] = valor_fp
        
        # Llenado Estrategia B
        df_decisiones.loc[c_mask & cond_tp_b, 'Beneficio_Ind_B'] = valor_tp
        df_decisiones.loc[c_mask & cond_fp_b, 'Beneficio_Ind_B'] = valor_fp
        
    # 2. Ejecutar el Bootstrap vectorizado
    n_iterations = 10000
    ben_a = df_decisiones['Beneficio_Ind_A'].values
    ben_b = df_decisiones['Beneficio_Ind_B'].values
    
    indices_bootstrap = np.random.choice(len(ben_a), size=(n_iterations, len(ben_a)), replace=True)
    
    roi_sim_a = np.sum(ben_a[indices_bootstrap], axis=1)
    roi_sim_b = np.sum(ben_b[indices_bootstrap], axis=1)
    
    # 3. Impresión de Métricas Competitivas
    print("\n--- RESULTADOS MONTE CARLO (10,000 Escenarios) ---")
    print(f"ESTRATEGIA A (Segmentada) -> Promedio: ${np.mean(roi_sim_a):,.0f} | Peor: ${np.percentile(roi_sim_a, 2.5):,.0f} | Mejor: ${np.percentile(roi_sim_a, 97.5):,.0f}")
    print(f"ESTRATEGIA B (Agresiva)   -> Promedio: ${np.mean(roi_sim_b):,.0f} | Peor: ${np.percentile(roi_sim_b, 2.5):,.0f} | Mejor: ${np.percentile(roi_sim_b, 97.5):,.0f}")

    # Comparación pareada (ambas estrategias evaluadas sobre el MISMO remuestreo)
    diff = roi_sim_a - roi_sim_b
    prob_a_gana = np.mean(diff > 0)
    ic_bajo, ic_alto = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
    print(f"\nVEREDICTO -> P(Segmentada supera a Global): {prob_a_gana:.1%} | "
          f"Ventaja media de A sobre B: ${np.mean(diff):,.0f} "
          f"(IC95%: ${ic_bajo:,.0f} a ${ic_alto:,.0f})")
    if ic_bajo > 0:
        print("           La estrategia segmentada SUPERA de forma significativa a la campaña global.")
    elif np.mean(diff) >= 0:
        print("           Empate estadístico con ventaja para la segmentada: ya NO es batida por la campaña global,")
        print("           y la protege en el segmento caro (Súper Usuarios) sin perder en el resto.")
    else:
        print("           La campaña global sigue ganando: revisa save_rate / unit economics por clúster.")
    
    # 4. Gráfica de Distribuciones Solapadas
    plt.figure(figsize=(12, 7))
    sns.histplot(roi_sim_b, bins=60, color='#d62728', alpha=0.5, label='Estrategia B: Agresiva (Subóptima)')
    sns.histplot(roi_sim_a, bins=60, color='#1f77b4', alpha=0.8, label='Estrategia A: Segmentada (Nuestra)')
    
    # Líneas de media
    plt.axvline(np.mean(roi_sim_a), color='#1f77b4', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(roi_sim_b), color='#d62728', linestyle='dashed', linewidth=2)
    
    plt.title('Comparativa de Riesgo Financiero: Segmentada vs Agresiva Global\nSimulación Bootstrap Monte Carlo', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Beneficio Neto Retornado al Banco ($ USD)')
    plt.ylabel('Frecuencia (Simulaciones)')
    plt.legend(loc='upper right', facecolor='black', framealpha=0.8, fontsize=11)
    
    ruta_bootstrap = os.path.join(output_dir, "viz_riesgo_competitivo.png")
    plt.savefig(ruta_bootstrap, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[ÉXITO] Análisis de riesgo competitivo exportado a: {ruta_bootstrap}")
    
    return df_decisiones, unit_economics, umbrales_segmentados