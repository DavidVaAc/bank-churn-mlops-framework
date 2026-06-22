import pandas as pd
import numpy as np

class SistemaRetencionBancaria:
    def __init__(self, scaler, kmeans, mapa_clusters, model, unit_economics, umbrales, X_val=None, y_val=None):
        """
        Ecosistema Unificado de Inferencia Operativa para la Retención de Churn.
        Encapsula el preprocesamiento, la segmentación geométrica canónica,
        la clasificación por Gradient Boosting y el motor de reglas asimétricas.
        """
        self.scaler = scaler
        self.kmeans = kmeans
        self.mapa_clusters = mapa_clusters
        self.model = model
        self.unit_economics = unit_economics
        self.umbrales = umbrales
        self.X_val = X_val  
        self.y_val = y_val
        
        # Variables que el modelo necesita para el clustering
        self.features_clustering = [
            'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio',
            'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1'
        ]
        
        # Variables categóricas que el LightGBM espera
        self.categorical_features = [
            'Gender', 'Education_Level', 'Marital_Status', 
            'Income_Category', 'Card_Category', 'Cluster_Cliente'
        ]

        # ==============================================================
        # CONTRATO DE ORDEN DEL MODELO (Esquema estricto de producción)
        # ==============================================================
        # Este es el orden EXACTO con el que se entrenó tu LightGBM (sin CLIENTNUM ni Attrition_Flag)
        self.model_features = [
            'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 
            'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book', 
            'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 
            'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 
            'Cluster_Cliente' # <-- Tu feature ingeniada va al final
        ]

    def procesar_e_inferir(self, df_crudo):
        """
        Recibe un DataFrame masivo de clientes nuevos y ejecuta el pipeline completo en frío.
        
        Parámetros:
            df_crudo (pd.DataFrame): Datos sin procesar extraídos directamente del core del banco.
                                     Debe contener las variables demográficas y transaccionales.
        Retorna:
            pd.DataFrame: Reporte limpio con la segmentación, probabilidad de fuga 
                          y la estrategia comercial recomendada.
        """

        # =========================================================================
        # VALIDACIÓN AUTOMÁTICA DEL CONTRATO DE DATOS (Instructivo integrado)
        # =========================================================================
        # Las variables necesarias para operar (todas las del modelo excepto la que nosotros inventamos)
        features_requeridas = [f for f in self.model_features if f != 'Cluster_Cliente'] + ['CLIENTNUM']
        
        columnas_faltantes = [col for col in features_requeridas if col not in df_crudo.columns]
        
        if columnas_faltantes:
            raise ValueError(
                f"\n[ERROR DE CONTRATO DE DATOS] El archivo provisto no cumple con el esquema requerido.\n"
                f"Faltan las siguientes {len(columnas_faltantes)} columnas obligatorias: {columnas_faltantes}\n"
                f"Por favor, verifica el manual de extracción del Core Bancario."
            )
        
        print(f"Procesando lote de producción ({len(df_crudo)} clientes)...")
        df_proc = df_crudo.copy()
        
        # 1. Limpieza idéntica a la de entrenamiento
        cols_to_drop = ['CLIENTNUM', 'Attrition_Flag']
        X = df_proc.drop(columns=cols_to_drop, errors='ignore')
        
        # 2. Transformación geométrica usando el StandardScaler CONGELADO (Anti-Leakage)
        X_scaled = self.scaler.transform(X[self.features_clustering])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.features_clustering, index=X.index)
        
        # 3. Predicción y Remapeo Canónico de Clústeres en frío
        raw_pred_clusters = self.kmeans.predict(X_scaled_df)
        remap_func = np.vectorize(self.mapa_clusters.get)
        X['Cluster_Cliente'] = remap_func(raw_pred_clusters)
        
        # 4. Forzado de tipos categóricos nativos para LightGBM
        for col in self.categorical_features:
            X[col] = X[col].astype('category')

        # ==================================
        # BLINDAJE DE ALINEACIÓN DE COLUMNAS
        # ==================================
        # Reordenamos el DataFrame X usando la lista estricta. No importa cómo venía el archivo, 
        # aquí se formatea exactamente al Layout requerido por el LightGBM.
        X = X[self.model_features]
            
        # 5. Inferencia Probabilística del Gradient Boosting
        X['Probabilidad_Churn'] = self.model.predict_proba(X)[:, 1]
        
        # 6. Motor Cost-Sensitive (Aplicación de umbrales óptimos calculados en validación)
        X['Prediccion_Final'] = 0
        X['Estrategia_Asignada'] = "Sin Acción (Cliente Estable)"
        
        for cluster, umbral in self.umbrales.items():
            cond_alarma = (X['Cluster_Cliente'] == cluster) & (X['Probabilidad_Churn'] > umbral)
            X.loc[cond_alarma, 'Prediccion_Final'] = 1
            
            if cluster == 1:
                X.loc[cond_alarma, 'Estrategia_Asignada'] = "C1: ALERTA PREMIUM (Contacto Humano / Condonación)"
            elif cluster == 3:
                X.loc[cond_alarma, 'Estrategia_Asignada'] = "C3: CAMPAÑA MASIVA (Email Automático / MSI)"
            elif cluster == 0:
                X.loc[cond_alarma, 'Estrategia_Asignada'] = "C0: ALIVIO FINANCIERO (Descuento de Tasa / Pagos Fijos)"
            elif cluster == 2:
                X.loc[cond_alarma, 'Estrategia_Asignada'] = "C2: DESPERTAR VIP (Bono de Cashback / Upgrade Tarjeta)"
                
        # Columnas limpias que necesita el equipo de operaciones de marketing
        cols_entrega = ['Cluster_Cliente', 'Probabilidad_Churn', 'Prediccion_Final', 'Estrategia_Asignada']

        df_entregable = X[cols_entrega].copy()
        
        # 3. Reinyectamos el identificador al inicio (posición 0) mapeando con el dataframe crudo
        if 'CLIENTNUM' in df_crudo.columns:
            df_entregable.insert(0, 'CLIENTNUM', df_crudo['CLIENTNUM'])
            
        return df_entregable