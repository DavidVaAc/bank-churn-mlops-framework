import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _mapa_canonico_clusters(X_train_orig, raw_labels):
    """
    Asigna IDs canónicos ESTABLES a los clústeres según su arquetipo conductual,
    haciendo el pipeline robusto a la permutación arbitraria de etiquetas de K-Means
    (que cambia cada vez que se reentrena con una partición distinta).

    Mapeo fijo que mantiene la coherencia con unit_economics del motor de decisiones:
      0 = Financiados   (mayor Avg_Utilization_Ratio)
      1 = Súper Usuarios(mayor Total_Trans_Amt)
      2 = VIP Pasivos   (mayor Credit_Limit)
      3 = Alerta Roja   (segmento restante)
    """
    centros = X_train_orig.copy()
    centros['_raw'] = np.asarray(raw_labels)
    centros = centros.groupby('_raw').mean()

    disponibles = list(centros.index)
    mapa = {}

    super_u = centros.loc[disponibles, 'Total_Trans_Amt'].idxmax()
    mapa[super_u] = 1; disponibles.remove(super_u)

    vip = centros.loc[disponibles, 'Credit_Limit'].idxmax()
    mapa[vip] = 2; disponibles.remove(vip)

    financ = centros.loc[disponibles, 'Avg_Utilization_Ratio'].idxmax()
    mapa[financ] = 0; disponibles.remove(financ)

    mapa[disponibles[0]] = 3
    return mapa


def entrenar_clustering_y_perfilamiento(df_original, X_train, X_val, X_test, output_dir="outputs"):
    """
    Entrena el StandardScaler y el modelo K-Means (K=4) usando únicamente el Train Set.
    Asigna las etiquetas a todo el portafolio y exporta el perfilamiento de negocio.
    """
    print("\nIniciando el proceso de segmentación conductual sin Data Leakage...")
    
    features_for_clustering = [
        'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio',
        'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1'
    ]
    
    # 1. Estandarización defensiva ajustada únicamente con Train Set
    scaler = StandardScaler()
    scaler.fit(X_train[features_for_clustering])
    
    X_train_scaled = scaler.transform(X_train[features_for_clustering])
    X_val_scaled = scaler.transform(X_val[features_for_clustering])
    X_test_scaled = scaler.transform(X_test[features_for_clustering])
    
    # 2. Convertimos de vuelta a DataFrame heredando los índices originales para evitar desalineación
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features_for_clustering, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features_for_clustering, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features_for_clustering, index=X_test.index)
    
    # 3. Entrenamiento blindado de K-Means (K=4)
    print("Ajustando centroides de K-Means con el conjunto de entrenamiento (K=4)...")
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_train_scaled_df)
    
    # 3b. Relabeling canónico por arquetipo (estable ante permutaciones de K-Means)
    raw_train = kmeans.predict(X_train_scaled_df)
    mapa = _mapa_canonico_clusters(X_train[features_for_clustering], raw_train)
    remap = np.vectorize(mapa.get)
    
    # 4. Asignación global de clústeres mediante predicción controlada (ya remapeada)
    df_original['Cluster_Cliente'] = remap(kmeans.predict(
            pd.DataFrame(scaler.transform(df_original[features_for_clustering]), columns=features_for_clustering)
        ))
    X_train['Cluster_Cliente'] = remap(raw_train)
    X_val['Cluster_Cliente'] = remap(kmeans.predict(X_val_scaled_df))
    X_test['Cluster_Cliente'] = remap(kmeans.predict(X_test_scaled_df))
    
    # 5. Generación del reporte de perfilamiento de negocio (exclusivo con índices de Train)
    columnas_analisis = [
        'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio',
        'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Target_Churn'
    ]
    
    perfiles = df_original.loc[X_train.index].groupby('Cluster_Cliente')[columnas_analisis].mean().reset_index()
    
    conteo_clientes = df_original.loc[X_train.index, 'Cluster_Cliente'].value_counts().reset_index()
    conteo_clientes.columns = ['Cluster_Cliente', 'Cantidad_Clientes']
    conteo_clientes['Porcentaje_Total'] = (conteo_clientes['Cantidad_Clientes'] / len(X_train)) * 100
    
    perfiles_completos = pd.merge(perfiles, conteo_clientes, on='Cluster_Cliente').round(2)
    
    os.makedirs(output_dir, exist_ok=True)
    ruta_csv = os.path.join(output_dir, "data_perfil_segmentos_final.csv")
    perfiles_completos.to_csv(ruta_csv, index=False)
    
    print(f"[ÉXITO] Perfil de los segmentos exportado a: {ruta_csv}")
    return df_original, X_train, X_val, X_test, mapa, scaler, kmeans