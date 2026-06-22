import os
import sys

import joblib

# Forzamos la inclusión del path para evitar errores de importación locales
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_processing import cargar_y_dividir_datos
from src.clustering_models import entrenar_clustering_y_perfilamiento
from src.classification_models import entrenar_y_evaluar_lgbm
from src.decision_engine import generar_lista_estrategias
from src.inference import SistemaRetencionBancaria

def ejecutar_pipeline():
    print("====================================================")
    print("  INICIANDO PIPELINE DE RETENCIÓN DE CHURN BANCARIO  ")
    print("====================================================\n")
    
    ruta_csv = "data/BankChurners.csv"
    output_dir = "outputs"
    
    # Validación defensiva de infraestructura
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encontró el archivo de datos en '{ruta_csv}'.")
        print("Por favor, crea la carpeta data/ y coloca el archivo original ahí.")
        return
        
    # Paso 1: Carga, Limpieza y Partición (Anti-Leakage)
    df, X_train, X_val, X_test, y_train, y_val, y_test = cargar_y_dividir_datos(ruta_csv)
    
    # Paso 2: Clustering sin Leakage y Perfilamiento
    df_etiquetado, X_train, X_val, X_test, mapa, scaler, kmeans = entrenar_clustering_y_perfilamiento(
        df, X_train, X_val, X_test, output_dir=output_dir
    )
    
    # Paso 3: Clasificación supervisada con LightGBM Nativo
    clf, X_train_final, X_val_final, X_test_final = entrenar_y_evaluar_lgbm(
        df_etiquetado, X_train.index, X_val.index, X_test.index, y_train, y_test, output_dir=output_dir
    )
    
    # Paso 4: Ejecución del Motor de Decisiones Asimétrico (cost-sensitive).
    # Los umbrales se calibran en VALIDACIÓN y se evalúan sobre TEST intacto.
    df_decisiones, unit_economics, umbrales_segmentados = generar_lista_estrategias(
        X_val_final, y_val, X_test_final, y_test, clf, save_rate=0.30, output_dir=output_dir
    )
    
    # Paso 5: Empaquetado del Ecosistema de Inferencia para Producción
    srb = SistemaRetencionBancaria(
        scaler=scaler, 
        kmeans=kmeans, 
        mapa_clusters=mapa, 
        model=clf, 
        unit_economics=unit_economics, 
        umbrales=umbrales_segmentados,
        X_val=X_val_final,  
        y_val=y_val         
    )
    
    joblib.dump(srb, "outputs/sistema_retencion_completo.joblib")
    print("[MIGRACIÓN EXITOSA] Ecosistema de inferencia empaquetado para producción.")

    print("\n====================================================")
    print(" [ÉXITO TOTAL] Pipeline ejecutado de principio a fin.")
    print(" Todos los entregables e imágenes consolidados en /outputs.")
    print("====================================================")

if __name__ == "__main__":
    ejecutar_pipeline()