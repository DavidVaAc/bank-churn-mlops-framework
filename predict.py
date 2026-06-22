import os
import argparse
import pandas as pd
import joblib

def ejecutar_inferencia_batch(ruta_entrada, ruta_salida):
    """
    Script operativo para la ejecución masiva y en lote de nuevos clientes
    utilizando el artefacto unificado de producción.
    """
    print("====================================================")
    print("      EXECUCCIÓN DE INFERENCIA EN LOTE (BATCH) ")
    print("====================================================\n")

    ruta_pipeline = "outputs/sistema_retencion_completo.joblib"

    # 1. Validación de infraestructura y artefactos
    if not os.path.exists(ruta_pipeline):
        print(f"[❌ ERROR] No se encontró el pipeline compilado en: '{ruta_pipeline}'")
        print("Por favor, ejecuta primero 'python main.py' para entrenar y generar el artefacto.")
        return

    if not os.path.exists(ruta_entrada):
        print(f"[❌ ERROR] No se encontró el archivo de clientes nuevos en: '{ruta_entrada}'")
        return

    # 2. Carga del ecosistema unificado en frío
    print(f"Cargando pipeline empaquetado desde {ruta_pipeline}...")
    pipeline = joblib.load(ruta_pipeline)
    print("✅ Ecosistema cargado y aislado estadísticamente.")

    # 3. Ingesta de los datos nuevos
    print(f"Leyendo lote de datos desde {ruta_entrada}...")
    df_nuevos = pd.read_csv(ruta_entrada)

    # 4. Inferencia protegida y automatizada (Maneja escalador, clústeres, LGBM y negocio)
    try:
        df_entregable = pipeline.procesar_e_inferir(df_nuevos)
        
        # 5. Exportación del reporte operativo de Marketing
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        df_entregable.to_csv(ruta_salida, index=False)
        
        print("\n====================================================")
        print(" [🔥 ÉXITO TOTAL] Lote procesado correctamente.")
        print(f" Clientes evaluados: {len(df_entregable)}")
        print(f" Reporte comercial exportado en: {ruta_salida}")
        print("====================================================")
        
    except Exception as e:
        print(f"\n[❌ CRÍTICO] El proceso falló debido a una violación del contrato de datos:")
        print(str(e))

if __name__ == "__main__":
    # Configuración de argumentos por consola para permitir automatización (ej. Tareas programadas / Cron Jobs)
    parser = argparse.ArgumentParser(description="Script de Inferencia Batch para Retención de Churn Bancario")
    parser.add_argument("--input", type=str, default="data/BankChurners.csv", help="Ruta del CSV de nuevos clientes")
    parser.add_argument("--output", type=str, default="outputs/plan_operativo_retencion.csv", help="Ruta de salida del reporte")
    
    args = parser.parse_args()
    ejecutar_inferencia_batch(args.input, args.output)