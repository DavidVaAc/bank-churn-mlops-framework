import pandas as pd
from sklearn.model_selection import train_test_split

def cargar_y_dividir_datos(ruta_datos):
    """
    Carga el dataset bancario, limpia las columnas irrelevantes
    y genera un split estratificado libre de Data Leakage.
    """
    print("Cargando y limpiando el conjunto de datos original...")
    df = pd.read_csv(ruta_datos)
    
    # Eliminamos CLIENTNUM y las columnas basura de Naive Bayes (The Kaggle Trap)
    cols_to_drop = [
        'CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Convertimos Attrition_Flag a la variable objetivo binaria
    df['Target_Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    
    # Split en TRES vías (60/20/20) con estratificación ANTES de cualquier transformación.
    # La validación se reserva para calibrar los umbrales del motor de decisiones,
    # dejando el test completamente intacto para la evaluación final (anti-leakage).
    X = df.drop(columns=['Target_Churn'])
    y = df['Target_Churn']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"[ÉXITO] Datos divididos. Entrenamiento: {X_train.shape} | "
          f"Validación: {X_val.shape} | Prueba: {X_test.shape}")
    return df, X_train, X_val, X_test, y_train, y_val, y_test