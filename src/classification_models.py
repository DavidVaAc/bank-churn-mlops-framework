import os
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def entrenar_y_evaluar_lgbm(df_etiquetado, train_index, val_index, test_index, y_train, y_test, output_dir="outputs"):
    """
    Configura tipos de variables categóricas, entrena LightGBM de forma nativa
    y exporta las curvas de optimización interactiva de Plotly en HTML.
    """
    columnas_irrelevantes = ['Attrition_Flag']
    X = df_etiquetado.drop(columns=columnas_irrelevantes + ['Target_Churn'], errors='ignore')
    
    categorical_features = [
        'Gender', 'Education_Level', 'Marital_Status', 
        'Income_Category', 'Card_Category', 'Cluster_Cliente'
    ]
    
    # Forzado de tipo categórico para el truco de producción de LightGBM
    for col in categorical_features:
        X[col] = X[col].astype('category')
        
    X_train_final = X.loc[train_index]
    X_val_final = X.loc[val_index]
    X_test_final = X.loc[test_index]
    
    print("\nEntrenando clasificador LightGBM con manejo nativo de categóricas...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    
    clf.fit(
        X_train_final, 
        y_train,
        feature_name='auto',
        categorical_feature=categorical_features
    )
    print("[ÉXITO] Modelo de Gradient Boosting entrenado correctamente.")
    
    # Evaluación del modelo base
    proba_test = clf.predict_proba(X_test_final)[:, 1]
    
    # Cálculo de curvas Precision-Recall e impacto del umbral
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, proba_test)
    df_plot = pd.DataFrame({
        'recall': recall[:-1],
        'precision': precision[:-1],
        'threshold': thresholds
    })
    df_plot['f1'] = (2 * df_plot['precision'] * df_plot['recall']) / (df_plot['precision'] + df_plot['recall'])
    df_plot['f1'] = df_plot['f1'].fillna(0)
    
    # Exportación segura de entregables interactivos en HTML
    os.makedirs(output_dir, exist_ok=True)
    
    fig_pr = px.scatter(
        df_plot, x='recall', y='precision', color='f1',
        title='Explorador de Umbrales: Precision vs Recall',
        labels={'recall': 'Recall (Sensibilidad)', 'precision': 'Precisión', 'threshold': 'Umbral'}
    )
    fig_pr.update_layout(plot_bgcolor='rgba(0,0,0,1)', paper_bgcolor='rgba(0,0,0,1)', font_color="white")
    fig_pr.write_html(os.path.join(output_dir, "interactive_precision_recall.html"))
    
    fig_tuning = px.scatter(
        df_plot, x='threshold', y=['precision', 'recall', 'f1'],
        title='Impacto del Umbral en Precision, Recall y F1 Score',
        labels={'value': 'Puntuación', 'threshold': 'Umbral (Threshold)', 'variable': 'Métrica'}
    )
    fig_tuning.update_layout(plot_bgcolor='rgba(0,0,0,1)', paper_bgcolor='rgba(0,0,0,1)', font_color="white")
    fig_tuning.write_html(os.path.join(output_dir, "interactive_tuning_umbrales.html"))    

    # ROC Curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test,proba_test)
    df_roc = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    })

    df_roc = df_roc.merge(df_plot[['threshold', 'f1']], on='threshold', how='left')

    fig_roc = px.scatter(
        df_roc, 
        x='fpr', 
        y='tpr',
        color='f1',
        color_continuous_scale='Viridis',
        title='Curva ROC',
        labels={'fpr': 'Tasa de Falsos Positivos (FPR)', 'tpr': 'Tasa de Verdaderos Positivos (TPR)', 'threshold': 'Umbral'},
        hover_data={'threshold': ':.2%', 'tpr': ':.2%', 'fpr': ':.2%', 'f1': ':.2%'}
    )

    fig_roc.update_layout(plot_bgcolor='rgba(0,0,0,1)',
                            paper_bgcolor='rgba(0,0,0,1)',
                            font_color="white",
                            margin=dict(t=50, b=50, l=50, r=50)
                            )
    fig_roc.write_html(os.path.join(output_dir, "interactive_roc_curve.html"))

    print("[ÉXITO] Reportes interactivos de Plotly exportados en formato HTML.")
    
    # =========================================================================
    #                  FEATURE IMPORTANCE (REPORTE VISUAL)
    # =========================================================================
    importances = clf.feature_importances_
    features = X_train_final.columns
    df_importance = pd.DataFrame({'Variable': features, 'Importancia': importances})
    df_importance = df_importance.sort_values(by='Importancia', ascending=False)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    # Graficamos las variables más importantes
    sns.barplot(
        x='Importancia', 
        y='Variable', 
        data=df_importance, 
        palette='cool', 
        hue='Importancia',
        legend=False
        )
    plt.title('Top 12 Variables Clave para la Predicción de Churn', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Importancia (Gain / Split Count)')
    plt.ylabel('Variable Conductual')
    plt.tight_layout()
    
    ruta_importance = os.path.join(output_dir, "viz_feature_importance.png")
    plt.savefig(ruta_importance, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ÉXITO] Gráfica de Feature Importance exportada en: {ruta_importance}")


    # Creamos el objeto explainer de SHAP para LightGBM
    explainer = shap.TreeExplainer(clf)
    # Calculamos los valores SHAP para el conjunto de validación/test
    shap_values = explainer.shap_values(X_test_final)
    # Graficamos el resumen global de SHAP para las 10 variables más importantes
    shap.summary_plot(shap_values, X_test_final, show=False)
    plt.title('Impacto Global de las Variables en la Predicción de Churn', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    ruta_shap = os.path.join(output_dir, "viz_shap_summary.png")
    plt.savefig(ruta_shap, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ÉXITO] Gráfica de resumen SHAP exportada en: {ruta_shap}")
    
    return clf, X_train_final, X_val_final, X_test_final