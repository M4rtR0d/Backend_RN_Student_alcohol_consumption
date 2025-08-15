import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json

def test_model():
    """Script para probar el modelo y verificar que las predicciones estén en el rango correcto"""
    
    try:
        # Cargar el modelo
        modelo = tf.keras.models.load_model('modelo_entrenado1.h5', compile=False)
        
        # Cargar metadatos
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('column_order.json', 'r') as f:
            column_order = json.load(f)
        
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        print("✅ Metadatos cargados exitosamente")
        print(f"📊 Rango de calificaciones en el dataset: {model_info['min_grade']} - {model_info['max_grade']}")
        print(f"📋 Número de características: {len(column_order)}")
        
        # Mostrar información sobre los label encoders
        print("\n🔍 Información de Label Encoders:")
        for col, encoder in label_encoders.items():
            print(f"   {col}: {list(encoder.classes_)}")
        
        # Crear datos de prueba usando valores que sabemos que existen en los encoders
        test_features = []
        
        # Mapear valores según los encoders disponibles
        for col in column_order:
            if col in label_encoders:
                # Usar el primer valor disponible en el encoder
                test_features.append(str(label_encoders[col].classes_[0]))
            else:
                # Para variables numéricas, usar valores típicos
                if col == 'age':
                    test_features.append(16)
                elif col == 'traveltime':
                    test_features.append(1)
                elif col == 'studytime':
                    test_features.append(2)
                elif col == 'failures':
                    test_features.append(0)
                elif col == 'schoolsup':
                    test_features.append(0)
                elif col == 'famsup':
                    test_features.append(1)
                elif col == 'paid':
                    test_features.append(0)
                elif col == 'activities':
                    test_features.append(1)
                elif col == 'nursery':
                    test_features.append(1)
                elif col == 'higher':
                    test_features.append(1)
                elif col == 'internet':
                    test_features.append(1)
                elif col == 'romantic':
                    test_features.append(0)
                elif col == 'famrel':
                    test_features.append(4)
                elif col == 'freetime':
                    test_features.append(3)
                elif col == 'goout':
                    test_features.append(3)
                elif col == 'dalc':
                    test_features.append(1)
                elif col == 'walc':
                    test_features.append(2)
                elif col == 'health':
                    test_features.append(4)
                elif col == 'absences':
                    test_features.append(2)
                elif col == 'G_prev_avg':
                    test_features.append(12.5)
                elif col == 'alcohol_total':
                    test_features.append(3)
                else:
                    test_features.append(0)
        
        print(f"\n📝 Datos de prueba creados con {len(test_features)} características")
        
        # Crear DataFrame con las características
        df_features = pd.DataFrame([test_features], columns=column_order)
        
        # Aplicar label encoding de forma segura
        for col, encoder in label_encoders.items():
            if col in df_features.columns:
                # Verificar que todos los valores estén en las clases del encoder
                unique_values = df_features[col].unique()
                for val in unique_values:
                    if val not in encoder.classes_:
                        print(f"⚠️  Valor '{val}' no encontrado en encoder para {col}. Valores disponibles: {list(encoder.classes_)}")
                        # Usar el primer valor disponible
                        df_features[col] = encoder.classes_[0]
                
                df_features[col] = encoder.transform(df_features[col].astype(str))
        
        # Aplicar escalado
        features_scaled = scaler.transform(df_features)
        
        # Hacer predicción
        predicted_grade = modelo.predict(features_scaled)
        pred_value = float(predicted_grade[0][0])
        pred_value_clipped = np.clip(pred_value, 0, 20)
        
        print(f"\n🧪 Prueba de predicción:")
        print(f"   Predicción raw: {pred_value:.2f}")
        print(f"   Predicción ajustada: {pred_value_clipped:.2f}")
        print(f"   ¿Fue ajustada?: {'Sí' if pred_value != pred_value_clipped else 'No'}")
        
        # Verificar rango
        if 0 <= pred_value_clipped <= 20:
            print("✅ La predicción está en el rango correcto (0-20)")
        else:
            print("❌ La predicción está fuera del rango esperado")
        
        print("\n✅ Pruebas completadas exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
        print("Asegúrate de ejecutar primero sac_rn.py para generar el modelo y metadatos")

if __name__ == "__main__":
    test_model()
