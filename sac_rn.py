import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import pickle
import json
import numpy as np
import os

def regenerate_model():
    
    
    try:
        # Cargar datos
        df_math = pd.read_csv("../../dataset/student-mat.csv")
        df_port = pd.read_csv("../../dataset/student-por.csv")
        df1 = pd.concat([df_math, df_port], ignore_index=True)
        
        # Crear características derivadas
        df1['G_prev_avg'] = (df1['G1'] + df1['G2']) / 2
        df1['alcohol_total'] = df1['Dalc'] + df1['Walc']
        
        print(f"📋 Dataset cargado: {df1.shape[0]} filas, {df1.shape[1]} columnas")
        
        # Preparar datos
        x1 = df1.drop(['G1','G2','G3'], axis=1)
        y1 = df1['G3']
        
        print(f"🎯 Variable objetivo G3 - Rango: {y1.min()} a {y1.max()}")
        
        # Guardar el orden de las columnas
        column_order = x1.columns.tolist()
        with open('column_order.json', 'w') as f:
            json.dump(column_order, f)
        
        print(f"📝 Orden de columnas guardado: {len(column_order)} características")
        
        # Identificar columnas categóricas y numéricas
        categorias_columns1 = x1.select_dtypes(include=['object']).columns
        numerica_colums1 = x1.select_dtypes(include=['int64', 'float64']).columns
        
        print(f"🏷️  Columnas categóricas: {list(categorias_columns1)}")
        print(f"🔢 Columnas numéricas: {list(numerica_colums1)}")
        
        # Aplicar label encoding
        label_encoders1 = {}
        for col in categorias_columns1:
            le1 = LabelEncoder()
            x1[col] = le1.fit_transform(x1[col].astype(str))
            label_encoders1[col] = le1
            print(f"✅ Codificador para {col}: {list(le1.classes_)}")
        
        # Guardar los label encoders
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders1, f)
        
        print("💾 Label encoders guardados")
        
        # Crear bins para estratificación
        y_bins1 = pd.cut(y1, bins=5, labels=[1, 2, 3, 4, 5])
        
        # Split de datos
        x_train1, x_test1, y_train1, y_test1 = train_test_split(
            x1, y1, test_size=0.2, random_state=42, stratify=y_bins1
        )
        
        print(f"📊 Datos divididos: {x_train1.shape[0]} entrenamiento, {x_test1.shape[0]} prueba")
        
        # Escalado
        scaler = RobustScaler()
        x_train_scaled1 = scaler.fit_transform(x_train1)
        x_test_scaled1 = scaler.transform(x_test1)
        
        # Guardar el scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("💾 Scaler guardado")
        
        # Crear modelo
        def create_model1(input_dim):
            model1 = keras.Sequential([
                layers.Dense(128, activation='relu', input_dim=input_dim),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),

                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Dense(1, activation='linear')
            ])
            return model1
        
        model1 = create_model1(x_train1.shape[1])
        
        # Compilar modelo
        optimizar = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
        )
        
        model1.compile(
            optimizer=optimizar,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='loss',
                patience=25,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=10,
                verbose=1
            )
        ]
        
        # Entrenamiento
        print("🏋️  Iniciando entrenamiento...")
        start_time = time.time()
        
        history1 = model1.fit(
            x_train_scaled1, y_train1,
            epochs=300,
            batch_size=32,
            validation_split=0.2,
            callbacks=callback_list,
            verbose=1,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Tiempo de entrenamiento: {training_time:.2f} segundos")
        
        # Evaluación
        y_predict1 = model1.predict(x_test_scaled1).flatten()
        
        print(f"📊 Rango de predicciones: {y_predict1.min():.2f} a {y_predict1.max():.2f}")
        print(f"📊 Rango de valores reales: {y_test1.min():.2f} a {y_test1.max():.2f}")
        
        # Aplicar clipping
        y_predict1_clipped = np.clip(y_predict1, 0, 20)
        
        # Métricas
        mse1 = mean_squared_error(y_test1, y_predict1_clipped)
        mae1 = mean_absolute_error(y_test1, y_predict1_clipped)
        r21 = r2_score(y_test1, y_predict1_clipped)
        
        print(f"📈 Métricas de evaluación:")
        print(f"   MSE: {mse1:.4f}")
        print(f"   MAE: {mae1:.4f}")
        print(f"   R²: {r21:.4f}")
        
        # Guardar modelo
        model1.save('modelo_entrenado1.h5')
        
        # Guardar información del modelo
        model_info = {
            'min_grade': float(y1.min()),
            'max_grade': float(y1.max()),
            'mean_grade': float(y1.mean()),
            'std_grade': float(y1.std()),
            'training_time': training_time,
            'mse': mse1,
            'mae': mae1,
            'r2': r21
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f)
        
        print("✅ Modelo y metadatos regenerados exitosamente")
        print(f"📁 Archivos generados:")
        print(f"   - modelo_entrenado1.h5")
        print(f"   - label_encoders.pkl")
        print(f"   - scaler.pkl")
        print(f"   - column_order.json")
        print(f"   - model_info.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la regeneración: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    regenerate_model()
