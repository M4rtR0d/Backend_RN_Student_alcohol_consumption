import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import pickle
import json
import numpy as np
import os

def create_neural_network(input_dim):
    """Crear modelo de red neuronal"""
    model = keras.Sequential([
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
    return model

def create_ml_pipeline():
    """Crear pipeline completo de ML siguiendo estándares de la industria"""
    
    print("🔄 Iniciando creación del pipeline de ML...")
    
    try:
        # Cargar datos
        print("📊 Cargando datos...")
        
        df1 = pd.read_csv("../../dataset/df_sac.csv")
       
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
        categorical_features = x1.select_dtypes(include=['object']).columns
        
        numerical_features = x1.select_dtypes(include=['int64', 'float64']).columns
        
        print(f"🏷️  Columnas categóricas: {categorical_features}")
        print(f"🔢 Columnas numéricas: {numerical_features}")
        
        # Crear preprocesadores
        categorical_transformer = Pipeline([
            ('ordinal_encoder', OrdinalEncoder())
            ])
        
        numerical_transformer = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        # Crear preprocesador completo
        preprocessor = ColumnTransformer([
            ('categorical', categorical_transformer, categorical_features),
            ('numerical', numerical_transformer, numerical_features)
        ])
        
        # Crear modelo de red neuronal
        neural_network = create_neural_network(len(column_order))
        
        # Compilar modelo
        neural_network.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', neural_network)
        ])
        
        print("✅ Pipeline creado exitosamente")
        
        # Split de datos
        x_train, x_test, y_train, y_test = train_test_split(
            x1, y1, test_size=0.2, random_state=42
        )
        
        print(f"📊 Datos divididos: {x_train.shape[0]} entrenamiento, {x_test.shape[0]} prueba")
        
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
        
        # Entrenar solo el preprocesador primero
        preprocessor.fit(x_train)
        
        # Transformar datos
        x_train_processed = preprocessor.transform(x_train)
        x_test_processed = preprocessor.transform(x_test)
        
        # Entrenar modelo
        history = neural_network.fit(
            x_train_processed, y_train,
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
        y_predict = neural_network.predict(x_test_processed).flatten()
        
        print(f"📊 Rango de predicciones: {y_predict.min():.2f} a {y_predict.max():.2f}")
        print(f"📊 Rango de valores reales: {y_test.min():.2f} a {y_test.max():.2f}")
        
        # Aplicar clipping
        y_predict_clipped = np.clip(y_predict, 0, 20)
        
        # Métricas
        mse = mean_squared_error(y_test, y_predict_clipped)
        mae = mean_absolute_error(y_test, y_predict_clipped)
        r2 = r2_score(y_test, y_predict_clipped)
        
        print(f"📈 Métricas de evaluación:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Guardar componentes por separado (para compatibilidad)
        neural_network.save('modelo_entrenado1.h5')
        
        # Extraer y guardar preprocesadores
        ordinal_encoder = {}
        for i, feature in enumerate(categorical_features):
            ordinal_encoder[feature] = preprocessor.named_transformers_['categorical'].named_steps['ordinal_encoder']
        
        with open('ordinal_encoders.pkl', 'wb') as f:
            pickle.dump(ordinal_encoder, f)
        
        scaler = preprocessor.named_transformers_['numerical'].named_steps['scaler']
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Guardar información del modelo
        model_info = {
            'min_grade': float(y1.min()),
            'max_grade': float(y1.max()),
            'mean_grade': float(y1.mean()),
            'std_grade': float(y1.std()),
            'training_time': training_time,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'categorical_features': categorical_features.tolist(),  # Convertir a lista
            'numerical_features': numerical_features.tolist() 
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)  # Agregué indent para mejor legibilidad
        
        print("✅ Pipeline y componentes guardados exitosamente")
        print(f"📁 Archivos generados:")
        print(f"   - modelo_entrenado1.h5")
        print(f"   - ordinal_encoders.pkl")
        print(f"   - scaler.pkl")
        print(f"   - column_order.json")
        print(f"   - model_info.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_ml_pipeline()
