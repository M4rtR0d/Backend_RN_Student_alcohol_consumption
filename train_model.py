import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import pickle
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_neural_network(input_dim, learning_rate=0.001, dropout_rates=[0.5, 0.4, 0.3]):
    """
    Args:
        input_dim: Dimensión de entrada
        learning_rate: Tasa de aprendizaje
        dropout_rates: Lista de tasas de dropout para cada capa
    """
    model = keras.Sequential([
        # Capa de entrada con mayor capacidad
        layers.Dense(256, activation='relu', input_dim=input_dim,
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[0]),
        
        # Capas ocultas con reducción gradual
        layers.Dense(128, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[1]),

        layers.Dense(64, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[2]),
        
        layers.Dense(32, activation='relu',
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Capa de salida
        layers.Dense(1, activation='linear')
    ])
    return model

def plot_training_history(history, save_path='training_history.png'):
    """Visualizar el historial de entrenamiento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # MAE
    axes[0,1].plot(history.history['mae'], label='Training MAE')
    axes[0,1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0,1].set_title('Model MAE')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # MSE
    axes[1,0].plot(history.history['mse'], label='Training MSE')
    axes[1,0].plot(history.history['val_mse'], label='Validation MSE')
    axes[1,0].set_title('Model MSE')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('MSE')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Learning Rate (si está disponible)
    if 'lr' in history.history:
        axes[1,1].plot(history.history['lr'])
        axes[1,1].set_title('Learning Rate')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True)
    else:
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_training_history(history):
    """Analizar el historial de entrenamiento de manera más completa"""
    print("\n📊 ANÁLISIS DETALLADO DEL ENTRENAMIENTO:")
    
    epochs_completed = len(history.history['loss'])
    print(f"   Épocas completadas: {epochs_completed}")
    
    # Mejor época basada en val_loss
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    
    print(f"   Mejor época: {best_epoch}")
    print(f"   Mejor val_loss: {best_val_loss:.4f}")
    
    # Métricas finales
    final_metrics = {
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'mae': history.history['mae'][-1],
        'val_mae': history.history['val_mae'][-1],
        'mse': history.history['mse'][-1],
        'val_mse': history.history['val_mse'][-1]
    }
    
    # Métricas en la mejor época
    best_metrics = {
        'loss': history.history['loss'][best_epoch-1],
        'val_loss': best_val_loss,
        'mae': history.history['mae'][best_epoch-1],
        'val_mae': history.history['val_mae'][best_epoch-1],
        'mse': history.history['mse'][best_epoch-1],
        'val_mse': history.history['val_mse'][best_epoch-1]
    }
    
    print(f"   Métricas finales:")
    for metric, value in final_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Análisis de overfitting/underfitting
    val_loss_trend = np.array(history.history['val_loss'][-10:])  # Últimas 10 épocas
    loss_trend = np.array(history.history['loss'][-10:])
    
    overfitting_ratio = final_metrics['val_loss'] / final_metrics['loss']
    
    if overfitting_ratio > 1.5:
        print("   ⚠️  Overfitting significativo detectado")
        status = "overfitting"
    elif overfitting_ratio > 1.2:
        print("   ⚠️  Ligero overfitting detectado")
        status = "slight_overfitting"
    elif final_metrics['loss'] > 0.8:  # Umbral para underfitting
        print("   ⚠️  Posible underfitting")
        status = "underfitting"
    else:
        print("   ✅ Buen balance bias-variance")
        status = "good_fit"
    
    # Estabilidad del entrenamiento
    val_loss_std = np.std(val_loss_trend)
    if val_loss_std < 0.01:
        print("   ✅ Entrenamiento estable")
        stability = "stable"
    elif val_loss_std < 0.05:
        print("   ⚠️  Entrenamiento moderadamente estable")
        stability = "moderate"
    else:
        print("   ❌ Entrenamiento inestable")
        stability = "unstable"
    
    # Información adicional para guardar
    training_info = {
        'epochs_completed': epochs_completed,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'best_metrics': {k: float(v) for k, v in best_metrics.items()},
        'overfitting_ratio': float(overfitting_ratio),
        'training_status': status,
        'stability': stability,
        'val_loss_std_last_10': float(val_loss_std),
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
            'mse': [float(x) for x in history.history['mse']],
            'val_mse': [float(x) for x in history.history['val_mse']]
        }
    }
    
    return training_info


def train_model():
    
    print("🔄 Iniciando Entrenamiento del modelo...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Cargar datos
        print("📊 Cargando datos...")
        
        if not os.path.exists("../../dataset/df_sac.csv"):
            print("❌ Error: No se encuentra el archivo de datos")
            return False
            
        df1 = pd.read_csv("../../dataset/df_sac.csv")
        
               
        # Crear características derivadas mejoradas
        df1['G_prev_avg'] = (df1['G1'] + df1['G2']) / 2
        df1['alcohol_total'] = df1['Dalc'] + df1['Walc']
        df1['study_time_total'] = df1.get('studytime', 0) + df1.get('traveltime', 0)
        df1['family_support'] = (df1.get('famsup', 0) == 'yes').astype(int)
        
        print(f"📋 Dataset cargado: {df1.shape[0]} filas, {df1.shape[1]} columnas")
        
        # Preparar datos
        target_cols = ['G1', 'G2', 'G3']
        feature_cols = [col for col in df1.columns if col not in target_cols]
        
        X = df1[feature_cols].copy()
        y = df1['G3'].copy()
        
        print(f"🎯 Variable objetivo G3 - Rango: {y.min()} a {y.max()}")
        print(f"🎯 Distribución G3 - Media: {y.mean():.2f}, Std: {y.std():.2f}")
        
        # Guardar el orden de las columnas
        column_order = X.columns.tolist()
        with open('column_order.json', 'w') as f:
            json.dump(column_order, f, indent=2)
        
        print(f"📝 Orden de columnas guardado: {len(column_order)} características")
        
        # Identificar columnas categóricas y numéricas
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"🏷️  Columnas categóricas ({len(categorical_features)}): {categorical_features}")
        print(f"🔢 Columnas numéricas ({len(numerical_features)}): {numerical_features}")
        
        # Codificación de variables categóricas
        label_encoders = {}
        X_encoded = X.copy()
        
        for col in categorical_features:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"✅ Codificador para {col}: {len(le.classes_)} clases únicas")
        
        # Crear bins para estratificación mejorada
        y_bins = pd.cut(y, bins=5, labels=[1, 2, 3, 4, 5])
        
        # Split de datos estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y_bins
        )
        
        print(f"📊 Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        print(f"📊 Distribución en entrenamiento - Media: {y_train.mean():.2f}")
        print(f"📊 Distribución en prueba - Media: {y_test.mean():.2f}")
        
        # Escalado con RobustScaler (mejor para outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("✅ Datos escalados con RobustScaler")
        
        # Crear modelo de red neuronal
        neural_network = create_neural_network(len(column_order))
        
        # Compilar modelo con métricas adicionales
        neural_network.compile(
            optimizer=optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("🧠 Modelo creado y compilado")
        print(f"📊 Parámetros del modelo: {neural_network.count_params():,}")
        
        # Callbacks mejorados
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenamiento
        print("🏋️  Iniciando entrenamiento...")
        start_time = time.time()
        
        history = neural_network.fit(
            X_train_scaled, y_train,
            epochs=300,
            batch_size=32,
            validation_split=0.2,
            callbacks=callback_list,
            verbose=1,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Tiempo de entrenamiento: {training_time:.2f} segundos")
        
        # Análisis completo del entrenamiento
        training_analysis = analyze_training_history(history)
        
        # Crear visualización del entrenamiento
        plot_training_history(history, 'training_history.png')
        print("📈 Gráficos de entrenamiento guardados")
        
        # Evaluación en conjunto de prueba
        print("\n📊 EVALUACIÓN FINAL:")
        y_pred = neural_network.predict(X_test_scaled).flatten()
        
        print(f"📊 Rango de predicciones: {y_pred.min():.2f} a {y_pred.max():.2f}")
        print(f"📊 Rango de valores reales: {y_test.min():.2f} a {y_test.max():.2f}")
        
        # Aplicar clipping para mantener predicciones en rango válido
        y_pred_clipped = np.clip(y_pred, 0, 20)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred_clipped)
        mae = mean_absolute_error(y_test, y_pred_clipped)
        r2 = r2_score(y_test, y_pred_clipped)
        rmse = np.sqrt(mse)
        
        print(f"📈 Métricas de evaluación:")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Guardar modelo y componentes
        model_filename = 'modelo_entrenado.h5'
        neural_network.save(model_filename)
        
        # Guardar preprocesadores
        encoders_filename = 'label_encoders.pkl'
        scaler_filename = 'scaler.pkl'
        
        with open(encoders_filename, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Información completa del modelo
        model_info = {
            'timestamp': timestamp,
            'model_version': '2.0',
            'dataset_info': {
                'total_samples': int(len(df1)),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test)),
                'n_features': int(len(column_order)),
                'categorical_features': categorical_features,
                'numerical_features': numerical_features
            },
            'target_statistics': {
                'min_grade': float(y.min()),
                'max_grade': float(y.max()),
                'mean_grade': float(y.mean()),
                'std_grade': float(y.std()),
                'median_grade': float(y.median())
            },
            'model_architecture': {
                'total_parameters': int(neural_network.count_params()),
                'layers': len(neural_network.layers),
                'optimizer': 'Adam',
                'loss_function': 'mse'
            },
            'training_info': training_analysis,
            'performance_metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'training_time_seconds': float(training_time)
            },
            'files_generated': {
                'model': model_filename,
                'encoders': encoders_filename,
                'scaler': scaler_filename,
                'column_order': 'column_order.json',
                'training_plot': 'training_history.png'
            }
        }
        
        info_filename = 'model_info.json'
        with open(info_filename, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print("\n✅ Pipeline completado exitosamente")
        print(f"📁 Archivos generados:")
        for key, filename in model_info['files_generated'].items():
            print(f"   - {filename}")
        print(f"   - {info_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configurar GPU si está disponible
    if tf.config.list_physical_devices('GPU'):
        print("🚀 GPU detectada, habilitando crecimiento de memoria")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error configurando GPU: {e}")

    train_model()