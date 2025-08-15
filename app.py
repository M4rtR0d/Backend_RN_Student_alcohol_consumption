import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el Modelo y metadatos
try:
name_model = 'modelo_entrenado1.h5'
modelo = tf.keras.models.load_model(name_model, compile=False)

    # Cargar metadatos del preprocesamiento
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('column_order.json', 'r') as f:
        column_order = json.load(f)
    
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
        
    logger.info("✅ Metadatos cargados exitosamente")
    logger.info(f"📊 Rango de calificaciones en el dataset: {model_info['min_grade']} - {model_info['max_grade']}")
    
except FileNotFoundError as e:
    logger.error(f"❌ Error: No se encontraron los archivos de metadatos: {e}")
    logger.error("Asegúrate de ejecutar primero train_model.py para generar los metadatos")
    raise

# Configurar FastAPI
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance based on various features following industry standards.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir esquemas de datos
class StudentData(BaseModel):
    features: List[str]  # Lista de características del estudiante (valores originales)

class PredictionResponse(BaseModel):
    predicted_grade: float
    raw_prediction: float
    was_clipped: bool
    confidence: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessors_loaded: bool
    timestamp: str
    model_info: dict

class PredictionPipeline:
    """Pipeline completo de predicción siguiendo estándares de ML"""
    
    def __init__(self, model, label_encoders, scaler, column_order, model_info):
        self.model = model
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.column_order = column_order
        self.model_info = model_info
        self.categorical_features = model_info.get('categorical_features', [])
        self.numerical_features = model_info.get('numerical_features', [])
    
    def validate_input(self, features: List[str]) -> bool:
        """Validar entrada de datos"""
        if len(features) != len(self.column_order):
            raise ValueError(f"Se esperaban {len(self.column_order)} características, se recibieron {len(features)}")
        
        # Validar valores categóricos
        for i, feature in enumerate(self.column_order):
            if feature in self.categorical_features:
                if features[i] not in self.label_encoders[feature].classes_:
                    raise ValueError(f"Valor '{features[i]}' no válido para {feature}. Valores permitidos: {list(self.label_encoders[feature].classes_)}")
        
        return True
    
    def preprocess(self, raw_data: List[str]) -> np.ndarray:
        """Preprocesamiento completo de datos"""
        try:
            # 1. Crear DataFrame con orden correcto
            df = pd.DataFrame([raw_data], columns=self.column_order)
            
            # 2. Aplicar label encoding a variables categóricas
            for col in self.categorical_features:
                if col in df.columns:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
            # 3. Aplicar escalado
            features_scaled = self.scaler.transform(df)
            
            return features_scaled
            
        except Exception as e:
            raise ValueError(f"Error en preprocesamiento: {str(e)}")
    
    def predict(self, raw_data: List[str]) -> dict:
        """Predicción completa con validación"""
        try:
            # Validar entrada
            self.validate_input(raw_data)
            
            # Preprocesamiento
            features_scaled = self.preprocess(raw_data)
            
            # Predicción
            prediction = self.model.predict(features_scaled, verbose=0)
            pred_value = float(prediction[0][0])
            
            # Validación de rango
            pred_value_clipped = np.clip(pred_value, 0, 20)
            
            # Determinar confianza
            confidence = "high"
            if pred_value != pred_value_clipped:
                confidence = "adjusted"
            elif pred_value < 5 or pred_value > 15:
                confidence = "low"
            
            return {
                "predicted_grade": round(pred_value_clipped, 2),
                "raw_prediction": round(pred_value, 2),
                "was_clipped": pred_value != pred_value_clipped,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ValueError(f"Error en predicción: {str(e)}")

# Inicializar pipeline
pipeline = PredictionPipeline(modelo, label_encoders, scaler, column_order, model_info)

# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessors_loaded": True,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "version": "2.0.0",
            "range_grades": f"{model_info['min_grade']} - {model_info['max_grade']}",
            "features_count": len(column_order),
            "categorical_features": len(model_info.get('categorical_features', [])),
            "numerical_features": len(model_info.get('numerical_features', []))
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessors_loaded": True,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "version": "2.0.0",
            "range_grades": f"{model_info['min_grade']} - {model_info['max_grade']}",
            "features_count": len(column_order)
        }
    }

@app.get("/status")
async def status():
    """Endpoint de estado (compatibilidad)"""
    return {
        "status": True,
        "Modelo": name_model,
        "Rango_calificaciones": f"{model_info['min_grade']} - {model_info['max_grade']}",
        "version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    """Endpoint de predicción con validación completa"""
    try:
        # Obtener datos
    data = await request.json()
    features = data.get("features")
        
        if not features:
            raise HTTPException(
                status_code=400, 
                detail="No se proporcionaron características"
            )
        
        # Validar tipo de datos
        if not isinstance(features, list):
            raise HTTPException(
                status_code=400, 
                detail="Las características deben ser una lista"
            )
        
        # Predicción usando pipeline
        result = pipeline.predict(features)
        
        logger.info(f"✅ Predicción exitosa: {result['predicted_grade']}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ Error de validación: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ Error interno: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Obtener información detallada del modelo"""
    return {
        "model_info": model_info,
        "column_order": column_order,
        "categorical_features": model_info.get('categorical_features', []),
        "numerical_features": model_info.get('numerical_features', []),
        "label_encoders_info": {
            feature: list(encoder.classes_) 
            for feature, encoder in label_encoders.items()
        }
    }

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejador global de excepciones"""
    logger.error(f"❌ Error no manejado: {str(exc)}")
    return {
        "error": "Error interno del servidor",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
