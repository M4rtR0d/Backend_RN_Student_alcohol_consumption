from fastapi import FastAPI,HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from websockets import Origin

# cargar el Modelo
name_model = 'modelo_entrenado1.h5'
modelo = tf.keras.models.load_model(name_model, compile=False)

app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance based on various features.",
    version="1.0.0" 
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las origines
    allow_credentials=True, # Permitir credenciales
    allow_methods=["*"],  # Permitir todos los métodos
    allow_headers=["*"],  # Permitir todos los encabezados
)

#Definir el esquema de entrada
class StudentData(BaseModel):
    features: list # Lista de características del estudiante

# Ruta de estado
@app.get("/status")
async def status():
    return {
        "status": True,
        "Modelo": name_model
    }

#Ruta de prediccion


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    features = data.get("features")
    if features is None:
        return {"prediction": None}
    # Convierte a float y asegura el shape correcto
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    predicted_grade = modelo.predict(features)
    # Si la predicción es un array, extrae el valor
    pred_value = float(predicted_grade[0][0])
    return {"predicted_grade": round(pred_value, 2)}
'''

@app.post("/predict")
def predict(data: StudentData):
    # Realizar la predicción
    try:
        # Convertir la lista de características a un array de numpy
        features = np.array(data.features, dtype=np.float32).reshape(1, -1)
        #realiza la prediccion
        prediction = modelo.predict(features)
        grade = float(prediction[0][0])

        return {"predicted_class": round(grade, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        https://student-alcohol-consumption-api.onrender.com
        '''