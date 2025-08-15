# Backend RN SAC v2.0 - Predicción de Calificaciones Estudiantiles

## 🏗️ Arquitectura Implementada (Estándares de la Industria)

### ✅ **Nueva Arquitectura v2.0**
Este proyecto implementa una **arquitectura completa de Machine Learning** siguiendo los estándares de la industria y las mejores prácticas de desarrollo de software.

## 🔧 **Componentes de la Arquitectura**

### **1. Pipeline Completo de ML**
- **Frontend**: Envía datos originales (texto)
- **Backend**: Pipeline completo con preprocesamiento
- **Modelo**: Recibe datos ya procesados
- **Validación**: En cada capa del sistema

### **2. Separación de Responsabilidades (SOLID)**
```
Frontend (UI Layer):
    - Presentación de datos
    - Validación de entrada
    - Envío de datos originales

Backend (Business Logic Layer):
    - Preprocesamiento de datos
    - Lógica de negocio
    - Predicción del modelo

Model Layer:
    - Algoritmos de ML
    - Transformaciones de datos
```

### **3. Estándares Implementados**
- ✅ **MLOps**: Pipeline reproducible
- ✅ **Microservicios**: Separación clara de responsabilidades
- ✅ **API Design**: Endpoints RESTful con validación
- ✅ **Error Handling**: Manejo robusto de errores
- ✅ **Logging**: Sistema de logs completo
- ✅ **Documentation**: API documentada automáticamente

## 📁 **Estructura del Proyecto**

```
Backend_RN_Sac/
├── train_model.py          # Pipeline completo de entrenamiento
├── app.py                  # API FastAPI con arquitectura correcta
├── index.html              # Frontend con valores originales
├── diagnose.py             # Diagnóstico del sistema
├── requirements.txt        # Dependencias
├── README.md              # Documentación
├── modelo_entrenado1.h5   # Modelo entrenado
├── label_encoders.pkl     # Codificadores de etiquetas
├── scaler.pkl             # Escalador de datos
├── column_order.json      # Orden de características
└── model_info.json        # Información del modelo
```

## 🚀 **Instrucciones de Uso**

### **1. Diagnóstico del proyecto**
```bash
python diagnose.py
```

### **2. Entrenar el modelo (si es necesario)**
```bash
python train_model.py
```

### **3. Ejecutar la API**
```bash
python app.py
```

### **4. Acceder a la interfaz web**
Abrir `index.html` en el navegador

### **5. Documentación de la API**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔍 **Endpoints de la API**

### **GET /** - Información del servicio
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessors_loaded": true,
  "timestamp": "2024-01-01T12:00:00",
  "model_info": {
    "version": "2.0.0",
    "range_grades": "0.0 - 20.0",
    "features_count": 32
  }
}
```

### **GET /health** - Health check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessors_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

### **POST /predict** - Predicción
```json
{
  "features": ["GP", "F", "16", "U", "LE3", "T", "secondary", "secondary", ...]
}
```

**Respuesta:**
```json
{
  "predicted_grade": 12.5,
  "raw_prediction": 12.3,
  "was_clipped": false,
  "confidence": "high",
  "timestamp": "2024-01-01T12:00:00"
}
```

### **GET /model-info** - Información del modelo
```json
{
  "model_info": {...},
  "column_order": [...],
  "categorical_features": [...],
  "numerical_features": [...],
  "label_encoders_info": {...}
}
```

## 🎯 **Mejoras Implementadas**

### **1. Pipeline Completo**
- ✅ Preprocesamiento automático
- ✅ Validación de entrada
- ✅ Manejo de errores robusto
- ✅ Logging completo

### **2. Frontend Mejorado**
- ✅ Valores originales (texto)
- ✅ Validación de entrada
- ✅ Información de confianza
- ✅ Mejor UX

### **3. Backend Robusto**
- ✅ Arquitectura de microservicios
- ✅ Validación de datos
- ✅ Manejo de excepciones
- ✅ Documentación automática

### **4. Estándares de Calidad**
- ✅ Código limpio y mantenible
- ✅ Separación de responsabilidades
- ✅ Testing y diagnóstico
- ✅ Documentación completa

## 🔧 **Arquitectura Técnica**

### **Flujo de Datos**
```
Usuario (Frontend) → API (Backend) → Pipeline → Modelo → Respuesta
     ↓                    ↓              ↓         ↓         ↓
   Datos originales   → Validación   → Preproc. → Predic. → Resultado
   (GP, F, U, etc.)     y logging      (encoding + scaling)
```

### **Componentes del Pipeline**
1. **Validación de Entrada**: Verifica formato y valores
2. **Label Encoding**: Convierte texto a números
3. **Scaling**: Normaliza datos numéricos
4. **Predicción**: Modelo de red neuronal
5. **Validación de Salida**: Clipping y confianza

## 📊 **Métricas y Monitoreo**

### **Health Checks**
- Estado del modelo
- Estado de preprocesadores
- Tiempo de respuesta
- Errores y excepciones

### **Logging**
- Predicciones exitosas
- Errores de validación
- Errores internos
- Métricas de rendimiento

## 🛠️ **Desarrollo y Mantenimiento**

### **Agregar Nuevas Características**
1. Actualizar `train_model.py`
2. Regenerar modelo y metadatos
3. Actualizar validaciones en `app.py`
4. Actualizar frontend si es necesario

### **Testing**
```bash
# Diagnóstico completo
python diagnose.py

# Pruebas de predicción
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": ["GP", "F", "16", "U", ...]}'
```

## 📈 **Escalabilidad**

### **Horizontal Scaling**
- Múltiples instancias del servicio
- Load balancing
- Health checks automáticos

### **Versioning**
- Diferentes versiones del modelo
- Migración gradual
- Rollback capabilities

## ✅ **Conclusión**

Esta implementación cumple con los **estándares de la industria** y las **mejores prácticas** de desarrollo de software:

- ✅ **Arquitectura limpia y mantenible**
- ✅ **Pipeline completo de ML**
- ✅ **Validación robusta**
- ✅ **Manejo de errores**
- ✅ **Documentación completa**
- ✅ **Escalabilidad**
- ✅ **Monitoreo y logging**

El proyecto está listo para **producción** y puede ser **escalado** según las necesidades del negocio.
