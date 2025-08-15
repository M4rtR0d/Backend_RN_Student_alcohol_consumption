def predict_new_student(student_data):
    # Cargar configuración
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    TARGET_MAX = config['target_max']
    
    # Cargar modelo entrenado
    model = tf.keras.models.load_model('modelo_entrenado.h5')
    
    # Hacer predicción
    prediction_norm = model.predict(new_data_scaled)
    prediction_final = prediction_norm * TARGET_MAX  # Desnormalizar
    
    return prediction_final