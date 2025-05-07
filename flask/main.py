from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib  # для загрузки скейлера, если вы его тоже сохраняете

app = Flask(__name__)

# Загружаем модель и скейлер
neiro_model = load_model('neiro_model.h5')
neiro_scaler = joblib.load('neiro_scaler.pkl')  # если вы сохраняете MinMaxScaler

ml_model = joblib.load('stacking_pipeline.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/neiro_predict', methods=['GET', 'POST'])
def neiro_predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Получаем данные из формы
            print(list(request.form.values()))
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)
            features_scaled = neiro_scaler.transform(features)
            prediction = neiro_model.predict(features_scaled)
            prediction = prediction.flatten()[0]  # Преобразовать результат в число
        except Exception as e:
            prediction = f"Ошибка: {str(e)}"
    
    return render_template('neiro.html', prediction=prediction)

@app.route('/ml_predict', methods=['GET', 'POST'])
def ml_predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Получаем данные из формы
            print(list(request.form.values()))
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)

            result = ml_model.predict(features)
            prediction = {
                "Модуль упругости при растяжении, ГПа": round(float(result[0][0]), 3),
                "Прочность при растяжении, МПа": round(float(result[0][1]), 3),
            }
        except Exception as e:
            prediction = f"Ошибка: {str(e)}"
    
    return render_template('ml.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)