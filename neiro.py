import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

# загрузка данных
X_bp = pd.read_csv("X_bp.csv", index_col=0)  # первый датасет
X_nup = pd.read_csv("X_nup.csv", index_col=0)  # второй датасет

# объединение по индексу
data = X_bp.join(X_nup, how="inner")

# Рассчитываем IQR для каждого признака
Q1 = data.quantile(0.1)
Q3 = data.quantile(0.9)
IQR = Q3 - Q1
# Вычисляем границы для выбросов
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Убираем выбросы
data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

# # Вычисляем корреляцию между признаками
# corr_matrix = data.corr()
# # Строим тепловую карту для наглядности
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.savefig("neiro_corr.png")

# определим целевую переменную (target) и признаки (все кроме целевой)
X = data.drop(columns=['Соотношение матрица-наполнитель', "Плотность, кг/м3", "Плотность нашивки"])
y = data['Соотношение матрица-наполнитель']

# масштабирование признаков
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# создание модели
model = Sequential()
model.add(tf.keras.layers.Input(shape=(X_train.shape[1],))) # входной слой (слой для признаков)
model.add(Dense(128, activation='tanh')) # скрытый слой
model.add(Dense(64, activation='tanh')) # скрытый слой
model.add(Dense(32, activation='tanh')) # скрытый слой
model.add(Dense(1)) # выходной слой (для прогнозирования соотношения)

# компиляция модели (loss = mse)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# оценка модели на тестовых данных
test_loss = model.evaluate(X_test, y_test)
print(f'Loss on test data: {test_loss}')

# прогнозирование для новых данных
new_data = [
    [738.7368421, 30, 22.26785714, 100, 210, 70, 3000, 220, 0, 4],
    [506, 129, 21.25, 300, 380, 75, 1800, 120, 0, 10],
    [478.2862473, 105.7869296, 17.87409991, 328.1545795, 526.6921594, 72.34570879, 3059.032991, 275.5758795, 90, 5],
    [641.0525494, 96.56329319, 22.98929056, 262.956722, 804.5926208, 74.51135922, 2288.967377, 126.8163389, 90, 7],
]
new_data_scaled = scaler.transform(new_data)
predicted_ratio = model.predict(new_data_scaled)
print(predicted_ratio)
print(f'Предсказанное соотношение матрица-наполнитель:') 
print(1.8571429 ,predicted_ratio[0][0])
print(4.193548387, predicted_ratio[1][0])
print(3.305535422, predicted_ratio[2][0])
print(2.709554095, predicted_ratio[3][0])

# plt.plot(history.history['loss'], label='Train loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.savefig("neiro_error.png")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape}")

plt.scatter(y_test, y_pred)
plt.xlabel('Настоящее значение')
plt.ylabel('Предсказанное значение')
plt.title('Сравнение реальных и предсказанных значений')
plt.grid()
plt.savefig("neiro_predskazan.png")


# model.save('neiro_model.h5')
# joblib.dump(scaler, 'neiro_scaler.pkl')