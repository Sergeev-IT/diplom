import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# загрузка данных
X_bp = pd.read_csv("X_bp.csv", index_col=0)  # первый датасет
X_nup = pd.read_csv("X_nup.csv", index_col=0)  # второй датасет

# объединение по индексу
data = X_bp.join(X_nup, how="inner")

target_columns = ['Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа']
low_corr_features = ['Соотношение матрица-наполнитель', 'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', 'Угол нашивки, град', 'Плотность нашивки']
X = data.drop(columns=target_columns)
X = X.drop(columns=low_corr_features)
y = data[target_columns]

# порог для корреляции
threshold = 0.04

# corr_matrix = X.corrwith(y['Модуль упругости при растяжении, ГПа']).to_frame(name='Корреляция с Модуль упругости')
# corr_matrix['Корреляция с Прочность при растяжении'] = X.corrwith(y['Прочность при растяжении, МПа'])
# # Найдем признаки с низкой корреляцией с целевыми переменными
# low_corr_features = corr_matrix[
#     (abs(corr_matrix['Корреляция с Модуль упругости']) < threshold) & 
#     (abs(corr_matrix['Корреляция с Прочность при растяжении']) < threshold)
# ].index
# # Удалим эти признаки из набора X
# print(low_corr_features.tolist())

# ограничение выбросов (IQR)
def limit_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df.clip(lower=lower_bound, upper=upper_bound)

X = X.apply(limit_outliers)
y = y.loc[X.index]

# разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42 
)

# масштабирование признаков и целевых переменных
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)  # Только для сравнения после

# базовые модели
base_models = [
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)), 
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)), 
    ('knn', KNeighborsRegressor(n_neighbors=5)), 
]

# финальный регрессор
final_model = LinearRegression()

# стэкинг как один регрессор
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=final_model,
    passthrough=True,
    n_jobs=-1
)

model = MultiOutputRegressor(stacking_regressor)
model.fit(X_train_scaled, y_train_scaled)

# предсказание в стандартизированном пространстве
y_pred_scaled = model.predict(X_test_scaled)

# обратное преобразование к исходным значениям
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)

# метрики
for i, col in enumerate(target_columns):
    mse = mean_squared_error(y_test_unscaled[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test_unscaled[:, i], y_pred[:, i])
    mape = mean_absolute_percentage_error(y_test_unscaled[:, i], y_pred[:, i])
    r2 = r2_score(y_test_unscaled[:, i], y_pred[:, i])
    print(f'{col}\n  MSE: {mse:.2f}\n  MAE: {mae:.2f}\n  MAPE: {mape:.2%}\n  R²: {r2:.3f}\n')


# # MSE и R2 для каждой цели
# for i, col in enumerate(target_columns):
#     mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
#     mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
#     mape = mean_absolute_percentage_error(y_test.iloc[:, i], y_pred[:, i])
#     r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
#     print(f'{col}\n  MSE: {mse:.2f}\n  MAE: {mae:.2f}\n  MAPE: {mape:.2%}\n  R²: {r2:.3f}\n')

# for i, col in enumerate(target_columns):
#     plt.figure(figsize=(6, 6))
#     sns.scatterplot(x=y_test.iloc[:, i], y=y_pred[:, i])
#     plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
#              [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
#              'r--')
#     plt.xlabel(f'Факт: {col}')
#     plt.ylabel(f'Предсказание: {col}')
#     plt.title(f'Предсказание vs Факт для {col}')
#     plt.grid(True)
#     plt.savefig("predskazan.png")

# # Сохраняем pipeline с моделью
# joblib.dump(stacking_pipeline, 'stacking_pipeline.pkl')