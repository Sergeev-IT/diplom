# Используем официальный образ Python
FROM python:3.12-slim

# Установка системных зависимостей (если нужно)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
# Создание рабочей директории
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем файлы проекта
COPY . .

# Указываем порт, который будет использовать контейнер
EXPOSE 5000

# Запуск Flask-приложения
CMD ["python", "main.py"]
