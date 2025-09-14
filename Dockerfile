# Используем более легкий базовый образ
FROM python:3.9-alpine as builder

# Устанавливаем системные зависимости для сборки
RUN apk add --no-cache \
    build-base \
    linux-headers \
    libffi-dev \
    openssl-dev \
    musl-dev

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем только самые необходимые зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics==8.0.196 && \
    pip install --no-cache-dir opencv-python-headless==4.8.0.74 && \
    pip install --no-cache-dir fastapi==0.104.1 && \
    pip install --no-cache-dir uvicorn==0.24.0 && \
    pip install --no-cache-dir python-multipart==0.0.6 && \
    pip install --no-cache-dir pyyaml==6.0.1 && \
    pip install --no-cache-dir requests==2.31.0

# Финальный образ
FROM python:3.9-alpine

# Устанавливаем только runtime зависимости
RUN apk add --no-cache \
    libstdc++ \
    libgomp

# Копируем виртуальное окружение из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только необходимые файлы
COPY src/ ./src/
COPY static/ ./static/
COPY weights/ ./weights/
COPY yolo11n.pt ./

# Создаем минимальную структуру data
RUN mkdir -p data/images/val data/labels/val

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
