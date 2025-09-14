# Используем самый легкий базовый образ
FROM python:3.9-alpine

# Устанавливаем системные зависимости
RUN apk add --no-cache --virtual .build-deps \
    build-base \
    linux-headers \
    libffi-dev \
    openssl-dev \
    musl-dev \
    && apk add --no-cache \
    libstdc++ \
    libgomp

# Устанавливаем Python зависимости в один слой (без кэширования)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy==1.24.3 \
    && pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics==8.0.196 \
    && pip install --no-cache-dir opencv-python-headless==4.8.0.74 \
    && pip install --no-cache-dir fastapi==0.104.1 \
    && pip install --no-cache-dir uvicorn==0.24.0 \
    && pip install --no-cache-dir python-multipart==0.0.6 \
    && pip install --no-cache-dir pyyaml==6.0.1 \
    && pip install --no-cache-dir requests==2.31.0 \
    && apk del .build-deps \
    && rm -rf /var/cache/apk/* \
    && rm -rf /root/.cache/pip/*

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
