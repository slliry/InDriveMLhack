# Многоэтапная сборка для Railway
FROM python:3.9-slim as builder

# Устанавливаем системные зависимости для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# requirements.txt больше не нужен - устанавливаем зависимости напрямую

# Устанавливаем зависимости с оптимизацией для CPU
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics>=8.0.0 && \
    pip install --no-cache-dir opencv-python-headless>=4.5.0 && \
    pip install --no-cache-dir fastapi>=0.100.0 && \
    pip install --no-cache-dir uvicorn[standard]>=0.20.0 && \
    pip install --no-cache-dir python-multipart>=0.0.6 && \
    pip install --no-cache-dir tqdm>=4.64.0 && \
    pip install --no-cache-dir pyyaml>=6.0 && \
    pip install --no-cache-dir requests>=2.25.0

# Финальный образ
FROM python:3.9-slim

# Устанавливаем только runtime зависимости
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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
COPY data/dataset.yaml ./data/
COPY data/example_labels.csv ./data/

# Создаем минимальную папку data с несколькими примерами
RUN mkdir -p data/images/val data/labels/val && \
    cp data/images/val/*.jpg data/images/val/ 2>/dev/null || true && \
    cp data/labels/val/*.txt data/labels/val/ 2>/dev/null || true

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
