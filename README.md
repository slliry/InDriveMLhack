# Car Damage Detection

Проект для детекции повреждений автомобилей с использованием YOLOv8.

## Описание

Этот проект использует YOLOv8 для детекции и локализации повреждений автомобилей:
- **Царапины** (scratch)
- **Вмятины** (dent) 
- **Ржавчина** (rust)
- **Трещины** (crack)

**Логика работы**: Если модель нашла объекты повреждений → автомобиль поврежден, иначе — целый.

## Структура проекта

```
car-condition/
├── data/                    # Папка для данных
│   ├── images/             # Изображения (train/val)
│   ├── labels/             # Аннотации YOLO (train/val)
│   └── dataset.yaml        # Конфигурация датасета
├── src/
│   ├── dataset.py          # Работа с датасетом YOLO
│   ├── train_yolo.py       # Обучение YOLOv8
│   ├── predict.py          # Детекция повреждений
│   └── api.py              # FastAPI сервис
├── weights/                # Папка для сохранения весов модели
├── runs/                   # Результаты обучения
├── requirements.txt        # Зависимости Python
└── README.md              # Этот файл
```

## Установка

### Локальная установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd car-condition
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Подготовка данных

### Формат данных

Создайте CSV файл с аннотациями в следующем формате:

```csv
image_path,annotations
car1.jpg,"[{'class': 'scratch', 'bbox': [100, 150, 200, 180]}, {'class': 'dent', 'bbox': [300, 200, 350, 250]}]"
car2.jpg,"[{'class': 'rust', 'bbox': [50, 100, 120, 140]}]"
car3.jpg,"[]"
```

### Структура аннотаций

- `class`: класс повреждения (scratch, dent, rust, crack)
- `bbox`: координаты [x1, y1, x2, y2] в пикселях

### Создание датасета YOLO

```python
from src.dataset import CarDamageDataset

# Создаем датасет в формате YOLO
dataset = CarDamageDataset("data")
dataset.create_yolo_dataset("path/to/your/csv_file.csv")
```

## Обучение модели

### Базовое обучение

```bash
python src/train_yolo.py --config data/dataset.yaml --epochs 100 --batch_size 16
```

### Расширенные параметры

```bash
python src/train_yolo.py \
    --config data/dataset.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch_size 32 \
    --imgsz 640 \
    --patience 50
```

### Параметры обучения

- `--config`: Путь к YAML конфигурации датасета
- `--model`: Модель YOLOv8 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `--epochs`: Количество эпох (по умолчанию: 100)
- `--batch_size`: Размер батча (по умолчанию: 16)
- `--imgsz`: Размер изображений (по умолчанию: 640)
- `--patience`: Терпение для ранней остановки (по умолчанию: 50)

### Валидация модели

```bash
python src/train_yolo.py --validate --model_path weights/best_model.pt
```

### Тест инференса

```bash
python src/train_yolo.py --test --test_image path/to/test_image.jpg
```

## Детекция повреждений

### Локальная детекция

#### Одно изображение

```bash
python src/predict.py \
    --model_path weights/best_model.pt \
    --image_path test_image.jpg \
    --conf_threshold 0.25
```

#### Батчевая детекция

```bash
python src/predict.py \
    --model_path weights/best_model.pt \
    --image_dir test_images/ \
    --output_file results.json
```

#### Визуализация

```bash
python src/predict.py \
    --model_path weights/best_model.pt \
    --image_path test_image.jpg \
    --visualize \
    --output_image result.jpg
```

### Параметры детекции

- `--model_path`: Путь к файлу с весами модели
- `--image_path`: Путь к изображению для детекции
- `--image_dir`: Папка с изображениями для батчевой детекции
- `--conf_threshold`: Порог уверенности (по умолчанию: 0.25)
- `--iou_threshold`: Порог IoU для NMS (по умолчанию: 0.7)
- `--output_file`: Файл для сохранения результатов (JSON)
- `--visualize`: Показать визуализацию

## API сервис

### Запуск API локально

```bash
python src/api.py
```

API будет доступен по адресу: http://localhost:8000

### Запуск через uvicorn

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

## Использование API

### Эндпоинты

- `GET /`: Информация о API
- `GET /health`: Проверка состояния сервиса
- `GET /model_info`: Информация о модели
- `POST /detect`: Детекция повреждений на одном изображении
- `POST /detect_batch`: Детекция для нескольких изображений
- `POST /visualize`: Получение изображения с визуализацией

### Примеры запросов

#### Детекция одного изображения

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg" \
     -F "conf_threshold=0.25"
```

#### Детекция с визуализацией

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg" \
     -F "include_visualization=true"
```

#### Получение визуализации

```bash
curl -X POST "http://localhost:8000/visualize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg" \
     --output result.jpg
```

### Формат ответа

```json
{
  "image_path": "test_image.jpg",
  "car_condition": "damaged",
  "has_damages": true,
  "total_damages": 2,
  "damage_stats": {
    "scratch": 1,
    "dent": 1,
    "rust": 0,
    "crack": 0
  },
  "detections": [
    {
      "class": "scratch",
      "class_id": 0,
      "confidence": 0.85,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 200,
        "y2": 180,
        "width": 100,
        "height": 30
      }
    },
    {
      "class": "dent",
      "class_id": 1,
      "confidence": 0.92,
      "bbox": {
        "x1": 300,
        "y1": 200,
        "x2": 350,
        "y2": 250,
        "width": 50,
        "height": 50
      }
    }
  ],
  "filename": "test_image.jpg",
  "file_size": 245760,
  "conf_threshold": 0.25
}
```

## Использование с Postman

1. Откройте Postman
2. Создайте новый POST запрос
3. URL: `http://localhost:8000/detect`
4. Перейдите на вкладку "Body"
5. Выберите "form-data"
6. Добавьте ключи:
   - `file` (тип File) - выберите изображение
   - `conf_threshold` (тип Text) - значение 0.25
   - `include_visualization` (тип Text) - значение true
7. Отправьте запрос

## Модель

### Архитектура

- **Модель**: YOLOv8 (nano, small, medium, large, xlarge)
- **Тип**: Object Detection
- **Классы**: 4 класса повреждений (scratch, dent, rust, crack)
- **Вход**: Изображения 640x640 пикселей
- **Выход**: Bounding boxes с классами и уверенностью

### Логика определения состояния

- **intact** (целый): если не найдено ни одного повреждения
- **damaged** (поврежден): если найдено хотя бы одно повреждение

### Метрики

- **mAP@0.5**: средняя точность при IoU=0.5
- **mAP@0.5:0.95**: средняя точность при IoU от 0.5 до 0.95
- **Precision**: точность детекции
- **Recall**: полнота детекции

## Требования к системе

- Python 3.10+
- CUDA (опционально, для GPU)
- Минимум 8GB RAM
- Минимум 4GB свободного места на диске

## Примеры использования

### Python

```python
from src.predict import CarDamageDetector

# Создаем детектор
detector = CarDamageDetector("weights/best_model.pt")

# Детектируем повреждения
result = detector.detect_damages("test_image.jpg")

print(f"Состояние: {result['car_condition']}")
print(f"Количество повреждений: {result['total_damages']}")

for detection in result['detections']:
    print(f"{detection['class']}: {detection['confidence']:.3f}")
```

### JavaScript (Node.js)

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('test_image.jpg'));
form.append('conf_threshold', '0.25');

axios.post('http://localhost:8000/detect', form, {
  headers: form.getHeaders()
}).then(response => {
  console.log(response.data);
});
```

## Лицензия

MIT License

## Поддержка

Если у вас возникли вопросы или проблемы, создайте issue в репозитории проекта.