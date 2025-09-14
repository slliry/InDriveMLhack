# Быстрый старт - Детекция повреждений автомобилей

## 1. Подготовка данных

Создайте CSV файл с аннотациями в папке `data/`:

```csv
image_path,annotations
car1.jpg,"[{'class': 'scratch', 'bbox': [100, 150, 200, 180]}, {'class': 'dent', 'bbox': [300, 200, 350, 250]}]"
car2.jpg,"[{'class': 'rust', 'bbox': [50, 100, 120, 140]}]"
car3.jpg,"[]"
```

Поместите изображения автомобилей в папку `data/`.

## 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 3. Создание датасета YOLO

```bash
python -c "from src.dataset import CarDamageDataset; dataset = CarDamageDataset('data'); dataset.create_yolo_dataset('data/labels.csv')"
```

## 4. Обучение модели

```bash
python src/train_yolo.py --config data/dataset.yaml --epochs 100 --batch_size 16
```

## 5. Детекция повреждений

```bash
python src/predict.py --model_path weights/best_model.pt --image_path test_image.jpg --visualize
```

## 6. Запуск API

```bash
python src/api.py
```

## 7. Тестирование

```bash
python test_api.py
```

## Полезные ссылки

- API документация: http://localhost:8000/docs
- Проверка здоровья: http://localhost:8000/health
- Информация о модели: http://localhost:8000/model_info
