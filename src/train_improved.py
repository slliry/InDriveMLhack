#!/usr/bin/env python3
"""
Улучшенный скрипт обучения YOLOv8 для детекции повреждений автомобилей
"""

import argparse
import os
import torch
from ultralytics import YOLO
import yaml

def main():
    parser = argparse.ArgumentParser(description='Обучение YOLOv8 для детекции повреждений')
    parser.add_argument('--config', type=str, default='data/dataset.yaml', help='Путь к конфигурации датасета')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Модель YOLOv8')
    parser.add_argument('--epochs', type=int, default=200, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=8, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображений')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (cuda/cpu/auto)')
    parser.add_argument('--workers', type=int, default=4, help='Количество воркеров')
    parser.add_argument('--patience', type=int, default=50, help='Терпение для early stopping')
    parser.add_argument('--save_period', type=int, default=10, help='Период сохранения чекпоинтов')
    
    args = parser.parse_args()
    
    # Проверяем устройство
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Используемое устройство: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Загружаем конфигурацию датасета
    with open(args.config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f) or {}

    # Безопасно читаем список классов
    class_names = dataset_config.get('names')
    if class_names is None:
        print("Предупреждение: ключ 'names' отсутствует в data/dataset.yaml или файл поврежден. Продолжаю без вывода классов.")
    else:
        print(f"Классы повреждений: {class_names}")
    print(f"Начинаем обучение модели {args.model}...")
    print(f"Конфигурация датасета: {args.config}")
    print(f"Эпохи: {args.epochs}, Батч: {args.batch_size}, Размер изображений: {args.imgsz}")
    print(f"Устройство: {device}")
    
    # Загружаем модель
    model = YOLO(args.model)
    
    # Параметры обучения с улучшениями
    train_params = {
        'data': args.config,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': device,
        'workers': args.workers,
        'patience': args.patience,
        'save_period': args.save_period,
        
        # Улучшенные параметры
        'lr0': 0.01,           # Начальная скорость обучения
        'lrf': 0.01,           # Финальная скорость обучения
        'momentum': 0.937,     # Момент
        'weight_decay': 0.0005, # Весовой распад
        'warmup_epochs': 3,    # Эпохи разогрева
        'warmup_momentum': 0.8, # Момент разогрева
        'warmup_bias_lr': 0.1, # Скорость обучения bias разогрева
        
        # Аугментации
        'hsv_h': 0.015,        # HSV hue аугментация
        'hsv_s': 0.7,          # HSV saturation аугментация
        'hsv_v': 0.4,          # HSV value аугментация
        'degrees': 0.0,        # Поворот изображений
        'translate': 0.1,      # Сдвиг изображений
        'scale': 0.5,          # Масштабирование
        'shear': 0.0,          # Сдвиг
        'perspective': 0.0,    # Перспектива
        'flipud': 0.0,         # Вертикальное отражение
        'fliplr': 0.5,         # Горизонтальное отражение
        'mosaic': 1.0,         # Mosaic аугментация
        'mixup': 0.0,          # Mixup аугментация
        'copy_paste': 0.0,     # Copy-paste аугментация
        
        # Детекция
        'conf': 0.001,         # Порог уверенности для детекции
        'iou': 0.6,            # IoU threshold для NMS
        'max_det': 300,        # Максимум детекций
        'agnostic_nms': False, # Agnostic NMS
        
        # Обучение
        'amp': True,           # Automatic Mixed Precision
        'fraction': 1.0,       # Доля датасета для обучения
        'profile': False,      # Профилирование
        'freeze': None,        # Заморозка слоев
        'multi_scale': False,  # Мульти-масштабное обучение
        
        # Валидация
        'val': True,           # Валидация
        'split': 'val',        # Разделение для валидации
        'save_json': False,    # Сохранение JSON результатов
        'save_hybrid': False,  # Сохранение hybrid labels
        'half': False,         # FP16 валидация
        'dnn': False,          # OpenCV DNN backend
        
        # Визуализация
        'plots': True,         # Графики
        'show': False,         # Показ изображений
        'save_txt': False,     # Сохранение txt результатов
        'save_conf': False,    # Сохранение confidence
        'save_crop': False,    # Сохранение обрезанных детекций
        'show_labels': True,   # Показ меток
        'show_conf': True,     # Показ уверенности
        'visualize': False,    # Визуализация
        'augment': True,       # Аугментация при валидации
        'verbose': True,       # Подробный вывод
        'seed': 0,             # Семя для воспроизводимости
        'deterministic': True, # Детерминистичность
        'single_cls': False,   # Один класс
        'rect': False,         # Прямоугольное обучение
        'cos_lr': False,       # Cosine LR scheduler
        'close_mosaic': 10,    # Эпохи до закрытия mosaic
        
        # Оптимизация
        'optimizer': 'AdamW',  # Оптимизатор
        'resume': False,       # Продолжение обучения
        'cache': False,        # Кэширование изображений
        'overlap_mask': True,  # Перекрытие масок
        'mask_ratio': 4,       # Соотношение масок
        'dropout': 0.0,        # Dropout
    }
    
    try:
        # Запускаем обучение
        results = model.train(**train_params)
        
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("="*50)
        
        # Копируем лучшую модель
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            import shutil
            weights_dir = 'weights'
            os.makedirs(weights_dir, exist_ok=True)
            shutil.copy2(best_model_path, f'{weights_dir}/best_model.pt')
            print(f"Лучшая модель скопирована в: {weights_dir}/best_model.pt")
        
        print(f"Результаты сохранены в: {results.save_dir}")
        
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    except Exception as e:
        print(f"\nОшибка при обучении: {e}")
        raise

if __name__ == "__main__":
    main()
