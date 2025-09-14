import os
import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd
from datetime import datetime


class CarDamageTrainer:
    """Класс для обучения YOLOv8 модели детекции повреждений автомобилей"""
    
    def __init__(self, config_path: str = "data/dataset.yaml"):
        """
        Args:
            config_path (str): Путь к YAML конфигурации датасета
        """
        self.config_path = config_path
        self.model_name = "yolov8n.pt"  # Можно изменить на yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Классы повреждений
        self.damage_classes = ["scratch", "dent", "rust", "crack"]
        
        print(f"Используемое устройство: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Классы повреждений: {self.damage_classes}")
    
    def train(self, 
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              patience: int = 50,
              save_period: int = 10,
              project: str = "runs/detect",
              name: str = None):
        """
        Обучает YOLOv8 модель
        
        Args:
            epochs (int): Количество эпох
            batch_size (int): Размер батча
            imgsz (int): Размер изображений
            patience (int): Терпение для ранней остановки
            save_period (int): Период сохранения чекпоинтов
            project (str): Папка проекта
            name (str): Название эксперимента
        """
        if name is None:
            name = f"car_damage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Начинаем обучение модели {self.model_name}...")
        print(f"Конфигурация датасета: {self.config_path}")
        print(f"Эпохи: {epochs}, Батч: {batch_size}, Размер изображений: {imgsz}")
        print(f"Устройство: {self.device}")
        
        # Проверяем наличие конфигурации
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Конфигурация датасета не найдена: {self.config_path}")
        
        # Загружаем модель
        model = YOLO(self.model_name)
        
        # Начинаем обучение
        results = model.train(
            data=self.config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=patience,
            save_period=save_period,
            project=project,
            name=name,
            device=self.device,
            workers=4,
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            label_smoothing=0.0,
            nbs=64,
            val=True,
            plots=True,
            save=True,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            vid_stride=1,
            line_width=3,
            visualize=False,
            augment=True,
            agnostic_nms=False,
            classes=None,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False
        )
        
        print(f"Обучение завершено!")
        print(f"Результаты сохранены в: {project}/{name}")
        
        # Копируем лучшую модель в папку weights
        best_model_path = Path(project) / name / "weights" / "best.pt"
        if best_model_path.exists():
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            
            import shutil
            shutil.copy2(best_model_path, weights_dir / "best_model.pt")
            print(f"Лучшая модель скопирована в: weights/best_model.pt")
        
        return results
    
    def validate(self, model_path: str = "weights/best_model.pt"):
        """
        Валидирует обученную модель
        
        Args:
            model_path (str): Путь к обученной модели
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        print(f"Валидация модели: {model_path}")
        
        # Загружаем модель
        model = YOLO(model_path)
        
        # Валидация
        results = model.val(
            data=self.config_path,
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            device=self.device,
            workers=4,
            save_json=True,
            save_hybrid=False,
            plots=True,
            verbose=True
        )
        
        print("Валидация завершена!")
        return results
    
    def test_inference(self, model_path: str = "weights/best_model.pt", test_image: str = None):
        """
        Тестирует инференс на одном изображении
        
        Args:
            model_path (str): Путь к модели
            test_image (str): Путь к тестовому изображению
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Ищем тестовое изображение
        if test_image is None:
            images_dir = Path("data/images/val")
            if images_dir.exists():
                test_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                if test_images:
                    test_image = str(test_images[0])
                else:
                    print("Тестовые изображения не найдены")
                    return
            else:
                print("Папка с тестовыми изображениями не найдена")
                return
        
        print(f"Тестируем инференс на: {test_image}")
        
        # Загружаем модель
        model = YOLO(model_path)
        
        # Предсказание
        results = model(test_image, conf=0.25, iou=0.7)
        
        # Выводим результаты
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"Найдено {len(boxes)} повреждений:")
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.damage_classes[class_id] if class_id < len(self.damage_classes) else "unknown"
                    print(f"  {i+1}. {class_name} (уверенность: {confidence:.3f})")
            else:
                print("Повреждения не обнаружены")
        
        # Сохраняем результат с визуализацией
        output_path = "test_result.jpg"
        results[0].save(output_path)
        print(f"Результат сохранен: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Обучение YOLOv8 для детекции повреждений автомобилей')
    parser.add_argument('--config', type=str, default='data/dataset.yaml', 
                       help='Путь к YAML конфигурации датасета')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='Название модели YOLOv8 (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображений')
    parser.add_argument('--patience', type=int, default=50, help='Терпение для ранней остановки')
    parser.add_argument('--validate', action='store_true', help='Только валидация модели')
    parser.add_argument('--test', action='store_true', help='Тест инференса')
    parser.add_argument('--test_image', type=str, help='Путь к тестовому изображению')
    
    args = parser.parse_args()
    
    # Создаем тренер
    trainer = CarDamageTrainer(config_path=args.config)
    trainer.model_name = args.model
    
    if args.validate:
        # Только валидация
        trainer.validate()
    elif args.test:
        # Тест инференса
        trainer.test_inference(test_image=args.test_image)
    else:
        # Обучение
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            patience=args.patience
        )


if __name__ == "__main__":
    main()