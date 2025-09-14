import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class CarDamageDetector:
    """Класс для детекции повреждений автомобилей с помощью YOLOv8"""
    
    def __init__(self, model_path: str = "yolo11n.pt", device: str = 'auto'):
        """
        Args:
            model_path (str): Путь к файлу с весами модели
            device (str): Устройство для инференса
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model_path = model_path
        self.damage_classes = ["scratch", "dent", "rust", "crack"]
        
        # Загружаем модель
        self.model = self._load_model()
        
        print(f"Детектор повреждений инициализирован")
        print(f"Устройство: {self.device}")
        print(f"Классы повреждений: {self.damage_classes}")
    
    def _load_model(self):
        """Загружает модель YOLOv8"""
        if not os.path.exists(self.model_path):
            print(f"Предупреждение: модель {self.model_path} не найдена")
            print("Используем предобученную модель YOLOv11n")
            return YOLO('yolo11n.pt')
        
        print(f"Загружаем модель: {self.model_path}")
        model = YOLO(self.model_path)
        return model
    
    def detect_damages(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        """
        Детектирует повреждения на изображении
        
        Args:
            image_path (str): Путь к изображению
            conf_threshold (float): Порог уверенности
            iou_threshold (float): Порог IoU для NMS
            
        Returns:
            dict: Результаты детекции
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        # Предсказание
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        
        # Обрабатываем результаты
        detections = []
        has_damages = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                has_damages = True
                
                for box in boxes:
                    # Координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Класс и уверенность
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Название класса
                    class_name = self.damage_classes[class_id] if class_id < len(self.damage_classes) else "unknown"
                    
                    detection = {
                        'class': class_name,
                        'class_id': class_id,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                    }
                    detections.append(detection)
        
        # Определяем общее состояние автомобиля
        car_condition = "damaged" if has_damages else "intact"
        
        # Подсчитываем статистику
        damage_stats = {}
        for class_name in self.damage_classes:
            damage_stats[class_name] = sum(1 for det in detections if det['class'] == class_name)
        
        result = {
            'image_path': image_path,
            'car_condition': car_condition,
            'has_damages': has_damages,
            'total_damages': len(detections),
            'damage_stats': damage_stats,
            'detections': detections
        }
        
        return result
    
    def detect_batch(self, image_paths: List[str], conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        """
        Детектирует повреждения на нескольких изображениях
        
        Args:
            image_paths (List[str]): Список путей к изображениям
            conf_threshold (float): Порог уверенности
            iou_threshold (float): Порог IoU для NMS
            
        Returns:
            List[dict]: Результаты детекции для каждого изображения
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.detect_damages(image_path, conf_threshold, iou_threshold)
                results.append(result)
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_detections(self, image_path: str, output_path: str = None, conf_threshold: float = 0.25):
        """
        Визуализирует детекции на изображении
        
        Args:
            image_path (str): Путь к изображению
            output_path (str): Путь для сохранения результата
            conf_threshold (float): Порог уверенности
        """
        # Получаем детекции
        result = self.detect_damages(image_path, conf_threshold)
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return
        
        # Рисуем детекции
        for detection in result['detections']:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Цвет для каждого класса
            colors = {
                'scratch': (0, 255, 0),    # Зеленый
                'dent': (255, 0, 0),        # Синий
                'rust': (0, 0, 255),       # Красный
                'crack': (255, 255, 0)     # Голубой
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
            
            # Добавляем текст
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (bbox['x1'], bbox['y1'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Добавляем общую информацию
        status_text = f"Состояние: {result['car_condition']} ({result['total_damages']} повреждений)"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Сохраняем или показываем
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Результат сохранен: {output_path}")
        else:
            cv2.imshow('Damage Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Детекция повреждений автомобилей')
    parser.add_argument('--model_path', type=str, default='weights/best_model.pt', 
                       help='Путь к файлу с весами модели')
    parser.add_argument('--image_path', type=str, help='Путь к изображению для детекции')
    parser.add_argument('--image_dir', type=str, help='Папка с изображениями для батчевой детекции')
    parser.add_argument('--output_file', type=str, help='Файл для сохранения результатов (JSON)')
    parser.add_argument('--output_image', type=str, help='Путь для сохранения изображения с визуализацией')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Порог уверенности')
    parser.add_argument('--iou_threshold', type=float, default=0.7, help='Порог IoU для NMS')
    parser.add_argument('--device', type=str, default='auto', help='Устройство для инференса (cuda/cpu/auto)')
    parser.add_argument('--visualize', action='store_true', help='Показать визуализацию')
    
    args = parser.parse_args()
    
    # Создаем детектор
    detector = CarDamageDetector(args.model_path, args.device)
    
    # Детекция для одного изображения
    if args.image_path:
        print(f"Детектируем повреждения на: {args.image_path}")
        
        if not os.path.exists(args.image_path):
            print(f"Ошибка: файл {args.image_path} не найден")
            return
        
        result = detector.detect_damages(args.image_path, args.conf_threshold, args.iou_threshold)
        
        print(f"\nРезультаты детекции:")
        print(f"Состояние автомобиля: {result['car_condition']}")
        print(f"Количество повреждений: {result['total_damages']}")
        print(f"Статистика повреждений: {result['damage_stats']}")
        
        if result['detections']:
            print(f"\nДетали повреждений:")
            for i, detection in enumerate(result['detections'], 1):
                print(f"  {i}. {detection['class']} (уверенность: {detection['confidence']:.3f})")
                print(f"     Координаты: ({detection['bbox']['x1']}, {detection['bbox']['y1']}) - ({detection['bbox']['x2']}, {detection['bbox']['y2']})")
        
        # Сохраняем результат в файл
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Результат сохранен в {args.output_file}")
        
        # Визуализация
        if args.visualize or args.output_image:
            output_path = args.output_image or "detection_result.jpg"
            detector.visualize_detections(args.image_path, output_path, args.conf_threshold)
    
    # Батчевая детекция
    elif args.image_dir:
        print(f"Батчевая детекция для папки: {args.image_dir}")
        
        if not os.path.exists(args.image_dir):
            print(f"Ошибка: папка {args.image_dir} не найдена")
            return
        
        # Находим все изображения
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for file_path in Path(args.image_dir).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
        
        if not image_paths:
            print("Изображения не найдены")
            return
        
        print(f"Найдено {len(image_paths)} изображений")
        
        # Детектируем
        results = detector.detect_batch(image_paths, args.conf_threshold, args.iou_threshold)
        
        # Выводим результаты
        for result in results:
            if 'error' in result:
                print(f"{result['image_path']}: ОШИБКА - {result['error']}")
            else:
                print(f"{result['image_path']}: {result['car_condition']} ({result['total_damages']} повреждений)")
        
        # Сохраняем результаты в файл
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Результаты сохранены в {args.output_file}")
    
    else:
        print("Ошибка: необходимо указать --image_path или --image_dir")
        parser.print_help()


if __name__ == "__main__":
    main()