import os
import yaml
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


class CarDamageDataset:
    """
    Класс для работы с датасетом детекции повреждений автомобилей в формате YOLO.
    
    Поддерживает классы повреждений:
    - scratch (царапина)
    - dent (вмятина) 
    - rust (ржавчина)
    - crack (трещина)
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir (str): Путь к папке с данными
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Классы повреждений
        self.damage_classes = {
            0: "scratch",    # царапина
            1: "dent",       # вмятина
            2: "rust",       # ржавчина
            3: "crack"      # трещина
        }
        
        self.class_names = list(self.damage_classes.values())
        
        # Создаем папки если их нет
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def create_yolo_dataset(self, csv_file: str, train_split: float = 0.8):
        """
        Создает датасет в формате YOLO из CSV файла
        
        Args:
            csv_file (str): Путь к CSV файлу с аннотациями
            train_split (float): Доля данных для обучения
        """
        print("Создаем датасет в формате YOLO...")
        
        # Читаем CSV файл
        df = pd.read_csv(csv_file)
        
        # Разделяем на train/val
        train_size = int(len(df) * train_split)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]
        
        # Создаем папки для train/val
        train_images_dir = self.images_dir / "train"
        train_labels_dir = self.labels_dir / "train"
        val_images_dir = self.images_dir / "val"
        val_labels_dir = self.labels_dir / "val"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Обрабатываем обучающую выборку
        print("Обрабатываем обучающую выборку...")
        self._process_split(train_df, train_images_dir, train_labels_dir)
        
        # Обрабатываем валидационную выборку
        print("Обрабатываем валидационную выборку...")
        self._process_split(val_df, val_images_dir, val_labels_dir)
        
        # Создаем YAML конфигурацию
        self._create_yaml_config()
        
        print(f"Датасет создан! Обучающих изображений: {len(train_df)}, валидационных: {len(val_df)}")
    
    def _process_split(self, df: pd.DataFrame, images_dir: Path, labels_dir: Path):
        """Обрабатывает одну часть датасета"""
        for idx, row in df.iterrows():
            image_path = Path(row['image_path'])
            annotations = eval(row['annotations'])  # Предполагаем, что аннотации в формате списка
            
            # Копируем изображение
            if image_path.exists():
                shutil.copy2(image_path, images_dir / image_path.name)
                
                # Создаем файл с аннотациями
                label_file = labels_dir / f"{image_path.stem}.txt"
                self._write_yolo_annotations(annotations, label_file)
            else:
                print(f"Предупреждение: изображение {image_path} не найдено")
    
    def _write_yolo_annotations(self, annotations: List[Dict], label_file: Path):
        """
        Записывает аннотации в формате YOLO
        
        Args:
            annotations: Список аннотаций в формате [{"class": "scratch", "bbox": [x1, y1, x2, y2]}, ...]
            label_file: Путь к файлу с аннотациями
        """
        with open(label_file, 'w') as f:
            for ann in annotations:
                class_name = ann['class']
                bbox = ann['bbox']  # [x1, y1, x2, y2]
                
                # Конвертируем в формат YOLO (class_id, x_center, y_center, width, height)
                if class_name in self.class_names:
                    class_id = self.class_names.index(class_name)
                    
                    # Предполагаем, что bbox в пикселях, нужно нормализовать
                    # Для этого нужны размеры изображения
                    image_path = label_file.parent.parent.parent / "images" / label_file.parent.name / f"{label_file.stem}.jpg"
                    if image_path.exists():
                        img = Image.open(image_path)
                        img_width, img_height = img.size
                        
                        x1, y1, x2, y2 = bbox
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _create_yaml_config(self):
        """Создает YAML конфигурацию для YOLO"""
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        config_file = self.data_dir / "dataset.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"YAML конфигурация создана: {config_file}")
    
    def visualize_annotations(self, image_path: str, label_path: str = None):
        """
        Визуализирует аннотации на изображении
        
        Args:
            image_path (str): Путь к изображению
            label_path (str): Путь к файлу с аннотациями (опционально)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return
        
        height, width = image.shape[:2]
        
        # Если путь к аннотациям не указан, ищем автоматически
        if label_path is None:
            image_name = Path(image_path).stem
            label_path = self.labels_dir / f"{image_name}.txt"
        
        if Path(label_path).exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                    
                    # Конвертируем обратно в пиксели
                    x_center *= width
                    y_center *= height
                    bbox_width *= width
                    bbox_height *= height
                    
                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)
                    
                    # Рисуем прямоугольник
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Добавляем название класса
                    class_name = self.class_names[int(class_id)]
                    cv2.putText(image, class_name, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Показываем изображение
        cv2.imshow('Annotations', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_dataset_stats(self):
        """Возвращает статистику датасета"""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_counts': {name: 0 for name in self.class_names}
        }
        
        for split in ['train', 'val']:
            images_dir = self.images_dir / split
            labels_dir = self.labels_dir / split
            
            if images_dir.exists():
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                stats['total_images'] += len(image_files)
                
                for image_file in image_files:
                    label_file = labels_dir / f"{image_file.stem}.txt"
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                if class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                    stats['class_counts'][class_name] += 1
                                    stats['total_annotations'] += 1
        
        return stats


def create_sample_dataset():
    """Создает пример датасета для тестирования"""
    print("Создаем пример датасета...")
    
    # Создаем пример CSV файла
    sample_data = {
        'image_path': ['car1.jpg', 'car2.jpg', 'car3.jpg'],
        'annotations': [
            "[{'class': 'scratch', 'bbox': [100, 150, 200, 180]}, {'class': 'dent', 'bbox': [300, 200, 350, 250]}]",
            "[{'class': 'rust', 'bbox': [50, 100, 120, 140]}]",
            "[]"  # Нет повреждений
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_dataset.csv', index=False)
    
    print("Пример CSV файла создан: data/sample_dataset.csv")
    print("Формат аннотаций:")
    print("- class: название класса повреждения (scratch, dent, rust, crack)")
    print("- bbox: координаты [x1, y1, x2, y2] в пикселях")


if __name__ == "__main__":
    # Тестирование
    dataset = CarDamageDataset()
    
    print("Классы повреждений:")
    for class_id, class_name in dataset.damage_classes.items():
        print(f"  {class_id}: {class_name}")
    
    # Создаем пример датасета
    create_sample_dataset()
    
    print("\nДля создания реального датасета используйте:")
    print("dataset.create_yolo_dataset('path/to/your/csv_file.csv')")