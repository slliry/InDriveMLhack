#!/usr/bin/env python3
"""
Тестовый скрипт для проверки детекции с разными порогами
"""

import requests
import base64
from PIL import Image
import io

def test_detection(image_path, conf_threshold=0.05):
    """Тестирует детекцию с разными порогами"""
    
    # Читаем изображение
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Отправляем запрос
    files = {'file': (image_path, image_bytes, 'image/jpeg')}
    data = {'conf_threshold': conf_threshold}
    
    try:
        response = requests.post('http://localhost:8000/detect', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n=== Порог уверенности: {conf_threshold} ===")
            print(f"Состояние автомобиля: {result.get('car_condition', 'unknown')}")
            print(f"Найдено повреждений: {result.get('total_damages', 0)}")
            print(f"Статистика: {result.get('damage_stats', {})}")
            
            detections = result.get('detections', [])
            if detections:
                print("Детекции:")
                for i, det in enumerate(detections):
                    print(f"  {i+1}. {det.get('class', 'unknown')} - {det.get('confidence', 0):.3f}")
            else:
                print("Детекции не найдены")
                
            return result
        else:
            print(f"Ошибка: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Ошибка при отправке запроса: {e}")
        return None

def main():
    print("Тестирование детекции повреждений автомобилей")
    print("=" * 50)
    
    # Список изображений для тестирования
    test_images = [
        "data/images/val/car-body-scratches_jpg.rf.e17bfa6eb25f91ba238ce42790d94c2f.jpg",
        "data/images/val/7p4p1hie7oa01_jpg.rf.be1569ae6864834079eca32528be4034.jpg",
        "data/images/val/damaged-car-after-a-car-accident-2021-04-06-18-22-05-utc-scaled-e1631415899344_jpg.rf.7bc18289902d624ad762063249dad646.jpg"
    ]
    
    # Разные пороги уверенности
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for image_path in test_images:
        print(f"\n🖼️ Тестируем: {image_path}")
        print("-" * 50)
        
        for threshold in thresholds:
            result = test_detection(image_path, threshold)
            if result and result.get('total_damages', 0) > 0:
                print(f"✅ Найдены повреждения при пороге {threshold}!")
                break
        else:
            print("❌ Повреждения не найдены ни при одном пороге")

if __name__ == "__main__":
    main()
