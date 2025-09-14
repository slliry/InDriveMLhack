# Скрипт для тестирования API детекции повреждений

import requests
import json
import os
from pathlib import Path

def test_api():
    """Тестирует API сервис детекции повреждений"""
    
    base_url = "http://localhost:8000"
    
    print("Тестируем API сервис детекции повреждений...")
    
    # Тест 1: Проверка здоровья сервиса
    print("\n1. Проверка здоровья сервиса...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Сервис работает")
            health_data = response.json()
            print(f"  Устройство: {health_data['device']}")
            print(f"  Классы повреждений: {health_data['damage_classes']}")
        else:
            print(f"✗ Ошибка: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Не удается подключиться к сервису")
        print("  Убедитесь, что API запущен на http://localhost:8000")
        return
    
    # Тест 2: Информация о модели
    print("\n2. Информация о модели...")
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            print("✓ Информация о модели получена")
            model_info = response.json()
            print(f"  Модель: {model_info['model_name']}")
            print(f"  Тип: {model_info['model_type']}")
            print(f"  Классы: {model_info['damage_classes']}")
        else:
            print(f"✗ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
    
    # Тест 3: Детекция повреждений (если есть тестовое изображение)
    print("\n3. Тест детекции повреждений...")
    
    # Ищем тестовое изображение
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(Path('data').glob(ext))
    
    if not test_images:
        print("  Нет тестовых изображений в папке data/")
        print("  Поместите изображение автомобиля в папку data/ для тестирования")
        return
    
    test_image = test_images[0]
    print(f"  Используем изображение: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            data = {'conf_threshold': '0.25'}
            response = requests.post(f"{base_url}/detect", files=files, data=data)
        
        if response.status_code == 200:
            print("✓ Детекция выполнена успешно")
            result = response.json()
            print(f"  Состояние автомобиля: {result['car_condition']}")
            print(f"  Количество повреждений: {result['total_damages']}")
            print(f"  Статистика повреждений: {result['damage_stats']}")
            
            if result['detections']:
                print("  Детали повреждений:")
                for i, detection in enumerate(result['detections'], 1):
                    print(f"    {i}. {detection['class']} (уверенность: {detection['confidence']:.3f})")
                    bbox = detection['bbox']
                    print(f"       Координаты: ({bbox['x1']}, {bbox['y1']}) - ({bbox['x2']}, {bbox['y2']})")
        else:
            print(f"✗ Ошибка детекции: {response.status_code}")
            print(f"  Ответ: {response.text}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
    
    # Тест 4: Получение визуализации
    print("\n4. Тест получения визуализации...")
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            data = {'conf_threshold': '0.25'}
            response = requests.post(f"{base_url}/visualize", files=files, data=data)
        
        if response.status_code == 200:
            print("✓ Визуализация получена успешно")
            with open('test_visualization.jpg', 'wb') as f:
                f.write(response.content)
            print("  Результат сохранен: test_visualization.jpg")
        else:
            print(f"✗ Ошибка получения визуализации: {response.status_code}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
    
    print("\nТестирование завершено!")

if __name__ == "__main__":
    test_api()
