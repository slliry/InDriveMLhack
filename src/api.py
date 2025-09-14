import os
import sys
import io
from pathlib import Path
from typing import Dict, Any, List
import torch
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import CarDamageDetector


class CarDamageAPI:
    """API сервис для детекции повреждений автомобилей"""
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Args:
            model_path (str): Путь к файлу с весами модели
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.damage_classes = ["scratch", "dent", "rust", "crack"]
        
        # Загружаем детектор
        self.detector = CarDamageDetector(model_path, str(self.device))
        
        print(f"API сервис детекции повреждений инициализирован на устройстве: {self.device}")
    
    def detect_damages_from_bytes(self, image_bytes: bytes, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Детектирует повреждения из байтов изображения
        
        Args:
            image_bytes (bytes): Байты изображения
            conf_threshold (float): Порог уверенности
            
        Returns:
            Dict[str, Any]: Результаты детекции
        """
        try:
            # Сохраняем временное изображение
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Детектируем повреждения
            result = self.detector.detect_damages(temp_image_path, conf_threshold)
            
            # Удаляем временный файл
            os.remove(temp_image_path)
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при детекции: {str(e)}")
    
    def create_visualization(self, image_bytes: bytes, detections: List[Dict], conf_threshold: float = 0.25) -> bytes:
        """
        Создает визуализацию детекций
        
        Args:
            image_bytes (bytes): Байты исходного изображения
            detections (List[Dict]): Результаты детекции
            conf_threshold (float): Порог уверенности
            
        Returns:
            bytes: Байты изображения с визуализацией
        """
        # Загружаем изображение
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")
        
        # Цвета для каждого класса
        colors = {
            'scratch': (0, 255, 0),    # Зеленый
            'dent': (255, 0, 0),        # Синий
            'rust': (0, 0, 255),       # Красный
            'crack': (255, 255, 0)     # Голубой
        }
        
        # Рисуем детекции
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
            
            # Добавляем текст
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (bbox['x1'], bbox['y1'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Добавляем общую информацию
        total_damages = len(detections)
        car_condition = "damaged" if total_damages > 0 else "intact"
        status_text = f"Состояние: {car_condition} ({total_damages} повреждений)"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Конвертируем обратно в байты
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()


# Создаем FastAPI приложение
app = FastAPI(
    title="Car Damage Detection API",
    description="API для детекции повреждений автомобилей с помощью YOLOv8",
    version="2.0.0"
)

# Инициализируем API сервис (может быть None если не удалось загрузить)
api_service = None

try:
    api_service = CarDamageAPI()
    print("API сервис успешно инициализирован")
except Exception as e:
    print(f"Ошибка инициализации API сервиса: {e}")
    print("Приложение запустится, но детекция будет недоступна")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статические файлы фронтенда
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Корневой эндпоинт - возвращает главную страницу фронтенда"""
    return {"message": "Car Damage Detection API", "status": "running", "version": "2.0.0"}

@app.get("/index.html")
async def index():
    """Возвращает главную страницу фронтенда"""
    return FileResponse("static/browser/index.html")

@app.get("/api")
async def api_info():
    """Информация об API"""
    return {
        "message": "Car Damage Detection API",
        "version": "2.0.0",
        "description": "Детекция повреждений автомобилей (царапины, вмятины, ржавчина, трещины)",
        "endpoints": {
            "/detect": "POST - Детекция повреждений на изображении",
            "/detect_batch": "POST - Детекция для нескольких изображений",
            "/health": "GET - Проверка состояния сервиса",
            "/model_info": "GET - Информация о модели"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {"status": "ok", "message": "Service is running"}

@app.get("/health/detailed")
async def detailed_health_check():
    """Детальная проверка состояния сервиса"""
    if api_service is None:
        return {
            "status": "unhealthy",
            "error": "API service not initialized",
            "model_loaded": False
        }
    
    return {
        "status": "healthy",
        "device": str(api_service.device),
        "model_loaded": True,
        "damage_classes": api_service.damage_classes
    }


@app.get("/model_info")
async def model_info():
    """Информация о модели"""
    if api_service is None:
        raise HTTPException(status_code=503, detail="API service not available")
    
    return {
        "model_name": "YOLOv8 Car Damage Detector",
        "model_type": "Object Detection",
        "damage_classes": api_service.damage_classes,
        "device": str(api_service.device),
        "description": "Детекция повреждений: царапины, вмятины, ржавчина, трещины"
    }


@app.post("/detect")
async def detect_damages(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    include_visualization: bool = True
):
    """
    Детектирует повреждения на изображении автомобиля
    
    Args:
        file (UploadFile): Загруженное изображение
        conf_threshold (float): Порог уверенности (0.0 - 1.0)
        include_visualization (bool): Включить визуализацию в ответ
        
    Returns:
        JSONResponse: Результат детекции
    """
    if api_service is None:
        raise HTTPException(status_code=503, detail="API service not available")
    
    # Проверяем тип файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Файл должен быть изображением (jpg, png, etc.)"
        )
    
    # Проверяем размер файла (максимум 10MB)
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Размер файла не должен превышать 10MB"
        )
    
    try:
        # Читаем содержимое файла
        image_bytes = await file.read()
        
        # Детектируем повреждения
        result = api_service.detect_damages_from_bytes(image_bytes, conf_threshold)
        
        # Добавляем информацию о файле
        result['filename'] = file.filename
        result['file_size'] = len(image_bytes)
        result['conf_threshold'] = conf_threshold
        
        # Добавляем визуализацию если запрошена
        if include_visualization:
            visualization_bytes = api_service.create_visualization(
                image_bytes, result['detections'], conf_threshold
            )
            # Конвертируем в base64 для JSON
            import base64
            result['visualization'] = base64.b64encode(visualization_bytes).decode('utf-8')
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@app.post("/detect_batch")
async def detect_damages_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = 0.25
):
    """
    Детектирует повреждения на нескольких изображениях
    
    Args:
        files (List[UploadFile]): Список загруженных изображений
        conf_threshold (float): Порог уверенности (0.0 - 1.0)
        
    Returns:
        JSONResponse: Результаты детекции
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Максимум 10 файлов за один запрос"
        )
    
    results = []
    
    for file in files:
        try:
            # Проверяем тип файла
            if not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'Файл должен быть изображением'
                })
                continue
            
            # Читаем содержимое файла
            image_bytes = await file.read()
            
            # Детектируем повреждения
            result = api_service.detect_damages_from_bytes(image_bytes, conf_threshold)
            result['filename'] = file.filename
            result['file_size'] = len(image_bytes)
            result['conf_threshold'] = conf_threshold
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return JSONResponse(content={'results': results})


@app.post("/visualize")
async def visualize_detections(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    """
    Возвращает изображение с визуализацией детекций
    
    Args:
        file (UploadFile): Загруженное изображение
        conf_threshold (float): Порог уверенности
        
    Returns:
        Response: Изображение с визуализацией
    """
    # Проверяем тип файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Файл должен быть изображением"
        )
    
    try:
        # Читаем содержимое файла
        image_bytes = await file.read()
        
        # Детектируем повреждения
        result = api_service.detect_damages_from_bytes(image_bytes, conf_threshold)
        
        # Создаем визуализацию
        visualization_bytes = api_service.create_visualization(
            image_bytes, result['detections'], conf_threshold
        )
        
        # Возвращаем изображение
        from fastapi.responses import Response
        return Response(
            content=visualization_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=detection_result_{file.filename}"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при создании визуализации: {str(e)}"
        )


@app.get("/visualize/{image_name}")
async def get_visualization(image_name: str, conf_threshold: float = 0.25):
    """
    Возвращает изображение с визуализацией по имени файла из папки data/images/val/
    
    Args:
        image_name (str): Имя файла изображения
        conf_threshold (float): Порог уверенности
        
    Returns:
        Response: Изображение с визуализацией
    """
    try:
        # Путь к изображению
        image_path = f"data/images/val/{image_name}"
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Изображение не найдено")
        
        # Читаем изображение
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Детектируем повреждения
        result = api_service.detect_damages_from_bytes(image_bytes, conf_threshold)
        
        # Создаем визуализацию
        visualization_bytes = api_service.create_visualization(
            image_bytes, result['detections'], conf_threshold
        )
        
        # Возвращаем изображение
        from fastapi.responses import Response
        return Response(
            content=visualization_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=detection_result_{image_name}"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при создании визуализации: {str(e)}"
        )


# Добавляем catch-all маршрут для фронтенда (должен быть последним!)
@app.get("/{file_path:path}")
async def serve_static_files(file_path: str):
    """Обслуживает статические файлы фронтенда"""
    static_file_path = f"static/browser/{file_path}"
    if os.path.exists(static_file_path) and os.path.isfile(static_file_path):
        return FileResponse(static_file_path)
    else:
        # Если файл не найден, возвращаем главную страницу (для SPA)
        return FileResponse("static/browser/index.html")


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )