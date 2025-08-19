
import uvicorn
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.AudioProcessingPipeline import AudioProcessingPipeline
import asyncio
from app.config import settings
import fastapi.openapi.utils
fastapi.openapi.utils.get_openapi = None
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_service")

app = FastAPI(title="Audio Dialog Service")

# Создаём pipeline и executor при старте
pipeline = AudioProcessingPipeline()
executor = ThreadPoolExecutor(max_workers=1)  # только 1 одновременная обработка (безопасно для GPU)

@app.get("/")
async def health_check():
    """Корневой эндпоинт для проверки работы сервиса"""
    return {
        "status": "running",
        "service": "Audio Dialog Service",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Файл пустой")

    loop = asyncio.get_running_loop()
    # Выполнение синхронного process_bytes в отдельном потоке (в очереди executor)
    try:
        fut = loop.run_in_executor(executor, pipeline.process_bytes, data, file.filename)
        # общий таймаут: суммируем TRANSCRIBE и ANALYSIS таймауты как верхний предел
        total_timeout = settings.TRANSCRIBE_TIMEOUT + settings.ANALYSIS_TIMEOUT
        result = await asyncio.wait_for(fut, timeout=total_timeout)
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        logger.exception("Processing timeout")
        raise HTTPException(status_code=504, detail="Processing timeout")
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="0.0.0.0", port=9988, reload=True)

def run_server():
    """Функция для запуска сервера"""
    uvicorn.run(
        app,
#        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1
    )

if __name__ == "__main__":
    run_server()
