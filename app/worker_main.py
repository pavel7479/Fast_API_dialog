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

# app/worker_main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI(title="Audio Dialog Service")

@app.get("/", include_in_schema=True)
async def health_check():
    return {"status": "running"}

@app.post("/process-audio", include_in_schema=True)
async def process_audio(file: UploadFile = File(...)):
    return JSONResponse(content={"filename": file.filename})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.worker_main:app", host="127.0.0.1", port=8000, reload=True)