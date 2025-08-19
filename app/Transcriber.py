
from faster_whisper import WhisperModel
from typing import Optional
from app.config import settings


class Transcriber:
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, compute_type: Optional[str] = None):
        model_size = model_size or settings.TRANSCRIBER_MODEL
        device = device or settings.DEVICE
        compute_type = compute_type or settings.TRANSCRIBER_COMPUTE

        if WhisperModel is None:
            raise RuntimeError("faster_whisper не установлен или импорт не удался. Установите зависимость faster-whisper.")
        self.model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str = "ru", beam_size: int = 5):
        segments, _ = self.model.transcribe(audio_path, beam_size=beam_size, language=language)
        return list(segments)

