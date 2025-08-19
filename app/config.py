
import logging
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    # Transcriber
    TRANSCRIBER_MODEL: str="large-v3"
    TRANSCRIBER_COMPUTE: str="float32"

    # Gemma
    GEMMA_BASE_URL: str="http://170.168.12.23:11434"
    GEMMA_MODEL_NAME: str="gemma3:27b"
    GEMMA_TIMEOUT: int=60
    GEMMA_TEMPERATURE: float=0.1
    GEMMA_TOP_P: float=0.9
    GEMMA_REPEAT_PENALTY: float=1.2
    GEMMA_NUM_BEAMS: int=50
    GEMMA_DO_SAMPLE: bool=False

    # Pipeline paths
    AUDIO_INPUT_FOLDER: str="/tmp/audio_in"
    TRANSCRIBER_OUTPUT_FOLDER: str="/tmp/transcripts"
    GEMMA_INPUT_FOLDER: str="/tmp/gemma_in"
    GEMMA_OUTPUT_FOLDER: str="/tmp/gemma_out"

    # General
    LANGUAGE: str="ru"
    BEAM_SIZE: int=9

    # API / timeouts / device
    API_KEY: str="internal-test"
    TRANSCRIBE_TIMEOUT: int=1200
    ANALYSIS_TIMEOUT: int=1200
    DEVICE: str="cuda"

    # Prompts (use paths to files in .env)
    SYSTEM_PROMPT_PATH: Optional[str]=None
    USER_PROMPT_PATH: Optional[str]=None
    
    ENV: str='development'
    LOG_PROMPTS: bool='true'       # true = показывать, false = отключить
    LOG_PROMPT_MAX_CHARS: int=0


    # Filled at runtime (text content)
    SYSTEM_PROMPT: Optional[str]=None
    USER_PROMPT: Optional[str]=None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "forbid"

def _load_text_from_path(path_str: Optional[str], name: str) -> str:
    """Read and return text from file path. Raise clear errors if missing/empty."""
    if not path_str:
        raise RuntimeError(f"{name} not set: please set {name}_PATH in .env (e.g. {name}_PATH=./prompts/{name.lower()}.txt)")
    p = Path(path_str)
    if not p.exists():
        raise RuntimeError(f"{name} file not found at: {p.resolve()}")
    try:
        text = p.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read {name} file {p.resolve()}: {e}")
    if not text:
        raise RuntimeError(f"{name} file is empty: {p.resolve()}")
    return text


# Создаём объект настроек (подхватит .env)
settings = Settings()

# Загружаем промты из файлов. Падение при отсутствии/пустоте файла — намеренно.
settings.SYSTEM_PROMPT = _load_text_from_path(settings.SYSTEM_PROMPT_PATH, "SYSTEM_PROMPT")
settings.USER_PROMPT = _load_text_from_path(settings.USER_PROMPT_PATH, "USER_PROMPT")

# Логируем короткую информацию (без вывода полного содержимого)
def _short_preview(s: str, length: int = 200) -> str:
    return (s[:length] + ("... [truncated]" if len(s) > length else ""))

logger.info("SYSTEM_PROMPT loaded: length=%d chars; preview: %s", len(settings.SYSTEM_PROMPT), _short_preview(settings.SYSTEM_PROMPT))
logger.info("USER_PROMPT loaded: length=%d chars; preview: %s", len(settings.USER_PROMPT), _short_preview(settings.USER_PROMPT))
