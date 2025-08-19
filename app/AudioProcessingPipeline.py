# app/pipeline.py
import os
import time
import glob
import tempfile
from tqdm import tqdm
import logging

from app.Transcriber import Transcriber
from app.GemmaClient import GemmaClient
import os
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

def log_prompts(system_prompt: str, user_prompt: str, transcript: str):
    """Логирует системный промт, пользовательский промт и текст транскрипции для отладки."""
    if not settings.LOG_PROMPTS or settings.ENV.lower() != "development":
        return

    max_len = int(settings.LOG_PROMPT_MAX_CHARS or 0)

    def format_block(title: str, text: str) -> str:
        length = len(text)
        if max_len > 0 and length > max_len:
            text = text[:max_len] + f"\n... [truncated, full length={length}]"
        return (
            f"\n{'═'*60}\n"
            f"📌 {title} (len={length})\n"
            f"{'-'*40}\n"
            f"{text}\n"
            f"{'═'*60}"
        )

    print(format_block("SYSTEM_PROMPT", system_prompt))
    print(format_block("USER_PROMPT", user_prompt))
    print(format_block("TRANSCRIPT", transcript))

class AudioProcessingPipeline:
    def __init__(self):
        self.audio_input_folder = settings.AUDIO_INPUT_FOLDER
        self.transcriber_output_folder = settings.TRANSCRIBER_OUTPUT_FOLDER
        self.gemma_input_folder = settings.GEMMA_INPUT_FOLDER
        self.gemma_output_folder = settings.GEMMA_OUTPUT_FOLDER

        self.language = settings.LANGUAGE
        self.beam_size = settings.BEAM_SIZE

        self.transcriber = Transcriber(
            model_size=settings.TRANSCRIBER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.TRANSCRIBER_COMPUTE
        )
        self.gemma = GemmaClient(
            base_url=settings.GEMMA_BASE_URL,
            model_name=settings.GEMMA_MODEL_NAME,
            default_timeout=settings.GEMMA_TIMEOUT
        )

    def run(self):
        self._run_transcription()
        self._run_gemma_analysis()

    def _run_transcription(self):
        os.makedirs(self.transcriber_output_folder, exist_ok=True)
        audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
        audio_files = [f for f in glob.glob(os.path.join(self.audio_input_folder, "*")) if f.lower().endswith(audio_extensions)]

        if not audio_files:
            print(f"⚠️ Нет аудиофайлов в: {self.audio_input_folder}")
            return

        for audio_path in tqdm(audio_files, desc="Транскрибирование", unit="файл"):
            try:
                segments = self.transcriber.transcribe(audio_path, language=self.language, beam_size=self.beam_size)
                text = " ".join(getattr(s, "text", str(s)) for s in segments)
                filename = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
                out_path = os.path.join(self.transcriber_output_folder, filename)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"❌ Ошибка при обработке {audio_path}: {e}")

    def _run_gemma_analysis(self):
        os.makedirs(self.gemma_output_folder, exist_ok=True)
        txt_files = glob.glob(os.path.join(self.gemma_input_folder, "*.txt"))

        if not txt_files:
            print(f"⚠️ Нет .txt файлов для анализа в: {self.gemma_input_folder}")
            return

        system_prompt = settings.SYSTEM_PROMPT or "Анализируй текст разговора строго по инструкциям."

        for txt_path in tqdm(txt_files, desc="Анализ Gemma", unit="файл"):

            # Важно! Код где загружаются все промты и транскибированный текств Gemma!
            try:
                text = self.gemma.load_text_file(txt_path)
                user_prompt = self.gemma.build_prompt(text)

                log_prompts(system_prompt, user_prompt, text) 

                answer = self.gemma.send_prompt(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=settings.GEMMA_TEMPERATURE,
                    repetition_penalty=settings.GEMMA_REPEAT_PENALTY,
                    do_sample=settings.GEMMA_DO_SAMPLE,
                    num_beams=settings.GEMMA_NUM_BEAMS,
                    timeout=settings.GEMMA_TIMEOUT,
                )
                self.gemma.save_response(txt_path, answer, self.gemma_output_folder)
            except Exception as e:
                print(f"❌ Ошибка при анализе {txt_path}: {e}")

    # def process_bytes(self, audio_bytes: bytes, filename: Optional[str] = None, language: Optional[str] = None, beam_size: Optional[int] = None) -> dict:
    #     t0 = time.time()
    #     language = language or self.language
    #     beam_size = beam_size or self.beam_size

    #     suffix = ".wav"
    #     if filename:
    #         ext = os.path.splitext(filename)[1]
    #         if ext:
    #             suffix = ext

    #     fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    #     os.close(fd)
    #     try:
    #         with open(tmp_path, "wb") as f:
    #             f.write(audio_bytes)

    #         t1 = time.time()
    #         segments = self.transcriber.transcribe(tmp_path, language=language, beam_size=beam_size)
    #         transcript_text = " ".join(getattr(s, "text", str(s)) for s in segments).strip()
    #         t_trans = time.time() - t1

    #         t2 = time.time()


    #         system_prompt = settings.SYSTEM_PROMPT or "Анализируй текст разговора строго по инструкциям."
    #         user_prompt = self.gemma.build_prompt(transcript_text)

    #         log_prompts(system_prompt, user_prompt, transcript_text)
    #         answer = self.gemma.send_prompt(
    #             user_prompt=user_prompt,
    #             system_prompt=system_prompt,
    #             temperature=settings.GEMMA_TEMPERATURE,
    #             repetition_penalty=settings.GEMMA_REPEAT_PENALTY,
    #             do_sample=settings.GEMMA_DO_SAMPLE,
    #             num_beams=settings.GEMMA_NUM_BEAMS,
    #             timeout=settings.GEMMA_TIMEOUT,
    #         )
    #         t_ana = time.time() - t2

    #         return {
    #             "transcript": transcript_text,
    #             "analysis": answer,
    #             "timings_sec": {
    #                 "save": round(t1 - t0, 3),
    #                 "transcription": round(t_trans, 3),
    #                 "analysis": round(t_ana, 3),
    #                 "total": round(time.time() - t0, 3)
    #             }
    #         }

    #     finally:
    #         try:
    #             if os.path.exists(tmp_path):
    #                 os.remove(tmp_path)
    #         except Exception:
    #             pass

#===============================================================================================
    def process_bytes(self, audio_bytes: bytes, filename: Optional[str] = None, 
                     language: Optional[str] = None, beam_size: Optional[int] = None) -> dict:
        """
        Обрабатывает аудиофайл: транскрибация и анализ текста
        Возвращает:
        {
            "transcript": str,  # текст расшифровки
            "analysis": dict,    # результат анализа (JSON от модели)
            "timings_sec": {
                "save": float,      # время сохранения файла
                "transcription": float,  # время транскрибации
                "analysis": float,      # время анализа
                "total": float          # общее время
            }
        }
        """
        t0 = time.time()
        language = language or self.language
        beam_size = beam_size or self.beam_size

        # Создаем временный файл
        suffix = ".wav"
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext:
                suffix = ext

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        try:
            # Сохраняем аудио во временный файл
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            t1 = time.time()

            # Транскрибация аудио
            segments = self.transcriber.transcribe(tmp_path, language=language, beam_size=beam_size)
            transcript_text = " ".join(getattr(s, "text", str(s)) for s in segments).strip()
            t_trans = time.time() - t1

            # Анализ текста
            t2 = time.time()
            system_prompt = settings.SYSTEM_PROMPT or "Анализируй текст разговора строго по инструкциям."
            user_prompt = self.gemma.build_prompt(transcript_text)

            # Логируем промты для отладки
            log_prompts(system_prompt, user_prompt, transcript_text)

            # Получаем ответ от модели
            raw_answer = self.gemma.send_prompt(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=settings.GEMMA_TEMPERATURE,
                repetition_penalty=settings.GEMMA_REPEAT_PENALTY,
                do_sample=settings.GEMMA_DO_SAMPLE,
                num_beams=settings.GEMMA_NUM_BEAMS,
                timeout=settings.GEMMA_TIMEOUT,
            )

            # Обработка JSON вывода
            analysis_data = self._parse_model_output(raw_answer)
            t_ana = time.time() - t2

            return {
                "transcript": transcript_text,
                "analysis": analysis_data,
                "timings_sec": {
                    "save": round(t1 - t0, 3),
                    "transcription": round(t_trans, 3),
                    "analysis": round(t_ana, 3),
                    "total": round(time.time() - t0, 3)
                }
            }

        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Error removing temp file: {str(e)}")

