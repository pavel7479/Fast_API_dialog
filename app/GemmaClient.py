# app/gemma_client.py
import os
import json
import logging
import requests
from typing import Optional
from requests.exceptions import RequestException, Timeout
from requests import post
from app.config import settings
from langchain_core.output_parsers import JsonOutputParser
from app.GemmaAnalysis import GemmaAnalysis

logger = logging.getLogger(__name__)

class GemmaClient:
    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None, default_timeout: Optional[int] = None):
        self.base_url = base_url or settings.GEMMA_BASE_URL
        self.model_name = model_name or settings.GEMMA_MODEL_NAME
        self.default_timeout = default_timeout or settings.GEMMA_TIMEOUT
        self.json_parser = JsonOutputParser(pydantic_object=GemmaAnalysis)

    def load_text_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def build_prompt(self, text: str) -> str:
        format_instructions = self.json_parser.get_format_instructions()
        
        if settings.USER_PROMPT:
            base_prompt = settings.USER_PROMPT.format(text=text)
            return f"{base_prompt}\n\n{format_instructions}"
        
        # fallback с инструкциями
        return f"""Проанализируй этот разговор и верни ответ строго в JSON формате:

    {text}

    {format_instructions}

    ТОЛЬКО JSON, никакого лишнего текста!"""

    def send_prompt(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        timeout: Optional[int] = None,
        retries: int = 3,
        backoff: float = 1.0
    ) -> str:
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt пустой")
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt пустой")

        temperature = temperature if temperature is not None else settings.GEMMA_TEMPERATURE
        top_p = top_p if top_p is not None else settings.GEMMA_TOP_P
        repetition_penalty = repetition_penalty if repetition_penalty is not None else settings.GEMMA_REPEAT_PENALTY
        num_beams = num_beams if num_beams is not None else settings.GEMMA_NUM_BEAMS
        do_sample = do_sample if do_sample is not None else settings.GEMMA_DO_SAMPLE
        timeout = timeout or self.default_timeout

        payload = {
            "model": self.model_name,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
                "num_beams": num_beams,
                "do_sample": do_sample
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        print("\n" + "="*40 + " GEMMA PROMPT " + "="*40)
        print("SYSTEM PROMPT:\n", system_prompt)
        print("\nUSER PROMPT:\n", user_prompt)
        print("="*95 + "\n")

        url = f"{self.base_url.rstrip('/')}/api/chat"
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                logger.info("Gemma request attempt %d/%d url=%s timeout=%s", attempt, retries, url, timeout)
                resp = post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                try:
                    result = resp.json()
                except json.JSONDecodeError as e:
                    logger.exception("Invalid JSON from Gemma")
                    raise RuntimeError(f"Invalid JSON response from Gemma: {e}")
                response_text = result.get("message", {}).get("content", "")
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                logger.info("Received response from Gemma len=%d", len(response_text))

                parsed_response = self.json_parser.parse(response_text)
                return parsed_response
                # return response_text
            except Timeout as e:
                last_exc = e
                logger.warning("Timeout on Gemma request (attempt %d): %s", attempt, e)
            except RequestException as e:
                last_exc = e
                logger.warning("Network/HTTP error on Gemma request (attempt %d): %s", attempt, e)
            # backoff
            if attempt < retries:
                sleep_for = backoff * (2 ** (attempt - 1))
                logger.info("Retrying after %.1f seconds...", sleep_for)
                import time; time.sleep(sleep_for)
        # all attempts failed
        logger.error("All attempts to contact Gemma failed.")
        raise RuntimeError(f"Failed to contact Gemma after {retries} attempts. Last error: {last_exc}")

    def save_response(self, input_path: str, answer: str, output_folder: str) -> str:
        
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{__import__('os').path.splitext(__import__('os').path.basename(input_path))[0]}_response.txt"
        output_path = __import__('os').path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(answer)
        return output_path
