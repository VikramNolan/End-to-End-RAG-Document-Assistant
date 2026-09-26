import os
import logging
from typing import List, Dict, Any, Generator, Optional
from app.config import settings

logger = logging.getLogger(__name__)

# Preferred fallback models in order of priority
FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.8-27b"
]

class GroqService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY.strip()
        self.model = settings.GROQ_MODEL or "openai/gpt-oss-20b"
        self._client = None
        self._tested_model = None

    @property
    def client(self):
        current_key = settings.GROQ_API_KEY.strip() or os.getenv("GROQ_API_KEY", "").strip()
        if not current_key:
            return None
        if self._client is None or self.api_key != current_key:
            from groq import Groq
            self.api_key = current_key
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(settings.GROQ_API_KEY.strip() or os.getenv("GROQ_API_KEY", "").strip())

    def get_missing_key_message(self) -> str:
        return (
            "[Notice] Groq API Key Not Configured.\n\n"
            "To enable live AI answers, please provide your free Groq API key:\n"
            "1. Visit https://console.groq.com to generate a free API key.\n"
            "2. Add it to your .env file as GROQ_API_KEY=your_key_here.\n"
            "3. Restart the backend to start chatting!"
        )

    def _get_active_model(self) -> str:
        if self._tested_model:
            return self._tested_model
        return self.model

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500
    ) -> str:
        if not self.is_configured() or not self.client:
            return self.get_missing_key_message()

        models_to_try = [self._get_active_model()] + [m for m in FALLBACK_MODELS if m != self._get_active_model()]
        last_error = None

        for model_name in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                self._tested_model = model_name
                self.model = model_name
                return response.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e)
                last_error = err_msg
                if (
                    "does not exist" in err_msg
                    or "model_not_found" in err_msg
                    or "404" in err_msg
                    or "rate_limit_exceeded" in err_msg
                    or "429" in err_msg
                ):
                    logger.warning(f"Model {model_name} unavailable or rate-limited on Groq, trying fallback...")
                    continue
                else:
                    logger.error(f"Groq API call error with model {model_name}: {err_msg}")
                    break

        return f"[Error] Groq API call failed: {last_error}"

    def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500
    ) -> Generator[str, None, None]:
        if not self.is_configured() or not self.client:
            yield self.get_missing_key_message()
            return

        models_to_try = [self._get_active_model()] + [m for m in FALLBACK_MODELS if m != self._get_active_model()]
        last_error = None

        for model_name in models_to_try:
            try:
                stream = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                self._tested_model = model_name
                self.model = model_name
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
                return
            except Exception as e:
                err_msg = str(e)
                last_error = err_msg
                if (
                    "does not exist" in err_msg
                    or "model_not_found" in err_msg
                    or "404" in err_msg
                    or "rate_limit_exceeded" in err_msg
                    or "429" in err_msg
                ):
                    logger.warning(f"Model {model_name} unavailable or rate-limited on Groq for streaming, trying fallback...")
                    continue
                else:
                    logger.error(f"Groq streaming error with {model_name}: {err_msg}")
                    break

        yield f"\n\n[Error] Groq API streaming failed: {last_error}"

groq_service = GroqService()
