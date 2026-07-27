"""Распознавание речи (speech-to-text).

Голосовые из Telegram приходят в формате OGG/Opus. Groq принимает его напрямую,
поэтому перекодирование через ffmpeg не нужно — файл уходит в API как есть.

Модель Whisper живёт на том же ключе и адресе, что и чат, поэтому здесь тот же
клиент, только другой метод API.
"""

import logging

from openai import AsyncOpenAI

from app.api_errors import friendly_errors
from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TIMEOUT_SECONDS,
    STT_LANGUAGE,
    STT_MODEL,
)

logger = logging.getLogger(__name__)


class SpeechToText:
    """Превращает аудиофайл в текст."""

    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = STT_MODEL,
        language: str = STT_LANGUAGE,
        timeout: int = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        # Явно указанный язык заметно повышает точность. Пустое значение
        # означает автоопределение — пригодится, если говоришь на двух языках.
        self._language = language or None

    async def transcribe(self, audio: bytes, filename: str = "voice.ogg") -> str:
        """Распознаёт речь и возвращает текст (может быть пустым, если тишина).

        Имя файла важно: по расширению провайдер определяет формат аудио,
        поэтому передаём его вместе с содержимым.
        """
        with friendly_errors("распознавание речи"):
            response = await self._client.audio.transcriptions.create(
                model=self._model,
                file=(filename, audio),
                language=self._language,
            )

        text = (response.text or "").strip()
        logger.info(
            "Распознано %s байт аудио -> %s символов текста", len(audio), len(text)
        )
        return text

    async def close(self) -> None:
        await self._client.close()
