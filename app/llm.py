"""Обёртка над провайдером LLM.

Единственное место в проекте, которое знает, что мы работаем с Groq. Всё
остальное общается через этот интерфейс: «дай историю — получи ответ».
Смена провайдера (OpenRouter, OpenAI, локальная модель) сводится к правке
LLM_BASE_URL и LLM_MODEL в .env.

Ответ модели описан как «текст ИЛИ запросы на вызов инструментов», хотя
инструментов пока нет. Так задумано: на шаге с напоминаниями останется
описать функции и передать их сюда, не переписывая логику бота.
"""

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.api_errors import friendly_errors
from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """Просьба модели выполнить нашу функцию."""

    call_id: str
    name: str
    arguments: dict
    raw_arguments: str


@dataclass(frozen=True)
class LLMReply:
    """Что вернула модель: либо текст, либо просьбы вызвать инструменты."""

    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """Асинхронный клиент к модели."""

    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        timeout: int = LLM_TIMEOUT_SECONDS,
    ) -> None:
        # Библиотека openai подходит и для Groq: у них совместимый протокол.
        # Асинхронный клиент нужен, чтобы запрос к модели не блокировал бота,
        # пока тот обрабатывает другие сообщения.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._temperature = temperature

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> LLMReply:
        """Отправляет диалог модели и разбирает ответ.

        system — инструкция с текущим временем, всегда идёт первой репликой.
        messages — история разговора в формате [{"role": ..., "content": ...}].
        """
        payload = [{"role": "system", "content": system}, *messages]

        with friendly_errors("запрос к модели"):
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=self._temperature,
                tools=tools or None,  # пустой список API не любит, шлём None
            )

        if usage := response.usage:
            # Полезно видеть расход: так заранее понятно, когда история
            # разрослась и пора урезать контекст.
            logger.info(
                "Модель %s: %s токенов на запрос, %s на ответ",
                self._model,
                usage.prompt_tokens,
                usage.completion_tokens,
            )

        return self._parse(response)

    @staticmethod
    def _parse(response) -> LLMReply:
        """Превращает ответ API в наш LLMReply."""
        message = response.choices[0].message

        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            raw = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                # Модель иногда присылает битый JSON. Не падаем: отдаём вызов
                # с пустыми аргументами, а решение примет вызывающий код.
                logger.warning("Не удалось разобрать аргументы вызова: %r", raw)
                arguments = {}
            calls.append(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                    raw_arguments=raw,
                )
            )

        text = (message.content or "").strip() or None
        return LLMReply(text=text, tool_calls=tuple(calls))

    async def close(self) -> None:
        """Закрывает HTTP-соединения при остановке бота."""
        await self._client.close()
