"""Поиск в интернете.

Единственное место, знающее, каким поисковиком мы пользуемся. Сейчас это
DuckDuckGo через библиотеку ddgs: работает без ключей и регистраций.

Почему не встроенный поиск Groq (модель groq/compound): на бесплатном тарифе
он нерабочий. Любой запрос, требующий похода в интернет, возвращает
413 Request Entity Too Large — модель подгружает найденные страницы себе в
контекст и упирается в ограничение на размер запроса. Вопросы из собственных
знаний при этом отвечаются нормально, так что проблема именно в поиске.

Мы забираем только сниппеты из выдачи, не скачивая сами страницы. Этого
обычно достаточно: в сниппетах уже есть и температура, и курс валюты. Если
окажется мало, здесь же добавится загрузка текста страниц.
"""

import asyncio
import logging
from dataclasses import dataclass

from ddgs import DDGS

from app.config import (
    SEARCH_BACKENDS,
    SEARCH_MAX_RESULTS,
    SEARCH_REGION,
    SEARCH_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Поиск не удался; текст пригоден для показа пользователю."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str


class WebSearch:
    """Поиск в интернете с приведением выдачи к нашему виду."""

    def __init__(
        self,
        max_results: int = SEARCH_MAX_RESULTS,
        region: str = SEARCH_REGION,
        timeout: int = SEARCH_TIMEOUT_SECONDS,
        backends: str = SEARCH_BACKENDS,
    ) -> None:
        self._max_results = max_results
        self._region = region
        self._timeout = timeout
        # Список бэкендов перечислением через запятую. Библиотека сама переберёт
        # их по очереди. Оставлять выбор на её усмотрение нельзя: часть
        # источников не отвечает вовсе, а часть на предметных запросах отдаёт
        # энциклопедические статьи вместо профильных сайтов.
        self._backends = backends

    async def search(self, query: str) -> list[SearchResult]:
        """Ищет и возвращает список результатов (может быть пустым)."""
        query = query.strip()
        if not query:
            raise SearchError("Пустой поисковый запрос.")

        try:
            # ddgs синхронная, а бот асинхронный: выполняем в отдельном потоке,
            # иначе на время запроса встанет вся обработка сообщений.
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._search_blocking, query),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("Поиск не уложился в %s с: %s", self._timeout, query)
            raise SearchError("Поиск занял слишком много времени.") from exc
        except Exception as exc:
            # Чаще всего это временная блокировка со стороны поисковика.
            logger.exception("Ошибка поиска по запросу %r", query)
            raise SearchError("Поиск сейчас недоступен, попробуй позже.") from exc

        results = [
            SearchResult(
                title=(item.get("title") or "").strip(),
                snippet=(item.get("body") or "").strip(),
                url=(item.get("href") or "").strip(),
            )
            for item in raw
        ]
        logger.info("Поиск %r: найдено %s результатов", query, len(results))
        return results

    def _search_blocking(self, query: str) -> list[dict]:
        with DDGS() as ddgs:
            return list(
                ddgs.text(
                    query,
                    region=self._region,
                    max_results=self._max_results,
                    backend=self._backends,
                )
            )
