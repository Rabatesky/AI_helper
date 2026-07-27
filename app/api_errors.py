"""Перевод ошибок провайдера в понятные пользователю фразы.

Модуль общий для чата и распознавания речи: и то и другое ходит к одному API,
и падать они могут одинаково — лимит, таймаут, неверный ключ, нет сети.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Сбой обращения к API с текстом, который можно показать пользователю."""


@contextmanager
def friendly_errors(action: str) -> Iterator[None]:
    """Превращает исключения библиотеки openai в ProviderError.

    action — что мы делали, попадает в лог: «запрос к модели», «распознавание речи».
    Пользователю показываем короткую фразу, разработчику оставляем подробности.
    """
    try:
        yield
    except RateLimitError as exc:
        # Бесплатный тариф ограничен запросами в минуту и в сутки.
        logger.warning("%s: превышен лимит запросов: %s", action, exc)
        raise ProviderError("Превышен лимит запросов к модели. Попробуй через минуту.") from exc
    except AuthenticationError as exc:
        # Это проблема конфигурации, а не случайный сбой, — отделяем от прочих.
        logger.error("%s: ключ API отклонён: %s", action, exc)
        raise ProviderError("Ключ API не принят. Проверь GROQ_API_KEY в .env.") from exc
    except APITimeoutError as exc:
        logger.warning("%s: превышено время ожидания: %s", action, exc)
        raise ProviderError("Провайдер слишком долго отвечает. Попробуй ещё раз.") from exc
    except APIConnectionError as exc:
        logger.warning("%s: нет связи с провайдером: %s", action, exc)
        raise ProviderError("Не могу связаться с провайдером — похоже, проблемы с сетью.") from exc
    except APIStatusError as exc:
        # Всё остальное, что вернул сервер: неверная модель, слишком большой
        # файл, слишком длинный контекст. Детали — в лог.
        logger.error("%s: ошибка API (%s): %s", action, exc.status_code, exc.message)
        raise ProviderError("Провайдер вернул ошибку. Подробности в логах бота.") from exc
